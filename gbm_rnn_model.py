import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

# ---- GBM path generator ----
from longstaff_schwartz import CorrelatedGBM


# ---------------------------
# Utils / activations
# ---------------------------

def set_seed(seed: int):
    import os, random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def get_device():
    # Force CPU usage (comment out the line below to use GPU again)
    # return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Swish(nn.Module):
    def forward(self, x): return x * torch.sigmoid(x)

class SoftplusKappa(nn.Module):
    def __init__(self, kappa: float = 10.0):
        super().__init__()
        self.kappa = float(kappa)
    def forward(self, x):
        k = self.kappa
        return F.softplus(k * x) / max(k, 1e-8)


# ---------------------------
# Arithmetic basket payoff & gradients
# ---------------------------

def arithmetic_basket(S: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    # S: (B,d) >0, w: (d,), sum w=1
    return (S * w.unsqueeze(0)).sum(dim=1)

def payoff_arith(S: torch.Tensor, K: float, w: torch.Tensor, kind: str) -> torch.Tensor:
    G = arithmetic_basket(S, w)
    return torch.clamp(G - K, min=0.0) if kind == "call" else torch.clamp(K - G, min=0.0)

def grad_arith(S: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    # ∂G/∂S_i = w_i for arithmetic basket
    B, d = S.shape
    return w.unsqueeze(0).expand(B, d)  # (B,d)

def grad_payoff_arith(S: torch.Tensor, K: float, w: torch.Tensor, kind: str) -> torch.Tensor:
    G = arithmetic_basket(S, w).unsqueeze(1)
    dG = grad_arith(S, w)
    if kind == "call":
        return dG * (G > K).float()
    else:
        return -dG * (G < K).float()

def smooth_payoff_and_grad(S: torch.Tensor, K: float, w: torch.Tensor, kind: str, kappa: float):
    """
    f_k(s) = softplus(kappa * g(s)) / kappa
    ∇f_k = sigmoid(kappa * g(s)) * ∇g(s)
    g(s) = G(s)-K (call) or K-G(s) (put)
    """
    G = arithmetic_basket(S, w)                 # (B,)
    if kind == "call":
        g = G - K;   grad_g = grad_arith(S, w)          # (B,), (B,d)
    else:
        g = K - G;   grad_g = -grad_arith(S, w)

    f_k  = F.softplus(kappa * g) / max(kappa, 1e-8)    # (B,)
    sig  = torch.sigmoid(kappa * g).unsqueeze(1)       # (B,1)
    grad_fk = sig * grad_g                              # (B,d)
    return f_k, grad_fk


# ---------------------------
# Two RNNs (price & delta)
# ---------------------------

class PriceGRU(nn.Module):
    def __init__(self, d: int, hidden_dim: int, num_layers: int, kappa: float):
        super().__init__()
        self.gru = nn.GRU(input_size=d+1, hidden_size=hidden_dim,
                          num_layers=num_layers, batch_first=True)
        self.embed = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), Swish())
        self.out = nn.Linear(hidden_dim, 1)
        self.softplus_k = SoftplusKappa(kappa)

    def forward(self, X_rev: torch.Tensor, h0: torch.Tensor = None) -> torch.Tensor:
        """
        X_rev: (B,N,d+1) reversed time (t_{N-1}..t_0). Returns (B,N,1) in forward order.
        """
        B, N, _ = X_rev.shape
        if h0 is None:
            h0 = torch.zeros(self.gru.num_layers, B, self.gru.hidden_size, device=X_rev.device)
        h, _ = self.gru(X_rev, h0)
        z = self.embed(h)
        y_rev = self.softplus_k(self.out(z))
        idx = torch.arange(N-1, -1, -1, device=X_rev.device)
        return y_rev.index_select(1, idx)

class DeltaGRU(nn.Module):
    def __init__(self, d: int, hidden_dim: int, num_layers: int, activation: str = "sigmoid"):
        super().__init__()
        self.gru = nn.GRU(input_size=d+1, hidden_size=hidden_dim,
                          num_layers=num_layers, batch_first=True)
        self.embed = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), Swish())
        self.out = nn.Linear(hidden_dim, d)
        self.activation = activation

    def forward(self, X_rev: torch.Tensor, h0: torch.Tensor = None) -> torch.Tensor:
        """
        Returns delta (B,N,d) in forward order.
        """
        B, N, _ = X_rev.shape
        if h0 is None:
            h0 = torch.zeros(self.gru.num_layers, B, self.gru.hidden_size, device=X_rev.device)
        h, _ = self.gru(X_rev, h0)
        z = self.embed(h)
        d_rev = self.out(z)
        if self.activation == "sigmoid":
            d_rev = torch.sigmoid(d_rev)
        elif self.activation == "tanh":
            d_rev = torch.tanh(d_rev)
        idx = torch.arange(N-1, -1, -1, device=X_rev.device)
        return d_rev.index_select(1, idx)


# ---------------------------
# Trainer (paper-faithful)
# ---------------------------

class RNNAmericanTrainer:
    def __init__(
        self,
        d: int, S0, K: float, r: float, T: float, N: int,
        sig: float, corr, kind: str = "put", weights=None,
        M: int = 5000, batch_size: int = 512, epochs: int = 20, seed: int = 42,
        hidden_dim: int = 64, num_layers: int = 2,
        lr: float = 1e-3, grad_clip: float = 1.0,
        kappa: float = 10.0, delta_activation: str = "tanh",
        beta: float = 0.5,                     # paper uses ~0.5 in experiments
        alpha_price: float = 1.0,              # paper blends deltas; keep price unblended by default
        z_weight: float = 1.0,                 # EXACT paper loss when z_weight=1.0
        smooth_labels: bool = True, smooth_only_at_maturity: bool = False,
        lookahead_window: int = None,
        shuffle: bool = False, drop_last: bool = True,
        resimulate_every: int = 1,             # 1 = resim each epoch; >1 keeps paths fixed for K-1 epochs
    ):
        assert kind in ("call", "put")
        self.dev = get_device()
        self.d, self.K, self.r, self.T, self.N = d, K, r, T, N
        self.sig, self.kind = float(sig), kind
        
        # Handle S0 as vector or scalar
        if np.isscalar(S0):
            self.S0 = np.full(d, float(S0), dtype=np.float64)
        else:
            S0_array = np.asarray(S0, dtype=np.float64)
            assert S0_array.shape == (d,), f"S0 must be scalar or shape ({d},), got {S0_array.shape}"
            self.S0 = S0_array
        
        if np.isscalar(corr) or (hasattr(corr, 'ndim') and corr.ndim == 0):
            self.corr = float(corr)
            self.corr_matrix = np.full((d, d), self.corr, dtype=np.float64)
            np.fill_diagonal(self.corr_matrix, 1.0)
        else:
            corr_np = np.asarray(corr, dtype=np.float64)
            self.corr_matrix = corr_np
            self.corr = corr_np
        
        self.M, self.batch_size, self.epochs, self.seed = M, batch_size, epochs, seed
        self.hidden_dim, self.num_layers = hidden_dim, num_layers
        self.lr, self.grad_clip = lr, grad_clip
        self.kappa, self.delta_activation = kappa, delta_activation
        self.beta = float(beta)
        self.alpha_price = float(alpha_price)
        self.z_weight = float(z_weight)
        self.smooth_labels, self.smooth_only_at_maturity = smooth_labels, smooth_only_at_maturity
        self.lookahead_window = lookahead_window
        self.shuffle, self.drop_last = shuffle, drop_last
        self.resim_every = max(int(resimulate_every), 1)

        if weights is None:
            w_np = np.full(d, 1.0 / d, dtype=np.float64)
        else:
            w_np = np.asarray(weights, dtype=np.float64)
            w_np = w_np / w_np.sum()
        self.w = torch.from_numpy(w_np).float().to(self.dev)

        if self.kind == "call" and self.delta_activation != "sigmoid":
            print("NOTE: For calls the paper uses sigmoid deltas; your delta_activation is:", self.delta_activation)
        self.price_net = PriceGRU(d, hidden_dim, num_layers, kappa).to(self.dev)
        self.delta_net = DeltaGRU(d, hidden_dim, num_layers, activation=self.delta_activation).to(self.dev)

        self.delta_h0_proj = nn.Linear(d, self.delta_net.gru.hidden_size, bias=False).to(self.dev)

        self.params = (list(self.price_net.parameters())
                       + list(self.delta_net.parameters())
                       + list(self.delta_h0_proj.parameters()))
        self.opt = torch.optim.Adam(self.params, lr=self.lr)
        self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, factor=0.5, patience=2, verbose=True)


        self.L_vol = None
        self.dt = None
        self.cached_paths_cpu = None
        self._cached_paths_dev = None
        
        # Cache for price_at_all_time results to avoid redundant computation
        self._cached_all_time_results = None
        self._cached_all_time_params = None  # Store params used for cache validation

        set_seed(self.seed)
    
    @staticmethod
    def _check_finite(name, *tensors):
        for t in tensors:
            if not torch.isfinite(t).all():
                raise RuntimeError(f"{name}: found NaN/Inf in tensor of shape {t.shape}")

    # ----- Simulation -----
    def _simulate_paths(self, epoch_seed: int):
        """Simulate (M, N+1, d) true-scale prices with CorrelatedGBM and build L_vol, dt."""
        cov = self.corr_matrix * (self.sig * self.sig)
        gbm = CorrelatedGBM(S0=self.S0, r=self.r, cov=cov, T=self.T, step=self.N, N=self.M)
        paths = gbm.generate_paths(seed=epoch_seed)             # (M, N+1, d)
        S = torch.from_numpy(paths).float().to(self.dev, non_blocking=True)
        cov_t = torch.from_numpy(cov).float().to(self.dev)
        L_vol = torch.linalg.cholesky(cov_t)                    # (d,d)
        dt = self.T / self.N
        return S, L_vol, dt

    def _make_loader(self, S_paths: torch.Tensor):
        ds = TensorDataset(S_paths)  # each sample is one path: (N+1, d)
        return DataLoader(
            ds, batch_size=self.batch_size,
            shuffle=self.shuffle, drop_last=self.drop_last
        )

    # ----- Look-ahead labels (BATCH-MEAN j*) -----
    def _build_lookahead_labels(self, S_batch: torch.Tensor, n: int, kappa_val: float):
        """
        Batch-mean look-ahead (single j* for the whole mini-batch):

        j* = argmax_{j ∈ {n+1,...,N}}  mean_b [ e^{-r (j-n) dt} f(S_j^{(b)}) ]

        Labels at that common j* for every path b in the batch:
        c_n^{(b)}  = D* f(S_{j*}^{(b)}),
        ∇c_n^{(b)} = D* ∇f(S_{j*}^{(b)}) ⊙ (S_{j*}^{(b)} / S_n^{(b)})

        Returns
        c_label: (B,1), dc_label: (B,d)
        """
        B, Np1, d = S_batch.shape
        N = Np1 - 1
        assert 0 <= n < N, "Continuation is defined only for n = 0..N-1"

        # ---- candidate times: n+1..N (continuation only) ----
        end = N if (self.lookahead_window is None) else min(N, n + self.lookahead_window)
        j_range = torch.arange(n + 1, end + 1, device=S_batch.device, dtype=torch.long)  # (J,)

        # payoffs at candidates (B,J)
        S_cand = S_batch[:, j_range, :]  # (B,J,d)
        pay = payoff_arith(S_cand.reshape(-1, d), self.K, self.w, self.kind).view(B, -1)  # (B,J)

        # discounts (J,)
        D = torch.exp(-self.r * (j_range.to(dtype=pay.dtype) - float(n)) * self.dt)  # (J,)

        # ---- batch-mean selection (single j* for the whole batch) ----
        disc_pay = pay * D.unsqueeze(0)        # (B,J)
        means    = disc_pay.mean(dim=0)        # (J,)
        j_idx    = torch.argmax(means)         # scalar long
        j_star   = int(j_range[j_idx].item())
        D_star   = D[j_idx]                    # scalar tensor

        # labels at the common j*
        rows    = torch.arange(B, device=S_batch.device)
        S_tilde = S_batch[rows, j_star, :]                 # (B,d)
        S_n     = S_batch[:, n, :].clamp_min(5e-4)         # (B,d)

        # payoff & gradient at j* (optionally smooth only at maturity)
        if self.smooth_labels:
            if self.smooth_only_at_maturity:
                if j_star == N:
                    f_t, g_t = smooth_payoff_and_grad(S_tilde, self.K, self.w, self.kind, kappa_val)
                else:
                    f_t = payoff_arith(S_tilde, self.K, self.w, self.kind)
                    g_t = grad_payoff_arith(S_tilde, self.K, self.w, self.kind)
            else:
                f_t, g_t = smooth_payoff_and_grad(S_tilde, self.K, self.w, self.kind, kappa_val)
        else:
            f_t = payoff_arith(S_tilde, self.K, self.w, self.kind)
            g_t = grad_payoff_arith(S_tilde, self.K, self.w, self.kind)

        c_label  = (D_star * f_t).unsqueeze(-1)            # (B,1)
        dc_label = (D_star * g_t) * (S_tilde / S_n)        # (B,d)
        return c_label, dc_label
    
    # ----- BSDE residual (paper) -----
    @staticmethod
    def _bsde_loss_step(y_n, dY_n, c_lab, dC_lab, S_n, L_vol, dt, r, z_weight: float):
        """
        Paper loss:
           E|c_n - y_n|^2 + (Δt / (1 + rΔt)^2) * E| Z_c - Z_y |^2
        with Z = (diag(S_n) * delta) * L_vol^T
        """
        val_term = (c_lab - y_n).pow(2).mean()
        Z_y = (S_n * dY_n) @ L_vol.T
        Z_c = (S_n * dC_lab) @ L_vol.T
        xi = Z_c - Z_y
        del_term = (xi.pow(2).sum(dim=1)).mean()
        loss = val_term + z_weight * (dt / ((1.0 + r * dt) ** 2)) * del_term
        return loss, val_term.detach(), del_term.detach()

    # ----- Training -----
    def train(self):
        for epoch in range(1, self.epochs + 1):

            # (Re)simulate paths every resim_every epochs; otherwise reuse CPU cache
            if (epoch - 1) % self.resim_every == 0 or (self._cached_paths_dev is None):
                S_paths, self.L_vol, self.dt = self._simulate_paths(self.seed + epoch)
                # Cache on CPU to avoid GPU OOM for large M
                self._cached_paths_dev = S_paths.detach().cpu()
            else:
                # Move cached paths back to GPU when needed
                S_paths = self._cached_paths_dev.to(self.dev)

            loader = self._make_loader(S_paths)

            # κ ≈ 2/Δt (use everywhere this epoch)
            kappa_eff = float(max(2.0 / self.dt, 1.0))
            self.price_net.softplus_k.kappa = kappa_eff

            self.price_net.train(); self.delta_net.train()
            run_loss = run_val = run_del = 0.0; nb = 0

            for (S_batch,) in loader:
                S_batch = S_batch.to(self.dev, non_blocking=True)   # (B, N+1, d)
                B, Np1, d = S_batch.shape
                N = Np1 - 1

                # Features: X_n = [S_n, g(S_n)]
                S_all = S_batch[:, :N, :]                       # (B,N,d)
                G_all = arithmetic_basket(S_all.reshape(-1, d), self.w).view(B, N, 1)
                g_n = (G_all - self.K) if self.kind == "call" else (self.K - G_all)   # (B,N,1)
                X = torch.cat([S_all, g_n], dim=2)              # (B,N,d+1)

                # Reverse time once
                X_rev = X.flip(dims=[1])                        # (B,N,d+1)

                # Hidden-state init with smoothed maturity info (κ_eff)
                S_T = S_batch[:, N, :]
                fT_k, gradT_k = smooth_payoff_and_grad(S_T, self.K, self.w, self.kind, kappa_eff)
                h0_price = fT_k.view(1, B, 1).repeat(self.price_net.gru.num_layers, 1, self.price_net.gru.hidden_size)
                # delta init via projection (keep directional info)
                h0_delta_base = self.delta_h0_proj(gradT_k)     # (B,H)
                h0_delta = h0_delta_base.unsqueeze(0).repeat(self.delta_net.gru.num_layers, 1, 1)

                # Forward both nets once
                y_raw = self.price_net(X_rev, h0=h0_price)      # (B,N,1)
                d_all = self.delta_net(X_rev, h0=h0_delta)      # (B,N,d)
                self._check_finite("pred", y_raw, d_all)

                # Continuation: by default no price blend (alpha_price=1.0)
                if self.alpha_price >= 1.0:
                    y_all = y_raw
                else:
                    n_idx = torch.arange(N, device=self.dev, dtype=S_batch.dtype)   # 0..N-1
                    Dn    = torch.exp(-self.r * (self.T - n_idx * self.dt))         # (N,)
                    baseline = fT_k.view(B,1,1) * Dn.view(1, N, 1)
                    y_all = self.alpha_price * y_raw + (1.0 - self.alpha_price) * baseline

                # Delta β-blend (paper)
                n_idx = torch.arange(N, device=self.dev, dtype=S_batch.dtype)
                Dn    = torch.exp(-self.r * (self.T - n_idx * self.dt))
                S_all_clamped = S_all.clamp_min(5e-4)
                base_delta = Dn.view(1,N,1) * (gradT_k.view(B,1,d) * (S_batch[:, N, :].view(B,1,d) / S_all_clamped))
                d_all = self.beta * d_all + (1.0 - self.beta) * base_delta
                self._check_finite("delta_blend", d_all)

                # Accumulate BSDE loss across time (paper form)
                loss_sum = val_sum = del_sum = 0.0
                for n in range(N):
                    c_lab, dC_lab = self._build_lookahead_labels(S_batch, n, kappa_eff)
                    self._check_finite(f"labels@{n}", c_lab, dC_lab)
                    y_n  = y_all[:, n, :].contiguous()
                    dY_n = d_all[:, n, :].contiguous()
                    S_n  = S_batch[:, n, :].contiguous()

                    loss_n, vterm, dterm = self._bsde_loss_step(
                        y_n, dY_n, c_lab, dC_lab, S_n, self.L_vol, self.dt, self.r, self.z_weight
                    )
                    loss_sum += loss_n; val_sum += vterm.item(); del_sum += dterm.item()

                loss = loss_sum / N
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                if self.grad_clip and self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.params, self.grad_clip)
                self.opt.step()

                run_loss += loss.item(); run_val += val_sum / N; run_del += del_sum / N; nb += 1
            

            print(f"Epoch {epoch:02d} | Loss {run_loss/nb:.6f} | Value {run_val/nb:.6f} | Delta {run_del/nb:.6f} | Batches {nb}")
            self.sched.step(run_loss/nb)

            # cache this epoch's paths to CPU for later reuse in inference
            self.cached_paths_cpu = S_paths.detach().to("cpu")
            # Clear GPU cache to free memory after training
            self._cached_paths_dev = None

        return self

    # ----- Inference for all time steps -----
    @torch.no_grad()
    def price_at_all_time(self, seed: int = 777, inference_batch_size: int | None = 1024, use_cached_paths: bool = False):
        """
        Compute option prices and deltas for ALL time steps (0 to N-1).
        Results are cached to avoid redundant computation when called multiple times with same parameters.
        
        Args:
            seed: Random seed for path generation (if not using cached paths)
            inference_batch_size: Batch size for inference (None = single pass)
            use_cached_paths: Whether to use cached training paths
            
        Returns:
            tuple: (y_all_mean, V_all, delta_all_mean)
            - y_all_mean: Continuation values at all time steps (N,)
            - V_all: American option values at all time steps (N,) 
            - delta_all_mean: Deltas at all time steps (N, d)
        """
        # Select source
        if use_cached_paths and (self.cached_paths_cpu is not None):
            S_paths_cpu = self.cached_paths_cpu  # (M, N+1, d) on CPU
        else:
            S_paths_dev, _, _ = self._simulate_paths(seed)
            S_paths_cpu = S_paths_dev.to("cpu")

        M, Np1, d = S_paths_cpu.shape
        N = Np1 - 1
        kappa_eff = float(max(2.0 / (self.T / self.N), 1.0))  # same rule at inference

        def _eval_batch(batch_paths: torch.Tensor):
            b = batch_paths.shape[0]
            S_all = batch_paths[:, :N, :]                      # (b,N,d)
            G_all = arithmetic_basket(S_all.reshape(-1, d), self.w).view(b, N, 1)
            g_n   = (G_all - self.K) if self.kind == "call" else (self.K - G_all)
            X     = torch.cat([S_all, g_n], dim=2)
            X_rev = X.flip(dims=[1])

            S_T   = batch_paths[:, N, :]
            fT_k, gradT_k = smooth_payoff_and_grad(S_T, self.K, self.w, self.kind, kappa_eff)
            h0_price = fT_k.view(1, b, 1).repeat(self.price_net.gru.num_layers, 1, self.price_net.gru.hidden_size)
            
            # Delta network initialization (same as in training)
            h0_delta_base = self.delta_h0_proj(gradT_k)     # (b,H)
            h0_delta = h0_delta_base.unsqueeze(0).repeat(self.delta_net.gru.num_layers, 1, 1)

            y_raw  = self.price_net(X_rev, h0=h0_price)    # (b,N,1)
            d_all  = self.delta_net(X_rev, h0=h0_delta)    # (b,N,d)
            y_all  = y_raw  # no price blend for inference (alpha_price=1.0)
            
            # Apply delta blending (same as in training)
            n_idx = torch.arange(N, device=self.dev, dtype=batch_paths.dtype)
            Dn    = torch.exp(-self.r * (self.T - n_idx * self.dt))
            S_all_clamped = S_all.clamp_min(5e-4)
            base_delta = Dn.view(1,N,1) * (gradT_k.view(b,1,d) * (batch_paths[:, N, :].view(b,1,d) / S_all_clamped))
            d_all = self.beta * d_all + (1.0 - self.beta) * base_delta

            return y_all.squeeze(-1), d_all  # (b,N), (b,N,d) - ALL time steps

        if inference_batch_size is None:
            y_all_paths, delta_all_paths = _eval_batch(S_paths_cpu.to(self.dev, non_blocking=True))
        else:
            y_all_list = []
            delta_all_list = []
            num_batches = (M + inference_batch_size - 1) // inference_batch_size
            for bidx in range(num_batches):
                s = bidx * inference_batch_size
                e = min((bidx + 1) * inference_batch_size, M)
                batch_paths = S_paths_cpu[s:e].to(self.dev, non_blocking=True)
                y_batch, delta_batch = _eval_batch(batch_paths)
                y_all_list.append(y_batch)
                delta_all_list.append(delta_batch)
            y_all_paths = torch.cat(y_all_list, dim=0)        # (M,N)
            delta_all_paths = torch.cat(delta_all_list, dim=0) # (M,N,d)

        # Average across paths
        y_all_mean = y_all_paths.mean(dim=0)      # (N,) - continuation values at all times
        delta_all_mean = delta_all_paths.mean(dim=0) # (N,d) - deltas at all times

        # Compute American values at all time steps: V_n = max(f(S_n), y_n)
        V_all = torch.zeros(N, device=self.dev)
        
        for n in range(N):
            # Get average stock prices at time n across all paths
            S_n_avg = S_paths_cpu[:, n, :].mean(dim=0).to(self.dev)  # (d,)
            f_n = payoff_arith(S_n_avg.unsqueeze(0), self.K, self.w, self.kind)[0]
            V_all[n] = torch.maximum(f_n, y_all_mean[n])
        
        # Convert to numpy
        results = (y_all_mean.cpu().numpy(),    # (N,) continuation values
                   V_all.cpu().numpy(),        # (N,) American values  
                   delta_all_mean.cpu().numpy()) # (N,d) delta time series
        
        return results

    # ----- Helper: Extract t=0 results from full time series -----
    def price_at_t0(self, seed: int = 777, inference_batch_size: int | None = 1024, use_cached_paths: bool = False):
        """
        Helper method that extracts t=0 results from the full time series.
        Maintains backward compatibility with existing code.
        
        Returns: (y0_mean, V0, delta0_mean) - continuation value, American value, and delta at t0
        """
        y_all_mean, V_all, delta_all_mean = self.price_at_all_time(seed, inference_batch_size, use_cached_paths)
        
        # Extract t=0 results
        y0_mean = y_all_mean[0]      # Continuation value at t=0
        V0 = V_all[0]                # American value at t=0  
        delta0_mean = delta_all_mean[0, :]  # Delta at t=0: (d,)
        
        return y0_mean, V0, delta0_mean

    # ----- Cache management -----
    # ----- Save -----
    def save(self, path: str = "data/american_arith_two_rnn_bsde.pth"):
        torch.save({
            "price_net": self.price_net.state_dict(),
            "delta_net": self.delta_net.state_dict(),
            "delta_h0_proj": self.delta_h0_proj.state_dict(),
            "meta": {
                "d": self.d, "S0": self.S0.tolist(), "K": self.K, "r": self.r, "T": self.T, "N": self.N,
                "sig": self.sig, "corr": self.corr_matrix.tolist(), "kind": self.kind, "weights": self.w.cpu().numpy(),
                "M": self.M, "batch_size": self.batch_size, "epochs": self.epochs, "seed": self.seed,
                "hidden_dim": self.hidden_dim, "num_layers": self.num_layers,
                "lr": self.lr, "grad_clip": self.grad_clip,
                "beta": self.beta, "alpha_price": self.alpha_price, "z_weight": self.z_weight,
                "kappa": self.kappa, "delta_activation": self.delta_activation,
                "smooth_labels": self.smooth_labels, "smooth_only_at_maturity": self.smooth_only_at_maturity,
                "lookahead_window": self.lookahead_window, "resimulate_every": self.resim_every,
            }
        }, path)
        print(f"Saved model to {path}")


# ---------------------------
# Minimal usage example
# ---------------------------
if __name__ == "__main__":
    d = 5
    # Lower correlation to see more independent behavior
    corr_matrix = np.eye(d)
    corr_matrix.fill(0.15)  # Reduced from 0.30 to 0.15
    np.fill_diagonal(corr_matrix, 1.0)

    # Different initial prices to break symmetry
    S0_vector = np.array([90.0, 95.0, 100.0, 105.0, 110.0])  # Range from 90 to 110
    
    # Different weights (must sum to 1)
    weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])  # Decreasing weights
    
    trainer = RNNAmericanTrainer(
        d=d, S0=S0_vector, K=100.0, r=0.1, T=0.5, N=126,   # match your LSM discretization
        sig=0.2, corr=corr_matrix, kind="put", weights=weights,
        M=50000, batch_size=4096, epochs=10, seed=12345,  # Increased for CPU
        hidden_dim=64, num_layers=3,
        lr=1e-3, grad_clip=1.0,
        
        alpha_price=1.0,     # price unblended
        beta=0.5,            # delta blend (paper used ~0.5)
        z_weight=1.0,        # EXACT paper loss
        # label smoothing at all j (helps stability)
        smooth_labels=True, smooth_only_at_maturity=False,
        lookahead_window=None,
        shuffle=False, drop_last=True,
        resimulate_every=1   # set to >1 to keep same paths fixed for K-1 epochs
    )

    trainer.train()

    # Print setup info
    print(f"\n--- Setup ---")
    print(f"Initial prices: {S0_vector}")
    print(f"Weights: {weights}")
    print(f"Strike: {trainer.K}")
    print(f"Correlation: {corr_matrix[0,1]:.2f} (off-diagonal)")

    # Inference on cached training paths (single pass for exact average)
    y0_cached, V0_cached, delta0_cached = trainer.price_at_t0(inference_batch_size=4096, use_cached_paths=True)
    print(f"\n--- Results ---")
    print("Cached paths -> t0 continuation:", round(y0_cached, 6))
    print("Cached paths -> t0 American   :", round(V0_cached, 6))
    print("Cached paths -> t0 deltas     :")
    for i, (s0, w, delta) in enumerate(zip(S0_vector, weights, delta0_cached)):
        print(f"  Asset {i+1}: S0={s0:6.1f}, weight={w:5.2f}, delta={delta:8.6f}")

    trainer.save()