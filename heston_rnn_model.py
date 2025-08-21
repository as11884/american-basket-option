# heston_rnn_model_fixed.py
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

# ---- Heston path generator ----
from longstaff_schwartz import CorrelatedHeston


# ---------------------------
# Utils / activations
# ---------------------------

def set_seed(seed: int):
    import os, random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def get_device():
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
    return (S * w.unsqueeze(0)).sum(dim=1)

def payoff_arith(S: torch.Tensor, K: float, w: torch.Tensor, kind: str) -> torch.Tensor:
    G = arithmetic_basket(S, w)
    return torch.clamp(G - K, min=0.0) if kind == "call" else torch.clamp(K - G, min=0.0)

def grad_arith(S: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    B, d = S.shape
    return w.unsqueeze(0).expand(B, d)

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
    G = arithmetic_basket(S, w)
    if kind == "call":
        g = G - K;   grad_g = grad_arith(S, w)
    else:
        g = K - G;   grad_g = -grad_arith(S, w)
    f_k  = F.softplus(kappa * g) / max(kappa, 1e-8)
    sig  = torch.sigmoid(kappa * g).unsqueeze(1)
    grad_fk = sig * grad_g
    return f_k, grad_fk


# ---------------------------
# Three heads: price, delta, alpha(=stock driver)
# Input features = [S_n, sqrt(v_n), g_n] ⇒ dim = 2d+1
# ---------------------------

class PriceGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, kappa: float):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim,
                          num_layers=num_layers, batch_first=True)
        self.embed = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), Swish())
        self.out = nn.Linear(hidden_dim, 1)
        self.softplus_k = SoftplusKappa(kappa)

    def forward(self, X_rev: torch.Tensor, h0: torch.Tensor = None) -> torch.Tensor:
        B, N, _ = X_rev.shape
        if h0 is None:
            h0 = torch.zeros(self.gru.num_layers, B, self.gru.hidden_size, device=X_rev.device)
        h, _ = self.gru(X_rev, h0)
        z = self.embed(h)
        y_rev = self.softplus_k(self.out(z))
        idx = torch.arange(N-1, -1, -1, device=X_rev.device)
        return y_rev.index_select(1, idx)

class DeltaGRU(nn.Module):
    def __init__(self, input_dim: int, d: int, hidden_dim: int, num_layers: int, activation: str = "tanh"):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim,
                          num_layers=num_layers, batch_first=True)
        self.embed = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), Swish())
        self.out = nn.Linear(hidden_dim, d)
        self.activation = activation

    def forward(self, X_rev: torch.Tensor, h0: torch.Tensor = None) -> torch.Tensor:
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

class AlphaGRU(nn.Module):
    def __init__(self, input_dim: int, d: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim,
                          num_layers=num_layers, batch_first=True)
        self.embed = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), Swish())
        self.out = nn.Linear(hidden_dim, d)   # α_y ∈ R^d per step

    def forward(self, X_rev: torch.Tensor, h0: torch.Tensor = None) -> torch.Tensor:
        B, N, _ = X_rev.shape
        if h0 is None:
            h0 = torch.zeros(self.gru.num_layers, B, self.gru.hidden_size, device=X_rev.device)
        h, _ = self.gru(X_rev, h0)
        z = self.embed(h)
        a_rev = self.out(z)
        idx = torch.arange(N-1, -1, -1, device=X_rev.device)
        return a_rev.index_select(1, idx)     # (B,N,d) forward order


# ---------------------------
# OLS helper for α-label
# ---------------------------

def _ridge_ols_alpha(F: torch.Tensor, X: torch.Tensor, ridge: float = 1e-6) -> torch.Tensor:
    """
    F: (B,)   discounted continuation increment / sqrt(dt)
    X: (B,d)  regressors = standardized log-return shocks (≈ L ε)
    returns α_c: (d,)  batch-shared slope in stock-shock (X) space
    """
    B, d = X.shape
    Xt = X.transpose(0, 1)                 # (d,B)
    G  = Xt @ X                            # (d,d)
    if ridge and ridge > 0:
        G = G + ridge * torch.eye(d, device=X.device, dtype=X.dtype)
    b  = Xt @ F                            # (d,)
    return torch.linalg.solve(G, b)        # (d,)


# ---------------------------
# Trainer (Heston + α-label regression)
# ---------------------------

class RNNAmericanHestonAlphaTrainer:
    def __init__(
        self,
        d: int, S0, K: float, r: float, T: float, N: int,
        # Heston params (scalar or (d,)):
        v0, theta, kappa_v, vol_of_vol, rho_sv,
        corr, kind: str = "put", weights=None,
        M: int = 5000, batch_size: int = 512, epochs: int = 20, seed: int = 42,
        hidden_dim: int = 64, num_layers: int = 2,
        lr: float = 1e-3, grad_clip: float = 1.0,
        kappa: float = 10.0, delta_activation: str = "tanh",
        beta: float = 0.5,                     # delta β-blend
        alpha_price: float = 1.0,              # price blend (0..1), keep 1.0
        z_weight: float = 1.0,                 # scale on Z-loss (paper scaling is applied)
        delta_aux_weight: float = 0.0,         # optional Δ auxiliary MSE weight
        smooth_labels: bool = True, smooth_only_at_maturity: bool = False,
        lookahead_window: int = None,
        shuffle: bool = False, drop_last: bool = True,
        resimulate_every: int = 1,
    ):
        assert kind in ("call", "put")
        self.dev = get_device()
        self.d, self.K, self.r, self.T, self.N = d, K, r, T, N
        self.kind = kind

        # vectorize params
        def _vec(x, name):
            if np.isscalar(x): return np.full(d, float(x), dtype=np.float64)
            arr = np.asarray(x, dtype=np.float64); assert arr.shape == (d,), f"{name} must be ({d},)"
            return arr
        self.S0 = _vec(S0, "S0")
        self.v0 = _vec(v0, "v0")
        self.theta = _vec(theta, "theta")
        self.kappa_v = _vec(kappa_v, "kappa_v")
        self.vol_of_vol = _vec(vol_of_vol, "vol_of_vol")
        self.rho_sv = _vec(rho_sv, "rho_sv")

        if np.isscalar(corr) or (hasattr(corr,'ndim') and np.asarray(corr).ndim == 0):
            self.corr_matrix = np.full((d, d), float(corr), dtype=np.float64); np.fill_diagonal(self.corr_matrix, 1.0)
        else:
            corr_np = np.asarray(corr, dtype=np.float64); assert corr_np.shape == (d,d)
            self.corr_matrix = corr_np

        # weights
        if weights is None:
            w_np = np.full(d, 1.0 / d, dtype=np.float64)
        else:
            w_np = np.asarray(weights, dtype=np.float64); w_np = w_np / w_np.sum()
        self.w = torch.from_numpy(w_np).float().to(self.dev)

        # hparams
        self.M, self.batch_size, self.epochs, self.seed = M, batch_size, epochs, seed
        self.hidden_dim, self.num_layers = hidden_dim, num_layers
        self.lr, self.grad_clip = lr, grad_clip
        self.kappa, self.delta_activation = kappa, delta_activation
        self.beta = float(beta); self.alpha_price=float(alpha_price); self.z_weight=float(z_weight)
        self.delta_aux_weight = float(delta_aux_weight)
        self.smooth_labels, self.smooth_only_at_maturity = smooth_labels, smooth_only_at_maturity
        self.lookahead_window = lookahead_window
        self.shuffle, self.drop_last = shuffle, drop_last
        self.resim_every = max(int(resimulate_every), 1)

        # nets: input features = [S, sqrt(v), g] ⇒ 2d+1
        inp_dim = 2 * d + 1
        self.price_net = PriceGRU(inp_dim, hidden_dim, num_layers, kappa).to(self.dev)
        self.delta_net = DeltaGRU(inp_dim, d, hidden_dim, num_layers, activation=self.delta_activation).to(self.dev)
        self.alpha_net = AlphaGRU(inp_dim, d, hidden_dim, num_layers).to(self.dev)

        # delta init projection
        self.delta_h0_proj = nn.Linear(d, self.delta_net.gru.hidden_size, bias=False).to(self.dev)

        # optimizer & scheduler
        self.params = (list(self.price_net.parameters())
                       + list(self.delta_net.parameters())
                       + list(self.alpha_net.parameters())
                       + list(self.delta_h0_proj.parameters()))
        self.opt = torch.optim.Adam(self.params, lr=self.lr)
        self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, factor=0.5, patience=2, verbose=True)

        # constants on device
        self.L_stock  = torch.linalg.cholesky(torch.from_numpy(self.corr_matrix).float().to(self.dev))  # (d,d)

        # caches
        self.dt = None
        self._cached_paths_dev = None
        self.cached_paths_cpu = None
        self.cached_var_cpu = None

        set_seed(self.seed)

    @staticmethod
    def _check_finite(name, *tensors):
        for t in tensors:
            if not torch.isfinite(t).all():
                raise RuntimeError(f"{name}: found NaN/Inf in {name} tensor of shape {t.shape}")

    def _alpha_to_shares(self, alpha_vec: torch.Tensor, S_vec: torch.Tensor, v_vec: torch.Tensor) -> torch.Tensor:
        """
        alpha_vec: (d,)   — coefficient in stock-shock (X=Lε) space
        S_vec:     (d,)   — spot
        v_vec:     (d,)   — inst. variance
        Hedge shares (continuation region): h = alpha / (S * sqrt(v))   elementwise.
        """
        eps = 1e-12
        denom = (S_vec * torch.sqrt(v_vec.clamp_min(eps))).clamp_min(1e-12)
        return alpha_vec / denom

    # ----- Simulation (Heston) -----
    def _simulate_paths(self, epoch_seed: int):
        hes = CorrelatedHeston(
            S0=self.S0, r=self.r, v0=self.v0, theta=self.theta,
            kappa=self.kappa_v, sigma=self.vol_of_vol, rho_sv=self.rho_sv,
            corr_matrix=self.corr_matrix, T=self.T, step=self.N, N=self.M
        )
        S_np = hes.generate_paths(seed=epoch_seed)      # (M, N+1, d)
        # ---- variance paths ----
        if hasattr(hes, "get_variance_paths"):
            V_np = hes.get_variance_paths()
        elif hasattr(hes, "variance_paths"):
            V_np = hes.variance_paths
        else:
            raise RuntimeError("CorrelatedHeston must expose variance paths via get_variance_paths() or .variance_paths")
        # Tensors
        S = torch.from_numpy(S_np).float().to(self.dev, non_blocking=True)
        V = torch.from_numpy(V_np).float().to(self.dev, non_blocking=True)
        dt = self.T / self.N
        return S, V, dt

    def _make_loader(self, S_paths: torch.Tensor, V_paths: torch.Tensor):
        ds = TensorDataset(S_paths, V_paths)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=self.shuffle, drop_last=self.drop_last)

    # ----- Look-ahead labels (batch-mean j*) -----
    def _build_lookahead_labels(self, S_batch: torch.Tensor, n: int, kappa_val: float):
        B, Np1, d = S_batch.shape
        N = Np1 - 1
        assert 0 <= n < N, "Continuation only for n=0..N-1"
        end = N if (self.lookahead_window is None) else min(N, n + self.lookahead_window)
        j_range = torch.arange(n + 1, end + 1, device=S_batch.device, dtype=torch.long)

        S_cand = S_batch[:, j_range, :]
        pay = payoff_arith(S_cand.reshape(-1, d), self.K, self.w, self.kind).view(B, -1)

        D = torch.exp(-self.r * (j_range.to(dtype=pay.dtype) - float(n)) * self.dt)
        disc_pay = pay * D.unsqueeze(0)
        means    = disc_pay.mean(dim=0)
        j_idx    = torch.argmax(means)
        j_star   = int(j_range[j_idx].item())
        D_star   = D[j_idx]

        rows    = torch.arange(B, device=S_batch.device)
        S_tilde = S_batch[rows, j_star, :]
        S_n     = S_batch[:, n, :].clamp_min(5e-4)

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

        c_label  = (D_star * f_t).unsqueeze(-1)            # (B,1) time-n PV
        dc_label = (D_star * g_t) * (S_tilde / S_n)        # (B,d)
        return c_label, dc_label

    # ----- Training -----
    def train(self):
        for epoch in range(1, self.epochs + 1):
            if (epoch - 1) % self.resim_every == 0 or (self._cached_paths_dev is None):
                S_paths, V_paths, self.dt = self._simulate_paths(self.seed + epoch)
                self._cached_paths_dev = (S_paths, V_paths)
            else:
                S_paths, V_paths = self._cached_paths_dev

            loader = self._make_loader(S_paths, V_paths)
            kappa_eff = float(max(2.0 / self.dt, 1.0))
            self.price_net.softplus_k.kappa = kappa_eff

            self.price_net.train(); self.delta_net.train(); self.alpha_net.train()
            run_loss = run_val = run_del = 0.0; nb = 0

            for (S_batch, V_batch) in loader:
                S_batch = S_batch.to(self.dev, non_blocking=True)  # (B,N+1,d)
                V_batch = V_batch.to(self.dev, non_blocking=True)
                B, Np1, d = S_batch.shape
                N = Np1 - 1

                # Features X_n = [S_n, sqrt(v_n), g(S_n)]  (forward), then reverse once for GRUs
                S_all     = S_batch[:, :N, :]
                sqrtV_all = torch.sqrt(V_batch[:, :N, :].clamp_min(0.0))
                G_all     = arithmetic_basket(S_all.reshape(-1, d), self.w).view(B, N, 1)
                g_n       = (G_all - self.K) if self.kind == "call" else (self.K - G_all)

                S_rev, sqrtV_rev, g_rev = S_all.flip(1), sqrtV_all.flip(1), g_n.flip(1)
                X_rev = torch.cat([S_rev, sqrtV_rev, g_rev], dim=2)  # (B,N,2d+1)

                # Hidden-state inits
                S_T = S_batch[:, N, :]
                fT_k, gradT_k = smooth_payoff_and_grad(S_T, self.K, self.w, self.kind, kappa_eff)
                h0_price = fT_k.view(1, B, 1).repeat(self.price_net.gru.num_layers, 1, self.price_net.gru.hidden_size)
                h0_delta = self.delta_h0_proj(gradT_k).unsqueeze(0).repeat(self.delta_net.gru.num_layers, 1, 1)

                # Forward
                y_raw   = self.price_net(X_rev, h0=h0_price)    # (B,N,1) forward order
                d_all   = self.delta_net(X_rev, h0=h0_delta)    # (B,N,d)
                alpha_y = self.alpha_net(X_rev)                 # (B,N,d)
                self._check_finite("pred", y_raw, d_all, alpha_y)

                # Optional price blend (often leave at 1.0)
                if self.alpha_price >= 1.0:
                    y_all = y_raw
                else:
                    n_idx = torch.arange(N, device=self.dev, dtype=S_batch.dtype)
                    Dn    = torch.exp(-self.r * (self.T - n_idx * self.dt))
                    baseline = fT_k.view(B,1,1) * Dn.view(1, N, 1)
                    y_all = self.alpha_price * y_raw + (1.0 - self.alpha_price) * baseline

                # Paper β-blend for deltas (aux head)
                n_idx = torch.arange(N, device=self.dev, dtype=S_batch.dtype)
                Dn    = torch.exp(-self.r * (self.T - n_idx * self.dt))
                S_all_clamped = S_all.clamp_min(5e-4)
                base_delta = Dn.view(1,N,1) * (gradT_k.view(B,1,d) * (S_batch[:, N, :].view(B,1,d) / S_all_clamped))
                d_all = self.beta * d_all + (1.0 - self.beta) * base_delta
                self._check_finite("delta_blend", d_all)

                # Accumulate loss across time
                loss_sum = val_sum = del_sum = 0.0
                for n in range(N):
                    # value labels at n
                    c_n,  dC_n  = self._build_lookahead_labels(S_batch, n,   kappa_eff)  # (B,1),(B,d)

                    # value MSE
                    y_n = y_all[:, n, :].contiguous()
                    val_term = (c_n - y_n).pow(2).mean()

                    # ----- Z(stock) loss via α-label from model's own Y (stop-grad) -----
                    y_det = y_all.detach()  # (B,N,1) stop gradient through labels
                    t_n = n * self.dt
                    disc_n   = math.exp(-self.r * t_n)
                    disc_np1 = math.exp(-self.r * (t_n + self.dt))

                    Y_n_hat   = disc_n   * y_det[:, n, 0]                           # (B,)
                    Y_np1_hat = disc_np1 * (y_det[:, n+1, 0] if n+1 < N else y_det[:, n, 0])
                    F_n       = (Y_np1_hat - Y_n_hat) / math.sqrt(self.dt)          # (B,)

                    # standardized log-return shocks (X = L ε)
                    S_n   = S_batch[:, n, :].clamp_min(5e-4)
                    S_np1 = S_batch[:, n+1, :] if n+1 < N else S_batch[:, n, :]
                    v_n   = V_batch[:, n, :].clamp_min(1e-12)
                    sqrt_vn = torch.sqrt(v_n)

                    log_ret = torch.log(S_np1 / S_n.clamp_min(1e-12))               # (B,d)
                    drift   = (self.r - 0.5 * v_n) * self.dt
                    Xn      = (log_ret - drift) / (sqrt_vn * math.sqrt(self.dt))    # (B,d)

                    alpha_c = _ridge_ols_alpha(F_n, Xn, ridge=1e-6)                 # (d,)
                    # compare in whitened Z space: Z = α L^T
                    Zy = (alpha_y[:, n, :] @ self.L_stock.T)                        # (B,d)
                    Zc = alpha_c @ self.L_stock.T                                   # (d,)
                    z_term = ((Zy - Zc.unsqueeze(0)).pow(2).sum(dim=1)).mean()

                    # optional small Δ auxiliary loss
                    if self.delta_aux_weight > 0:
                        dY_n = d_all[:, n, :].contiguous()
                        delta_aux = (dY_n - dC_n).pow(2).sum(dim=1).mean()
                    else:
                        delta_aux = torch.tensor(0.0, device=self.dev)

                    z_scale = self.z_weight * (self.dt / ((1.0 + self.r*self.dt) ** 2))
                    loss_n = val_term + z_scale * z_term + self.delta_aux_weight * delta_aux

                    loss_sum += loss_n
                    val_sum  += val_term.item()
                    del_sum  += z_term.item()

                loss = loss_sum / N
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                if self.grad_clip and self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.params, self.grad_clip)
                self.opt.step()

                run_loss += loss.item(); run_val += val_sum / N; run_del += del_sum / N; nb += 1

            z_scaled = z_scale * z_term
            print(f"Epoch {epoch:02d} | Loss {run_loss/nb:.6f} | Value {run_val/nb:.6f} | "
                  f"Z(stock) {run_del/nb:.6f} (scaled {z_scaled:.3f}) | Batches {nb}")
            self.sched.step(run_loss/nb)

            # cache last epoch paths
            self.cached_paths_cpu = S_paths.detach().to("cpu")
            self.cached_var_cpu   = V_paths.detach().to("cpu")

        return self

    # ----- Inference at t0 -----
    @torch.no_grad()
    def price_at_all_time(self, seed: int = 777, inference_batch_size: int | None = 1024, use_cached_paths: bool = False):
        """
        Compute option prices, deltas, and alphas for ALL time steps (0 to N-1) in Heston model.
        
        Returns:
            tuple: (y_all_mean, V_all, delta_all_mean, alpha_all_mean)
            - y_all_mean: Continuation values at all time steps (N,)
            - V_all: American option values at all time steps (N,) 
            - delta_all_mean: Deltas at all time steps (N, d)
            - alpha_all_mean: Alphas (volatility hedges) at all time steps (N, d)
        """
        if use_cached_paths and (self.cached_paths_cpu is not None):
            S_paths_cpu = self.cached_paths_cpu
            V_paths_cpu = self.cached_var_cpu
        else:
            S_paths_dev, V_paths_dev, _ = self._simulate_paths(seed)
            S_paths_cpu = S_paths_dev.to("cpu"); V_paths_cpu = V_paths_dev.to("cpu")

        M, Np1, d = S_paths_cpu.shape
        N = Np1 - 1
        kappa_eff = float(max(2.0 / (self.T / self.N), 1.0))

        def _eval_batch(S_batch, V_batch):
            b = S_batch.shape[0]
            S_all     = S_batch[:, :N, :]
            sqrtV_all = torch.sqrt(V_batch[:, :N, :].clamp_min(0.0))
            G_all     = arithmetic_basket(S_all.reshape(-1, d), self.w).view(b, N, 1)
            g_n       = (G_all - self.K) if self.kind == "call" else (self.K - G_all)
            X_rev     = torch.cat([S_all.flip(1), sqrtV_all.flip(1), g_n.flip(1)], dim=2)

            S_T = S_batch[:, N, :]
            fT_k, gradT_k = smooth_payoff_and_grad(S_T, self.K, self.w, self.kind, kappa_eff)
            h0_price = fT_k.view(1, b, 1).repeat(self.price_net.gru.num_layers, 1, self.price_net.gru.hidden_size)
            h0_delta = self.delta_h0_proj(gradT_k).unsqueeze(0).repeat(self.delta_net.gru.num_layers, 1, 1)

            y_raw   = self.price_net(X_rev, h0=h0_price)
            d_all   = self.delta_net(X_rev, h0=h0_delta)
            alpha_y = self.alpha_net(X_rev)
            y_all   = y_raw

            # delta β-blend
            n_idx = torch.arange(N, device=self.dev, dtype=S_batch.dtype)
            Dn    = torch.exp(-self.r * (self.T - n_idx * self.dt))
            S_all_clamped = S_all.clamp_min(5e-4)
            base_delta = Dn.view(1,N,1) * (gradT_k.view(b,1,d) * (S_batch[:, N, :].view(b,1,d) / S_all_clamped))
            d_all = self.beta * d_all + (1.0 - self.beta) * base_delta

            return y_all.squeeze(-1), d_all, alpha_y  # (b,N), (b,N,d), (b,N,d) - ALL time steps

        if inference_batch_size is None:
            y_all_paths, delta_all_paths, alpha_all_paths = _eval_batch(S_paths_cpu.to(self.dev), V_paths_cpu.to(self.dev))
        else:
            y_list, d_list, a_list = [], [], []
            num_batches = (M + inference_batch_size - 1) // inference_batch_size
            for bidx in range(num_batches):
                s = bidx * inference_batch_size
                e = min((bidx + 1) * inference_batch_size, M)
                Sb = S_paths_cpu[s:e].to(self.dev, non_blocking=True)
                Vb = V_paths_cpu[s:e].to(self.dev, non_blocking=True)
                yb, db, ab = _eval_batch(Sb, Vb)
                y_list.append(yb); d_list.append(db); a_list.append(ab)
            y_all_paths = torch.cat(y_list, dim=0); delta_all_paths = torch.cat(d_list, dim=0); alpha_all_paths = torch.cat(a_list, dim=0)

        # Average across paths
        y_all_mean = y_all_paths.mean(dim=0)      # (N,) - continuation values at all times
        delta_all_mean = delta_all_paths.mean(dim=0) # (N,d) - deltas at all times
        alpha_all_mean = alpha_all_paths.mean(dim=0) # (N,d) - alphas at all times

        # Compute American values at all time steps: V_n = max(f(S_n), y_n)
        V_all = torch.zeros(N, device=self.dev)
        
        for n in range(N):
            # Get average stock prices at time n across all paths
            S_n_avg = S_paths_cpu[:, n, :].mean(dim=0).to(self.dev)  # (d,)
            f_n = payoff_arith(S_n_avg.unsqueeze(0), self.K, self.w, self.kind)[0]
            V_all[n] = torch.maximum(f_n, y_all_mean[n])
        
        # Convert to numpy
        results = (y_all_mean.cpu().numpy(),      # (N,) continuation values
                   V_all.cpu().numpy(),          # (N,) American values  
                   delta_all_mean.cpu().numpy(), # (N,d) delta time series
                   alpha_all_mean.cpu().numpy()) # (N,d) alpha time series
        
        return results

    def price_at_t0(self, seed: int = 777, inference_batch_size: int | None = 1024, use_cached_paths: bool = False):
        """
        Helper method that extracts t=0 results from the full time series.
        
        Returns: (y0_mean, V0, delta0_mean, alpha0_mean, hedge_final, hedge_note)
        """
        y_all_mean, V_all, delta_all_mean, alpha_all_mean = self.price_at_all_time(seed, inference_batch_size, use_cached_paths)
        
        # Extract t=0 results
        y0_mean = y_all_mean[0]      # Continuation value at t=0
        V0 = V_all[0]                # American value at t=0  
        delta0_mean = delta_all_mean[0, :]  # Delta at t=0: (d,)
        alpha0_mean = alpha_all_mean[0, :]  # Alpha at t=0: (d,)
        
        # Hedge calculation (same as before) - fix device placement
        S0_t = torch.from_numpy(self.S0).float().to(self.dev)
        v0_t = torch.from_numpy(self.v0).float().to(self.dev)
        S0_batch = torch.from_numpy(self.S0).float().unsqueeze(0).to(self.dev)
        f0 = payoff_arith(S0_batch, self.K, self.w, self.kind)[0]
        
        if f0 >= y0_mean:
            hedge_final = (self.w if self.kind == "call" else -self.w).cpu().numpy()
            hedge_note = "exercise-region hedge (payoff gradient)"
        else:
            h_alpha = self._alpha_to_shares(torch.from_numpy(alpha0_mean).float().to(self.dev), S0_t, v0_t)
            hedge_final = h_alpha.cpu().numpy()
            hedge_note = "continuation-region hedge (h = α/(S√v))"
        
        return y0_mean, V0, delta0_mean, alpha0_mean, hedge_final, hedge_note

    # ----- Save -----
    def save(self, path: str = "data/american_heston_alpha.pth"):
        torch.save({
            "price_net": self.price_net.state_dict(),
            "delta_net": self.delta_net.state_dict(),
            "alpha_net": self.alpha_net.state_dict(),
            "delta_h0_proj": self.delta_h0_proj.state_dict(),
            "meta": {
                "d": self.d, "S0": self.S0.tolist(), "K": self.K, "r": self.r, "T": self.T, "N": self.N,
                "v0": self.v0.tolist(), "theta": self.theta.tolist(), "kappa_v": self.kappa_v.tolist(),
                "vol_of_vol": self.vol_of_vol.tolist(), "rho_sv": self.rho_sv.tolist(),
                "corr": self.corr_matrix.tolist(), "kind": self.kind, "weights": self.w.cpu().numpy(),
                "M": self.M, "batch_size": self.batch_size, "epochs": self.epochs, "seed": self.seed,
                "hidden_dim": self.hidden_dim, "num_layers": self.num_layers,
                "lr": self.lr, "grad_clip": self.grad_clip,
                "beta": self.beta, "alpha_price": self.alpha_price, "z_weight": self.z_weight,
                "delta_aux_weight": self.delta_aux_weight,
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
    corr_matrix = np.eye(d); corr_matrix[corr_matrix==0] = 0.15
    S0_vector = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
    weights   = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

    # Heston params (scalars or (d,))
    v0 = 0.04; theta = 0.04; kappa_v = 1.5; vol_of_vol = 0.5; rho_sv = -0.5

    trainer = RNNAmericanHestonAlphaTrainer(
        d=d, S0=S0_vector, K=120.0, r=0.05, T=0.5, N=126,
        v0=v0, theta=theta, kappa_v=kappa_v, vol_of_vol=vol_of_vol, rho_sv=rho_sv,
        corr=corr_matrix, kind="put", weights=weights,
        M=100000, batch_size=4096, epochs=5, seed=12345,
        hidden_dim=64, num_layers=3, lr=1e-3, grad_clip=1.0,
        alpha_price=1.0, z_weight=1, beta=0.5, delta_aux_weight=max(0.001, 0.005*5/d),
        smooth_labels=True, smooth_only_at_maturity=False,
        lookahead_window=None, shuffle=False, drop_last=True, resimulate_every=1
    )

    trainer.train()

    print(f"\n--- Setup ---")
    print(f"Initial prices: {S0_vector}")
    print(f"Weights: {weights}")
    print(f"Strike: {trainer.K}")
    print(f"Correlation: {corr_matrix[0,1]:.2f} (off-diagonal)")

    y0_cached, V0_cached, delta0_cached, alpha0_cached, hedge, note = trainer.price_at_t0(inference_batch_size=4096, use_cached_paths=True)
    print(f"\n--- Results ---")
    print("Cached paths -> t0 continuation:", round(y0_cached, 6))
    print("Cached paths -> t0 American   :", round(V0_cached, 6))
    print("Cached paths -> t0 deltas     :", np.round(delta0_cached, 6))
    print("Cached paths -> t0 alphas     :", np.round(alpha0_cached, 6))

    print("\n--- Hedges at t0 ---")
    print("Delta-only (ref):   ", np.round(delta0_cached, 6))
    print("Alpha (driver):     ", np.round(alpha0_cached, 6))
    print("Final shares:       ", np.round(hedge, 6), f"[{note}]")