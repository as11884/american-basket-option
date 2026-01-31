from __future__ import annotations

import math
import copy
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
        g = G - K
        grad_g = grad_arith(S, w)              # (B,d)
    else:
        g = K - G
        grad_g = -grad_arith(S, w)             # (B,d)

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
# Trainer
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
        beta: float = 0.5,
        alpha_price: float = 1.0,
        z_weight: float = 1.0,
        smooth_labels: bool = True, smooth_only_at_maturity: bool = False,
        lookahead_window: int | None = None,
        shuffle: bool = False, drop_last: bool = True,
        resimulate_every: int = 1,

        label_mode: str = "snell_one_step",   # "snell_one_step" or "paper_lookahead"
        target_tau: float = 0.01,
        target_update_every: int = 1,
    ):
        assert kind in ("call", "put")
        assert label_mode in ("snell_one_step", "paper_lookahead")

        self.dev = get_device()
        self.d, self.K, self.r, self.T, self.N = d, K, r, T, N
        self.sig, self.kind = float(sig), kind
        self.label_mode = label_mode

        # Handle S0 as vector or scalar
        if np.isscalar(S0):
            self.S0 = np.full(d, float(S0), dtype=np.float64)
        else:
            S0_array = np.asarray(S0, dtype=np.float64)
            assert S0_array.shape == (d,), f"S0 must be scalar or shape ({d},), got {S0_array.shape}"
            self.S0 = S0_array

        # correlation
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

        # ---- Target networks (for stable bootstrapped labels) ----
        self.target_price_net = copy.deepcopy(self.price_net).to(self.dev).eval()
        self.target_delta_net = copy.deepcopy(self.delta_net).to(self.dev).eval()
        self.target_delta_h0_proj = copy.deepcopy(self.delta_h0_proj).to(self.dev).eval()
        for m in (self.target_price_net, self.target_delta_net, self.target_delta_h0_proj):
            for p in m.parameters():
                p.requires_grad_(False)

        self.target_tau = float(target_tau)
        self.target_update_every = int(max(target_update_every, 1))
        self._opt_step = 0

        self.params = (list(self.price_net.parameters())
                       + list(self.delta_net.parameters())
                       + list(self.delta_h0_proj.parameters()))
        self.opt = torch.optim.Adam(self.params, lr=self.lr)
        self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, factor=0.5, patience=2, verbose=True)

        self.L_vol = None
        self.dt = None
        self.cached_paths_cpu = None
        self._cached_paths_dev = None

        set_seed(self.seed)

    @staticmethod
    def _check_finite(name, *tensors):
        for t in tensors:
            if not torch.isfinite(t).all():
                raise RuntimeError(f"{name}: found NaN/Inf in tensor of shape {t.shape}")

    @torch.no_grad()
    def _soft_update_targets(self):
        tau = self.target_tau
        for tp, p in zip(self.target_price_net.parameters(), self.price_net.parameters()):
            tp.data.mul_(1.0 - tau).add_(p.data, alpha=tau)
        for td, d in zip(self.target_delta_net.parameters(), self.delta_net.parameters()):
            td.data.mul_(1.0 - tau).add_(d.data, alpha=tau)
        for tt, t in zip(self.target_delta_h0_proj.parameters(), self.delta_h0_proj.parameters()):
            tt.data.mul_(1.0 - tau).add_(t.data, alpha=tau)

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

    # ----- Original paper look-ahead labels (kept for comparison) -----
    def _build_lookahead_labels(self, S_batch: torch.Tensor, n: int, kappa_val: float):
        """
        Batch-mean look-ahead (single j* for the whole mini-batch):
        j* = argmax_{j ∈ {n+1,...,N}}  mean_b [ e^{-r (j-n) dt} f(S_j^{(b)}) ].

        Label:
        c_label  = D* f(S_{j*})
        dC_label = D* ∇f(S_{j*}) ⊙ (S_{j*} / S_n)

        Returns (B,1), (B,d)
        """
        B, Np1, d = S_batch.shape
        N = Np1 - 1
        assert 0 <= n < N

        end = N if (self.lookahead_window is None) else min(N, n + self.lookahead_window)
        j_range = torch.arange(n + 1, end + 1, device=S_batch.device, dtype=torch.long)  # (J,)

        S_cand = S_batch[:, j_range, :]  # (B,J,d)
        pay = payoff_arith(S_cand.reshape(-1, d), self.K, self.w, self.kind).view(B, -1)  # (B,J)

        D = torch.exp(-self.r * (j_range.to(dtype=pay.dtype) - float(n)) * self.dt)  # (J,)
        disc_pay = pay * D.unsqueeze(0)        # (B,J)
        means    = disc_pay.mean(dim=0)        # (J,)
        j_idx    = torch.argmax(means)
        j_star   = int(j_range[j_idx].item())
        D_star   = D[j_idx]

        rows    = torch.arange(B, device=S_batch.device)
        S_tilde = S_batch[rows, j_star, :]
        S_n     = S_batch[:, n, :].clamp_min(5e-4)

        if self.smooth_labels:
            if self.smooth_only_at_maturity and (j_star != N):
                f_t = payoff_arith(S_tilde, self.K, self.w, self.kind)
                g_t = grad_payoff_arith(S_tilde, self.K, self.w, self.kind)
            else:
                f_t, g_t = smooth_payoff_and_grad(S_tilde, self.K, self.w, self.kind, kappa_val)
        else:
            f_t = payoff_arith(S_tilde, self.K, self.w, self.kind)
            g_t = grad_payoff_arith(S_tilde, self.K, self.w, self.kind)

        c_label  = (D_star * f_t).unsqueeze(-1)            # (B,1)
        dC_label = (D_star * g_t) * (S_tilde / S_n)        # (B,d)
        return c_label, dC_label

    # ----- NEW: One-step Snell bootstrap labels (policy-consistent) -----
    @torch.no_grad()
    def _build_snell_one_step_labels(self, S_batch: torch.Tensor, y_tgt: torch.Tensor, d_tgt: torch.Tensor, n: int, kappa_eff: float):
        """
        cont_mask = 1{ f_n < y_tgt_n }.

        Label random variables:
            V_{n+1} = max( f_{n+1}, y_tgt_{n+1} )  (or maturity payoff if n+1=N)
            c_label = e^{-r dt} V_{n+1}
            dC_label ≈ e^{-r dt} * (∂V/∂S_{n+1}) ⊙ (S_{n+1}/S_n)   [GBM chain rule]
        """
        B, Np1, d = S_batch.shape
        N = Np1 - 1
        assert 0 <= n < N

        S_n = S_batch[:, n, :].clamp_min(5e-4)

        # use smoothed payoff for boundary if requested
        if self.smooth_labels:
            f_n, _ = smooth_payoff_and_grad(S_batch[:, n, :], self.K, self.w, self.kind, kappa_eff)
        else:
            f_n = payoff_arith(S_batch[:, n, :], self.K, self.w, self.kind)

        ycn = y_tgt[:, n, 0]
        cont_mask = (f_n < ycn).float().unsqueeze(1)  # (B,1)

        disc = torch.exp(torch.tensor(-self.r * self.dt, device=S_batch.device, dtype=S_batch.dtype))

        # next step
        if n + 1 == N:
            S_next = S_batch[:, N, :]
            if self.smooth_labels:
                V_next, gradV_next = smooth_payoff_and_grad(S_next, self.K, self.w, self.kind, kappa_eff)
            else:
                V_next = payoff_arith(S_next, self.K, self.w, self.kind)
                gradV_next = grad_payoff_arith(S_next, self.K, self.w, self.kind)
        else:
            S_next = S_batch[:, n + 1, :]

            if self.smooth_labels:
                f_next, gradf_next = smooth_payoff_and_grad(S_next, self.K, self.w, self.kind, kappa_eff)
            else:
                f_next = payoff_arith(S_next, self.K, self.w, self.kind)
                gradf_next = grad_payoff_arith(S_next, self.K, self.w, self.kind)

            yc_next = y_tgt[:, n + 1, 0]
            d_next  = d_tgt[:, n + 1, :]   # delta wrt S_{n+1}

            ex_next = (f_next >= yc_next).float().unsqueeze(1)
            V_next = torch.maximum(f_next, yc_next)
            gradV_next = ex_next * gradf_next + (1.0 - ex_next) * d_next

        c_label = (disc * V_next).unsqueeze(1)  # (B,1)

        ratio = S_next.clamp_min(5e-4) / S_n
        dC_label = (disc * gradV_next) * ratio  # (B,d)

        return c_label, dC_label, cont_mask

    # ----- BSDE residual loss (masked on continuation region) -----
    @staticmethod
    def _bsde_loss_step(y_n, dY_n, c_lab, dC_lab, S_n, L_vol, dt, r, z_weight: float, cont_mask: torch.Tensor | None = None):
        if cont_mask is None:
            denom = y_n.shape[0]
            w = None
        else:
            denom = cont_mask.sum().clamp_min(1.0)
            w = cont_mask  # (B,1)

        err = (c_lab - y_n).pow(2)  # (B,1)
        val_term = (err.sum() / denom) if (w is None) else ((err * w).sum() / denom)

        Z_y = (S_n * dY_n) @ L_vol.T
        Z_c = (S_n * dC_lab) @ L_vol.T
        xi = Z_c - Z_y
        del_err = xi.pow(2).sum(dim=1, keepdim=True)  # (B,1)
        del_term = (del_err.sum() / denom) if (w is None) else ((del_err * w).sum() / denom)

        loss = val_term + z_weight * (dt / ((1.0 + r * dt) ** 2)) * del_term
        return loss, val_term.detach(), del_term.detach()

    # ----- Training -----
    def train(self):
        for epoch in range(1, self.epochs + 1):

            # (Re)simulate paths every resim_every epochs; otherwise reuse CPU cache
            if (epoch - 1) % self.resim_every == 0 or (self._cached_paths_dev is None):
                S_paths, self.L_vol, self.dt = self._simulate_paths(self.seed + epoch)
                self._cached_paths_dev = S_paths.detach().cpu()
            else:
                S_paths = self._cached_paths_dev.to(self.dev)

            loader = self._make_loader(S_paths)

            # κ ≈ 2/Δt
            kappa_eff = float(max(2.0 / self.dt, 1.0))
            self.price_net.softplus_k.kappa = kappa_eff
            self.target_price_net.softplus_k.kappa = kappa_eff

            self.price_net.train(); self.delta_net.train()
            run_loss = run_val = run_del = 0.0; nb = 0

            for (S_batch,) in loader:
                S_batch = S_batch.to(self.dev, non_blocking=True)   # (B, N+1, d)
                B, Np1, d = S_batch.shape
                N = Np1 - 1

                # Features: X_n = [S_n, g(S_n)]
                S_all = S_batch[:, :N, :]                       # (B,N,d)
                G_all = arithmetic_basket(S_all.reshape(-1, d), self.w).view(B, N, 1)
                g_n = (G_all - self.K) if self.kind == "call" else (self.K - G_all)
                X = torch.cat([S_all, g_n], dim=2)              # (B,N,d+1)
                X_rev = X.flip(dims=[1])                        # (B,N,d+1)

                # Hidden-state init with smoothed maturity info
                S_T = S_batch[:, N, :]
                fT_k, gradT_k = smooth_payoff_and_grad(S_T, self.K, self.w, self.kind, kappa_eff)

                h0_price = fT_k.view(1, B, 1).repeat(self.price_net.gru.num_layers, 1, self.price_net.gru.hidden_size)
                h0_delta_base = self.delta_h0_proj(gradT_k)     # (B,H)
                h0_delta = h0_delta_base.unsqueeze(0).repeat(self.delta_net.gru.num_layers, 1, 1)

                # Forward (online nets)
                y_raw = self.price_net(X_rev, h0=h0_price)      # (B,N,1)
                d_all = self.delta_net(X_rev, h0=h0_delta)      # (B,N,d)
                self._check_finite("pred", y_raw, d_all)

                # price blend (off by default)
                if self.alpha_price >= 1.0:
                    y_all = y_raw
                else:
                    n_idx = torch.arange(N, device=self.dev, dtype=S_batch.dtype)
                    Dn    = torch.exp(-self.r * (self.T - n_idx * self.dt))
                    baseline = fT_k.view(B,1,1) * Dn.view(1, N, 1)
                    y_all = self.alpha_price * y_raw + (1.0 - self.alpha_price) * baseline

                # Delta β-blend (paper)
                n_idx = torch.arange(N, device=self.dev, dtype=S_batch.dtype)
                Dn    = torch.exp(-self.r * (self.T - n_idx * self.dt))
                S_all_clamped = S_all.clamp_min(5e-4)
                base_delta = Dn.view(1,N,1) * (gradT_k.view(B,1,d) * (S_batch[:, N, :].view(B,1,d) / S_all_clamped))
                d_all = self.beta * d_all + (1.0 - self.beta) * base_delta
                self._check_finite("delta_blend", d_all)

                # ---- Target nets outputs (for labeling) ----
                with torch.no_grad():
                    self.target_price_net.eval()
                    self.target_delta_net.eval()
                    self.target_delta_h0_proj.eval()

                    h0_price_tgt = h0_price
                    h0_delta_base_tgt = self.target_delta_h0_proj(gradT_k)
                    h0_delta_tgt = h0_delta_base_tgt.unsqueeze(0).repeat(self.target_delta_net.gru.num_layers, 1, 1)

                    y_tgt_raw = self.target_price_net(X_rev, h0=h0_price_tgt)  # (B,N,1)
                    d_tgt_all = self.target_delta_net(X_rev, h0=h0_delta_tgt)  # (B,N,d)

                    # apply SAME delta blending to targets
                    base_delta_t = Dn.view(1,N,1) * (gradT_k.view(B,1,d) * (S_batch[:, N, :].view(B,1,d) / S_all_clamped))
                    d_tgt_all = self.beta * d_tgt_all + (1.0 - self.beta) * base_delta_t
                    y_tgt_all = y_tgt_raw

                # Accumulate BSDE loss across time
                loss_sum = val_sum = del_sum = 0.0
                for n in range(N):
                    if self.label_mode == "paper_lookahead":
                        c_lab, dC_lab = self._build_lookahead_labels(S_batch, n, kappa_eff)
                        cont_mask = None
                    else:
                        c_lab, dC_lab, cont_mask = self._build_snell_one_step_labels(
                            S_batch, y_tgt_all, d_tgt_all, n, kappa_eff
                        )

                    self._check_finite(f"labels@{n}", c_lab, dC_lab)

                    y_n  = y_all[:, n, :].contiguous()
                    dY_n = d_all[:, n, :].contiguous()
                    S_n  = S_batch[:, n, :].contiguous()

                    loss_n, vterm, dterm = self._bsde_loss_step(
                        y_n, dY_n, c_lab, dC_lab, S_n, self.L_vol, self.dt, self.r,
                        self.z_weight, cont_mask=cont_mask
                    )
                    loss_sum += loss_n
                    val_sum += vterm.item()
                    del_sum += dterm.item()

                loss = loss_sum / N
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                if self.grad_clip and self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.params, self.grad_clip)
                self.opt.step()

                # ---- target update ----
                self._opt_step += 1
                if self._opt_step % self.target_update_every == 0:
                    self._soft_update_targets()

                run_loss += loss.item()
                run_val  += val_sum / N
                run_del  += del_sum / N
                nb += 1

            print(f"Epoch {epoch:02d} | Loss {run_loss/nb:.6f} | Value {run_val/nb:.6f} | Delta {run_del/nb:.6f} | Batches {nb}")
            self.sched.step(run_loss/nb)

            self.cached_paths_cpu = S_paths.detach().to("cpu")
            self._cached_paths_dev = None

        return self

    # ---------------------------
    # Inference helpers (GBM)
    # ---------------------------

    @torch.no_grad()
    def _get_inference_paths(self, seed: int, use_cached_paths: bool) -> torch.Tensor:
        """
        Returns S_paths on CPU with shape (M, N+1, d).
        Ensures self.dt is set.
        """
        if use_cached_paths and (self.cached_paths_cpu is not None):
            if self.dt is None:
                self.dt = self.T / self.N
            return self.cached_paths_cpu.clone()

        S_dev, _, dt = self._simulate_paths(seed)
        self.dt = dt
        return S_dev.detach().cpu()

    @torch.no_grad()
    def _eval_nets_on_paths(
        self,
        S_paths_cpu: torch.Tensor,
        inference_batch_size: int | None = 4096,
        use_target_nets: bool = False,
    ):
        """
        Evaluate nets on CPU paths in GPU batches.

        Returns (all on CPU):
            y_all_paths: (M,N)    continuation predictions y_n(path)
            d_all_paths: (M,N,d)  network deltas Δ_n(path)
            f_all:       (M,N)    immediate payoff f(S_n(path))
            f_T:         (M,)     payoff at maturity f(S_N(path))
        """
        dev = self.dev
        M, Np1, d = S_paths_cpu.shape
        N = Np1 - 1

        dt = self.dt if self.dt is not None else (self.T / self.N)
        kappa_eff = float(max(2.0 / dt, 1.0))

        # pick nets
        if use_target_nets:
            price_net = self.target_price_net
            delta_net = self.target_delta_net
            delta_h0_proj = self.target_delta_h0_proj
        else:
            price_net = self.price_net
            delta_net = self.delta_net
            delta_h0_proj = self.delta_h0_proj

        price_net.eval()
        delta_net.eval()
        delta_h0_proj.eval()

        price_net.softplus_k.kappa = kappa_eff
        if use_target_nets:
            self.target_price_net.softplus_k.kappa = kappa_eff

        # ---- compute payoffs on CPU (avoid moving all paths to GPU) ----
        w_cpu = self.w.detach().cpu()
        S_all_cpu = S_paths_cpu[:, :N, :]  # (M,N,d)

        f_all = payoff_arith(S_all_cpu.reshape(-1, d), self.K, w_cpu, self.kind).view(M, N)  # (M,N)
        f_T = payoff_arith(S_paths_cpu[:, N, :], self.K, w_cpu, self.kind)                   # (M,)

        # batching for net eval
        if inference_batch_size is None:
            inference_batch_size = M

        y_list, d_list = [], []
        for s in range(0, M, inference_batch_size):
            e = min(s + inference_batch_size, M)
            Sb = S_paths_cpu[s:e].to(dev, non_blocking=True)  # (b,N+1,d)
            b = Sb.shape[0]

            # features X_n = [S_n, g(S_n)]
            S_all_b = Sb[:, :N, :]  # (b,N,d)
            G_b = arithmetic_basket(S_all_b.reshape(-1, d), self.w).view(b, N, 1)
            g_b = (G_b - self.K) if self.kind == "call" else (self.K - G_b)
            X = torch.cat([S_all_b, g_b], dim=2)   # (b,N,d+1)
            X_rev = X.flip(dims=[1])

            # h0 from smoothed maturity payoff
            S_Tb = Sb[:, N, :]
            fT_k, gradT_k = smooth_payoff_and_grad(S_Tb, self.K, self.w, self.kind, kappa_eff)
            h0_price = fT_k.view(1, b, 1).repeat(price_net.gru.num_layers, 1, price_net.gru.hidden_size)
            h0_delta = delta_h0_proj(gradT_k).unsqueeze(0).repeat(delta_net.gru.num_layers, 1, 1)

            y_raw = price_net(X_rev, h0=h0_price)  # (b,N,1)
            d_raw = delta_net(X_rev, h0=h0_delta)  # (b,N,d)

            # optional price blend (normally alpha_price=1.0)
            if self.alpha_price >= 1.0:
                y_all = y_raw
            else:
                n_idx = torch.arange(N, device=dev, dtype=Sb.dtype)
                Dn = torch.exp(-self.r * (self.T - n_idx * dt))  # (N,)
                baseline = fT_k.view(b, 1, 1) * Dn.view(1, N, 1)
                y_all = self.alpha_price * y_raw + (1.0 - self.alpha_price) * baseline

            # delta β-blend (same as training)
            n_idx = torch.arange(N, device=dev, dtype=Sb.dtype)
            Dn = torch.exp(-self.r * (self.T - n_idx * dt))
            S_all_clamped = S_all_b.clamp_min(5e-4)
            base_delta = Dn.view(1, N, 1) * (gradT_k.view(b, 1, d) * (Sb[:, N, :].view(b, 1, d) / S_all_clamped))
            d_all = self.beta * d_raw + (1.0 - self.beta) * base_delta

            y_list.append(y_all.squeeze(-1).detach().cpu())  # (b,N)
            d_list.append(d_all.detach().cpu())              # (b,N,d)

        y_all_paths = torch.cat(y_list, dim=0)  # (M,N)
        d_all_paths = torch.cat(d_list, dim=0)  # (M,N,d)
        return y_all_paths, d_all_paths, f_all, f_T

    @torch.no_grad()
    def price_at_all_time(
        self,
        seed: int = 777,
        inference_batch_size: int | None = 4096,
        use_cached_paths: bool = False,
        use_target_nets: bool = False,
        return_continuation: bool = False,
    ):
        """
        Pathwise diagnostics (NOT the true price function at each time, but useful checks).

        Returns:
            V_all_mean:     (N,)   mean over paths of max(f_n(path), y_n(path))
            delta_all_mean: (N,d)  mean over paths of [ exercise payoff-grad : net-delta ]
            ex_prob:        (N,)   P(exercise at n under rule f_n >= y_n)  (unconditional)
        Optionally:
            y_all_mean:     (N,)   mean continuation prediction (if return_continuation=True)
        """
        S_paths = self._get_inference_paths(seed, use_cached_paths)   # (M,N+1,d) CPU
        M, Np1, d = S_paths.shape
        N = Np1 - 1

        y_all_paths, d_all_paths, f_all, _ = self._eval_nets_on_paths(
            S_paths, inference_batch_size=inference_batch_size, use_target_nets=use_target_nets
        )  # all CPU

        # exercise indicator at each time on each path
        ex_now = (f_all >= y_all_paths)  # (M,N) bool

        # V_n(path) = max(f_n(path), y_n(path)) then average
        V_all_paths = torch.maximum(f_all, y_all_paths)  # (M,N)
        V_all_mean = V_all_paths.mean(dim=0).numpy()     # (N,)

        # policy-consistent deltas at each time: exercise -> payoff gradient, else -> net delta
        w_cpu = self.w.detach().cpu()
        delta_policy_means = []
        for n in range(N):
            S_n = S_paths[:, n, :]  # (M,d) CPU
            grad_n = grad_payoff_arith(S_n, self.K, w_cpu, self.kind)  # (M,d)
            exn = ex_now[:, n].float().view(M, 1)                      # (M,1)
            d_pol = exn * grad_n + (1.0 - exn) * d_all_paths[:, n, :]  # (M,d)
            delta_policy_means.append(d_pol.mean(dim=0).view(1, d))

        delta_all_mean = torch.cat(delta_policy_means, dim=0).numpy()  # (N,d)
        ex_prob = ex_now.float().mean(dim=0).numpy()                   # (N,)

        if not return_continuation:
            return V_all_mean, delta_all_mean, ex_prob

        y_all_mean = y_all_paths.mean(dim=0).numpy()
        return y_all_mean, V_all_mean, delta_all_mean, ex_prob

    @torch.no_grad()
    def price_at_t0(
        self,
        seed: int = 777,
        inference_batch_size: int | None = 4096,
        use_cached_paths: bool = False,
        use_target_nets: bool = False,
        return_diag: bool = False,
    ):
        """
        Policy evaluation on simulated paths:
          exercise at the FIRST time n where f_n(path) >= y_n(path),
          else exercise at maturity.

        Returns by default:
            price_hat, delta0_hat
        If return_diag=True:
            price_hat, delta0_hat, diag
        """
        S_paths = self._get_inference_paths(seed, use_cached_paths)  # CPU
        M, Np1, d = S_paths.shape
        N = Np1 - 1
        dt = self.dt if self.dt is not None else (self.T / self.N)

        y_all_paths, d_all_paths, f_all, f_T = self._eval_nets_on_paths(
            S_paths, inference_batch_size=inference_batch_size, use_target_nets=use_target_nets
        )  # CPU

        # stopping rule: first n where f_n >= y_n; maturity always exercise
        ex_now = (f_all >= y_all_paths)  # (M,N)
        ex_ext = torch.cat([ex_now, torch.ones(M, 1, dtype=torch.bool)], dim=1)  # (M,N+1)
        pay_ext = torch.cat([f_all, f_T.view(M, 1)], dim=1)                      # (M,N+1)

        tau = ex_ext.float().argmax(dim=1).long()                                # (M,)

        disc = torch.exp(-self.r * tau.float() * dt)                             # (M,)
        payoff_tau = pay_ext.gather(1, tau.view(M, 1)).squeeze(1)                # (M,)
        pv = disc * payoff_tau                                                   # (M,)

        price_hat = pv.mean().item()
        stderr = pv.std(unbiased=True).item() / math.sqrt(max(M, 1))

        # delta0 under the same decision: if exercised at 0 -> payoff grad else net delta at 0
        w_cpu = self.w.detach().cpu()
        S0_paths = S_paths[:, 0, :]                                # (M,d)
        grad0 = grad_payoff_arith(S0_paths, self.K, w_cpu, self.kind)  # (M,d)
        ex0 = ex_now[:, 0].float().view(M, 1)                      # (M,1)
        delta0_paths = ex0 * grad0 + (1.0 - ex0) * d_all_paths[:, 0, :]  # (M,d)
        delta0_hat = delta0_paths.mean(dim=0).numpy()

        if not return_diag:
            return price_hat, delta0_hat

        hist = torch.bincount(tau, minlength=N + 1).cpu().numpy()
        diag = {
            "price_stderr": float(stderr),
            "exercise_counts": hist,
            "exercise_probs": hist / hist.sum(),
            "mean_exercise_index": float(tau.float().mean().item()),
            "mean_exercise_time": float(tau.float().mean().item() * dt),
            # keep y0 only as a DEBUG number (not reported)
            "y0_mean_debug": float(y_all_paths[:, 0].mean().item()),
        }
        return price_hat, delta0_hat, diag

# ---------------------------
# Minimal usage example
# ---------------------------
if __name__ == "__main__":
    d = 5

    corr_matrix = np.eye(d)
    corr_matrix.fill(0.5)
    np.fill_diagonal(corr_matrix, 1.0)

    S0_vector = np.array([98, 98, 180, 150, 104])
    weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

    trainer = RNNAmericanTrainer(
        d=d, S0=S0_vector, K=100.0, r=0.02, T=0.5, N=126,
        sig=0.5, corr=corr_matrix, kind="put", weights=weights,
        M=50000, batch_size=4096, epochs=5, seed=12345,
        hidden_dim=64, num_layers=3,
        lr=1e-3, grad_clip=1.0,

        alpha_price=1.0,
        beta=0.5,
        z_weight=1.0,

        smooth_labels=True, smooth_only_at_maturity=False,
        lookahead_window=None,
        shuffle=False, drop_last=True,
        resimulate_every=1,

        # ---- key switches ----
        label_mode="snell_one_step",     
        target_tau=0.01,
        target_update_every=1,
    )

    trainer.train()

    price_hat, delta0_hat, diag = trainer.price_at_t0(
        inference_batch_size=4096,
        use_cached_paths=True,
        return_diag=True
    )

    print(f"\n--- Results ---")
    print("t0 American price:", round(price_hat, 6), "stderr:", round(diag["price_stderr"], 6))
    print("t0 deltas:", np.round(delta0_hat, 6))
    print("mean exercise time:", round(diag["mean_exercise_time"], 6))