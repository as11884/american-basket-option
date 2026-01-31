# heston_rnn_model_snell_fixed.py
from __future__ import annotations

import math
import copy
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
        g = G - K
        grad_g = grad_arith(S, w)
    else:
        g = K - G
        grad_g = -grad_arith(S, w)

    f_k  = F.softplus(kappa * g) / max(kappa, 1e-8)
    sig  = torch.sigmoid(kappa * g).unsqueeze(1)
    grad_fk = sig * grad_g
    return f_k, grad_fk


# ---------------------------
# Three heads: price, delta, alpha(stock-shock)
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
        return a_rev.index_select(1, idx)     # (B,N,d)


# ---------------------------
# OLS helper for α-label
# ---------------------------

def _ridge_ols_alpha(F: torch.Tensor, X: torch.Tensor, ridge: float = 1e-6) -> torch.Tensor:
    """
    F: (B,)   response (discounted increment)/sqrt(dt)
    X: (B,d)  regressors ≈ L ε  (stock Brownian shocks scaled)
    returns α_c: (d,)  batch-shared slope in X-space
    """
    Xt = X.transpose(0, 1)                 # (d,B)
    G  = Xt @ X                            # (d,d)
    if ridge and ridge > 0:
        G = G + ridge * torch.eye(G.shape[0], device=X.device, dtype=X.dtype)
    b  = Xt @ F                            # (d,)
    return torch.linalg.solve(G, b)        # (d,)


# ---------------------------
# Trainer (Heston + Snell one-step bootstrap labels)
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
        beta: float = 0.5,
        alpha_price: float = 1.0,
        z_weight: float = 1.0,
        delta_aux_weight: float = 0.0,
        smooth_labels: bool = True, smooth_only_at_maturity: bool = False,
        lookahead_window: int | None = None,
        shuffle: bool = False, drop_last: bool = True,
        resimulate_every: int = 1,

        # ---- NEW: labeling mode / target nets ----
        label_mode: str = "snell_one_step",   # "snell_one_step" or "paper_lookahead"
        target_tau: float = 0.01,
        target_update_every: int = 1,
        ridge_alpha: float = 1e-6,
    ):
        assert kind in ("call", "put")
        assert label_mode in ("snell_one_step", "paper_lookahead")

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

        # NEW labeling controls
        self.label_mode = label_mode
        self.target_tau = float(target_tau)
        self.target_update_every = int(max(target_update_every, 1))
        self.ridge_alpha = float(ridge_alpha)
        self._opt_step = 0

        # nets: input features = [S, sqrt(v), g] ⇒ 2d+1
        inp_dim = 2 * d + 1
        self.price_net = PriceGRU(inp_dim, hidden_dim, num_layers, kappa).to(self.dev)
        self.delta_net = DeltaGRU(inp_dim, d, hidden_dim, num_layers, activation=self.delta_activation).to(self.dev)
        self.alpha_net = AlphaGRU(inp_dim, d, hidden_dim, num_layers).to(self.dev)

        # delta init projection
        self.delta_h0_proj = nn.Linear(d, self.delta_net.gru.hidden_size, bias=False).to(self.dev)

        # ---- Target networks (for stable bootstrapped labels) ----
        self.target_price_net = copy.deepcopy(self.price_net).to(self.dev).eval()
        self.target_delta_net = copy.deepcopy(self.delta_net).to(self.dev).eval()
        self.target_alpha_net = copy.deepcopy(self.alpha_net).to(self.dev).eval()
        self.target_delta_h0_proj = copy.deepcopy(self.delta_h0_proj).to(self.dev).eval()
        for m in (self.target_price_net, self.target_delta_net, self.target_alpha_net, self.target_delta_h0_proj):
            for p in m.parameters():
                p.requires_grad_(False)

        # optimizer & scheduler
        self.params = (list(self.price_net.parameters())
                       + list(self.delta_net.parameters())
                       + list(self.alpha_net.parameters())
                       + list(self.delta_h0_proj.parameters()))
        self.opt = torch.optim.Adam(self.params, lr=self.lr)
        self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, factor=0.5, patience=2, verbose=True)

        # correlation factor for stock Brownian
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
                raise RuntimeError(f"{name}: found NaN/Inf in tensor of shape {t.shape}")

    @torch.no_grad()
    def _soft_update_targets(self):
        tau = self.target_tau
        for tp, p in zip(self.target_price_net.parameters(), self.price_net.parameters()):
            tp.data.mul_(1.0 - tau).add_(p.data, alpha=tau)
        for td, d in zip(self.target_delta_net.parameters(), self.delta_net.parameters()):
            td.data.mul_(1.0 - tau).add_(d.data, alpha=tau)
        for ta, a in zip(self.target_alpha_net.parameters(), self.alpha_net.parameters()):
            ta.data.mul_(1.0 - tau).add_(a.data, alpha=tau)
        for th, h in zip(self.target_delta_h0_proj.parameters(), self.delta_h0_proj.parameters()):
            th.data.mul_(1.0 - tau).add_(h.data, alpha=tau)

    def _alpha_to_shares(self, alpha_vec: torch.Tensor, S_vec: torch.Tensor, v_vec: torch.Tensor) -> torch.Tensor:
        """
        alpha_vec: (d,) coefficient in X-space (≈ L ε) for stock shocks
        Hedge shares: h = alpha / (S * sqrt(v)) elementwise.
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
        if hasattr(hes, "get_variance_paths"):
            V_np = hes.get_variance_paths()
        elif hasattr(hes, "variance_paths"):
            V_np = hes.variance_paths
        else:
            raise RuntimeError("CorrelatedHeston must expose variance paths via get_variance_paths() or .variance_paths")

        S = torch.from_numpy(S_np).float().to(self.dev, non_blocking=True)
        V = torch.from_numpy(V_np).float().to(self.dev, non_blocking=True)
        dt = self.T / self.N
        return S, V, dt

    def _make_loader(self, S_paths: torch.Tensor, V_paths: torch.Tensor):
        ds = TensorDataset(S_paths, V_paths)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=self.shuffle, drop_last=self.drop_last)

    # ----- Original paper look-ahead labels (kept for comparison) -----
    def _build_lookahead_labels(self, S_batch: torch.Tensor, n: int, kappa_val: float):
        B, Np1, d = S_batch.shape
        N = Np1 - 1
        assert 0 <= n < N
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
            if self.smooth_only_at_maturity and (j_star != N):
                f_t = payoff_arith(S_tilde, self.K, self.w, self.kind)
                g_t = grad_payoff_arith(S_tilde, self.K, self.w, self.kind)
            else:
                f_t, g_t = smooth_payoff_and_grad(S_tilde, self.K, self.w, self.kind, kappa_val)
        else:
            f_t = payoff_arith(S_tilde, self.K, self.w, self.kind)
            g_t = grad_payoff_arith(S_tilde, self.K, self.w, self.kind)

        c_label  = (D_star * f_t).unsqueeze(-1)            # (B,1)
        dc_label = (D_star * g_t) * (S_tilde / S_n)        # (B,d)
        return c_label, dc_label, None, None  # no mask / no V_next

    # ----- NEW: Snell one-step bootstrap label -----
    @torch.no_grad()
    def _build_snell_one_step_labels(
        self,
        S_batch: torch.Tensor,
        V_batch: torch.Tensor,
        y_tgt: torch.Tensor,       # (B,N,1) continuation targets
        d_tgt: torch.Tensor,       # (B,N,d) deltas at S_n targets
        n: int,
        kappa_eff: float
    ):
        """
        Continuation label:
            c_n = e^{-r dt} V_{n+1},  V_{n+1} = max(f_{n+1}, y_{n+1}^{tgt})
        Delta label (GBM-style chain rule still holds pathwise):
            dC_n ≈ e^{-r dt} (∂V_{n+1}/∂S_{n+1}) ⊙ (S_{n+1}/S_n)
        Mask:
            1{ f_n < y_n^{tgt} }  (only enforce BSDE on continuation region)
        """
        B, Np1, d = S_batch.shape
        N = Np1 - 1
        assert 0 <= n < N

        disc = torch.exp(torch.tensor(-self.r * self.dt, device=S_batch.device, dtype=S_batch.dtype))

        # continuation mask at time n
        S_n = S_batch[:, n, :].clamp_min(5e-4)
        if self.smooth_labels:
            f_n, _ = smooth_payoff_and_grad(S_batch[:, n, :], self.K, self.w, self.kind, kappa_eff)
        else:
            f_n = payoff_arith(S_batch[:, n, :], self.K, self.w, self.kind)
        ycn = y_tgt[:, n, 0]
        cont_mask = (f_n < ycn).float().unsqueeze(1)  # (B,1)

        # build V_{n+1}
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
            d_next  = d_tgt[:, n + 1, :]  # delta wrt S_{n+1}
            ex_next = (f_next >= yc_next).float().unsqueeze(1)

            V_next = torch.maximum(f_next, yc_next)
            gradV_next = ex_next * gradf_next + (1.0 - ex_next) * d_next

        # continuation label at n
        c_label = (disc * V_next).unsqueeze(1)  # (B,1)

        # delta label at n (chain rule via S_{n+1}/S_n)
        ratio = S_next.clamp_min(5e-4) / S_n
        dc_label = (disc * gradV_next) * ratio  # (B,d)

        return c_label, dc_label, cont_mask, V_next  # V_next returned for alpha-label

    # ----- Training -----
    def train(self):
        for epoch in range(1, self.epochs + 1):

            if (epoch - 1) % self.resim_every == 0 or (self._cached_paths_dev is None):
                S_paths, V_paths, self.dt = self._simulate_paths(self.seed + epoch)
                self._cached_paths_dev = (S_paths.detach().cpu(), V_paths.detach().cpu())
            else:
                S_paths, V_paths = self._cached_paths_dev
                S_paths = S_paths.to(self.dev)
                V_paths = V_paths.to(self.dev)

            loader = self._make_loader(S_paths, V_paths)

            kappa_eff = float(max(2.0 / self.dt, 1.0))
            self.price_net.softplus_k.kappa = kappa_eff
            self.target_price_net.softplus_k.kappa = kappa_eff

            self.price_net.train(); self.delta_net.train(); self.alpha_net.train()
            run_loss = run_val = run_z = 0.0; nb = 0

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
                X_rev     = torch.cat([S_all.flip(1), sqrtV_all.flip(1), g_n.flip(1)], dim=2)  # (B,N,2d+1)

                # Hidden-state inits (based on smoothed maturity payoff)
                S_T = S_batch[:, N, :]
                fT_k, gradT_k = smooth_payoff_and_grad(S_T, self.K, self.w, self.kind, kappa_eff)
                h0_price = fT_k.view(1, B, 1).repeat(self.price_net.gru.num_layers, 1, self.price_net.gru.hidden_size)
                h0_delta = self.delta_h0_proj(gradT_k).unsqueeze(0).repeat(self.delta_net.gru.num_layers, 1, 1)

                # ---- Online nets ----
                y_raw   = self.price_net(X_rev, h0=h0_price)    # (B,N,1)
                d_all   = self.delta_net(X_rev, h0=h0_delta)    # (B,N,d)
                alpha_y = self.alpha_net(X_rev)                 # (B,N,d)
                self._check_finite("pred", y_raw, d_all, alpha_y)

                # Optional price blend
                if self.alpha_price >= 1.0:
                    y_all = y_raw
                else:
                    n_idx = torch.arange(N, device=self.dev, dtype=S_batch.dtype)
                    Dn    = torch.exp(-self.r * (self.T - n_idx * self.dt))
                    baseline = fT_k.view(B,1,1) * Dn.view(1, N, 1)
                    y_all = self.alpha_price * y_raw + (1.0 - self.alpha_price) * baseline

                # β-blend for deltas (same as your old code)
                n_idx = torch.arange(N, device=self.dev, dtype=S_batch.dtype)
                Dn    = torch.exp(-self.r * (self.T - n_idx * self.dt))
                S_all_clamped = S_all.clamp_min(5e-4)
                base_delta = Dn.view(1,N,1) * (gradT_k.view(B,1,d) * (S_batch[:, N, :].view(B,1,d) / S_all_clamped))
                d_all = self.beta * d_all + (1.0 - self.beta) * base_delta
                self._check_finite("delta_blend", d_all)

                # ---- Target nets (for labels) ----
                with torch.no_grad():
                    self.target_price_net.eval()
                    self.target_delta_net.eval()
                    self.target_alpha_net.eval()
                    self.target_delta_h0_proj.eval()

                    h0_delta_tgt = self.target_delta_h0_proj(gradT_k).unsqueeze(0).repeat(self.target_delta_net.gru.num_layers, 1, 1)
                    y_tgt = self.target_price_net(X_rev, h0=h0_price)          # (B,N,1)
                    d_tgt = self.target_delta_net(X_rev, h0=h0_delta_tgt)      # (B,N,d)

                    # apply same β-blend to target deltas
                    base_delta_t = Dn.view(1,N,1) * (gradT_k.view(B,1,d) * (S_batch[:, N, :].view(B,1,d) / S_all_clamped))
                    d_tgt = self.beta * d_tgt + (1.0 - self.beta) * base_delta_t

                # ---- Loss across time ----
                loss_sum = val_sum = z_sum = 0.0
                z_scale = self.z_weight * (self.dt / ((1.0 + self.r*self.dt) ** 2))
                sqrt_dt = math.sqrt(self.dt)

                for n in range(N):
                    if self.label_mode == "paper_lookahead":
                        c_n, dC_n, cont_mask, V_next = self._build_lookahead_labels(S_batch, n, kappa_eff)
                        # no mask in paper mode
                        cont_mask = None
                        V_next = None
                    else:
                        c_n, dC_n, cont_mask, V_next = self._build_snell_one_step_labels(
                            S_batch, V_batch, y_tgt, d_tgt, n, kappa_eff
                        )

                    # Value term (masked on continuation)
                    y_n = y_all[:, n, :].contiguous()  # (B,1)
                    err = (c_n - y_n).pow(2)           # (B,1)
                    if cont_mask is None:
                        val_term = err.mean()
                        denom = None
                    else:
                        denom = cont_mask.sum().clamp_min(1.0)
                        val_term = (err * cont_mask).sum() / denom

                    # ---- α-label (stock-shock exposure) via OLS on discounted increment ----
                    # Build response using target continuation at n and Snell value at n+1
                    # Y_n = e^{-r t_n} * y_tgt_n  (continuation)
                    # Y_{n+1} = e^{-r t_{n+1}} * V_{n+1} (Snell)
                    t_n = n * self.dt
                    disc_n = math.exp(-self.r * t_n)
                    disc_np1 = math.exp(-self.r * (t_n + self.dt))

                    y_n_tgt = y_tgt[:, n, 0]  # (B,)

                    if self.label_mode == "paper_lookahead":
                        # keep old behavior for paper mode (not recommended)
                        # approximate "next" as target continuation (or itself at end)
                        y_np1 = y_tgt[:, n+1, 0] if (n+1 < N) else y_tgt[:, n, 0]
                        Y_np1 = disc_np1 * y_np1
                    else:
                        # Snell at n+1 (already computed): V_next is at time n+1
                        # for n=N-1, V_next is payoff at maturity
                        Y_np1 = disc_np1 * V_next

                    Y_n = disc_n * y_n_tgt
                    F_n = (Y_np1 - Y_n) / max(sqrt_dt, 1e-12)  # (B,)

                    # regressors Xn ≈ L ε  from standardized log returns
                    S_n_vec = S_batch[:, n, :].clamp_min(5e-4)
                    S_np1_vec = S_batch[:, n+1, :]
                    v_n_vec = V_batch[:, n, :].clamp_min(1e-12)
                    sqrt_vn = torch.sqrt(v_n_vec)

                    log_ret = torch.log(S_np1_vec / S_n_vec.clamp_min(1e-12))  # (B,d)
                    drift   = (self.r - 0.5 * v_n_vec) * self.dt
                    Xn      = (log_ret - drift) / (sqrt_vn * max(sqrt_dt, 1e-12))  # (B,d)

                    alpha_c = _ridge_ols_alpha(F_n, Xn, ridge=self.ridge_alpha)     # (d,)

                    z_err = (alpha_y[:, n, :] - alpha_c.unsqueeze(0)).pow(2).sum(dim=1, keepdim=True)  # (B,1)

                    if cont_mask is None:
                        z_term = z_err.mean()
                    else:
                        z_term = (z_err * cont_mask).sum() / denom

                    # optional delta auxiliary (masked)
                    if self.delta_aux_weight > 0:
                        dY_n = d_all[:, n, :].contiguous()
                        d_aux = (dY_n - dC_n).pow(2).sum(dim=1, keepdim=True)  # (B,1)
                        if cont_mask is None:
                            delta_aux = d_aux.mean()
                        else:
                            delta_aux = (d_aux * cont_mask).sum() / denom
                    else:
                        delta_aux = torch.tensor(0.0, device=self.dev)

                    loss_n = val_term + z_scale * z_term + self.delta_aux_weight * delta_aux

                    loss_sum += loss_n
                    val_sum  += float(val_term.item())
                    z_sum    += float(z_term.item())

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
                run_z    += z_sum / N
                nb += 1

            print(f"Epoch {epoch:02d} | Loss {run_loss/nb:.6f} | Value {run_val/nb:.6f} | "
                  f"Z(stock) {run_z/nb:.6f} (scaled {z_scale*run_z/nb:.6f}) | Batches {nb}")
            self.sched.step(run_loss/nb)

            # cache last epoch paths
            self.cached_paths_cpu = S_paths.detach().to("cpu")
            self.cached_var_cpu   = V_paths.detach().to("cpu")

        return self

    # ---------------------------
    # Inference helpers (Heston)
    # ---------------------------

    @torch.no_grad()
    def _get_inference_paths(self, seed: int, use_cached_paths: bool):
        """
        Returns (S_paths_cpu, V_paths_cpu) with shapes:
          S: (M, N+1, d), V: (M, N+1, d)
        Also ensures self.dt is set.
        """
        if use_cached_paths and (self.cached_paths_cpu is not None) and (self.cached_var_cpu is not None):
            if self.dt is None:
                self.dt = self.T / self.N
            return self.cached_paths_cpu.clone(), self.cached_var_cpu.clone()

        S_dev, V_dev, dt = self._simulate_paths(seed)
        self.dt = dt
        return S_dev.detach().cpu(), V_dev.detach().cpu()

    @torch.no_grad()
    def _eval_nets_on_paths(
        self,
        S_paths_cpu: torch.Tensor,
        V_paths_cpu: torch.Tensor,
        inference_batch_size: int | None = 4096,
        use_target_nets: bool = False,
    ):
        """
        Evaluate nets on CPU paths in GPU batches.

        Returns (all on CPU):
            y_all_paths: (M,N)    continuation predictions y_n(path)
            d_all_paths: (M,N,d)  delta head Δ_n(path)
            a_all_paths: (M,N,d)  alpha head α_n(path)  (stock-shock exposure)
            f_all:       (M,N)    immediate payoff f(S_n(path))
            f_T:         (M,)     payoff at maturity f(S_N(path))
        """
        dev = self.dev
        M, Np1, d = S_paths_cpu.shape
        N = Np1 - 1

        dt = float(self.dt) if (self.dt is not None) else float(self.T / self.N)
        kappa_eff = float(max(2.0 / dt, 1.0))

        # pick nets
        if use_target_nets:
            price_net = self.target_price_net
            delta_net = self.target_delta_net
            alpha_net = self.target_alpha_net
            delta_h0_proj = self.target_delta_h0_proj
        else:
            price_net = self.price_net
            delta_net = self.delta_net
            alpha_net = self.alpha_net
            delta_h0_proj = self.delta_h0_proj

        price_net.eval()
        delta_net.eval()
        alpha_net.eval()
        delta_h0_proj.eval()
        price_net.softplus_k.kappa = kappa_eff

        # payoffs on CPU
        w_cpu = self.w.detach().cpu()
        f_all = payoff_arith(
            S_paths_cpu[:, :N, :].reshape(-1, d),
            self.K, w_cpu, self.kind
        ).view(M, N)  # (M,N)
        f_T = payoff_arith(S_paths_cpu[:, N, :], self.K, w_cpu, self.kind)  # (M,)

        # batching
        if inference_batch_size is None:
            inference_batch_size = M

        y_list, d_list, a_list = [], [], []
        w_dev = self.w.to(dev)

        for s in range(0, M, inference_batch_size):
            e = min(s + inference_batch_size, M)

            Sb = S_paths_cpu[s:e].to(dev, non_blocking=True)  # (b,N+1,d)
            Vb = V_paths_cpu[s:e].to(dev, non_blocking=True)  # (b,N+1,d)
            b = Sb.shape[0]

            # features: [S_n, sqrt(v_n), g_n]
            S_all = Sb[:, :N, :]                                        # (b,N,d)
            sqrtV_all = torch.sqrt(Vb[:, :N, :].clamp_min(0.0))          # (b,N,d)
            G_all = arithmetic_basket(S_all.reshape(-1, d), w_dev).view(b, N, 1)
            g_n = (G_all - self.K) if self.kind == "call" else (self.K - G_all)

            X = torch.cat([S_all, sqrtV_all, g_n], dim=2)                # (b,N,2d+1)
            X_rev = X.flip(dims=[1])

            # init from smoothed maturity payoff
            S_T = Sb[:, N, :]
            fT_k, gradT_k = smooth_payoff_and_grad(S_T, self.K, w_dev, self.kind, kappa_eff)

            h0_price = fT_k.view(1, b, 1).repeat(price_net.gru.num_layers, 1, price_net.gru.hidden_size)
            h0_delta = delta_h0_proj(gradT_k).unsqueeze(0).repeat(delta_net.gru.num_layers, 1, 1)

            y_raw = price_net(X_rev, h0=h0_price)       # (b,N,1)
            d_raw = delta_net(X_rev, h0=h0_delta)       # (b,N,d)
            a_raw = alpha_net(X_rev)                    # (b,N,d)

            # optional price blend
            if self.alpha_price >= 1.0:
                y_all = y_raw
            else:
                n_idx = torch.arange(N, device=dev, dtype=Sb.dtype)
                Dn = torch.exp(-self.r * (self.T - n_idx * dt))
                baseline = fT_k.view(b, 1, 1) * Dn.view(1, N, 1)
                y_all = self.alpha_price * y_raw + (1.0 - self.alpha_price) * baseline

            # delta β-blend (same as training)
            n_idx = torch.arange(N, device=dev, dtype=Sb.dtype)
            Dn = torch.exp(-self.r * (self.T - n_idx * dt))
            S_all_clamped = S_all.clamp_min(5e-4)
            base_delta = Dn.view(1, N, 1) * (gradT_k.view(b, 1, d) * (Sb[:, N, :].view(b, 1, d) / S_all_clamped))
            d_all = self.beta * d_raw + (1.0 - self.beta) * base_delta

            y_list.append(y_all.squeeze(-1).detach().cpu())  # (b,N)
            d_list.append(d_all.detach().cpu())              # (b,N,d)
            a_list.append(a_raw.detach().cpu())              # (b,N,d)

        y_all_paths = torch.cat(y_list, dim=0)  # (M,N)
        d_all_paths = torch.cat(d_list, dim=0)  # (M,N,d)
        a_all_paths = torch.cat(a_list, dim=0)  # (M,N,d)
        return y_all_paths, d_all_paths, a_all_paths, f_all, f_T

    # ---------------------------
    # Public: diagnostics across all times (pathwise max then average)
    # ---------------------------

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
        Pathwise diagnostics (NOT the survival-conditioned value curve):
            V_all_mean[n] = E[ max(f_n(path), y_n(path)) ].
            delta_all_mean[n] = E[ exercise? payoff-grad : delta_head ].
            alpha_all_mean[n] = E[ (1-exercise) * alpha_head ]  (alpha irrelevant if exercised).
            ex_prob[n] = P(f_n >= y_n).

        By default returns NO continuation mean.
        """
        S_paths, V_paths = self._get_inference_paths(seed, use_cached_paths)
        M, Np1, d = S_paths.shape
        N = Np1 - 1

        y_paths, d_paths, a_paths, f_all, _ = self._eval_nets_on_paths(
            S_paths, V_paths,
            inference_batch_size=inference_batch_size,
            use_target_nets=use_target_nets
        )

        ex_now = (f_all >= y_paths)  # (M,N)
        ex_prob = ex_now.float().mean(dim=0).numpy()

        # V diag: pathwise max then mean
        V_all_mean = torch.maximum(f_all, y_paths).mean(dim=0).numpy()  # (N,)

        # policy-consistent deltas/alphas at each time
        w_cpu = self.w.detach().cpu()
        delta_means = []
        alpha_means = []
        for n in range(N):
            S_n = S_paths[:, n, :]  # (M,d)
            grad_n = grad_payoff_arith(S_n, self.K, w_cpu, self.kind)  # (M,d)
            exn = ex_now[:, n].float().view(M, 1)

            d_pol = exn * grad_n + (1.0 - exn) * d_paths[:, n, :]
            a_pol = (1.0 - exn) * a_paths[:, n, :]  # alpha only used when continuing

            delta_means.append(d_pol.mean(dim=0).view(1, d))
            alpha_means.append(a_pol.mean(dim=0).view(1, d))

        delta_all_mean = torch.cat(delta_means, dim=0).numpy()  # (N,d)
        alpha_all_mean = torch.cat(alpha_means, dim=0).numpy()  # (N,d)

        if not return_continuation:
            return V_all_mean, delta_all_mean, alpha_all_mean, ex_prob

        y_all_mean = y_paths.mean(dim=0).numpy()
        return y_all_mean, V_all_mean, delta_all_mean, alpha_all_mean, ex_prob

    # ---------------------------
    # Public: t=0 policy evaluation (this is the price you should report)
    # ---------------------------

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
        Policy evaluation:
          tau(path) = first n where f_n(path) >= y_n(path), else maturity.
          price_hat = E[ exp(-r tau dt) * payoff_tau ].

        Returns by default:
            price_hat, delta0_hat, alpha0_hat, hedge_shares_hat

        If return_diag=True, also returns:
            diag dict with stderr, exercise histogram, mean tau, and y0_mean_debug.
        """
        S_paths, V_paths = self._get_inference_paths(seed, use_cached_paths)
        M, Np1, d = S_paths.shape
        N = Np1 - 1
        dt = float(self.dt) if (self.dt is not None) else float(self.T / self.N)

        y_paths, d_paths, a_paths, f_all, f_T = self._eval_nets_on_paths(
            S_paths, V_paths,
            inference_batch_size=inference_batch_size,
            use_target_nets=use_target_nets
        )

        ex_now = (f_all >= y_paths)  # (M,N)
        ex_ext = torch.cat([ex_now, torch.ones(M, 1, dtype=torch.bool)], dim=1)  # (M,N+1)

        pay_ext = torch.cat([f_all, f_T.view(M, 1)], dim=1)  # (M,N+1)
        tau = ex_ext.float().argmax(dim=1).long()            # (M,)

        payoff_tau = pay_ext.gather(1, tau.view(M, 1)).squeeze(1)  # (M,)
        pv = torch.exp(-self.r * tau.float() * dt) * payoff_tau    # (M,)

        price_hat = pv.mean().item()
        stderr = pv.std(unbiased=True).item() / math.sqrt(max(M, 1))

        # delta0/alpha0 under same decision at t=0
        w_cpu = self.w.detach().cpu()
        S0_paths = S_paths[:, 0, :]
        grad0 = grad_payoff_arith(S0_paths, self.K, w_cpu, self.kind)  # (M,d)
        ex0 = ex_now[:, 0].float().view(M, 1)

        delta0_paths = ex0 * grad0 + (1.0 - ex0) * d_paths[:, 0, :]
        alpha0_paths = (1.0 - ex0) * a_paths[:, 0, :]  # alpha irrelevant if exercised

        delta0_hat = delta0_paths.mean(dim=0).numpy()
        alpha0_hat = alpha0_paths.mean(dim=0).numpy()

        # hedge shares at t=0: h = alpha / (S0 * sqrt(v0))
        S0_vec = torch.from_numpy(self.S0).float()
        v0_vec = torch.from_numpy(self.v0).float()
        hedge = self._alpha_to_shares(torch.from_numpy(alpha0_hat).float(), S0_vec, v0_vec).numpy()

        if not return_diag:
            return price_hat, delta0_hat, alpha0_hat, hedge

        hist = torch.bincount(tau, minlength=N + 1).cpu().numpy()
        diag = {
            "price_stderr": float(stderr),
            "exercise_counts": hist,
            "exercise_probs": hist / hist.sum(),
            "mean_exercise_index": float(tau.float().mean().item()),
            "mean_exercise_time": float(tau.float().mean().item() * dt),
            "y0_mean_debug": float(y_paths[:, 0].mean().item()),
        }
        return price_hat, delta0_hat, alpha0_hat, hedge, diag
    
# ---------------------------
# Minimal usage example
# ---------------------------
if __name__ == "__main__":
    d = 5
    corr_matrix = np.eye(d)
    corr_matrix[corr_matrix == 0] = 0.5
    S0_vector = np.array([150, 100, 100, 100.0, 100.0])
    weights   = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

    # Heston params (scalars or (d,))
    v0 = 0.04; theta = 0.04; kappa_v = 1.5; vol_of_vol = 0.5; rho_sv = -0.5

    trainer = RNNAmericanHestonAlphaTrainer(
        d=d, S0=S0_vector, K=100.0, r=0.05, T=0.5, N=126,
        v0=v0, theta=theta, kappa_v=kappa_v, vol_of_vol=vol_of_vol, rho_sv=rho_sv,
        corr=corr_matrix, kind="put", weights=weights,
        M=100000, batch_size=4096, epochs=5, seed=12345,
        hidden_dim=64, num_layers=3, lr=1e-3, grad_clip=1.0,

        alpha_price=1.0,
        z_weight=1.0,
        beta=0.5,
        delta_aux_weight=max(0.001, 0.005*5/d),
        smooth_labels=True,
        smooth_only_at_maturity=False,
        lookahead_window=None,
        shuffle=False,
        drop_last=True,
        resimulate_every=1,

        # ---- key switches ----
        label_mode="snell_one_step",  
        target_tau=0.01,
        target_update_every=1,
        ridge_alpha=1e-6,
    )

    trainer.train()

    price_hat, delta0_hat, alpha0_hat, hedge, diag = trainer.price_at_t0(
        inference_batch_size=4096,
        use_cached_paths=True,
        return_diag=True
    )

    print(f"\n--- Results ---")
    print("t0 American price:", round(price_hat, 6), "stderr:", round(diag["price_stderr"], 6))
    print("t0 deltas:", np.round(delta0_hat, 6))
    print("t0 alphas:", np.round(alpha0_hat, 6))
    print("t0 hedge shares:", np.round(hedge, 6))
    print("mean exercise time:", round(diag["mean_exercise_time"], 6))