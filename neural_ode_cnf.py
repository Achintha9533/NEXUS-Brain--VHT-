from sklearn.impute import SimpleImputer
from torchdiffeq import odeint  
import numpy as np
import torch
import torch.nn as nn

class CNFODEFunc(nn.Module):
    def __init__(self, state_dim, cond_dim=0, hidden_dim=128):
        super().__init__()
        self.state_dim = state_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim + cond_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim)
        )
        self._cond = None

    def set_condition(self, c):
        self._cond = c

    def forward(self, t, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if self._cond is None or self.cond_dim == 0:
            inp = x
        else:
            c = self._cond
            if c.dim() == 1:
                c = c.unsqueeze(0)
            if c.shape[0] != x.shape[0]:
                c = c.expand(x.shape[0], -1)
            inp = torch.cat([x, c], dim=-1)
        return self.net(inp)


class CNFWrapper(nn.Module):
    def __init__(self, state_dim, cond_dim=0, hidden_dim=128, device=None):
        super().__init__()
        self.state_dim = state_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.odefunc = CNFODEFunc(state_dim, cond_dim, hidden_dim).to(self.device)
        self.is_fitted = False
        self.state_imputer = SimpleImputer(strategy="median")
        self.cond_imputer = None
        self.d_log_mean = None
        self.d_log_std = None

    def fit_transition_pairs(self, X_t, X_t1, C_t=None, epochs=100, batch_size=32, lr=1e-3):
        X_t = np.asarray(X_t, dtype=float)
        X_t1 = np.asarray(X_t1, dtype=float)
        n = min(len(X_t), len(X_t1))
        if n < 2:
            self.is_fitted = False
            return self

        X_t = X_t[:n]
        X_t1 = X_t1[:n]

        has_cond = C_t is not None and len(C_t) > 0
        if has_cond:
            C_t = np.asarray(C_t, dtype=float)[:n]
            if C_t.ndim == 1:
                C_t = C_t.reshape(-1, 1)
            if C_t.shape[1] == 0:
                has_cond = False
        if not has_cond:
            C_t = np.zeros((n, 0), dtype=float)

        X_t = self.state_imputer.fit_transform(X_t)
        X_t1 = self.state_imputer.transform(X_t1)

        if has_cond:
            self.cond_imputer = SimpleImputer(strategy="median")
            C_t = self.cond_imputer.fit_transform(C_t)
        else:
            self.cond_imputer = None

        log_X_t = np.log1p(np.clip(X_t, 0, None))
        log_X_t1 = np.log1p(np.clip(X_t1, 0, None))
        d_log = log_X_t1 - log_X_t

        self.d_log_mean = np.mean(d_log, axis=0)
        self.d_log_std = np.std(d_log, axis=0)
        self.d_log_std[self.d_log_std == 0] = 1.0

        # Pre-convert parameters into tensors for speed and compatibility inside the loop
        d_log_mean_t = torch.tensor(self.d_log_mean, dtype=torch.float32, device=self.device)
        d_log_std_t = torch.tensor(self.d_log_std, dtype=torch.float32, device=self.device)

        d_log_scaled = (d_log - self.d_log_mean) / self.d_log_std

        log_X_t = torch.tensor(log_X_t, dtype=torch.float32, device=self.device)
        d_log_scaled = torch.tensor(d_log_scaled, dtype=torch.float32, device=self.device)
        C_t = torch.tensor(C_t, dtype=torch.float32, device=self.device) if C_t.shape[1] > 0 else None

        opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.train()
        t_span = torch.tensor([0.0, 1.0], device=self.device)

        for _ in range(epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                x0 = log_X_t[idx]
                d_log_true_scaled = d_log_scaled[idx]
                c0 = C_t[idx] if C_t is not None else None

                self.odefunc.set_condition(c0)
                # Call odeint safely; output tensor is shape [2, batch, state_dim] -> isolate final state [-1]
                x_pred = odeint(self.odefunc, x0, t_span, method="euler", options={"step_size": 0.2})[-1]
                
                d_log_pred_raw = x_pred - x0
                # Fixed broadcasting mismatched tensors here
                d_log_pred_scaled = (d_log_pred_raw - d_log_mean_t) / d_log_std_t

                loss = torch.mean((d_log_pred_scaled - d_log_true_scaled) ** 2)
                opt.zero_grad()
                loss.backward()
                opt.step()

        self.is_fitted = True
        return self

    def sample_counterfactual_path(self, x0, c0=None, steps=10, n_samples=20, treatment_plan=None, step_scale=1.0, noise_scale=0.01):
        if not self.is_fitted:
            raise RuntimeError("CNFWrapper has not been fitted yet.")

        x0 = np.asarray(x0, dtype=float).reshape(1, -1)
        x0 = self.state_imputer.transform(x0)

        if c0 is None:
            c0 = np.zeros((1, self.cond_dim), dtype=float)
        else:
            c0 = np.asarray(c0, dtype=float).reshape(1, -1)
            if c0.shape[1] > 0 and self.cond_imputer is not None:
                c0 = self.cond_imputer.transform(c0)

        paths = []
        mean_log = torch.tensor(self.d_log_mean, device=self.device, dtype=torch.float32)
        std_log = torch.tensor(self.d_log_std, device=self.device, dtype=torch.float32)
        t_span = torch.tensor([0.0, 1.0], device=self.device)

        with torch.no_grad():
            for _ in range(n_samples):
                x_physical = torch.tensor(x0.copy(), dtype=torch.float32, device=self.device)
                x_log = torch.log1p(torch.clamp(x_physical, min=0.0))
                path = [x_physical.squeeze(0).cpu().numpy().copy()]

                for s in range(steps):
                    c_step = c0.copy()
                    if treatment_plan is not None:
                        c_step = treatment_plan(s, c_step)

                    c_step_t = torch.tensor(c_step, dtype=torch.float32, device=self.device) if c_step.shape[1] > 0 else None
                    self.odefunc.set_condition(c_step_t)

                    x_next_log = odeint(self.odefunc, x_log, t_span, method="euler", options={"step_size": 0.2})[-1]
                    d_log_normalized = x_next_log - x_log
                    d_log_physical = (d_log_normalized * std_log) + mean_log
                    x_log = x_log + step_scale * d_log_physical + noise_scale * torch.randn_like(x_log)

                    x_physical_step = torch.expm1(x_log)
                    x_physical_step = torch.clamp(x_physical_step, min=0.0)
                    path.append(x_physical_step.squeeze(0).cpu().numpy().copy())

                paths.append(np.asarray(path))
        return paths


def build_treatment_plan(context_cols, **kwargs):
    cidx = {c: i for i, c in enumerate(context_cols)}

    def plan(step, c_step):
        c = np.asarray(c_step, dtype=float).reshape(1, -1)

        def set_if_present(name, value):
            if name in cidx:
                c[0, cidx[name]] = float(value)

        if step == 0:
            for key, value in kwargs.items():
                set_if_present(key, value)
        return c

    return plan