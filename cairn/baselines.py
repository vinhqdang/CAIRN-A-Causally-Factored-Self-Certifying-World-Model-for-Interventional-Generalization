"""Baselines (algorithm.md section 3, adapted to the state-space suite):

- MonolithicWM: one MLP transition map z_{t+1} = f(z_t, a_t) with a joint
  quantile head — the canonical monolithic world-model comparison class.
- EnsembleWM: PETS-style probabilistic ensemble with trajectory sampling
  (member sampled per trajectory), ensemble variance for detection.
- GlobalEGate: the "monolithic + e-gate on the whole model" ablation —
  a single wealth process fed by all dimensions' PITs, showing that
  localization requires structure.
- CusumDetector: best-tuned two-sided CUSUM on standardized residuals, the
  classical change-detection baseline for RQ2.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from cairn.egate import EGate
from cairn.quantile import (MEDIAN_IDX, N_TAUS, monotone_quantiles,
                            pinball_loss, pit_value, sample_from_quantiles)


class MonolithicWM(nn.Module):
    """z_{t+1} ~ f_theta(z_t, a_t): every latent depends on every other,
    actions condition the whole map — the structural commitment CAIRN
    removes."""

    def __init__(self, d: int, m: int, hidden: int = 160):
        super().__init__()
        self.d, self.m = d, m
        self.net = nn.Sequential(
            nn.Linear(d + m, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, d * N_TAUS),
        )

    def predict_quantiles(self, z, a):
        raw = self.net(torch.cat([z, a], dim=-1))
        return monotone_quantiles(raw.view(*z.shape[:-1], self.d, N_TAUS))

    def median_prediction(self, z, a):
        return self.predict_quantiles(z, a)[..., MEDIAN_IDX]

    @torch.no_grad()
    def rollout(self, z0, actions, n_samples: int = 16, generator=None):
        """actions: (H, B, m) -> samples (n_samples, H, B, d)."""
        H, B = actions.shape[0], z0.shape[0]
        z = z0.unsqueeze(0).expand(n_samples, B, self.d).reshape(-1, self.d)
        outs = []
        for t in range(H):
            a = actions[t].unsqueeze(0).expand(n_samples, B, self.m)
            q = self.predict_quantiles(z, a.reshape(-1, self.m))
            z = sample_from_quantiles(q, generator=generator)
            outs.append(z.view(n_samples, B, self.d))
        return torch.stack(outs, dim=1)


def train_monolithic(model: MonolithicWM, episodes, steps: int = 4000,
                     batch: int = 256, lr: float = 1e-3, seed: int = 0,
                     verbose: bool = False):
    from cairn.train import episodes_to_tensors
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    data = episodes_to_tensors(episodes)
    n = data["z"].shape[0]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(steps):
        idx = torch.tensor(rng.integers(0, n, size=batch))
        q = model.predict_quantiles(data["z"][idx], data["a"][idx])
        loss = pinball_loss(q, data["z_next"][idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if verbose and step % 1000 == 0:
            print(f"  monolithic step={step} pin={float(loss):.4f}",
                  flush=True)
    return model


class EnsembleWM(nn.Module):
    """Probabilistic ensemble of monolithic quantile models with
    trajectory sampling (PETS-style)."""

    def __init__(self, d: int, m: int, n_members: int = 5,
                 hidden: int = 128):
        super().__init__()
        self.d, self.m = d, m
        self.members = nn.ModuleList(
            [MonolithicWM(d, m, hidden) for _ in range(n_members)])

    def predict_quantiles(self, z, a):
        """Ensemble mean of member quantile sets (for point eval)."""
        return torch.stack([f.predict_quantiles(z, a)
                            for f in self.members]).mean(dim=0)

    def median_prediction(self, z, a):
        return self.predict_quantiles(z, a)[..., MEDIAN_IDX]

    @torch.no_grad()
    def member_disagreement(self, z, a) -> torch.Tensor:
        """Std of member medians per dim — the heuristic detection signal."""
        meds = torch.stack([f.median_prediction(z, a)
                            for f in self.members])
        return meds.std(dim=0)

    @torch.no_grad()
    def rollout(self, z0, actions, n_samples: int = 16, generator=None):
        H, B = actions.shape[0], z0.shape[0]
        outs = []
        per = max(1, n_samples // len(self.members))
        for f in self.members:
            outs.append(f.rollout(z0, actions, n_samples=per,
                                  generator=generator))
        return torch.cat(outs, dim=0)


def train_ensemble(model: EnsembleWM, episodes, steps: int = 4000, **kw):
    for k, f in enumerate(model.members):
        train_monolithic(f, episodes, steps=steps,
                         seed=kw.pop("seed", 0) * 100 + k, **dict(kw))
    return model


class GlobalEGate:
    """One e-gate over the whole monolithic model: PITs of all d dimensions
    feed a single wealth process.  Detects *that* something is wrong, cannot
    say *where* — the ablation isolating why localization requires
    structure."""

    def __init__(self, model, delta: float = 0.05):
        self.model = model
        self.gate = EGate(delta)

    @torch.no_grad()
    def observe(self, z, a, z_next, generator=None) -> bool:
        q = self.model.predict_quantiles(z.unsqueeze(0), a.unsqueeze(0))[0]
        fired = False
        for i in range(q.shape[0]):
            u = pit_value(q[i], z_next[i], generator=generator).item()
            fired = self.gate.update(u) or fired
        return fired


class CusumDetector:
    """Two-sided CUSUM per node on standardized residuals (classical
    baseline).  ``threshold`` must be tuned; RQ2 grants it oracle tuning on
    a null stream at matched false-alarm rate — CAIRN needs no tuning."""

    def __init__(self, d: int, drift: float = 0.25, threshold: float = 8.0):
        self.d = d
        self.drift = drift
        self.threshold = threshold
        self.pos = np.zeros(d)
        self.neg = np.zeros(d)

    def update(self, standardized_residual: np.ndarray) -> list[int]:
        """Feed |resid|-based statistic; returns alarmed node indices."""
        s = np.abs(standardized_residual) - 1.0 - self.drift
        self.pos = np.maximum(0.0, self.pos + s)
        alarmed = [i for i in range(self.d) if self.pos[i] > self.threshold]
        return alarmed

    def reset(self, i: int | None = None):
        if i is None:
            self.pos[:] = 0
        else:
            self.pos[i] = 0
