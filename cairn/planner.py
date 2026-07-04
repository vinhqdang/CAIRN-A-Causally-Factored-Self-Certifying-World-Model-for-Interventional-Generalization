"""Sampling-based planning with CAIRN (algorithm.md section 2.4).

CEM over action sequences, scored by risk-sensitive (CVaR) returns over
quantile-propagated rollout samples.  Because rollout samples are drawn
through the evidence-weighted mechanism mixtures *with uncertainty
inflation*, candidate action sequences whose effects flow through
low-wealth (healthy) mechanisms score better under CVaR — the planner
routes around the broken part of the world model without any extra code.
"""

from __future__ import annotations

import torch


class CEMPlanner:
    def __init__(self, model, reward_fn, horizon: int = 8,
                 population: int = 64, elites: int = 8, iters: int = 4,
                 n_rollout_samples: int = 12, cvar_alpha: float = 0.3,
                 action_low: float = -2.0, action_high: float = 2.0,
                 inflate: bool = True):
        self.model = model
        self.reward_fn = reward_fn          # (z: (..., d)) -> reward (...,)
        self.horizon = horizon
        self.population = population
        self.elites = elites
        self.iters = iters
        self.n_rollout_samples = n_rollout_samples
        self.cvar_alpha = cvar_alpha
        self.action_low, self.action_high = action_low, action_high
        self.inflate = inflate

    @torch.no_grad()
    def plan(self, z: torch.Tensor,
             generator: torch.Generator | None = None) -> torch.Tensor:
        """z: (d,) current state.  Returns the first action (m,)."""
        H, m = self.horizon, self.model.m
        mean = torch.zeros(H, m)
        std = torch.ones(H, m)
        for _ in range(self.iters):
            noise = torch.randn(self.population, H, m, generator=generator)
            cand = (mean + std * noise).clamp(self.action_low,
                                              self.action_high)
            scores = self._score(z, cand, generator=generator)
            elite_idx = scores.topk(self.elites).indices
            elite = cand[elite_idx]
            mean = elite.mean(dim=0)
            std = elite.std(dim=0) + 1e-3
        return mean[0]

    @torch.no_grad()
    def _score(self, z: torch.Tensor, cand: torch.Tensor,
               generator=None) -> torch.Tensor:
        """CVaR_alpha of quantile-propagated returns per candidate."""
        P = cand.shape[0]
        actions = cand.permute(1, 0, 2)                     # (H, P, m)
        z0 = z.unsqueeze(0).expand(P, self.model.d)
        samples = self.model.rollout(
            z0, actions, n_samples=self.n_rollout_samples,
            inflate=self.inflate, generator=generator)      # (S, H, P, d)
        rewards = self.reward_fn(samples).sum(dim=1)        # (S, P)
        k = max(1, int(self.cvar_alpha * rewards.shape[0]))
        worst, _ = rewards.topk(k, dim=0, largest=False)
        return worst.mean(dim=0)                            # (P,)
