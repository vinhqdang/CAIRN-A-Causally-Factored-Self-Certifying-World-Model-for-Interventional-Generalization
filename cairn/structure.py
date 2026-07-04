"""Learned causal structure: adjacency A and action-target mask M.

Both are parameterized with Gumbel-sigmoid relaxations and straight-through
estimation (algorithm.md section 2.1).  Edges run t -> t+1 (a dynamic
Bayesian network), so no acyclicity constraint is needed.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def gumbel_sigmoid(logits: torch.Tensor, temperature: float,
                   hard: bool = True) -> torch.Tensor:
    """Binary Gumbel-sigmoid sample with straight-through gradients."""
    u = torch.rand_like(logits).clamp(1e-6, 1 - 1e-6)
    logistic_noise = torch.log(u) - torch.log1p(-u)
    soft = torch.sigmoid((logits + logistic_noise) / temperature)
    if not hard:
        return soft
    hard_sample = (soft > 0.5).float()
    return hard_sample + soft - soft.detach()


class GumbelStructure(nn.Module):
    """Holds logits for A (d x d latent parents) and M (m x d action targets).

    ``A[j, i] = 1`` means latent j at time t is a parent of latent i at t+1.
    ``M[k, i] = 1`` means action coordinate k intervenes on latent i.
    Self-edges A[i, i] are initialized strongly positive (a variable is
    almost always a parent of its own next value in physical systems), but
    remain learnable.
    """

    def __init__(self, d: int, m: int, init_logit: float = 0.0,
                 self_edge_logit: float = 3.0):
        super().__init__()
        self.d, self.m = d, m
        a_init = torch.full((d, d), init_logit)
        a_init += torch.eye(d) * (self_edge_logit - init_logit)
        self.logits_A = nn.Parameter(a_init)
        self.logits_M = nn.Parameter(torch.full((m, d), init_logit))
        self.temperature = 1.0

    def sample(self, hard: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """Training-time stochastic masks (straight-through)."""
        A = gumbel_sigmoid(self.logits_A, self.temperature, hard=hard)
        M = gumbel_sigmoid(self.logits_M, self.temperature, hard=hard)
        return A, M

    @torch.no_grad()
    def hard_masks(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Deterministic deployment-time masks."""
        return ((self.logits_A > 0).float(), (self.logits_M > 0).float())

    def sparsity_loss(self) -> torch.Tensor:
        """L1 penalty on edge probabilities (algorithm.md section 2.3)."""
        probs_A = torch.sigmoid(self.logits_A)
        probs_M = torch.sigmoid(self.logits_M)
        off_diag = probs_A * (1.0 - torch.eye(self.d, device=probs_A.device))
        return off_diag.sum() / self.d + probs_M.sum() / max(self.m, 1)

    @torch.no_grad()
    def descendants(self, targets: torch.Tensor, horizon: int) -> torch.Tensor:
        """Soft reachability from ``targets`` (d,) within ``horizon`` steps.

        Returns a (d,) tensor in [0, 1]; 1 means reachable from the target
        set through the current hard adjacency.  Used for evaluation and for
        the interventional-consistency split.
        """
        A, _ = self.hard_masks()
        reach = targets.clone().float()
        frontier = targets.clone().float()
        for _ in range(horizon):
            frontier = ((frontier @ A) > 0).float()
            reach = torch.maximum(reach, frontier)
        return reach
