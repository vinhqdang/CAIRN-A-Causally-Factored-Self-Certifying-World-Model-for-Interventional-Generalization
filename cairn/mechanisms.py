"""Factored mechanisms: one small quantile-headed MLP per latent variable,
plus the per-node mechanism library used for evidence-gated adaptation
(algorithm.md sections 2.1 and 2.2, use #1).
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn

from cairn.quantile import N_TAUS, monotone_quantiles


class MechanismMLP(nn.Module):
    """Mechanism f_i: (z_t masked by A_{.i}, a_t masked by M_{.i}) -> quantiles.

    The full (d + m)-dimensional input is multiplied elementwise by the
    node's parent/target mask columns before the MLP, so structure gradients
    flow through the Gumbel-sigmoid masks.
    """

    def __init__(self, d: int, m: int, hidden: int = 48):
        super().__init__()
        self.d, self.m = d, m
        self.net = nn.Sequential(
            nn.Linear(d + m, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, N_TAUS),
        )

    def forward(self, z: torch.Tensor, a: torch.Tensor,
                a_col: torch.Tensor, m_col: torch.Tensor) -> torch.Tensor:
        """z: (B, d); a: (B, m); a_col: (d,) parent mask; m_col: (m,).

        Returns non-crossing quantiles (B, N_TAUS).
        """
        x = torch.cat([z * a_col, a * m_col], dim=-1)
        return monotone_quantiles(self.net(x))


class NodeLibrary:
    """Per-node library {f_i^(1), ..., f_i^(K)} with evidence weights.

    Rollouts combine members' quantile functions with weights
    pi^(k) proportional to exp(-beta * log W^(k))  (algorithm.md 2.2, use #1):
    members whose wealth (evidence of invalidity) grows are smoothly
    down-weighted.  Wealth values live in the paired EGate; this class only
    stores the mechanisms and computes the mixture.
    """

    def __init__(self, base: MechanismMLP, beta: float = 1.0,
                 max_members: int = 4):
        self.members: list[MechanismMLP] = [base]
        self.beta = beta
        self.max_members = max_members

    def spawn(self) -> tuple[MechanismMLP, int | None]:
        """Copy the most recently added member as a fresh adaptation.

        Returns (new_member, dropped_index): when the library is full the
        oldest non-base adaptation is dropped to bound memory, and the
        caller must drop the matching e-gate to stay in sync."""
        new = copy.deepcopy(self.members[-1])
        dropped = None
        if len(self.members) >= self.max_members:
            dropped = 1 if len(self.members) > 1 else 0
            self.members.pop(dropped)
        self.members.append(new)
        return new, dropped

    def weights(self, log_wealths: list[float]) -> torch.Tensor:
        """pi^(k) ∝ exp(-beta * log W^(k)), normalized."""
        lw = torch.tensor([min(w, 50.0) for w in log_wealths])
        logits = -self.beta * lw
        return torch.softmax(logits, dim=0)

    def mixture_quantiles(self, z: torch.Tensor, a: torch.Tensor,
                          a_col: torch.Tensor, m_col: torch.Tensor,
                          log_wealths: list[float]) -> torch.Tensor:
        """Evidence-weighted average of member quantile functions.

        Averaging quantile functions (vincentization) preserves monotonicity
        and yields a valid quantile set for the mixture-of-mechanisms.
        """
        assert len(log_wealths) == len(self.members), \
            "library members and e-gates out of sync"
        pis = self.weights(log_wealths).to(z.device)
        qs = torch.stack(
            [f(z, a, a_col, m_col) for f in self.members], dim=0)
        return (pis.view(-1, 1, 1) * qs).sum(dim=0)

    def __len__(self) -> int:
        return len(self.members)
