"""The e-gate: per-mechanism anytime-valid betting martingale (algorithm.md 2.2).

Each mechanism i maintains a wealth process over the PIT values of its own
realized next-state variable:

    W_t = prod_{s<=t} (1 + lambda_s * g(u_s)),   E[g(U)] = 0 for U ~ Unif(0,1)

with bets lambda adapted by Online Newton Step (ONS).  By Ville's inequality,
P(exists t: W_t >= 1/delta) <= delta while the mechanism is valid — a
time-uniform, per-mechanism certificate with no multiple-testing correction
across continuous monitoring.

We run two elementary e-processes per node — a location bet
g1(u) = 2u - 1 and a dispersion bet g2(u) = 6(u - 1/2)^2 - 1/2 (both bounded
in [-1, 1] with mean 0 under uniformity) — and combine them by averaging
(an average of e-processes is an e-process), so both shifted and
over/under-dispersed predictive distributions are detected.
"""

from __future__ import annotations

import math

import torch

# ONS constant for exp-concave losses (Cutkosky & Orabona; Waudby-Smith &
# Ramdas "estimating means by betting").
_ONS_C = 2.0 / (2.0 - math.log(3.0))
_LAMBDA_MAX = 0.5  # keeps 1 + lambda * g strictly positive for |g| <= 1
_LOG_WEALTH_CAP = 80.0  # numerical cap; alarms fire far below this


def _g_location(u: float) -> float:
    return 2.0 * u - 1.0

def _g_dispersion(u: float) -> float:
    return 6.0 * (u - 0.5) ** 2 - 0.5


class EProcess:
    """A single betting e-process with ONS-adapted bets and a tolerance
    (composite) null.

    With tolerance ``eps`` the per-step payoff is g(u) - eps * sign(lam),
    so the wealth is a supermartingale under ANY PIT distribution with
    |E g(U)| <= eps — not just exact uniformity.  Ville's inequality then
    holds for the composite null "mechanism approximately valid", which is
    the operationally meaningful null when the quantile heads carry
    irreducible approximation error (algorithm.md 5.5): a learned model
    with small residual bias never accumulates wealth, while a genuine
    mechanism shift (|E g| >> eps) is still detected within O(1/(Eg-eps))
    steps.  eps=0 recovers the exact-uniformity null."""

    def __init__(self, g, eps: float = 0.0):
        self.g = g
        self.eps = eps
        self.log_wealth = 0.0
        self.lam = 0.0
        self._grad_sq_sum = 1.0  # ONS second-order accumulator (A_0 = 1)

    def update(self, u: float) -> None:
        sign = 1.0 if self.lam > 0 else (-1.0 if self.lam < 0 else 0.0)
        g = self.g(u) - self.eps * sign
        g = max(-1.0 - self.eps, min(1.0 + self.eps, g))
        self.log_wealth = min(self.log_wealth + math.log1p(self.lam * g),
                              _LOG_WEALTH_CAP)
        # ONS ascent step on log(1 + lam * g).
        grad = g / (1.0 + self.lam * g)
        self._grad_sq_sum += grad * grad
        self.lam = max(-_LAMBDA_MAX,
                       min(_LAMBDA_MAX,
                           self.lam + _ONS_C * grad / self._grad_sq_sum))

    def reset(self) -> None:
        self.log_wealth = 0.0
        self.lam = 0.0
        self._grad_sq_sum = 1.0


class EGate:
    """Composite e-gate for one mechanism: mean of two e-processes.

    ``log_wealth`` is log of W = (W_loc + W_disp) / 2, itself an e-process,
    so the Ville threshold 1/delta applies unchanged.
    """

    def __init__(self, delta: float = 0.05, eps: float = 0.0):
        self.delta = delta
        self.eps = eps
        self.procs = [EProcess(_g_location, eps),
                      EProcess(_g_dispersion, eps)]
        self.steps = 0

    @property
    def log_wealth(self) -> float:
        lws = [p.log_wealth for p in self.procs]
        mx = max(lws)
        return mx + math.log(sum(math.exp(lw - mx) for lw in lws) / len(lws))

    @property
    def wealth(self) -> float:
        return math.exp(min(self.log_wealth, _LOG_WEALTH_CAP))

    def update(self, u: float) -> bool:
        """Feed one PIT value; returns True if the anytime-valid alarm fires."""
        self.steps += 1
        for p in self.procs:
            p.update(u)
        return self.alarmed

    @property
    def alarmed(self) -> bool:
        return self.log_wealth >= math.log(1.0 / self.delta)

    def reset(self) -> None:
        for p in self.procs:
            p.reset()
        self.steps = 0


class GateBank:
    """One e-gate per (node, library member).  Convenience container that
    also exposes the per-node mixture weights' inputs (log-wealth of the
    member currently monitored).
    """

    def __init__(self, d: int, delta: float = 0.05, eps: float = 0.0):
        self.d = d
        self.delta = delta
        self.eps = eps
        self.gates: list[list[EGate]] = [[EGate(delta, eps)]
                                         for _ in range(d)]

    def active(self, i: int) -> EGate:
        """Gate of the most recently spawned member at node i."""
        return self.gates[i][-1]

    def add_member(self, i: int, dropped: int | None = None) -> EGate:
        """Append a fresh gate for a newly spawned member; drop the gate of
        an evicted member first so gates stay aligned with the library."""
        if dropped is not None:
            self.gates[i].pop(dropped)
        gate = EGate(self.delta, self.eps)
        self.gates[i].append(gate)
        return gate

    def log_wealths(self, i: int) -> list[float]:
        return [g.log_wealth for g in self.gates[i]]

    def node_log_wealth(self, i: int) -> float:
        """Evidence against the node's *current best* member: the minimum
        member wealth (if any member is still valid, the node is usable)."""
        return min(self.log_wealths(i))
