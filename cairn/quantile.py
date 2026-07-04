"""Quantile utilities: pinball loss, monotone heads, PIT, quantile sampling.

Each CAIRN mechanism outputs a set of quantiles of its next-state variable
(algorithm.md section 2.1).  This module provides the shared machinery:

- ``TAUS``: the fixed quantile levels used by every distributional head.
- ``monotone_quantiles``: maps raw network outputs to non-crossing quantiles.
- ``pinball_loss``: the quantile-regression training loss.
- ``pit_value``: the (randomized-tail) probability-integral transform of a
  realized value under a predicted quantile set — the residual the e-gate
  bets against (section 2.2).
- ``sample_from_quantiles``: inverse-CDF sampling used for quantile-propagated
  rollouts (section 2.1) with optional evidence-driven spread inflation
  (section 2.2, use #2).
"""

from __future__ import annotations

import torch

# Fixed quantile levels shared by all mechanisms.  Symmetric around the
# median so that interval coverage at nominal 80%/90% can be read off
# directly (0.05-0.95 -> 90%, 0.10-0.90 -> 80%).
TAUS = torch.tensor([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
N_TAUS = len(TAUS)
MEDIAN_IDX = 3


def monotone_quantiles(raw: torch.Tensor) -> torch.Tensor:
    """Map raw head outputs (..., N_TAUS) to non-crossing quantiles.

    The first output is the lowest quantile; subsequent quantiles are obtained
    by accumulating softplus-positive increments, which guarantees
    monotonicity in tau without constrained optimization.
    """
    base = raw[..., :1]
    deltas = torch.nn.functional.softplus(raw[..., 1:])
    return torch.cat([base, base + torch.cumsum(deltas, dim=-1)], dim=-1)


def pinball_loss(quantiles: torch.Tensor, target: torch.Tensor,
                 taus: torch.Tensor | None = None) -> torch.Tensor:
    """Mean pinball (quantile) loss.

    quantiles: (..., N_TAUS); target: (...,) broadcastable to quantiles[..., 0].
    """
    if taus is None:
        taus = TAUS.to(quantiles.device)
    diff = target.unsqueeze(-1) - quantiles
    return torch.maximum(taus * diff, (taus - 1.0) * diff).mean()


def pit_value(quantiles: torch.Tensor, target: torch.Tensor,
              generator: torch.Generator | None = None) -> torch.Tensor:
    """Probability-integral-transform of ``target`` under the predicted CDF.

    The CDF is piecewise-linear through the (quantile, tau) knots.  Outside
    the outermost quantiles the PIT is drawn uniformly from the corresponding
    tail mass ([0, tau_min] or [tau_max, 1]): under the null that the stated
    quantiles are valid, the randomized PIT is exactly uniform in the tails
    and approximately uniform in the (linearly interpolated) interior, which
    is the null the e-gate tests (algorithm.md section 5.5 states this
    honestly).

    quantiles: (..., N_TAUS); target: (...,).  Returns u in (0, 1), same
    shape as target.
    """
    taus = TAUS.to(quantiles.device)
    q = quantiles
    y = target.unsqueeze(-1)

    # Interior: piecewise-linear interpolation of the CDF.
    # For each knot interval [q_j, q_{j+1}], fraction of the way through.
    q_lo, q_hi = q[..., :-1], q[..., 1:]
    t_lo, t_hi = taus[:-1], taus[1:]
    width = (q_hi - q_lo).clamp_min(1e-12)
    frac = ((y - q_lo) / width).clamp(0.0, 1.0)
    seg = t_lo + frac * (t_hi - t_lo)
    inside = (y >= q_lo) & (y < q_hi)
    u = torch.where(inside.any(dim=-1),
                    (seg * inside).sum(dim=-1)
                    / inside.sum(dim=-1).clamp_min(1),
                    torch.zeros_like(target))

    below = target < q[..., 0]
    above = target >= q[..., -1]
    if generator is not None:
        r = torch.rand(target.shape, generator=generator,
                       device=target.device)
    else:
        r = torch.rand_like(target)
    u = torch.where(below, r * taus[0], u)
    u = torch.where(above, taus[-1] + r * (1.0 - taus[-1]), u)
    return u.clamp(1e-6, 1.0 - 1e-6)


class PitCalibrator:
    """Conformal recalibration of PIT values (algorithm.md section 6: the
    conformalized-quantile variant restoring exact finite-sample PIT
    validity).

    Fit on held-out calibration PITs; ``transform`` maps a new PIT through
    the randomized empirical CDF, which is exactly Unif(0,1) under
    exchangeability with the calibration set — so the e-gate's null holds
    exactly even when the quantile head is slightly miscalibrated."""

    def __init__(self):
        self.sorted_u: torch.Tensor | None = None

    def fit(self, u_cal: torch.Tensor) -> "PitCalibrator":
        self.sorted_u = torch.sort(u_cal.flatten()).values
        return self

    def transform(self, u: float,
                  generator: torch.Generator | None = None) -> float:
        if self.sorted_u is None:
            return u
        n = len(self.sorted_u)
        rank = int(torch.searchsorted(self.sorted_u,
                                      torch.tensor(u)).item())
        r = torch.rand((), generator=generator).item()
        return min(max((rank + r) / (n + 1), 1e-6), 1 - 1e-6)


def sample_from_quantiles(quantiles: torch.Tensor,
                          u: torch.Tensor | None = None,
                          inflation: torch.Tensor | float = 1.0,
                          generator: torch.Generator | None = None
                          ) -> torch.Tensor:
    """Inverse-CDF sample from a predicted quantile set.

    quantiles: (..., N_TAUS).  ``u`` optional uniform draws of shape
    quantiles.shape[:-1].  ``inflation`` >= 1 scales the spread around the
    median (evidence-driven uncertainty inflation, algorithm.md 2.2 use #2).
    """
    taus = TAUS.to(quantiles.device)
    med = quantiles[..., MEDIAN_IDX:MEDIAN_IDX + 1]
    if not (isinstance(inflation, float) and inflation == 1.0):
        infl = inflation if isinstance(inflation, torch.Tensor) else \
            torch.as_tensor(inflation, device=quantiles.device)
        quantiles = med + (quantiles - med) * infl.unsqueeze(-1)
    if u is None:
        if generator is not None:
            u = torch.rand(quantiles.shape[:-1], generator=generator,
                           device=quantiles.device)
        else:
            u = torch.rand(quantiles.shape[:-1], device=quantiles.device)
    u = u.clamp(taus[0].item(), taus[-1].item())  # clip to supported range
    # Piecewise-linear inverse CDF through the knots.
    idx = torch.searchsorted(taus, u.contiguous(), right=True).clamp(1, N_TAUS - 1)
    t_lo, t_hi = taus[idx - 1], taus[idx]
    q_lo = torch.gather(quantiles, -1, (idx - 1).unsqueeze(-1)).squeeze(-1)
    q_hi = torch.gather(quantiles, -1, idx.unsqueeze(-1)).squeeze(-1)
    frac = (u - t_lo) / (t_hi - t_lo)
    return q_lo + frac * (q_hi - q_lo)
