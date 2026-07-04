"""The CAIRN training objective (algorithm.md section 2.3):

    L = sum_i L_pin  + g1 (|A|_1 + |M|_1)  + g2 L_int  + g3 L_inv  + g4 L_cal

- L_pin: pinball loss per mechanism (teacher-forced 1-step) plus multi-step
  latent-overshooting variants.
- L_int: interventional consistency — on segments with labeled do-targets the
  do-surgery forward pass must match observed outcomes, supervising A and M
  with interventional signal.
- L_inv: sparse-mechanism-shift regularizer — a concave (sqrt) penalty on the
  per-(node, regime) residual-distribution discrepancy, so a factorization
  that spreads a regime shift across many pseudo-mechanisms pays more than
  one that concentrates it in the truly shifted mechanisms.
- L_cal: differentiable (smoothed-indicator) coverage penalty on h-step
  rollout intervals, so calibration is optimized in training.
"""

from __future__ import annotations

import torch

from cairn.quantile import MEDIAN_IDX, TAUS, pinball_loss

# Interval index pairs used for the calibration penalty: (lo, hi, nominal).
_CAL_INTERVALS = [(1, 5, 0.80), (0, 6, 0.90)]


def one_step_pinball(model, z, a, z_next, do_mask=None, do_values=None):
    """Teacher-forced 1-step pinball over all mechanisms, with graph surgery
    applied at labeled intervention steps (the intervened node then matches
    its substituted value exactly, so its loss vanishes and the remaining
    nodes are supervised under the intervention)."""
    q = model.predict_quantiles(z, a)
    q = model.apply_surgery(q, do_mask, do_values)
    return pinball_loss(q, z_next), q


def overshoot_losses(model, seg_z, seg_a, seg_do_mask, seg_do_values,
                     smooth: float = 0.05):
    """Multi-step latent overshooting + interventional consistency + smoothed
    coverage, in one pass over a batch of segments.

    seg_z: (B, h+1, d); seg_a: (B, h, m); seg_do_*: (B, h, d).
    The rollout propagates the predicted *median* (differentiable), applies
    do-surgery at labeled steps, and accumulates (i) pinball at every step,
    split into intervention segments (L_int) vs plain segments (overshoot
    part of L_pin), and (ii) the smoothed coverage penalty (L_cal).
    """
    B, h = seg_a.shape[0], seg_a.shape[1]
    has_do = seg_do_mask.sum(dim=(1, 2)) > 0            # (B,)
    z = seg_z[:, 0]
    pin_plain = z.new_zeros(())
    pin_int = z.new_zeros(())
    cal = z.new_zeros(())
    taus = TAUS.to(z.device)
    for t in range(h):
        q = model.predict_quantiles(z, seg_a[:, t])
        q = model.apply_surgery(q, seg_do_mask[:, t], seg_do_values[:, t])
        target = seg_z[:, t + 1]
        diff = target.unsqueeze(-1) - q
        pin = torch.maximum(taus * diff, (taus - 1.0) * diff).mean(dim=(1, 2))
        pin_plain = pin_plain + (pin * (~has_do).float()).sum() / B
        pin_int = pin_int + (pin * has_do.float()).sum() / B
        for lo, hi, nominal in _CAL_INTERVALS:
            inside = torch.sigmoid((target - q[..., lo]) / smooth) * \
                torch.sigmoid((q[..., hi] - target) / smooth)
            cal = cal + (inside.mean() - nominal) ** 2
        # Propagate the median, but keep observed values at intervened nodes.
        z = q[..., MEDIAN_IDX]
    return pin_plain / h, pin_int / h, cal / (h * len(_CAL_INTERVALS))


def invariance_loss(residuals: torch.Tensor, regime_ids: torch.Tensor,
                    n_regimes: int, eps: float = 1e-4) -> torch.Tensor:
    """Sparse-mechanism-shift penalty (L_inv).

    residuals: (B, d) standardized 1-step residuals; regime_ids: (B,).
    For each (node, regime) we measure the first- and second-moment
    discrepancy of the residual distribution against the pooled one and
    apply a concave sqrt penalty — the smooth relaxation of counting shifted
    mechanisms, which pushes regime shifts to be explained by few mechanisms.
    """
    mu_all = residuals.mean(dim=0)
    sd_all = residuals.std(dim=0) + 1e-6
    total = residuals.new_zeros(())
    for e in range(n_regimes):
        mask = regime_ids == e
        if mask.sum() < 8:
            continue
        r = residuals[mask]
        d2 = (r.mean(0) - mu_all) ** 2 + (r.std(0) + 1e-6 - sd_all) ** 2
        total = total + torch.sqrt(d2 + eps).sum()
    return total / max(n_regimes, 1)
