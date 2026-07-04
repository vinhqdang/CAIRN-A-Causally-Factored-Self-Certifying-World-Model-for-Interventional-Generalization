import math

import numpy as np

from cairn.egate import EGate, EProcess, _g_dispersion, _g_location


def test_betting_functions_zero_mean_under_uniform():
    u = np.random.default_rng(0).uniform(size=200000)
    assert abs(np.mean([_g_location(x) for x in u])) < 0.01
    assert abs(np.mean([_g_dispersion(x) for x in u])) < 0.01


def test_ville_false_alarm_control_under_null():
    """While the mechanism is valid (uniform PITs), P(W ever >= 1/delta)
    <= delta.  Empirically over many independent gates the alarm fraction
    must not exceed delta (it is typically far below)."""
    rng = np.random.default_rng(1)
    delta, n_gates, horizon = 0.05, 300, 3000
    alarmed = 0
    for _ in range(n_gates):
        gate = EGate(delta)
        for u in rng.uniform(size=horizon):
            if gate.update(float(u)):
                alarmed += 1
                break
    assert alarmed / n_gates <= delta


def test_detection_power_under_shifted_pits():
    """A mechanism whose predictions are biased yields non-uniform PITs;
    wealth must cross 1/delta quickly."""
    rng = np.random.default_rng(2)
    delays = []
    for _ in range(20):
        gate = EGate(0.05)
        # Shifted mechanism: realized values fall in the upper quantiles.
        for t in range(2000):
            u = float(rng.beta(3.0, 1.2))
            if gate.update(u):
                delays.append(t + 1)
                break
        else:
            delays.append(2000)
    assert np.median(delays) < 200


def test_detection_power_under_overdispersion():
    """Variance blow-up pushes PITs to both tails; the dispersion bet must
    catch it even though the location bet cannot."""
    rng = np.random.default_rng(3)
    gate = EGate(0.05)
    fired_at = None
    for t in range(3000):
        u = float(rng.beta(0.35, 0.35))          # U-shaped PITs
        if gate.update(u):
            fired_at = t + 1
            break
    assert fired_at is not None and fired_at < 500


def test_tolerance_null_absorbs_small_bias_but_detects_shifts():
    """With tolerance eps, PIT distributions with |E g| <= eps (small
    approximation bias) must not accumulate wealth, while strong shifts
    must still alarm quickly."""
    rng = np.random.default_rng(5)
    # Small location bias: E[2U-1] ~ 0.05 < eps = 0.15.
    gate = EGate(0.05, eps=0.15)
    for _ in range(20000):
        gate.update(float(rng.beta(1.105, 1.0)))
    assert not gate.alarmed and gate.log_wealth < 1.0
    # Strong shift: E[2U-1] ~ 0.43 >> eps.
    gate = EGate(0.05, eps=0.15)
    fired_at = None
    for t in range(2000):
        if gate.update(float(rng.beta(3.0, 1.2))):
            fired_at = t + 1
            break
    assert fired_at is not None and fired_at < 400


def test_ons_lambda_bounded():
    proc = EProcess(_g_location)
    rng = np.random.default_rng(4)
    for u in rng.uniform(size=5000):
        proc.update(float(u))
        assert -0.5 <= proc.lam <= 0.5
    assert math.isfinite(proc.log_wealth)
