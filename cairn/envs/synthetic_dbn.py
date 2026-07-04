"""Ground-truth synthetic dynamic Bayesian network environment.

This is the "synthetic-DBN sanity suite" of algorithm.md (section 6, M1-2):
a nonlinear DBN with a known sparse causal graph A*, known action-target
mask M*, scriptable per-mechanism regime shifts (the sparse-mechanism-shift
setting of section 2.3), and do-interventions on individual nodes.  Ground
truth enables exact scoring of structure recovery (RQ6), localization (RQ2)
and descendant/non-descendant error splits (RQ1).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Regime:
    """A sparse mechanism shift: nodes in ``shifted`` get their input gain
    multiplied by ``gain`` and their noise scale by ``noise_gain``."""
    shifted: tuple[int, ...] = ()
    gain: float = 1.0
    noise_gain: float = 1.0


@dataclass
class Episode:
    z: np.ndarray                     # (T+1, d) states
    a: np.ndarray                     # (T, m) actions
    do_mask: np.ndarray               # (T, d) 1 where node was intervened
    do_values: np.ndarray             # (T, d) intervened values
    regime: int = 0


class SyntheticDBN:
    """z_{t+1}^i = rho_i z_t^i + tanh(w_i . pa_i(z_t) + c_i . (a_t on M*_i))
                   + sigma_i eps.

    Parents (including self) given by A*; action coordinates enter node i's
    mechanism only where M*[k, i] = 1.  A do-intervention on node i replaces
    the mechanism output entirely for that step.
    """

    def __init__(self, d: int = 8, m: int = 3, extra_parents: int = 2,
                 sigma: float = 0.08, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.d, self.m = d, m
        self.sigma = np.full(d, sigma)
        # Ground-truth adjacency: self-edge + `extra_parents` random parents.
        A = np.eye(d, dtype=int)
        for i in range(d):
            others = [j for j in range(d) if j != i]
            for j in rng.choice(others, size=extra_parents, replace=False):
                A[j, i] = 1
        self.A_true = A
        # Ground-truth action targets: each action hits 1-2 nodes.
        M = np.zeros((m, d), dtype=int)
        for k in range(m):
            for i in rng.choice(d, size=rng.integers(1, 3), replace=False):
                M[k, i] = 1
        # Ensure every action targets at least one node.
        self.M_true = M
        # Mechanism parameters.  Edge weights are bounded away from zero:
        # a near-zero-weight "edge" is unidentifiable from finite data in
        # any method and would make structure-recovery scores ill-posed
        # (standard practice in causal-discovery benchmarks).
        self.rho = rng.uniform(0.55, 0.75, size=d)
        self.W = (rng.uniform(0.4, 1.2, size=(d, d))
                  * rng.choice([-1.0, 1.0], size=(d, d))) * A   # parent gains
        self.C = (rng.uniform(0.5, 1.5, size=(m, d))
                  * rng.choice([-1.0, 1.0], size=(m, d))) * M   # action gains
        self.b = rng.normal(0.0, 0.2, size=d)
        self.rng = rng

    # ------------------------------------------------------------------ #

    def step(self, z: np.ndarray, a: np.ndarray, regime: Regime,
             do_mask: np.ndarray | None = None,
             do_values: np.ndarray | None = None,
             rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or self.rng
        gain = np.ones(self.d)
        noise_gain = np.ones(self.d)
        for i in regime.shifted:
            gain[i] = regime.gain
            noise_gain[i] = regime.noise_gain
        drive = np.tanh(gain * (z @ self.W + a @ self.C + self.b))
        z_next = self.rho * z + drive \
            + self.sigma * noise_gain * rng.standard_normal(self.d)
        if do_mask is not None:
            z_next = np.where(do_mask > 0, do_values, z_next)
        return z_next

    def descendants(self, targets: set[int], horizon: int) -> set[int]:
        reach = set(targets)
        frontier = set(targets)
        for _ in range(horizon):
            nxt = {i for i in range(self.d)
                   if any(self.A_true[j, i] for j in frontier)}
            frontier = nxt - reach
            reach |= nxt
            if not frontier:
                break
        return reach

    # ------------------------------------------------------------------ #
    # Data generation                                                      #
    # ------------------------------------------------------------------ #

    def generate_episode(self, T: int, regime: Regime, regime_id: int = 0,
                         action_scale: float = 1.0,
                         p_do: float = 0.0,
                         do_low: float = -1.5, do_high: float = 1.5,
                         do_nodes: list[int] | None = None,
                         rng: np.random.Generator | None = None) -> Episode:
        """Roll one episode with random smooth actions; with probability
        ``p_do`` per step, a random single-node do-intervention is applied
        and *labeled* (target + value), providing the interventional signal
        for L^int (algorithm.md 2.3)."""
        rng = rng or self.rng
        z = np.zeros((T + 1, self.d))
        z[0] = rng.normal(0.0, 0.5, size=self.d)
        a = np.zeros((T, self.m))
        do_mask = np.zeros((T, self.d))
        do_values = np.zeros((T, self.d))
        a_cur = rng.normal(0.0, action_scale, size=self.m)
        for t in range(T):
            a_cur = 0.8 * a_cur + 0.2 * rng.normal(0, action_scale, self.m)
            a[t] = a_cur
            if p_do > 0 and rng.random() < p_do:
                i = (rng.integers(self.d) if do_nodes is None
                     else rng.choice(do_nodes))
                do_mask[t, i] = 1.0
                do_values[t, i] = rng.uniform(do_low, do_high)
            z[t + 1] = self.step(z[t], a[t], regime,
                                 do_mask[t], do_values[t], rng=rng)
        return Episode(z, a, do_mask, do_values, regime_id)

    def generate_dataset(self, regimes: list[Regime], episodes_per_regime: int,
                         T: int, p_do: float = 0.05,
                         do_nodes: list[int] | None = None,
                         seed: int = 1) -> list[Episode]:
        """``do_nodes`` restricts which nodes receive labeled training
        interventions, so evaluation can hold out do-targets entirely
        (compositional interventional generalization)."""
        rng = np.random.default_rng(seed)
        data = []
        for rid, reg in enumerate(regimes):
            for _ in range(episodes_per_regime):
                data.append(self.generate_episode(
                    T, reg, regime_id=rid, p_do=p_do, do_nodes=do_nodes,
                    rng=rng))
        return data


def default_regimes(env: SyntheticDBN, n_regimes: int,
                    seed: int = 7) -> list[Regime]:
    """Regime 0 is nominal; each additional regime shifts 1-2 mechanisms
    (the sparse-mechanism-shift premise)."""
    rng = np.random.default_rng(seed)
    regimes = [Regime()]
    for _ in range(n_regimes - 1):
        k = int(rng.integers(1, 3))
        nodes = tuple(int(x) for x in rng.choice(env.d, size=k, replace=False))
        regimes.append(Regime(shifted=nodes,
                              gain=float(rng.choice([-1.0, 1.6, 0.4])),
                              noise_gain=float(rng.choice([1.0, 2.0]))))
    return regimes
