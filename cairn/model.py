"""The CAIRN world model: factored transition kernel over a learned causal
graph, interventions as graph surgery, quantile-propagated rollouts with
evidence-driven uncertainty inflation (algorithm.md section 2).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from cairn.egate import GateBank
from cairn.mechanisms import MechanismMLP, NodeLibrary
from cairn.quantile import (MEDIAN_IDX, N_TAUS, PitCalibrator, pit_value,
                            sample_from_quantiles)
from cairn.structure import GumbelStructure


class CairnWorldModel(nn.Module):
    """CAIRN in state space (identity encoder; the latent partition is given,
    as in the finance instantiation of algorithm.md 2.1 — a learned encoder
    is pluggable upstream without changing anything here).

    Parameters
    ----------
    d : number of latent variables (one mechanism each)
    m : action dimension
    hidden : mechanism MLP width
    delta : anytime-valid alarm level of each e-gate
    beta : evidence-weight sharpness for the mechanism mixtures
    inflation_scale : strength of rollout uncertainty inflation (2.2 use #2)
    """

    def __init__(self, d: int, m: int, hidden: int = 48, delta: float = 0.05,
                 beta: float = 1.0, inflation_scale: float = 1.0):
        super().__init__()
        self.d, self.m = d, m
        self.delta = delta
        self.inflation_scale = inflation_scale
        self.structure = GumbelStructure(d, m)
        self.base_mechanisms = nn.ModuleList(
            [MechanismMLP(d, m, hidden) for _ in range(d)])
        self.libraries = [NodeLibrary(f, beta=beta)
                          for f in self.base_mechanisms]
        self.gates = GateBank(d, delta)
        self.calibrators: list[PitCalibrator | None] = [None] * d
        self._last_raw_pits: list[float] = [0.5] * d

    # ------------------------------------------------------------------ #
    # Prediction                                                          #
    # ------------------------------------------------------------------ #

    def predict_quantiles(self, z: torch.Tensor, a: torch.Tensor,
                          hard: bool | None = None,
                          use_mixture: bool = False) -> torch.Tensor:
        """One-step predictive quantiles for every node: (B, d, N_TAUS).

        During training ``hard=None`` samples straight-through Gumbel masks;
        at deployment pass ``hard=True`` for deterministic structure.
        ``use_mixture=True`` engages the evidence-weighted mechanism
        libraries (deployment path).
        """
        if hard is None:
            A, M = self.structure.sample(hard=True)
        else:
            A, M = self.structure.hard_masks()
        qs = []
        for i in range(self.d):
            if use_mixture and len(self.libraries[i]) > 1:
                q = self.libraries[i].mixture_quantiles(
                    z, a, A[:, i], M[:, i], self.gates.log_wealths(i))
            else:
                q = self.libraries[i].members[-1](z, a, A[:, i], M[:, i])
            qs.append(q)
        return torch.stack(qs, dim=1)

    def median_prediction(self, z: torch.Tensor, a: torch.Tensor,
                          **kw) -> torch.Tensor:
        return self.predict_quantiles(z, a, **kw)[..., MEDIAN_IDX]

    # ------------------------------------------------------------------ #
    # Interventions: graph surgery                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def apply_surgery(quantiles: torch.Tensor,
                      do_mask: torch.Tensor | None,
                      do_values: torch.Tensor | None) -> torch.Tensor:
        """do(z^i <- v): replace mechanism i's output by the intervened value.

        Severed parents + substituted value = Pearl graph surgery implemented
        architecturally (algorithm.md 2.1).  ``do_mask``: (d,) or (B, d) in
        {0,1}; ``do_values``: broadcastable to (B, d).  The intervened node's
        entire quantile set collapses to the substituted value (a point mass).
        """
        if do_mask is None:
            return quantiles
        dm = do_mask.unsqueeze(-1)
        dv = do_values.unsqueeze(-1).expand_as(quantiles)
        return quantiles * (1 - dm) + dv * dm

    # ------------------------------------------------------------------ #
    # Rollouts                                                            #
    # ------------------------------------------------------------------ #

    def node_inflation(self) -> torch.Tensor:
        """Per-node quantile-spread inflation factor, increasing in log W
        (algorithm.md 2.2 use #2).  Healthy nodes (W <= 1) get factor 1."""
        log_thresh = math.log(1.0 / self.delta)
        factors = []
        for i in range(self.d):
            lw = max(0.0, self.gates.node_log_wealth(i))
            factors.append(1.0 + self.inflation_scale * lw / log_thresh)
        return torch.tensor(factors)

    @torch.no_grad()
    def rollout(self, z0: torch.Tensor, actions: torch.Tensor,
                n_samples: int = 16,
                do_mask: torch.Tensor | None = None,
                do_values: torch.Tensor | None = None,
                do_steps: slice | None = None,
                inflate: bool = False,
                use_mixture: bool = True,
                generator: torch.Generator | None = None) -> torch.Tensor:
        """Quantile-propagated imagination.

        z0: (B, d); actions: (H, B, m).  Returns samples (n_samples, H, B, d).
        ``do_mask``/``do_values`` apply graph surgery at the steps selected by
        ``do_steps`` (default: every step).  ``inflate=True`` widens each
        node's predictive spread by its e-gate inflation factor, so
        invalid-looking mechanisms contribute honest extra uncertainty to
        exactly the variables they generate, propagating to descendants.
        """
        H = actions.shape[0]
        B = z0.shape[0]
        infl = self.node_inflation().to(z0.device) if inflate else \
            torch.ones(self.d, device=z0.device)
        z = z0.unsqueeze(0).expand(n_samples, B, self.d).reshape(-1, self.d)
        outs = []
        for t in range(H):
            a = actions[t].unsqueeze(0).expand(n_samples, B, self.m)
            a = a.reshape(-1, self.m)
            q = self.predict_quantiles(z, a, hard=True,
                                       use_mixture=use_mixture)
            do_now = do_mask is not None and (
                do_steps is None or (do_steps.start <= t < do_steps.stop))
            if do_now:
                q = self.apply_surgery(q, do_mask.to(z.device),
                                       do_values.to(z.device))
            z = sample_from_quantiles(
                q, inflation=infl.unsqueeze(0).expand(q.shape[0], self.d),
                generator=generator)
            outs.append(z.view(n_samples, B, self.d))
        return torch.stack(outs, dim=1)

    @torch.no_grad()
    def rollout_intervals(self, z0, actions, lo_idx: int = 1,
                          hi_idx: int = N_TAUS - 2, n_samples: int = 64,
                          **kw):
        """Horizon-resolved predictive intervals from rollout samples.

        Returns (lo, hi) each of shape (H, B, d) at the empirical quantile
        levels TAUS[lo_idx], TAUS[hi_idx] of the sampled trajectories.
        """
        from cairn.quantile import TAUS
        samples = self.rollout(z0, actions, n_samples=n_samples, **kw)
        lo = torch.quantile(samples, TAUS[lo_idx].item(), dim=0)
        hi = torch.quantile(samples, TAUS[hi_idx].item(), dim=0)
        return lo, hi

    # ------------------------------------------------------------------ #
    # Monitoring (deployment loop, algorithm.md 2.5)                      #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def observe(self, z: torch.Tensor, a: torch.Tensor,
                z_next: torch.Tensor,
                generator: torch.Generator | None = None) -> list[int]:
        """Feed one transition to every node's active e-gate.

        z, a, z_next: single transition, shapes (d,), (m,), (d,).
        Returns the list of node indices whose gate crossed 1/delta at this
        step (anytime-valid local alarms).
        """
        q = self.predict_quantiles(z.unsqueeze(0), a.unsqueeze(0),
                                   hard=True, use_mixture=True)[0]
        alarms = []
        self._last_raw_pits = [0.0] * self.d
        for i in range(self.d):
            u_raw = pit_value(q[i], z_next[i], generator=generator).item()
            self._last_raw_pits[i] = u_raw
            u = u_raw
            if self.calibrators[i] is not None:
                u = self.calibrators[i].transform(u, generator=generator)
            fired_before = self.gates.active(i).alarmed
            fired = self.gates.active(i).update(u)
            if fired and not fired_before:
                alarms.append(i)
        return alarms

    @torch.no_grad()
    def calibrate_pits(self, z: torch.Tensor, a: torch.Tensor,
                       z_next: torch.Tensor,
                       generator: torch.Generator | None = None) -> None:
        """Fit per-node PIT recalibrators on held-out data not used for
        fitting (the train/monitor separation of algorithm.md 2.3): the
        e-gate null then holds exactly despite quantile-head approximation
        error."""
        q = self.predict_quantiles(z, a, hard=True, use_mixture=True)
        for i in range(self.d):
            u = pit_value(q[:, i], z_next[:, i], generator=generator)
            self.calibrators[i] = PitCalibrator().fit(u)
