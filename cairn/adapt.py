"""Deployment loop: e-gate monitoring, evidence-triggered localized
adaptation, and online structure repair (algorithm.md sections 2.2 and 2.5).

On each observed transition every node's active e-gate is updated with the
PIT of the realized value.  When a node's wealth crosses 1/delta:

1. a fresh mechanism copy is spawned and few-shot fitted on a short recent
   window (localized adaptation — the rest of the graph is untouched);
2. if alarms at the same node repeat after refit, a local re-search over the
   node's parent set A_{.i} is run, scoring candidate parent sets by
   held-out pinball loss on the recent window (structure repair).
"""

from __future__ import annotations

from collections import deque

import torch

from cairn.quantile import pinball_loss


class OnlineAdapter:
    def __init__(self, model, buffer_size: int = 96,
                 refit_epochs: int = 300, refit_lr: float = 5e-3,
                 min_refit_samples: int = 24,
                 repair: bool = True, repair_after_alarms: int = 2,
                 adapt: bool = True):
        self.model = model
        self.buffer: deque = deque(maxlen=buffer_size)
        self.refit_epochs = refit_epochs
        self.refit_lr = refit_lr
        self.min_refit_samples = min_refit_samples
        self.repair = repair
        self.repair_after_alarms = repair_after_alarms
        self.adapt = adapt
        self.alarm_counts = [0] * model.d
        self.alarm_log: list[tuple[int, int]] = []   # (step, node)
        self.repair_log: list[tuple[int, int]] = []
        self._pending: set[int] = set()
        self._step = 0

    # ------------------------------------------------------------------ #

    def step(self, z, a, z_next,
             generator: torch.Generator | None = None) -> list[int]:
        """Feed one transition; returns nodes that alarmed at this step."""
        self._step += 1
        self.buffer.append((z.clone(), a.clone(), z_next.clone()))
        alarms = self.model.observe(z, a, z_next, generator=generator)
        for i in alarms:
            self.alarm_counts[i] += 1
            self.alarm_log.append((self._step, i))
            self._pending.add(i)
        # Alarms that fire before the refit window has filled are deferred,
        # not dropped.
        if self.adapt and len(self.buffer) >= self.min_refit_samples:
            for i in sorted(self._pending):
                self.handle_alarm(i)
            self._pending.clear()
        return alarms

    # ------------------------------------------------------------------ #

    def _buffer_tensors(self):
        zs, as_, zn = zip(*self.buffer)
        return (torch.stack(zs), torch.stack(as_), torch.stack(zn))

    def handle_alarm(self, i: int) -> None:
        if (self.repair
                and self.alarm_counts[i] >= self.repair_after_alarms):
            self.repair_structure(i)
        new_mech, dropped = self.model.libraries[i].spawn()
        self.few_shot_fit(i, new_mech)
        self.model.gates.add_member(i, dropped)  # fresh wealth, kept in sync

    def few_shot_fit(self, i: int, mech, a_col=None) -> float:
        """Fit one spawned mechanism on the recent window (few-shot by
        construction: f_i is small and its parent set sparse)."""
        z, a, zn = self._buffer_tensors()
        A, M = self.model.structure.hard_masks()
        a_col = A[:, i] if a_col is None else a_col
        opt = torch.optim.Adam(mech.parameters(), lr=self.refit_lr)
        for _ in range(self.refit_epochs):
            opt.zero_grad()
            q = mech(z, a, a_col, M[:, i])
            loss = pinball_loss(q, zn[:, i])
            loss.backward()
            opt.step()
        with torch.no_grad():
            return float(pinball_loss(mech(z, a, a_col, M[:, i]), zn[:, i]))

    # ------------------------------------------------------------------ #

    def repair_structure(self, i: int, margin: float = 0.02) -> bool:
        """Local re-search over A_{.i}: score the current parent set and all
        single-edge flips by held-out pinball after a quick refit; adopt the
        best flip if it beats the current set by a relative margin."""
        z, a, zn = self._buffer_tensors()
        n = z.shape[0]
        split = max(int(0.75 * n), 1)
        A, M = self.model.structure.hard_masks()
        m_col = M[:, i]

        def score(a_col: torch.Tensor) -> float:
            import copy
            mech = copy.deepcopy(self.model.libraries[i].members[-1])
            opt = torch.optim.Adam(mech.parameters(), lr=self.refit_lr)
            for _ in range(self.refit_epochs // 2):
                opt.zero_grad()
                loss = pinball_loss(
                    mech(z[:split], a[:split], a_col, m_col), zn[:split, i])
                loss.backward()
                opt.step()
            with torch.no_grad():
                return float(pinball_loss(
                    mech(z[split:], a[split:], a_col, m_col), zn[split:, i]))

        base_col = A[:, i]
        base_score = score(base_col)
        best_j, best_score = None, base_score
        for j in range(self.model.d):
            cand = base_col.clone()
            cand[j] = 1.0 - cand[j]
            s = score(cand)
            if s < best_score:
                best_j, best_score = j, s
        if best_j is not None and best_score < base_score * (1 - margin):
            with torch.no_grad():
                flip_on = base_col[best_j] == 0
                self.model.structure.logits_A[best_j, i] = \
                    3.0 if flip_on else -3.0
            self.repair_log.append((self._step, i))
            return True
        return False
