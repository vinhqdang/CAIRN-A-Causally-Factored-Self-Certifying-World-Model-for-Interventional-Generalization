"""Two-timescale training of CAIRN (algorithm.md section 2.3):
mechanism parameters at a fast learning rate, structure parameters (A, M)
at a slow rate with straight-through Gumbel estimation and temperature
annealing.  E-gates carry guarantees only at deployment, on data not used
for fitting, so they are untouched here (ONS self-tunes online).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from cairn.losses import (invariance_loss, one_step_pinball,
                          overshoot_losses)
from cairn.quantile import MEDIAN_IDX


@dataclass
class TrainConfig:
    steps: int = 4000
    batch: int = 256
    seg_batch: int = 64
    seg_len: int = 4
    lr_mech: float = 2e-3
    lr_struct: float = 5e-3           # slower *effective* timescale via
    struct_delay: int = 300           # delayed start + temperature anneal
    gamma_sparsity: float = 2e-3
    gamma_int: float = 1.0
    gamma_inv: float = 1e-3
    gamma_cal: float = 0.3
    temp_start: float = 1.0
    temp_end: float = 0.3
    seed: int = 0
    log_every: int = 500


def episodes_to_tensors(episodes) -> dict[str, torch.Tensor]:
    """Flatten episodes into 1-step transitions and index segments."""
    zs, as_, zn, dm, dv, rid, ep_idx, t_idx = [], [], [], [], [], [], [], []
    for k, ep in enumerate(episodes):
        T = ep.a.shape[0]
        zs.append(ep.z[:-1]); zn.append(ep.z[1:]); as_.append(ep.a)
        dm.append(ep.do_mask); dv.append(ep.do_values)
        rid.append(np.full(T, ep.regime))
        ep_idx.append(np.full(T, k)); t_idx.append(np.arange(T))
    out = {
        "z": torch.tensor(np.concatenate(zs), dtype=torch.float32),
        "a": torch.tensor(np.concatenate(as_), dtype=torch.float32),
        "z_next": torch.tensor(np.concatenate(zn), dtype=torch.float32),
        "do_mask": torch.tensor(np.concatenate(dm), dtype=torch.float32),
        "do_values": torch.tensor(np.concatenate(dv), dtype=torch.float32),
        "regime": torch.tensor(np.concatenate(rid), dtype=torch.long),
        "ep": torch.tensor(np.concatenate(ep_idx), dtype=torch.long),
        "t": torch.tensor(np.concatenate(t_idx), dtype=torch.long),
    }
    return out


def _sample_segments(episodes, n: int, h: int, rng: np.random.Generator):
    """Sample n segments of length h; returns stacked tensors."""
    zs, as_, dm, dv = [], [], [], []
    lengths = [ep.a.shape[0] for ep in episodes]
    for _ in range(n):
        k = int(rng.integers(len(episodes)))
        T = lengths[k]
        s = int(rng.integers(0, T - h))
        ep = episodes[k]
        zs.append(ep.z[s:s + h + 1]); as_.append(ep.a[s:s + h])
        dm.append(ep.do_mask[s:s + h]); dv.append(ep.do_values[s:s + h])
    to = lambda x: torch.tensor(np.stack(x), dtype=torch.float32)
    return to(zs), to(as_), to(dm), to(dv)


def train_cairn(model, episodes, cfg: TrainConfig = TrainConfig(),
                verbose: bool = True) -> list[dict]:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    data = episodes_to_tensors(episodes)
    n = data["z"].shape[0]
    n_regimes = int(data["regime"].max().item()) + 1

    mech_params = [p for f in model.base_mechanisms for p in f.parameters()]
    opt_mech = torch.optim.Adam(mech_params, lr=cfg.lr_mech)
    opt_struct = torch.optim.Adam(model.structure.parameters(),
                                  lr=cfg.lr_struct)
    history = []
    for step in range(cfg.steps):
        frac = step / max(cfg.steps - 1, 1)
        model.structure.temperature = (
            cfg.temp_start + (cfg.temp_end - cfg.temp_start) * frac)

        idx = torch.tensor(rng.integers(0, n, size=cfg.batch))
        z, a = data["z"][idx], data["a"][idx]
        z_next = data["z_next"][idx]
        pin1, q = one_step_pinball(model, z, a, z_next,
                                   data["do_mask"][idx],
                                   data["do_values"][idx])

        seg = _sample_segments(episodes, cfg.seg_batch, cfg.seg_len, rng)
        pin_over, pin_int, cal = overshoot_losses(model, *seg)

        med = q[..., MEDIAN_IDX]
        resid = (z_next - med) / (z_next.std(dim=0, keepdim=True) + 1e-6)
        inv = invariance_loss(resid, data["regime"][idx], n_regimes)

        loss = (pin1 + pin_over
                + cfg.gamma_sparsity * model.structure.sparsity_loss()
                + cfg.gamma_int * pin_int
                + cfg.gamma_inv * inv
                + cfg.gamma_cal * cal)

        opt_mech.zero_grad(); opt_struct.zero_grad()
        loss.backward()
        opt_mech.step()
        if step >= cfg.struct_delay:
            opt_struct.step()

        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            rec = {"step": step, "loss": loss.item(),
                   "pin1": pin1.item(), "pin_int": pin_int.item(),
                   "cal": cal.item(), "inv": inv.item()}
            history.append(rec)
            if verbose:
                print("  " + "  ".join(f"{k}={v:.4f}" if k != "step"
                                       else f"step={v}"
                                       for k, v in rec.items()), flush=True)
    return history
