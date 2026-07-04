"""Online structure repair (algorithm.md 2.2, use #3): persistent wealth
growth at a node after mechanism refit indicates a wrong parent set and
triggers a local re-search over A_{.i} scored on held-out data."""

import numpy as np
import torch

from cairn.adapt import OnlineAdapter
from cairn.envs.synthetic_dbn import Regime, SyntheticDBN
from cairn.model import CairnWorldModel
from cairn.train import TrainConfig, train_cairn


def test_repair_restores_deleted_parent():
    torch.manual_seed(0)
    env = SyntheticDBN(d=5, m=2, extra_parents=1, sigma=0.2, seed=4)
    episodes = env.generate_dataset([Regime()], 30, 80, p_do=0.0, seed=2)

    # Oracle-structure model, trained on nominal data only.
    model = CairnWorldModel(d=5, m=2, hidden=32)
    with torch.no_grad():
        model.structure.logits_A.copy_(
            torch.tensor(env.A_true, dtype=torch.float32) * 12 - 6)
        model.structure.logits_M.copy_(
            torch.tensor(env.M_true, dtype=torch.float32) * 12 - 6)
    train_cairn(model, episodes,
                TrainConfig(steps=1200, struct_delay=10 ** 9, seed=0,
                            log_every=10 ** 9), verbose=False)

    # Corrupt: delete one true non-self parent edge j -> i.
    i = 0
    parents = [j for j in np.nonzero(env.A_true[:, i])[0] if j != i]
    j = int(parents[0])
    with torch.no_grad():
        model.structure.logits_A[j, i] = -6.0

    adapter = OnlineAdapter(model, buffer_size=96, refit_epochs=150,
                            repair=True, repair_after_alarms=2)
    rng = np.random.default_rng(3)
    gen = torch.Generator().manual_seed(3)
    z = rng.normal(0, 0.5, 5)
    for t in range(500):
        if t % 100 == 0:
            z = rng.normal(0, 0.5, 5)
        a = rng.normal(0, 1.0, 2)
        zn = env.step(z, a, Regime(), rng=rng)
        adapter.step(torch.tensor(z, dtype=torch.float32),
                     torch.tensor(a, dtype=torch.float32),
                     torch.tensor(zn, dtype=torch.float32), generator=gen)
        z = zn
        if model.structure.logits_A[j, i] > 0:
            break
    assert any(node == i for _, node in adapter.alarm_log), \
        "corrupted mechanism never alarmed"
    assert model.structure.logits_A[j, i] > 0, \
        "structure repair did not restore the deleted parent edge"
