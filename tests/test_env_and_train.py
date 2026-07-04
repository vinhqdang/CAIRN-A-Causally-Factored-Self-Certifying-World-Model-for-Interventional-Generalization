import numpy as np
import torch

from cairn.adapt import OnlineAdapter
from cairn.envs.synthetic_dbn import Regime, SyntheticDBN, default_regimes
from cairn.model import CairnWorldModel
from cairn.train import TrainConfig, train_cairn


def test_env_ground_truth_shapes_and_descendants():
    env = SyntheticDBN(d=6, m=2, seed=3)
    assert env.A_true.shape == (6, 6) and np.all(np.diag(env.A_true) == 1)
    assert env.M_true.shape == (2, 6) and env.M_true.sum() >= 2
    desc = env.descendants({0}, horizon=6)
    assert 0 in desc


def test_env_do_intervention_pins_value():
    env = SyntheticDBN(d=6, m=2, seed=3)
    z = np.zeros(6); a = np.zeros(2)
    do_mask = np.zeros(6); do_mask[4] = 1
    do_values = np.zeros(6); do_values[4] = 9.0
    z_next = env.step(z, a, Regime(), do_mask, do_values)
    assert z_next[4] == 9.0


def test_training_reduces_loss_smoke():
    env = SyntheticDBN(d=5, m=2, seed=1)
    regimes = default_regimes(env, 2, seed=1)
    episodes = env.generate_dataset(regimes, episodes_per_regime=6, T=60,
                                    p_do=0.05, seed=2)
    model = CairnWorldModel(d=5, m=2, hidden=24)
    cfg = TrainConfig(steps=400, batch=128, seg_batch=16, seg_len=3,
                      log_every=200, seed=0)
    hist = train_cairn(model, episodes, cfg, verbose=False)
    assert hist[-1]["pin1"] < hist[0]["pin1"] * 0.7


def test_adapter_spawns_and_improves_after_shift():
    torch.manual_seed(0)
    env = SyntheticDBN(d=5, m=2, seed=1)
    regimes = default_regimes(env, 2, seed=1)
    episodes = env.generate_dataset(regimes, episodes_per_regime=8, T=60,
                                    p_do=0.0, seed=2)
    model = CairnWorldModel(d=5, m=2, hidden=24)
    train_cairn(model, episodes,
                TrainConfig(steps=600, batch=128, seg_batch=16, seg_len=3,
                            log_every=300, seed=0), verbose=False)
    adapter = OnlineAdapter(model, buffer_size=64, refit_epochs=120,
                            repair=False)
    # Deploy into a hard shift of node 2's mechanism.
    shift = Regime(shifted=(2,), gain=-1.5)
    rng = np.random.default_rng(5)
    gen = torch.Generator().manual_seed(5)
    z = np.zeros(5)
    fired = []
    for t in range(400):
        a = rng.normal(0, 1.0, 2)
        z_next = env.step(z, a, shift, rng=rng)
        fired += adapter.step(torch.tensor(z, dtype=torch.float32),
                              torch.tensor(a, dtype=torch.float32),
                              torch.tensor(z_next, dtype=torch.float32),
                              generator=gen)
        z = z_next
    assert 2 in fired, f"expected node 2 alarm, got {sorted(set(fired))}"
    assert len(model.libraries[2]) > 1        # localized adaptation spawned
