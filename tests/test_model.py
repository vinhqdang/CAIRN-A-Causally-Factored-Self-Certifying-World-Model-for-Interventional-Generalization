import numpy as np
import torch

from cairn.model import CairnWorldModel
from cairn.quantile import N_TAUS


def _oracle_model(d=5, m=2):
    """CAIRN with a fixed chain graph 0 -> 1 -> 2 -> 3 -> 4 (plus self
    edges) and action 0 targeting node 0, action 1 targeting node 4."""
    torch.manual_seed(0)
    model = CairnWorldModel(d=d, m=m, hidden=16)
    with torch.no_grad():
        model.structure.logits_A.fill_(-5.0)
        for i in range(d):
            model.structure.logits_A[i, i] = 5.0
        for i in range(d - 1):
            model.structure.logits_A[i, i + 1] = 5.0
        model.structure.logits_M.fill_(-5.0)
        model.structure.logits_M[0, 0] = 5.0
        model.structure.logits_M[1, d - 1] = 5.0
    return model


def test_surgery_replaces_intervened_node_exactly():
    model = _oracle_model()
    z = torch.randn(3, 5)
    a = torch.randn(3, 2)
    do_mask = torch.zeros(5); do_mask[2] = 1.0
    do_values = torch.zeros(5); do_values[2] = 7.5
    q = model.predict_quantiles(z, a, hard=True)
    q_do = model.apply_surgery(q, do_mask, do_values)
    assert torch.allclose(q_do[:, 2], torch.full((3, N_TAUS), 7.5))
    assert torch.allclose(q_do[:, 0], q[:, 0])


def test_rollout_intervention_affects_only_descendants():
    """do(z^2 <- v) in the chain 0->1->2->3->4 must leave nodes 0, 1
    untouched at every horizon and node 2's immediate value pinned —
    the structural signature of graph surgery."""
    model = _oracle_model()
    z0 = torch.zeros(1, 5)
    H = 4
    actions = torch.zeros(H, 1, 2)
    do_mask = torch.zeros(5); do_mask[2] = 1.0
    do_values = torch.zeros(5); do_values[2] = 5.0
    gen1 = torch.Generator().manual_seed(42)
    plain = model.rollout(z0, actions, n_samples=64, generator=gen1)
    gen2 = torch.Generator().manual_seed(42)
    doped = model.rollout(z0, actions, n_samples=64,
                          do_mask=do_mask, do_values=do_values,
                          do_steps=slice(0, 1), generator=gen2)
    # Non-descendants of node 2 (nodes 0 and 1): identical trajectories.
    assert torch.allclose(plain[..., :2], doped[..., :2], atol=1e-5)
    # Intervened node pinned at the surgery step.
    assert torch.allclose(doped[:, 0, :, 2], torch.tensor(5.0))
    # Descendants (3, 4) must differ after propagation.
    assert (plain[:, 1:, :, 3:] - doped[:, 1:, :, 3:]).abs().mean() > 1e-3


def test_descendants_reachability():
    model = _oracle_model()
    targets = torch.zeros(5); targets[2] = 1.0
    reach = model.structure.descendants(targets, horizon=4)
    assert reach.tolist() == [0.0, 0.0, 1.0, 1.0, 1.0]


def test_node_inflation_grows_with_wealth():
    model = _oracle_model()
    base = model.node_inflation()
    assert torch.allclose(base, torch.ones(5))
    # Force wealth up at node 3 by feeding biased PITs.
    rng = np.random.default_rng(0)
    for _ in range(300):
        model.gates.active(3).update(float(rng.beta(4, 1)))
    infl = model.node_inflation()
    assert infl[3] > 1.5 and infl[0] == 1.0


def test_observe_alarms_localize_to_broken_node():
    """Feed transitions where node 1's realized value is wildly biased
    while others are drawn from the model's own predictive distribution
    (valid mechanisms => uniform PITs): only node 1's gate should alarm."""
    from cairn.quantile import sample_from_quantiles
    model = _oracle_model()
    gen = torch.Generator().manual_seed(0)
    alarmed_nodes = set()
    z = torch.zeros(5)
    a = torch.zeros(2)
    for _ in range(400):
        q = model.predict_quantiles(z.unsqueeze(0), a.unsqueeze(0),
                                    hard=True)[0]
        z_next = sample_from_quantiles(q, generator=gen)   # healthy nodes
        z_next[1] = q[1, -1] + 1.0             # far above the top quantile
        alarmed_nodes.update(model.observe(z, a, z_next, generator=gen))
        z = torch.zeros(5)                     # keep the stream stationary
    assert 1 in alarmed_nodes
    assert alarmed_nodes.issubset({1})
