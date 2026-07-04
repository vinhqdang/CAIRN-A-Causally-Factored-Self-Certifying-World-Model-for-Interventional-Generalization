import torch

from cairn.quantile import (TAUS, monotone_quantiles, pinball_loss,
                            pit_value, sample_from_quantiles)


def test_monotone_quantiles_never_cross():
    raw = torch.randn(500, len(TAUS)) * 5
    q = monotone_quantiles(raw)
    assert (q[..., 1:] >= q[..., :-1]).all()


def test_pinball_minimized_at_true_quantiles():
    torch.manual_seed(0)
    y = torch.randn(20000)
    true_q = torch.quantile(y, TAUS).expand(20000, -1)
    shifted = true_q + 0.5
    assert pinball_loss(true_q, y) < pinball_loss(shifted, y)


def test_pit_uniform_under_valid_quantiles():
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(1)
    y = torch.randn(50000)
    q = torch.distributions.Normal(0, 1).icdf(TAUS).expand(50000, -1)
    u = pit_value(q, y, generator=gen)
    # Mean ~ 0.5, variance ~ 1/12, and decile counts roughly flat.
    assert abs(u.mean().item() - 0.5) < 0.01
    assert abs(u.var().item() - 1 / 12) < 0.005
    hist = torch.histc(u, bins=10, min=0, max=1) / len(u)
    assert (hist - 0.1).abs().max().item() < 0.03


def test_sample_pit_roundtrip():
    gen = torch.Generator().manual_seed(2)
    q = torch.distributions.Normal(0.3, 2.0).icdf(TAUS).expand(2000, -1)
    u = torch.rand(2000, generator=gen) * 0.8 + 0.1  # interior levels
    x = sample_from_quantiles(q, u=u)
    u_back = pit_value(q, x, generator=gen)
    assert (u - u_back).abs().max().item() < 1e-4


def test_inflation_widens_spread_around_median():
    q = torch.distributions.Normal(0.0, 1.0).icdf(TAUS).expand(1000, -1)
    gen = torch.Generator().manual_seed(3)
    x1 = sample_from_quantiles(q, generator=gen)
    gen = torch.Generator().manual_seed(3)
    x2 = sample_from_quantiles(q, inflation=3.0, generator=gen)
    assert x2.std() > 2.0 * x1.std()
