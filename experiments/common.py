"""Shared experimental setup for the synthetic-DBN evaluation suite
(algorithm.md sections 3-5, state-space instantiation).

All experiments use the same ground-truth DBN (d=8 latents, m=3 actions,
sparse graph), four training regimes (nominal + three sparse mechanism
shifts), and the same trained model zoo:

- cairn          : full CAIRN (learned structure, all loss terms)
- cairn_noinv    : ablation without the sparse-mechanism-shift regularizer
- cairn_oracle   : ablation with the ground-truth graph fixed (upper bound)
- monolithic     : single quantile-headed MLP transition map
- ensemble       : 5-member PETS-style probabilistic ensemble
"""

from __future__ import annotations

import copy
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cairn.baselines import (EnsembleWM, MonolithicWM, train_ensemble,
                             train_monolithic)
from cairn.envs.synthetic_dbn import Regime, SyntheticDBN, default_regimes
from cairn.model import CairnWorldModel
from cairn.train import TrainConfig, train_cairn

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "results")

D, M = 8, 3
N_TRAIN_REGIMES = 4
EPISODES_PER_REGIME = 50
EPISODE_LEN = 100
P_DO = 0.15
TRAIN_STEPS = 6000
DELTA = 0.05


def save_json(name: str, payload: dict) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"[saved] {path}", flush=True)


def build_env(seed: int = 0):
    # sigma=0.2 keeps the signal-to-noise ratio realistic: with
    # near-deterministic dynamics every residual approximation error of the
    # quantile heads is statistically detectable, and no learned model can
    # satisfy the e-gate's conditional-validity null on long streams.
    env = SyntheticDBN(d=D, m=M, extra_parents=2, sigma=0.2, seed=seed)
    regimes = default_regimes(env, N_TRAIN_REGIMES, seed=seed + 7)
    return env, regimes


def build_dataset(env, regimes, seed: int = 1):
    return env.generate_dataset(regimes, EPISODES_PER_REGIME, EPISODE_LEN,
                                p_do=P_DO, seed=seed)


def make_cairn(seed: int, oracle_env=None) -> CairnWorldModel:
    torch.manual_seed(seed)
    model = CairnWorldModel(d=D, m=M, hidden=48, delta=DELTA)
    if oracle_env is not None:
        with torch.no_grad():
            model.structure.logits_A.copy_(
                torch.tensor(oracle_env.A_true, dtype=torch.float32) * 12 - 6)
            model.structure.logits_M.copy_(
                torch.tensor(oracle_env.M_true, dtype=torch.float32) * 12 - 6)
    return model


def train_zoo(env, episodes, seed: int = 0, steps: int = TRAIN_STEPS,
              which: tuple[str, ...] = ("cairn", "cairn_noinv",
                                        "cairn_oracle", "monolithic",
                                        "ensemble")) -> dict:
    zoo = {}
    t0 = time.time()
    if "cairn" in which:
        print("== training cairn ==", flush=True)
        zoo["cairn"] = make_cairn(seed)
        train_cairn(zoo["cairn"], episodes,
                    TrainConfig(steps=steps, seed=seed))
    if "cairn_noinv" in which:
        print("== training cairn_noinv ==", flush=True)
        zoo["cairn_noinv"] = make_cairn(seed + 1)
        train_cairn(zoo["cairn_noinv"], episodes,
                    TrainConfig(steps=steps, gamma_inv=0.0, seed=seed))
    if "cairn_oracle" in which:
        print("== training cairn_oracle ==", flush=True)
        zoo["cairn_oracle"] = make_cairn(seed + 2, oracle_env=env)
        train_cairn(zoo["cairn_oracle"], episodes,
                    TrainConfig(steps=steps, struct_delay=10 ** 9,
                                seed=seed))
    if "monolithic" in which:
        print("== training monolithic ==", flush=True)
        zoo["monolithic"] = MonolithicWM(D, M)
        train_monolithic(zoo["monolithic"], episodes, steps=steps,
                         seed=seed, verbose=True)
    if "ensemble" in which:
        print("== training ensemble ==", flush=True)
        zoo["ensemble"] = EnsembleWM(D, M, n_members=5)
        train_ensemble(zoo["ensemble"], episodes, steps=steps, seed=seed)
    print(f"== zoo trained in {time.time() - t0:.0f}s ==", flush=True)
    return zoo


def refit_for_deployment(model, nominal_episodes, steps: int = 4000,
                         seed: int = 0):
    """Regime-entry adaptation: freeze the learned structure, reinitialize
    the mechanisms, and fit them to deployment-regime data (two-phase, with
    a low-learning-rate polishing pass) before monitoring begins.

    Mechanisms trained on pooled multi-regime data state a regime-mixture
    conditional, which is genuinely invalid on any single regime — the
    e-gates would (correctly) flag it.  Deployment therefore starts from
    mechanisms adapted to the current regime; the gates then certify
    *continued* validity, which is their job.  Reinitialization avoids the
    regime-mixture local minimum."""
    model = copy.deepcopy(model)
    for f in model.base_mechanisms:
        for layer in f.net:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
    for cfg in [TrainConfig(steps=steps, struct_delay=10 ** 9,
                            gamma_inv=0.0, gamma_int=0.0, lr_mech=2e-3,
                            seed=seed, log_every=10 ** 9),
                TrainConfig(steps=steps // 2, struct_delay=10 ** 9,
                            gamma_inv=0.0, gamma_int=0.0, lr_mech=2e-4,
                            seed=seed + 1, log_every=10 ** 9)]:
        train_cairn(model, nominal_episodes, cfg, verbose=False)
    return model


def refit_monolithic_for_deployment(model, nominal_episodes,
                                    steps: int = 1500, seed: int = 0):
    model = copy.deepcopy(model)
    train_monolithic(model, nominal_episodes, steps=steps, lr=5e-4,
                     seed=seed)
    return model


# --------------------------------------------------------------------- #
# Shared evaluation helpers                                              #
# --------------------------------------------------------------------- #

def true_mean_trajectory(env, z0, actions, regime, H,
                         do_mask=None, do_values=None,
                         n_noise: int = 128, seed: int = 0) -> np.ndarray:
    """Monte-Carlo estimate of the true conditional-mean trajectory of the
    DBN under the given action sequence (and optional step-0 intervention),
    isolating model error from irreducible noise."""
    rng = np.random.default_rng(seed)
    acc = np.zeros((H, env.d))
    for _ in range(n_noise):
        z = z0.copy()
        for t in range(H):
            dm = do_mask if (do_mask is not None and t == 0) else None
            dv = do_values if (do_values is not None and t == 0) else None
            z = env.step(z, actions[t], regime, dm, dv, rng=rng)
            acc[t] += z
    return acc / n_noise


def model_mean_rollout(model, z0, actions, H, do_mask=None, do_values=None,
                       n_samples: int = 128, seed: int = 0,
                       inflate: bool = False) -> np.ndarray:
    """Mean over quantile-propagated rollout samples: (H, d)."""
    gen = torch.Generator().manual_seed(seed)
    z0_t = torch.tensor(z0, dtype=torch.float32).unsqueeze(0)
    a_t = torch.tensor(actions, dtype=torch.float32).unsqueeze(1)
    kw = {}
    if hasattr(model, "structure"):          # CAIRN path supports surgery
        kw = dict(do_mask=None if do_mask is None else
                  torch.tensor(do_mask, dtype=torch.float32),
                  do_values=None if do_values is None else
                  torch.tensor(do_values, dtype=torch.float32),
                  do_steps=slice(0, 1), inflate=inflate)
        samples = model.rollout(z0_t, a_t, n_samples=n_samples,
                                generator=gen, **kw)
    else:
        with torch.no_grad():
            samples = _baseline_rollout_with_do(model, z0_t, a_t, n_samples,
                                                do_mask, do_values, gen)
    return samples.mean(dim=0)[:, 0, :].numpy()


def _baseline_rollout_with_do(model, z0_t, a_t, n_samples,
                              do_mask, do_values, gen):
    """Monolithic/ensemble rollout with the same output-substitution
    intervention at step 0 (fair comparison: every model gets the do value
    clamped; only CAIRN additionally severs parents by construction)."""
    from cairn.quantile import sample_from_quantiles
    if isinstance(model, EnsembleWM):
        per = max(1, n_samples // len(model.members))
        return torch.cat(
            [_baseline_rollout_with_do(f, z0_t, a_t, per, do_mask,
                                       do_values, gen)
             for f in model.members], dim=0)
    H, B = a_t.shape[0], z0_t.shape[0]
    d, m = model.d, model.m
    z = z0_t.unsqueeze(0).expand(n_samples, B, d).reshape(-1, d)
    outs = []
    for t in range(H):
        a = a_t[t].unsqueeze(0).expand(n_samples, B, m).reshape(-1, m)
        q = model.predict_quantiles(z, a)
        z = sample_from_quantiles(q, generator=gen)
        if t == 0 and do_mask is not None:
            dm = torch.tensor(do_mask, dtype=torch.float32)
            dv = torch.tensor(do_values, dtype=torch.float32)
            z = z * (1 - dm) + dv * dm
        outs.append(z.view(n_samples, B, d))
    return torch.stack(outs, dim=1)


def standardized_residual_stats(model, env, regime, n_steps: int = 1000,
                                seed: int = 0):
    """Per-node std of 1-step median residuals on a stream — used to
    standardize residuals for the CUSUM baseline."""
    rng = np.random.default_rng(seed)
    z = np.zeros(env.d)
    resids = []
    for _ in range(n_steps):
        a = rng.normal(0, 1.0, env.m)
        z_next = env.step(z, a, regime, rng=rng)
        with torch.no_grad():
            zt = torch.tensor(z, dtype=torch.float32).unsqueeze(0)
            at = torch.tensor(a, dtype=torch.float32).unsqueeze(0)
            kw = {"hard": True} if hasattr(model, "structure") else {}
            med = model.median_prediction(zt, at, **kw)
        resids.append(z_next - med[0].numpy())
        z = z_next
    r = np.array(resids)
    return r.mean(axis=0), r.std(axis=0) + 1e-8
