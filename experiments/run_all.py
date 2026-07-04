"""Run the CAIRN evaluation suite (RQ1-RQ6, algorithm.md section 5) on the
ground-truth synthetic DBN and write one JSON per research question into
results/.  Usage:

    python experiments/run_all.py [--only rq1,rq2,...] [--seed 0]

Each JSON is written as soon as its RQ finishes, so partial results survive
interruption; rerunning with --only recomputes just the requested parts.
"""

from __future__ import annotations

import argparse
import copy
import time

import numpy as np
import torch

from common import (D, DELTA, DO_TRAIN_NODES, M, RESULTS_DIR,
                    build_dataset, build_env,
                    make_cairn, model_mean_rollout,
                    refit_for_deployment, refit_monolithic_for_deployment,
                    save_json, standardized_residual_stats, train_zoo,
                    true_mean_trajectory, TRAIN_STEPS)

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cairn.adapt import OnlineAdapter
from cairn.baselines import CusumDetector, GlobalEGate
from cairn.envs.synthetic_dbn import Regime
from cairn.planner import CEMPlanner
from cairn.quantile import pinball_loss
from cairn.train import TrainConfig, episodes_to_tensors, train_cairn

NOMINAL = Regime()
SMOKE = False


def _smooth_actions(rng, H, m, scale=1.0):
    a = np.zeros((H, m))
    cur = rng.normal(0, scale, m)
    for t in range(H):
        cur = 0.8 * cur + 0.2 * rng.normal(0, scale, m)
        a[t] = cur
    return a


def _stream(env, regime, n_steps, rng, action_scale=1.0,
            reset_every=100):
    """Yield transitions (z, a, z_next) from a random-action stream.

    The state resets every ``reset_every`` steps so the deployment
    distribution matches the episodic training/calibration distribution
    (the DBN has multiple attractors, and a never-resetting stream settles
    into a single basin the model was not calibrated on).  Gate wealth
    carries across resets — anytime validity is exactly what makes
    continuous monitoring over concatenated episodes sound."""
    z = rng.normal(0, 0.5, env.d)
    a_cur = rng.normal(0, action_scale, env.m)
    for t in range(n_steps):
        if reset_every and t > 0 and t % reset_every == 0:
            z = rng.normal(0, 0.5, env.d)
            a_cur = rng.normal(0, action_scale, env.m)
        a_cur = 0.8 * a_cur + 0.2 * rng.normal(0, action_scale, env.m)
        z_next = env.step(z, a_cur, regime, rng=rng)
        yield z.copy(), a_cur.copy(), z_next.copy()
        z = z_next


def _t(x):
    return torch.tensor(x, dtype=torch.float32)


# ===================================================================== #
# RQ1: interventional generalization                                     #
# ===================================================================== #

def run_rq1(env, zoo, seed=0, H=10, do_values=(-4.0, 4.0)):
    """Do-interventions with values far outside the reachable state range
    (|z| <= ~1/(1-rho) + noise < 3, do values +-4), on nodes never
    intervened during training (heldout) and on trained do-targets; h-step
    rollout error of the model-mean trajectory vs the true conditional
    mean, split into descendants / non-descendants of the intervened node.
    Off-support do-states are where factorization matters: a non-child
    mechanism provably ignores the intervened coordinate, while a
    monolithic map extrapolates with it as an input everywhere."""
    print("== RQ1 ==", flush=True)
    rng = np.random.default_rng(seed + 100)
    horizons = [1, 3, 5, 10]
    eval_models = ["cairn", "cairn_noinv", "cairn_oracle", "monolithic",
                   "ensemble"]
    groups = ["heldout", "trained"]
    per_model = {name: {g: {"sq": np.zeros((H, env.d)), "n": 0,
                            "desc": {h: [] for h in horizons},
                            "nondesc": {h: [] for h in horizons},
                            "nondesc_ref": {h: [] for h in horizons}}
                        for g in groups}
                 for name in eval_models if name in zoo}
    heldout_nodes = [i for i in range(env.d) if i not in DO_TRAIN_NODES]
    cases = [(i, v, "heldout") for i in heldout_nodes for v in do_values]
    cases += [(i, v, "trained") for i in DO_TRAIN_NODES for v in do_values]
    if SMOKE:
        cases = cases[:3]
    for ci, (i, v, group) in enumerate(cases):
        z0 = rng.normal(0, 0.5, env.d)
        actions = _smooth_actions(rng, H, env.m)
        do_mask = np.zeros(env.d); do_mask[i] = 1.0
        do_val = np.zeros(env.d); do_val[i] = v
        true_do = true_mean_trajectory(env, z0, actions, NOMINAL, H,
                                       do_mask, do_val, seed=seed + ci)
        true_ref = true_mean_trajectory(env, z0, actions, NOMINAL, H,
                                        seed=seed + ci)
        desc = env.descendants({i}, H)
        nondesc = [j for j in range(env.d) if j not in desc]
        desc = sorted(desc)
        for name in per_model:
            model = zoo[name]
            acc = per_model[name][group]
            pred_do = model_mean_rollout(model, z0, actions, H,
                                         do_mask, do_val, seed=seed + ci)
            pred_ref = model_mean_rollout(model, z0, actions, H,
                                          seed=seed + ci)
            err = (pred_do - true_do) ** 2
            acc["sq"] += err
            acc["n"] += 1
            for h in horizons:
                acc["desc"][h].append(
                    float(np.sqrt(err[h - 1, desc].mean())))
                if nondesc:
                    acc["nondesc"][h].append(
                        float(np.sqrt(err[h - 1, nondesc].mean())))
                    acc["nondesc_ref"][h].append(float(np.sqrt(
                        ((pred_ref - true_ref) ** 2)[h - 1, nondesc].mean())))
    out = {"horizons": list(range(1, H + 1)),
           "do_train_nodes": DO_TRAIN_NODES, "models": {}}
    for name, by_group in per_model.items():
        out["models"][name] = {}
        for g, acc in by_group.items():
            if acc["n"] == 0:
                continue
            rmse_h = np.sqrt(acc["sq"].mean(axis=1) / acc["n"])
            out["models"][name][g] = {
                "rmse_by_horizon": rmse_h.tolist(),
                "desc_rmse": {h: float(np.mean(vs))
                              for h, vs in acc["desc"].items()},
                "nondesc_rmse": {h: float(np.mean(vs))
                                 for h, vs in acc["nondesc"].items()},
                "nondesc_rmse_no_intervention": {
                    h: float(np.mean(vs))
                    for h, vs in acc["nondesc_ref"].items()},
            }
            print(f"  {name}[{g}]: rmse@h10={rmse_h[-1]:.4f}", flush=True)
    return out


# ===================================================================== #
# RQ2: detection & localization                                          #
# ===================================================================== #

def _tune_cusum(model, env, seed):
    """Oracle-tuned CUSUM: standardization stats + per-node threshold set to
    the max statistic seen on a null calibration stream (zero false alarms
    on calibration by construction — the strongest tuning a practitioner
    could do, which CAIRN must beat without any tuning)."""
    mu, sd = standardized_residual_stats(model, env, NOMINAL,
                                         n_steps=150 if SMOKE else 1500,
                                         seed=seed + 11)
    det = CusumDetector(env.d, threshold=np.inf)
    rng = np.random.default_rng(seed + 12)
    max_stat = np.zeros(env.d)
    for z, a, zn in _stream(env, NOMINAL, 200 if SMOKE else 4000, rng):
        with torch.no_grad():
            med = model.median_prediction(_t(z).unsqueeze(0),
                                          _t(a).unsqueeze(0), hard=True)
        r = (zn - med[0].numpy() - mu) / sd
        det.update(r)
        max_stat = np.maximum(max_stat, det.pos)
    return mu, sd, max_stat * 1.05


def _oracle_null_check(env, seed=0):
    """Exact-null Ville verification: gates are fed PITs computed from the
    TRUE environment conditional distribution, so the stated predictive
    distribution is exactly valid and P(W ever >= 1/delta) <= delta must
    hold per gate.  This verifies the e-gate machinery end-to-end on real
    streams, independently of any learned model."""
    from math import erf, sqrt
    from cairn.egate import EGate
    n_streams, stream_len = (2, 300) if SMOKE else (16, 6000)
    alarmed, total = 0, 0
    for st in range(n_streams):
        rng = np.random.default_rng(seed + 3000 + st)
        gates = [EGate(DELTA) for _ in range(env.d)]
        for z, a, zn in _stream(env, NOMINAL, stream_len, rng,
                                reset_every=50):
            drive = np.tanh(z @ env.W + a @ env.C + env.b)
            mean = env.rho * z + drive
            u = 0.5 * (1.0 + np.array(
                [erf(x / sqrt(2.0)) for x in (zn - mean) / env.sigma]))
            for i, g in enumerate(gates):
                g.update(float(np.clip(u[i], 1e-6, 1 - 1e-6)))
        alarmed += sum(1 for g in gates if g.alarmed or g.log_wealth >=
                       np.log(1.0 / DELTA))
        total += env.d
    return alarmed / total, n_streams, stream_len


def _make_monitor(model):
    """Deployment monitor: e-gates + sliding-holdout maintenance, no
    adaptation (pure detection, so alarms are observable)."""
    return OnlineAdapter(copy.deepcopy(model), adapt=False, repair=False,
                         maintain=True,
                         recal_every=100 if SMOKE else 300)


def run_rq2(env, zoo, seed=0):
    print("== RQ2 ==", flush=True)
    cairn = zoo["cairn_deploy"]
    oracle = zoo["cairn_oracle_deploy"]
    mono = zoo["monolithic_deploy"]
    ens = zoo["ensemble_deploy"]
    mu, sd, cusum_thresh = _tune_cusum(cairn, env, seed)

    # Ensemble-disagreement threshold, tuned the same oracle way.
    rng = np.random.default_rng(seed + 13)
    max_dis = np.zeros(env.d)
    for z, a, zn in _stream(env, NOMINAL, 200 if SMOKE else 2000, rng,
                            reset_every=50):
        dis = ens.member_disagreement(_t(z).unsqueeze(0),
                                      _t(a).unsqueeze(0))[0].numpy()
        max_dis = np.maximum(max_dis, dis)
    dis_thresh = max_dis * 1.05

    oracle_frac, on_streams, on_len = _oracle_null_check(env, seed)
    print(f"  exact-null Ville check: alarm fraction "
          f"{oracle_frac:.4f} (delta={DELTA})", flush=True)

    # ---- null streams: empirical false-alarm rate ---- #
    null = {"egate": 0, "egate_oracle_graph": 0, "cusum": 0,
            "ensemble": 0, "global_egate": 0}
    n_null_streams, null_len = (1, 200) if SMOKE else (3, 6000)
    gate_units = 0
    for s in range(n_null_streams):
        mon = _make_monitor(cairn)
        mon_o = _make_monitor(oracle)
        gen = torch.Generator().manual_seed(seed + s)
        gen_o = torch.Generator().manual_seed(seed + 50 + s)
        cus = CusumDetector(env.d)
        glob = GlobalEGate(mono, delta=DELTA)
        if "_cal" in zoo:
            glob.calibrate(*zoo["_cal"], generator=gen)
        rng = np.random.default_rng(seed + 20 + s)
        alarmed_c, alarmed_d = set(), set()
        glob_fired = False
        for z, a, zn in _stream(env, NOMINAL, null_len, rng,
                                reset_every=50):
            mon.step(_t(z), _t(a), _t(zn), generator=gen)
            mon_o.step(_t(z), _t(a), _t(zn), generator=gen_o)
            with torch.no_grad():
                med = cairn.median_prediction(_t(z).unsqueeze(0),
                                              _t(a).unsqueeze(0), hard=True)
            r = (zn - med[0].numpy() - mu) / sd
            cus.update(r)
            for i in range(env.d):
                if cus.pos[i] > cusum_thresh[i]:
                    alarmed_c.add(i)
            dis = ens.member_disagreement(_t(z).unsqueeze(0),
                                          _t(a).unsqueeze(0))[0].numpy()
            alarmed_d.update(np.nonzero(dis > dis_thresh)[0].tolist())
            glob_fired = glob.observe(_t(z), _t(a), _t(zn),
                                      generator=gen) or glob_fired
        alarmed_e = {i for _, i in mon.alarm_log}
        alarmed_o = {i for _, i in mon_o.alarm_log}
        null["egate"] += len(alarmed_e)
        null["egate_oracle_graph"] += len(alarmed_o)
        null["cusum"] += len(alarmed_c)
        null["ensemble"] += len(alarmed_d)
        null["global_egate"] += int(glob_fired)
        gate_units += env.d
        print(f"  null stream {s}: egate={sorted(alarmed_e)} "
              f"oracle-graph={sorted(alarmed_o)} cusum={sorted(alarmed_c)} "
              f"ens={sorted(alarmed_d)} global={glob_fired}", flush=True)

    # ---- shift scenarios ---- #
    scenarios = [{"S": (i,), "gain": -1.0} for i in range(env.d)]
    scenarios += [{"S": (0, 4), "gain": 0.3}, {"S": (2, 6), "gain": -1.0}]
    if SMOKE:
        scenarios = scenarios[:2]
    pre, post, loc_window = (30, 90, 60) if SMOKE else (300, 900, 300)
    res = {det: {"delays": [], "precision": [], "recall": [], "f1": []}
           for det in ["egate", "egate_oracle_graph", "cusum", "ensemble"]}
    glob_delays = []
    for sc_i, sc in enumerate(scenarios):
        S = set(sc["S"])
        shift = Regime(shifted=tuple(S), gain=sc["gain"])
        mon = _make_monitor(cairn)
        mon_o = _make_monitor(oracle)
        gen = torch.Generator().manual_seed(seed + 40 + sc_i)
        gen_o = torch.Generator().manual_seed(seed + 90 + sc_i)
        cus = CusumDetector(env.d)
        glob = GlobalEGate(mono, delta=DELTA)
        if "_cal" in zoo:
            glob.calibrate(*zoo["_cal"], generator=gen)
        rng = np.random.default_rng(seed + 60 + sc_i)
        first = {det: {} for det in res}
        glob_first = None
        stream = list(_stream(env, NOMINAL, pre, rng, reset_every=50)) + \
            list(_stream(env, shift, post, rng, reset_every=50))
        for t, (z, a, zn) in enumerate(stream):
            for i in mon.step(_t(z), _t(a), _t(zn), generator=gen):
                first["egate"].setdefault(i, t)
            for i in mon_o.step(_t(z), _t(a), _t(zn), generator=gen_o):
                first["egate_oracle_graph"].setdefault(i, t)
            with torch.no_grad():
                med = cairn.median_prediction(_t(z).unsqueeze(0),
                                              _t(a).unsqueeze(0), hard=True)
            r = (zn - med[0].numpy() - mu) / sd
            cus.update(r)
            for i in range(env.d):
                if cus.pos[i] > cusum_thresh[i]:
                    first["cusum"].setdefault(i, t)
            dis = ens.member_disagreement(_t(z).unsqueeze(0),
                                          _t(a).unsqueeze(0))[0].numpy()
            for i in np.nonzero(dis > dis_thresh)[0]:
                first["ensemble"].setdefault(int(i), t)
            if glob_first is None and glob.observe(_t(z), _t(a), _t(zn),
                                                   generator=gen):
                glob_first = t
        for det in res:
            hits = {i: ft for i, ft in first[det].items() if ft >= pre}
            in_S = [ft for i, ft in hits.items() if i in S]
            delay = min(in_S) - pre if in_S else post
            res[det]["delays"].append(delay)
            loc = {i for i, ft in hits.items() if ft < pre + loc_window}
            tp = len(loc & S)
            prec = tp / len(loc) if loc else 0.0
            rec = tp / len(S)
            f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
            res[det]["precision"].append(prec)
            res[det]["recall"].append(rec)
            res[det]["f1"].append(f1)
        glob_delays.append((glob_first - pre) if glob_first is not None
                           and glob_first >= pre else post)
        print(f"  scenario S={sorted(S)}: "
              + " ".join(f"{det}:delay={res[det]['delays'][-1]} "
                         f"f1={res[det]['f1'][-1]:.2f}" for det in res),
              flush=True)

    out = {
        "delta": DELTA,
        "oracle_null_alarm_fraction": oracle_frac,
        "oracle_null_streams": on_streams,
        "oracle_null_stream_steps": on_len,
        "null": {det: {"alarm_fraction": null[det] / gate_units
                       if det != "global_egate"
                       else null[det] / n_null_streams,
                       "unit": "per-gate" if det != "global_egate"
                       else "per-stream"}
                 for det in null},
        "null_stream_steps": null_len,
        "detectors": {det: {
            "median_delay": float(np.median(res[det]["delays"])),
            "mean_delay": float(np.mean(res[det]["delays"])),
            "localization_f1": float(np.mean(res[det]["f1"])),
            "localization_precision": float(np.mean(res[det]["precision"])),
            "localization_recall": float(np.mean(res[det]["recall"])),
        } for det in res},
        "global_egate_median_delay": float(np.median(glob_delays)),
        "note": "CUSUM/ensemble thresholds oracle-tuned on null streams; "
                "e-gates use delta=0.05 with no tuning. egate rows use the "
                "deployment-refit models with sliding-holdout maintenance.",
    }
    return out


# ===================================================================== #
# RQ3: adaptation speed                                                  #
# ===================================================================== #

def _eval_rmse(model, eval_z, eval_a, eval_zn, mixture=False, nodes=None):
    with torch.no_grad():
        if hasattr(model, "structure"):
            q = model.predict_quantiles(eval_z, eval_a, hard=True,
                                        use_mixture=mixture)
        else:
            q = model.predict_quantiles(eval_z, eval_a)
        from cairn.quantile import MEDIAN_IDX
        med = q[..., MEDIAN_IDX]
        err = (med - eval_zn) ** 2
        if nodes is not None:
            err = err[:, list(nodes)]
        return float(torch.sqrt(err.mean()))


def run_rq3(env, zoo, seed=0):
    print("== RQ3 ==", flush=True)
    checkpoints = [10, 20] if SMOKE else [25, 50, 100, 200, 400]
    scenarios = [{"S": (1,), "gain": -1.0}, {"S": (3, 5), "gain": -1.0}]
    out = {"checkpoints": checkpoints, "scenarios": []}
    for sc_i, sc in enumerate(scenarios):
        shift = Regime(shifted=sc["S"], gain=sc["gain"])
        # Held-out evaluation transitions from the shifted regime.
        eval_eps = env.generate_dataset([shift], 2 if SMOKE else 8, 80, p_do=0.0,
                                        seed=seed + 900 + sc_i)
        ev = episodes_to_tensors(eval_eps)
        eval_z, eval_a, eval_zn = ev["z"], ev["a"], ev["z_next"]
        nom_eps = env.generate_dataset([NOMINAL], 2 if SMOKE else 8, 80, p_do=0.0,
                                       seed=seed + 950 + sc_i)
        nv = episodes_to_tensors(nom_eps)

        S = list(sc["S"])
        methods = {}
        # --- CAIRN localized (e-gate triggered mechanism refits) --- #
        model = copy.deepcopy(zoo["cairn_deploy"])
        # Recovery is judged on the shifted nodes: averaging over all d
        # nodes would dilute a 1-node shift below the noise floor.
        pre_rmse = _eval_rmse(model, nv["z"], nv["a"], nv["z_next"],
                              nodes=S)
        adapter = OnlineAdapter(model, buffer_size=96,
                                refit_epochs=30 if SMOKE else 300,
                                min_refit_samples=8 if SMOKE else 24)
        gen = torch.Generator().manual_seed(seed + sc_i)
        rng = np.random.default_rng(seed + 70 + sc_i)
        curve, curve_all, n_seen = [], [], 0
        stream = _stream(env, shift, checkpoints[-1], rng)
        curve.append(_eval_rmse(model, eval_z, eval_a, eval_zn,
                                mixture=True, nodes=S))
        curve_all.append(_eval_rmse(model, eval_z, eval_a, eval_zn,
                                    mixture=True))
        for z, a, zn in stream:
            adapter.step(_t(z), _t(a), _t(zn), generator=gen)
            n_seen += 1
            if n_seen in checkpoints:
                curve.append(_eval_rmse(model, eval_z, eval_a, eval_zn,
                                        mixture=True, nodes=S))
                curve_all.append(_eval_rmse(model, eval_z, eval_a, eval_zn,
                                            mixture=True))
        methods["cairn_local"] = {
            "rmse_curve": curve, "rmse_curve_all_nodes": curve_all,
            "refit_nodes": sorted({i for _, i in adapter.alarm_log}),
        }

        # --- full fine-tuning baselines (matched gradient budget) --- #
        for name, base in [("cairn_full_ft", zoo["cairn_deploy"]),
                           ("monolithic_ft", zoo["monolithic_deploy"])]:
            model = copy.deepcopy(base)
            rng = np.random.default_rng(seed + 70 + sc_i)  # same stream
            buf = []
            curve = [_eval_rmse(model, eval_z, eval_a, eval_zn, nodes=S)]
            curve_all = [_eval_rmse(model, eval_z, eval_a, eval_zn)]
            params = ([p for f in model.base_mechanisms
                       for p in f.parameters()]
                      if hasattr(model, "base_mechanisms")
                      else list(model.parameters()))
            opt = torch.optim.Adam(params, lr=5e-3)
            n_seen = 0
            for z, a, zn in _stream(env, shift, checkpoints[-1], rng):
                buf.append((z, a, zn))
                buf = buf[-96:]
                n_seen += 1
                if n_seen in checkpoints:
                    bz = _t(np.stack([b[0] for b in buf]))
                    ba = _t(np.stack([b[1] for b in buf]))
                    bzn = _t(np.stack([b[2] for b in buf]))
                    for _ in range(30 if SMOKE else 300):  # matched to CAIRN refit epochs
                        opt.zero_grad()
                        if hasattr(model, "structure"):
                            q = model.predict_quantiles(bz, ba, hard=True)
                        else:
                            q = model.predict_quantiles(bz, ba)
                        loss = pinball_loss(q, bzn)
                        loss.backward()
                        opt.step()
                    curve.append(_eval_rmse(model, eval_z, eval_a, eval_zn,
                                            nodes=S))
                    curve_all.append(
                        _eval_rmse(model, eval_z, eval_a, eval_zn))
            methods[name] = {"rmse_curve": curve,
                             "rmse_curve_all_nodes": curve_all}

        thresh = 1.10 * pre_rmse
        for name, rec in methods.items():
            cps = [0] + checkpoints
            rec["samples_to_recovery"] = next(
                (cps[k] for k, r in enumerate(rec["rmse_curve"])
                 if r <= thresh), None)
        out["scenarios"].append({
            "S": list(sc["S"]), "gain": sc["gain"],
            "pre_shift_rmse": pre_rmse, "recovery_threshold": thresh,
            "methods": methods,
        })
        print(f"  S={sc['S']}: " + " ".join(
            f"{n}:rec={m['samples_to_recovery']}"
            for n, m in methods.items()), flush=True)
    return out


# ===================================================================== #
# RQ4: calibration                                                       #
# ===================================================================== #

def _coverage(model, env, regime, horizons, n_cases, seed,
              inflate=False, mixture=True, node=None):
    """Empirical coverage of 80%/90% rollout intervals vs realized
    trajectories.  ``node`` restricts scoring to one variable (e.g. the
    shifted mechanism's output, where interval honesty is actually at
    stake — the all-node average dilutes a single-node shift)."""
    rng = np.random.default_rng(seed)
    H = max(horizons)
    hits80 = {h: [] for h in horizons}
    hits90 = {h: [] for h in horizons}
    for c in range(n_cases):
        z0 = rng.normal(0, 0.5, env.d)
        actions = _smooth_actions(rng, H, env.m)
        # One realized trajectory.
        z = z0.copy()
        traj = []
        for t in range(H):
            z = env.step(z, actions[t], regime, rng=rng)
            traj.append(z.copy())
        traj = np.array(traj)
        z0_t = _t(z0).unsqueeze(0)
        a_t = _t(actions).unsqueeze(1)
        if hasattr(model, "structure"):
            samples = model.rollout(z0_t, a_t, n_samples=64,
                                    inflate=inflate, use_mixture=mixture,
                                    generator=torch.Generator()
                                    .manual_seed(seed + c))
        else:
            samples = model.rollout(z0_t, a_t, n_samples=64,
                                    generator=torch.Generator()
                                    .manual_seed(seed + c))
        s = samples[:, :, 0, :].numpy()          # (S, H, d)
        lo90, hi90 = np.quantile(s, 0.05, axis=0), np.quantile(s, 0.95, 0)
        lo80, hi80 = np.quantile(s, 0.10, axis=0), np.quantile(s, 0.90, 0)
        sel = slice(None) if node is None else slice(node, node + 1)
        for h in horizons:
            hits90[h].append(float(np.mean(
                (traj[h - 1, sel] >= lo90[h - 1, sel])
                & (traj[h - 1, sel] <= hi90[h - 1, sel]))))
            hits80[h].append(float(np.mean(
                (traj[h - 1, sel] >= lo80[h - 1, sel])
                & (traj[h - 1, sel] <= hi80[h - 1, sel]))))
    return ({h: float(np.mean(v)) for h, v in hits80.items()},
            {h: float(np.mean(v)) for h, v in hits90.items()})


def run_rq4(env, zoo, seed=0):
    print("== RQ4 ==", flush=True)
    horizons = [1, 5, 10, 15]
    out = {"horizons": horizons, "nominal": {}, "shift": {}}
    for name in ["cairn_deploy", "ensemble_deploy"]:
        c80, c90 = _coverage(zoo[name], env, NOMINAL, horizons,
                             8 if SMOKE else 96,
                             seed + 200)
        out["nominal"][name.replace("_deploy", "")] = {"cov80": c80,
                                                       "cov90": c90}
        print(f"  nominal {name}: cov80={c80} cov90={c90}", flush=True)

    # Shift condition: node 2 sign-flip; warm the gates on 150 post-shift
    # transitions WITHOUT adaptation, then measure interval honesty.
    shift = Regime(shifted=(2,), gain=-1.0)
    model = copy.deepcopy(zoo["cairn_deploy"])
    gen = torch.Generator().manual_seed(seed + 5)
    rng = np.random.default_rng(seed + 300)
    for z, a, zn in _stream(env, shift, 30 if SMOKE else 150, rng):
        model.observe(_t(z), _t(a), _t(zn), generator=gen)
    for label, inflate in [("pre_adapt_no_inflation", False),
                           ("pre_adapt_inflated", True)]:
        c80, c90 = _coverage(model, env, shift, horizons,
                             8 if SMOKE else 96, seed + 201,
                             inflate=inflate)
        n80, n90 = _coverage(model, env, shift, horizons,
                             8 if SMOKE else 96, seed + 201,
                             inflate=inflate, node=2)
        out["shift"][label] = {"cov80": c80, "cov90": c90,
                               "cov80_shifted_node": n80,
                               "cov90_shifted_node": n90}
        print(f"  shift {label}: cov90={c90} node2={n90}", flush=True)
    # Ensemble under the same shift (no mechanism for honesty).
    c80, c90 = _coverage(zoo["ensemble_deploy"], env, shift, horizons,
                         8 if SMOKE else 96, seed + 201)
    out["shift"]["ensemble"] = {"cov80": c80, "cov90": c90}

    # Post-adaptation: run the full deployment loop for 400 steps.
    model = copy.deepcopy(zoo["cairn_deploy"])
    adapter = OnlineAdapter(model, buffer_size=96,
                            refit_epochs=30 if SMOKE else 300,
                            min_refit_samples=8 if SMOKE else 24)
    gen = torch.Generator().manual_seed(seed + 6)
    rng = np.random.default_rng(seed + 301)
    for z, a, zn in _stream(env, shift, 60 if SMOKE else 400, rng):
        adapter.step(_t(z), _t(a), _t(zn), generator=gen)
    c80, c90 = _coverage(model, env, shift, horizons,
                         8 if SMOKE else 96, seed + 202, inflate=True)
    n80, n90 = _coverage(model, env, shift, horizons,
                         8 if SMOKE else 96, seed + 202, inflate=True,
                         node=2)
    out["shift"]["post_adapt"] = {"cov80": c80, "cov90": c90,
                                  "cov80_shifted_node": n80,
                                  "cov90_shifted_node": n90}
    print(f"  shift post_adapt: cov90={c90}", flush=True)
    return out


# ===================================================================== #
# RQ5 (lite): downstream planning utility                                #
# ===================================================================== #

def run_rq5(env, zoo, seed=0):
    print("== RQ5 ==", flush=True)
    rng = np.random.default_rng(seed + 400)
    # A reachable goal: terminal state of a random action sequence.
    z = np.zeros(env.d)
    for t in range(30):
        z = env.step(z, rng.normal(0, 1.0, env.m), NOMINAL, rng=rng)
    goal = _t(z)

    def reward_fn(zs):
        return -((zs - goal) ** 2).sum(dim=-1)

    def episode_return(model, regime, ep_seed, inflate=True):
        planner = CEMPlanner(model, reward_fn, horizon=6, population=40,
                             elites=6, iters=3, n_rollout_samples=10,
                             inflate=inflate)
        rng_e = np.random.default_rng(ep_seed)
        gen = torch.Generator().manual_seed(ep_seed)
        z = rng_e.normal(0, 0.5, env.d)
        total = 0.0
        for t in range(3 if SMOKE else 20):
            a = planner.plan(_t(z), generator=gen).numpy()
            z = env.step(z, a, regime, rng=rng_e)
            total += -float(((z - goal.numpy()) ** 2).sum())
        return total

    shift = Regime(shifted=(2,), gain=-1.0)
    n_eps = 1 if SMOKE else 6
    out = {"goal": goal.tolist(), "episodes": n_eps, "conditions": {}}

    conds = {}
    conds["nominal"] = {
        "cairn": [episode_return(zoo["cairn_deploy"], NOMINAL, seed + 500 + k)
                  for k in range(n_eps)],
        "monolithic": [episode_return(zoo["monolithic_deploy"], NOMINAL,
                                      seed + 500 + k) for k in range(n_eps)],
    }
    # Post-shift, pre-adaptation: CAIRN's gates warmed on 150 transitions.
    warmed = copy.deepcopy(zoo["cairn_deploy"])
    gen = torch.Generator().manual_seed(seed + 7)
    rng2 = np.random.default_rng(seed + 501)
    for z0_, a0_, zn0_ in _stream(env, shift, 30 if SMOKE else 150, rng2):
        warmed.observe(_t(z0_), _t(a0_), _t(zn0_), generator=gen)
    conds["shift_pre_adapt"] = {
        "cairn": [episode_return(warmed, shift, seed + 510 + k)
                  for k in range(n_eps)],
        "monolithic": [episode_return(zoo["monolithic_deploy"], shift,
                                      seed + 510 + k) for k in range(n_eps)],
    }
    # Post-adaptation.
    adapted = copy.deepcopy(zoo["cairn_deploy"])
    adapter = OnlineAdapter(adapted, buffer_size=96,
                            refit_epochs=30 if SMOKE else 300,
                            min_refit_samples=8 if SMOKE else 24)
    gen = torch.Generator().manual_seed(seed + 8)
    rng3 = np.random.default_rng(seed + 502)
    for z0_, a0_, zn0_ in _stream(env, shift, 60 if SMOKE else 400, rng3):
        adapter.step(_t(z0_), _t(a0_), _t(zn0_), generator=gen)
    conds["shift_post_adapt"] = {
        "cairn": [episode_return(adapted, shift, seed + 520 + k)
                  for k in range(n_eps)],
    }
    for cname, models in conds.items():
        out["conditions"][cname] = {
            n: {"mean_return": float(np.mean(v)),
                "sem": float(np.std(v) / np.sqrt(len(v))),
                "returns": v}
            for n, v in models.items()}
        print(f"  {cname}: " + " ".join(
            f"{n}={np.mean(v):.1f}" for n, v in models.items()), flush=True)
    return out


# ===================================================================== #
# RQ6: structure recovery vs regime diversity                            #
# ===================================================================== #

def _structure_scores(model, env):
    A_hat, M_hat = model.structure.hard_masks()
    A_hat = A_hat.numpy().astype(int)
    M_hat = M_hat.numpy().astype(int)
    A_true, M_true = env.A_true, env.M_true

    def f1(pred, true):
        tp = int(((pred == 1) & (true == 1)).sum())
        fp = int(((pred == 1) & (true == 0)).sum())
        fn = int(((pred == 0) & (true == 1)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        return 2 * p * r / (p + r) if p + r else 0.0

    return {"shd_A": int(np.sum(A_hat != A_true)),
            "f1_A": f1(A_hat, A_true),
            "shd_M": int(np.sum(M_hat != M_true)),
            "f1_M": f1(M_hat, M_true)}


def run_rq6(env, seed=0, steps=3000):
    if SMOKE:
        steps = 120
    print("== RQ6 ==", flush=True)
    from cairn.envs.synthetic_dbn import default_regimes
    out = {"regime_counts": [], "scores": []}
    for n_regimes in ([1, 2] if SMOKE else [1, 2, 4, 6]):
        regs = default_regimes(env, n_regimes, seed=seed + 7)
        eps = env.generate_dataset(regs, max(160 // n_regimes, 27), 100,
                                   p_do=0.08, seed=seed + 1)
        model = make_cairn(seed + n_regimes)
        train_cairn(model, eps, TrainConfig(steps=steps, seed=seed),
                    verbose=False)
        sc = _structure_scores(model, env)
        out["regime_counts"].append(n_regimes)
        out["scores"].append(sc)
        print(f"  regimes={n_regimes}: {sc}", flush=True)
    return out


# ===================================================================== #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="rq1,rq2,rq3,rq4,rq5,rq6")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=TRAIN_STEPS)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    only = set(args.only.split(","))
    global SMOKE
    if args.smoke:
        SMOKE = True
        args.steps = min(args.steps, 200)
    torch.set_num_threads(4)

    t0 = time.time()
    env, regimes = build_env(args.seed)
    episodes = build_dataset(env, regimes, seed=args.seed + 1)
    need_zoo = only & {"rq1", "rq2", "rq3", "rq4", "rq5"}
    cache_path = os.path.join(RESULTS_DIR,
                              f"zoo_seed{args.seed}.pt") if not SMOKE else None
    zoo = {}
    if need_zoo and cache_path and os.path.exists(cache_path):
        print(f"== loading cached zoo from {cache_path} ==", flush=True)
        zoo = torch.load(cache_path, weights_only=False)
    elif need_zoo:
        zoo = train_zoo(env, episodes, seed=args.seed, steps=args.steps)
    if zoo and "cairn_deploy" not in zoo:
        # Deployment pipeline (regime-entry adaptation + conformal PIT
        # calibration, algorithm.md 2.3): monitored/adapted/planned models
        # are refit to the deployment (nominal) regime with the learned
        # structure frozen, then their PITs are recalibrated on held-out
        # nominal data so each e-gate's null holds at deployment.
        refit_steps = 400 if SMOKE else 4000
        nom_eps = env.generate_dataset([NOMINAL], 8 if SMOKE else 120, 100,
                                       p_do=0.0, seed=args.seed + 555)
        print("== deployment refits ==", flush=True)
        zoo["cairn_deploy"] = refit_for_deployment(
            zoo["cairn"], nom_eps, steps=refit_steps, seed=args.seed)
        zoo["cairn_oracle_deploy"] = refit_for_deployment(
            zoo["cairn_oracle"], nom_eps, steps=refit_steps,
            seed=args.seed)
        zoo["monolithic_deploy"] = refit_monolithic_for_deployment(
            zoo["monolithic"], nom_eps, steps=refit_steps // 2,
            seed=args.seed)
        zoo["ensemble_deploy"] = copy.deepcopy(zoo["ensemble"])
        from cairn.baselines import train_monolithic as _tm
        for k, f in enumerate(zoo["ensemble_deploy"].members):
            _tm(f, nom_eps, steps=refit_steps // 2, lr=5e-4,
                seed=args.seed * 100 + k)
        cal_eps = env.generate_dataset([NOMINAL], 20, 100, p_do=0.0,
                                       seed=args.seed + 777)
        cal = episodes_to_tensors(cal_eps)
        gen = torch.Generator().manual_seed(args.seed + 777)
        for name in ["cairn_deploy", "cairn_oracle_deploy"]:
            zoo[name].calibrate_pits(cal["z"], cal["a"], cal["z_next"],
                                     generator=gen)
        zoo["_cal"] = (cal["z"], cal["a"], cal["z_next"])
        if cache_path:
            torch.save(zoo, cache_path)
            print(f"== zoo cached to {cache_path} ==", flush=True)
    meta = {"d": D, "m": M, "seed": args.seed, "train_steps": args.steps,
            "A_true": env.A_true.tolist(), "M_true": env.M_true.tolist()}
    save_json("meta.json", meta)

    runners = {"rq1": lambda: run_rq1(env, zoo, args.seed),
               "rq2": lambda: run_rq2(env, zoo, args.seed),
               "rq3": lambda: run_rq3(env, zoo, args.seed),
               "rq4": lambda: run_rq4(env, zoo, args.seed),
               "rq5": lambda: run_rq5(env, zoo, args.seed),
               "rq6": lambda: run_rq6(env, args.seed)}
    for rq in ["rq1", "rq2", "rq3", "rq4", "rq5", "rq6"]:
        if rq in only:
            t = time.time()
            save_json(f"{rq}.json", runners[rq]())
            print(f"[{rq} done in {time.time() - t:.0f}s]", flush=True)
    print(f"[all done in {time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
