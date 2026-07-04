# CAIRN: A Causally-Factored, Self-Certifying World Model for Interventional Generalization

**CAIRN** (**C**ausal **A**ction-conditioned **I**nterventional **R**ollout
**N**etwork) is a world-model architecture and training algorithm in which
(i) latent dynamics are factored over a learned sparse causal graph,
(ii) actions and external shocks enter as *interventions on specific
mechanisms* (Pearl graph surgery, implemented architecturally), and
(iii) every mechanism carries an internal betting martingale — the
**e-gate** — so the model knows, with anytime-valid statistical evidence,
which parts of itself are currently wrong, enabling mechanism-level
adaptation instead of whole-model failure under distribution shift.

The full specification, motivation, related work, theory targets, and paper
plan are in [`algorithm.md`](algorithm.md).  This repository implements the
complete algorithm and its evaluation suite on a ground-truth synthetic
dynamic Bayesian network (the M1–2 milestone of the plan).

## Repository layout

```
cairn/
  quantile.py       Quantile machinery: monotone heads, pinball loss, PIT,
                    inverse-CDF sampling, conformal PIT recalibration
  structure.py      Learned adjacency A and action-target mask M
                    (Gumbel-sigmoid, straight-through; DBN => no acyclicity
                    constraint needed)
  mechanisms.py     Per-node quantile-headed mechanism MLPs and the
                    evidence-weighted mechanism libraries
  egate.py          The e-gate: ONS-adapted betting e-processes over PIT
                    validity (location + dispersion bets), Ville alarms
  model.py          CairnWorldModel: factored transition kernel, do-surgery,
                    quantile-propagated rollouts with evidence-driven
                    uncertainty inflation, deployment monitoring
  losses.py         Training objective: pinball + sparsity + interventional
                    consistency + sparse-mechanism-shift invariance +
                    differentiable rollout-calibration penalty
  train.py          Two-timescale training (fast mechanisms, slow structure)
  adapt.py          Deployment loop: evidence-triggered localized mechanism
                    refits, online structure repair, sliding-holdout
                    PIT maintenance
  planner.py        CEM planning with CVaR over quantile rollouts
  baselines.py      Monolithic WM, PETS-style ensemble, global e-gate
                    ablation, oracle-tuned CUSUM detector
  envs/synthetic_dbn.py   Ground-truth nonlinear DBN with scriptable sparse
                          mechanism shifts and do-interventions
experiments/
  run_all.py        RQ1-RQ6 evaluation suite (writes results/*.json)
  make_tables.py    Regenerates RESULTS.md from the result JSONs
  common.py         Shared setup: model zoo, deployment refits, calibration
tests/              Unit tests (e-gate validity & power, surgery semantics,
                    PIT uniformity, training smoke, adaptation smoke)
```

## Quick start

```bash
pip install -r requirements.txt
python -m pytest tests -q                 # unit tests
python experiments/run_all.py --smoke     # fast end-to-end check (~2 min)
python experiments/run_all.py             # full suite (CPU, ~1.5 h)
python experiments/make_tables.py         # regenerate RESULTS.md
```

Results are reported in [`RESULTS.md`](RESULTS.md), regenerated only via
`make_tables.py` from the JSONs in `results/`.

## What the experiments show (see RESULTS.md for numbers)

- **RQ1 Interventional generalization**: rollout error under held-out
  do-interventions, with the structural descendant/non-descendant split.
- **RQ2 Detection & localization**: anytime-valid per-mechanism alarms vs
  oracle-tuned CUSUM, ensemble disagreement, and an unlocalized global
  e-gate; includes an exact-null verification of the Ville guarantee.
- **RQ3 Adaptation speed**: e-gate-triggered localized mechanism refits vs
  full fine-tuning at a matched gradient budget.
- **RQ4 Calibration**: horizon-resolved rollout interval coverage
  in-distribution, under shift (with and without evidence-driven
  uncertainty inflation), and after adaptation.
- **RQ5 Planning**: CVaR-CEM control before/after mechanism shifts.
- **RQ6 Structure recovery**: SHD/F1 of the learned graph and action-target
  mask vs regime diversity.
