# CAIRN: A Causally-Factored, Self-Certifying World Model for Interventional Generalization

**CAIRN** = **C**ausal **A**ction-conditioned **I**nterventional **R**ollout **N**etwork

**Target venues (primary → fallback):** IEEE RA-L → CoRL 2027 → IROS 2027; finance instantiation → ACM ICAIF 2026 → Journal of Financial Stability

**Positioning in one sentence:** CAIRN is a *new world-model architecture and training algorithm* — not a post-hoc wrapper — in which (i) latent dynamics are factored over a learned sparse causal graph, (ii) actions and external shocks enter as *interventions on specific mechanisms* rather than global conditioning, and (iii) every mechanism carries an internal betting martingale (e-gate) so the model *knows, with anytime-valid statistical evidence, which parts of itself are currently wrong* — enabling mechanism-level adaptation instead of whole-model failure under distribution shift.

---

## 1. Motivation and Research Gap

### 1.1 What today's world models get wrong

State-of-the-art world models — RSSM/DreamerV3, TD-MPC2's latent dynamics, and the 2025–2026 video world models (Genie 3, Cosmos-Predict 2.5, IRASim, CtrlWorld) — all share a structural commitment: **the transition function is a monolithic map** $\hat{s}_{t+1} = f_\theta(s_t, a_t)$ (or its video-token equivalent). Every latent dimension depends on every other; actions condition the whole map. Three well-documented failures follow directly from this monolithic design:

1. **Interventional brittleness.** When one mechanism of the environment changes — object mass, surface friction, a swapped tool; in markets, a depegged stablecoin or an exploited bridge — the *entire* learned transition is off-support. The model cannot localize the change, so it cannot adapt to it, and rollouts fail globally even though most of the world is unchanged. The 2026 robotics world-model survey identifies exactly this: rollouts may look convincing while violating dynamics in ways that break closed-loop control, and ACT-Bench documents that SOTA driving world models do not faithfully execute the very actions they are conditioned on.
2. **No native uncertainty semantics.** Monolithic models emit point predictions (or ensembles with heuristic variance). There is no principled per-component notion of "this part of my dynamics is stale," so planners either trust everything or truncate rollouts by hand-tuned depth.
3. **Actions are not interventions.** Physically and economically, an action manipulates *specific* variables (the gripper pose; the collateral ratio of one protocol) and effects propagate through structure. Conditioning a monolithic map on the action vector discards this sparsity, which is precisely the inductive bias needed for compositional generalization to unseen action–state combinations.

Recent causal world-model work (e.g., "Causal World Modeling for Robot Control," 2026) begins to address point 3 but retains monolithic uncertainty handling (points 1–2): when its causal graph or a mechanism becomes wrong at deployment, nothing inside the model detects or localizes this.

### 1.2 The gap, stated precisely

> **No existing world model (a) factors its transition kernel over an explicitly learned causal graph with actions as targeted interventions, (b) equips each causal mechanism with an internal, anytime-valid evidence process that measures whether that mechanism is currently valid, and (c) uses this mechanism-level evidence to drive localized adaptation and calibrated, horizon-resolved rollout uncertainty — trained jointly, end-to-end, in a single algorithm.**

The three ingredients exist separately (causal dynamics factorization; betting e-processes; quantile/distributional heads) but **their joint construction is new**, and the joint construction is what produces the qualitatively new capability: a world model that degrades *locally* and *detectably* under intervention instead of globally and silently.

### 1.3 Why an applied venue cares

- **Robotics:** A manipulation robot whose world model localizes "the friction mechanism changed" can keep planning with the 90% of dynamics that remain valid, re-learn one mechanism from a handful of samples (mechanism-level fine-tuning is few-shot by construction, since each $f_i$ is small), and report calibrated confidence per rollout horizon. This is deployment safety framed constructively, not as a monitor bolted on.
- **Finance/DeFi:** Contagion simulation is *the* canonical interventional question — "what happens to the network if protocol X fails?" is a do-operation, not a conditional expectation. A world model of the protocol network that natively represents interventions, and whose per-protocol mechanisms carry validity evidence across regime shifts, is a stress-testing instrument regulators and risk teams currently lack. This instantiation reuses the VEIN data pipeline directly.
- **Feasible without frontier compute:** CAIRN's novelty is architectural and algorithmic, demonstrated in latent/state space (dim 32–256), not pixel space. All experiments fit on one A100.

### 1.4 Nearest prior work and differentiation

| Work | Shares | Lacks |
|---|---|---|
| DreamerV3 / RSSM, TD-MPC2 | Latent dynamics for planning | Any causal factorization; any validity semantics; monolithic failure under shift |
| Causal dynamics learning for MBRL (Wang et al. 2022; CDL; Denoised MDPs) | Sparse causal transition structure | Actions as learned intervention targets; no internal validity evidence; no calibrated rollouts; offline-only structure learning |
| "Causal World Modeling for Robot Control" (2026) | Causal graph in a robot world model | Mechanism-level anytime-valid monitoring; localized adaptation; native calibration |
| Sparse-mechanism-shift / interventional causal representation learning (Schölkopf et al.; Ahuja et al.) | The theoretical premise (shifts hit few mechanisms) | Any dynamics model, planner integration, or online evidence machinery |
| Conformal/ACI time-series UQ; conformal test martingales | Validity statistics | These are model-external; CAIRN internalizes the martingale as a differentiable-in-effect gating signal inside the architecture |
| DeXposure-FM (2026, DeFi graph foundation model) | Financial network forecasting | Not action/intervention-conditioned; no causal semantics; authors explicitly note forecasts degrade at structural breaks — the exact failure CAIRN targets |

---

## 2. The CAIRN Algorithm

### 2.1 Model class

**State.** An encoder $q_\phi$ maps observations to a latent state partitioned into $d$ scalar-or-block variables $z_t = (z_t^1, \dots, z_t^d)$. In the finance instantiation the partition is given (one block per protocol/asset node); in robotics it is learned with a disentanglement-encouraging prior (block-wise VAE or slot-style encoder).

**Causal graph.** A learned binary adjacency $A \in \{0,1\}^{d \times d}$ (parents among latents) and an **action-target mask** $M \in \{0,1\}^{m \times d}$ (which action coordinates intervene on which latents). Both are parameterized with Gumbel-sigmoid relaxations and sparsity penalties; acyclicity within a timestep is not required because edges run $t \to t{+}1$ (a dynamic Bayesian network, so structure learning avoids NOTEARS-style acyclicity constraints entirely — a deliberate simplification that removes the least stable part of differentiable causal discovery).

**Factored mechanisms.** The transition kernel factorizes:
$$
p_\theta(z_{t+1} \mid z_t, a_t) \;=\; \prod_{i=1}^{d} p_{\theta_i}\!\big( z_{t+1}^i \;\big|\; z_t \odot A_{\cdot i},\; a_t \odot M_{\cdot i} \big),
$$
where each mechanism $f_i$ is a small MLP (or per-node GNN message function in the graph instantiation) with a **distributional head**: it outputs a set of quantiles $\{\hat{z}^{i}_{t+1}(\tau)\}_{\tau \in T}$ (pinball-trained), not a point. Rollouts propagate quantile-sampled trajectories, giving horizon-resolved predictive intervals natively.

**Interventions as first-class inputs.** An intervention $\text{do}(z^i \leftarrow \cdot)$ — an agent action hitting its targets via $M$, or an exogenous shock label in finance — *replaces* mechanism $i$'s output rather than conditioning it. This is the graph-surgery semantics of Pearl's do-operator implemented architecturally: severed parents, substituted value. Counterfactual rollouts (needed for stress testing and for the training loss below) are therefore a forward pass, not a fine-tune.

### 2.2 The e-gate: internal anytime-valid mechanism evidence (core novelty)

Each mechanism $i$ maintains a **betting wealth process** over its own recent predictive validity. Let $u_t^i \in (0,1)$ be the probability-integral-transform (PIT) value of the realized $z_{t+1}^i$ under mechanism $i$'s predictive quantile distribution. If mechanism $i$ is valid, $u_t^i$ is (approximately) uniform. Mechanism $i$ bets against uniformity:
$$
W_t^i \;=\; \prod_{u \le t} \big(1 + \lambda_u^i \, g(u_u^i)\big), \qquad \mathbb{E}[g(U)] = 0 \text{ for } U \sim \text{Unif}(0,1),
$$
with bets $\lambda^i$ adapted by Online Newton Step. By Ville's inequality, $W_t^i \ge 1/\delta$ ever occurring has probability $\le \delta$ while mechanism $i$ is valid — a **time-uniform, per-mechanism certificate** with no tuning and no multiple-testing correction across continuous monitoring.

The wealth enters the model in three places, which is what makes this an *algorithm* rather than a monitor:

1. **Gated mixture-of-mechanisms.** Each node keeps a small library $\{f_i^{(1)}, \dots, f_i^{(K)}\}$ (base mechanism + spawned adaptations). The rollout uses evidence weights $\pi_i^{(k)} \propto \exp(-\beta \log W_t^{i,(k)})$ — mechanisms currently accumulating evidence of invalidity are smoothly down-weighted; when wealth crosses $1/\delta$, a fresh mechanism copy is spawned and rapidly fitted on a short recent window (few-shot, because $f_i$ is small and its parent set is sparse). This is **localized, evidence-triggered adaptation**: the rest of the graph is untouched.
2. **Rollout uncertainty inflation.** During $h$-step imagination, each mechanism's predictive quantile spread at step $u$ is inflated by a factor increasing in $\log W_u^i$ — invalid-looking mechanisms contribute honest extra uncertainty to exactly the state variables they generate, propagating structurally to descendants. The planner sees *where* the imagined future is untrustworthy, not just *that* it is.
3. **Structure repair signal.** Persistent wealth growth at node $i$ *after* mechanism refit indicates a wrong parent set, triggering a local re-search over $A_{\cdot i}$ (add/remove candidate parents scored by held-out wealth decay) — online causal-structure repair driven by anytime-valid evidence, which to our knowledge has no precedent in world-model literature.

### 2.3 Training objective

Joint loss over trajectory data spanning multiple regimes $e \in \mathcal{E}$ (in sim: scripted perturbation regimes; in finance: dated market regimes):

$$
\mathcal{L} = \underbrace{\textstyle\sum_i \mathcal{L}^{\text{pin}}_i}_{\text{quantile prediction}}
+ \gamma_1 \underbrace{\|A\|_1 + \|M\|_1}_{\text{sparsity}}
+ \gamma_2 \underbrace{\mathcal{L}^{\text{int}}}_{\text{interventional consistency}}
+ \gamma_3 \underbrace{\mathcal{L}^{\text{inv}}}_{\text{mechanism invariance}}
+ \gamma_4 \underbrace{\mathcal{L}^{\text{cal}}}_{\text{rollout calibration}}
$$

- $\mathcal{L}^{\text{pin}}$: pinball loss per mechanism per quantile (teacher-forced 1-step) **plus** multi-step latent-overshooting variants to fight compounding error.
- $\mathcal{L}^{\text{int}}$ (interventional consistency): on trajectory segments where an action/shock with known targets occurs, the do-surgery forward pass must match observed outcomes on *descendants* of the target while *non-descendants'* predictions must be unchanged relative to the counterfactual no-intervention pass. This supervises $M$ and $A$ with interventional (not merely observational) signal — the key to identifiability.
- $\mathcal{L}^{\text{inv}}$ (sparse-mechanism-shift regularizer): across regimes $e$, penalize the number of mechanisms whose per-regime residual distributions differ (energy-distance penalty with a Gumbel top-k relaxation). This encodes the assumption that regime changes hit few mechanisms, pushing the factorization toward the *true* one (a wrong factorization spreads any shift across many pseudo-mechanisms and pays the penalty).
- $\mathcal{L}^{\text{cal}}$: differentiable coverage penalty (smoothed indicator) on $h$-step rollout intervals against held-out trajectory batches, so calibration is optimized in training, not patched afterward.

Optimization: standard two-timescale scheme — mechanism parameters at fast learning rate, structure parameters $(A, M)$ at slow rate with straight-through Gumbel estimation; e-gate wealth processes run in inference mode during training epochs solely to initialize betting hyperparameters (they carry guarantees only at deployment, on data not used for fitting — this train/monitor separation is stated honestly and enforced by a sliding holdout buffer online).

### 2.4 Planning with CAIRN

Any sampling-based planner (CEM/MPPI) or Dreamer-style actor-critic plugs in unchanged, with two upgrades exploited in experiments: (i) trajectory scoring uses quantile-propagated returns (risk-sensitive planning via CVaR over rollout quantiles); (ii) candidate action sequences whose effects flow through low-wealth (healthy) mechanisms are preferred — the planner *routes around* the broken part of the world model, a behavior impossible for monolithic models.

### 2.5 Pseudocode (deployment loop)

```
# offline: train (q_φ, {f_i}, A, M) with L on multi-regime data
loop t:
  z_t ← q_φ(o_t)
  # plan: quantile rollouts with evidence-weighted mixtures, CVaR scoring
  a_t ← planner({f_i^(k)}, π from wealth W, A, M, z_t)
  execute a_t, observe o_{t+1}, z_{t+1}
  for i in 1..d:
      u ← PIT of z^i_{t+1} under mechanism i's predictive quantiles
      W^i ← W^i · (1 + λ^i g(u));  update λ^i (ONS)
      if W^i ≥ 1/δ:                      # anytime-valid local alarm
          spawn f_i^(K+1); few-shot fit on recent window; reset its wealth
          if repeated alarms: local parent re-search over A_{·i}
```

### 2.6 Theory targets

1. **Per-mechanism false-alarm control:** $\Pr(\exists t: W_t^i \ge 1/\delta \mid \text{mechanism } i \text{ valid}) \le \delta$ (Ville), robust to optional stopping and to other mechanisms adapting — validity of one e-gate does not depend on the rest of the model, by construction of PIT residuals. Family-wide control via e-value merging, no Bonferroni.
2. **Detection & localization:** for a shift confined to mechanism set $S$ raising PIT non-uniformity by KL $\ge \kappa$, expected detection delay $O(\log(1/\delta)/\kappa)$ at nodes in $S$, while nodes outside $S$ alarm with probability $\le \delta$ — this *localization guarantee* is the formal statement of "degrades locally, not globally," and is the paper's flagship proposition.
3. **Identifiability (conditional):** under the sparse-mechanism-shift assumption and sufficient regime diversity, the loss $\mathcal{L}^{\text{int}} + \mathcal{L}^{\text{inv}}$ is minimized only by factorizations Markov-equivalent to the true DBN restricted to shifted components (adapting existing interventional CRL results to the temporal setting; stated with its assumptions, verified empirically in sim where ground-truth graphs are known).
4. **Regret decomposition (informal → empirical):** planning regret after a mechanism-$S$ shift scales with $|S|$ and the refit sample complexity of $\{f_i\}_{i\in S}$, versus full-model sample complexity for monolithic baselines.

---

## 3. Baselines

**Monolithic world models (the main comparison class):**
1. **DreamerV3** (RSSM) — canonical latent world model, with and without online fine-tuning after shift.
2. **TD-MPC2** — latent dynamics + MPC, current strong MBRL baseline.
3. **Transformer world model** (IRIS-style token dynamics, small scale) — represents the autoregressive-token family at matched parameter count.

**Uncertainty-equipped variants (isolating the calibration claim):**
4. **PETS-style probabilistic ensemble** (5–7 members) with trajectory sampling.
5. **DreamerV3 + deep-ensemble heads**; **DreamerV3 + ACI-style post-hoc interval adaptation** (a fair "wrapper" competitor — CAIRN must beat what a wrapper can do, since we argued wrappers are insufficient).

**Causal/structured world models (isolating the causal-factorization claim):**
6. **CDL / causal dynamics learning** (Wang et al.) adapted to our environments.
7. **"Causal World Modeling for Robot Control" (2026)** — closest 2026 competitor; expected parity on interventional prediction, expected CAIRN advantage on detection/localization/adaptation speed and calibration.
8. **CAIRN ablations as internal baselines:** no e-gate (pure causal factorization); no causal factorization (monolithic + e-gate on the whole model, showing localization requires structure); no $\mathcal{L}^{\text{inv}}$; point heads instead of quantile heads; oracle graph $A^\ast$ (upper bound).

**Finance instantiation baselines:**
9. **DeXposure-FM** (published checkpoints) and temporal-GNN forecasters (TGN/EvolveGCN) — none action/intervention-conditioned; comparison on shock-window forecasting where they are documented to degrade.
10. Classical contagion models (Eisenberg–Noe clearing, DebtRank) as interpretable non-learned references for the stress-testing task.

---

## 4. Datasets and Environments

### 4.1 Robotics / embodied track (primary)

| Environment | Role | Interventions available |
|---|---|---|
| **CausalWorld** (TriFinger causal benchmark) | Flagship: ground-truth causal structure of the do-interventions API enables Theory-target-3 verification and exact localization scoring | Mass, friction, size, shape, actuator gains — scriptable per-variable |
| **Meta-World+ / RLBench** | Breadth across manipulation tasks; standard MBRL comparability for DreamerV3/TD-MPC2 numbers | Scripted physics perturbations at known changepoints |
| **DeepMind Control (walker, quadruped)** | Locomotion under actuator damage = classic localized-mechanism shift | Joint damping/gain edits mid-episode |
| **DROID (real data, replay)** | Offline evaluation of calibration + PIT validity on real trajectories; no robot required | Natural nonstationarity across labs/scenes as unlabeled shifts |

State-space (proprioceptive + object poses) is the primary observation mode; a pixel-input section uses the frozen-DINOv2-features variant to show the architecture is not sim-state-bound.

### 4.2 Finance / DeFi track (Component for ICAIF/JFS or a section)

| Resource | Role |
|---|---|
| **DeFiLlama TVL API** (5,000+ protocols, daily, 2021–present, typed) | Node states of the protocol-network CAIRN (one latent block per protocol category or top-N protocol) |
| **DeXposure-FM public dataset** | Inter-protocol exposure graph → prior/initialization for $A$; also the head-to-head baseline |
| **Ethereum BigQuery public dataset** | Flow features on edges where needed |
| **Dated incident catalogs** (UST depeg 2022-05; FTX 2022-11; major bridge exploits from the 60-bridge/34-attack SoK; 181-incident catalog) | Labeled natural interventions: training regimes for $\mathcal{L}^{\text{inv}}$ and held-out shocks for detection/localization/stress-test evaluation |

The finance world model answers do-queries directly: "do(protocol X TVL → −80%)" is a graph-surgery rollout — the stress-testing deliverable.

### 4.3 Compute

All models ≤ 15M parameters; CausalWorld/Meta-World training runs are hours each on one A100; total budget ≈ 400–500 GPU-hours including seeds and ablations. No video-scale pretraining anywhere in the paper.

---

## 5. Evaluation Protocol

### 5.1 Research questions

- **RQ1 — Interventional generalization:** Trained on regimes $\mathcal{E}_{\text{train}}$, how does CAIRN's $h$-step rollout error under *held-out* interventions compare with monolithic and causal baselines? (Report per-horizon error, and error on descendants vs. non-descendants of the intervened variable — the structural signature.)
- **RQ2 — Detection & localization:** After an unannounced mechanism shift, what are (a) detection delay of the correct node's e-gate, (b) localization precision/recall (alarmed set vs. true shifted set), (c) empirical false-alarm rate over long stationary streams vs. the $\delta$ guarantee? Baselines: ensemble-variance thresholds, ACI miscoverage bursts, CUSUM on residuals, monolithic e-gate ablation.
- **RQ3 — Adaptation speed:** Post-shift samples needed to recover pre-shift planning performance: CAIRN mechanism-level refit vs. full fine-tuning of DreamerV3/TD-MPC2 vs. causal baselines without evidence gating. Hypothesis: order-of-magnitude reduction when $|S| \ll d$.
- **RQ4 — Calibration:** Horizon-resolved empirical coverage of rollout intervals (nominal 80/90%) in-distribution, under shift *before* adaptation (uncertainty-inflation honesty), and after adaptation — vs. ensembles and post-hoc conformal wrappers.
- **RQ5 — Downstream utility:** Task success / return of CVaR planning with CAIRN vs. baselines, especially in the episodes straddling shifts; finance: stress-loss estimation error at held-out incidents vs. DeXposure-FM, temporal GNNs, and Eisenberg–Noe.
- **RQ6 — Structure recovery:** SHD/F1 of learned $A, M$ vs. ground truth in CausalWorld and synthetic DBNs; sensitivity to regime diversity (how many distinct training regimes are needed).

### 5.2 Metrics

Per-horizon rollout RMSE/quantile loss; descendant vs. non-descendant error split; detection delay and ARL; localization F1; false-alarm rate vs. $\delta$; interval coverage and width per horizon; post-shift sample-to-recovery; task success/return (10 seeds, 95% CIs); SHD/edge-F1 for $A$ and target-F1 for $M$; finance: shock-window forecast error, stress-loss error, and DebtRank-correlation of do-rollout contagion estimates.

### 5.3 Ablations (beyond §3 item 8)

E-gate wealth → weights mapping ($\beta$ sweep, hard vs. soft gating); mixture library size $K$; sparsity strengths $\gamma_1$; regime count in training; latent partition granularity $d$; ONS vs. fixed-bet vs. mixture betting; uncertainty-inflation schedule; structure-repair on/off.

### 5.4 Success criteria (pre-registered)

- RQ1: ≥ 30% relative reduction in held-out interventional rollout error vs. DreamerV3/TD-MPC2 at horizon 10+, and non-descendant error statistically indistinguishable from the no-shift condition (the localization signature).
- RQ2: localization F1 ≥ 0.8 on CausalWorld shifts; empirical false alarms ≤ δ over ≥ 10⁵-step null streams; delay ≤ 50% of best-tuned CUSUM without CAIRN requiring tuning.
- RQ3: ≥ 5× fewer post-shift samples to 90% performance recovery vs. full fine-tuning, on shifts touching ≤ 20% of mechanisms.
- RQ4: coverage within ±3% of nominal at horizons ≤ 20 across conditions.
- RQ5 (finance): beat DeXposure-FM on shock-window forecasting at ≥ 3 of 5 held-out incidents; do-rollout contagion rankings rank-correlate ρ ≥ 0.6 with DebtRank on historical episodes.

### 5.5 Limitations to state up front

Latent-space (not pixel-native) primary instantiation; identifiability holds under sparse-mechanism-shift and regime-diversity assumptions that real data may violate (we quantify degradation as regimes shrink); e-gate guarantees are exact for PIT validity of the *stated* predictive distribution — approximation error of quantile heads is absorbed into what is being tested, which is the operationally correct but philosophically weaker null; finance latents are protocol aggregates, not micro-level agents.

---

## 6. Paper Plan, Risks, Timeline

**Structure:** Intro (monolithic-failure motivation, Fig. 1: same shift → global degradation in Dreamer vs. one alarmed node in CAIRN) → Related work (§1.4 expanded) → Method (§2) → Theory (4 propositions; localization guarantee as flagship) → CausalWorld + Meta-World experiments (RQ1–RQ4, RQ6) → Planning results (RQ5) → DeFi case study → Limitations.

| Risk | Mitigation |
|---|---|
| Differentiable structure learning unstable | DBN setting removes acyclicity constraint (hardest part); oracle-graph and given-partition (finance) variants de-risk the story if learned-$A$ results are noisy |
| Closest 2026 causal-WM competitor code unavailable | Reimplement its core factorization as baseline 7; our differentiators (e-gate, adaptation, calibration) don't depend on beating it at raw prediction |
| E-gate guarantee vs. learned-quantile approximation criticized | Address head-on in §5.5; add a conformalized-quantile variant of the heads restoring exact finite-sample PIT validity as an appendix option |
| "Kitchen sink" review | Ablation table maps each component to exactly one claimed capability; every component removable with a measured, reported cost |
| Concurrent work in a hot area | arXiv early; the e-gate + localized-adaptation combination and the finance instantiation are distinctive |

**Timeline (lead + 1 student):** M1–2 model + CausalWorld infrastructure + synthetic-DBN sanity suite; M3–4 robotics experiments and theory write-up; M5 planning results + DeFi study (reuse VEIN pipeline); M6 writing, RA-L submission; ICAIF workshop version of the finance section in parallel.
