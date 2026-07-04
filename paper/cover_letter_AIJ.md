# Cover letter — Artificial Intelligence (Elsevier)

Dear Editors,

Please consider our manuscript, "CAIRN: A causally-factored,
self-certifying world model for interventional generalization," for
publication in Artificial Intelligence.

World models — learned simulators that agents plan inside — have become
central to model-based decision making, yet every leading architecture
shares a structural commitment: a monolithic transition map in which all
latent variables depend on all others. Our manuscript argues, and then
demonstrates, that this commitment is what makes deployed world models
fail globally and silently when a single mechanism of the environment
changes. We propose CAIRN, an architecture and training algorithm in
which dynamics factor over a learned sparse causal graph, actions enter
as Pearl-style interventions implemented as graph surgery, and — the
central novelty — every mechanism carries an internal betting
martingale, the e-gate, giving the model anytime-valid statistical
evidence about which parts of itself are currently wrong. We prove
time-uniform false-alarm control under a composite null that survives
learned-model approximation error, bound detection delay, and give
conditions under which alarms localize to the truly shifted mechanisms.
Empirically, across three independent seeds of a ground-truth causal
benchmark, the gates raise zero false alarms while localizing
unannounced mechanism shifts within a median of seventeen steps at an
F1 of 0.90, where oracle-tuned CUSUM and ensemble disagreement fail in
one direction or the other; an ablation shows that the same statistical
machinery without the causal factorization cannot detect localized
shifts at all.

We believe the paper suits Artificial Intelligence's tradition of work
that combines formal guarantees with algorithmic contributions of broad
relevance — here spanning model-based reinforcement learning, causal
representation learning, and game-theoretic statistics. The complete
implementation, benchmark, and scripts that regenerate every table are
publicly available, and the manuscript reports negative findings (no
long-horizon accuracy advantage; preliminary planning gains) alongside
the positive ones.

This manuscript is not under consideration elsewhere, and all results
are original. We have no competing interests to declare.

Corresponding author:
Quang-Vinh Dang
British University Vietnam, Hung Yen, Vietnam
vinh.dq4@buv.edu.vn

Thank you for your consideration.

Sincerely,
Quang-Vinh Dang
