"""Generate the LaTeX tables of the manuscript from the result JSONs.

Never hand-edit paper/tables/*.tex — rerun this script instead:

    python experiments/make_paper_tables.py
"""

from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "paper", "tables")

LABELS = {
    "cairn": r"CAIRN (learned graph)",
    "cairn_noinv": r"CAIRN w/o $\mathcal{L}^{\mathrm{inv}}$",
    "cairn_oracle": r"CAIRN (oracle graph)",
    "monolithic": r"Monolithic WM",
    "ensemble": r"Probabilistic ensemble",
}
DET_LABELS = {
    "egate": r"CAIRN e-gates",
    "egate_oracle_graph": r"e-gates, oracle graph",
    "cusum": r"CUSUM",
    "ensemble": r"Ensemble disagreement",
}


def load(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


def write(name, content):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w") as f:
        f.write(content)
    print(f"[saved] paper/tables/{name}")


def f3(x):
    return f"{x:.3f}" if x is not None else "--"


def rq1_table():
    r = load("rq1.json")
    rows = []
    for group, glabel in [("heldout", "held-out"), ("trained", "trained")]:
        for name in ["cairn", "cairn_noinv", "cairn_oracle", "monolithic",
                     "ensemble"]:
            rec = r["models"][name].get(group)
            if not rec:
                continue
            rm = rec["rmse_by_horizon"]
            rows.append(
                f"{glabel} & {LABELS[name]} & {f3(rm[0])} & {f3(rm[2])} & "
                f"{f3(rm[4])} & {f3(rec['desc_rmse']['3'])} & "
                f"{f3(rec['nondesc_rmse']['3'])} & "
                f"{f3(rec['nondesc_rmse_no_intervention']['3'])} \\\\")
        rows.append(r"\midrule")
    body = "\n".join(rows[:-1])
    write("rq1.tex", r"""\begin{table}[t]
\caption{Interventional generalization (RQ1). Rollout RMSE of the
model-mean trajectory against the true conditional mean under off-support
do-interventions ($\mathrm{do}(z^i{=}\pm 4)$), for do-targets never
intervened during training (held-out) and for trained do-targets, together
with the structural split at horizon $3$: error on descendants versus
non-descendants of the intervened variable, and the no-intervention
reference error of the same non-descendant set. A structurally correct
model must leave non-descendants unchanged, so its non-descendant column
must match the reference column.}\label{tab:rq1}
\footnotesize\setlength{\tabcolsep}{3pt}%
\begin{tabular}{llcccccc}
\toprule
Targets & Model & $h{=}1$ & $h{=}3$ & $h{=}5$ & desc.\ & non-desc.\ & ref.\\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def rq2_tables():
    import glob as _glob
    import numpy as np
    seeds = sorted(_glob.glob(os.path.join(RESULTS, "rq2*.json")))
    rs = [json.load(open(f)) for f in seeds]
    r = rs[0]
    n_seeds = len(rs)

    def agg(path):
        vals = []
        for rr in rs:
            v = rr
            for k in path:
                v = v[k]
            vals.append(v)
        m, sd = float(np.mean(vals)), float(np.std(vals))
        return f"{m:.3f} $\\pm$ {sd:.3f}" if n_seeds > 1 else f"{m:.3f}"

    null_rows = []
    order = ["egate", "egate_oracle_graph", "cusum", "ensemble"]
    for det in order:
        null_rows.append(
            f"{DET_LABELS[det]} & "
            f"{agg(['null', det, 'alarm_fraction'])} & "
            f"{agg(['detectors', det, 'median_delay'])} & "
            f"{agg(['detectors', det, 'localization_f1'])} \\\\")
    glob = agg(["null", "global_egate", "alarm_fraction"])
    gdelay = agg(["global_egate_median_delay"])
    null_rows.append(
        rf"Global e-gate (monolithic) & {glob}$^{{\ast}}$ & "
        rf"{gdelay} & undefined \\")
    write("rq2.tex", r"""\begin{table}[t]
\caption{Detection and localization (RQ2): mean $\pm$ standard deviation
over """ + str(n_seeds) + r""" independent benchmark seeds, each aggregating
ten unannounced mechanism-shift scenarios and three $6{,}000$-step
stationary null streams at level $\delta=0.05$. The e-gates run with no tuning; CUSUM and ensemble
thresholds receive oracle tuning on a separate null stream. The exact-null
verification (gates fed probability integral transforms from the true
environment conditional) produced an alarm fraction of
$""" + f"{r['oracle_null_alarm_fraction']:.4f}" + r"""$, consistent with
Proposition~\ref{prop:validity}. $^{\ast}$per stream rather than per gate.}
\label{tab:rq2}
\footnotesize\setlength{\tabcolsep}{3.5pt}%
\begin{tabular}{lccc}
\toprule
Detector & Null alarm fraction & Median delay & Localization $F_1$\\
\midrule
""" + "\n".join(null_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def rq3_table():
    r = load("rq3.json")
    cps = r["checkpoints"]
    rows = []
    meth_labels = {"cairn_local": "CAIRN localized refit",
                   "cairn_full_ft": "CAIRN full fine-tune",
                   "monolithic_ft": "Monolithic fine-tune"}
    for sc in r["scenarios"]:
        s_lab = ",".join(str(x) for x in sc["S"])
        for mname in ["cairn_local", "cairn_full_ft", "monolithic_ft"]:
            m = sc["methods"][mname]
            c = m["rmse_curve"]
            ca = m["rmse_curve_all_nodes"]
            rows.append(
                rf"$S{{=}}\{{{s_lab}\}}$ & {meth_labels[mname]} & shifted & "
                + " & ".join(f3(x) for x in c) + r" \\")
            rows.append(
                rf" &  & all & "
                + " & ".join(f3(x) for x in ca) + r" \\")
        rows.append(r"\midrule")
    head = " & ".join([f"$n{{=}}{c}$" for c in [0] + cps])
    write("rq3.tex", r"""\begin{table}[t]
\caption{Localized adaptation versus fine-tuning (RQ3): one-step median
RMSE after an unannounced shift of mechanism set $S$, as a function of the
number of post-shift samples $n$, evaluated on the shifted variables
(``shifted'') and on all variables (``all''). All methods receive the same
stream and the same gradient budget. Full fine-tuning suffers catastrophic
interference on the seven healthy mechanisms; the evidence-triggered
localized refit leaves them untouched.}\label{tab:rq3}
\footnotesize\setlength{\tabcolsep}{2.5pt}%
\begin{tabular}{lllcccccc}
\toprule
Shift & Method & Nodes & """ + head + r"""\\
\midrule
""" + "\n".join(rows[:-1]) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def rq4_table():
    r = load("rq4.json")
    hs = r["horizons"]
    rows = []
    for name in ["cairn", "ensemble"]:
        rec = r["nominal"][name]
        rows.append(f"in-distribution & {LABELS[name]} & "
                    + " & ".join(f3(rec['cov90'][str(h)]) for h in hs)
                    + r" \\")
    rows.append(r"\midrule")
    labels = {"pre_adapt_no_inflation": "shift, stale intervals",
              "pre_adapt_inflated": "shift, evidence-inflated",
              "ensemble": "shift, ensemble",
              "post_adapt": "shift, after adaptation"}
    for key in ["pre_adapt_no_inflation", "pre_adapt_inflated", "ensemble",
                "post_adapt"]:
        rec = r["shift"][key]
        row = f"{labels[key]} & "
        if "cov90_shifted_node" in rec:
            row += ("shifted node & "
                    + " & ".join(f3(rec['cov90_shifted_node'][str(h)])
                                 for h in hs))
        else:
            row += ("all nodes & "
                    + " & ".join(f3(rec['cov90'][str(h)]) for h in hs))
        rows.append(row + r" \\")
    write("rq4.tex", r"""\begin{table}[t]
\caption{Calibration (RQ4): empirical coverage of nominal $90\%$ rollout
intervals by horizon. In-distribution rows average over all variables;
shift rows are scored on the shifted mechanism's variable when available
(all-node averages otherwise), where interval honesty is actually at
stake.}\label{tab:rq4}
\footnotesize\setlength{\tabcolsep}{4pt}%
\begin{tabular}{llcccc}
\toprule
Condition & Scope & """ + " & ".join(f"$h{{=}}{h}$" for h in hs) + r"""\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def rq6_table():
    r = load("rq6.json")
    rows = []
    for n, sc in zip(r["regime_counts"], r["scores"]):
        rows.append(f"{n} & {sc['shd_A']} & {sc['f1_A']:.3f} & "
                    f"{sc['shd_M']} & {sc['f1_M']:.3f} \\\\")
    write("rq6.tex", r"""\begin{table}[t]
\caption{Structure recovery versus regime diversity (RQ6): structural
Hamming distance and edge $F_1$ of the learned adjacency $A$ and
action-target mask $M$ against ground truth, as the number of training
regimes grows.}\label{tab:rq6}
\begin{tabular}{ccccc}
\toprule
Regimes & SHD$(A)$ & $F_1(A)$ & SHD$(M)$ & $F_1(M)$\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


if __name__ == "__main__":
    rq1_table()
    rq2_tables()
    rq3_table()
    rq4_table()
    rq6_table()
