import sys, os, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/home/user/CAIRN-A-Causally-Factored-Self-Certifying-World-Model-for-Interventional-Generalization")
from cairn.envs.synthetic_dbn import SyntheticDBN, Regime
from cairn.model import CairnWorldModel
from cairn.train import TrainConfig, train_cairn, episodes_to_tensors
from cairn.quantile import TAUS, pit_value

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
INK="#1A2332"; ACCENT="#C6363C"; TEAL="#2C7A6E"; SLATE="#5B6B7C"; GRID="#E3E1D8"
plt.rcParams.update({"figure.dpi":110, "font.size":9, "axes.edgecolor":SLATE,
  "axes.labelcolor":INK, "text.color":INK, "xtick.color":SLATE, "ytick.color":SLATE,
  "axes.grid":True, "grid.color":GRID, "grid.linewidth":0.5, "font.family":"DejaVu Sans"})
t = lambda x: torch.tensor(x, dtype=torch.float32)
NOM = Regime()
env = SyntheticDBN(d=6, m=2, extra_parents=1, sigma=0.2, seed=3)
model = CairnWorldModel(d=6, m=2, hidden=32, gate_eps=0.1)
with torch.no_grad():
    model.structure.logits_A.copy_(t(env.A_true)*12.-6.)
    model.structure.logits_M.copy_(t(env.M_true)*12.-6.)
model.load_state_dict(torch.load(f"{OUT}/tiny_model.pt", weights_only=True))
cal = episodes_to_tensors(env.generate_dataset([NOM], 10, 100, p_do=0.0, seed=77))
model.calibrate_pits(cal["z"], cal["a"], cal["z_next"],
                     generator=torch.Generator().manual_seed(7))

NODE = 2
rng = np.random.default_rng(21)
z = rng.normal(0, 0.5, 6)
for _ in range(20):
    z = env.step(z, rng.normal(0,1,2), NOM, rng=rng)
a = rng.normal(0, 1, 2)
zn = env.step(z, a, NOM, rng=rng)
with torch.no_grad():
    q = model.predict_quantiles(t(z).unsqueeze(0), t(a).unsqueeze(0), hard=True)[0]
A_hat, M_hat = model.structure.hard_masks()
parents = A_hat[:, NODE].numpy().astype(bool)
atargets = M_hat[:, NODE].numpy().astype(bool)
u = pit_value(q[NODE], t(zn[NODE]), generator=torch.Generator().manual_seed(3)).item()

fig, axes = plt.subplots(1, 4, figsize=(11.6, 2.9))
# P1: observation
ax = axes[0]; ax.grid(axis="y")
ax.bar(range(6), z, color=SLATE, width=0.55, alpha=0.8)
ax.set_xticks(range(6)); ax.set_xticklabels([f"$z^{i}$" for i in range(6)], fontsize=8)
ax.set_title("1 · Observation at time $t$\n(partitioned into 6 variables)", fontsize=9)
ax.axhline(0, color=SLATE, lw=0.7)
# P2: masked input to f_NODE
ax = axes[1]; ax.grid(axis="y")
xs = np.arange(8)
vals = np.concatenate([z, a])
keep = np.concatenate([parents, atargets])
cols = [TEAL if k else "#D6D4CC" for k in keep]
ax.bar(xs, vals, color=cols, width=0.55)
ax.set_xticks(xs); ax.set_xticklabels([f"$z^{i}$" for i in range(6)] + ["$a_0$","$a_1$"], fontsize=8)
ax.axhline(0, color=SLATE, lw=0.7)
kept = [f"z{i}" for i in range(6) if parents[i]] + [f"a{k}" for k in range(2) if atargets[k]]
ax.set_title(f"2 · Mechanism $f_{{{NODE}}}$ sees only its parents\n(learned masks keep {', '.join(kept)})", fontsize=9)
# P3: predicted quantiles + realized
ax = axes[2]
qs = q[NODE].numpy()
ax.plot(qs, TAUS.numpy(), color=TEAL, lw=1.8, marker="o", ms=3.5, label="predicted CDF (7 quantiles)")
ax.axvline(zn[NODE], color=ACCENT, lw=1.6, ls="--")
ax.text(zn[NODE]+0.04, 0.08, "observed\n$z^2_{t+1}$", color=ACCENT, fontsize=8)
ax.axhline(u, color=SLATE, lw=0.9, ls=":")
ax.text(qs[0], u+0.03, f"PIT $u$ = {u:.2f}", fontsize=8, color=INK)
ax.set_xlabel("value"); ax.set_ylabel("probability")
ax.set_title("3 · Distributional prediction,\nreality lands at PIT $u$", fontsize=9)
# P4: the bet
ax = axes[3]; ax.grid(False)
uu = np.linspace(0.001, 0.999, 200)
ax.plot(uu, 2*uu-1, color=TEAL, lw=1.5, label="location bet $g_1$")
ax.plot(uu, 6*(uu-.5)**2-.5, color=SLATE, lw=1.5, label="dispersion bet $g_2$")
ax.axvline(u, color=ACCENT, lw=1.6, ls="--")
ax.scatter([u,u], [2*u-1, 6*(u-.5)**2-.5], color=ACCENT, zorder=5, s=22)
ax.axhline(0, color=GRID, lw=0.8)
ax.set_xlabel("PIT $u$"); ax.set_ylabel("payoff $g(u)$")
ax.set_title("4 · Gate bets on $u$:\n$W \\leftarrow W\\,(1+\\lambda\\, g(u))$", fontsize=9)
ax.legend(frameon=False, fontsize=7.5, loc="upper center")
fig.suptitle("One transition through CAIRN: observe → mask by the causal graph → predict a distribution → score reality → bet",
             fontsize=10.5, y=1.06)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_pipeline.png", bbox_inches="tight")
print("pipeline fig done; u =", round(u,3), "| parents:", kept, flush=True)
