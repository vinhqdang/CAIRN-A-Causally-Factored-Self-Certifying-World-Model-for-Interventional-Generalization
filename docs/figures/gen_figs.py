import sys, os, numpy as np, torch, copy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/home/user/CAIRN-A-Causally-Factored-Self-Certifying-World-Model-for-Interventional-Generalization")
from cairn.envs.synthetic_dbn import SyntheticDBN, Regime
from cairn.model import CairnWorldModel
from cairn.train import TrainConfig, train_cairn, episodes_to_tensors
from cairn.adapt import OnlineAdapter
from cairn.planner import CEMPlanner

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
INK="#1A2332"; ACCENT="#C6363C"; TEAL="#2C7A6E"; SLATE="#5B6B7C"; GRID="#E3E1D8"
plt.rcParams.update({"figure.dpi":110, "font.size":10, "axes.edgecolor":SLATE,
  "axes.labelcolor":INK, "text.color":INK, "xtick.color":SLATE, "ytick.color":SLATE,
  "axes.grid":True, "grid.color":GRID, "grid.linewidth":0.6, "font.family":"DejaVu Sans"})
t = lambda x: torch.tensor(x, dtype=torch.float32)
NOM = Regime()

env = SyntheticDBN(d=6, m=2, extra_parents=1, sigma=0.2, seed=3)
print("A_true:\n", env.A_true, "\nM_true:\n", env.M_true, flush=True)
data = env.generate_dataset([NOM], 80, 100, p_do=0.05, seed=11)
model = CairnWorldModel(d=6, m=2, hidden=32, gate_eps=0.1)
with torch.no_grad():
    model.structure.logits_A.copy_(t(env.A_true)*12.-6.)
    model.structure.logits_M.copy_(t(env.M_true)*12.-6.)
for lr, st in [(2e-3, 2000), (2e-4, 800)]:
    train_cairn(model, data, TrainConfig(steps=st, struct_delay=10**9, gamma_inv=0, gamma_int=0,
                lr_mech=lr, seed=0, log_every=10**9), verbose=False)
cal = episodes_to_tensors(env.generate_dataset([NOM], 10, 100, p_do=0.0, seed=77))
gen = torch.Generator().manual_seed(7)
model.calibrate_pits(cal["z"], cal["a"], cal["z_next"], generator=gen)
print("trained", flush=True)

# ---------------- fig 1: imagination vs reality ----------------
rng = np.random.default_rng(42)
z0 = rng.normal(0, 0.5, 6)
H = 15
acts = np.zeros((H, 2)); cur = rng.normal(0,1,2)
for k in range(H):
    cur = 0.8*cur + 0.2*rng.normal(0,1,2); acts[k]=cur
true_trajs = []
for _ in range(60):
    z = z0.copy(); tr=[]
    for k in range(H):
        z = env.step(z, acts[k], NOM, rng=rng); tr.append(z.copy())
    true_trajs.append(tr)
true_trajs = np.array(true_trajs)          # (60,H,6)
samples = model.rollout(t(z0).unsqueeze(0), t(acts).unsqueeze(1), n_samples=60,
                        generator=torch.Generator().manual_seed(1))[:, :, 0, :].numpy()
node = 1
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2), sharey=True)
ts = np.arange(1, H+1)
for tr in true_trajs[:40]: axes[0].plot(ts, tr[:, node], color=SLATE, alpha=0.18, lw=0.8)
axes[0].plot(ts, true_trajs[:, :, node].mean(0), color=INK, lw=2, label="mean future")
axes[0].set_title("Reality: the environment's possible futures", fontsize=10)
axes[0].set_xlabel("steps ahead"); axes[0].set_ylabel(f"state variable $z^{{{node}}}$")
for tr in samples[:40]: axes[1].plot(ts, tr[:, node], color=TEAL, alpha=0.18, lw=0.8)
axes[1].plot(ts, samples[:, :, node].mean(0), color=TEAL, lw=2, label="imagined mean")
axes[1].plot(ts, true_trajs[:, :, node].mean(0), color=INK, lw=1.4, ls="--", label="true mean")
axes[1].set_title("Imagination: CAIRN's quantile-propagated rollouts", fontsize=10)
axes[1].set_xlabel("steps ahead"); axes[1].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_imagination.png", bbox_inches="tight"); plt.close(fig)
print("fig1 done", flush=True)

# ---------------- fig 2: planning inside the model ----------------
goal = np.array([0.8, -0.5, 0.6, 0.0, -0.6, 0.4])
goal_t = t(goal)
def reward_fn(zs): return -((zs - goal_t)**2).sum(dim=-1)
planner = CEMPlanner(model, reward_fn, horizon=6, population=48, elites=6,
                     iters=3, n_rollout_samples=10)
rng_p = np.random.default_rng(5)
z = rng_p.normal(0, 0.5, 6); traj=[z.copy()]
genp = torch.Generator().manual_seed(5)
for k in range(18):
    a = planner.plan(t(z), generator=genp).numpy()
    z = env.step(z, a, NOM, rng=rng_p); traj.append(z.copy())
traj = np.array(traj)
zr = rng_p.normal(0, 0.5, 6); rnd=[zr.copy()]
for k in range(18):
    zr = env.step(zr, rng_p.normal(0,1,2), NOM, rng=rng_p); rnd.append(zr.copy())
rnd = np.array(rnd)
fig, ax = plt.subplots(figsize=(9.2, 3.0))
d_plan = np.linalg.norm(traj - goal, axis=1); d_rnd = np.linalg.norm(rnd - goal, axis=1)
ax.plot(d_plan, color=TEAL, lw=2, marker="o", ms=3, label="CVaR-CEM planning in imagination")
ax.plot(d_rnd, color=SLATE, lw=1.6, ls="--", marker="s", ms=3, label="random actions")
ax.set_xlabel("environment step"); ax.set_ylabel("distance to goal state")
ax.legend(frameon=False, fontsize=9); ax.set_title("Acting through the world model: the planner rehearses futures internally, then acts", fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_planning.png", bbox_inches="tight"); plt.close(fig)
print("fig2 done", flush=True)

# ---------------- fig 3: causal graph ----------------
fig, ax = plt.subplots(figsize=(6.4, 4.6)); ax.axis("off"); ax.grid(False)
n = 6
pos = {i: (np.cos(2*np.pi*i/n - np.pi/2)*1.0, np.sin(2*np.pi*i/n - np.pi/2)*1.0) for i in range(n)}
apos = {0: (-2.1, 0.65), 1: (-2.1, -0.65)}
for j in range(n):
    for i in range(n):
        if env.A_true[j, i] and j != i:
            x1,y1 = pos[j]; x2,y2 = pos[i]
            ax.annotate("", xy=(x2*0.85, y2*0.85), xytext=(x1*0.85, y1*0.85),
                arrowprops=dict(arrowstyle="-|>", color=SLATE, lw=1.4, shrinkA=14, shrinkB=14,
                                connectionstyle="arc3,rad=0.12"))
for k in range(2):
    for i in range(n):
        if env.M_true[k, i]:
            x1,y1 = apos[k]; x2,y2 = pos[i]
            ax.annotate("", xy=(x2*0.9, y2*0.9), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.6, ls=(0,(4,2)),
                                shrinkA=12, shrinkB=14, connectionstyle="arc3,rad=-0.08"))
for i,(x,y) in pos.items():
    ax.add_patch(plt.Circle((x,y), 0.19, facecolor="white", edgecolor=INK, lw=1.6, zorder=5))
    ax.text(x, y, f"$z^{{{i}}}$", ha="center", va="center", zorder=6, fontsize=11)
    ax.annotate("self", xy=(x*1.28, y*1.28), fontsize=6.5, color=SLATE, ha="center")
for k,(x,y) in apos.items():
    ax.add_patch(plt.Rectangle((x-0.17, y-0.15), 0.36, 0.3, facecolor="#FDF0F0",
                 edgecolor=ACCENT, lw=1.6, zorder=5))
    ax.text(x+0.01, y, f"$a_{{{k}}}$", ha="center", va="center", zorder=6, fontsize=11, color=ACCENT)
ax.set_xlim(-2.6, 1.7); ax.set_ylim(-1.7, 1.7); ax.set_aspect("equal")
ax.set_title("Ground-truth causal graph: state variables (circles), actions as targeted\ninterventions (red dashed), mechanism parents (grey arrows; self-loops omitted)", fontsize=9.5)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_graph.png", bbox_inches="tight"); plt.close(fig)
print("fig3 done", flush=True)

# ---------------- fig 4: do-intervention surgery ----------------
node_do = 1
desc = env.descendants({node_do}, 8) - {node_do}
nondesc = [j for j in range(6) if j not in desc and j != node_do]
child = sorted(desc)[0]; nd = nondesc[0] if nondesc else None
print("do node", node_do, "desc", desc, "nondesc", nondesc, flush=True)
H2 = 8
acts2 = np.zeros((H2, 2))
do_mask = torch.zeros(6); do_mask[node_do]=1.
do_val = torch.zeros(6); do_val[node_do]=2.5
plain = model.rollout(t(z0).unsqueeze(0), t(acts2).unsqueeze(1), n_samples=60,
                      generator=torch.Generator().manual_seed(2))[:, :, 0, :].numpy()
doped = model.rollout(t(z0).unsqueeze(0), t(acts2).unsqueeze(1), n_samples=60,
                      do_mask=do_mask, do_values=do_val, do_steps=slice(0,1),
                      generator=torch.Generator().manual_seed(2))[:, :, 0, :].numpy()
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))
ts2 = np.arange(1, H2+1)
for ax, nd_i, name in [(axes[0], node_do, f"intervened node $z^{{{node_do}}}$"),
                        (axes[1], child, f"descendant $z^{{{child}}}$"),
                        (axes[2], nd, f"non-descendant $z^{{{nd}}}$")]:
    lo_p, hi_p = np.percentile(plain[:,:,nd_i], [10,90], axis=0)
    lo_d, hi_d = np.percentile(doped[:,:,nd_i], [10,90], axis=0)
    ax.fill_between(ts2, lo_p, hi_p, color=SLATE, alpha=0.25, label="no intervention")
    ax.fill_between(ts2, lo_d, hi_d, color=ACCENT, alpha=0.3, label="do($z^1$=2.5) at t=1")
    ax.plot(ts2, plain[:,:,nd_i].mean(0), color=SLATE, lw=1.6)
    ax.plot(ts2, doped[:,:,nd_i].mean(0), color=ACCENT, lw=1.6)
    ax.set_title(name, fontsize=10); ax.set_xlabel("steps ahead")
axes[0].set_ylabel("value"); axes[2].legend(frameon=False, fontsize=8)
fig.suptitle("Graph surgery: the intervention replaces one mechanism — effects reach descendants only", fontsize=10.5, y=1.03)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_surgery.png", bbox_inches="tight"); plt.close(fig)
print("fig4 done", flush=True)

# ---------------- fig 5: e-gate wealth under a shift ----------------
mon_model = copy.deepcopy(model)
mon = OnlineAdapter(mon_model, adapt=False, repair=False, maintain=False)
shift_node = 2
wealth = {i: [] for i in range(6)}
rngs = np.random.default_rng(9); gen3 = torch.Generator().manual_seed(9)
z = rngs.normal(0,0.5,6); a_cur = rngs.normal(0,1,2)
alarm_time = None
for step in range(700):
    if step % 50 == 0 and step: z = rngs.normal(0,0.5,6); a_cur = rngs.normal(0,1,2)
    reg = NOM if step < 300 else Regime(shifted=(shift_node,), gain=-1.0)
    a_cur = 0.8*a_cur + 0.2*rngs.normal(0,1,2)
    zn = env.step(z, a_cur, reg, rng=rngs)
    fired = mon.step(t(z), t(a_cur), t(zn), generator=gen3)
    if alarm_time is None and shift_node in fired: alarm_time = step
    for i in range(6): wealth[i].append(mon_model.gates.node_log_wealth(i))
    z = zn
fig, ax = plt.subplots(figsize=(9.2, 3.4))
for i in range(6):
    if i == shift_node: continue
    ax.plot(wealth[i], color=SLATE, alpha=0.5, lw=1.0)
ax.plot(wealth[shift_node], color=ACCENT, lw=2, label=f"shifted mechanism $f_{{{shift_node}}}$")
ax.axhline(np.log(20), color=INK, lw=1, ls=":")
ax.text(8, np.log(20)+0.25, "anytime-valid alarm  $W \\geq 1/\\delta$  ($\\delta$=0.05)", fontsize=8.5, color=INK)
ax.axvline(300, color=INK, lw=1, ls="--")
ax.text(305, min(min(w) for w in wealth.values())+0.4, "unannounced mechanism shift", fontsize=8.5, rotation=90, va="bottom")
if alarm_time:
    ax.annotate(f"alarm fires at t={alarm_time}\n({alarm_time-300} steps after shift)",
        xy=(alarm_time, np.log(20)), xytext=(alarm_time+60, np.log(20)+2.2),
        fontsize=8.5, arrowprops=dict(arrowstyle="->", color=ACCENT), color=ACCENT)
ax.set_xlabel("deployment step"); ax.set_ylabel("log betting wealth  $\\log W_t^i$")
ax.set_title("The e-gate: each mechanism bets against its own validity — only the broken one gets rich", fontsize=10.5)
ax.legend(frameon=False, fontsize=9, loc="upper left")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_egate.png", bbox_inches="tight"); plt.close(fig)
print("fig5 done, alarm at", alarm_time, flush=True)

# ---------------- fig 6: uncertainty inflation ----------------
warm = copy.deepcopy(mon_model)   # gates already rich at node 2
z0b = rngs.normal(0,0.5,6)
actsb = np.zeros((10,2))
lo_n, hi_n = model.rollout_intervals(t(z0b).unsqueeze(0), t(actsb).unsqueeze(1),
    n_samples=64, inflate=False, generator=torch.Generator().manual_seed(4))
lo_i, hi_i = warm.rollout_intervals(t(z0b).unsqueeze(0), t(actsb).unsqueeze(1),
    n_samples=64, inflate=True, generator=torch.Generator().manual_seed(4))
truth = []
for _ in range(50):
    zb = z0b.copy(); tr=[]
    for k in range(10):
        zb = env.step(zb, actsb[k], Regime(shifted=(2,), gain=-1.0), rng=rngs); tr.append(zb.copy())
    truth.append(tr)
truth = np.array(truth)
fig, ax = plt.subplots(figsize=(9.2, 3.2))
ts3 = np.arange(1, 11)
ax.fill_between(ts3, lo_n[:,0,2], hi_n[:,0,2], color=SLATE, alpha=0.3, label="stale model, no inflation (overconfident)")
ax.fill_between(ts3, lo_i[:,0,2], hi_i[:,0,2], color=TEAL, alpha=0.25, label="evidence-inflated intervals (honest)")
for tr in truth[:30]: ax.plot(ts3, tr[:,2], color=ACCENT, alpha=0.2, lw=0.8)
ax.plot([], [], color=ACCENT, lw=1, label="actual futures under the shift")
ax.set_xlabel("steps ahead"); ax.set_ylabel("$z^2$")
ax.set_title("Evidence-driven uncertainty inflation: the alarmed mechanism widens exactly its own variable", fontsize=10.5)
ax.legend(frameon=False, fontsize=8.5)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_inflation.png", bbox_inches="tight"); plt.close(fig)
print("fig6 done", flush=True)
