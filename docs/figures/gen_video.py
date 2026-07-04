import sys, os, numpy as np, torch, copy
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
sys.path.insert(0, "/home/user/CAIRN-A-Causally-Factored-Self-Certifying-World-Model-for-Interventional-Generalization")
from cairn.envs.synthetic_dbn import SyntheticDBN, Regime
from cairn.model import CairnWorldModel
from cairn.train import TrainConfig, train_cairn, episodes_to_tensors
from cairn.adapt import OnlineAdapter

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
INK="#1A2332"; ACCENT="#C6363C"; TEAL="#2C7A6E"; SLATE="#5B6B7C"; GRID="#E3E1D8"
plt.rcParams.update({"figure.dpi":80, "font.size":9, "axes.edgecolor":SLATE,
  "axes.labelcolor":INK, "text.color":INK, "xtick.color":SLATE, "ytick.color":SLATE,
  "axes.grid":True, "grid.color":GRID, "grid.linewidth":0.5, "font.family":"DejaVu Sans"})
t = lambda x: torch.tensor(x, dtype=torch.float32)
NOM = Regime()

CKPT = os.path.join(OUT, "tiny_model.pt")
env = SyntheticDBN(d=6, m=2, extra_parents=1, sigma=0.2, seed=3)
model = CairnWorldModel(d=6, m=2, hidden=32, gate_eps=0.1)
with torch.no_grad():
    model.structure.logits_A.copy_(t(env.A_true)*12.-6.)
    model.structure.logits_M.copy_(t(env.M_true)*12.-6.)
if os.path.exists(CKPT):
    model.load_state_dict(torch.load(CKPT, weights_only=True))
else:
    data = env.generate_dataset([NOM], 80, 100, p_do=0.05, seed=11)
    for lr, st in [(2e-3, 2000), (2e-4, 800)]:
        train_cairn(model, data, TrainConfig(steps=st, struct_delay=10**9, gamma_inv=0,
                    gamma_int=0, lr_mech=lr, seed=0, log_every=10**9), verbose=False)
    torch.save(model.state_dict(), CKPT)
cal = episodes_to_tensors(env.generate_dataset([NOM], 10, 100, p_do=0.0, seed=77))
model.calibrate_pits(cal["z"], cal["a"], cal["z_next"],
                     generator=torch.Generator().manual_seed(7))
print("model ready", flush=True)

# ---------- simulate the whole deployment episode, recording everything ----
SHIFT_AT, TOTAL, SHIFT_NODE = 300, 600, 2
adapter = OnlineAdapter(model, buffer_size=96, refit_epochs=250, repair=False)
rng = np.random.default_rng(9); gen = torch.Generator().manual_seed(9)
z = rng.normal(0,0.5,6); a_cur = rng.normal(0,1,2)
Zs, As, Ws, alarms, refits = [], [], [], [], []
pred_med, pred_lo, pred_hi = [], [], []
for step in range(TOTAL):
    if step % 50 == 0 and step: z = rng.normal(0,0.5,6); a_cur = rng.normal(0,1,2)
    reg = NOM if step < SHIFT_AT else Regime(shifted=(SHIFT_NODE,), gain=-1.0)
    a_cur = 0.8*a_cur + 0.2*rng.normal(0,1,2)
    with torch.no_grad():
        q = model.predict_quantiles(t(z).unsqueeze(0), t(a_cur).unsqueeze(0),
                                    hard=True, use_mixture=True)[0]
    pred_med.append(q[:,3].numpy()); pred_lo.append(q[:,1].numpy()); pred_hi.append(q[:,5].numpy())
    zn = env.step(z, a_cur, reg, rng=rng)
    n_lib_before = len(model.libraries[SHIFT_NODE])
    fired = adapter.step(t(z), t(a_cur), t(zn), generator=gen)
    if fired: alarms += [(step, i) for i in fired]
    if len(model.libraries[SHIFT_NODE]) > n_lib_before: refits.append(step)
    Zs.append(zn.copy()); As.append(a_cur.copy())
    Ws.append([model.gates.node_log_wealth(i) for i in range(6)])
    z = zn
Zs, Ws = np.array(Zs), np.array(Ws)
pred_med, pred_lo, pred_hi = map(np.array, (pred_med, pred_lo, pred_hi))
print("episode simulated; alarms:", alarms, "refits:", refits, flush=True)

# ---------------- animation ----------------
STRIDE, WIN = 3, 120
frames = list(range(WIN, TOTAL, STRIDE))
fig = plt.figure(figsize=(8.6, 6.4))
gs = fig.add_gridspec(3, 1, height_ratios=[1.1, 1.15, 1.0], hspace=0.42)
ax0 = fig.add_subplot(gs[0]); ax1 = fig.add_subplot(gs[1]); ax2 = fig.add_subplot(gs[2])

def draw(step):
    for ax in (ax0, ax1, ax2): ax.clear()
    sl = slice(step-WIN, step)
    xs = np.arange(step-WIN, step)
    shifted_now = step >= SHIFT_AT
    alarmed_now = any(s <= step and i == SHIFT_NODE for s, i in alarms)
    refit_done = any(r <= step for r in refits)
    # -- world panel: the 6 variables as physical dials --
    ax0.grid(False)
    vals = Zs[step-1]
    colors = [ACCENT if (i==SHIFT_NODE and shifted_now) else TEAL for i in range(6)]
    ax0.bar(range(6), vals, color=colors, width=0.55, alpha=0.85)
    ax0.axhline(0, color=SLATE, lw=0.8)
    ax0.set_ylim(-3, 3); ax0.set_xticks(range(6))
    ax0.set_xticklabels([f"$z^{i}$" for i in range(6)])
    status = "physics of $z^2$ silently flipped" if shifted_now else "nominal physics"
    ax0.set_title(f"THE WORLD  —  step {step}   ({status})",
                  fontsize=9.5, color=ACCENT if shifted_now else SLATE, loc="left")
    if alarmed_now and not refit_done:
        ax0.text(SHIFT_NODE, 2.5, "ALARM", ha="center", color=ACCENT, fontsize=10, fontweight="bold")
    if refit_done:
        ax0.text(SHIFT_NODE, 2.5, "refitted", ha="center", color=TEAL, fontsize=9, fontweight="bold")
    # -- observations panel: raw signal + model 80% band for the shifted node --
    ax1.fill_between(xs, pred_lo[sl, SHIFT_NODE], pred_hi[sl, SHIFT_NODE],
                     color=TEAL, alpha=0.22, label="model 80% band")
    ax1.plot(xs, pred_med[sl, SHIFT_NODE], color=TEAL, lw=1.1, label="model median")
    ax1.plot(xs, Zs[sl, SHIFT_NODE], color=INK, lw=1.0, label="observed $z^2$")
    if step > SHIFT_AT: ax1.axvline(SHIFT_AT, color=ACCENT, ls="--", lw=1)
    for r in refits:
        if step > r: ax1.axvline(r, color=TEAL, ls=":", lw=1.2)
    ax1.set_ylim(-3.2, 3.2)
    ax1.set_title("WHAT THE MODEL SEES  —  observation stream vs its own predictive band ($z^2$)",
                  fontsize=9.5, loc="left", color=SLATE)
    ax1.legend(frameon=False, fontsize=7.5, loc="upper left", ncols=3)
    # -- wealth panel --
    for i in range(6):
        if i == SHIFT_NODE: continue
        ax2.plot(xs, Ws[sl, i], color=SLATE, alpha=0.45, lw=0.9)
    ax2.plot(xs, Ws[sl, SHIFT_NODE], color=ACCENT, lw=1.8, label="gate of $f_2$")
    ax2.axhline(np.log(20), color=INK, lw=0.9, ls=":")
    ax2.text(xs[2], np.log(20)+0.4, "alarm threshold $1/\\delta$", fontsize=7.5)
    if step > SHIFT_AT: ax2.axvline(SHIFT_AT, color=ACCENT, ls="--", lw=1)
    for s_, i_ in alarms:
        if i_ == SHIFT_NODE and step > s_:
            ax2.annotate("alarm", xy=(s_, np.log(20)), xytext=(s_-40, np.log(20)+4),
                         fontsize=8, color=ACCENT, arrowprops=dict(arrowstyle="->", color=ACCENT))
    for r in refits:
        if step > r:
            ax2.annotate("spawn + few-shot refit\n(only mechanism 2)", xy=(r, 0.5),
                         xytext=(r+8, 6.5), fontsize=7.5, color=TEAL,
                         arrowprops=dict(arrowstyle="->", color=TEAL))
    ax2.set_ylim(-6, 14)
    ax2.set_title("THE E-GATES  —  log betting wealth per mechanism (its own lie detector)",
                  fontsize=9.5, loc="left", color=SLATE)
    ax2.set_xlabel("deployment step")
anim = FuncAnimation(fig, draw, frames=frames, interval=90)
anim.save(f"{OUT}/cairn_live.gif", writer=PillowWriter(fps=12))
plt.close(fig)
print("gif saved:", os.path.getsize(f"{OUT}/cairn_live.gif")//1024, "KB", flush=True)
