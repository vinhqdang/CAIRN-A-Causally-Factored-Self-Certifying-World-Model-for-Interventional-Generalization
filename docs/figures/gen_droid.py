import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import imageio.v3 as iio

S = os.path.dirname(os.path.abspath(__file__))
OUT = f"{S}/figs"
INK="#1A2332"; ACCENT="#C6363C"; TEAL="#2C7A6E"; SLATE="#5B6B7C"; GRID="#E3E1D8"
plt.rcParams.update({"figure.dpi":100, "font.size":9, "font.family":"DejaVu Sans",
  "text.color":INK, "axes.labelcolor":INK, "xtick.color":SLATE, "ytick.color":SLATE,
  "axes.edgecolor":SLATE, "axes.grid":True, "grid.color":GRID, "grid.linewidth":0.5})

df = pd.read_parquet(f"{S}/droid/data.parquet")
EP = 0
ep = df[df["episode_index"]==EP]
n = len(ep)
state = np.stack(ep["observation.state"].values)     # (n, 8)
action = np.stack(ep["action"].values)
print("state dim:", state.shape, "action dim:", action.shape, flush=True)

# read the first n frames of both cameras (episodes are concatenated in order)
ext = np.stack([f for i, f in zip(range(n), iio.imiter(f"{S}/droid/exterior_image_1_left.mp4"))])
wrist = np.stack([f for i, f in zip(range(n), iio.imiter(f"{S}/droid/wrist_image_left.mp4"))])
print("frames:", ext.shape, wrist.shape, flush=True)

TASK = "Put the marker in the pot"
# ---- keyframe strip + real state traces ----
keys = [0, n//4, n//2, 3*n//4, n-1]
fig = plt.figure(figsize=(11, 6.2))
gs = fig.add_gridspec(3, len(keys), height_ratios=[1.15, 1.15, 1.0], hspace=0.3, wspace=0.04)
for k, fi in enumerate(keys):
    ax = fig.add_subplot(gs[0, k]); ax.axis("off"); ax.grid(False)
    ax.imshow(ext[fi]); ax.set_title(f"t = {fi/15:.1f}s", fontsize=9, color=SLATE)
    ax2 = fig.add_subplot(gs[1, k]); ax2.axis("off"); ax2.grid(False)
    ax2.imshow(wrist[fi])
fig.text(0.13, 0.905, "exterior camera", fontsize=9, color=SLATE, rotation=0)
fig.text(0.13, 0.615, "wrist camera", fontsize=9, color=SLATE)
ax = fig.add_subplot(gs[2, :])
tt = np.arange(n)/15
for j in range(6):
    ax.plot(tt, state[:, j], lw=1.1, color=plt.cm.viridis(j/6), alpha=0.85,
            label=f"joint {j+1}" if j in (0, 5) else None)
ax.plot(tt, state[:, 6], lw=1.8, color=ACCENT, label="gripper")
ax.set_xlabel("time (s)"); ax.set_ylabel("state (rad / open frac)")
ax.legend(frameon=False, fontsize=8, ncols=3)
fig.suptitle(f'DROID (real robot data, public sample) — episode 0: "{TASK}"\n'
             "Two real camera streams for humans; the 7 proprioceptive signals below are the state stream a CAIRN mechanism would own per variable",
             fontsize=10.5, y=0.99)
fig.savefig(f"{OUT}/droid_strip.png", bbox_inches="tight")
print("strip saved", flush=True)

# ---- GIF: exterior + wrist side by side, with state ticker ----
fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9))
for ax in axes: ax.axis("off"); ax.grid(False)
im0 = axes[0].imshow(ext[0]); axes[0].set_title("exterior", fontsize=9, color=SLATE)
im1 = axes[1].imshow(wrist[0]); axes[1].set_title("wrist", fontsize=9, color=SLATE)
sup = fig.suptitle("", fontsize=10)
def draw(i):
    im0.set_data(ext[i]); im1.set_data(wrist[i])
    g = state[i, 6]
    sup.set_text(f'DROID ep. 0: "{TASK}"  |  t={i/15:4.1f}s  |  gripper {"CLOSED" if g > 0.5 else "open"}')
    sup.set_color(ACCENT if g > 0.5 else INK)
anim = FuncAnimation(fig, draw, frames=range(0, n, 2), interval=133)
anim.save(f"{OUT}/droid_episode.gif", writer=PillowWriter(fps=7))
plt.close(fig)
print("gif:", os.path.getsize(f"{OUT}/droid_episode.gif")//1024, "KB", flush=True)
