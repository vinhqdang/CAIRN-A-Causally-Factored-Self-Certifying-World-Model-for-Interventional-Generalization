import os, numpy as np
os.environ["MUJOCO_GL"] = "osmesa"
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from dm_control import suite

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
INK="#1A2332"; ACCENT="#C6363C"; TEAL="#2C7A6E"; SLATE="#5B6B7C"; GRID="#E3E1D8"
plt.rcParams.update({"figure.dpi":100, "font.size":9, "font.family":"DejaVu Sans",
  "text.color":INK, "axes.labelcolor":INK, "xtick.color":SLATE, "ytick.color":SLATE,
  "axes.edgecolor":SLATE, "axes.grid":True, "grid.color":GRID, "grid.linewidth":0.5})

env = suite.load("walker", "walk", task_kwargs={"random": 4})
spec = env.action_spec()
rng = np.random.default_rng(4)
T, DAMAGE_AT = 300, 150
act_names = [env.physics.model.id2name(i, "actuator") for i in range(env.physics.model.nu)]
knee_idx = act_names.index("right_knee")
print("actuators:", act_names, flush=True)

ts = env.reset()
frames, qpos_hist, act_hist = [], [], []
a = np.zeros(spec.shape)
for t in range(T):
    a = 0.85*a + 0.3*rng.uniform(spec.minimum, spec.maximum)
    a = np.clip(a, spec.minimum, spec.maximum)
    if t == DAMAGE_AT:
        env.physics.model.actuator_gear[knee_idx] *= 0.05   # actuator damage
        env.physics.model.dof_damping[:] *= 1.0
    env.step(a)
    if t % 3 == 0:
        frames.append(env.physics.render(height=240, width=320, camera_id=0))
    qpos_hist.append(env.physics.data.qpos.copy())
    act_hist.append(a.copy())
qpos_hist = np.array(qpos_hist)
print("frames:", len(frames), "qpos dim:", qpos_hist.shape, flush=True)

# joint index for right knee in qpos: walker qpos: root(3) + joints
jnames = [env.physics.model.id2name(i, "joint") for i in range(env.physics.model.njnt)]
print("joints:", jnames, flush=True)
rk = jnames.index("right_knee") + 2   # rootz,rootx,rooty are first 3 qpos... offset check
# walker: qpos = [rootz, rootx, rooty, right_hip, right_knee, right_ankle, left_hip, left_knee, left_ankle]
rk_q = 4

# ---- GIF: frames with status banner ----
fig, ax = plt.subplots(figsize=(4.0, 3.3))
ax.axis("off"); ax.grid(False)
im = ax.imshow(frames[0])
title = ax.set_title("", fontsize=10, loc="left")
def draw(i):
    im.set_data(frames[i])
    step = i*3
    if step < DAMAGE_AT:
        title.set_text(f"DeepMind Control 'walker' — step {step}  |  nominal")
        title.set_color(SLATE)
    else:
        title.set_text(f"step {step}  |  right-knee actuator at 5% torque")
        title.set_color(ACCENT)
anim = FuncAnimation(fig, draw, frames=len(frames), interval=80)
anim.save(f"{OUT}/walker_damage.gif", writer=PillowWriter(fps=12))
plt.close(fig)
print("gif:", os.path.getsize(f"{OUT}/walker_damage.gif")//1024, "KB", flush=True)

# ---- keyframe strip + proprioceptive traces ----
keys = [0, 25, 50, 75, 99]   # frame indices (x3 steps)
fig = plt.figure(figsize=(10.5, 5.4))
gs = fig.add_gridspec(2, len(keys), height_ratios=[1.15, 1.0], hspace=0.32, wspace=0.05)
for k, fi in enumerate(keys):
    ax = fig.add_subplot(gs[0, k]); ax.axis("off"); ax.grid(False)
    ax.imshow(frames[fi])
    step = fi*3
    dam = step >= DAMAGE_AT
    ax.set_title(f"t={step}" + ("  (damaged)" if dam else ""), fontsize=9,
                 color=ACCENT if dam else SLATE)
ax = fig.add_subplot(gs[1, :])
tt = np.arange(T)
ax.plot(tt, qpos_hist[:, 0], color=SLATE, lw=1.4, label="torso height (m)")
ax.plot(tt, qpos_hist[:, rk_q], color=ACCENT, lw=1.4, label="right knee angle (rad)")
ax.plot(tt, qpos_hist[:, rk_q+3], color=TEAL, lw=1.4, label="left knee angle (rad)")
ax.axvline(DAMAGE_AT, color=INK, ls="--", lw=1.2)
ax.text(DAMAGE_AT+4, ax.get_ylim()[0]+0.2, "actuator damage:\nright knee gear -> 5%",
        fontsize=8.5, color=INK)
ax.set_xlabel("simulation step"); ax.set_ylabel("proprioceptive signal")
ax.legend(frameon=False, fontsize=8.5, ncols=3, loc="upper right")
fig.suptitle("The robotics instantiation's data: pixels for humans, proprioceptive state for CAIRN —\n"
             "one mechanism (the right knee) silently loses torque mid-episode, exactly the localized shift the e-gates are built to catch",
             fontsize=10.5, y=1.00)
fig.savefig(f"{OUT}/walker_strip.png", bbox_inches="tight")
print("strip saved", flush=True)
