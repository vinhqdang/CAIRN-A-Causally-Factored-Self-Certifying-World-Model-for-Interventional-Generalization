import os, glob, numpy as np, torch
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

base = f"{S}/kitti/2011_09_26/2011_09_26_drive_0002_sync"
imgs = sorted(glob.glob(f"{base}/image_02/data/*.png"))
oxts = sorted(glob.glob(f"{base}/oxts/data/*.txt"))
frames = np.stack([iio.imread(p)[::2, ::2] for p in imgs])   # (77, 187, 621, 3)
ox = np.array([[float(x) for x in open(p).read().split()] for p in oxts])
vf, yaw_rate, ax_ = ox[:, 8], ox[:, 19], ox[:, 11]
n = len(frames)
print("kitti frames:", frames.shape, "oxts:", ox.shape, flush=True)

# ---- DINOv2 frozen features (the pixel-variant encoder from algorithm.md) ----
feat_label, feats = None, None
try:
    import timm
    dinov2 = timm.create_model("vit_small_patch14_dinov2.lvd142m",
                               pretrained=True, num_classes=0,
                               img_size=(154, 308))
    dinov2.eval()
    import torchvision.transforms as T
    tr = T.Compose([T.ToTensor(), T.Resize((154, 308)),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    with torch.no_grad():
        fs = []
        for i in range(n):
            x = tr(frames[i]).unsqueeze(0)
            fs.append(dinov2(x).squeeze(0).numpy())
    feats = np.stack(fs)
    feat_label = "frozen DINOv2 ViT-S/14 features (384-dim per frame)"
    print("dinov2 features:", feats.shape, flush=True)
except Exception as e:
    print("dinov2 unavailable:", str(e)[:200], flush=True)

# PCA of features
if feats is not None:
    F = feats - feats.mean(0)
    U, Sv, Vt = np.linalg.svd(F, full_matrices=False)
    pcs = F @ Vt[:3].T
    var = (Sv**2 / (Sv**2).sum())[:3]
    print("PCA var:", var.round(3), flush=True)

# ---- strip figure ----
keys = [0, n//4, n//2, 3*n//4, n-1]
rows = 3 if feats is not None else 2
fig = plt.figure(figsize=(11, 2.6 + 1.9*rows))
gs = fig.add_gridspec(rows, len(keys), height_ratios=[1.4] + [1.0]*(rows-1),
                      hspace=0.42, wspace=0.04)
for k, fi in enumerate(keys):
    ax = fig.add_subplot(gs[0, k]); ax.axis("off"); ax.grid(False)
    ax.imshow(frames[fi]); ax.set_title(f"t = {fi/10:.1f}s", fontsize=9, color=SLATE)
axst = fig.add_subplot(gs[1, :])
tt = np.arange(n)/10
axst.plot(tt, vf, color=INK, lw=1.6, label="forward speed (m/s)")
axst.plot(tt, ax_, color=TEAL, lw=1.2, label="forward accel (m/s²)")
axst.plot(tt, yaw_rate*10, color=ACCENT, lw=1.2, label="yaw rate ×10 (rad/s)")
axst.set_ylabel("vehicle state"); axst.legend(frameon=False, fontsize=8, ncols=3)
if feats is not None:
    axf = fig.add_subplot(gs[2, :])
    for j in range(3):
        axf.plot(tt, pcs[:, j], lw=1.3, label=f"PC{j+1} ({var[j]*100:.0f}% var)")
    axf.set_ylabel("frozen-encoder\nlatents"); axf.set_xlabel("time (s)")
    axf.legend(frameon=False, fontsize=8, ncols=3)
else:
    axst.set_xlabel("time (s)")
fig.suptitle("KITTI raw (real street video, public) — drive 2011_09_26_0002\n"
             "Camera frames for humans; the vehicle's GPS/IMU state stream and the frozen-encoder latent stream are what a world model consumes",
             fontsize=10.5, y=0.99)
fig.savefig(f"{OUT}/kitti_strip.png", bbox_inches="tight")
print("kitti strip saved", flush=True)

# ---- GIF with speed overlay ----
fig, ax = plt.subplots(figsize=(6.4, 2.5)); ax.axis("off"); ax.grid(False)
im = ax.imshow(frames[0]); title = ax.set_title("", fontsize=9, loc="left")
def draw(i):
    im.set_data(frames[i])
    title.set_text(f"KITTI drive 0002  |  t={i/10:4.1f}s  |  speed {vf[i]*3.6:4.1f} km/h")
    title.set_color(INK)
anim = FuncAnimation(fig, draw, frames=n, interval=100)
anim.save(f"{OUT}/kitti_drive.gif", writer=PillowWriter(fps=10))
plt.close(fig)
print("kitti gif:", os.path.getsize(f"{OUT}/kitti_drive.gif")//1024, "KB", flush=True)

# ---- UCF101 person clips strip ----
clips = [("v_BabyCrawling_g19_c02.avi", "BabyCrawling"),
         ("v_BasketballDunk_g14_c06.avi", "BasketballDunk")]
fig, axes = plt.subplots(2, 5, figsize=(11, 3.6))
for r, (fn, name) in enumerate(clips):
    fr = np.stack(list(iio.imiter(f"{S}/ucf/{fn}")))
    ks = np.linspace(0, len(fr)-1, 5).astype(int)
    for c, fi in enumerate(ks):
        ax = axes[r, c]; ax.axis("off"); ax.grid(False)
        ax.imshow(fr[fi])
        if c == 0:
            ax.set_ylabel(name)
        ax.set_title(f"frame {fi}", fontsize=8, color=SLATE)
    axes[r, 0].text(-0.12, 0.5, name, transform=axes[r, 0].transAxes,
                    rotation=90, va="center", fontsize=9, color=INK)
fig.suptitle("UCF101 (real human-action video, public subset) — the kind of footage frontier video world models (Genie, Cosmos) train on",
             fontsize=10.5)
fig.tight_layout()
fig.savefig(f"{OUT}/ucf_strip.png", bbox_inches="tight")
print("ucf strip saved", flush=True)
