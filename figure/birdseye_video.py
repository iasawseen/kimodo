#!/usr/bin/env python
"""Bird's-eye video of the lifted Figure skeleton over the VGGT-Omega reconstruction.

TOP: source figure.mp4 frame. BOTTOM: top-down view - reconstruction cloud (gray, static),
pelvis trail (orange, grows over time), current skeleton (green bones / yellow joints),
camera marker. Same timeline as the source (23.976 fps).

Usage: DEPTH_WORK=<pose_full dir> python figure/birdseye_video.py
Output: outputs/figure/pose/birdseye.mp4
"""
import os

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from skel_draw import draw_skeleton, draw_trail  # noqa: E402
from fastvid import VideoWriter  # noqa: E402

WORK = os.environ["DEPTH_WORK"]
_PKG = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_PKG)
OUTP = os.environ.get("BEV_OUT", os.path.join(REPO, "outputs", "figure", "pose", "birdseye.mp4"))
FPS = float(os.environ.get("MR_FPS", 24000 / 1001))

z = np.load(os.environ.get("POSE_NPZ", os.path.join(WORK, "lift3d.npz")))
J, t, ok, P = z["joints_w"], z["t"], z["ok"], z["scene"]
cam = z["cam_pos"]

KPN = ["nose", "lsho", "rsho", "lelb", "relb", "lhip", "rhip", "lkne", "rkne",
       "lank", "rank", "lbtoe", "lheel", "rbtoe", "rheel", "rwri", "lwri", "neck"]
I = {n: i for i, n in enumerate(KPN)}
BONES = [("neck", "nose"), ("lsho", "rsho"), ("neck", "lsho"), ("neck", "rsho"),
         ("lsho", "lelb"), ("lelb", "lwri"), ("rsho", "relb"), ("relb", "rwri"),
         ("lhip", "rhip"), ("neck", "lhip"), ("neck", "rhip"),
         ("lhip", "lkne"), ("lkne", "lank"), ("rhip", "rkne"), ("rkne", "rank")]

# ---- BEV mapping (equal scale, whole scene + camera in view)
H = W = 720
allx = np.concatenate([P[:, 0], [cam[0]]]); ally = np.concatenate([P[:, 1], [cam[1]]])
x0, x1 = np.percentile(allx, [0.5, 99.5]); y0, y1 = np.percentile(ally, [0.5, 99.5])
pad = 0.4
scale = min((W - 40) / (x1 - x0 + 2 * pad), (H - 40) / (y1 - y0 + 2 * pad))
cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)


def px(p):
    return (W / 2 + (p[0] - cx) * scale, H / 2 - (p[1] - cy) * scale)


# static background: scene cloud + camera
bg = Image.new("RGB", (W, H), (16, 16, 20))
d = ImageDraw.Draw(bg)
for p in P[::2]:
    u, v = px(p)
    if 0 <= u < W and 0 <= v < H:
        g = 70 + int(min(60, max(0.0, float(p[2])) * 40))
        d.point((u, v), fill=(g, g, g))
u, v = px(cam)
d.rectangle([u - 5, v - 5, u + 5, v + 5], fill=(230, 230, 230))
d.text((u + 8, v - 6), "camera", fill=(230, 230, 230))
BG = np.asarray(bg)

mid = 0.5 * (J[:, I["lhip"], :2] + J[:, I["rhip"], :2])
N = len(t)
from PIL import Image as _I
_s0 = _I.open(os.path.join(WORK, "frames", "f00000.jpg")).size
_srch = 2 * ((W * _s0[1] // _s0[0]) // 2)
wr = VideoWriter(OUTP, W, H + _srch, FPS)
for n in range(N):
    bev = BG.copy()
    tr = [px(mid[k]) for k in range(0, n + 1, 3) if ok[k]]
    bev = draw_trail(bev, tr)
    if ok[n]:
        pts = np.array([px(J[n, i, :2]) for i in range(len(KPN))])
        bev = draw_skeleton(bev, pts)
    frame = Image.fromarray(bev)
    ImageDraw.Draw(frame).text((12, 10), f"t = {t[n]:6.2f} s", fill=(250, 250, 250))
    bev = np.asarray(frame)
    # top pane: source frame, same width as the BEV; vertical stack
    src = Image.open(os.path.join(WORK, "frames", f"f{n:05d}.jpg"))
    src = src.resize((W, 2 * ((W * src.size[1] // src.size[0]) // 2)))
    wr.write(np.concatenate([np.asarray(src), bev], 0))
    if n % 1200 == 0:
        print(f"[bev] {n}/{N}", flush=True)
wr.close()
print(f"[bev] WROTE {OUTP}")
