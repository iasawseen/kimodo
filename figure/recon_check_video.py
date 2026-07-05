#!/usr/bin/env python
"""Raw VGGT-Omega reconstruction verification videos (no skeleton, no robot pose).

1) recon_points.mp4 - LEFT: source figure.mp4 frame; RIGHT: that frame's depth map unprojected
   to a 3D point cloud and re-rendered from the ORIGINAL camera angle (identity view, the
   pose_enc focal). A legit reconstruction reproduces the source; depth errors show as holes,
   smears, and flicker. Low-confidence points are dropped (black holes = model unsure).
2) depth_maps.mp4 - LEFT: source frame; RIGHT: the raw depth map colorized (turbo, fixed
   global range across the whole video, so brightness is comparable frame to frame).

Usage: DEPTH_WORK=<pose_full dir> python figure/recon_check_video.py
"""
import os

import imageio.v2 as imageio
import numpy as np
from matplotlib import cm
from PIL import Image

WORK = os.environ["DEPTH_WORK"]
_PKG = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_PKG)
OUTD = os.path.join(REPO, "outputs", "figure", "pose")
FPS = 24000 / 1001
CONF_MIN = 1.2
TILT = np.radians(float(os.environ.get("TILT_DEG", "33")))  # negative pitches the view downward
SKIP_DM = os.environ.get("SKIP_DM") == "1"

# depth-map resolution (VGGT) and output pane
d0 = np.load(os.path.join(WORK, "depth", "f00000.npz"))
h, w = d0["depth"].shape
SC = 960 / w                                             # 512x288 -> 960x540
PW, PH = int(w * SC) // 2 * 2, int(h * SC) // 2 * 2

# global depth range for stable colorization
samp = []
for k in range(0, 5189, 97):
    f = os.path.join(WORK, "depth", f"f{k:05d}.npz")
    if os.path.exists(f):
        samp.append(np.load(f)["depth"].astype(np.float32).ravel()[::7])
samp = np.concatenate(samp)
D0, D1 = np.percentile(samp, [2, 98])
print(f"[check] depth range [{D0:.3f}, {D1:.3f}] (raw units)", flush=True)
TURBO = (np.asarray(cm.turbo(np.linspace(0, 1, 256)))[:, :3] * 255).astype(np.uint8)

vs, us = np.mgrid[0:h, 0:w].astype(np.float32)

wr_pc = imageio.get_writer(os.path.join(OUTD, "recon_points.mp4"), fps=FPS, quality=7,
                           macro_block_size=1)
wr_dm = imageio.get_writer(os.path.join(OUTD, "depth_maps.mp4"), fps=FPS, quality=7,
                           macro_block_size=1)
N = 5189
for n in range(N):
    dz = np.load(os.path.join(WORK, "depth", f"f{n:05d}.npz"))
    D = dz["depth"].astype(np.float32)
    C = dz["conf"].astype(np.float32)
    pe = dz["pose_enc"]
    fy = (h / 2.0) / np.tan(pe[7] / 2.0)
    fx = (w / 2.0) / np.tan(pe[8] / 2.0)
    src = np.asarray(Image.open(os.path.join(WORK, "frames", f"f{n:05d}.jpg"))
                     .resize((w, h)), dtype=np.uint8)
    left = np.asarray(Image.open(os.path.join(WORK, "frames", f"f{n:05d}.jpg"))
                      .resize((PW, PH)), dtype=np.uint8)

    # ---- point-cloud re-renders: the camera orbits a mid-scene pivot a little ABOVE and a
    # little BELOW the original viewpoint, so depth errors show as parallax distortion
    good = C > CONF_MIN
    X = (us - w / 2) / fx * D
    Y = (vs - h / 2) / fy * D
    zmid = float(np.median(D[good])) if good.any() else 1.0
    panes = [left]
    for tilt in (TILT, -TILT):                           # above, then below
        ca, sa = np.cos(tilt), np.sin(tilt)
        Yt = ca * Y - sa * (D - zmid)
        Zt = np.maximum(sa * Y + ca * (D - zmid) + zmid, 1e-4)
        u2 = (PW / 2 + X / Zt * fx * SC)[good]
        v2 = (PH / 2 + Yt / Zt * fy * SC)[good]
        zc = Zt[good]
        order = np.argsort(-zc)                          # painter's, far first
        ui = np.clip(u2[order].astype(int), 0, PW - 2)
        vi = np.clip(v2[order].astype(int), 0, PH - 2)
        pc = np.zeros((PH, PW, 3), np.uint8)
        cc = src[good][order]
        pc[vi, ui] = cc; pc[vi, ui + 1] = cc             # 2x2 splats close most gaps
        pc[vi + 1, ui] = cc; pc[vi + 1, ui + 1] = cc
        panes.append(pc)
    wr_pc.append_data(np.concatenate(panes, 0))          # source / above / below

    # ---- colorized depth map
    if SKIP_DM:
        if n % 1200 == 0:
            print(f"[check] {n}/{N}", flush=True)
        continue
    idx = np.clip((D - D0) / (D1 - D0) * 255, 0, 255).astype(np.uint8)
    dm = TURBO[idx]
    dm[~good] //= 4                                      # dim low-confidence regions
    dm = np.asarray(Image.fromarray(dm).resize((PW, PH), Image.NEAREST))
    wr_dm.append_data(np.concatenate([left, dm], 1))
    if n % 1200 == 0:
        print(f"[check] {n}/{N}", flush=True)
wr_pc.close(); wr_dm.close()
print(f"[check] WROTE {OUTD}/recon_points.mp4 and depth_maps.mp4", flush=True)
