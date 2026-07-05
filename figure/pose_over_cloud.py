#!/usr/bin/env python
"""Fitted 3D robot skeleton over the per-frame VGGT-Omega point cloud, +33 deg inclination.

TOP: source figure.mp4 frame. BOTTOM: that frame's cloud (RGB-colored, confidence-gated)
re-rendered from the original camera orbited +33 deg vertically about a mid-scene pivot, with
the rigid-fitter skeleton (fit3d.npz, green bones / yellow joints) drawn in the same view.

Usage: DEPTH_WORK=<pose_full dir> python figure/pose_over_cloud.py
Output: outputs/figure/pose/pose_over_cloud.mp4
"""
import os

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from skel_draw import draw_skeleton  # noqa: E402
from fastvid import VideoWriter, gpu_splat  # noqa: E402

WORK = os.environ["DEPTH_WORK"]
_PKG = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_PKG)
OUTP = os.environ.get("POC_OUT", os.path.join(REPO, "outputs", "figure", "pose", "pose_over_cloud.mp4"))
FPS = float(os.environ.get("MR_FPS", 24000 / 1001))
CONF_MIN = 1.2
TILT = np.radians(float(os.environ.get("TILT_DEG", "33")))

fz = np.load(os.environ.get("POSE_NPZ", os.path.join(REPO, "outputs", "figure", "pose", "fit3d.npz")))
J_w, t_arr, ok = fz["joints_w"].astype(np.float64), fz["t"], fz["ok"]
KAPPA = float(fz["kappa"]) if "kappa" in fz.files else 1.0
lz = np.load(os.environ.get("LIFT_NPZ", os.path.join(REPO, "outputs", "figure", "pose", "lift3d.npz")))
Rcw, scale, cam_pos = lz["R_cam2world"], float(lz["scale"]), lz["cam_pos"]
# world -> VGGT camera frame (inverse of the lift alignment; per-frame if the camera moves)
if "M_c2w" in lz:
    Minv = np.linalg.inv(lz["M_c2w"])
    J_cam = np.einsum("nij,nkj->nki", Minv, J_w - lz["c_c2w"][:, None])
else:
    J_cam = np.einsum("ji,nkj->nki", Rcw, (J_w - cam_pos[None, None]) / scale)

KPN = ["nose", "lsho", "rsho", "lelb", "relb", "lhip", "rhip", "lkne", "rkne",
       "lank", "rank", "lbtoe", "lheel", "rbtoe", "rheel", "rwri", "lwri", "neck"]
I = {n: i for i, n in enumerate(KPN)}
BONES = [("neck", "nose"), ("lsho", "rsho"), ("neck", "lsho"), ("neck", "rsho"),
         ("lsho", "lelb"), ("lelb", "lwri"), ("rsho", "relb"), ("relb", "rwri"),
         ("lhip", "rhip"), ("neck", "lhip"), ("neck", "rhip"),
         ("lhip", "lkne"), ("lkne", "lank"), ("rhip", "rkne"), ("rkne", "rank")]

d0 = np.load(os.path.join(WORK, "depth", "f00000.npz"))
h, w = d0["depth"].shape
SC = 960 / w
PW, PH = int(w * SC) // 2 * 2, int(h * SC) // 2 * 2
vs, us = np.mgrid[0:h, 0:w].astype(np.float32)
FW, FH = Image.open(os.path.join(WORK, "frames", "f00000.jpg")).size

# SAM's focal (processed-frame px): the fitted joints subtend SAM's ANGULAR sizes - projecting
# with VGGT's focal shrinks the drawn skeleton by f_vggt/f_sam. Same fix as collate_video.
import glob as _glob0
_mhr = sorted(_glob0.glob(os.path.join(WORK, "mhr", "f*.npz")))
_f = [float(z["focal"]) for f in _mhr[:: max(1, len(_mhr) // 25)]
      for z in [np.load(f)] if "focal" in z.files]
F_SAM = float(np.median(_f)) if _f else 1468.6           # older caches: fixed SAM heuristic


def body_fix(Jc, fx, fy):
    """Rescale about mid-hip: undo kappa (world-only calibration) and the VGGT/SAM focal ratio;
    the pelvis stays on its kp2d ray so its pixel is untouched."""
    pelv = 0.5 * (Jc[I["lhip"]] + Jc[I["rhip"]])
    s = F_SAM * (w / FW) / np.sqrt(fx * fy) / KAPPA
    return pelv + (Jc - pelv) * s


def tilt_project(X, Y, Z, zmid, fx, fy):
    ca, sa = np.cos(TILT), np.sin(TILT)
    Yt = ca * Y - sa * (Z - zmid)
    Zt = np.maximum(sa * Y + ca * (Z - zmid) + zmid, 1e-4)
    return PW / 2 + X / Zt * fx * SC, PH / 2 + Yt / Zt * fy * SC, Zt


wr = VideoWriter(OUTP, PW, PH * 2, FPS)
import glob as _g
N = len(_g.glob(os.path.join(WORK, 'depth', 'f*.npz')))
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
    good = C > CONF_MIN
    X = (us - w / 2) / fx * D
    Y = (vs - h / 2) / fy * D
    zmid = float(np.median(D[good])) if good.any() else 1.0
    u2, v2, zc = tilt_project(X[good], Y[good], D[good], zmid, fx, fy)
    pc = gpu_splat(np.stack([u2, v2], 1), zc, src[good], PH, PW)
    if ok[n]:
        Jc = body_fix(J_cam[n], fx, fy)
        ju, jv, _ = tilt_project(Jc[:, 0], Jc[:, 1], Jc[:, 2], zmid, fx, fy)
        pc = draw_skeleton(pc, np.stack([ju, jv], 1))
    wr.write(np.concatenate([left, pc], 0))          # vertical stack: source / cloud
    if n % 1200 == 0:
        print(f"[poc] {n}/{N}", flush=True)
wr.close()
print(f"[poc] WROTE {OUTP}")
