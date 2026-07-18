#!/usr/bin/env python
"""Superimpose the retargeted G1 over the source video (ghost overlay, example.png style).

Renders the G1 surface mesh driven by MR_OUT/g1_replay.csv through the reconstruction's
CALIBRATED camera chain onto the processed source frames, as a semi-transparent colored
ghost. The retarget world is the fit world uniformly scaled to G1 size (SOMA_SCALE auto);
dividing the robot's world vertices by that scale puts it back in the subject's world, so
the ghost aligns limb-for-limb with the person (feet stay on the same floor plane).

Projection = the pipeline's body_fix reprojection (collate_video/mesh_world_video):
world -> per-frame VGGT camera via inv(M_c2w) + c_c2w, then a pure rescale about the
camera-frame mid-hip (undoes kappa and the VGGT/SAM focal ratio), then pinhole with the
frame-center principal point. Splatting/encoding via humanoid_motion_recon gpurender +
fastvid (NVENC).

Usage (kimodo env):
    MR_OUT=outputs/figure/pose_vshort FRAMES=<dir with f%05d.jpg> MR_FPS=59.94 \
        [OUT=...] [ALPHA=0.6] [COLOR=235,150,45] [DENSITY=24] \
        MUJOCO_GL=egl python figure/render_overlay.py

FRAMES must be the processed-frame space the reconstruction ran on (same size/count);
re-extract with: ffmpeg -i <src.mp4> -vf scale=<W>:<H> -start_number 0 f%05d.jpg
"""
import os
import sys

import numpy as np
import mujoco
import torch
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from humanoid_motion_recon import gpurender as gr           # noqa: E402
from humanoid_motion_recon.fastvid import VideoWriter       # noqa: E402

OUTD = os.environ.get("MR_OUT", os.path.join(REPO, "outputs", "figure", "pose_vshort"))
FRAMES = os.environ["FRAMES"]
FPS = float(os.environ.get("MR_FPS", "59.94005994"))
OUTP = os.environ.get("OUT", os.path.join(OUTD, "g1_overlay.mp4"))
ALPHA = float(os.environ.get("ALPHA", "0.6"))
COLOR = tuple(int(c) for c in os.environ.get("COLOR", "235,150,45").split(","))
DENSITY = float(os.environ.get("DENSITY", "24"))
F_SAM = float(os.environ.get("F_SAM", "1468.6"))            # SAM focal, processed-frame px
G1_XML = os.path.join(REPO, "kimodo/assets/skeletons/g1skel34/xml/g1.xml")

KPN = ["nose", "lsho", "rsho", "lelb", "relb", "lhip", "rhip", "lkne", "rkne",
       "lank", "rank", "lbtoe", "lheel", "rbtoe", "rheel", "rwri", "lwri", "neck"]
K = {n: i for i, n in enumerate(KPN)}

fz = np.load(os.path.join(OUTD, "fit3d.npz"))
lz = np.load(os.path.join(OUTD, "lift3d.npz"))
Jw_fit = fz["joints_w"].astype(np.float64)
KAPPA = float(fz["kappa"]) if "kappa" in fz.files else 1.0
N = len(Jw_fit)
if "M_c2w" in lz.files:
    M_c2w, c_c2w = lz["M_c2w"].astype(np.float64), lz["c_c2w"].astype(np.float64)
else:
    M_c2w = np.repeat((float(lz["scale"]) * lz["R_cam2world"])[None], N, 0)
    c_c2w = np.repeat(lz["cam_pos"][None], N, 0)
Minv = np.linalg.inv(M_c2w)

# retarget world scale, recomputed exactly as soma_retarget SOMA_SCALE=auto
_pel = 0.5 * (Jw_fit[:, K["lhip"], 2] + Jw_fit[:, K["rhip"], 2])
_pel = _pel[np.isfinite(_pel)]
F_SC = 0.72 / float(np.median(_pel[_pel > np.percentile(_pel, 85) * 0.97]))

qpos = np.loadtxt(os.path.join(OUTD, "g1_replay.csv"), delimiter=",")
frame_files = sorted(f for f in os.listdir(FRAMES) if f.endswith(".jpg"))
T0 = float(os.environ.get("T0", "0"))                       # qpos row 0 <-> video t=T0
OFF = int(round(T0 * FPS))
T = min(OFF + len(qpos), len(frame_files), N)
FW, FH = Image.open(os.path.join(FRAMES, frame_files[0])).size
f_frame = (FH / 2.0) / np.tan(float(lz["fov_h"]) / 2.0)     # VGGT focal in frame px
s_fix = F_SAM / f_frame / KAPPA                             # body_fix rescale factor

# ---- G1 mesh: constant topology, per-frame rigid transforms
m = mujoco.MjModel.from_xml_path(G1_XML)
d = mujoco.MjData(m)
visual = [g for g in range(m.ngeom)
          if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
          and m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0]
local_verts, faces_list, off = [], [], 0
for g in visual:
    mid = m.geom_dataid[g]
    va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
    fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
    local_verts.append(m.mesh_vert[va:va + vn].astype(np.float32))
    faces_list.append(m.mesh_face[fa:fa + fn].astype(np.int32) + off)
    off += vn
faces = np.concatenate(faces_list)
Vtot = off


def assemble(n):
    d.qpos[:] = qpos[n]
    mujoco.mj_kinematics(m, d)
    out = np.empty((Vtot, 3), np.float32)
    o = 0
    for i, g in enumerate(visual):
        v = local_verts[i]
        R = d.geom_xmat[g].reshape(3, 3).astype(np.float32)
        out[o:o + len(v)] = v @ R.T + d.geom_xpos[g].astype(np.float32)
        o += len(v)
    return out


F_T, FIDX, BARY = gr.sample_mesh(faces, assemble(0), density=DENSITY)
Kt = (f_frame, f_frame, FW / 2.0, FH / 2.0)
wr = VideoWriter(OUTP, FW, FH, FPS)
print(f"[overlay] {T} frames, F_SC {F_SC:.4f}, body_fix {s_fix:.3f}, "
      f"f {f_frame:.1f}px, V {Vtot}, alpha {ALPHA}")
for n in range(T):
    frame = gr.from_np(np.asarray(Image.open(os.path.join(FRAMES, frame_files[n]))))
    if n < OFF:                                             # before the qpos window
        wr.write(gr.to_np(frame))
        continue
    vw = assemble(n - OFF).astype(np.float64) / F_SC        # robot -> subject world
    vc = (vw - c_c2w[n]) @ Minv[n].T
    # body_fix rescale about the SUBJECT's camera-frame mid-hip (the fitter's fixed point)
    hips_w = 0.5 * (Jw_fit[n, K["lhip"]] + Jw_fit[n, K["rhip"]])
    if not np.isfinite(hips_w).all() or not np.isfinite(vc).all():
        wr.write(gr.to_np(frame))
        continue
    pelv = (hips_w - c_c2w[n]) @ Minv[n].T
    vc = pelv + (vc - pelv) * s_fix
    ghost = frame.clone()
    gr.mesh_splat(ghost, torch.as_tensor(vc, dtype=torch.float32, device=gr.DEV),
                  F_T, FIDX, BARY, Kt, base_color=COLOR)
    out = (ALPHA * ghost.float() + (1.0 - ALPHA) * frame.float()).clamp(0, 255).byte()
    wr.write(gr.to_np(out))
    if n % 100 == 0:
        print(f"[overlay] {n}/{T}")
wr.close()
print(f"[overlay] WROTE {OUTP} ({T} frames @ {FPS:.3f})")
