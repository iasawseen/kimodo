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
        [ROBOT_XML=<robot mjcf>] [ROBOT_STAND_Z=<standing root height, 0.72=G1>] \
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
# UPSCALE renders at N x the processed-frame size (frames Lanczos-upscaled, ghost
# rasterized sharp at the higher res; all intrinsics scale together) - worth it when
# the processed frames are small (xpeng source is only 480x540). CQ = NVENC quality.
UPSCALE = float(os.environ.get("UPSCALE", "1"))
CQ = int(os.environ.get("CQ", "27"))
# GHOST_SCALE enlarges the ghost about its own per-frame ground-contact point (root xy
# at z=0): feet stay planted on the tracked ground and the walked path stays aligned.
GHOST_SCALE = float(os.environ.get("GHOST_SCALE", "1"))
# VIEW2=1 adds a second pane (collate style). With DEPTH_WORK set (dir holding the VGGT
# depth cache depth/f%05d.npz + the video frames) it renders the DENSE RGB-colored
# per-frame cloud exactly like pose_over_cloud (conf-gated backprojection, +33 deg orbit
# about the median scene depth, painter's splat) with the ghost mesh in the same tilted
# camera; without DEPTH_WORK it falls back to the sparse lift3d scene cloud.
VIEW2 = os.environ.get("VIEW2", "0") != "0"
V2_TILT = np.radians(float(os.environ.get("VIEW2_TILT", "33")))
DEPTH_WORK = os.environ.get("DEPTH_WORK", "")
V2_CONF = float(os.environ.get("V2_CONF", "1.2"))
ROBOT_XML = os.environ.get("ROBOT_XML",
                           os.path.join(REPO, "kimodo/assets/skeletons/g1skel34/xml/g1.xml"))
STAND_Z = float(os.environ.get("ROBOT_STAND_Z", "0.72"))    # robot root height at stand

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

# retarget-world -> fit-world scale. For NV-soma trajectories pass SOMA_HIPS_SCALE
# (the scaler's effective Hips scale: model_height/1.8 x joint_scales.Hips — G1 0.7744,
# Chel 0.883): the chain-consistent factor is template-midhip (0.918) x hips-scale /
# subject plateau, which makes the ghost's walked path match the subject's exactly.
# The STAND_Z/plateau fallback equates the robot ROOT with the subject MID-HIP — an
# ~9% mismatch for robots whose root rides the SOMA Hips joint (8.5 cm above the hip
# line), which shrank the ghost and its path (depth lag -> reads small + floating).
_pel = 0.5 * (Jw_fit[:, K["lhip"], 2] + Jw_fit[:, K["rhip"], 2])
_pel = _pel[np.isfinite(_pel)]
_plat = float(np.median(_pel[_pel > np.percentile(_pel, 85) * 0.97]))
_hips = os.environ.get("SOMA_HIPS_SCALE", "")
F_SC = (0.918 * float(_hips) / _plat) if _hips else (STAND_Z / _plat)

# subject ground level over time: the fit world's z=0 floor is calibrated once and can
# drift (xpeng: heels sink to -0.2 m as the subject nears the camera), which leaves a
# z=0-standing ghost hovering above the subject's feet. Track the planted-heel level
# (per-frame min heel z, ~0.8 s box smooth) and ride the ghost on it. FLOOR_TRACK=0
# restores the fixed z=0 floor.
if os.environ.get("FLOOR_TRACK", "1") != "0":
    _hl = np.nanmin(Jw_fit[:, [K["lheel"], K["rheel"]], 2], axis=1)
    _m = np.isfinite(_hl)
    _hl = np.interp(np.arange(N), np.flatnonzero(_m), _hl[_m])
    _k = max(3, 2 * int(round(0.4 * FPS)) + 1)
    _pad = np.pad(_hl, (_k // 2, _k // 2), mode="edge")
    floor_z = np.convolve(_pad, np.ones(_k) / _k, "valid")
    print(f"[overlay] floor track: heel level {floor_z.min():.3f}..{floor_z.max():.3f} m")
else:
    floor_z = np.zeros(N)

qpos = np.loadtxt(os.path.join(OUTD, "g1_replay.csv"), delimiter=",")
frame_files = sorted(f for f in os.listdir(FRAMES) if f.endswith(".jpg"))
T0 = float(os.environ.get("T0", "0"))                       # qpos row 0 <-> video t=T0
OFF = int(round(T0 * FPS))
T = min(OFF + len(qpos), len(frame_files), N)
FW, FH = Image.open(os.path.join(FRAMES, frame_files[0])).size
FW0, FH0, F_SAM0 = FW, FH, F_SAM                            # original processed-frame px
if UPSCALE != 1.0:                                          # everything scales together:
    FW, FH = 2 * round(FW * UPSCALE / 2), 2 * round(FH * UPSCALE / 2)
    F_SAM *= UPSCALE                                        # SAM focal in upscaled px
f_frame = (FH / 2.0) / np.tan(float(lz["fov_h"]) / 2.0)     # VGGT focal in frame px
s_fix = F_SAM / f_frame / KAPPA                             # body_fix rescale factor

# ---- G1 mesh: constant topology, per-frame rigid transforms
m = mujoco.MjModel.from_xml_path(ROBOT_XML)
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

PW2 = FW
if VIEW2 and DEPTH_WORK:                                    # dense colored per-frame cloud
    _d0 = np.load(os.path.join(DEPTH_WORK, "depth", "f00000.npz"))
    D_H, D_W = _d0["depth"].shape
    SC2 = FH / D_H
    PW2 = 2 * round(D_W * SC2 / 2)
    _vs2, _us2 = torch.meshgrid(
        torch.arange(D_H, dtype=torch.float32, device=gr.DEV),
        torch.arange(D_W, dtype=torch.float32, device=gr.DEV), indexing="ij")
    SQ2 = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], device=gr.DEV)
elif VIEW2:                                                 # fallback: sparse lift3d cloud
    scene_w = lz["scene"].astype(np.float64)                # colored by height
    _z0, _z1 = np.percentile(scene_w[:, 2], 2), np.percentile(scene_w[:, 2], 98)
    _t = np.clip((scene_w[:, 2] - _z0) / max(_z1 - _z0, 1e-6), 0, 1)[:, None]
    scene_rgb = torch.as_tensor(
        ((1 - _t) * np.array([95.0, 115.0, 140.0]) + _t * np.array([230.0, 240.0, 250.0]))
        .astype(np.uint8), device=gr.DEV)

_CA2, _SA2 = float(np.cos(V2_TILT)), float(np.sin(V2_TILT))


def pane2_dense(n, vw, pelv, frame):
    """pose_over_cloud pane: conf-gated RGB cloud orbited +V2_TILT about the median
    scene depth, ghost mesh body-fixed + tilted into the same camera."""
    dz = np.load(os.path.join(DEPTH_WORK, "depth", f"f{n:05d}.npz"))
    D = gr.from_np(dz["depth"]).float()
    C = gr.from_np(dz["conf"]).float()
    pe = dz["pose_enc"]
    fy = (D_H / 2.0) / np.tan(pe[7] / 2.0)
    fx = (D_W / 2.0) / np.tan(pe[8] / 2.0)
    src = gr.resize(frame, D_H, D_W)
    good = C > V2_CONF
    Dg = D[good]
    # pivot the orbit on the SUBJECT's depth when known: in dark scenes the confident
    # points are all far (ceiling lights) and a scene-median pivot swings the near-field
    # robot out of frame
    if pelv is not None:
        zmid = float(pelv[2])
    else:
        zmid = float(torch.quantile(Dg, 0.5)) if Dg.numel() else 1.0
    X = (_us2[good] - D_W / 2) / fx * Dg
    Y = (_vs2[good] - D_H / 2) / fy * Dg
    Yt = _CA2 * Y - _SA2 * (Dg - zmid)
    Zt = (_SA2 * Y + _CA2 * (Dg - zmid) + zmid).clamp(min=1e-4)
    u = (PW2 / 2 + X / Zt * fx * SC2).clamp(0, PW2 - 2)
    v = (FH / 2 + Yt / Zt * fy * SC2).clamp(0, FH - 2)
    cv2_ = gr.canvas(FH, PW2)
    gr.splat_cloud(cv2_, torch.stack([u, v], 1), Zt, src[good], SQ2)
    if vw is not None:
        vc = (vw - c_c2w[n]) @ Minv[n].T
        s2 = F_SAM0 * (D_W / FW0) / np.sqrt(fx * fy) / KAPPA
        vc = pelv + (vc - pelv) * s2                        # match the cloud's angular size
        Zr = vc[:, 2] - zmid
        gv = np.stack([vc[:, 0], _CA2 * vc[:, 1] - _SA2 * Zr,
                       _SA2 * vc[:, 1] + _CA2 * Zr + zmid], 1)
        gr.mesh_splat(cv2_, torch.as_tensor(gv, dtype=torch.float32, device=gr.DEV),
                      F_T, FIDX, BARY, (fx * SC2, fy * SC2, PW2 / 2.0, FH / 2.0),
                      base_color=COLOR)
    return gr.to_np(cv2_)


def pane2(n, vw, pelv, frame):
    if DEPTH_WORK:
        return pane2_dense(n, vw, pelv, frame)
    pelv_z = float(pelv[2]) if pelv is not None else \
        float(np.median(((scene_w - c_c2w[n]) @ Minv[n].T)[:, 2]))

    def tilt(P):
        Zr = P[:, 2] - pelv_z
        Yt = _CA2 * P[:, 1] - _SA2 * Zr
        Zt = _SA2 * P[:, 1] + _CA2 * Zr + pelv_z
        return np.stack([P[:, 0], Yt, Zt], 1)

    cv2_ = gr.canvas(FH, FW, color=(10, 12, 16))
    pc = tilt((scene_w - c_c2w[n]) @ Minv[n].T)
    m = pc[:, 2] > 0.2
    u = FW / 2 + pc[m, 0] / pc[m, 2] * f_frame
    v = FH / 2 + pc[m, 1] / pc[m, 2] * f_frame
    inb = (u >= 0) & (u < FW - 1) & (v >= 0) & (v < FH - 1)
    idx = np.flatnonzero(m)[inb]
    gr.splat_cloud(cv2_, torch.as_tensor(np.stack([u[inb], v[inb]], 1),
                                         dtype=torch.float32, device=gr.DEV),
                   torch.as_tensor(pc[m, 2][inb], dtype=torch.float32, device=gr.DEV),
                   scene_rgb[torch.as_tensor(idx, device=gr.DEV)], stencil=gr.DISK2)
    if vw is not None:
        gv = tilt((vw - c_c2w[n]) @ Minv[n].T)
        gr.mesh_splat(cv2_, torch.as_tensor(gv, dtype=torch.float32, device=gr.DEV),
                      F_T, FIDX, BARY, Kt, base_color=COLOR)
    return gr.to_np(cv2_)


wr = VideoWriter(OUTP, FW + (PW2 if VIEW2 else 0), FH, FPS, cq=CQ)
print(f"[overlay] {T} frames, F_SC {F_SC:.4f}, body_fix {s_fix:.3f}, "
      f"f {f_frame:.1f}px, V {Vtot}, alpha {ALPHA}, view2 {VIEW2}")
for n in range(T):
    img = Image.open(os.path.join(FRAMES, frame_files[n]))
    if img.size != (FW, FH):
        img = img.resize((FW, FH), Image.LANCZOS)
    frame = gr.from_np(np.asarray(img))
    vw = None
    pelv = None
    if n >= OFF:
        vw = assemble(n - OFF).astype(np.float64) / F_SC    # robot -> subject world
        if GHOST_SCALE != 1.0:
            c = np.array([qpos[n - OFF, 0] / F_SC, qpos[n - OFF, 1] / F_SC, 0.0])
            vw = c + (vw - c) * GHOST_SCALE
        vw[:, 2] += floor_z[n]                              # ride the subject's ground
    hips_w = 0.5 * (Jw_fit[n, K["lhip"]] + Jw_fit[n, K["rhip"]])
    if vw is not None and np.isfinite(hips_w).all() and np.isfinite(vw).all():
        vc = (vw - c_c2w[n]) @ Minv[n].T
        # body_fix rescale about the SUBJECT's camera-frame mid-hip (fitter's fixed point)
        pelv = (hips_w - c_c2w[n]) @ Minv[n].T
        vc = pelv + (vc - pelv) * s_fix
        ghost = frame.clone()
        gr.mesh_splat(ghost, torch.as_tensor(vc, dtype=torch.float32, device=gr.DEV),
                      F_T, FIDX, BARY, Kt, base_color=COLOR)
        out = (ALPHA * ghost.float() + (1.0 - ALPHA) * frame.float()).clamp(0, 255).byte()
    else:
        vw = None                                           # no ghost in either pane
        pelv = None
        out = frame
    main = gr.to_np(out)
    if VIEW2:
        main = np.concatenate([main, pane2(n, vw, pelv, frame)], 1)
    wr.write(main)
    if n % 100 == 0:
        print(f"[overlay] {n}/{T}")
wr.close()
print(f"[overlay] WROTE {OUTP} ({T} frames @ {FPS:.3f})")
