#!/usr/bin/env python
"""Replay comparison video: source video frames LEFT, G1 replay in MuJoCo RIGHT (LAYOUT=v
stacks them vertically instead).

Two scene modes:
  SCENE=plain   (default) bare floor, camera at the RECONSTRUCTED video camera (lift3d pose +
                VGGT fovy) - the robot appears where/at the scale the real one does.
  SCENE=kitchen the reproduction kitchen (outputs/figure/kitchen_g1.xml) with its video-matched
                camera; the replay trajectory is translated recon->scene by (scene work spot -
                kitchen_fit pocket); dishwasher door/rack held open (video work phase).

Usage: MUJOCO_GL=egl [SCENE=kitchen] [T0=4.0] python figure/render_replay.py
Env: MR_OUT (g1_replay.csv dir), POSE_WORK (frames), MR_FPS, OUT (mp4 path).
"""
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

import mujoco

_PKG = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_PKG)
from humanoid_motion_recon.fastvid import VideoWriter   # noqa: E402  (pip install -e ../humanoid-motion-reconstruction)

OUTD = os.environ.get("MR_OUT", os.path.join(REPO, "outputs", "figure", "pose"))
WORK = os.environ.get("POSE_WORK",
                      "/tmp/claude-1000/-home-lucius-data-personal-hive-code-kimodo/"
                      "853101ea-754a-452e-bac7-b2f1af4f21a0/pose_full")
FIG = os.path.join(REPO, "outputs", "figure")
SCENE = os.environ.get("SCENE", "plain")
FPS = float(os.environ.get("MR_FPS", 24000 / 1001))
T0 = float(os.environ.get("T0", "4.0"))
OUTP = os.environ.get("OUT", os.path.join(OUTD, f"g1_replay_{SCENE}.mp4"))
PW, PH = 960, 540

qpos = np.loadtxt(os.path.join(OUTD, "g1_replay.csv"), delimiter=",")

if SCENE == "kitchen":
    XML = os.path.join(FIG, "kitchen_g1.xml")
    meta = json.load(open(os.path.join(FIG, "helix_kitchen_scene.json")))
    hk = np.loadtxt(os.path.join(FIG, "helix_kitchen.csv"), delimiter=",")
    bounds = dict(json.load(open(os.path.join(FIG, "helix_kitchen_bounds.json")))["bounds"])
    pW = hk[bounds["open_door"], :2]                    # scene work spot
    # recon -> scene translation (both frames +X work-facing): the robot is PLANTED at the
    # work pocket for most of the video (t ~ 12-190 s), so the replay's own work-phase median
    # IS the recon pocket. (kitchen_fit.json's pocket is stale - solved against an earlier
    # reconstruction's world frame - and put the robot ~9 m off-camera.)
    i0, i1 = int((12.0 - T0) * FPS), int((190.0 - T0) * FPS)
    pR = np.nanmedian(qpos[max(0, i0):max(1, i1), :2], axis=0)
    shift = pW - pR
    qpos = qpos.copy()
    qpos[:, 0] += shift[0]; qpos[:, 1] += shift[1]
    cam_lookat = meta["lookat"]; cam_az = meta["azimuth"]
    cam_el = meta["elevation"]; cam_dist = meta["distance"]; fovy = meta.get("fovy", 30.0)
else:
    XML = os.environ.get("ROBOT_XML",
                         os.path.join(REPO, "kimodo/assets/skeletons/g1skel34/xml/g1.xml"))
    lz = np.load(os.path.join(OUTD, "lift3d.npz"))
    view = lz["R_cam2world"] @ np.array([0.0, 0.0, 1.0]); view /= np.linalg.norm(view)
    fovy = float(np.degrees(lz["fov_h"])) if "fov_h" in lz.files else 30.0
    _d = os.environ.get("DIST", "auto")                  # auto: ~2.2 m vertical span at the
    DIST = (1.1 / np.tan(np.radians(fovy) / 2)          # robot, whatever the lens
            if _d == "auto" else float(_d))
    cam_lookat = (lz["cam_pos"] + DIST * view).tolist()
    cam_az = float(np.degrees(np.arctan2(view[1], view[0])))
    cam_el = float(np.degrees(np.arcsin(np.clip(view[2], -1, 1))))
    cam_dist = DIST
    # CAM=track (default): follow the robot like the source tracking shot - lookat rides a
    # smoothed pelvis path (fixed az/el/dist from the reconstructed camera). CAM=fixed keeps
    # the static reconstructed-camera framing.
    if os.environ.get("CAM", "track") == "track":
        k = int(2 * round(1.0 * FPS / 2) + 1)            # ~1 s box smooth, odd
        pad = np.pad(qpos[:, :3], ((k // 2, k // 2), (0, 0)), mode="edge")
        cam_lookat = np.stack([np.convolve(pad[:, c], np.ones(k) / k, "valid")
                               for c in range(3)], 1)
        cam_lookat[:, 2] = float(os.environ.get("LOOKAT_Z", "0.7"))  # frame the whole body

m = mujoco.MjModel.from_xml_path(XML)
m.vis.global_.offwidth, m.vis.global_.offheight = PW, PH
m.vis.global_.fovy = fovy
d = mujoco.MjData(m)
r = mujoco.Renderer(m, height=PH, width=PW)
cam = mujoco.MjvCamera()
cam.lookat[:] = cam_lookat[0] if np.ndim(cam_lookat) == 2 else cam_lookat
cam.azimuth, cam.elevation, cam.distance = cam_az, cam_el, cam_dist

# kitchen extras: hold the dishwasher door open + rack out (the video's work phase)
adr_door = adr_rack = None
jnames = [m.joint(i).name for i in range(m.njnt)]
if "dw_door_hinge" in jnames:
    adr_door = m.joint("dw_door_hinge").qposadr[0]
    adr_rack = m.joint("dw_rack_slide").qposadr[0]
    meta_s = json.load(open(os.path.join(FIG, "helix_kitchen_scene.json")))
    DOOR_OPEN = meta_s.get("door_open_angle", -1.31)
    RACK_OUT = meta_s.get("rack_out", 0.38)

frames = sorted(glob.glob(os.path.join(WORK, "frames", "f*.jpg")))
n0 = int(round(T0 * FPS))
n_frames = min(len(qpos), len(frames) - n0)
HORIZ = os.environ.get("LAYOUT", "h") == "h"             # h: video | MuJoCo side by side
AXIS = 1 if HORIZ else 0
w0, h0 = Image.open(frames[0]).size                      # keep the source aspect ratio
if HORIZ:
    VW, VH = 2 * round(PH * w0 / h0 / 2), PH
    wr = VideoWriter(OUTP, VW + PW, PH, FPS)
else:
    VW, VH = PW, 2 * round(PW * h0 / w0 / 2)
    wr = VideoWriter(OUTP, PW, VH + PH, FPS)
nq = min(36, qpos.shape[1])
for i in range(n_frames):
    top = np.asarray(Image.open(frames[n0 + i]).resize((VW, VH)), dtype=np.uint8)
    d.qpos[:] = 0.0
    d.qpos[:nq] = qpos[i, :nq]
    if adr_door is not None:
        d.qpos[adr_door] = DOOR_OPEN
        d.qpos[adr_rack] = RACK_OUT
    mujoco.mj_forward(m, d)
    if np.ndim(cam_lookat) == 2:
        cam.lookat[:] = cam_lookat[min(i, len(cam_lookat) - 1)]
    r.update_scene(d, camera=cam)
    wr.write(np.concatenate([top, r.render()], AXIS))
    if i % 1200 == 0:
        print(f"[replay] {i}/{n_frames}", flush=True)
wr.close()
print(f"[replay] WROTE {OUTP} ({n_frames} frames @ {FPS:.3f}, scene={SCENE})")
