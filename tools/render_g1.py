#!/usr/bin/env python
"""Headless MuJoCo render of a G1 qpos CSV -> mp4. Camera follows the pelvis.

Usage:  MUJOCO_GL=egl python tools/render_g1.py <qpos.csv> <out.mp4> [fps]
Override the skeleton XML with the G1_XML env var if needed.
"""
import os
import sys

import imageio.v2 as imageio
import numpy as np

import mujoco

CSV = sys.argv[1]
OUT = sys.argv[2]
FPS = int(sys.argv[3]) if len(sys.argv) > 3 else 30
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # tools/ -> repo root
XML = os.environ.get("G1_XML", os.path.join(_REPO, "kimodo/assets/skeletons/g1skel34/xml/g1.xml"))
W, H = 960, 720

qpos = np.loadtxt(CSV, delimiter=",")
if qpos.ndim == 1:
    qpos = qpos[None]
model = mujoco.MjModel.from_xml_path(XML)
# default offscreen framebuffer is 640x480; enlarge it for our target resolution
model.vis.global_.offwidth = W
model.vis.global_.offheight = H
data = mujoco.MjData(model)
print(f"[render] nq={model.nq} csv_cols={qpos.shape[1]} frames={qpos.shape[0]} fps={FPS}")
n = min(model.nq, qpos.shape[1])
if model.nq != qpos.shape[1]:
    print(f"[render] WARN nq({model.nq}) != csv cols({qpos.shape[1]}); using first {n}")

renderer = mujoco.Renderer(model, height=H, width=W)
cam = mujoco.MjvCamera()
cam.azimuth = 130.0
cam.elevation = -15.0
cam.distance = 3.5

frames = []
for i in range(qpos.shape[0]):
    data.qpos[:] = 0.0
    data.qpos[:n] = qpos[i, :n]
    mujoco.mj_forward(model, data)
    root = np.array(data.qpos[:3])
    cam.lookat[:] = [float(root[0]), float(root[1]), 0.8]
    renderer.update_scene(data, camera=cam)
    frames.append(renderer.render())

os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
imageio.mimsave(OUT, frames, fps=FPS, quality=8, macro_block_size=1)
print(f"[render] WROTE {OUT} ({len(frames)} frames @ {FPS}fps)")
