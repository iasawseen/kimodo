#!/usr/bin/env python
"""Ground-clamp post-process: lift the root height per frame so the feet stop sinking into the
floor (the G1 retarget pushes feet up to ~7 cm underground in deep bends). Lift = the frame's max
foot-floor penetration, smoothed with a short moving average so the correction never pops.

Usage: python figure/ground_clamp.py            (in-place on outputs/figure/helix_kitchen.csv;
                                                 original saved once as helix_kitchen_raw.csv)
"""
import os
import shutil

import numpy as np

import mujoco

_PKG = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_PKG)
OUT = os.path.join(REPO, "outputs", "figure")
CSV = os.path.join(OUT, "helix_kitchen.csv")
RAW = os.path.join(OUT, "helix_kitchen_raw.csv")
XML = os.path.join(OUT, "kitchen_g1.xml")
WIN = 11                                       # smoothing window (frames)

if os.path.exists(RAW):
    qpos = np.loadtxt(RAW, delimiter=",")      # idempotent: always clamp from the raw motion
else:
    qpos = np.loadtxt(CSV, delimiter=",")
    shutil.copy(CSV, RAW)

m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)
foot, floor = set(), None
for i in range(m.ngeom):
    nm = m.geom(i).name
    if nm and "foot" in nm and nm.endswith("_collision"):
        foot.add(i)
        m.geom_contype[i] = 1; m.geom_conaffinity[i] = 1
    elif nm == "floor":
        floor = i
        m.geom_contype[i] = 1; m.geom_conaffinity[i] = 1
    else:
        m.geom_contype[i] = 0; m.geom_conaffinity[i] = 0

n = min(36, qpos.shape[1])
lift = np.zeros(len(qpos))
for i in range(len(qpos)):
    d.qpos[:] = 0.0
    d.qpos[:n] = qpos[i, :n]
    mujoco.mj_forward(m, d)
    pen = 0.0
    for k in range(d.ncon):
        c = d.contact[k]
        pair = {c.geom1, c.geom2}
        if floor in pair and (pair & foot) and c.dist < 0:
            pen = max(pen, -c.dist)
    lift[i] = pen

kernel = np.ones(WIN) / WIN
smooth = np.convolve(np.pad(lift, WIN // 2, mode="edge"), kernel, mode="valid")[:len(qpos)]
qpos[:, 2] += smooth
np.savetxt(CSV, qpos, delimiter=",")
print(f"[clamp] lift mean={smooth.mean()*100:.1f}cm max={smooth.max()*100:.1f}cm -> {CSV}")
