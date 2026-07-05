#!/usr/bin/env python
"""Render a qpos CSV inside the kitchen scene with the video-matched FIXED camera, animating the
dishwasher door (hinge) and bottom rack (slide) in sync with the choreography segment bounds.

Usage: MUJOCO_GL=egl python figure/render_scene.py <csv> <xml> <out.mp4|out.png> [--sheet]
--sheet: 8-frame contact sheet PNG (framing / gesture QA) instead of the video.
Camera + door meta from outputs/figure/helix_kitchen_scene.json; animation windows from
outputs/figure/helix_kitchen_bounds.json (skipped for sub-segment CSVs that don't match).
See design/figure.md."""
import json
import os
import sys

import imageio.v2 as imageio
import numpy as np

import mujoco

_PKG = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_PKG)
OUT = os.path.join(REPO, "outputs", "figure")
STEM = "helix_kitchen"

CSV, XML, OUTP = sys.argv[1], sys.argv[2], sys.argv[3]
SHEET = "--sheet" in sys.argv
FPS = 30
W, H = 960, 720

qpos = np.loadtxt(CSV, delimiter=",")
meta = json.load(open(os.path.join(OUT, STEM + "_scene.json")))
OPEN_ANGLE = meta.get("door_open_angle", 1.31 * meta.get("door_open_sign", -1.0))
RACK_OUT = meta.get("rack_out", 0.30)


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def anim_tracks(nframes, fps):
    """Door angle + rack slide per frame, from the segment bounds (identity if bounds don't apply)."""
    door = np.zeros(nframes); rack = np.zeros(nframes)
    bpath = os.path.join(OUT, STEM + "_bounds.json")
    if not os.path.exists(bpath):
        return door + OPEN_ANGLE, rack                   # standalone clip: assume door open
    bj = json.load(open(bpath))
    bounds = dict(bj["bounds"])
    order = sorted(bj["bounds"], key=lambda kv: kv[1])
    ends = {name: (order[i + 1][1] if i + 1 < len(order) else nframes)
            for i, (name, _) in enumerate(order)}
    if max(bounds.values()) >= nframes + 1:              # sub-segment CSV: leave everything open
        return door + OPEN_ANGLE, rack
    t = np.arange(nframes) / fps

    def window(seg, t0, t1):                             # seg-local seconds -> global frame times
        s = bounds[seg] / fps
        return s + t0, s + t1

    o0, o1 = window("open_door", 1.5, 3.1)               # hands swing the door down (robot retreats)
    c0, c1 = window("close_door", 1.8, 3.3)              # foot lifts it back up
    door = OPEN_ANGLE * (smoothstep((t - o0) / (o1 - o0)) - smoothstep((t - c0) / (c1 - c0)))
    p0, p1 = window("pull_rack", 1.2, 2.2)
    q0, q1 = window("push_rack", 1.2, 2.2)
    rack = RACK_OUT * (smoothstep((t - p0) / (p1 - p0)) - smoothstep((t - q0) / (q1 - q0)))
    return door, rack


m = mujoco.MjModel.from_xml_path(XML)
m.vis.global_.offwidth, m.vis.global_.offheight = W, H
m.vis.global_.fovy = float(meta.get("fovy", 45.0))
d = mujoco.MjData(m)
r = mujoco.Renderer(m, height=H, width=W)
cam = mujoco.MjvCamera()
cam.lookat[:] = meta["lookat"]
cam.azimuth = float(meta["azimuth"])
cam.elevation = float(meta["elevation"])
cam.distance = float(meta["distance"])

# ---- constraint-target skeleton overlay (helix_kitchen_targets.npz from the generator)
_az, _el = np.radians(cam.azimuth), np.radians(cam.elevation)
_fwd = np.array([np.cos(_el) * np.cos(_az), np.cos(_el) * np.sin(_az), np.sin(_el)])
_cpos = np.array(meta["lookat"]) - cam.distance * _fwd
_right = np.cross(_fwd, [0.0, 0.0, 1.0]); _right /= np.linalg.norm(_right)
_upc = np.cross(_right, _fwd)
_foc = 0.5 * H / np.tan(0.5 * np.radians(m.vis.global_.fovy))

TW = {}
_tpath = os.path.join(OUT, STEM + "_targets.npz")
if os.path.exists(_tpath):
    _tz = np.load(_tpath)
    for fr, pj in zip(_tz["frames"], _tz["pj"]):
        TW.setdefault(int(fr), pj)

try:
    import sys as _sys
    _sys.path.insert(0, REPO)
    from kimodo.skeleton.definitions import G1Skeleton34
    _names = [n for n, _ in G1Skeleton34.bone_order_names_with_parents]
    _idx = {n: i for i, n in enumerate(_names)}
    _EDGES = [(_idx[p], _idx[c]) for c, p in G1Skeleton34.bone_order_names_with_parents if p]
except Exception:
    _EDGES = []
from PIL import Image, ImageDraw  # noqa: E402


def project(p):
    v = p - _cpos
    depth = float(v @ _fwd)
    if depth < 0.1:
        return None
    return (W / 2 + (v @ _right) / depth * _foc, H / 2 - (v @ _upc) / depth * _foc)


def draw_targets(img, i, hold=9):
    """Overlay the nearest constraint pin skeleton within `hold` frames (green = what Kimodo
    was asked to hit at this moment)."""
    if not TW or not _EDGES:
        return img
    near = [fr for fr in range(i - hold, i + hold + 1) if fr in TW]
    if not near:
        return img
    fr = min(near, key=lambda fr: abs(fr - i))
    pj = TW[fr]
    pil = Image.fromarray(img)
    dr = ImageDraw.Draw(pil)
    pts = [project(p) for p in pj]
    for a_, b_ in _EDGES:
        if pts[a_] and pts[b_]:
            dr.line([pts[a_], pts[b_]], fill=(40, 255, 80), width=3)
    for pt in pts:
        if pt:
            dr.ellipse([pt[0] - 3, pt[1] - 3, pt[0] + 3, pt[1] + 3], fill=(255, 230, 40))
    return np.asarray(pil)

n = min(36, qpos.shape[1])
adr_door = m.joint("dw_door_hinge").qposadr[0] if "dw_door_hinge" in [m.joint(i).name for i in range(m.njnt)] else None
adr_rack = m.joint("dw_rack_slide").qposadr[0] if adr_door is not None else None
door_tr, rack_tr = anim_tracks(len(qpos), FPS)


def render_frame(i):
    d.qpos[:] = 0.0
    d.qpos[:n] = qpos[i, :n]
    if adr_door is not None:
        d.qpos[adr_door] = door_tr[i]
        d.qpos[adr_rack] = rack_tr[i]
    mujoco.mj_forward(m, d)
    r.update_scene(d, camera=cam)
    return draw_targets(r.render(), i)


if SHEET:
    idx = np.linspace(0, len(qpos) - 1, 8).astype(int)
    shots = [render_frame(i) for i in idx]
    rows = [np.concatenate(shots[:4], axis=1), np.concatenate(shots[4:], axis=1)]
    imageio.imwrite(OUTP, np.concatenate(rows, axis=0))
    print(f"[scene] SHEET {OUTP} frames={list(idx)}")
else:
    wr = imageio.get_writer(OUTP, fps=FPS, quality=8, macro_block_size=1)
    for i in range(len(qpos)):
        wr.append_data(render_frame(i))
    wr.close()
    print(f"[scene] WROTE {OUTP} ({len(qpos)} frames @ {FPS})")
