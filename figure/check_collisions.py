#!/usr/bin/env python
"""Collision QA gate for the Helix-02 reproduction (the design/gait.md /goal conditions applied
here: no interpenetration with scene solids, no self-collision).

Two complementary checks per frame (door/rack driven with the same schedule as the renderer):
 1. MuJoCo contact pipeline: the robot's *_collision geoms vs scene geoms
      - SOLID scene geoms: gated at TOL_SOLID
      - PROP geoms (door, racks, dishes, faucet, floor): reported FYI (mime contact expected)
      - robot self-collisions: gated at TOL_SELF
 2. BODY-POINT test: G1's collision geoms are sparse (no torso/knee/wrist/forearm cover), so key
    body origins (torso, knees, elbows, wrists, head) are tested point-in-box against every solid
    scene box with TOL_POINT depth - this catches mesh-level clipping the contact pipeline misses.

Usage: python figure/check_collisions.py [csv] [xml]   (defaults: outputs/figure/helix_kitchen.*)
Exit 1 on any gate failure.
"""
import json
import os
import sys

import numpy as np

import mujoco

_PKG = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_PKG)
OUT = os.path.join(REPO, "outputs", "figure")
STEM = "helix_kitchen"

CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUT, STEM + ".csv")
XML = sys.argv[2] if len(sys.argv) > 2 else os.path.join(OUT, "kitchen_g1.xml")
TOL_SOLID = 0.02
TOL_SELF = 0.01
TOL_POINT = 0.03          # m: how deep a body ORIGIN may sit inside a solid box
POINT_BODIES = ["torso_link", "left_knee_link", "right_knee_link", "left_elbow_link",
                "right_elbow_link", "left_wrist_roll_link", "right_wrist_roll_link"]
# limb midpoints (thigh/shin/upper arm): G1 collision geoms and body origins both skip the limb
# shafts, so a thin slab (tabletop) can pass between two origins undetected
MID_PAIRS = [("left_hip_pitch_link", "left_knee_link"), ("right_hip_pitch_link", "right_knee_link"),
             ("left_knee_link", "left_ankle_roll_link"), ("right_knee_link", "right_ankle_roll_link"),
             ("left_shoulder_roll_link", "left_elbow_link"),
             ("right_shoulder_roll_link", "right_elbow_link")]

qpos = np.loadtxt(CSV, delimiter=",")
meta = json.load(open(os.path.join(OUT, STEM + "_scene.json")))
SOLIDS = set(meta["solids"])
bounds = json.load(open(os.path.join(OUT, STEM + "_bounds.json")))["bounds"]
seg_of = np.empty(len(qpos), dtype=object)
for i, (name, start) in enumerate(bounds):
    end = bounds[i + 1][1] if i + 1 < len(bounds) else len(qpos)
    seg_of[start:end] = name

m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)

names = [m.geom(i).name for i in range(m.ngeom)]
robot, scene = set(), set()
for i, nm in enumerate(names):
    root = m.geom_bodyid[i]
    while m.body_parentid[root] != 0:
        root = m.body_parentid[root]
    if m.body(root).name == "pelvis":
        if nm.endswith("_collision"):
            robot.add(i)
    elif nm and nm != "floor":
        scene.add(i)
for i in range(m.ngeom):
    on = i in robot or i in scene
    m.geom_contype[i] = 1 if on else 0
    m.geom_conaffinity[i] = 1 if on else 0

solid_boxes = []                               # (name, center[3], halfsize[3]) - all axis-aligned
for i, nm in enumerate(names):
    if nm in SOLIDS and m.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX:
        solid_boxes.append((nm, m.geom_pos[i].copy(), m.geom_size[i].copy()))
point_ids = [m.body(nm).id for nm in POINT_BODIES]
mid_ids = []
for a, bnm in MID_PAIRS:
    try:
        mid_ids.append((m.body(a).id, m.body(bnm).id, f"mid:{a[:12]}"))
    except Exception:
        pass

OPEN_ANGLE = meta.get("door_open_angle", 1.31 * meta.get("door_open_sign", -1.0))
RACK_OUT = meta.get("rack_out", 0.30)


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


bmap = dict(bounds); fps = 30.0
t = np.arange(len(qpos)) / fps


def win(seg, t0, t1):
    return bmap[seg] / fps + t0, bmap[seg] / fps + t1


o0, o1 = win("open_door", 1.5, 3.1); c0, c1 = win("close_door", 1.8, 3.3)
door_tr = OPEN_ANGLE * (smoothstep((t - o0) / (o1 - o0)) - smoothstep((t - c0) / (c1 - c0)))
p0, p1 = win("pull_rack", 1.2, 2.2); q0, q1 = win("push_rack", 1.2, 2.2)
rack_tr = RACK_OUT * (smoothstep((t - p0) / (p1 - p0)) - smoothstep((t - q0) / (q1 - q0)))
adr_door = m.joint("dw_door_hinge").qposadr[0]
adr_rack = m.joint("dw_rack_slide").qposadr[0]

contact_rep, point_rep = {}, {}
n = min(36, qpos.shape[1])
for i in range(len(qpos)):
    d.qpos[:] = 0.0
    d.qpos[:n] = qpos[i, :n]
    d.qpos[adr_door] = door_tr[i]
    d.qpos[adr_rack] = rack_tr[i]
    mujoco.mj_forward(m, d)
    for k in range(d.ncon):
        con = d.contact[k]
        if con.dist >= 0:
            continue
        g1, g2 = con.geom1, con.geom2
        in1, in2 = g1 in robot, g2 in robot
        if in1 and in2:
            kind = "self"
        elif in1 or in2:
            sg = g2 if in1 else g1
            kind = "solid" if names[sg] in SOLIDS else "prop"
        else:
            continue
        pair = tuple(sorted((names[g1] or "?", names[g2] or "?")))
        pen = -con.dist
        cur = contact_rep.setdefault((kind, pair), [0, 0.0, ""])
        cur[0] += 1
        if pen > cur[1]:
            cur[1], cur[2] = pen, str(seg_of[i])
    # body-point vs solid boxes (origins + limb midpoints)
    probes = [(bid, bname) for bid, bname in zip(point_ids, POINT_BODIES)]
    probe_pts = [(d.xpos[bid], bname) for bid, bname in probes]
    probe_pts += [((d.xpos[a] + d.xpos[bb]) / 2, nm) for a, bb, nm in mid_ids]
    for p, bname in probe_pts:
        for nm, c, hs in solid_boxes:
            depth = float(min((hs[0] - abs(p[0] - c[0])), (hs[1] - abs(p[1] - c[1])),
                              (hs[2] - abs(p[2] - c[2]))))
            if depth > 0:                      # origin inside the box; depth = distance to exit
                cur = point_rep.setdefault((bname, nm), [0, 0.0, ""])
                cur[0] += 1
                if depth > cur[1]:
                    cur[1], cur[2] = depth, str(seg_of[i])

fail = False
for kind, label, tol in [("solid", "SOLID scene contact penetrations", TOL_SOLID),
                         ("self", "robot SELF-collisions", TOL_SELF),
                         ("prop", "prop contacts (mime, FYI)", None)]:
    rows = sorted(((p, v) for (kk, p), v in contact_rep.items() if kk == kind),
                  key=lambda kv: -kv[1][1])
    print(f"\n[collide] {label}: {len(rows)} pairs")
    for pair, (cnt, pen, seg) in rows[:12]:
        flag = "  <-- FAIL" if (tol is not None and pen > tol) else ""
        fail |= bool(flag)
        print(f"  {pair[0]:28s} x {pair[1]:22s} frames={cnt:4d} maxpen={pen*100:5.1f}cm @{seg}{flag}")

rows = sorted(point_rep.items(), key=lambda kv: -kv[1][1])
print(f"\n[collide] BODY-POINT inside solid boxes: {len(rows)} pairs")
for (bname, nm), (cnt, dep, seg) in rows[:12]:
    flag = "  <-- FAIL" if dep > TOL_POINT else ""
    fail |= bool(flag)
    print(f"  {bname:28s} in {nm:22s} frames={cnt:4d} depth={dep*100:5.1f}cm @{seg}{flag}")

print(f"\n[collide] {'FAIL' if fail else 'PASS'} "
      f"(tol solid {TOL_SOLID*100:.0f}cm, self {TOL_SELF*100:.0f}cm, point {TOL_POINT*100:.0f}cm)")
sys.exit(1 if fail else 0)
