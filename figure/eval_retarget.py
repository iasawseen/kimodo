#!/usr/bin/env python
"""Retarget eval: quantify the discrepancy between the motion capture (MotionRecon corrected
world joints, `g1_targets.npz:mocap`, G1 scale by construction) and the retargeted G1 replay
(`g1_replay.csv` FK'd through the real MuJoCo model).

Metrics (all meters/degrees, computed on ok frames):
  MPJPE-G   global mean per-joint position error (trajectory + pose)
  MPJPE-L   pelvis-aligned (translation removed) - pose fidelity, the score to optimize
  per-joint local error breakdown - WHERE it is wrong
  bone-direction cosine error - direction-transfer quality per bone
  heading   |mocap hip-line yaw - G1 root yaw| (deg)
  lag       per-joint temporal offset maximizing velocity correlation (frames; smoothing lag)
  foot-skate  G1 foot horizontal speed while the mocap heel is planted (<0.05 m/s)
  timeline  MPJPE-L per 10 s bucket - WHEN it is wrong

Usage (kimodo env): python figure/eval_retarget.py   [MR_OUT=outputs/figure/pose]
Writes <MR_OUT>/eval_retarget.json and prints the report.
"""
import json
import os
import sys

import mujoco
import numpy as np

_PKG = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_PKG)
OUTD = os.environ.get("MR_OUT", os.path.join(REPO, "outputs", "figure", "pose"))
FPS = float(os.environ.get("MR_FPS", 24000 / 1001))
G1_XML = os.path.join(REPO, "kimodo/assets/skeletons/g1skel34/xml/g1.xml")

KPN = ["nose", "lsho", "rsho", "lelb", "relb", "lhip", "rhip", "lkne", "rkne",
       "lank", "rank", "lbtoe", "lheel", "rbtoe", "rheel", "rwri", "lwri", "neck"]
K = {n: i for i, n in enumerate(KPN)}

# mocap joint -> G1 body whose origin should track it (the retarget's own TRACK set + hips)
PAIRS = [
    ("lkne", "left_knee_link"), ("rkne", "right_knee_link"),
    ("lank", "left_ankle_roll_link"), ("rank", "right_ankle_roll_link"),
    ("lsho", "left_shoulder_pitch_link"), ("rsho", "right_shoulder_pitch_link"),
    ("lelb", "left_elbow_link"), ("relb", "right_elbow_link"),
    ("lwri", "left_wrist_yaw_link"), ("rwri", "right_wrist_yaw_link"),
    ("lhip", "left_hip_pitch_link"), ("rhip", "right_hip_pitch_link"),
]
BONES = [("lhip", "lkne"), ("lkne", "lank"), ("rhip", "rkne"), ("rkne", "rank"),
         ("lsho", "lelb"), ("lelb", "lwri"), ("rsho", "relb"), ("relb", "rwri"),
         ("lhip", "rhip"), ("lsho", "rsho")]

tgt_path = os.path.join(OUTD, "g1_targets.npz")
csv_path = os.path.join(OUTD, "g1_replay.csv")
tz = np.load(tgt_path)
if "mocap" not in tz.files:
    sys.exit("g1_targets.npz has no 'mocap' - rerun figure/soma_retarget.py first")
age = os.path.getmtime(csv_path) - os.path.getmtime(tgt_path)
if abs(age) > 1800:
    print(f"[eval] WARNING: g1_replay.csv and g1_targets.npz differ by {age/60:.0f} min - "
          f"one may be stale (a crashed retarget leaves the previous csv in place)")
MC, ok, tsec = tz["mocap"].astype(np.float64), tz["ok"].astype(bool), tz["t"]
Q_rows = np.loadtxt(csv_path, delimiter=",")
# csv rows are the CONTIGUOUS global frames first..last (soma crops to the valid window);
# align onto the mocap frame axis - a row-0-to-frame-0 comparison is off by ~4 s and poisons
# every time-varying metric (caught by the saturated +24-frame lag on all joints)
first, last = np.flatnonzero(ok)[0], np.flatnonzero(ok)[-1]
assert len(Q_rows) == last - first + 1, \
    f"csv rows {len(Q_rows)} != ok window {last - first + 1} - stale csv?"
N = len(MC)
Q = np.full((N, Q_rows.shape[1]), np.nan)
Q[first:last + 1] = Q_rows
ok &= np.isfinite(Q[:, 0]) & np.isfinite(MC[:, 0, 0])

# FK the replay
m = mujoco.MjModel.from_xml_path(G1_XML)
d = mujoco.MjData(m)
BID = {b: m.body(b).id for _, b in PAIRS}
G1P = np.full((N, len(PAIRS), 3), np.nan)
G1_ROOT_YAW = np.full(N, np.nan)
for n in range(N):
    if not ok[n]:
        continue
    d.qpos[:] = Q[n]
    mujoco.mj_kinematics(m, d)
    for j, (_, b) in enumerate(PAIRS):
        G1P[n, j] = d.xpos[BID[b]]
    w, x, y, z = Q[n, 3:7]
    G1_ROOT_YAW[n] = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

MCP = np.stack([MC[:, K[a]] for a, _ in PAIRS], 1)      # [N,P,3] mocap partners

oki = np.flatnonzero(ok)
# pelvis frames for local (pose-only) comparison
mc_pelv = 0.5 * (MC[:, K["lhip"]] + MC[:, K["rhip"]])
g1_pelv = 0.5 * (G1P[:, [i for i, (a, _) in enumerate(PAIRS) if a == "lhip"][0]]
                 + G1P[:, [i for i, (a, _) in enumerate(PAIRS) if a == "rhip"][0]])
E_g = np.linalg.norm(MCP - G1P, axis=2)                 # [N,P] global
E_l = np.linalg.norm((MCP - mc_pelv[:, None]) - (G1P - g1_pelv[:, None]), axis=2)

# bone direction cosine error
cosd = {}
for a, b in BONES:
    va = MC[:, K[b]] - MC[:, K[a]]
    ia = [i for i, (x, _) in enumerate(PAIRS) if x == a]
    ib = [i for i, (x, _) in enumerate(PAIRS) if x == b]
    if not ia or not ib:
        continue
    vb = G1P[:, ib[0]] - G1P[:, ia[0]]
    c = (va * vb).sum(1) / (np.linalg.norm(va, axis=1) * np.linalg.norm(vb, axis=1) + 1e-9)
    cosd[f"{a}->{b}"] = float(np.degrees(np.arccos(np.clip(c[oki], -1, 1))).mean())

# heading error
hl = MC[:, K["lhip"]] - MC[:, K["rhip"]]
mc_yaw = np.arctan2(hl[:, 1], hl[:, 0]) - np.pi / 2     # hip line -> facing (+X convention)
dyaw = np.degrees(np.abs(np.angle(np.exp(1j * (mc_yaw - G1_ROOT_YAW)))))

# temporal lag per joint (velocity cross-correlation, +/-24 frames)
def lag_of(a_sig, b_sig):
    a = np.diff(a_sig, axis=0); b = np.diff(b_sig, axis=0)
    a = np.nan_to_num(np.linalg.norm(a, axis=1)); b = np.nan_to_num(np.linalg.norm(b, axis=1))
    a -= a.mean(); b -= b.mean()
    best, bl = -1e9, 0
    for L in range(-24, 25):
        v = (a[max(0, L):len(a) + min(0, L)] * b[max(0, -L):len(b) - max(0, L)]).mean()
        if v > best:
            best, bl = v, L
    return bl

lags = {a: lag_of(MCP[oki, i], G1P[oki, i]) for i, (a, _) in enumerate(PAIRS)}

# foot skate: G1 ankle horizontal speed while the mocap heel is planted
skate = {}
for heel, ank in (("lheel", "lank"), ("rheel", "rank")):
    hv = np.linalg.norm(np.diff(MC[:, K[heel], :2], axis=0), axis=1) * FPS
    ia = [i for i, (x, _) in enumerate(PAIRS) if x == ank][0]
    gv = np.linalg.norm(np.diff(G1P[:, ia, :2], axis=0), axis=1) * FPS
    planted = ok[1:] & ok[:-1] & (hv < 0.05)
    skate[heel] = float(np.nanmean(gv[planted])) if planted.sum() else float("nan")

# report
rep = {
    "frames": int(ok.sum()),
    "MPJPE_global_cm": round(float(E_g[oki].mean()) * 100, 2),
    "MPJPE_local_cm": round(float(E_l[oki].mean()) * 100, 2),
    "MPJPE_local_p95_cm": round(float(np.percentile(E_l[oki], 95)) * 100, 2),
    "heading_err_deg_mean": round(float(dyaw[oki].mean()), 2),
    "heading_err_deg_p95": round(float(np.percentile(dyaw[oki], 95)), 2),
    "per_joint_local_cm": {a: round(float(E_l[oki, i].mean()) * 100, 2)
                           for i, (a, _) in enumerate(PAIRS)},
    "bone_dir_err_deg": {k: round(v, 1) for k, v in cosd.items()},
    "lag_frames": lags,
    "foot_skate_mps": {k: round(v, 3) for k, v in skate.items()},
}
buckets = {}
for lo in range(int(tsec[oki][0]) // 10 * 10, int(tsec[oki][-1]) + 1, 10):
    mb = ok & (tsec >= lo) & (tsec < lo + 10)
    if mb.sum() > 5:
        buckets[f"{lo:03d}-{lo+10:03d}s"] = round(float(E_l[mb].mean()) * 100, 2)
rep["timeline_local_cm"] = buckets

print(f"[eval] frames {rep['frames']}  MPJPE global {rep['MPJPE_global_cm']} cm  "
      f"LOCAL {rep['MPJPE_local_cm']} cm (p95 {rep['MPJPE_local_p95_cm']})  "
      f"heading {rep['heading_err_deg_mean']} deg (p95 {rep['heading_err_deg_p95']})")
print("[eval] per-joint local cm:",
      "  ".join(f"{k}:{v}" for k, v in sorted(rep["per_joint_local_cm"].items(),
                                              key=lambda kv: -kv[1])))
print("[eval] bone dir err deg:",
      "  ".join(f"{k}:{v}" for k, v in sorted(rep["bone_dir_err_deg"].items(),
                                              key=lambda kv: -kv[1])))
print("[eval] lag frames:", "  ".join(f"{k}:{v:+d}" for k, v in rep["lag_frames"].items()))
print("[eval] foot skate m/s:", rep["foot_skate_mps"])
print("[eval] timeline local cm:",
      "  ".join(f"{k}:{v}" for k, v in rep["timeline_local_cm"].items()))
with open(os.path.join(OUTD, "eval_retarget.json"), "w") as f:
    json.dump(rep, f, indent=1)
print(f"[eval] WROTE {os.path.join(OUTD, 'eval_retarget.json')}")
