#!/usr/bin/env python
"""SOMA-style retarget: figure.mp4 MotionRecon output -> Unitree G1 replay in MuJoCo.

Pipeline (design/figure.md section 7; gait.md pipeline stage 1):
  fit3d.npz (world z-up, G1 scale by MR_PELVIS_H construction, smoothed, 58 joints;
             chirality-consistent - the fit MUST be produced with MR_FLIPFIX=1 for the
             faceless figure robot, see fit_pose.py)
    1. chirality sanity check (forward vs pelvis velocity on walking frames)
    2. direction transfer -> G1-34 joint-position targets with exact G1 bone lengths,
       placed at the fit3d pelvis with the corrected heading (world frame)
    3. root pose set analytically (mid-hip + orthonormal hip/torso frame);
       29 hinge dofs solved by damped-least-squares IK on the actual MuJoCo G1 model
       (joint limits from the XML, warm-started frame to frame)
    4. soft ground clamp (downward root shift only, smoothed) + qpos gap interpolation
Outputs: outputs/figure/pose/g1_replay.csv [T,36] @ video fps + _sxs.mp4 (video | MuJoCo).

Usage (kimodo env): MUJOCO_GL=egl python figure/soma_retarget.py [--no-render]
Scenario override: MR_OUT (fit3d/lift3d dir), POSE_WORK (frames dir for the side-by-side).
"""
import os
import sys

import numpy as np

import mujoco

_PKG = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_PKG)
sys.path.insert(0, REPO)
sys.path.insert(0, _PKG)
from kimodo.skeleton.definitions import G1Skeleton34  # noqa: E402

OUTD = os.environ.get("MR_OUT", os.path.join(REPO, "outputs", "figure", "pose"))
WORK = os.environ.get("POSE_WORK",
                      "/tmp/claude-1000/-home-lucius-data-personal-hive-code-kimodo/"
                      "853101ea-754a-452e-bac7-b2f1af4f21a0/pose_full")
G1_XML = os.path.join(REPO, "kimodo/assets/skeletons/g1skel34/xml/g1.xml")
FPS = float(os.environ.get("MR_FPS", 24000 / 1001))

# ---- fit3d joint order (body 18 + hands, fit_pose.KP order)
KPN = ["nose", "lsho", "rsho", "lelb", "relb", "lhip", "rhip", "lkne", "rkne",
       "lank", "rank", "lbtoe", "lheel", "rbtoe", "rheel", "rwri", "lwri", "neck"]
K = {n: i for i, n in enumerate(KPN)}
SWAP18 = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 13, 14, 11, 12, 16, 15, 17]

fz = np.load(os.path.join(OUTD, "fit3d.npz"))
lz = np.load(os.path.join(OUTD, "lift3d.npz"))
Jw_all, tsec, ok = fz["joints_w"].astype(np.float64), fz["t"], fz["ok"].astype(bool)
N, NJ = Jw_all.shape[:2]
CAM = lz["c_c2w"] if "c_c2w" in lz.files else np.repeat(lz["cam_pos"][None], N, 0)

# ---------------------------------------------------------------- 1. chirality check
# fit3d arrives chirality-consistent: fit_pose.py (MR_FLIPFIX=1) undoes SAM's front/back
# mirror on the raw kp3d BEFORE smoothing. Verify here: forward must align with the pelvis
# velocity on walking frames.
mid = 0.5 * (Jw_all[:, K["lhip"]] + Jw_all[:, K["rhip"]])
idx_all = np.arange(N)
okm = np.isfinite(mid[:, 0])
mid_i = np.stack([np.interp(idx_all, idx_all[okm], mid[okm, c]) for c in range(3)], 1)
vel = np.gradient(mid_i, axis=0) * FPS
speed = np.nan_to_num(np.linalg.norm(vel[:, :2], axis=1))
hipline0 = Jw_all[:, K["lhip"]] - Jw_all[:, K["rhip"]]
fwd0 = np.cross(hipline0, [0.0, 0.0, 1.0])
fwd0 /= np.linalg.norm(fwd0, axis=1, keepdims=True) + 1e-9
w = ok & (speed > 0.3)
if w.sum() > 10:
    align = ((fwd0[w, :2] * vel[w, :2]).sum(1) > 0).mean()
    print(f"[soma] chirality check: fwd·vel>0 on {align*100:.0f}% of {int(w.sum())} walk frames "
          f"(expect ~100 with MR_FLIPFIX fit)")
Jw = Jw_all

# ---------------------------------------------------------------- 2. direction transfer targets
BONES34 = G1Skeleton34.bone_order_names_with_parents
NAMES = [n for n, _ in BONES34]
IDX = {n: i for i, n in enumerate(NAMES)}
PARENT = [IDX[p] if p else -1 for _, p in BONES34]
ref = np.load(os.path.join(REPO, "outputs", "gait", "go_forward.npz"))
STAND = ref["posed_joints"][0]                          # kimodo frame: y-up, faces +Z
M_W2K = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])   # world(z-up)->kimodo(y-up)
BONE_LEN = np.zeros(len(NAMES)); STAND_DIR = np.zeros((len(NAMES), 3))
for i, p in enumerate(PARENT):
    if p < 0:
        continue
    v = STAND[i] - STAND[p]
    BONE_LEN[i] = np.linalg.norm(v)
    STAND_DIR[i] = v / max(BONE_LEN[i], 1e-9)

SEG = {  # G1 bone -> (a, b) human segment in KPN names; None = stand direction (heading-rotated)
    "left_hip_pitch_skel": ("midhip", "lhip"), "right_hip_pitch_skel": ("midhip", "rhip"),
    "left_hip_roll_skel": None, "left_hip_yaw_skel": None,
    "right_hip_roll_skel": None, "right_hip_yaw_skel": None,
    "left_knee_skel": ("lhip", "lkne"), "right_knee_skel": ("rhip", "rkne"),
    "left_ankle_pitch_skel": ("lkne", "lank"), "right_ankle_pitch_skel": ("rkne", "rank"),
    "left_ankle_roll_skel": ("lkne", "lank"), "right_ankle_roll_skel": ("rkne", "rank"),
    "left_toe_base": ("lheel", "lbtoe"), "right_toe_base": ("rheel", "rbtoe"),
    "waist_yaw_skel": ("midhip", "neck"), "waist_roll_skel": ("midhip", "neck"),
    "waist_pitch_skel": ("midhip", "neck"),
    "left_shoulder_pitch_skel": ("lumbar", "lsho"), "right_shoulder_pitch_skel": ("lumbar", "rsho"),
    "left_shoulder_roll_skel": ("lsho", "lelb"), "left_shoulder_yaw_skel": ("lsho", "lelb"),
    "right_shoulder_roll_skel": ("rsho", "relb"), "right_shoulder_yaw_skel": ("rsho", "relb"),
    "left_elbow_skel": ("lsho", "lelb"), "right_elbow_skel": ("rsho", "relb"),
    "left_wrist_roll_skel": ("lelb", "lwri"), "left_wrist_pitch_skel": ("lelb", "lwri"),
    "left_wrist_yaw_skel": ("lelb", "lwri"), "left_hand_roll_skel": ("lelb", "lwri"),
    "right_wrist_roll_skel": ("relb", "rwri"), "right_wrist_pitch_skel": ("relb", "rwri"),
    "right_wrist_yaw_skel": ("relb", "rwri"), "right_hand_roll_skel": ("relb", "rwri"),
}
LEG_DAMP = {"left_knee_skel", "right_knee_skel", "left_ankle_pitch_skel",
            "right_ankle_pitch_skel", "left_ankle_roll_skel", "right_ankle_roll_skel"}

hipline_c = Jw[:, K["lhip"]] - Jw[:, K["rhip"]]        # corrected hip lines
yaw_w = np.arctan2(hipline_c[:, 1], hipline_c[:, 0])   # hip-line angle vs +X (world, about z)
targets = np.full((N, len(NAMES), 3), np.nan)          # world-frame G1-34 joint targets
for n in range(N):
    if not ok[n]:
        continue
    kp_k = (Jw[n] - mid[n]) @ M_W2K.T                  # pelvis-centred, kimodo frame
    hip = kp_k[K["lhip"]] - kp_k[K["rhip"]]
    yaw = np.arctan2(hip[2], hip[0])                   # kimodo yaw of the hip line
    cy, sy = np.cos(yaw), np.sin(yaw)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    kp = kp_k @ Ry.T                                   # roty(+yaw): hip line -> +X (pose_retarget)
    midhip = 0.5 * (kp[K["lhip"]] + kp[K["rhip"]])

    def hpt(name):
        if name == "midhip":
            return midhip
        if name == "lumbar":
            return midhip + 0.35 * (kp[K["neck"]] - midhip)
        return kp[K[name]]

    pose = np.zeros((len(NAMES), 3))
    for i, name in enumerate(NAMES):
        p = PARENT[i]
        if p < 0:
            continue
        seg = SEG.get(name)
        if seg is None:
            d = STAND_DIR[i]
        else:
            d = hpt(seg[1]) - hpt(seg[0])
            nl = np.linalg.norm(d)
            d = STAND_DIR[i] if nl < 1e-6 else d / nl
        if name in LEG_DAMP:                           # SAM overestimates the robot's leg splay
            d = d.copy(); d[0] *= 0.55
            d /= np.linalg.norm(d)
        pose[i] = pose[p] + BONE_LEN[i] * d
    for j, name in enumerate(NAMES):                   # partial leg straightening (see doc 7.4)
        if any(s in name for s in ("hip", "knee", "ankle", "toe")):
            pose[j] = 0.85 * pose[j] + 0.15 * (STAND[j] - STAND[IDX["pelvis_skel"]])
    pose -= pose[IDX["pelvis_skel"]]
    if n % 500 == 0:                                   # residual-yaw self-check (pose_retarget)
        hp = kp[K["lhip"]] - kp[K["rhip"]]
        assert abs(np.degrees(np.arctan2(hp[2], hp[0]))) < 5.0, \
            f"yaw normalization broken at frame {n}"
    # back to world: un-yaw (kimodo), kimodo->world, translate to the fit3d mid-hip
    targets[n] = (pose @ Ry) @ M_W2K + mid[n]

np.savez_compressed(os.path.join(OUTD, "g1_targets.npz"), targets=targets.astype(np.float32),
                    ok=ok, t=tsec)

# ---------------------------------------------------------------- 3. MuJoCo DLS IK
m = mujoco.MjModel.from_xml_path(G1_XML)
d = mujoco.MjData(m)
TRACK = [  # (G1-34 target joint, xml body, weight)
    ("left_knee_skel", "left_knee_link", 1.0), ("right_knee_skel", "right_knee_link", 1.0),
    ("left_ankle_roll_skel", "left_ankle_roll_link", 2.0),
    ("right_ankle_roll_skel", "right_ankle_roll_link", 2.0),
    ("left_shoulder_pitch_skel", "left_shoulder_pitch_link", 1.2),
    ("right_shoulder_pitch_skel", "right_shoulder_pitch_link", 1.2),
    ("left_elbow_skel", "left_elbow_link", 1.0), ("right_elbow_skel", "right_elbow_link", 1.0),
    ("left_wrist_yaw_skel", "left_wrist_yaw_link", 1.5),
    ("right_wrist_yaw_skel", "right_wrist_yaw_link", 1.5),
]
BID = [m.body(b).id for _, b, _ in TRACK]
WGT = np.array([w for _, _, w in TRACK])
TIDX = [IDX[j] for j, _, _ in TRACK]
# toe point fixed in each ankle body (stand-pose offset), tracks the toe target
TOE = {"left": (IDX["left_toe_base"], m.body("left_ankle_roll_link").id),
       "right": (IDX["right_toe_base"], m.body("right_ankle_roll_link").id)}
# stand-frame local toe offsets from the G1-34 stand pose
ank_l = STAND[IDX["left_ankle_roll_skel"]]; toe_l = STAND[IDX["left_toe_base"]]
TOE_LOC = (toe_l - ank_l) @ M_W2K                      # kimodo->world axes (rest-aligned)
lo = m.jnt_range[1:, 0].copy(); hi = m.jnt_range[1:, 1].copy()
free = m.jnt_range[1:, 0] >= m.jnt_range[1:, 1]        # unlimited joints
lo[free], hi[free] = -np.pi, np.pi

up_body = Jw[:, K["neck"]] - mid


def root_pose(n):
    l = hipline_c[n] / (np.linalg.norm(hipline_c[n]) + 1e-9)
    z = up_body[n] / (np.linalg.norm(up_body[n]) + 1e-9)
    x = np.cross(l, z); x /= np.linalg.norm(x) + 1e-9
    y = np.cross(z, x)
    R = np.stack([x, y, z], 1)
    pos = mid[n] + R @ np.array([0.0, 0.0, 0.1027])    # xml hip joints sit 0.1027 below pelvis
    return pos, R


def mat2quat(R):
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, R.ravel())
    return q


qpos_out = np.full((N, 36), np.nan)
q_warm = None
nv = m.nv
for n in range(N):
    if not ok[n]:
        continue
    pos, R = root_pose(n)
    d.qpos[:] = 0.0
    d.qpos[:3] = pos
    d.qpos[3:7] = mat2quat(R)
    d.qpos[7:] = q_warm if q_warm is not None else 0.0
    tgt = targets[n]
    for it in range(6):
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        rows, Js = [], []
        for (tj, bid, w) in zip(TIDX, BID, WGT):
            jacp = np.zeros((3, nv)); jacr = np.zeros((3, nv))
            mujoco.mj_jacBody(m, d, jacp, jacr, bid)
            rows.append(w * (tgt[tj] - d.xpos[bid]))
            Js.append(w * jacp[:, 6:])
        for side, (tj, bid) in TOE.items():
            pt = d.xpos[bid] + d.xmat[bid].reshape(3, 3) @ (TOE_LOC * (1 if side == "left" else 1))
            jacp = np.zeros((3, nv)); jacr = np.zeros((3, nv))
            mujoco.mj_jac(m, d, jacp, jacr, pt, bid)
            rows.append(1.5 * (tgt[tj] - pt))
            Js.append(1.5 * jacp[:, 6:])
        r = np.concatenate(rows)
        J = np.concatenate(Js, 0)
        dq = np.linalg.solve(J.T @ J + 1e-4 * np.eye(29), J.T @ r)
        d.qpos[7:] = np.clip(d.qpos[7:] + dq, lo, hi)
        if np.linalg.norm(dq) < 1e-4:
            break
    q_warm = d.qpos[7:].copy()
    qpos_out[n] = d.qpos.copy()
    if n % 600 == 0:
        print(f"[soma] IK {n}/{N} res={np.linalg.norm(r)/max(len(rows),1):.3f}", flush=True)

# ---------------------------------------------------------------- 4. gaps, ground, save
okq = np.isfinite(qpos_out[:, 0])
first, last = np.flatnonzero(okq)[0], np.flatnonzero(okq)[-1]
idx = np.arange(N)
qpos_out[:, 3:7] *= np.where((qpos_out[:, 3:4] < 0) & okq[:, None], -1.0, 1.0)  # hemisphere-align
for c in range(36):                                    # linear gap fill (quat renormalized below)
    qpos_out[:, c] = np.interp(idx, idx[okq], qpos_out[okq, c])
qpos_out = qpos_out[first:last + 1]
# temporal smoothing (targets are smooth; this kills residual IK jitter / axis flips):
# 5-frame median + 7-frame box on every column, quat renormalized after
for c in range(36):
    v = qpos_out[:, c]
    for kk, op in ((5, "med"), (7, "box")):
        pad = np.pad(v, kk // 2, mode="edge")
        from numpy.lib.stride_tricks import sliding_window_view
        wins = sliding_window_view(pad, kk)
        v = np.median(wins, 1) if op == "med" else wins.mean(1)
    qpos_out[:, c] = v
qpos_out[:, 3:7] /= np.linalg.norm(qpos_out[:, 3:7], axis=1, keepdims=True)

# soft ground clamp: shift root down/up so the lowest foot geom kisses z=0 when near it
mfl = mujoco.MjModel.from_xml_path(G1_XML)
dfl = mujoco.MjData(mfl)
foot_z = np.zeros(len(qpos_out))
fb = [mfl.body("left_ankle_roll_link").id, mfl.body("right_ankle_roll_link").id]
for i in range(len(qpos_out)):
    dfl.qpos[:] = 0.0; dfl.qpos[:36] = qpos_out[i]
    mujoco.mj_kinematics(mfl, dfl)
    lowest = min(dfl.xpos[fb[0]][2], dfl.xpos[fb[1]][2]) - 0.055   # ankle centre -> sole
    foot_z[i] = lowest
corr = np.where(foot_z < 0.03, -foot_z, 0.0)           # lift/press only when near the floor
kk = 11; pad = kk // 2
corr = np.array([np.median(corr[max(0, i - pad):i + pad + 1]) for i in range(len(corr))])
qpos_out[:, 2] += corr

csv = os.path.join(OUTD, "g1_replay.csv")
np.savetxt(csv, qpos_out, delimiter=",")
res_fk = []
print(f"[soma] {len(qpos_out)} frames ({len(qpos_out)/FPS:.1f}s at {FPS:.3f} fps) -> {csv}")
print(f"[soma] t window {tsec[first]:.2f}..{tsec[last]:.2f}s  ground corr median {np.median(corr)*100:.1f}cm")
print("DONE_SOMA")
