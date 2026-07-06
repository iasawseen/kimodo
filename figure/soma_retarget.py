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
Jw_fit, tsec, ok = fz["joints_w"].astype(np.float64), fz["t"], fz["ok"].astype(bool)
N, NJ = Jw_fit.shape[:2]
CAM = lz["c_c2w"] if "c_c2w" in lz.files else np.repeat(lz["cam_pos"][None], N, 0)

# ---------------------------------------------------------------- 1. chirality-corrected joints
# fit3d articulation is UNUSABLE for retargeting: SAM's per-frame front/back mirror flickers
# were smoothed into the fit (design/figure.md 7.1) and cannot be undone post-hoc. Rebuild the
# articulation from the RAW per-frame kp3d (flips still crisp there), resolve chirality at THIS
# layer by hypothesis selection - per frame the truth is either the raw estimate or its mirror
# (L/R label swap + depth flip about the pelvis); both hypotheses' geometry is computed
# NUMERICALLY, and a global 2-state Viterbi picks the sequence: unaries anchor moving frames to
# the (mirror-invariant, cleanly smoothed) fit3d pelvis velocity, pairwise = hip-line
# continuity + a switch penalty. Only the trajectory (pelvis path) is taken from fit3d.
# Smoothing of the articulation happens AFTER correction, where it is safe.
# MHR-70 indices in fit_pose.KP order: body 18 + right hand 20 + left hand 20
KIDX70 = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 20, 41, 62, 69] + \
    [b + 4 * f + j for b in (21, 42) for f in range(5) for j in range(4)]
SWAP58 = SWAP18 + list(range(38, 58)) + list(range(18, 38))     # body swap + r-hand <-> l-hand

# MHR rig joints for rotation targets, identified geometrically from joint_coords (stable
# across frames): 1=pelvis (== global_rot), 112=chest/neck, 41=r wrist, 77=l wrist
ROT_J = [1, 112, 41, 77]
raw_cache = os.path.join(WORK, "raw58.npz")
rc = np.load(raw_cache) if os.path.exists(raw_cache) else {}
if "rel" in rc and "rots" in rc:
    rel_raw, ok_raw, rots_raw = rc["rel"], rc["ok"], rc["rots"]
else:
    import glob as _glob
    mfiles = sorted(_glob.glob(os.path.join(WORK, "mhr", "f*.npz")))
    rel_raw = np.full((N, 58, 3), np.nan, np.float32)
    rots_raw = np.full((N, 4, 3, 3), np.nan, np.float32)
    ok_raw = np.zeros(N, bool)
    for mf in mfiles:
        n = int(os.path.basename(mf)[1:6])
        if n >= N:
            continue
        mz = np.load(mf)
        k3 = mz["kp3d"]
        if not np.isfinite(k3).all():
            continue
        midc = 0.5 * (k3[9] + k3[10])
        rel_raw[n] = (k3 - midc)[KIDX70]
        if "joint_rots" in mz.files:
            rots_raw[n] = mz["joint_rots"][ROT_J].astype(np.float32)
        ok_raw[n] = True
    np.savez_compressed(raw_cache, rel=rel_raw, ok=ok_raw, rots=rots_raw)
    print(f"[soma] cached raw58 rel offsets + rig rotations -> {raw_cache}")
ok &= ok_raw

# mirrored hypothesis: L/R label swap + depth mirror about the pelvis (z=0 in rel coords)
rel_mir = rel_raw[:, SWAP58].copy()
rel_mir[:, :, 2] *= -1.0

# world scale/orientation chain from the fit (a = offset ratio, kappa = body-size calib)
a_s = fz["a"].astype(np.float64)
kappa = float(fz["kappa"]) if "kappa" in fz.files else 1.0
if "M_c2w" in lz.files:
    R_c2w = lz["M_c2w"].astype(np.float64)               # per-frame affine (scale folded in)
else:
    R_c2w = np.repeat((float(lz["scale"]) * lz["R_cam2world"])[None], N, 0)
off_raw = kappa * np.einsum("nij,nkj->nki", R_c2w, a_s[:, None, None] * rel_raw)
off_mir = kappa * np.einsum("nij,nkj->nki", R_c2w, a_s[:, None, None] * rel_mir)

# smoothed, mirror-invariant trajectory + velocity from fit3d
mid = 0.5 * (Jw_fit[:, K["lhip"]] + Jw_fit[:, K["rhip"]])
idx_all = np.arange(N)
okm = np.isfinite(mid[:, 0])
mid_i = np.stack([np.interp(idx_all, idx_all[okm], mid[okm, c]) for c in range(3)], 1)
vel = np.gradient(mid_i, axis=0) * FPS
speed = np.nan_to_num(np.linalg.norm(vel[:, :2], axis=1))


def _fwd(off):
    l = off[:, K["lhip"]] - off[:, K["rhip"]]
    f = np.cross(l, [0.0, 0.0, 1.0])
    return f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-9), \
        l / (np.linalg.norm(l, axis=1, keepdims=True) + 1e-9)


FWD = [None, None]; HIP = [None, None]
FWD[0], HIP[0] = _fwd(off_raw)
FWD[1], HIP[1] = _fwd(off_mir)

# global 2-state Viterbi over the flip sequence
oki = np.flatnonzero(ok)
nfr = len(oki)
W_VEL, W_SW = 4.0, 1.5
cost = np.zeros((nfr, 2))
for s in (0, 1):
    f = FWD[s][oki]
    ali = (f[:, :2] * vel[oki, :2]).sum(1) / (speed[oki] + 1e-9)
    cost[:, s] = np.where(speed[oki] > 0.2,
                          W_VEL * np.maximum(0.0, -ali) * np.minimum(speed[oki], 1.0), 0.0)
acc = cost[0].copy()
back = np.zeros((nfr, 2), np.int8)
for k in range(1, nfr):
    n, p = oki[k], oki[k - 1]
    new = np.empty(2)
    for s in (0, 1):
        cont = [1.0 - float(HIP[s][n] @ HIP[sp][p]) for sp in (0, 1)]
        tr = [acc[sp] + cont[sp] + (W_SW if s != sp else 0.0) for sp in (0, 1)]
        back[k, s] = int(np.argmin(tr))
        new[s] = min(tr) + cost[k, s]
    acc = new
flip = np.zeros(N, bool)
s = int(np.argmin(acc))
for k in range(nfr - 1, -1, -1):
    flip[oki[k]] = bool(s)
    s = int(back[k, s])
print(f"[soma] chirality (raw, hypothesis-select): {int(flip[oki].sum())}/{nfr} frames mirrored")

# corrected world offsets, THEN temporal smoothing (safe post-correction), + fit3d trajectory
off_c = np.where(flip[:, None, None], off_mir, off_raw)
from numpy.lib.stride_tricks import sliding_window_view
flat = off_c.reshape(N, -1)
for c in range(flat.shape[1]):
    v = flat[:, c]
    m = np.isfinite(v)
    if m.sum() < 10:
        continue
    vi = np.interp(idx_all, idx_all[m], v[m])
    for kk, op in ((7, "med"), (11, "box")):
        pad = np.pad(vi, kk // 2, mode="edge")
        wins = sliding_window_view(pad, kk)
        vi = np.median(wins, 1) if op == "med" else wins.mean(1)
    v[m] = vi[m]
Jw = mid_i[:, None] + off_c

# acceptance metric: corrected forward vs velocity on walking frames
fwd_c, _ = _fwd(off_c)
w = ok & (speed > 0.3)
if w.sum() > 10:
    align = ((fwd_c[w, :2] * vel[w, :2]).sum(1) > 0).mean()
    print(f"[soma] chirality check: fwd·vel>0 on {align*100:.0f}% of {int(w.sum())} walk frames "
          f"(expect ~100)")

# ------------------------------------------------- 1b. rotation targets (mesh/rig rotations)
# The MHR mesh is a deterministic function of the rig's per-joint GLOBAL rotations - so
# "retarget from mesh" = consume joint_rots. They carry what bone directions cannot: twist
# about the bone (all 3 wrist dofs) and true torso orientation. Chirality: the mirror of a
# rotation across the camera z-plane is the reflection conjugate S R S (S = diag(1,1,-1)),
# plus L/R label swap for the wrists; applied per frame from the SAME Viterbi flips.
S_MIR = np.diag([1.0, 1.0, -1.0])
rots_sel = rots_raw.astype(np.float64).copy()           # [N,4,3,3] (pelvis, chest, rwri, lwri)
fl = np.flatnonzero(flip)
for n in fl:
    rp, rc_, rr, rl = rots_raw[n].astype(np.float64)
    rots_sel[n, 0] = S_MIR @ rp @ S_MIR
    rots_sel[n, 1] = S_MIR @ rc_ @ S_MIR
    rots_sel[n, 2] = S_MIR @ rl @ S_MIR                 # label swap: mirrored L drives R
    rots_sel[n, 3] = S_MIR @ rr @ S_MIR
# RELATIVE rotations, WORLD axes. Two failure modes bracketed this design: global deltas
# double-count the chain (the analytic root already encodes torso lean -> waist clamps and
# recruits yaw/roll, twisted bows); rig-local right-composed deltas assume the rig joint's
# local axes match the G1 link's - they don't (v4: garbage targets, body-wide saturation).
# Convention-free form: the world-axes rotation from parent to child, Q = R_child @ R_parent^T;
# transfer its delta vs the stand reference onto the G1's own per-frame parent.
R_wc = R_c2w / np.cbrt(np.maximum(np.linalg.det(R_c2w), 1e-12))[:, None, None]
rots_w = np.einsum("nkj,nqjm->nqkm", R_wc, rots_sel)    # [N,4,3,3] world (pelvis,chest,rw,lw)
Q_chest = np.einsum("nkl,nml->nkm", rots_w[:, 1], rots_w[:, 0])   # chest @ pelvis^T
Q_rwri = np.einsum("nkl,nml->nkm", rots_w[:, 2], rots_w[:, 1])    # r wrist @ chest^T
Q_lwri = np.einsum("nkl,nml->nkm", rots_w[:, 3], rots_w[:, 1])
# reference frame: middle of the stand window (heading-consistent with the G1 reference pose)
st0 = float(os.environ.get("MR_STAND0", "7.5")); st1 = float(os.environ.get("MR_STAND1", "9.5"))
cand = np.flatnonzero(ok & (tsec >= st0) & (tsec <= st1))
REF_N = int(cand[len(cand) // 2]) if len(cand) else int(np.flatnonzero(ok)[0])
D_chest = np.einsum("nkl,ml->nkm", Q_chest, Q_chest[REF_N])       # dQ = Q(n) @ Q(ref)^T
D_rwri = np.einsum("nkl,ml->nkm", Q_rwri, Q_rwri[REF_N])
D_lwri = np.einsum("nkl,ml->nkm", Q_lwri, Q_lwri[REF_N])

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
LEG_SPLAY = float(os.environ.get("SOMA_LEG_SPLAY", "1.0"))
STRAIGHT = float(os.environ.get("SOMA_STRAIGHT", "0.0"))
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
            d = d.copy(); d[0] *= LEG_SPLAY
            d /= np.linalg.norm(d)
        pose[i] = pose[p] + BONE_LEN[i] * d
    for j, name in enumerate(NAMES):                   # partial leg straightening (see doc 7.4)
        if any(s in name for s in ("hip", "knee", "ankle", "toe")):
            pose[j] = (1 - STRAIGHT) * pose[j] + STRAIGHT * (STAND[j] - STAND[IDX["pelvis_skel"]])
    pose -= pose[IDX["pelvis_skel"]]
    if n % 500 == 0:                                   # residual-yaw self-check (pose_retarget)
        hp = kp[K["lhip"]] - kp[K["rhip"]]
        assert abs(np.degrees(np.arctan2(hp[2], hp[0]))) < 5.0, \
            f"yaw normalization broken at frame {n}"
    # back to world: un-yaw (kimodo), kimodo->world, translate to the fit3d mid-hip
    targets[n] = (pose @ Ry) @ M_W2K + mid[n]

# re-anchor arm targets on the REACHABLE chain: the direction-transfer composes G1-34 skeleton
# bone lengths from the lumbar, which lands elbow/wrist targets 15-20 cm off the XML chain's
# manifold (measured: ankle target->achieved 1.2 cm, arms 15-20 cm). Rebuild them from the
# shoulder target with the XML segment lengths along the MOCAP directions - feasible by
# construction; arm fidelity is then a direction match (mocap arms are ~30% longer than G1's,
# so absolute wrist position has an irreducible proportional floor).
_mx = mujoco.MjModel.from_xml_path(G1_XML)
_dx = mujoco.MjData(_mx)
_dx.qpos[:] = 0; _dx.qpos[3] = 1
mujoco.mj_kinematics(_mx, _dx)


def _xml_len(a, b):
    return float(np.linalg.norm(_dx.xpos[_mx.body(a).id] - _dx.xpos[_mx.body(b).id]))


L_UA = {"l": _xml_len("left_shoulder_pitch_link", "left_elbow_link"),
        "r": _xml_len("right_shoulder_pitch_link", "right_elbow_link")}
L_FA = {"l": _xml_len("left_elbow_link", "left_wrist_yaw_link"),
        "r": _xml_len("right_elbow_link", "right_wrist_yaw_link")}
for sd, pre in (("l", "left"), ("r", "right")):
    sho_t = targets[:, IDX[f"{pre}_shoulder_pitch_skel"]]
    d_ue = Jw[:, K[f"{sd}elb"]] - Jw[:, K[f"{sd}sho"]]
    d_ue /= np.linalg.norm(d_ue, axis=1, keepdims=True) + 1e-9
    elb_t = sho_t + L_UA[sd] * d_ue
    d_fw = Jw[:, K[f"{sd}wri"]] - Jw[:, K[f"{sd}elb"]]
    d_fw /= np.linalg.norm(d_fw, axis=1, keepdims=True) + 1e-9
    wri_t = elb_t + L_FA[sd] * d_fw
    targets[:, IDX[f"{pre}_elbow_skel"]] = elb_t
    for wn in (f"{pre}_wrist_roll_skel", f"{pre}_wrist_pitch_skel",
               f"{pre}_wrist_yaw_skel", f"{pre}_hand_roll_skel"):
        targets[:, IDX[wn]] = wri_t

np.savez_compressed(os.path.join(OUTD, "g1_targets.npz"), targets=targets.astype(np.float32),
                    ok=ok, t=tsec,
                    mocap=Jw.astype(np.float32), flip=flip)   # corrected mocap for eval_retarget

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
ARM_SCALE = float(os.environ.get("SOMA_W_ARM", "1.0"))
BID = [m.body(b).id for _, b, _ in TRACK]
WGT = np.array([w * (ARM_SCALE if any(k in j for k in ("shoulder", "elbow", "wrist")) else 1.0)
                for j, _, w in TRACK])
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


def rotvec_err(R_tgt, R_cur):
    """axis*angle of R_tgt @ R_cur.T (the rotation still needed)."""
    q = mat2quat(R_tgt @ R_cur.T)
    v = np.empty(3)
    mujoco.mju_quat2Vel(v, q, 1.0)
    return v


# rig-rotation targets -> G1 link orientations by LOCAL delta transfer: the rig's
# chest-in-pelvis delta drives torso-in-root; each wrist-in-chest delta drives
# wrist-in-torso. Composed per frame onto the G1's OWN root/torso, so nothing is
# double-counted with the analytic root or the position-tracked arm chain.
pos_r, R_r = root_pose(REF_N)
d.qpos[:] = 0.0
d.qpos[:3] = pos_r
d.qpos[3:7] = mat2quat(R_r)
mujoco.mj_kinematics(m, d)
B_TORSO = m.body("torso_link").id
B_LWRI = m.body("left_wrist_yaw_link").id
B_RWRI = m.body("right_wrist_yaw_link").id
R_torso0 = d.xmat[B_TORSO].reshape(3, 3).copy()
TQ0 = R_torso0 @ R_r.T                                  # G1 ref world-axes torso-vs-root
WQ0_L = d.xmat[B_LWRI].reshape(3, 3) @ R_torso0.T       # G1 ref world-axes wrist-vs-torso
WQ0_R = d.xmat[B_RWRI].reshape(3, 3) @ R_torso0.T
W_ORI_T = float(os.environ.get("SOMA_W_TORSO", "0.0"))
W_ORI_W = float(os.environ.get("SOMA_W_WRIST", "0.0"))

# dynamic arm anchoring: elbow/wrist targets rebuilt EVERY IK iteration from the CURRENT
# shoulder link along the mocap directions (pure direction matching, feasible from wherever
# the shoulder actually is - static arm targets assumed the shoulder hits its own target,
# which it misses by ~10 cm, and the least-squares compromise bent the forearm ~50 deg)
D_UE, D_FW = {}, {}
for sd, pre in (("l", "left"), ("r", "right")):
    v = Jw[:, K[f"{sd}elb"]] - Jw[:, K[f"{sd}sho"]]
    D_UE[sd] = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
    v = Jw[:, K[f"{sd}wri"]] - Jw[:, K[f"{sd}elb"]]
    D_FW[sd] = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
ARM_DYN = [(sd,
            m.body(f"{pre}_shoulder_pitch_link").id,
            m.body(f"{pre}_elbow_link").id,
            m.body(f"{pre}_wrist_yaw_link").id,
            1.0, 1.5)
           for sd, pre in (("l", "left"), ("r", "right"))]
ARM_BIDS = {b for _, _, be, bw, _, _ in ARM_DYN for b in (be, bw)}
# legs: same dynamic anchoring off the live hip link (same off-manifold disease as the arms)
D_TH, D_SH = {}, {}
for sd in ("l", "r"):
    v = Jw[:, K[f"{sd}kne"]] - Jw[:, K[f"{sd}hip"]]
    D_TH[sd] = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
    v = Jw[:, K[f"{sd}ank"]] - Jw[:, K[f"{sd}kne"]]
    D_SH[sd] = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
L_TH = {sd: _xml_len(f"{pre}_hip_pitch_link", f"{pre}_knee_link")
        for sd, pre in (("l", "left"), ("r", "right"))}
L_SH = {sd: _xml_len(f"{pre}_knee_link", f"{pre}_ankle_roll_link")
        for sd, pre in (("l", "left"), ("r", "right"))}
LEG_DYN = [(sd,
            m.body(f"{pre}_hip_pitch_link").id,
            m.body(f"{pre}_knee_link").id,
            m.body(f"{pre}_ankle_roll_link").id,
            1.0, 2.0)
           for sd, pre in (("l", "left"), ("r", "right"))]
LEG_BIDS = {b for _, _, bk, ba, _, _ in LEG_DYN for b in (bk, ba)}
ARM_BIDS |= LEG_BIDS

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
    for it in range(int(os.environ.get("SOMA_ITERS", "6"))):
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        rows, Js = [], []
        for (tj, bid, w) in zip(TIDX, BID, WGT):
            if bid in ARM_BIDS:
                continue                                # arm elbow/wrist handled dynamically
            jacp = np.zeros((3, nv)); jacr = np.zeros((3, nv))
            mujoco.mj_jacBody(m, d, jacp, jacr, bid)
            rows.append(w * (tgt[tj] - d.xpos[bid]))
            Js.append(w * jacp[:, 6:])
        for sd, bs, be, bw, we, ww in ARM_DYN:          # dynamic arm targets off the live shoulder
            elb_t = d.xpos[bs] + L_UA[sd] * D_UE[sd][n]
            wri_t = elb_t + L_FA[sd] * D_FW[sd][n]
            for bid, w, t_ in ((be, we, elb_t), (bw, ww, wri_t)):
                jacp = np.zeros((3, nv)); jacr = np.zeros((3, nv))
                mujoco.mj_jacBody(m, d, jacp, jacr, bid)
                rows.append(w * (t_ - d.xpos[bid]))
                Js.append(w * jacp[:, 6:])
        for sd, bh, bk, ba, wk, wa in LEG_DYN:          # dynamic leg targets off the live hip
            kne_t = d.xpos[bh] + L_TH[sd] * D_TH[sd][n]
            ank_t = kne_t + L_SH[sd] * D_SH[sd][n]
            for bid, w, t_ in ((bk, wk, kne_t), (ba, wa, ank_t)):
                jacp = np.zeros((3, nv)); jacr = np.zeros((3, nv))
                mujoco.mj_jacBody(m, d, jacp, jacr, bid)
                rows.append(w * (t_ - d.xpos[bid]))
                Js.append(w * jacp[:, 6:])
        for side, (tj, bid) in TOE.items():
            pt = d.xpos[bid] + d.xmat[bid].reshape(3, 3) @ (TOE_LOC * (1 if side == "left" else 1))
            jacp = np.zeros((3, nv)); jacr = np.zeros((3, nv))
            mujoco.mj_jac(m, d, jacp, jacr, pt, bid)
            rows.append(1.5 * (tgt[tj] - pt))
            Js.append(1.5 * jacp[:, 6:])
        if np.isfinite(D_chest[n, 0, 0]):               # rig-rotation orientation residuals
            R_tgt_torso = D_chest[n] @ TQ0 @ R          # world-axes deltas on the G1's chain
            R_cur_torso = d.xmat[B_TORSO].reshape(3, 3)
            for bid, w, R_tgt in ((B_TORSO, W_ORI_T, R_tgt_torso),
                                  (B_LWRI, W_ORI_W, D_lwri[n] @ WQ0_L @ R_cur_torso),
                                  (B_RWRI, W_ORI_W, D_rwri[n] @ WQ0_R @ R_cur_torso)):
                jacp = np.zeros((3, nv)); jacr = np.zeros((3, nv))
                mujoco.mj_jacBody(m, d, jacp, jacr, bid)
                rows.append(w * rotvec_err(R_tgt, d.xmat[bid].reshape(3, 3)))
                Js.append(w * jacr[:, 6:])
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
# 5-frame median + 7-frame box on every column, quat renormalized after.
# SOMA_QSMOOTH=0 disables (eval instrumentation: hinge-space smoothing compounds across the
# 4-hinge arm chain and can rotate the forearm during fast unloading motions)
QS = int(os.environ.get("SOMA_QSMOOTH", "1"))
for c in range(36 if QS else 0):
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
