#!/usr/bin/env python
"""Retarget SAM-3D-Body MHR-70 keypoints (from figure.mp4) to G1-34 anchor poses.

Direction-transfer retarget: every G1 bone takes the DIRECTION of its matching human segment
(thigh, shank, torso, upper arm, forearm, ...) and keeps its own G1 LENGTH (measured from the
reference stand pose). Poses are yaw-normalized (hip line -> +X, i.e. facing +Z like all Kimodo
anchor poses) and grounded (lowest foot joint on the stand-pose floor level). The pre-
normalization hip-line yaw and the camera-frame pelvis are saved too, so segment scripts can
drive Root2D heading/position from the video as well.

In:  <work>/mhr/f%05d.npz (kp3d [70,3] camera frame: x right, y DOWN, z forward)
Out: <work>/g1_anchors.npz  pj [N,34,3], yaw [N], t [N] seconds, pelvis_cam [N,3]

Usage (kimodo env, CPU):  POSE_WORK=<work> python figure/pose_retarget.py
"""
import glob
import os

import numpy as np

WORK = os.environ.get("POSE_WORK", "")
_PKG = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_PKG)
FPS_SAMPLE = 2.0

# ---- G1 skeleton (34): tree + reference stand pose (no GPU / no model load needed)
import sys
sys.path.insert(0, REPO)
from kimodo.skeleton.definitions import G1Skeleton34  # noqa: E402

BONES = G1Skeleton34.bone_order_names_with_parents
NAMES = [n for n, _ in BONES]
IDX = {n: i for i, n in enumerate(NAMES)}
PARENT = [IDX[p] if p else -1 for _, p in BONES]
ref = np.load(os.path.join(REPO, "outputs", "gait", "go_forward.npz"))
STAND = ref["posed_joints"][0]                        # [34,3], +Y up, faces +Z
BONE_LEN = np.zeros(len(NAMES))
STAND_DIR = np.zeros((len(NAMES), 3))
for i, p in enumerate(PARENT):
    if p < 0:
        continue
    v = STAND[i] - STAND[p]
    BONE_LEN[i] = np.linalg.norm(v)
    STAND_DIR[i] = v / max(BONE_LEN[i], 1e-9)
FLOOR_Y = min(STAND[IDX["left_toe_base"], 1], STAND[IDX["right_toe_base"], 1])

# ---- MHR-70 keypoint indices (sam_3d_body/metadata/mhr70.py order)
MHR = {n: i for i, n in enumerate([
    "nose", "left-eye", "right-eye", "left-ear", "right-ear", "left-shoulder", "right-shoulder",
    "left-elbow", "right-elbow", "left-hip", "right-hip", "left-knee", "right-knee", "left-ankle",
    "right-ankle", "left-big-toe-tip", "left-small-toe-tip", "left-heel", "right-big-toe-tip",
    "right-small-toe-tip", "right-heel"])}
MHR["left-wrist"], MHR["right-wrist"], MHR["neck"] = 62, 41, 69

# G1 bone -> human segment (a, b): direction b-a (None = keep stand direction; "midhip" special)
SEG = {
    "left_hip_pitch_skel": ("midhip", "left-hip"), "right_hip_pitch_skel": ("midhip", "right-hip"),
    "left_hip_roll_skel": None, "left_hip_yaw_skel": None,
    "right_hip_roll_skel": None, "right_hip_yaw_skel": None,
    "left_knee_skel": ("left-hip", "left-knee"), "right_knee_skel": ("right-hip", "right-knee"),
    "left_ankle_pitch_skel": ("left-knee", "left-ankle"),
    "right_ankle_pitch_skel": ("right-knee", "right-ankle"),
    "left_ankle_roll_skel": ("left-knee", "left-ankle"),
    "right_ankle_roll_skel": ("right-knee", "right-ankle"),
    "left_toe_base": ("left-heel", "left-big-toe-tip"),
    "right_toe_base": ("right-heel", "right-big-toe-tip"),
    "waist_yaw_skel": ("midhip", "neck"), "waist_roll_skel": ("midhip", "neck"),
    "waist_pitch_skel": ("midhip", "neck"),
    # waist->shoulder is G1's long torso link: use lumbar->shoulder (has the vertical rise;
    # neck->shoulder is nearly horizontal and collapses the chest)
    "left_shoulder_pitch_skel": ("lumbar", "left-shoulder"),
    "right_shoulder_pitch_skel": ("lumbar", "right-shoulder"),
    "left_shoulder_roll_skel": ("left-shoulder", "left-elbow"),
    "left_shoulder_yaw_skel": ("left-shoulder", "left-elbow"),
    "right_shoulder_roll_skel": ("right-shoulder", "right-elbow"),
    "right_shoulder_yaw_skel": ("right-shoulder", "right-elbow"),
    "left_elbow_skel": ("left-shoulder", "left-elbow"),
    "right_elbow_skel": ("right-shoulder", "right-elbow"),
    "left_wrist_roll_skel": ("left-elbow", "left-wrist"),
    "left_wrist_pitch_skel": ("left-elbow", "left-wrist"),
    "left_wrist_yaw_skel": ("left-elbow", "left-wrist"),
    "left_hand_roll_skel": ("left-elbow", "left-wrist"),
    "right_wrist_roll_skel": ("right-elbow", "right-wrist"),
    "right_wrist_pitch_skel": ("right-elbow", "right-wrist"),
    "right_wrist_yaw_skel": ("right-elbow", "right-wrist"),
    "right_hand_roll_skel": ("right-elbow", "right-wrist"),
}

LEG_DAMP = {"left_knee_skel", "right_knee_skel", "left_ankle_pitch_skel",
            "right_ankle_pitch_skel", "left_ankle_roll_skel", "right_ankle_roll_skel"}

files = sorted(glob.glob(os.path.join(WORK, "mhr", "f*.npz")))
kp_all, keys = [], []
for f in files:
    z = np.load(f)
    kp_all.append(z["kp3d"])
    keys.append(int(os.path.basename(f)[1:6]))
kp_all = np.stack(kp_all)                              # [N,70,3] camera frame, y-down

# ---- camera-up from the most-extended (standing) frames: up = -(neck-midhip) is y-up already?
mid = 0.5 * (kp_all[:, MHR["left-hip"]] + kp_all[:, MHR["right-hip"]])
torso = kp_all[:, MHR["neck"]] - mid
tsec = np.array(keys) / FPS_SAMPLE
stand = (tsec >= 7.5) & (tsec <= 9.5)                  # upright stand at the dishwasher
up_cam = torso[stand].mean(0)
up_cam /= np.linalg.norm(up_cam)
# rotation taking up_cam -> +Y (world up), leaving an arbitrary yaw (normalized per-pose below)
y = np.array([0.0, 1.0, 0.0])
v = np.cross(up_cam, y); c = float(up_cam @ y); s = np.linalg.norm(v)
if s < 1e-8:
    R_up = np.eye(3)
else:
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R_up = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))

# ---- front/back disambiguation: the Figure robot is faceless and shows the camera its BACK for
# most of the video; SAM resolves it as facing the camera, i.e. a depth-mirrored, L/R-swapped
# estimate. Detect (estimated forward pointing AT the camera) and undo per frame: swap left/right
# keypoint labels and mirror depth about the pelvis plane.
SWAP = list(range(70))
_mhr_names = list(MHR.keys())
for nm, i in list(MHR.items()):
    if nm.startswith("left-"):
        SWAP[i] = MHR[nm.replace("left-", "right-")]
    elif nm.startswith("right-"):
        SWAP[i] = MHR[nm.replace("right-", "left-")]
# flip decision with temporal smoothing + motion vote: the raw per-frame sign flickers at
# side-profile views (forward ~ perpendicular to the camera axis), and a wrong mirror there
# REVERSES the walking stride
fwd_z = np.empty(len(files))
for n in range(len(files)):
    hip_cam = kp_all[n][MHR["left-hip"]] - kp_all[n][MHR["right-hip"]]
    fwd_z[n] = np.cross(hip_cam, -up_cam)[2]
flip = np.array([np.median(fwd_z[max(0, n - 2):n + 3]) < 0 for n in range(len(files))])
pelv = mid + 0.0                                       # camera-frame pelvis (root-relative ok)
vel = np.gradient(pelv, axis=0) * FPS_SAMPLE
speed = np.linalg.norm(vel[:, [0, 2]], axis=1)
for n in range(len(files)):
    if speed[n] > 0.25:                                # walking: forward should match motion
        hip_cam = kp_all[n][MHR["left-hip"]] - kp_all[n][MHR["right-hip"]]
        f3 = np.cross(hip_cam, -up_cam)
        if flip[n]:
            f3 = f3 * np.array([1.0, 1.0, -1.0])
        if np.dot(f3[[0, 2]], vel[n][[0, 2]]) < -0.1 * speed[n] * np.linalg.norm(f3[[0, 2]]):
            flip[n] = not flip[n]
n_flipped = 0

pj_out = np.zeros((len(files), len(NAMES), 3), dtype=np.float32)
yaw_out = np.zeros(len(files), dtype=np.float32)
for n in range(len(files)):
    kp_cam = kp_all[n]
    if flip[n]:                                        # facing the camera per SAM -> mirrored
        kp_cam = kp_cam[SWAP].copy()
        zp = 0.5 * (kp_cam[MHR["left-hip"], 2] + kp_cam[MHR["right-hip"], 2])
        kp_cam[:, 2] = 2 * zp - kp_cam[:, 2]
        n_flipped += 1
    kp = kp_cam @ R_up.T                               # world-up frame
    midhip = 0.5 * (kp[MHR["left-hip"]] + kp[MHR["right-hip"]])

    def hpt(name):
        if name == "midhip":
            return midhip
        if name == "lumbar":
            return midhip + 0.35 * (kp[MHR["neck"]] - midhip)
        return kp[MHR[name]]

    # yaw normalization: hip line (right->left) -> +X.  roty(phi) changes atan2(z,x) by -phi,
    # so phi = +yaw zeroes it (verified below per-frame).
    hip = kp[MHR["left-hip"]] - kp[MHR["right-hip"]]
    yaw = np.arctan2(hip[2], hip[0])                   # angle of hip line vs +X in xz plane
    yaw_out[n] = yaw
    cy, sy = np.cos(yaw), np.sin(yaw)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])

    pose = np.zeros((len(NAMES), 3))
    for i, name in enumerate(NAMES):
        p = PARENT[i]
        if p < 0:
            continue
        seg = SEG.get(name)
        if seg is None:
            d = STAND_DIR[i]                           # keep structural micro-bones as in stand
        else:
            d = (hpt(seg[1]) - hpt(seg[0])) @ Ry.T
            nl = np.linalg.norm(d)
            d = STAND_DIR[i] if nl < 1e-6 else d / nl
        if name in LEG_DAMP:                           # SAM overestimates the robot's leg splay
            d = d.copy(); d[0] *= 0.55
            d /= np.linalg.norm(d)
        pose[i] = pose[p] + BONE_LEN[i] * d
    # G1 reads Figure's habitual knee-flex as a squat (shorter thighs, same absolute pelvis
    # drop): straighten the legs 35% toward the G1 stand, keep the upper body pure video.
    # Blend in the common pelvis-at-origin frame (pose is built with pelvis at the origin).
    for j, name in enumerate(NAMES):
        if "hip" in name or "knee" in name or "ankle" in name or "toe" in name:
            pose[j] = 0.65 * pose[j] + 0.35 * (STAND[j] - STAND[IDX["pelvis_skel"]])
    # ground: lowest foot joint sits on the stand-pose floor
    feet = [IDX["left_toe_base"], IDX["right_toe_base"],
            IDX["left_ankle_roll_skel"], IDX["right_ankle_roll_skel"]]
    pose[:, 1] += FLOOR_Y - pose[feet, 1].min()
    pose[:, [0, 2]] -= pose[IDX["pelvis_skel"], [0, 2]]
    pj_out[n] = pose

# self-check: residual hip yaw of the normalized poses must be ~0
_res = []
for n in range(0, len(files), 37):
    hp = pj_out[n, IDX["left_hip_pitch_skel"]] - pj_out[n, IDX["right_hip_pitch_skel"]]
    _res.append(abs(np.degrees(np.arctan2(hp[2], hp[0]))))
print(f"[retarget] residual hip yaw after normalization: max {max(_res):.1f} deg")

np.savez_compressed(os.path.join(WORK, "g1_anchors.npz"),
                    pj=pj_out, yaw=yaw_out, t=np.array(keys) / FPS_SAMPLE,
                    pelvis_cam=mid.astype(np.float32))
print(f"[retarget] front/back flips undone: {n_flipped}/{len(files)}")
print(f"[retarget] {len(files)} poses -> g1_anchors.npz  "
      f"(pelvis h range {pj_out[:, 0, 1].min():.2f}..{pj_out[:, 0, 1].max():.2f} m)")
