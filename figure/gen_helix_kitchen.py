#!/usr/bin/env python
"""Reproduce the Figure 'Introducing Helix 02' kitchen scenario (outputs/figure/figure.mp4) with G1,
using Kimodo as a CONSTRAINT-DRIVEN inbetweener. See design/figure.md.

v3 - anchor-pose keyframing, NO text interface:
  * cfg_weight=[0.0, 2.0] (separated CFG): text guidance weight is ZERO - the prompt is a dead
    placeholder; all content comes from constraints.
  * A small library of ANCHOR POSES (clean stand, half-bend incline, squat-at-rack, reach-up,
    put-down) is harvested once into outputs/figure/anchors/ and then placed at scene-correct
    times/yaws as FullBodyConstraintSet keyframes ("in front of the furniture -> inclined over it
    -> back to stand"), with Root2D station pins + smooth heading profiles between them. Kimodo
    inbetweens the anchors.
  * Door open / rack pull+push / door close use hand/foot EE keyframe arcs with geometry that
    respects the scene: hands never cross the counter plane (dz <= 0.45), the fold-down door edge
    arcs TOWARD the robot as it opens, the closing foot lift stays <= 0.24 m.
  * The robot yaws toward the dishwasher (right-front) while working it, like the video.

Segments are stand-bookended, chained by SE(2) alignment + crossfade. Per-segment qpos cached in
outputs/figure/segs/ (CLI args force regeneration: segment names or `all`).

Run from repo root (env per design/gait.md section 6):
    CUDA_VISIBLE_DEVICES=0 TEXT_ENCODERS_DIR=... HF_HOME=... PYTHONPATH=. \
        python -m figure.gen_helix_kitchen [all|<segment> ...]
"""
import json
import os

import numpy as np
import torch

from kimodo.constraints import (FullBodyConstraintSet, LeftFootConstraintSet,
                                LeftHandConstraintSet,
                                RightFootConstraintSet, RightHandConstraintSet,
                                Root2DConstraintSet)
from kimodo.exports.mujoco import MujocoQposConverter
from kimodo.model.load_model import load_model
from kimodo.tools import seed_everything

_PKG = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_PKG)
OUT = os.path.join(REPO, "outputs", "figure")
GAIT_OUT = os.path.join(REPO, "outputs", "gait")
ANCH = os.path.join(OUT, "anchors")
STEM = "helix_kitchen"
CROSS = 10
CFG = [0.0, 2.0]                               # (text, constraint): TEXT OFF
PROMPT = "a person"                            # dead placeholder (w_text = 0)
STANDOFF = 0.58                                # work-spot distance to the counter front
LAT = 0.55                                     # dishwasher center offset: robot's front-LEFT
YAW_DW = 0.60                                  # rad (~34 deg): body yaw toward the dishwasher
# (video chirality: the robot stands in the corner pocket BETWEEN the dishwasher (its left) and
#  the corner/range (its right), working leftward - LAT/yaw keep the fold-down door (0.53 m) and
#  slid-out rack clear of the legs; the robot also steps BACK while the door swings down)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = load_model("kimodo-g1-rp", device=device, default_family="Kimodo")
skel = model.skeleton; fps = float(model.fps)
conv = MujocoQposConverter(skel)

_ref = np.load(os.path.join(GAIT_OUT, "go_forward.npz"))
STAND_PJ = torch.tensor(_ref["posed_joints"][0], dtype=torch.float32, device=device)      # [34,3]
STAND_GR = torch.tensor(_ref["global_rot_mats"][0], dtype=torch.float32, device=device)   # [34,3,3]
RFOOT = list(skel.right_foot_joint_names)
LFOOT = list(skel.left_foot_joint_names)
LHAND = list(skel.left_hand_joint_names)
RHAND = list(skel.right_hand_joint_names)
ROOT = skel.root_idx
b = skel.bone_index
hL, hR = b["left_hand_roll_skel"], b["right_hand_roll_skel"]


# ---------------------------------------------------------------- low-level generation
def qpos_of(out):
    q = conv.dict_to_qpos(out, device)
    q = q.cpu().numpy() if torch.is_tensor(q) else np.asarray(q)
    return q[0] if q.ndim == 3 else q


def gen(cons, nf, seed=0, prompt=PROMPT, cfg=CFG, want=("qpos",)):
    seed_everything(seed)
    out = model([prompt], [nf], constraint_lst=cons, num_denoising_steps=100, num_samples=1,
                multi_prompt=False, post_processing=False, cfg_type="separated",
                cfg_weight=list(cfg), return_numpy=True)
    res = []
    for w in want:
        if w == "qpos":
            res.append(qpos_of(out))
        else:
            v = np.asarray(out[w])
            res.append(v[0] if v.ndim in (4, 5) else v)     # strip batch dim
    return res[0] if len(res) == 1 else res


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def station(nf, theta=None):
    """Root2D pin at the origin for all frames; heading follows the yaw profile theta [nf]."""
    f = torch.arange(nf, device=device)
    root2d = torch.zeros(nf, 2, dtype=torch.float32, device=device)
    th = np.zeros(nf) if theta is None else theta
    heading = torch.tensor(np.stack([np.cos(th), -np.sin(th)], 1), dtype=torch.float32, device=device)
    return Root2DConstraintSet(skel, f, root2d, global_root_heading=heading).to(device)


def yaw_profile(nf, theta, t_in=0.5, t_hold_end=None):
    """0 -> theta (by ~1s) -> hold -> back to 0 over the last second."""
    t = np.arange(nf) / fps
    T = nf / fps
    t_hold_end = T - 1.2 if t_hold_end is None else t_hold_end
    up = smoothstep((t - t_in) / 0.7)
    down = smoothstep((t - t_hold_end) / 0.9)
    return theta * (up - down)


def _roty(a):
    c, s = np.cos(a), np.sin(a)
    return torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float32, device=device)


def pose_yawed(pj, gr, yaw):
    """Rotate an anchor pose about +Y (up) around its root ground point."""
    R = _roty(yaw)
    p = pj.clone()
    r0 = p[ROOT].clone(); r0[1] = 0.0
    p = (p - r0) @ R.T + r0
    p[:, [0, 2]] -= p[ROOT, [0, 2]].clone()    # root back over the station point
    g = R @ gr
    return p, g


TLOG = []                                       # (frame, pj[34,3] kimodo local) per FullBody pin


def _tlog(frames, p):
    pn = p.detach().cpu().numpy()
    for fr in (frames.tolist() if torch.is_tensor(frames) else list(frames)):
        TLOG.append((int(fr), pn.copy()))


def fullbody_at(frames, pj, gr, yaw=0.0):
    p, g = pose_yawed(pj, gr, yaw)
    _tlog(frames, p)
    f = torch.tensor(frames, device=device, dtype=torch.long)
    return FullBodyConstraintSet(skel, f, p.unsqueeze(0).repeat(len(f), 1, 1),
                                 g.unsqueeze(0).repeat(len(f), 1, 1, 1)).to(device)


def ee_keys(cls, joints, keys, yaw=0.0):
    """EE keyframes: stand pose with the EE chain displaced by (t, dx_left, dy_up, dz_fwd) in the
    (possibly yawed) body frame; the whole target pose is yawed so it matches the heading."""
    frames, poss, rots = [], [], []
    for ts, dx, dy, dz in keys:
        pose = STAND_PJ.clone()
        for j in joints:
            pose[b[j], 0] += dx
            pose[b[j], 1] += dy
            pose[b[j], 2] += dz
        p, g = pose_yawed(pose, STAND_GR, yaw)
        frames.append(int(round(ts * fps))); poss.append(p); rots.append(g)
    return cls(skel, torch.tensor(frames, device=device), torch.stack(poss),
               torch.stack(rots), None).to(device)


# ---------------------------------------------------------------- anchor library
# Harvested ONCE from text-styled probe motions (offline asset authoring; the demo pipeline itself
# never uses text). Selection rules pick the most articulated natural frame of each probe.
ANCHOR_SPECS = {
    "squat":    ("a person squats down, picks something up from the ground, and stands back up", 5.0, 0),
    "reach":    ("a person reaches up high with both hands to place an object on a shelf", 4.5, 0),
    "putdown":  ("a person places an object down on a table in front of them", 4.0, 3),
}


def harvest_anchors():
    os.makedirs(ANCH, exist_ok=True)
    need = [k for k in list(ANCHOR_SPECS) + ["halfbend"] if not os.path.exists(os.path.join(ANCH, k + ".npz"))]
    if not need:
        return
    print(f"[helix] harvesting anchors: {need}")
    for name in [n for n in need if n != "halfbend"] + (["squat"] if "halfbend" in need and "squat" not in need else []):
        prompt, dur, seed = ANCHOR_SPECS[name]
        nf = int(round(dur * fps))
        pj, gr = gen([station(nf)], nf, seed=seed, prompt=prompt, cfg=[2.0, 2.0],
                     want=("posed_joints", "global_rot_mats"))
        drop = pj[0, ROOT, 1] - pj[:, ROOT, 1]
        handy = (pj[:, hL, 1] + pj[:, hR, 1]) / 2
        fwd = (pj[:, hL, 2] + pj[:, hR, 2]) / 2 - pj[:, ROOT, 2]
        if name == "squat":
            k = int(np.argmax(drop))
        elif name == "reach":
            k = int(np.argmax(handy))
        else:
            k = int(np.argmax(fwd * (handy < 0.9)))
        np.savez(os.path.join(ANCH, name + ".npz"), pj=pj[k], gr=gr[k])
        print(f"[helix]   anchor {name}: frame {k} (drop={drop[k]*100:.0f}cm handY={handy[k]*100:.0f}cm fwd={fwd[k]*100:.0f}cm)")
        if name == "squat" and "halfbend" in need:
            half = int(np.argmin(np.abs(drop[:k] - 0.45 * drop[k])))
            np.savez(os.path.join(ANCH, "halfbend.npz"), pj=pj[half], gr=gr[half])
            print(f"[helix]   anchor halfbend: frame {half} (drop={drop[half]*100:.0f}cm)")


def anchor(name):
    if name == "stand":
        return STAND_PJ, STAND_GR
    z = np.load(os.path.join(ANCH, name + ".npz"))
    return (torch.tensor(z["pj"], dtype=torch.float32, device=device),
            torch.tensor(z["gr"], dtype=torch.float32, device=device))


# ---------------------------------------------------------------- segments (constraints only)
def frames_at(*ts):
    return [int(round(t * fps)) for t in ts]


def end_stand(nf, tail=0.35):
    """Stand anchor pinned over the last `tail` seconds -> clean square stand at the seam."""
    k = int(round(tail * fps))
    return fullbody_at(list(range(nf - k, nf)), STAND_PJ, STAND_GR, 0.0)


def seg_walk(dist, vpeak, t_pre=1.0, t_post=1.0, t_ramp=1.2, back=False, seed=0, xbow=0.0,
             end_pose=None, start_pose=None, approach=0.0, path_xy=None):
    """approach (rad): walk direction relative to the ARRIVAL facing - the video robot
    approaches the pocket diagonally (~0.7 rad) before squaring up to the counter.
    path_xy: MEASURED world-frame pelvis path [(X, Y) at 30 fps] (VGGT lift, lift_traj());
    replaces the synthetic velocity/arc profile - root2d, heading, and pin positions all
    follow the measurement. dist/vpeak/t_ramp/approach are ignored in that mode."""
    if path_xy is not None:
        npre, npost = int(round(t_pre * fps)), int(round(t_post * fps))
        pw = np.concatenate([np.repeat(path_xy[:1], npre, 0), path_xy,
                             np.repeat(path_xy[-1:], npost, 0)], 0)
        nf = len(pw)
        x, z = pw[:, 1].copy(), pw[:, 0].copy()          # kimodo x=world Y, z=world X
        d = np.gradient(pw, axis=0)
        spd = np.hypot(d[:, 0], d[:, 1])
        th = np.arctan2(d[:, 1], d[:, 0])                # world yaw == kimodo heading angle
        # hold heading through slow/hold phases (tangent is noise there)
        moving = spd * fps > 0.12
        if moving.any():
            first, last = np.argmax(moving), nf - 1 - np.argmax(moving[::-1])
            th[:first] = th[first]; th[last + 1:] = th[last]
            th = np.unwrap(th)
            k = int(0.3 * fps) | 1
            th = np.convolve(np.pad(th, k // 2, mode="edge"), np.ones(k) / k, "valid")
        approach = float(th[0])
    else:
        t_cruise = max(0.0, (dist - vpeak * t_ramp) / vpeak)
        T = t_pre + t_ramp + t_cruise + t_ramp + t_post
        nf = int(round(T * fps)); t = np.arange(nf) / fps
        v = np.zeros(nf)
        m = (t >= t_pre) & (t < t_pre + t_ramp)
        v[m] = vpeak * smoothstep((t[m] - t_pre) / t_ramp)
        m = (t >= t_pre + t_ramp) & (t < t_pre + t_ramp + t_cruise)
        v[m] = vpeak
        m = (t >= t_pre + t_ramp + t_cruise) & (t < T - t_post)
        v[m] = vpeak * smoothstep((T - t_post - t[m]) / t_ramp)
        z = np.cumsum(v) / fps * (-1.0 if back else 1.0)
        # xbow: lateral bow (robot's left, +X) peaking early - a diagonal entry like the video's
        # walk around the dining table into the pocket; starts and ends on the axis
        # ARC approach (video: enter nearly parallel to the back wall, straighten into the pocket):
        # heading decays from `approach` to 0 along the walk; the path integrates the heading.
        prog = np.clip(z / (z[-1] if abs(z[-1]) > 1e-6 else 1.0), 0.0, 1.0)
        th = approach * (1.0 - smoothstep(prog / 0.35))
        step = np.diff(z, prepend=0.0)
        x = np.cumsum(step * np.sin(th))
        z = np.cumsum(step * np.cos(th))
    f = torch.arange(nf, device=device)
    root2d = torch.tensor(np.stack([x, z], 1), dtype=torch.float32, device=device)
    th[0] = approach                                 # frame-0 heading defines the chain seam
    heading = torch.tensor(np.stack([np.cos(th), -np.sin(th)], 1),
                           dtype=torch.float32, device=device)
    cons = [Root2DConstraintSet(skel, f, root2d, global_root_heading=heading).to(device)]
    # pin the arrival pose over the deceleration hold: with the text prior off, unconstrained
    # walks collapse/kneel at the end instead of stopping. Default target is a clean stand;
    # end_pose=(pj, yaw) targets a VIDEO pose instead (SAM-3D skeleton at the reference frame),
    # so the walk is generated as an approach INTO the video's arrival stance.
    k = int(round(min(t_post * 0.8, 0.6) * fps))
    if end_pose is not None:
        p, g = pose_yawed(end_pose[0], STAND_GR, end_pose[1])
        heading[nf - k:, 0] = float(np.cos(end_pose[1]))
        heading[nf - k:, 1] = float(-np.sin(end_pose[1]))
        cons[0] = Root2DConstraintSet(skel, f, root2d, global_root_heading=heading).to(device)
    else:
        p, g = pose_yawed(STAND_PJ, STAND_GR, 0.0)
    p = p.clone()
    p[:, 0] += float(x[-1]); p[:, 2] += float(z[-1])     # arrive AT the endpoint, not the origin
    ff = torch.arange(nf - k, nf, device=device)
    _tlog(ff, p)
    cons.append(FullBodyConstraintSet(skel, ff, p.unsqueeze(0).repeat(k, 1, 1),
                                      g.unsqueeze(0).repeat(k, 1, 1, 1)).to(device))
    if start_pose is not None:                           # video skeleton at the walk's first frame
        sp, sg = pose_yawed(start_pose[0], STAND_GR, approach)
        sp = sp.clone()
        sp[:, 0] += float(x[0]); sp[:, 2] += float(z[0])  # measured paths don't start at origin
        k0 = int(round(0.35 * fps))
        f0 = torch.arange(0, k0, device=device)
        _tlog(f0, sp)
        cons.append(FullBodyConstraintSet(skel, f0, sp.unsqueeze(0).repeat(k0, 1, 1),
                                          sg.unsqueeze(0).repeat(k0, 1, 1, 1)).to(device))
    return gen(cons, nf, seed=seed)


def seg_anchored(dur, anchors, yaw=0.0, seed=0, stand_from=None):
    """Station-pinned segment: yaw toward the target, hit FullBody anchors, return to stand.
    anchors: list of (t0, t1, name) - the pose held over [t0, t1]. A dense stand anchor covers
    [stand_from, end] (default: the last 1.2 s) - with the text prior off, the model does not
    return upright on its own."""
    nf = int(round(dur * fps))
    T = nf / fps
    stand_from = T - 1.2 if stand_from is None else stand_from
    th = yaw_profile(nf, yaw, t_hold_end=stand_from - 0.7)
    cons = [station(nf, th)]
    for t0, t1, name in anchors:
        pj, gr = anchor(name)
        ff = list(range(int(round(t0 * fps)), min(int(round(t1 * fps)) + 1, nf), 3))
        cons.append(fullbody_at(ff, pj, gr, yaw))
    cons.append(fullbody_at(list(range(int(round(stand_from * fps)), nf, 3)), STAND_PJ, STAND_GR, 0.0))
    return gen(cons, nf, seed=seed)


# ---------------------------------------------------------------- video-derived anchors (v4)
# SAM-3D-Body poses lifted from figure.mp4 at 2 fps and retargeted to G1-34 by direction transfer
# (figure/pose_video.py + figure/pose_retarget.py -> outputs/figure/pose/g1_anchors.npz).
_VID = None

# design-frame work pocket: where the dw-work stance sits in the chain/scene frame (build_scene
# measures the same value from the chained targets). Anchors the measured lift trajectory.
PW_DESIGN = np.array([1.96, 0.38])
_LIFT = None


def lift_traj(t0, t1):
    """Measured pelvis path from the VGGT-Omega lift (outputs/figure/pose/lift3d.npz), mapped
    into the scene frame by pinning the dw-work median position to the design pocket. Both
    frames share +X = work facing and z-up, so the map is a pure translation."""
    global _LIFT
    if _LIFT is None:
        z = np.load(os.path.join(OUT, "pose", "fit3d.npz"))   # rigid fitter output
        mid = 0.5 * (z["joints_w"][:, 5, :2] + z["joints_w"][:, 6, :2])   # lhip=5, rhip=6
        _LIFT = (mid.astype(np.float64), z["t"], z["ok"])
    mid, tt, ok = _LIFT
    w = ok & (tt >= 8.0) & (tt <= 13.5)                  # same window as the anchor yaw ref
    pocket_l = np.median(mid[w], 0)
    m = ok & (tt >= t0) & (tt <= t1)
    grid = np.arange(t0, t1, 1.0 / fps)
    gx = np.interp(grid, tt[m], mid[m, 0])
    gy = np.interp(grid, tt[m], mid[m, 1])
    k = int(0.7 * fps) | 1                               # ~0.7 s moving average: kill lift jitter
    ker = np.ones(k) / k
    gx = np.convolve(np.pad(gx, k // 2, mode="edge"), ker, "valid")
    gy = np.convolve(np.pad(gy, k // 2, mode="edge"), ker, "valid")
    path = np.stack([gx, gy], 1) - pocket_l + PW_DESIGN
    # chord alignment: the ARRIVAL (close to camera) is well measured, but the entry is both
    # depth-warped and rotated by the lift frame's work-facing ambiguity (same ~90 deg the
    # anchor yaws needed). Keep the measured arrival point + path SHAPE (arc-length profile,
    # lateral wobble) and rotate the whole path about the arrival so the chord runs down the
    # aisle axis +X (entry in front of the range, like the video).
    chord = path[-1] - path[0]
    phi = -np.arctan2(chord[1], chord[0])
    c, s = np.cos(phi), np.sin(phi)
    rel = path - path[-1]
    path = path[-1] + rel @ np.array([[c, -s], [s, c]]).T
    # depth cap vs the counter face (fitted path still wanders in the weak depth axis): never
    # deeper than the work stance minus a stride margin, released over the last 0.6 s
    cap = np.full(len(path), PW_DESIGN[0] - 0.05)
    rel_n = int(0.6 * fps)
    cap[-rel_n:] += np.linspace(0.0, 0.4, rel_n)
    path[:, 0] = np.minimum(path[:, 0], cap)
    path -= path[0]                                      # Kimodo generates roots from the origin;
                                                         # the scene re-anchors to the pocket anyway
    print(f"[lift_traj] {t0}-{t1}s: start {path[0].round(2)} end {path[-1].round(2)} "
          f"len {np.sum(np.hypot(*np.diff(path, axis=0).T)):.2f} chord_rot {np.degrees(phi):+.0f}deg")
    return path


def _vid():
    global _VID
    if _VID is None:
        z = np.load(os.path.join(OUT, "pose", "g1_anchors.npz"))
        yaw = np.unwrap(z["yaw"].astype(np.float64))
        ref = np.median(yaw[(z["t"] >= 8.0) & (z["t"] <= 13.5)])   # dw work: body square to counter
        # poses are normalized to face +Z; a facing of (yaw-ref) relative to the counter-facing
        # reference is restored by roty(ref-yaw) (roty(phi) moves hip atan2 by -phi).
        # +pi/2: the ref window (dw work) has the body angled ALONG the counter, not square to
        # it - verified against the video (user calibration: rotate G1 CCW 90 deg)
        _VID = {"pj": z["pj"], "t": z["t"], "dyaw": (ref - yaw) + np.pi / 2}
    return _VID


def _hand_leg_clearance(pj, min_gap=0.12):
    """Monocular depth errors can sink a video hand INTO the same-side shin or the pelvis; push
    the hand chain forward (+Z) until every offending hand joint clears the body by min_gap."""
    pj = pj.copy()
    for side in ("left", "right"):
        hand = [b[f"{side}_wrist_roll_skel"], b[f"{side}_wrist_pitch_skel"],
                b[f"{side}_wrist_yaw_skel"], b[f"{side}_hand_roll_skel"]]
        leg = [b[f"{side}_knee_skel"], b[f"{side}_ankle_pitch_skel"],
               b[f"{side}_ankle_roll_skel"], b[f"{side}_toe_base"]]
        core = [b["pelvis_skel"], b["left_hip_pitch_skel"], b["right_hip_pitch_skel"]]
        push = 0.0
        for h in hand:
            for g in core:                      # pelvis/hips: no height gate
                if np.linalg.norm(pj[h] - pj[g]) < min_gap:
                    push = max(push, (pj[g, 2] + min_gap) - pj[h, 2])
            if pj[h, 1] > 0.55:
                continue
            for g in leg:
                if np.linalg.norm(pj[h] - pj[g]) < min_gap:
                    push = max(push, (pj[g, 2] + min_gap) - pj[h, 2])
        if push > 0:
            pj[hand, 2] += push
    return pj


def seg_video(dur, r0, r1, step=0.6, zpath=None, stand_tail=0.8, seed=0, yaw_gain=1.0):
    """Segment driven by VIDEO poses: FullBody anchors every `step` s resampled from the
    retargeted SAM-3D-Body track over video window [r0, r1]; heading profile measured from the
    video hip line (relative to the final-stand reference). Station-pinned (optional zpath gives
    the root a scripted in/out path, e.g. the door-swing retreat); stand-bookended for chaining."""
    v = _vid()
    nf = int(round(dur * fps))
    tseg = np.arange(nf) / fps
    tvid = r0 + tseg * (r1 - r0) / dur
    dyaw = yaw_gain * np.interp(tvid, v["t"], v["dyaw"])
    dyaw *= smoothstep(tseg / 0.5) * smoothstep((dur - stand_tail + 0.3 - tseg) / 0.6)
    cons = [station(nf, dyaw)]
    if zpath is not None:
        f = torch.arange(nf, device=device)
        root2d = torch.tensor(np.stack([np.zeros(nf), zpath(tseg)], 1),
                              dtype=torch.float32, device=device)
        heading = torch.tensor(np.stack([np.cos(dyaw), -np.sin(dyaw)], 1),
                               dtype=torch.float32, device=device)
        cons[0] = Root2DConstraintSet(skel, f, root2d, global_root_heading=heading).to(device)
    for ts in np.arange(0.4, dur - stand_tail, step):
        k = int(np.argmin(np.abs(v["t"] - (r0 + ts * (r1 - r0) / dur))))
        pj = torch.tensor(_hand_leg_clearance(v["pj"][k]), dtype=torch.float32, device=device)
        fr = int(round(ts * fps))
        cons.append(fullbody_at([fr], pj, STAND_GR, float(dyaw[fr])))
    cons.append(fullbody_at(list(range(int(round((dur - stand_tail) * fps)), nf, 3)),
                            STAND_PJ, STAND_GR, 0.0))
    return gen(cons, nf, seed=seed)



def vid_pose(tv):
    """Retargeted SAM-3D-Body pose + measured heading at video time tv (clearance-corrected)."""
    v = _vid()
    k = int(np.argmin(np.abs(v["t"] - tv)))
    pj = torch.tensor(_hand_leg_clearance(v["pj"][k]), dtype=torch.float32, device=device)
    return pj, float(v["dyaw"][k])


def seg_open_door(dur=6.0, seed=0):
    """Video beat: lean, grab the door's near top corner, then STEP BACK while the door folds down
    toward the robot (hands follow its edge), ending upright out of the door's arc."""
    nf = int(round(dur * fps))
    t = np.arange(nf) / fps
    th = yaw_profile(nf, YAW_DW, t_hold_end=3.8)
    zpath = -0.35 * smoothstep((t - 1.8) / 1.0)          # back-step during the swing
    f = torch.arange(nf, device=device)
    root2d = torch.tensor(np.stack([np.zeros(nf), zpath], 1), dtype=torch.float32, device=device)
    heading = torch.tensor(np.stack([np.cos(th), -np.sin(th)], 1), dtype=torch.float32, device=device)
    cons = [Root2DConstraintSet(skel, f, root2d, global_root_heading=heading).to(device)]
    hb, _ = anchor("halfbend")
    cons.append(fullbody_at(frames_at(1.3), hb, anchor("halfbend")[1], YAW_DW))
    # hand keys in the yawed body frame: grab the near top corner (lean supplies the reach), then
    # follow the falling edge down-and-in while the body retreats
    keys = [(1.1, 0.0, -0.08, 0.58), (1.5, 0.0, -0.10, 0.56), (2.1, 0.0, -0.22, 0.40),
            (2.6, 0.0, -0.32, 0.26)]
    cons.append(ee_keys(LeftHandConstraintSet, LHAND, keys, yaw=YAW_DW))
    cons.append(ee_keys(RightHandConstraintSet, RHAND, keys, yaw=YAW_DW))
    # stand at the retreated point for the last second
    p, g = pose_yawed(STAND_PJ, STAND_GR, 0.0)
    p = p.clone(); p[:, 2] += -0.35
    ff = torch.arange(int(round(4.6 * fps)), nf, 3, device=device)
    cons.append(FullBodyConstraintSet(skel, ff, p.unsqueeze(0).repeat(len(ff), 1, 1),
                                      g.unsqueeze(0).repeat(len(ff), 1, 1, 1)).to(device))
    return gen(cons, nf, seed=seed)


def seg_rack(dur, out_dir, seed=0):
    """Pull (out_dir=-1) or push (out_dir=+1) the bottom rack: half-bend, hands at rack height."""
    nf = int(round(dur * fps))
    th = yaw_profile(nf, YAW_DW, t_hold_end=dur - 1.6)
    hb, hbg = anchor("halfbend")
    if out_dir < 0:
        keys = [(1.2, 0.0, -0.30, 0.40), (2.0, 0.0, -0.30, 0.24)]
    else:
        keys = [(1.2, 0.0, -0.30, 0.24), (2.0, 0.0, -0.30, 0.40)]
    cons = [station(nf, th),
            fullbody_at(frames_at(1.0), hb, hbg, YAW_DW),
            ee_keys(LeftHandConstraintSet, LHAND, keys, yaw=YAW_DW),
            ee_keys(RightHandConstraintSet, RHAND, keys, yaw=YAW_DW),
            fullbody_at(list(range(int(round((dur - 1.0) * fps)), nf, 3)), STAND_PJ, STAND_GR, 0.0)]
    return gen(cons, nf, seed=seed)


def seg_place_ctr(dur=4.4, yaw=-0.10, seed=0):
    """Two-hand put-down onto the countertop: raise, extend over the top, set down, retract."""
    nf = int(round(dur * fps))
    th = yaw_profile(nf, yaw, t_hold_end=dur - 1.6)
    keys = [(1.0, 0.0, 0.10, 0.30), (1.6, 0.0, 0.06, 0.42), (2.1, 0.0, -0.02, 0.42),
            (2.7, 0.0, -0.06, 0.16)]
    cons = [station(nf, th),
            ee_keys(LeftHandConstraintSet, LHAND, keys, yaw=yaw),
            ee_keys(RightHandConstraintSet, RHAND, keys, yaw=yaw),
            fullbody_at(list(range(int(round((dur - 1.0) * fps)), nf, 3)), STAND_PJ, STAND_GR, 0.0)]
    return gen(cons, nf, seed=seed)


def seg_close_door(dur=5.2, seed=0):
    """LEFT-foot lift under the open door tip (door is at the robot's left); plant, settle."""
    nf = int(round(dur * fps))
    th = yaw_profile(nf, 0.35, t_hold_end=3.4)
    keys = [(1.2, 0.0, 0.03, 0.18), (1.7, 0.0, 0.06, 0.34), (2.3, 0.0, 0.22, 0.40),
            (2.9, 0.0, 0.08, 0.16), (3.4, 0.0, 0.0, 0.02)]
    cons = [station(nf, th),
            ee_keys(LeftFootConstraintSet, LFOOT, keys, yaw=0.35),
            fullbody_at(list(range(int(round(4.0 * fps)), nf, 3)), STAND_PJ, STAND_GR, 0.0)]
    return gen(cons, nf, seed=seed)


# ---------------------------------------------------------------- SE(2) chaining
def yaw_of(q):
    w, x, y, z = q[3], q[4], q[5], q[6]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def quat_mul_z(dpsi, q):
    cw, sw = np.cos(dpsi / 2), np.sin(dpsi / 2)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([cw * w - sw * z, cw * x - sw * y, cw * y + sw * x, cw * z + sw * w], 1)


def place(seg, x, y, psi):
    s = seg.copy()
    psi0 = yaw_of(s[0]); xy0 = s[0, :2].copy()
    d = psi - psi0; R = np.array([[np.cos(d), -np.sin(d)], [np.sin(d), np.cos(d)]])
    s[:, :2] = (s[:, :2] - xy0) @ R.T + [x, y]
    s[:, 3:7] = quat_mul_z(d, s[:, 3:7])
    return s


def chain(segments, seam_yaw=0.0, blend=16):
    """Chain at the DESIGNED seam yaw (every segment is authored to start/end facing +X), not the
    measured end yaw: constraint guidance leaves a 10-20 deg per-segment yaw residual which would
    otherwise accumulate across the chain. The mismatch is absorbed in the stand-to-stand
    crossfade (a small stationary pivot, ~0.5 s)."""
    glob, bounds, cursor = None, [], 0
    t_frames, t_pj = [], []
    for name, seg, tf, tp in segments:
        # first segment: keep its authored absolute position (measured paths don't start at
        # the origin); later segments continue from the previous end
        x, y = (float(seg[0][0]), float(seg[0][1])) if glob is None else (glob[-1][0], glob[-1][1])
        psi0 = yaw_of(seg[0]); xy0 = seg[0, :2].copy()
        d = (psi0 if glob is None else seam_yaw) - psi0   # first segment keeps its author frame
        R = np.array([[np.cos(d), -np.sin(d)], [np.sin(d), np.cos(d)]])
        placed = place(seg, x, y, psi0 + d)
        if glob is not None:
            e = glob[-1]
            k = min(blend, len(placed) - 1)
            for i in range(k):
                w = (i + 1) / (k + 1)
                mix = (1 - w) * e + w * placed[i]
                qa, qb = e[3:7], placed[i, 3:7]
                if np.dot(qa, qb) < 0:
                    qb = -qb
                bq = (1 - w) * qa + w * qb
                mix[3:7] = bq / np.linalg.norm(bq)
                placed[i] = mix
        # constraint pins -> world: kimodo (x_left, y_up, z_fwd) maps to qpos ground (X=z, Y=x)
        for fr, pj in zip(tf, tp):
            g = np.stack([pj[:, 2], pj[:, 0]], 1)        # ground plane
            g = (g - xy0) @ R.T + [x, y]
            t_frames.append(cursor + int(fr))
            t_pj.append(np.stack([g[:, 0], g[:, 1], pj[:, 1]], 1))
        bounds.append([name, cursor])
        cursor += len(placed)
        glob = placed if glob is None else np.concatenate([glob, placed])
    targets = (np.array(t_frames, dtype=int),
               np.stack(t_pj) if t_pj else np.zeros((0, 34, 3)))
    return glob, bounds, targets


# ---------------------------------------------------------------- storyboard (video-faithful)
# unload arc: halfbend on the way down, squat hold, halfbend on the way up, dense stand at the end
UNLOAD = [(1.2, 1.2, "halfbend"), (1.8, 2.4, "squat"), (3.0, 3.0, "halfbend")]
story = [
    # walk path is SYNTHETIC: the video's walking happens at 2.4-4.0 s, inside the discarded
    # SAM window (first 4 s unreliable), and the measured 4.0-6.5 s remainder is only a settle.
    # Start/end poses are the (valid) video anchors at 4.0 / 6.5 s.
    ("walk_in",    lambda: seg_walk(2.10, 0.85, t_pre=0.3, t_post=0.7, approach=1.15, seed=2,
                                start_pose=vid_pose(4.0), end_pose=vid_pose(6.5))),
    # video window 6-12 s; scripted retreat keeps the legs out of the door's swing arc
    ("open_door",  lambda: seg_video(6.0, 6.0, 12.0,
                                     zpath=lambda t: -0.35 * smoothstep((t - 1.8) / 1.0))),
    ("step_in",    lambda: seg_walk(0.35, 0.25, t_pre=0.5, t_post=0.9, t_ramp=0.6)),
    ("pull_rack",  lambda: seg_rack(4.0, -1)),
    ("unload_1",   lambda: seg_anchored(5.2, UNLOAD, yaw=YAW_DW, stand_from=3.8)),
    ("place_ctr1", lambda: seg_place_ctr(4.4, yaw=-0.10)),
    ("unload_2",   lambda: seg_anchored(5.2, UNLOAD, yaw=YAW_DW, seed=1, stand_from=3.8)),
    ("place_up",   lambda: seg_anchored(4.8, [(1.7, 2.3, "reach")], yaw=0.15, stand_from=3.4)),
    ("unload_3",   lambda: seg_anchored(5.0, [(1.5, 2.4, "halfbend")], yaw=YAW_DW, seed=2, stand_from=3.4)),
    ("place_ctr2", lambda: seg_place_ctr(4.4, yaw=-0.10, seed=1)),
    ("push_rack",  lambda: seg_rack(4.0, +1)),
    ("step_back",  lambda: seg_walk(0.35, 0.25, t_pre=0.5, t_post=0.9, t_ramp=0.6, back=True)),
    ("close_door", lambda: seg_close_door(5.2)),
    ("final_stand", lambda: seg_anchored(3.0, [], yaw=0.0, stand_from=0.4)),
]

if __name__ == "__main__":
    import sys
    force = set(sys.argv[1:])
    SEGS = os.path.join(OUT, "segs")
    os.makedirs(SEGS, exist_ok=True)
    harvest_anchors()
    print("[helix] generating segments (text OFF, constraints only)...")
    segments = []
    for name, fn in story:
        cache = os.path.join(SEGS, name + ".csv")
        tcache = os.path.join(SEGS, name + "_targets.npz")
        if os.path.exists(cache) and name not in force and "all" not in force:
            seg = np.loadtxt(cache, delimiter=",")
            if os.path.exists(tcache):
                tz = np.load(tcache)
                tf, tp = tz["tf"], tz["tp"]
            else:
                tf, tp = np.zeros(0, dtype=int), np.zeros((0, 34, 3))
            src = "cache"
        else:
            TLOG.clear()
            seg = fn()
            tf = np.array([fr for fr, _ in TLOG], dtype=int)
            tp = (np.stack([pp for _, pp in TLOG])
                  if TLOG else np.zeros((0, 34, 3)))
            np.savetxt(cache, seg, delimiter=",")
            np.savez_compressed(tcache, tf=tf, tp=tp)
            src = "gen"
        segments.append((name, seg, tf, tp))
        print(f"[helix]   {name:11s} {len(seg):4d} frames ({len(seg)/fps:.1f}s) "
              f"[{src}, {len(tf)} pins]")

    qpos, bounds, targets = chain(segments)
    np.savez_compressed(os.path.join(OUT, STEM + "_targets.npz"),
                        frames=targets[0], pj=targets[1])
    np.savetxt(os.path.join(OUT, STEM + ".csv"), qpos, delimiter=",")
    json.dump({"fps": fps, "bounds": bounds}, open(os.path.join(OUT, STEM + "_bounds.json"), "w"), indent=1)
    print(f"[helix] chained {len(qpos)} frames = {len(qpos)/fps:.1f}s -> {STEM}.csv")

    try:
        from figure.build_scene import build
    except ImportError:
        from build_scene import build
    build()
    print("DONE_HELIX_GEN")
