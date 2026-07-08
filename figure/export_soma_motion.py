#!/usr/bin/env python
"""Export MotionRecon fit3d (+ optional mesh) to Kimodo's SOMA-77 motion npz format.

Produces the exact schema of a Kimodo motion seed (local_rot_mats, global_rot_mats,
posed_joints, root_positions, smooth_root_pos, foot_contacts, global_root_heading), so the
reconstruction can be consumed anywhere Kimodo motions go (stitching, constraints, replay).

Method: direction transfer onto the SOMA-77 rig. Torso joints get full two-vector frames
(hip line + spine for Hips, shoulder line + spine for Chest, slerp between for Spine1/2;
neck/head slerp toward a nose-up frame). Limb joints get minimal rotations aligning each
rest bone direction to the measured fit3d bone direction (wrist/toe globals follow their
parent - no twist source in an 18-joint fit). Fingers come from the rig's relaxed rest
pose. posed_joints/global_rot_mats are produced by the rig's own FK from the exported
locals, so the file is kinematically self-consistent by construction.

Conventions (validated against a reference seed + the rig's rest pose):
    kimodo frame: Y-up, +Z forward, +X left  <-  MR world: Z-up, +X forward, +Y left
    scale: standing pelvis height 0.72 (MR world) -> rig rest pelvis-above-floor
    root == Hips == mid-hip; heading == hip-line (x, z) normalized
    foot_contacts columns: [LeftFoot, LeftToeBase, LeftToeEnd, Right...] (height+velocity)

Usage (kimodo env):
    MR_OUT=<dir with fit3d.npz> [MR_FPS=30] [MESH_NPZ=<mesh_w.npz>] \
        python figure/export_soma_motion.py <out_motion.npz>
"""
import os
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

from kimodo.skeleton.registry import build_skeleton

KPN = ["nose", "lsho", "rsho", "lelb", "relb", "lhip", "rhip", "lkne", "rkne",
       "lank", "rank", "lbtoe", "lheel", "rbtoe", "rheel", "rwri", "lwri", "neck"]
I = {n: i for i, n in enumerate(KPN)}
PELVIS_H = float(os.environ.get("MR_PELVIS_H", "0"))       # 0 (default) = measure the
# standing pelvis height from the fit itself (pelvis-plateau median) - input fits differ in
# their calibration conventions (stance-window vs plateau anchoring, subject height)
FPS = float(os.environ.get("MR_FPS", "30"))                # source fps (fit3d timebase)
FPS_OUT = float(os.environ.get("FPS_OUT", "30"))           # kimodo motions are 30 fps by
# convention (the seed schema carries no fps field) - resample unless FPS_OUT=0

W2K = np.array([[0.0, 1.0, 0.0],                           # kimodo x (left)  = world y
                [0.0, 0.0, 1.0],                           # kimodo y (up)    = world z
                [1.0, 0.0, 0.0]])                          # kimodo z (fwd)   = world x


def unit(v, axis=-1):
    return v / np.maximum(np.linalg.norm(v, axis=axis, keepdims=True), 1e-9)


def minrot(a, b):
    """[T,3,3] minimal rotations taking unit vectors a -> b (batched Rodrigues)."""
    a, b = unit(a), unit(b)
    v = np.cross(a, b)
    c = np.einsum("ni,ni->n", a, b)
    s2 = np.einsum("ni,ni->n", v, v)
    R = np.tile(np.eye(3), (len(a), 1, 1))
    m = s2 > 1e-12
    K = np.zeros((len(a), 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -v[:, 2], v[:, 1]
    K[:, 1, 0], K[:, 1, 2] = v[:, 2], -v[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -v[:, 1], v[:, 0]
    R[m] = (np.eye(3) + K[m] + K[m] @ K[m] * ((1 - c[m]) / s2[m])[:, None, None])
    return R


def frame_from(x_raw, y_raw):
    """[T,3,3] right-handed frames: x along x_raw (orthogonalized against y), y along y_raw."""
    y = unit(y_raw)
    x = unit(x_raw - np.einsum("ni,ni->n", x_raw, y)[:, None] * y)
    z = np.cross(x, y)
    return np.stack([x, y, z], -1)                         # columns are the axes


def slerp_batch(Ra, Rb, t):
    out = np.empty_like(Ra)
    for n in range(len(Ra)):
        s = Slerp([0, 1], Rotation.from_matrix([Ra[n], Rb[n]]))
        out[n] = s([t]).as_matrix()[0]
    return out


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "motion_export.npz"
    outd = os.environ.get("MR_OUT", "mr_out")
    fz = np.load(os.path.join(outd, "fit3d.npz"))
    Jw, ok = fz["joints_w"][:, :18].astype(np.float64), fz["ok"].astype(bool)

    first, last = np.flatnonzero(ok)[0], np.flatnonzero(ok)[-1]
    Jw = Jw[first:last + 1]
    T = len(Jw)
    for j in range(18):                                    # interpolate interior gaps
        for c in range(3):
            v = Jw[:, j, c]
            m = np.isfinite(v)
            if not m.all():
                Jw[:, j, c] = np.interp(np.arange(T), np.flatnonzero(m), v[m])
    mesh_sel = np.arange(T)                                # mesh frames follow the resample
    if FPS_OUT and abs(FPS - FPS_OUT) > 1e-3:              # resample to the kimodo timebase
        t_src = np.arange(T) / FPS
        t_dst = np.arange(int(t_src[-1] * FPS_OUT) + 1) / FPS_OUT
        Jw = np.stack([[np.interp(t_dst, t_src, Jw[:, j, c]) for c in range(3)]
                       for j in range(18)], 0).transpose(2, 0, 1)
        mesh_sel = np.clip(np.round(t_dst * FPS).astype(int), 0, T - 1)
        print(f"[soma] resampled {T} frames @ {FPS:.3f} fps -> {len(Jw)} @ {FPS_OUT:g} fps")
        T = len(Jw)

    skel = build_skeleton(77)
    names = [n for n, _ in skel.bone_order_names_with_parents]
    parents = {n: p for n, p in skel.bone_order_names_with_parents}
    ji = {n: i for i, n in enumerate(names)}
    rest = skel.fk(torch.eye(3).repeat(1, 77, 1, 1), torch.zeros(1, 3))[1][0].numpy()
    rest_pelvis_h = -rest[:, 1].min()                      # pelvis y=0, lowest joint = floor
    pel_w = 0.5 * (Jw[:, I["lhip"], 2] + Jw[:, I["rhip"], 2])
    if PELVIS_H > 0:
        stand_h = PELVIS_H
    else:                                                  # plateau = subject at full height
        stand_h = float(np.median(pel_w[pel_w > np.percentile(pel_w, 85) * 0.97]))
    s = rest_pelvis_h / stand_h
    off_dir = {n: unit((rest[ji[n]] - rest[ji[parents[n]]])[None])[0]
               for n in names if parents[n] is not None}

    J = np.einsum("ij,ntj->nti", W2K, Jw) * s              # -> kimodo frame, rig scale
    # floor re-anchor: rig floor is y=0; the input fit's floor may sit off zero (different
    # scenario calibrations) - anchor on the lower-heel percentile over the whole clip
    heel_lo = np.minimum(J[:, I["lheel"], 1], J[:, I["rheel"], 1])
    y_floor = float(np.percentile(heel_lo[np.isfinite(heel_lo)], 5))
    J[:, :, 1] -= y_floor

    def d(a, b):
        return J[:, I[b]] - J[:, I[a]]

    midhip = 0.5 * (J[:, I["lhip"]] + J[:, I["rhip"]])
    hipline = J[:, I["lhip"]] - J[:, I["rhip"]]            # rest hip-line = +X (left)
    sholine = J[:, I["lsho"]] - J[:, I["rsho"]]
    spine = J[:, I["neck"]] - midhip

    G = {}
    G["Hips"] = frame_from(hipline, spine)
    G["Chest"] = frame_from(sholine, spine)
    G["Spine1"] = slerp_batch(G["Hips"], G["Chest"], 1 / 3)
    G["Spine2"] = slerp_batch(G["Hips"], G["Chest"], 2 / 3)
    G["Head"] = frame_from(sholine, unit(d("neck", "nose")))
    G["Neck1"] = slerp_batch(G["Chest"], G["Head"], 1 / 3)
    G["Neck2"] = slerp_batch(G["Chest"], G["Head"], 2 / 3)
    G["LeftShoulder"], G["RightShoulder"] = G["Chest"], G["Chest"]
    for S_, sh, el, wr in (("Left", "lsho", "lelb", "lwri"), ("Right", "rsho", "relb", "rwri")):
        G[S_ + "Arm"] = minrot(np.tile(off_dir[S_ + "ForeArm"], (T, 1)), d(sh, el))
        G[S_ + "ForeArm"] = minrot(np.tile(off_dir[S_ + "Hand"], (T, 1)), d(el, wr))
        G[S_ + "Hand"] = G[S_ + "ForeArm"]
    for S_, hp, kn, an, to in (("Left", "lhip", "lkne", "lank", "lbtoe"),
                               ("Right", "rhip", "rkne", "rank", "rbtoe")):
        G[S_ + "Leg"] = minrot(np.tile(off_dir[S_ + "Shin"], (T, 1)), d(hp, kn))
        G[S_ + "Shin"] = minrot(np.tile(off_dir[S_ + "Foot"], (T, 1)), d(kn, an))
        G[S_ + "Foot"] = minrot(np.tile(off_dir[S_ + "ToeBase"], (T, 1)), d(an, to))
        G[S_ + "ToeBase"] = G[S_ + "Foot"]

    loc = skel.relaxed_hands_rest_pose.clone().float().unsqueeze(0).repeat(T, 1, 1, 1)
    for n in G:
        p = parents[n]
        Rl = G[n] if p is None else np.einsum("nji,njk->nik", G[p], G[n])
        loc[:, ji[n]] = torch.from_numpy(Rl).float()

    root = torch.from_numpy(midhip).float()
    g_out, p_out = skel.fk(loc, root)[:2]

    # smooth root (short box, matches the seed's smooth_root_pos character)
    k = 11
    pad = np.pad(midhip, ((k // 2, k // 2), (0, 0)), mode="edge")
    smooth_root = np.stack([np.convolve(pad[:, c], np.ones(k) / k, "valid") for c in range(3)], 1)

    heading = unit(hipline[:, [0, 2]])

    P = p_out.numpy()
    contacts = np.zeros((T, 6), bool)
    rest_floor = rest[:, 1].min()
    for c_, nm in enumerate(["LeftFoot", "LeftToeBase", "LeftToeEnd",
                             "RightFoot", "RightToeBase", "RightToeEnd"]):
        y = P[:, ji[nm], 1]
        v = np.zeros(T)
        v[1:] = np.linalg.norm(np.diff(P[:, ji[nm]], axis=0), axis=1) * (FPS_OUT or FPS)
        # each joint rides at its own height above the sole when planted (ankle ~7 cm),
        # and fits carry cm-level per-joint biases (e.g. camera-far heel rides ~3 cm high),
        # so anchor on the joint's own planted level (p10; robust to penetration spikes,
        # unlike min), floored by the rig rest height
        thr = max((rest[ji[nm], 1] - rest_floor) + 0.04, np.percentile(y, 10) + 0.03)
        contacts[:, c_] = (y < thr) & (v < 0.45)

    np.savez(out_path,
             local_rot_mats=loc.numpy().astype(np.float32),
             global_rot_mats=g_out.numpy().astype(np.float32),
             posed_joints=P.astype(np.float32),
             root_positions=midhip.astype(np.float32),
             smooth_root_pos=smooth_root.astype(np.float32),
             foot_contacts=contacts,
             global_root_heading=heading.astype(np.float32))

    # ---- validation: FK bone directions vs measured
    errs = []
    for a, b, nm in [("lsho", "lelb", "LeftArm->LeftForeArm"), ("lelb", "lwri", "LeftForeArm->LeftHand"),
                     ("rsho", "relb", "RightArm->RightForeArm"), ("relb", "rwri", "RightForeArm->RightHand"),
                     ("lhip", "lkne", "LeftLeg->LeftShin"), ("lkne", "lank", "LeftShin->LeftFoot"),
                     ("rhip", "rkne", "RightLeg->RightShin"), ("rkne", "rank", "RightShin->RightFoot")]:
        child = nm.split("->")[1]
        fkd = unit(P[:, ji[child]] - P[:, ji[parents[child]]])
        med = unit(d(a, b))
        ang = np.degrees(np.arccos(np.clip(np.einsum("ni,ni->n", fkd, med), -1, 1)))
        errs.append((np.median(ang), nm))
    print(f"[soma] {T} frames -> {out_path} (stand_h {stand_h:.3f}w, scale {s:.3f}, "
          f"floor shift {y_floor:+.3f}, rig pelvis {rest_pelvis_h:.3f})")
    print("[soma] FK-vs-measured bone dirs (median deg): "
          + "  ".join(f"{nm.split('->')[0]}:{a:.1f}" for a, nm in errs))
    print(f"[soma] contact fractions: {contacts.mean(0).round(2)}")
    print(f"[soma] pelvis y median {np.median(midhip[:,1]):.3f} (rig standing ~{rest_pelvis_h:.2f}); "
          f"lowest posed joint {P[:,:,1].min():+.3f}")

    mesh_path = os.environ.get("MESH_NPZ", "")
    if mesh_path and os.path.exists(mesh_path):
        mz = np.load(mesh_path)
        V = mz["verts_w"][first:last + 1][mesh_sel].astype(np.float32)
        Vk = np.einsum("ij,ntj->nti", W2K.astype(np.float32), V) * np.float32(s)
        faces_p = os.environ.get("FACES_NPY", os.path.join(
            os.environ.get("DEPTH_WORK", ""), "mhr", "faces.npy"))
        faces = np.load(faces_p) if os.path.exists(faces_p) else np.zeros((0, 3), np.int32)
        np.savez_compressed(os.path.splitext(out_path)[0] + "_mesh.npz",
                            verts=Vk.astype(np.float16), faces=faces,
                            t=np.arange(T) / (FPS_OUT or FPS))
        print(f"[soma] mesh track: {Vk.shape} + faces {faces.shape} -> "
              f"{os.path.splitext(out_path)[0]}_mesh.npz")


if __name__ == "__main__":
    main()
