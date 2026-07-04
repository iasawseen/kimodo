#!/usr/bin/env python
"""Reusable stitcher: build an arbitrarily long continuous locomotion trajectory by tiling
constant-velocity Root2D segments joined with FullBody pose-continuity pins.

Each segment is an ~8 s constant-velocity cruise (no start/stop ramp -> no mid-clip deceleration).
Segment i>=1 pins its first `overlap` frames to segment i-1's tail so the gait phase flows across
the seam; segments are translated into one timeline and the overlap is dropped (or crossfaded).

Import:
    from tools.stitch import stitch
    qpos, info = stitch("go forward", speed=1.0, duration=60, out_stem="go_forward_60s")

CLI (run from repo root, with the usual env: PYTHONPATH=. CUDA_VISIBLE_DEVICES=0, text-encoder vars):
    python tools/stitch.py "go forward" 1.0 60 --out go_forward_60s --render --blend

Limitations: translation-only join -> STRAIGHT-LINE forward locomotion only (curved stitching would
need per-seam heading rotation). Constant speed across the whole trajectory. Duration is met by
generating ceil() segments and trimming to the exact frame count.
"""
import argparse
import os

import numpy as np
import torch

from kimodo.constraints import FullBodyConstraintSet, Root2DConstraintSet
from kimodo.exports.mujoco import MujocoQposConverter
from kimodo.model.load_model import load_model
from kimodo.tools import seed_everything

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # tools/ -> repo root
DEFAULT_OUT_DIR = os.path.join(_REPO, "outputs", "gait")
_MODELS = {}


def _get_model(model_name, device):
    if model_name not in _MODELS:
        _MODELS[model_name] = load_model(model_name, device=device, default_family="Kimodo")
    return _MODELS[model_name]


def stitch(prompt, speed, duration, *, seg_s=8.0, overlap=8, out_stem=None, out_dir=DEFAULT_OUT_DIR,
           blend=False, model=None, model_name="kimodo-g1-rp", device=None, seed0=0, denoise=100):
    """Generate a continuous trajectory `duration` seconds long at constant `speed` (m/s).

    Returns (qpos [T,36] numpy, info dict). Saves `<out_dir>/<out_stem>.csv` if out_stem given.
    """
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model or _get_model(model_name, device)
    skel = model.skeleton; fps = float(model.fps); conv = MujocoQposConverter(skel)
    ridx = skel.root_idx
    NF = int(round(seg_s * fps)); K = int(overlap)
    total = int(round(duration * fps))
    nseg = 1 if total <= NF else int(np.ceil((total - NF) / (NF - K))) + 1

    def to_single(out):
        n = int(out["posed_joints"].shape[0])
        return {k: (v[0] if hasattr(v, "shape") and len(getattr(v, "shape", ())) > 0 and v.shape[0] == n else v)
                for k, v in out.items()}

    def qpos_of(out):
        q = conv.dict_to_qpos(out, device)
        q = q.cpu().numpy() if torch.is_tensor(q) else np.asarray(q)
        return q[0] if q.ndim == 3 else q

    def gen(i, tail_pj=None, tail_grot=None):
        head = torch.tensor(np.tile([1.0, 0.0], (NF, 1)), dtype=torch.float32, device=device)
        if tail_pj is None:
            z = speed * np.arange(NF) / fps
            r = torch.tensor(np.stack([np.zeros(NF), z], 1), dtype=torch.float32, device=device)
            cons = [Root2DConstraintSet(skel, torch.arange(NF, device=device), r, global_root_heading=head).to(device)]
        else:
            cons = [FullBodyConstraintSet(skel, torch.arange(K, device=device), tail_pj, tail_grot).to(device)]
            z0 = float(tail_pj[K - 1, ridx, 2]); zc = z0 + speed * (np.arange(K, NF) - (K - 1)) / fps
            r2 = torch.tensor(np.stack([np.zeros(NF - K), zc], 1), dtype=torch.float32, device=device)
            cons.append(Root2DConstraintSet(skel, torch.arange(K, NF, device=device), r2,
                                            global_root_heading=head[K:]).to(device))
        seed_everything(seed0 + i)
        return model([prompt], [NF], constraint_lst=cons, num_denoising_steps=denoise, num_samples=1,
                     multi_prompt=False, post_processing=False, cfg_type="separated",
                     cfg_weight=[2.0, 2.0], return_numpy=True)

    g_pj = g_grot = g_qpos = None; seams = []
    for i in range(nseg):
        if i == 0:
            out = gen(0)
        else:
            t = g_pj[-K:].copy(); off = t[0, ridx, [0, 2]].copy(); t[..., 0] -= off[0]; t[..., 2] -= off[1]
            out = gen(i, torch.tensor(t, dtype=torch.float32, device=device),
                      torch.tensor(g_grot[-K:], dtype=torch.float32, device=device))
        s = to_single(out); pj = np.asarray(s["posed_joints"]); grot = np.asarray(s["global_rot_mats"]); q = qpos_of(out)
        if i == 0:
            g_pj, g_grot, g_qpos = pj, grot, q
        else:
            opj = g_pj[-K, ridx, [0, 2]]; pj = pj.copy(); pj[..., 0] += opj[0]; pj[..., 2] += opj[1]
            oq = g_qpos[-K, :2] - q[0, :2]; q = q.copy(); q[:, 0] += oq[0]; q[:, 1] += oq[1]
            seams.append(float(np.linalg.norm(q[K, :2] - g_qpos[-1, :2])))
            if blend and K >= 2:
                w = np.linspace(0, 1, K + 2)[1:-1][:, None]
                g_qpos[-K:] = (1 - w) * g_qpos[-K:] + w * q[:K]
                quat = g_qpos[-K:, 3:7]; g_qpos[-K:, 3:7] = quat / np.linalg.norm(quat, axis=1, keepdims=True)
            g_pj = np.concatenate([g_pj, pj[K:]]); g_grot = np.concatenate([g_grot, grot[K:]])
            g_qpos = np.concatenate([g_qpos, q[K:]])

    g_qpos = g_qpos[:total]
    if out_stem:
        os.makedirs(out_dir, exist_ok=True)
        np.savetxt(os.path.join(out_dir, out_stem + ".csv"), g_qpos, delimiter=",")
    dist = float(np.linalg.norm(g_qpos[-1, :2] - g_qpos[0, :2]))
    info = dict(frames=len(g_qpos), duration=len(g_qpos) / fps, dist=dist,
                speed=dist / (len(g_qpos) / fps), nseg=nseg, seams=seams,
                max_seam=max(seams) if seams else 0.0)
    return g_qpos, info


def _cli():
    ap = argparse.ArgumentParser(description="Stitch a long continuous locomotion trajectory.")
    ap.add_argument("prompt"); ap.add_argument("speed", type=float); ap.add_argument("duration", type=float)
    ap.add_argument("--out", default=None, help="output stem (writes <out>.csv)")
    ap.add_argument("--seg", type=float, default=8.0); ap.add_argument("--overlap", type=int, default=8)
    ap.add_argument("--blend", action="store_true", help="crossfade seams instead of hard-drop")
    ap.add_argument("--render", action="store_true", help="also render <out>.mp4 via render_g1.py")
    ap.add_argument("--model", default="kimodo-g1-rp"); ap.add_argument("--denoise", type=int, default=100)
    a = ap.parse_args()
    _, info = stitch(a.prompt, a.speed, a.duration, seg_s=a.seg, overlap=a.overlap, out_stem=a.out,
                     blend=a.blend, model_name=a.model, denoise=a.denoise)
    print(f"[stitch] {info['frames']}f = {info['duration']:.1f}s, dist={info['dist']:.1f}m, "
          f"speed={info['speed']:.2f} m/s, nseg={info['nseg']}, max_seam={info['max_seam']*100:.0f}cm")
    if a.render and a.out:
        import subprocess
        here = os.path.dirname(os.path.abspath(__file__))
        env = {**os.environ, "MUJOCO_GL": "egl"}
        csv = os.path.join(DEFAULT_OUT_DIR, a.out + ".csv"); mp4 = os.path.join(DEFAULT_OUT_DIR, a.out + ".mp4")
        subprocess.run(["python", os.path.join(here, "render_g1.py"), csv, mp4, "30"], check=True, env=env)
        print(f"[stitch] rendered {mp4}")


if __name__ == "__main__":
    _cli()
