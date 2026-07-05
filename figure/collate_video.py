#!/usr/bin/env python
"""One-pass MotionRecon collate: horizontal stack [raw + pose overlay | depth | points +33 | BEV].

Composes every pane per frame directly from the cached assets (frames/, depth/, fit3d/lift3d)
- no intermediate video decoding, no ffmpeg filter graph - and writes via NVENC (fastvid).
The raw pane carries the fitted humanoid skeleton reprojected onto the video.

Worker-parallel: the parent splits the frame range across WORKERS subprocesses (default 8 -
the consumer NVENC concurrent-session cap on current drivers), each renders its chunk to a
part file, and the parts are losslessly concatenated (ffmpeg -c copy). Frames are independent
(the BEV trail is a deterministic function of the frame index), so chunking is exact.

Layout: LAYOUT=row -> 1x4 horizontal strip [raw | depth | points | BEV] (default for portrait
sources); LAYOUT=grid -> 2x2 [raw, points / depth, BEV] in equal padded tiles (default for
landscape - e.g. figure gives 1920x1080).

Usage: DEPTH_WORK=<workdir> MR_OUT=<outdir> [MR_FPS=..] [TILT_DEG=33] [WORKERS=8] \\
       [LAYOUT=row|grid] python figure/collate_video.py
Output: <MR_OUT>/collate.mp4
"""
import glob
import os
import sys

import numpy as np
from matplotlib import cm
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fastvid import VideoWriter, gpu_splat               # noqa: E402
from skel_draw import KPN, I, draw_skeleton, draw_trail  # noqa: E402

WORK = os.environ["DEPTH_WORK"]
OUTD = os.environ.get("MR_OUT", "outputs/figure/pose")
FPS = float(os.environ.get("MR_FPS", 24000 / 1001))
TILT = np.radians(float(os.environ.get("TILT_DEG", "33")))
CONF_MIN = 1.2

fz = np.load(os.path.join(OUTD, "fit3d.npz"))
lz = np.load(os.path.join(OUTD, "lift3d.npz"))
J_w, t_arr, ok = fz["joints_w"].astype(np.float64), fz["t"], fz["ok"]
KAPPA = float(fz["kappa"]) if "kappa" in fz.files else 1.0
Rcw, scale, cam_pos = lz["R_cam2world"], float(lz["scale"]), lz["cam_pos"]
P_scene = lz["scene"].astype(np.float64)
if "M_c2w" in lz:                                        # moving camera: per-frame inverse map
    Minv = np.linalg.inv(lz["M_c2w"])
    J_cam = np.einsum("nij,nkj->nki", Minv, J_w - lz["c_c2w"][:, None])
else:
    J_cam = np.einsum("ji,nkj->nki", Rcw, (J_w - cam_pos[None, None]) / scale)

# SAM's focal (processed-frame px): the fitted joints subtend SAM's ANGULAR sizes, which map to
# correct pixels only under this focal - projecting them with VGGT's pose_enc focal shrinks the
# drawn skeleton by f_vggt/f_sam (0.91 figure, 0.62 vera). Median over the mhr cache; older
# caches predate the focal key and used SAM's fixed heuristic 1468.6.
_mhr = sorted(glob.glob(os.path.join(WORK, "mhr", "f*.npz")))
_f = [float(z["focal"]) for f in _mhr[:: max(1, len(_mhr) // 25)]
      for z in [np.load(f)] if "focal" in z.files]
F_SAM = float(np.median(_f)) if _f else 1468.6


def body_fix(Jc, fx, fy):
    """Undo the two un-inverted scale factors on the reprojected skeleton: the kappa heel
    calibration (world-space, legitimate there) and the VGGT-vs-SAM focal ratio. Pure rescale
    about the mid-hip: the pelvis stays on its kp2d ray, so its pixel is untouched.
    (w/FW converts F_SAM from processed-frame px to depth-resolution px.)"""
    pelv = 0.5 * (Jc[I["lhip"]] + Jc[I["rhip"]])
    s = F_SAM * (w / FW) / np.sqrt(fx * fy) / KAPPA
    return pelv + (Jc - pelv) * s

dfiles = sorted(glob.glob(os.path.join(WORK, "depth", "f*.npz")))
N = len(dfiles)
d0 = np.load(dfiles[0])["depth"]
h, w = d0.shape
FW, FH = Image.open(os.path.join(WORK, "frames", "f00000.jpg")).size
H = 1080 if h > w else 540                               # pane height by aspect
SCp = H / h                                              # cloud/depth pane scale
PW = int(w * SCp) // 2 * 2
RW = int(FW * H / FH) // 2 * 2                           # raw pane width
BW = H                                                   # BEV square

# depth colorization range (global)
samp = np.concatenate([np.load(dfiles[k])["depth"].astype(np.float32).ravel()[::9]
                       for k in range(0, N, max(1, N // 40))])
D0, D1 = np.percentile(samp, [2, 98])
TURBO = (np.asarray(cm.turbo(np.linspace(0, 1, 256)))[:, :3] * 255).astype(np.uint8)

# BEV mapping (from lift3d scene, like birdseye_video)
allx = np.concatenate([P_scene[:, 0], [cam_pos[0]]])
ally = np.concatenate([P_scene[:, 1], [cam_pos[1]]])
x0, x1 = np.percentile(allx, [0.5, 99.5]); y0, y1 = np.percentile(ally, [0.5, 99.5])
bscale = min((BW - 40) / (x1 - x0 + 0.8), (H - 40) / (y1 - y0 + 0.8))
bcx, bcy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)


def bpx(p):
    return (BW / 2 + (p[0] - bcx) * bscale, H / 2 - (p[1] - bcy) * bscale)


bev_bg = np.zeros((H, BW, 3), np.uint8); bev_bg[:] = (16, 16, 20)
uu = (BW / 2 + (P_scene[:, 0] - bcx) * bscale).astype(int)
vv = (H / 2 - (P_scene[:, 1] - bcy) * bscale).astype(int)
m = (uu >= 0) & (uu < BW) & (vv >= 0) & (vv < H)
g = (70 + np.clip(P_scene[:, 2], 0, 1.5) * 40).astype(np.uint8)
bev_bg[vv[m], uu[m]] = np.stack([g[m]] * 3, 1)
mid_w = 0.5 * (J_w[:, I["lhip"], :2] + J_w[:, I["rhip"], :2])

vs, us = np.mgrid[0:h, 0:w].astype(np.float32)
LAYOUT = os.environ.get("LAYOUT", "row" if FH > FW else "grid")
TW = max(RW, PW, BW)                                     # grid tile width
OW, OH = (RW + PW * 2 + BW, H) if LAYOUT == "row" else (TW * 2, H * 2)


def tile(pane):
    out = np.zeros((H, TW, 3), np.uint8)
    x0 = (TW - pane.shape[1]) // 2
    out[:, x0:x0 + pane.shape[1]] = pane
    return out


def render_frame(n):
    dz = np.load(dfiles[n])
    D = dz["depth"].astype(np.float32)
    C = dz["conf"].astype(np.float32)
    pe = dz["pose_enc"]
    fy = (h / 2.0) / np.tan(pe[7] / 2.0)
    fx = (w / 2.0) / np.tan(pe[8] / 2.0)
    src = Image.open(os.path.join(WORK, "frames", f"f{n:05d}.jpg"))
    Jc = body_fix(J_cam[n], fx, fy) if ok[n] else None
    # ---- pane 1: raw + humanoid pose overlay
    raw = np.asarray(src.resize((RW, H)), dtype=np.uint8)
    if ok[n]:
        ju = (Jc[:, 0] / Jc[:, 2] * fx + w / 2) * (RW / w)
        jv = (Jc[:, 1] / Jc[:, 2] * fy + h / 2) * (H / h)
        raw = draw_skeleton(raw, np.stack([ju, jv], 1), scale=H / 1080 + 0.3)
    # ---- pane 2: depth
    idx = np.clip((D - D0) / (D1 - D0) * 255, 0, 255).astype(np.uint8)
    dm = TURBO[idx]
    dm[C <= CONF_MIN] //= 4
    dm = np.asarray(Image.fromarray(dm).resize((PW, H), Image.NEAREST))
    # ---- pane 3: points at TILT + skeleton
    good = C > CONF_MIN
    X = (us - w / 2) / fx * D
    Y = (vs - h / 2) / fy * D
    zmid = float(np.median(D[good])) if good.any() else 1.0
    ca, sa = np.cos(TILT), np.sin(TILT)
    Yt = ca * Y - sa * (D - zmid)
    Zt = np.maximum(sa * Y + ca * (D - zmid) + zmid, 1e-4)
    u2 = (PW / 2 + X / Zt * fx * SCp)[good]
    v2 = (H / 2 + Yt / Zt * fy * SCp - (h * SCp - H) / 2)[good]
    pts = gpu_splat(np.stack([u2, v2], 1), Zt[good], np.asarray(src.resize((w, h)))[good], H, PW)
    if ok[n]:
        Yj = ca * Jc[:, 1] - sa * (Jc[:, 2] - zmid)
        Zj = np.maximum(sa * Jc[:, 1] + ca * (Jc[:, 2] - zmid) + zmid, 1e-4)
        ju = PW / 2 + Jc[:, 0] / Zj * fx * SCp
        jv = H / 2 + Yj / Zj * fy * SCp - (h * SCp - H) / 2
        pts = draw_skeleton(pts, np.stack([ju, jv], 1), scale=H / 1080 + 0.3)
    # ---- pane 4: BEV
    bev = bev_bg.copy()
    tr = [bpx(mid_w[k]) for k in range(0, n + 1, 3) if ok[k]]
    bev = draw_trail(bev, tr)
    if ok[n]:
        bev = draw_skeleton(bev, np.array([bpx(J_w[n, i, :2])                # body only: hands
                                           for i in range(min(18, J_w.shape[1]))]),  # are sub-px in BEV
                            scale=H / 1080 + 0.3)
    if LAYOUT == "row":
        return np.concatenate([raw, dm, pts, bev], 1)
    return np.concatenate([np.concatenate([tile(raw), tile(pts)], 1),
                           np.concatenate([tile(dm), tile(bev)], 1)], 0)


def render_range(n0, n1, out_path):
    wr = VideoWriter(out_path, OW, OH, FPS)
    for n in range(n0, n1):
        wr.write(render_frame(n))
        if (n - n0) % 600 == 0:
            print(f"[collate] {n0}-{n1}: {n - n0}/{n1 - n0}", flush=True)
    wr.close()


if __name__ == "__main__":
    part = os.environ.get("COLLATE_PART")
    final = os.path.join(OUTD, "collate.mp4")
    if part:                                             # worker mode
        n0, n1 = map(int, part.split(":"))
        render_range(n0, n1, os.environ["PART_OUT"])
        sys.exit(0)
    K = int(os.environ.get("WORKERS", "8"))
    if K <= 1:
        render_range(0, N, final)
    else:
        import subprocess
        import tempfile
        tmpd = tempfile.mkdtemp(prefix="collate_")
        bounds = np.linspace(0, N, K + 1).astype(int)
        procs, parts = [], []
        for k in range(K):
            part_out = os.path.join(tmpd, f"part{k:02d}.mp4")
            parts.append(part_out)
            env = dict(os.environ, COLLATE_PART=f"{bounds[k]}:{bounds[k + 1]}",
                       PART_OUT=part_out)
            procs.append(subprocess.Popen([sys.executable, os.path.abspath(__file__)], env=env))
        for pr in procs:
            if pr.wait() != 0:
                sys.exit("[collate] worker failed")
        lst = os.path.join(tmpd, "list.txt")
        with open(lst, "w") as f:
            f.writelines(f"file '{q}'\n" for q in parts)
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "concat",
                        "-safe", "0", "-i", lst, "-c", "copy", final], check=True)
        for q in parts + [lst]:
            os.remove(q)
        os.rmdir(tmpd)
    print(f"[collate] WROTE {final}", flush=True)
