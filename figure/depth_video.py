#!/usr/bin/env python
"""Per-frame depth reconstruction of figure.mp4 with VGGT-Omega (facebookresearch/vggt-omega).

Frames are processed in windows of S consecutive frames (one multi-view forward per window ->
temporally consistent depth inside each window). Per frame we save the predicted depth map,
its confidence, and the frame's camera pose encoding - everything needed to lift SAM-3D-Body
2D keypoints into 3D (sample depth at kp2d, back-project with the VGGT camera).

Usage (env sam_3d_body, GPU 0):
    CUDA_VISIBLE_DEVICES=0 DEPTH_WORK=<workdir> python figure/depth_video.py <ckpt.pt>
Outputs: <workdir>/depth/f%05d.npz  {depth[h,w] f16, conf[h,w] f16, pose_enc[9] f32}
"""
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vggt-omega"))
from vggt_omega.models.vggt_omega import VGGTOmega           # noqa: E402
from vggt_omega.utils.load_fn import load_and_preprocess_images  # noqa: E402

WORK = os.environ["DEPTH_WORK"]
CKPT = sys.argv[1]
S = 16                                                       # frames per forward

frames = sorted(glob.glob(os.path.join(WORK, "frames", "f*.jpg")))
outd = os.path.join(WORK, "depth")
os.makedirs(outd, exist_ok=True)

model = VGGTOmega().eval()
state = torch.load(CKPT, map_location="cpu", weights_only=False)
state = state.get("model", state) if isinstance(state, dict) else state
model.load_state_dict(state)
model = model.cuda()

todo = [f for f in frames if not os.path.exists(
    os.path.join(outd, os.path.basename(f)[:-4] + ".npz"))]
print(f"[depth] {len(todo)}/{len(frames)} frames to process, window={S}", flush=True)

with torch.no_grad():
    for w0 in range(0, len(todo), S):
        batch_files = todo[w0:w0 + S]
        images = load_and_preprocess_images(batch_files).cuda()
        pred = model(images)
        depth = pred["depth"][0].float().cpu().numpy()       # [S,h,w,1]
        conf = pred["depth_conf"][0].float().cpu().numpy()
        pose = pred["pose_enc"][0].float().cpu().numpy()     # [S,9]
        for i, f in enumerate(batch_files):
            key = os.path.basename(f)[:-4]
            np.savez_compressed(os.path.join(outd, key + ".npz"),
                                depth=depth[i, :, :, 0].astype(np.float16),
                                conf=conf[i].astype(np.float16),
                                pose_enc=pose[i].astype(np.float32))
        if (w0 // S) % 20 == 0:
            print(f"[depth] {w0 + len(batch_files)}/{len(todo)}", flush=True)
print("DONE_DEPTH", flush=True)
