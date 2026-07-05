#!/usr/bin/env python
"""Estimate 3D robot poses from figure.mp4 frames with SAM 3D Body (facebookresearch/sam-3d-body).

The Figure robot defeats the human detector (white shell / black joints on a B&W kitchen), so the
detector is bypassed: bboxes come from median-background subtraction (see design/figure.md) stored
in <workdir>/boxes.npz. Frames were extracted at 2 fps into <workdir>/frames.

Outputs per frame into <workdir>/mhr/f%05d.npz:
    kp3d [70,3]  MHR-70 keypoints, camera frame     kp2d [70,2]  image px
    cam_t [3]    person->camera translation          focal        px
plus an overlay jpg every SPOT frames into <workdir>/overlay for visual QA.

Usage (env sam_3d_body, GPU 1):
    CUDA_VISIBLE_DEVICES=1 python figure/pose_video.py <checkpoint_dir> [workdir]
"""
import os
import sys

import cv2
import numpy as np
import torch

WORK = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("POSE_WORK", "")
CKPT_DIR = sys.argv[1]
SPOT = 20

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "sam-3d-body"))
from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator  # noqa: E402

device = torch.device("cuda")
model, model_cfg = load_sam_3d_body(
    os.path.join(CKPT_DIR, "model.ckpt"), device=device,
    mhr_path=os.path.join(CKPT_DIR, "assets", "mhr_model.pt"),
)
estimator = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=model_cfg)

boxes = np.load(os.path.join(WORK, "boxes.npz"))
os.makedirs(os.path.join(WORK, "mhr"), exist_ok=True)
os.makedirs(os.path.join(WORK, "overlay"), exist_ok=True)
np.save(os.path.join(WORK, "mhr", "faces.npy"), estimator.faces)   # constant mesh topology
FORCE = os.environ.get("PV_FORCE", "0") == "1"          # re-run frames that lack the new keys

keys = sorted(boxes.files)
for n, key in enumerate(keys):
    out_path = os.path.join(WORK, "mhr", key + ".npz")
    if os.path.exists(out_path):
        if not FORCE:
            continue
        try:                                            # idempotent force: skip if already upgraded
            if "verts" in np.load(out_path).files:
                continue
        except Exception:
            pass
    img_path = os.path.join(WORK, "frames", key + ".png")
    if not os.path.exists(img_path):
        img_path = os.path.join(WORK, "frames", key + ".jpg")
    bbox = boxes[key][None].astype(np.float32)          # [1,4] xyxy
    outputs = estimator.process_one_image(img_path, bboxes=bbox, use_mask=False)
    if not outputs:
        print(f"[pose] {key}: NO OUTPUT")
        continue
    o = outputs[0]
    np.savez_compressed(
        out_path,
        kp3d=o["pred_keypoints_3d"].astype(np.float32),
        kp2d=o["pred_keypoints_2d"].astype(np.float32),
        cam_t=o["pred_cam_t"].astype(np.float32),
        focal=np.float32(o["focal_length"]),
        bbox=boxes[key].astype(np.float32),
        # full MHR body: mesh + explicit DIRECTIONS (orientation / per-joint global rotations)
        verts=o["pred_vertices"].astype(np.float16),
        global_rot=o["global_rot"].astype(np.float32),
        joint_rots=o["pred_global_rots"].astype(np.float16),
        joint_coords=o["pred_joint_coords"].astype(np.float32),
        body_pose=o["body_pose_params"].astype(np.float32),
        shape=o["shape_params"].astype(np.float32),
        scale=o["scale_params"].astype(np.float32),
    )
    if n % SPOT == 0:
        img = cv2.imread(img_path)
        x0, y0, x1, y1 = boxes[key].astype(int)
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 3)
        for u, v in o["pred_keypoints_2d"][:21]:        # body kps only
            cv2.circle(img, (int(u), int(v)), 5, (0, 255, 0), -1)
        cv2.imwrite(os.path.join(WORK, "overlay", key + ".jpg"), img)
        print(f"[pose] {n+1}/{len(keys)}")
print("DONE_POSE")
