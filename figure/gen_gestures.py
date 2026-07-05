#!/usr/bin/env python
"""Gesture probe for the Helix-02 reproduction: which gesture prompts survive a Root2D station pin?
Each gesture generates with the root pinned in place (hip-bump gets a bookend-only pin so the pelvis
can sway) and reports numeric descriptors + an 8-frame contact-sheet PNG for visual QA.

Probe verdicts (2026-07, seed 0): squat OK (24cm pelvis drop), reach OK (+44cm hands),
put-down usable; kick / crouch / hip-bump DEAD as prompts -> use constraints instead
(see figure/gen_helix_kitchen.py seg_kick / seg_hipbump and design/figure.md).

Run from repo root: CUDA_VISIBLE_DEVICES=0 TEXT_ENCODERS_DIR=... HF_HOME=... MUJOCO_GL=egl \
    PYTHONPATH=. python -m figure.gen_gestures
Outputs -> outputs/figure/gestures/<name>.{csv,png}
"""
import os

import numpy as np
import torch

from kimodo.constraints import Root2DConstraintSet
from kimodo.exports.mujoco import MujocoQposConverter
from kimodo.model.load_model import load_model
from kimodo.tools import seed_everything

_PKG = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_PKG)
OUT = os.path.join(REPO, "outputs", "figure", "gestures")
G1_XML = os.path.join(REPO, "kimodo/assets/skeletons/g1skel34/xml/g1.xml")
os.makedirs(OUT, exist_ok=True)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = load_model("kimodo-g1-rp", device=device, default_family="Kimodo")
skel = model.skeleton; fps = float(model.fps)
conv = MujocoQposConverter(skel)
b = skel.bone_index
ROOT = skel.root_idx
aL, aR = b["left_ankle_roll_skel"], b["right_ankle_roll_skel"]
hL, hR = b["left_hand_roll_skel"], b["right_hand_roll_skel"]

GESTURES = [
    ("kick",    "a person kicks forward with their right foot", 4.0, "full"),
    ("squat",   "a person squats down, picks something up from the ground, and stands back up", 5.0, "full"),
    ("crouch",  "a person bends down and picks something up", 5.0, "full"),
    ("reach",   "a person reaches up high with both hands to place an object on a shelf", 4.5, "full"),
    ("hipbump", "a person bumps their hip sideways to close a drawer", 4.0, "bookend"),
    ("putdown", "a person places an object down on a table in front of them", 4.0, "full"),
]


def station_pin(nf, mode):
    if mode == "full":
        f = torch.arange(nf, device=device)
    else:                                             # bookend: pin first/last 8 frames only
        f = torch.cat([torch.arange(8, device=device), torch.arange(nf - 8, nf, device=device)])
    root2d = torch.zeros(len(f), 2, dtype=torch.float32, device=device)
    heading = torch.tensor([[1.0, 0.0]], device=device).repeat(len(f), 1)
    return Root2DConstraintSet(skel, f, root2d, global_root_heading=heading).to(device)


def contact_sheet(qpos, path, nshots=8):
    import imageio.v2 as imageio
    import mujoco
    m = mujoco.MjModel.from_xml_path(G1_XML)
    m.vis.global_.offwidth, m.vis.global_.offheight = 480, 360
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, height=360, width=480)
    cam = mujoco.MjvCamera(); cam.azimuth, cam.elevation, cam.distance = 135.0, -12.0, 2.6
    shots = []
    idx = np.linspace(0, qpos.shape[0] - 1, nshots).astype(int)
    n = min(m.nq, qpos.shape[1])
    for i in idx:
        d.qpos[:] = 0.0; d.qpos[:n] = qpos[i, :n]
        mujoco.mj_forward(m, d)
        cam.lookat[:] = [float(d.qpos[0]), float(d.qpos[1]), 0.7]
        r.update_scene(d, camera=cam)
        shots.append(r.render())
    rows = [np.concatenate(shots[:4], axis=1), np.concatenate(shots[4:], axis=1)]
    imageio.imwrite(path, np.concatenate(rows, axis=0))


for name, prompt, dur, mode in GESTURES:
    nf = int(round(dur * fps))
    seed_everything(0)
    out = model([prompt], [nf], constraint_lst=[station_pin(nf, mode)], num_denoising_steps=100,
                num_samples=1, multi_prompt=False, post_processing=False,
                cfg_type="separated", cfg_weight=[2.0, 2.0], return_numpy=True)
    pj = np.asarray(out["posed_joints"])
    pj = pj[0] if pj.ndim == 4 else pj                                   # [T,34,3], Kimodo Y-up
    q = conv.dict_to_qpos(out, device)
    q = q.cpu().numpy() if torch.is_tensor(q) else np.asarray(q)
    q = q[0] if q.ndim == 3 else q
    np.savetxt(os.path.join(OUT, name + ".csv"), q, delimiter=",")

    footY = max(np.ptp(pj[:, aL, 1]), np.ptp(pj[:, aR, 1]))              # kick: foot rise
    pelv_drop = pj[0, ROOT, 1] - pj[:, ROOT, 1].min()                    # squat depth
    handY = max(pj[:, hL, 1].max(), pj[:, hR, 1].max())                  # reach height (Y up)
    hand0 = max(pj[0, hL, 1], pj[0, hR, 1])
    sway = np.ptp(pj[:, ROOT, 0])                                        # lateral pelvis excursion
    endpose = np.linalg.norm(pj[-1, ROOT, [0, 2]] - pj[0, ROOT, [0, 2]])
    still = np.mean(np.abs(np.diff(pj[-10:], axis=0)))
    print(f"[gest] {name:8s} footRise={footY*100:5.1f}cm pelvDrop={pelv_drop*100:5.1f}cm "
          f"handMax={handY*100:5.1f}cm(start {hand0*100:.0f}) sway={sway*100:5.1f}cm "
          f"endOff={endpose*100:4.1f}cm still={still*1000:.1f}mm/f")
    contact_sheet(q, os.path.join(OUT, name + ".png"))

print("DONE_GESTURES")
