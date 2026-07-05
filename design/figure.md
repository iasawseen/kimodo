# Reproducing the Figure "Helix 02" kitchen demo with G1 in MuJoCo

Practical guide for the `figure/` package: reproducing the **Figure AI "Introducing Helix 02"**
kitchen video (`outputs/figure/figure.mp4`, 216 s) with the **Unitree G1** in MuJoCo, using
**Kimodo as motion generator + stitcher**. Whole-body motion is the deliverable; manipulation is
mimed (no object physics). Everything here is empirical. Companion doc: [gait.md](gait.md) — the
per-lever control recipes (speed / step length / turns / EE constraints) this build is made of.

> **One-line architecture:** decompose the scenario into distinct **stand-bookended** motion
> segments → generate each independently with Kimodo **constraints** (prompts only style them) →
> chain by **SE(2) alignment + crossfade** at the stand-to-stand seams → build the scene **around
> the measured key poses** → gate with a **collision check** → render with the video-matched
> fixed camera, animating the dishwasher door/rack in sync.

---

## 0. Files

| File | Role |
|---|---|
| `figure/gen_helix_kitchen.py` | segment generators + storyboard + SE(2) chain-stitcher (GPU). Caches per-segment qpos in `outputs/figure/segs/`; CLI args name segments to force-regenerate (`all` for everything) |
| `figure/build_scene.py` | kitchen XML built around measured poses + camera meta (no GPU; iterate freely) |
| `figure/render_scene.py` | fixed-camera renderer; animates door hinge + rack slide from segment bounds; `--sheet` for an 8-frame QA contact sheet |
| `figure/check_collisions.py` | collision QA gate (robot vs scene solids + self-collision) |
| `figure/gen_gestures.py` | gesture prompt probe (which gestures work under a station pin) |
| `figure/pose_video.py`, `figure/pose_retarget.py` | SAM-3D-Body runner (per-frame MHR-70) + direction-transfer retarget to G1-34 (§7.1) |
| `figure/depth_video.py`, `figure/lift_skeleton.py`, `figure/fit_pose.py` | MotionRecon core: VGGT-Omega depth, world alignment + naive lift, rigid fitter + smoother (§7.2–7.5) |
| `figure/fit_kitchen.py` | kitchen fit + camera solve from the reconstruction (§7.6) |
| `figure/skel_draw.py`, `figure/birdseye_video.py`, `figure/pose_over_cloud.py`, `figure/recon3d_video.py`, `figure/recon_check_video.py` | visualization / verification videos (§7.6) |
| `figure/build_verify.sh` | rebuilds `outputs/figure/verify/` clips (video top / render bottom, time-synced) |
| `outputs/figure/` | `figure.mp4` (source video), `helix_kitchen.csv` (qpos [T,36]), `_bounds.json` (segment index), `kitchen_g1.xml`, `helix_kitchen_scene.json` (camera + solids), `helix_kitchen.mp4`, `pose/` (anchors + `lift3d.npz` / `fit3d.npz` + visualization mp4s), `verify/` |

Run order (env per [gait.md](gait.md) §6):

```bash
PYTHONPATH=. python -m figure.gen_helix_kitchen        # GPU; uses segment cache when present
PYTHONPATH=. python -m figure.build_scene              # scene + camera (re-runs in seconds)
python figure/check_collisions.py                      # QA gate — fix and re-roll until PASS
MUJOCO_GL=egl python figure/render_scene.py outputs/figure/helix_kitchen.csv \
    outputs/figure/kitchen_g1.xml outputs/figure/helix_kitchen.mp4
```

---

## 1. Scenario decomposition (what the video actually does)

From 1 s-sampled frames of `figure.mp4`: one fixed camera; the robot walks in from frame-right,
then works **planted at the dishwasher pocket** for ~3 minutes — there are no long walks, no turns
across the room. The motion vocabulary is:

| Video beat (t≈) | Reproduction segment | Generator recipe |
|---|---|---|
| 0–6 s walk in from the right | `walk_in` | Root2D ramped velocity profile (stand-walk-stand) |
| 6–11 s **open door by hand**, pull rack out | `open_door`, `pull_rack` | **two-hand EE keyframe arcs** + station pin |
| 12–190 s unload cycles: bend to rack ↔ place on counter ↔ reach up to uppers | `unload_1..3`, `place_ctr1/2`, `place_up` | prompt gestures + Root2D station pin |
| 192–197 s push rack in | `push_rack` | two-hand EE push arc |
| 198–209 s **lift door closed with the foot** | `step_back`, `close_door` | short backward Root2D step, then **right-foot EE lift arc** |
| 210 s+ stand | `final_stand` | station pin |

13 segments, ~58 s (a faithful condensation: 3 unload cycles instead of ~6).

**Every segment is stand-bookended** (starts and ends standing at a fixed spot) — that is the
composability contract that makes the chain-stitch clean.

---

## 2. Constraints do the work; prompts only style

The probe (`gen_gestures.py`, station-pinned, seed 0) splits gestures into prompt-alive and
prompt-dead:

| Gesture | Prompt result | Verdict |
|---|---|---|
| squat + pick from ground | 24 cm pelvis drop, hand to floor, clean recovery | ✅ use prompt |
| reach up with both hands | +44 cm hand raise (to 1.46 m) | ✅ use prompt |
| put object down in front | moderate forward place | ⚠ usable (gate hand clearance, §5) |
| kick / foot lift | **0.5 cm** foot rise — nothing | ❌ **foot-EE constraint** |
| crouch ("bends down and picks up") | 0.2 cm — nothing | ❌ drop (squat covers it) |
| hip bump | 2.5 cm sway — nothing | ❌ Root2D sway path (not used in v2) |

The dead ones are exactly the off-distribution motions [gait.md](gait.md) §1 warns get silently
substituted. Recipes that replace them:

- **Foot/hand EE keyframe arcs** — `EndEffectorConstraintSet` subclasses
  (`Left/RightHandConstraintSet`, `Left/RightFootConstraintSet`) pin the EE chain at sparse
  keyframes: tile a **reference stand pose** (`outputs/gait/go_forward.npz` frame 0, the same
  source the turn-settle recipe uses) and displace the EE joint chain
  (`skel.left_hand_joint_names` etc.) by `(dx_left, dy_up, dz_fwd)` offsets per keyframe. Add a
  full-frame **Root2D station pin** so the body stays planted. Kimodo fills in natural whole-body
  coordination (weight shift, counterbalance) between keys.
- Gestures aim at the robot's **front-right** (`LAT=-0.28` lateral offset in the EE keys): the
  robot works the dishwasher from the side, like the video — and this keeps its feet clear of the
  open door (zero solid collisions, §5).

---

## 3. SE(2) chain-stitching (heterogeneous segments)

`tools/stitch.py` (gait.md §10) is translation-only. Choreography needs full **SE(2)**: each
segment generates in its local frame (origin, facing +X in qpos terms), then is rigidly placed so
its frame-0 root pose coincides with the previous segment's final pose:

```
psi = yaw(prev[-1]);  R = Rz(psi - yaw(seg[0]))
seg.xy  = (seg.xy - seg.xy[0]) @ R.T + prev[-1].xy
seg.quat = qz(dpsi) * seg.quat            # then crossfade first K=10 frames toward prev[-1]
```

Because seams are stand-to-stand, the crossfade correction is centimeters. Segment boundaries are
saved to `_bounds.json` — the renderer, the collision gate, and the scene builder all key off it.

---

## 4. Scene: built around measured poses, animated, camera-matched

- **Layout from the video** (1 s frame sampling): L-shaped kitchen — sink/dishwasher wall receding
  image-left (black bases, white tops, black arch faucet, window over the sink, black uppers, tall
  cabinet), **dishwasher at the L-corner**, frontal wall with **pro range + stainless hood**,
  black uppers, the toy-dinosaur prop, tall column at frame-right; dining table + chairs
  foreground-right; light wood floor. All at **G1 scale (~0.75× human)**: counters 0.65 m, uppers
  from 1.05 m, door 0.41 m.
- **Placed around measured poses**: the dishwasher is centered where the `open_door` gestures
  actually aim (`work_pose + 0.45·facing − 0.28·left`), the counter runs through it, the back wall
  hangs off the corner. Retarget drift cannot misalign the set — the set follows the motion.
  Heading snapping keeps it axis-aligned and turn-convention-invariant.
- **Jointed dishwasher**: the door is a **hinge body**, the bottom rack a **slide body** (with a
  dish on it). `render_scene.py` and `check_collisions.py` drive them on a schedule keyed to the
  segment bounds: door swings down during `open_door` (1.0–2.6 s in-segment), rack out during
  `pull_rack`, in during `push_rack`, door up under the foot during `close_door` (1.4–2.7 s).
  Their qpos live at indices ≥36 (robot csv is [T,36]); the renderer looks them up by joint name.
- **Camera** matched to the video: fixed, in the dining area, ~1.5 m high, elevation ≈ −10°,
  fovy 45, framed on the corner pocket. Stored in `helix_kitchen_scene.json`; iterate with
  `build_scene.py` + `render_scene.py --sheet` (seconds per iteration, no GPU).

---

## 5. Collision gate (the /goal conditions, enforced)

`check_collisions.py` applies the acceptance conditions from [gait.md](gait.md) ("Pipeline &
acceptance goal"): enable the robot's `*_collision` geoms against the scene in MuJoCo's collision
pipeline, `mj_forward` per frame (door/rack driven to their scheduled poses), and report
penetrations in three classes:

- **solid** (counters, walls, range, hood, uppers, table, chairs — the `solids` list in the scene
  json): **gated at 2 cm** — any deeper penetration fails the run;
- **self** (robot-robot, adjacent bodies auto-filtered): **gated at 1 cm**;
- **prop** (door, racks, dishes, faucet, floor): reported FYI — mime contact is expected there
  (foot-floor shows the usual 2–4 cm retarget sink-in).

v2 result: **0 solid pairs**. The one initial FAIL was self-collision — the seed-0 put-down
clasped its hands 2.1 cm into each other. Fix per the closing-the-loop rule: `seg_putdown()` now
**re-rolls seeds until the minimum hand-hand gap ≥ 5 cm**, measured from `posed_joints` at
generation time — the gate condition checked at the source instead of after the fact.

---

## 6. Known limitations / v2 upgrade path

- Manipulation is mimed: no grasped objects, no contact dynamics; dishes don't move (except the
  animated rack dish). The door "responds" to the foot only via the synced schedule.
- Prompt gestures don't aim: squats reach forward, not toward the lateral rack offset. Hand-EE
  nudges on the squats would fix aim at the cost of more constraint tuning.
- This is stage-0 output in the [gait.md](gait.md) pipeline sense — kinematic only. WBC replay
  (stage 2) would add balance/contact truth; the collision gate here is the kinematic
  approximation of that acceptance test.
- **Pose-from-video**: implemented — see §7 (MotionRecon). Video anchors now pin the walk start /
  arrival poses and the work-facing calibration; the constraint plumbing above is exactly the
  interface that consumes them.

---

## 7. MotionRecon pipeline (v4): video → 3D robot motion

Perception stack that turns `figure.mp4` into metric 3D supervision for the generator:
**SAM-3D-Body** (per-frame robot skeleton) + **VGGT-Omega** (per-frame depth + camera) →
**rigid fitter** (fuse into world-frame motion) → **temporal smoother**. Everything runs on the
5189-frame full video.

```
figure.mp4 (1280x720 frames, 23.976 fps)
   ├─ SAM-3D-Body  (pose_video.py)   → mhr/f%05d.npz   kp2d, kp3d, cam_t      [GPU 1, ~75 min]
   └─ VGGT-Omega   (depth_video.py)  → depth/f%05d.npz depth, conf, pose_enc  [GPU 0, ~10 min]
        → lift_skeleton.py   alignment (up/floor/scale/yaw) + naive lift  → lift3d.npz
        → fit_pose.py        rigid fitter + smoother                      → fit3d.npz
        → consumers: gen_helix_kitchen (anchors/trajectory), fit_kitchen (camera solve),
          visualization videos (birdseye_fit, pose_over_cloud, recon checks)
```

### 7.1 SAM-3D-Body (github.com/facebookresearch/sam-3d-body)

- Checkpoint from the **ModelScope mirror** (`facebook/sam-3d-body-dinov3`; HF repo is gated);
  torch.jit `mhr_model.pt` — no detectron2 needed. Manual/interpolated bboxes
  (`boxes.npz`) bypass the human detector.
- Output per frame: **MHR-70** keypoints — `kp2d` (pixels), `kp3d` (metric, root-relative),
  `cam_t` (camera translation). Indices used: hips 9/10, knees 11/12, ankles 13/14, shoulders
  5/6, elbows 7/8, wrists L62/R41, neck 69, heels 17/20, toes 15/18, nose 0.
- **kp2d is in the processed-frame pixel space (1280×720 here) — not the source 1920×1080.**
  A wrong `W_IMG` scaled every ray by 2/3 and silently corrupted all downstream world positions;
  the symptom was skeletons floating off the robot in every 3D view.
- Reliability: bone lengths stable to 3.4–6.9 % per frame; **absolute depth (`cam_t`) wobbles
  ±0.36 m even when the robot is static** (scale-from-appearance noise); L/R flips on the
  faceless robot (~40 % of frames) are handled in `pose_retarget.py` by mirror-vote
  stabilization. **First 4 s are discarded** (`ok &= t >= 4.0`): robot absent/partial there.

### 7.2 VGGT-Omega (github.com/facebookresearch/vggt-omega)

- `vggt_omega_1b_512.pt` via ModelScope (HF gated). Feed-forward; frames processed in
  **16-frame windows** (`depth_video.py`), saving `depth`/`conf` (float16) + 9-dof `pose_enc`
  (decode: `encoding_to_camera` — fov_h/fov_w → intrinsics). Static camera ⇒ each window's world
  frame ≈ the camera frame.
- **What it's good for** (verified with `recon_check_video.py` — identity re-render + ±33°
  parallax views + colorized depth maps): per-frame relative structure — depth ordering,
  occlusion boundaries, planarity, temporal stability. `fovy ≈ 30°` from `pose_enc` fixed the
  render camera's lens (was hand-tuned to 44°).
- **What it's NOT good for** (measured, decisive):
  - the whole scene is compressed into raw depth [0.64, 1.36] (2:1) and carries a **global
    low-frequency warp** (fitted floor slopes ~1.7 u across the room; near field stretched) —
    absolute box-fitting from the cloud is unreliable beyond ~0.3 m;
  - depth over the **thin robot is context-inpainted**: corr(SAM joint depth, VGGT depth at
    joints) ≈ 0.24; a per-frame affine depth fit flips sign frame to frame even with a planar
    bias field. Per-joint depth reads are unrecoverable — only **one body-level depth per
    frame** is usable;
  - **per-window scale/shift drift**: pelvis jumps of ~3.3 cm at every 16-frame seam in the
    naive lift.

### 7.3 Alignment + naive lift (`lift_skeleton.py` → `lift3d.npz`)

World frame: floor z=0, +X = the robot's dishwasher-work facing, G1 scale. Calibrations — each
replaced a scene-derived estimate that measurably failed:

- **Up from the robot's torso** (median walk/stand torso direction in camera coords). Floor-plane
  RANSAC is 30–39° off — VGGT's textureless-floor depth is bowed (measured by standing-torso
  verticality).
- **Floor + metric scale from the robot**: heels during the dw stand define z=0; standing pelvis
  height = 0.72 G1-units sets the scale (scene z-histogram peaks latch onto tables/racks).
  Self-check after: stand pelvis 0.72, heels ≈ 0.
- **Yaw anchor**: +X = median hip-line facing over the dw-work window t∈[8, 13.5] s (carries the
  same ~90° ambiguity as the anchor yaws — consumers that need direction use chord alignment).
- Naive per-keypoint lift (kp2d ray × depth sample) is kept as the baseline: silhouette
  keypoints bleed onto background depth → bone-length scatter 14–82 %.

### 7.4 Rigid fitter (`fit_pose.py` → `fit3d.npz`)

Division of labor set by the diagnostics above: **articulation entirely from SAM's rigid kp3d;
the cloud contributes exactly one number per frame** — a robust body depth:

- interior **bone-tube** mask (bones dilated by per-part radii, surface = axis depth − radius),
  conf-gated; body depth = median of tube pixels, median-filtered with **k=19 > window length**
  so every VGGT seam is straddled;
- pelvis rides its kp2d ray at that depth; all other joints are SAM kp3d offsets from mid-hip,
  scaled by the (heavily smoothed) raw-per-meter ratio → **bone lengths constant by
  construction**;
- results: bone scatter 20–79 % → **4–6 %** (SAM's own floor), pelvis seam jumps 3.3 → 1.4 cm.

### 7.5 Smoother

Per joint/coordinate in camera frame, before the world transform: **7-frame median** (kills
single-frame SAM spikes/flips) + **11-frame moving average** (~0.45 s). Walking/reaching
dynamics survive; work-phase pelvis steps drop to 0.3 cm within windows, **0.4 cm across seams**
(seam artifact below natural motion level).

### 7.6 Consumers + limits

- `gen_helix_kitchen.py`: walk start/arrival = video anchor poses (t=4.0 / 6.5 s);
  `lift_traj()` can lay the measured pelvis path into the scene (chord-aligned about the
  arrival, because absolute yaw/depth of the entry are the weak axes). Currently the walk path
  is synthetic again: the video's actual walking happens at 2.4–4.0 s — inside the discarded
  SAM window — and the valid remainder is only a settle.
- `fit_kitchen.py`: camera solve (fovy 30°, azimuth −74.5° w.r.t. work facing, blend-refined
  el/height) — baked into `build_scene.py`.
- Visualizations (all source-synced; `skel_draw.py` styling — L cyan / R orange / spine mint,
  supersampled): `birdseye_fit.mp4` (BEV + fading trail), `pose_over_cloud.mp4` (+33° cloud +
  skeleton), `recon_points.mp4` / `depth_maps.mp4` (recon QA), `collate.mp4` (2×2:
  raw / depth / points+33° / BEV).
- Honest limits: absolute trajectory depth is ±0.3 m-class (both estimators weak, corr 0.31 —
  keep chord alignment); the entry walk is under-measured; SAM per-frame scale wobble survives
  smoothing at low frequency.
