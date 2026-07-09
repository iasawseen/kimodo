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

`humanoid_motion_recon.*` entries live in the standalone sibling repo
`../humanoid_motion_recon` (mono video → world skeletons + meshes; `pip install -e`'d into
both envs, run as `python -m humanoid_motion_recon.<tool>`). Its README is the pipeline
doc; §7 here keeps the Figure-scenario findings.

| File | Role |
|---|---|
| `figure/gen_helix_kitchen.py` | segment generators + storyboard + SE(2) chain-stitcher (GPU). Caches per-segment qpos in `outputs/figure/segs/`; CLI args name segments to force-regenerate (`all` for everything) |
| `figure/build_scene.py` | kitchen XML built around measured poses + camera meta (no GPU; iterate freely) |
| `figure/render_scene.py` | fixed-camera renderer; animates door hinge + rack slide from segment bounds; `--sheet` for an 8-frame QA contact sheet |
| `figure/check_collisions.py` | collision QA gate (robot vs scene solids + self-collision) |
| `figure/gen_gestures.py` | gesture prompt probe (which gestures work under a station pin) |
| `humanoid_motion_recon.pose_video`, `figure/pose_retarget.py` | SAM-3D-Body runner (per-frame MHR: keypoints + mesh + rotations, §7.1) + direction-transfer anchor retarget (§7.1) |
| `figure/soma_retarget.py`, `figure/render_replay.py` | SOMA-style retarget fit3d → G1 qpos (direction transfer + MuJoCo DLS IK) + vertical video/MuJoCo replay renderer, plain or kitchen scene (§7.8) |
| `humanoid_motion_recon.sam3d_dummy_video`, `humanoid_motion_recon.sam3d_mesh_video` | chirality evidence: raw SAM 3D skeleton / 3D mesh renders (mesh cache-first from saved `verts`; §7.1) |
| `humanoid_motion_recon.gpurender` | shared torch-CUDA raster utils (splat/lines/cloud/resize/colormap) used by all video renderers (§7.9) |
| `humanoid_motion_recon.lift_mesh`, `humanoid_motion_recon.mesh_world_video` | world-lift + temporal smoothing of the SAM meshes (fit_pose transform chain per vertex) + two-pane render (§7.9) |
| `humanoid_motion_recon.depth_video`, `humanoid_motion_recon.lift_skeleton`, `humanoid_motion_recon.fit_pose` | MotionRecon core: VGGT-Omega depth, world alignment + naive lift, rigid fitter + smoother (§7.2–7.5) |
| `figure/fit_kitchen.py` | kitchen fit + camera solve from the reconstruction (§7.6) |
| `humanoid_motion_recon.skel_draw`, `humanoid_motion_recon.birdseye_video`, `humanoid_motion_recon.pose_over_cloud`, `humanoid_motion_recon.recon3d_video`, `humanoid_motion_recon.recon_check_video` | visualization / verification videos (§7.6) |
| `figure/export_soma_motion.py` | fit3d (+ mesh) → Kimodo SOMA-77 motion npz (motion-seed schema: local/global rot mats, FK posed_joints, root, heading, foot contacts); direction-transfer retarget, validated round-trip vs a reference seed |
| `figure/build_verify.sh` | rebuilds `outputs/figure/verify/` clips (video top / render bottom, time-synced) |
| `outputs/figure/` | `figure.mp4` (symlink; source videos + recon render mp4s live in `../humanoid_motion_recon/videos/`), `helix_kitchen.csv` (qpos [T,36]), `_bounds.json` (segment index), `kitchen_g1.xml`, `helix_kitchen_scene.json` (camera + solids), `helix_kitchen.mp4`, `pose/` (anchors + `lift3d.npz` / `fit3d.npz` + visualization mp4s), `verify/` |

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
- Output per frame (`mhr/f%05d.npz`): **MHR-70** keypoints — `kp2d` (pixels), `kp3d` (metric,
  root-relative), `cam_t` (camera translation) — **plus the full MHR body**: `verts` (mesh
  vertices, f16; topology in `mhr/faces.npy`), `global_rot` (body orientation — the explicit
  facing direction), `joint_rots` (per-joint global rotations, f16), `joint_coords`,
  `body_pose`/`shape`/`scale` params. Keypoint indices used: hips 9/10, knees 11/12, ankles
  13/14, shoulders 5/6, elbows 7/8, wrists L62/R41, neck 69, heels 17/20, toes 15/18, nose 0;
  hands R 21–41 / L 42–62 (5 fingers × 4 joints, tip→MCP, + wrist).
  Older caches carry only the keypoint keys; `PV_FORCE=1` re-runs frames to add the rest.
- **kp2d is in the processed-frame pixel space (1280×720 here) — not the source 1920×1080.**
  A wrong `W_IMG` scaled every ray by 2/3 and silently corrupted all downstream world positions;
  the symptom was skeletons floating off the robot in every 3D view.
- Reliability: bone lengths stable to 3.4–6.9 % per frame; **absolute depth (`cam_t`) wobbles
  ±0.36 m even when the robot is static** (scale-from-appearance noise). **First 4 s are
  discarded** (`ok &= t >= 4.0`): robot absent/partial there.
- **Front/back chirality**: on the faceless robot SAM resolves roughly half the frames as a
  depth-mirrored, L/R-swapped estimate (a mirrored pose projects to the SAME 2D image — the
  ambiguity is invisible in every 2D overlay and only bites 3D consumers, e.g. the retarget,
  where a surviving mirrored stretch reads as the robot spinning 180°). Confirmed on the raw
  MESH renders: consecutive side views snap 180° while the video robot stands still
  (`sam3d_mesh_video.py`: mesh overlay + 90° side view — front/back is unambiguous on the
  shaped body; `sam3d_dummy_video.py`: 3D skeleton variant). **An in-fitter fix
  (`MR_FLIPFIX`) was tried and REVERTED**: un-mirroring frames inside `fit_pose.py` makes the
  corrected 3D deliberately disagree with the 2D image on flipped frames, which visibly breaks
  every image-space fit3d consumer (collate cloud pane, pose_over_cloud, BEV). Architecture
  lesson: `fit3d.npz` stays faithful to SAM's per-frame estimates (2D-consistent); chirality
  correction belongs DOWNSTREAM in the retarget layer (or in a side-channel artifact), never
  inside the shared fit. Attempt history for that layer: per-frame median vote → greedy
  hip-line continuity → global 2-state Viterbi (velocity-alignment unaries + hip-line
  continuity + switch penalty) reached 88 % walk-frame alignment but left 23 >45°/frame
  heading jumps; greedy and median rules specifically fail in PROFILE views where the mirror
  barely changes the hip line. Two hard-won constraints for the next attempt: correct on raw
  per-frame estimates BEFORE smoothing (smoothing blends mirror flickers into unrecoverable
  averages), and prefer the now-saved `global_rot`/`joint_rots` (§7.1) — a full-orientation
  observation — over reconstructed hip lines.

### 7.2 VGGT-Omega (github.com/facebookresearch/vggt-omega)

- `vggt_omega_1b_512.pt` via ModelScope (HF gated). Feed-forward; frames processed in **large
  overlapping windows** (`depth_video.py`, default 160-frame window / 120 stride; 192 frames fit
  in 24 GiB). Consecutive windows are renormalized by VGGT, so the chain is stitched on the
  **shared frames**: robust affine `d ← a·d + b` on subsampled shared-frame depths, then linear
  blend across the 40-frame overlap. This removed the window seams at the source (naive-lift
  pelvis steps: uniform 0.5 cm, no seam spikes) and needs no static-scene assumption.
- **Camera extrinsics are chained the same way** (per-frame `cam_R`/`cam_T` =
  camera-from-chain, saved in each depth npz): each window's `pose_enc` poses live in that
  window's own frame, so the window→chain sim(3) is estimated from the shared frames (rotation:
  SVD-averaged relative orientation; scale: the depth affine `a`; translation: mean
  camera-center residual). This is what makes lifted skeletons world-consistent under a
  **moving camera** (vera: measured camera travel 1.5 u — a static-camera assumption put all of
  that into the skeleton trajectory).
- Saves `depth`/`conf` (float16) + 9-dof `pose_enc` (decode: `encoding_to_camera` — fov_h/fov_w
  → intrinsics; quaternion is xyzw scalar-last).
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
  - **per-window scale/shift drift**: pelvis jumps of ~3.3 cm at every window seam in the naive
    lift before the overlap stitching (historical 16-frame-window runs; the overlap stitch
    removed them).

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
- **Moving camera**: joints are first mapped camera→chain via the per-frame `cam_R`/`cam_T`
  (identity fallback for pre-chaining depth caches), and all calibrations run in the chain
  frame. `lift3d.npz` carries per-frame affine world maps `M_c2w [N,3,3]` / `c_c2w [N,3]`
  (`world(p_cam, n) = M_n p + c_n`; `c_n` is the camera center); the static keys
  (`R_cam2world`, `cam_pos`, `scale`) remain for static-camera consumers.
- **Scenario parameters are env vars** (`MR_FPS`, `MR_TMIN`, `MR_STAND0/1`, `MR_YAW0/1`,
  `MR_UP0/1`, `MR_PELVIS_H`, `MR_OUT`); frame size is auto-detected from `frames/f00000.jpg`
  (portrait phone video works — extraction must respect rotation metadata, e.g. `scale=720:1280`
  for a rotate-90 4K source). Calibration windows must sample **upright** phases: on vera the
  windows initially covered a shelf-crouch, which tilted the torso-up axis and shrank the
  floor/scale estimates.

### 7.4 Rigid fitter (`fit_pose.py` → `fit3d.npz`)

Division of labor set by the diagnostics above: **articulation entirely from SAM's rigid kp3d;
the cloud contributes exactly one number per frame** — a robust body depth:

- interior **bone-tube** mask (bones dilated by per-part radii, surface = axis depth − radius),
  conf-gated; body depth = median of tube pixels, median-filtered with **k=19 > window length**
  so every VGGT seam is straddled;
- pelvis rides its kp2d ray at that depth; all other joints are SAM kp3d offsets from mid-hip,
  scaled by the (heavily smoothed) raw-per-meter ratio → **bone lengths constant by
  construction**;
- **body-size calibration from the subject**: the pelvis height is right by construction (ray ×
  calibrated depth), but the offset sizes inherit **SAM's absolute depth**, which is the
  flakiest number in the stack — the fitted body comes out uniformly small (figure: heels
  floated +0.15 above the floor, κ=1.26; vera: +0.28, κ=1.42). Fix: uniform rescale about
  mid-hip so stand-window heels land on z=0 (`MR_STAND0/1`). After it, heels ≈ 0 in every phase
  (walk / stand / crouch) with pelvis height preserved. **κ is persisted in `fit3d.npz`** — it
  is a *world-space-only* calibration; anything that reprojects `joints_w` back onto the video
  must divide it out (§7.7);
- degenerate SAM frames (kp2d span > 1.5× frame size, e.g. figure f05188) are skipped at load;
- results: bone scatter 20–79 % → **4–6 %** on the robot, 1–2 % on a human (SAM's native
  domain); pelvis steps 0.2 cm with no seam signature.

### 7.5 Smoother

Per joint/coordinate in **world space**, after the per-frame world transform (in camera space a
moving camera's shake would leak into the joints): **7-frame median** (kills single-frame SAM
spikes/flips) + **11-frame moving average** (~0.45 s). Walking/reaching dynamics survive;
work-phase pelvis steps drop to 0.3 cm within windows, seam artifact below natural motion level.

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
  skeleton), `recon_points.mp4` / `depth_maps.mp4` (recon QA), `collate_video.py` — one-pass
  composition straight from the cached assets with the fitted skeleton reprojected on the raw
  pane (2×2 grid for landscape sources / 1×4 row for portrait, `LAYOUT=grid|row`). Rendering is
  GPU-fast (`fastvid.py`: NVENC writer + torch splats) and **worker-parallel** (`WORKERS=8`
  chunks concatenated via `-c copy`; figure collate 14 min → ~2 min). NVENC caps frames at
  4096 px per side — portrait pane stacks must go horizontal.
- Honest limits: absolute trajectory depth is ±0.3 m-class (both estimators weak, corr 0.31 —
  keep chord alignment); the entry walk is under-measured; SAM per-frame scale wobble survives
  smoothing at low frequency.

### 7.7 Reprojecting `fit3d` onto the video — the scale invariant

Diagnosed from "the overlay skeleton is shorter than the subject" (mild on figure, prominent on
vera_short). The on-screen size of the reprojected skeleton factorizes **exactly** as

```
overlay / kp2d = κ × (f_vggt / f_sam)
```

with the mid-hip pinned to the subject (the pelvis rides its kp2d ray, so its pixel is exact to
~1 px and the error appears as feet floating / head sitting low — which reads as "shorter").

- **What cancels and can never cause it**: the `MR_PELVIS_H` lift scale is baked into
  `M_c2w`/`c_c2w` and exactly inverted by every renderer's world→camera map — changing it cannot
  fix or break the overlay. Pixel-space mismatch was also refuted (processed frames = raw video
  size in both scenarios).
- **What does not cancel** (both fixed in `collate_video.py` + `pose_over_cloud.py` via
  `body_fix()` — a rescale about the camera-frame mid-hip by `f_sam·(w/FW) / √(fx·fy) / κ`):
  1. **Focal substitution**: the fitted offsets subtend **SAM's angular sizes** (`r_s = d_s /
     z_sam` construction, §7.4), pixel-correct only under SAM's fixed heuristic focal
     (**1468.6 px**, recovered from the caches at 0.00 px residual) — but the renderers project
     with **VGGT's `pose_enc` focal**. `f_vggt/f_sam`: figure **0.910**, vera_short **0.619** —
     this single ratio was the mild-vs-prominent axis.
  2. **κ** (§7.4): legitimate in world space (BEV, gait metrics — do NOT undo it there), but it
     rides into the reprojection un-inverted (figure ×1.256, vera_short ×1.418; recovered for
     pre-existing `fit3d.npz` from world-vs-SAM bone lengths and stamped in).
- Measured overlay/kp2d span after the fix: figure 1.15 → **1.01**, vera_short 0.87 → **0.99**
  (residual = smoothing wobble).
- **Irreducible residual**: MHR-70 has no head-top keypoint — the skeleton tops at the nose, so
  it under-covers the subject's silhouette by construction (figure: kp2d spans 0.71 of the
  detector bbox). Cosmetic, not a scale error; extrapolate a head-top point if it matters.

### 7.8 SOMA-style retarget: fit3d → G1 replay in MuJoCo (`soma_retarget.py`)

Stage 1 of the gait.md pipeline, from the video reconstruction instead of Kimodo:

- **Source = the 3D poses** (`fit3d.npz`): articulation from SAM, trajectory from the VGGT
  world calibration (already G1 scale). SAM alone is enough for *directions* but not for the
  trajectory (`cam_t` wobbles ±0.36 m standing still) — and its chirality ambiguity hits every
  3D consumer regardless of source (§7.1).
- **Direction transfer** (per frame, yaw-normalized about the hip line like `pose_retarget.py`)
  → G1-34 joint-position targets with exact G1 bone lengths, placed at the fit3d mid-hip.
  Leg-splay damp 0.55 and 15 % leg straightening carried over. A **residual-yaw assert** guards
  the normalization: the original sign bug here (rotating by −yaw, i.e. ~180° off in the
  work phase where the hip line sits at yaw ≈ 90°) produced arms-overhead nonsense.
- **Root analytically** (mid-hip + orthonormal hip/torso frame; pelvis origin sits 0.1027 above
  the hip line in the XML), **29 hinge dofs by damped-least-squares IK** on the real G1 MuJoCo
  model: 10 tracked bodies (knees/ankles ×2.0/elbows/wrists ×1.5/shoulders ×1.2) + toe points
  fixed in the ankle frames, XML joint limits clamped per iteration, warm-started; ~0.04–0.06
  weighted residual. Then gap interpolation, 5-median+7-box qpos smoothing (hemisphere-aligned
  quats), soft ground clamp.
- **Renderers** (`render_replay.py`, vertical video-over-MuJoCo): `SCENE=plain` puts the camera
  at the reconstructed video camera (lift3d pose + VGGT fovy) — the robot appears where the
  real one is; `SCENE=kitchen` translates the trajectory into the reproduction kitchen by
  (scene work spot − `kitchen_fit.json` pocket) — both frames are +X = work-facing by
  construction — with the dishwasher door/rack held open.
- **Chirality (v2, SOLVED here)**: articulation comes from RAW per-frame kp3d (cached
  `raw58.npz`), NOT from fit3d whose smoothing blended mirror flickers irrecoverably. Per
  frame both hypotheses (raw / mirrored) are computed numerically and a global 2-state Viterbi
  selects the sequence — velocity unaries anchored to fit3d's mirror-invariant pelvis path,
  hip-line continuity pairwise, switch penalty. No polarity convention exists to invert
  (hypothesis selection, not sign correction). Smoothing runs AFTER selection. fit3d
  contributes only the trajectory + scale chain (`a`, kappa). Result: 2 heading jumps
  >45°/frame (was 23; max 161°→54°), 90 % walk-frame forward-velocity alignment (residual =
  genuine pivots), 898/4970 frames un-mirrored.
- **Rotation targets from the rig ("retarget from mesh", v5)**: the MHR mesh is a
  deterministic function of `joint_rots` — so mesh-retargeting = consuming the rotation field,
  which carries what bone directions cannot: twist about the bone (all 3 wrist dofs) and true
  torso orientation. Rig joints identified geometrically from `joint_coords` (pelvis 1 ==
  `global_rot`, chest 112, wrists 41/77; stable across frames). Two failure modes bracket the
  correct transfer: GLOBAL deltas double-count the chain (the analytic root already encodes
  torso lean → waist clamps at ±30° and least-squares recruits yaw/roll → twisted bows;
  global wrist targets demand the whole arm chain's rotation from 3 wrist dofs → 25–49 %
  saturation); RIG-LOCAL right-composed deltas assume the rig joint's local axes match the G1
  link's (they don't → garbage targets, body-wide saturation). The convention-free form is the
  **world-axes relative rotation** `Q = R_child @ R_parent^T`; transfer `ΔQ = Q(n) @ Q(ref)^T`
  (ref = stand window) composed onto the G1's OWN per-frame root/torso. Mirror hypothesis for
  rotations = reflection conjugate `S R S`, S = diag(1,1,−1), + L/R wrist swap, driven by the
  same Viterbi flips. Weights 0.6 torso / 0.5 wrists, below the position targets. Sustained
  waist-pitch saturation during the work phase is CORRECT — the video robot bends past G1's
  ±0.52 rad; the excess lives in the root pitch.
- **Kitchen placement**: self-anchoring — the robot is planted at the work pocket for
  t ≈ 12–190 s, so the replay's own work-phase median is the recon pocket
  (`kitchen_fit.json`'s pocket went stale when the reconstruction frame changed and put the
  robot ~9 m off-camera).
- **Chirality is a per-LIMB, per-frame problem and SAM's 2D head is the reliable witness.**
  The original 2-state Viterbi (raw / whole-body mirror) mirrored ALL of vera_short and 606
  figure frames - visibly swapping left/right arms in every downstream retarget - because
  SAM's wrong 3D interpretations are internally self-consistent: no internal anatomical
  check (facing-vs-velocity, wrist-side-of-body) can catch a consistent mislabel, and
  whole-body mirrors cannot fix SAM's per-limb failures at all. Measured kp3d-vs-kp2d
  lateral agreement: figure 98-100% everywhere (SAM's 3D was right; the mirrors were OUR
  error), vera wrists 98% but ankles 70% (per-frame leg-only flips). The fix in
  `soma_retarget.py`: 7 hypotheses (raw, swap+z-mirror, swap+x-mirror, yaw-180, whole-body
  label swap, ARM-pair swap, LEG-pair swap) + a per-frame 2D-witness unary scoring each
  hypothesis's predicted image lateral order (wri/ank/sho/hip, dead-banded) against kp2d,
  which is appearance-driven and chirality-reliable even when the 3D head mirrors. Result:
  vera 365/365-mirrored -> 335 raw + 30 leg-swap; figure 4932 raw + 38 lateral; raised-arm
  side verified against video frames on both scenarios (kp2d overlays + replay renders).
- **Eval-driven convergence** (`eval_retarget.py`: mocap = corrected MotionRecon joints vs FK
  of the replay; MPJPE global/local, per-joint, bone directions, heading, lag, foot skate,
  10 s timeline; guards against stale artifacts — a crashed retarget leaves the previous csv
  in place, which silently poisoned two earlier "results"). Optimization trace (MPJPE local):
  rotation-targets 19.8 cm → drop rotation targets 12.0 → drop leg heuristics 11.9 → static
  reachable arm targets 11.6 → **dynamic arm anchoring 9.8** → **dynamic legs 8.4 cm**
  (p95 15.8, every bone direction ≤ 3.1°, heading 3.2°, foot skate 0.03 m/s)
  → **uniform target rescale to G1 size 4.18 cm** (p95 8.4; `SOMA_SCALE=auto`, the default:
  pelvis-plateau → 0.72 m, applied to trajectory + articulation chain). What had looked like
  a "morphology floor" was mostly reachability: targets lived at fit-world scale (figure
  0.826 m pelvis, human subjects 0.93), so the IK chased a larger skeleton. Cross-scenario
  (all local MPJPE, auto scale): figure 4.18 cm, xpeng walking 4.51 cm, vera_short (human,
  fast dance turn) 6.2 cm — vera's residual heading error (13° mean / 50° p95) is turn lag,
  not scale.
  The decisive idea: **dynamic anchoring** — child position targets (elbow/wrist, knee/ankle)
  are rebuilt EVERY IK iteration from the live parent link along the mocap bone directions
  with XML segment lengths. Static pre-computed targets assume each parent hits its own
  target; every parent miss cascades down-chain as direction error (measured: ankle
  target→achieved 1.2 cm vs arms 15–20 cm before the change; forearm direction 50°→2.5°).
  Limb tracking is thereby pure DIRECTION matching — the correct objective across mismatched
  morphologies: the mocap's arms are ~30 % longer than G1's (23.8 vs 18.4 cm forearm), so
  absolute limb-end positions have an irreducible proportional floor (~10–13 cm) while
  directions converge to ~1–3°.
- **The rig-rotation targets were net-negative and are OFF by default** (SOMA_W_TORSO/WRIST=0):
  global deltas double-count the chain, rig-local deltas assume axis conventions that do not
  hold, and the world-relative form still lost to position-only on the eval (19.8 vs 12.0).
  The leg splay-damp/straightening heuristics are likewise off (SOMA_LEG_SPLAY=1, STRAIGHT=0).
- **Honest gaps**: limb-end absolute positions carry the ~30 % proportion gap (directions are
  matched instead); G1 bends less than the Figure robot (waist ±0.52 rad; morphology); hands
  are rigid mittens vs the video's articulated fingers; ~10 % of walk frames keep ambiguous
  facing through pivots.
- **NVIDIA soma-retargeter A/B** (fit -> SOMA BVH -> their Newton/Warp IK -> G1 CSV; bridge
  modules `export_soma_bvh`/`import_soma_csv` in humanoid_motion_recon, all conventions
  calibrated from their sample BVHs): local MPJPE figure 16.8 cm / xpeng 8.0 / vera 15.9
  vs ours 4.2 / 4.5 / 7.3 on the same corrected mocap targets; heading xpeng 1.4 deg vs
  ours 3.4 / 3.2. Their dominant residual is a CONSTANT ~24 deg work-phase yaw offset:
  the feet-stabilized root undershoots the fast walk-in pivot and never recovers once
  feet pin (pre-chirality-fix numbers looked better only because the mirrored walk-in
  made the apparent pivot smaller). Foot skate 0.02 (matches ours). Four debugging lessons,
  each isolated by a ROUND-TRIP harness (re-encode their own sample's joint positions
  through our exporter, retarget, diff vs their reference CSV - final agreement 3.6 deg
  mean DOF, so the format layer is provably faithful): (1) the SOMA Hips joint is NOT
  mid-hip (sits ~8.5 cm above the hip line) - conflating them inflated the skeleton 9 %
  and their proportional IK crouched + pitched 27 deg; (2) fit3d articulation (SAM mirror
  flickers smoothed in) makes the robot pivot ~140 deg on mirrored stretches - export from
  the chirality-corrected mocap (`MR_MOCAP_NPZ`, soma_retarget's Viterbi output); (3) limb
  TWIST must be reference-anchored (`minrot(d_ref->d_meas) @ G_ref`) - anchoring at the
  rig's zero orientation scrambles hip yaw ~150 deg because their IK reads link rotations
  (round-trip DOF error 27.5 -> 3.6 deg); (4) their per-frame smoothing objective is tuned
  on 120 fps SEED data - upsample the BVH to 120 fps (`BVH_FPS`, resample back on import)
  or low-fps input over-smooths in wall time (xpeng heading 12.3 -> 1.4 deg). Residual gap
  vs our retargeter: their IK executes the fit's bend-phase heel float literally
  (single-leg lifts in the figure work phase) where our dynamic anchoring + ground clamp
  absorb it, and fast turns keep p95 heading tails (~50-60 deg).

### 7.9 Mesh lifting + GPU rendering infrastructure

- **World-lifted, time-smoothed meshes** (`lift_mesh.py`): every frame's 18439-vertex SAM mesh
  goes through the IDENTICAL transform chain as the skeleton fit — pelvis on the kp2d mid-hip
  ray at the fitted body depth `b[n]`, offsets scaled by the fitted ratio `a[n]`, per-frame
  `M_c2w`, kappa rescale about the pelvis — all read back from `fit3d.npz`/`lift3d.npz`, no
  recomputation. Constant topology ⇒ vertices are in correspondence across frames and smooth
  exactly like joints (same 7-median + 11-box, batched per-vertex on GPU). Output
  `<DEPTH_WORK>/mesh_w.npz` (cache dir — ~0.5 GB f16 for figure). `mesh_world_video.py`
  renders [frame + reprojected smoothed mesh | fixed 3/4 world view over the scene cloud];
  the reprojection pane must apply the §7.7 `body_fix` (divide pelvis-relative offsets by
  `kappa`, focal ratio F_SAM/√(fx·fy)) — verified: mesh/kp2d span 0.84 pre-fix → 0.95–0.98
  post-fix on vera_short. Figure caveat: SAM's mirror flicker (§7.1) smears the smoothed mesh
  inside flicker runs until chirality is resolved downstream.
- **GPU rendering** (`gpurender.py` + ports): all video renderers moved to torch-CUDA —
  point-cloud splats in painter's order, resizes, colormaps, pane compositing on device; NVENC
  encode via `fastvid`; skeleton/text overlays stay on PIL/skel_draw (cheap, already correct,
  now drawn in bbox crops). Verified frame-matched vs the CPU references on vera_short (365
  frames): birdseye 52→7 s (the CPU version was also O(N²) in the trail — now incremental),
  collate 22→13 s (57 ms/frame single-proc, 4.5×), pose_over_cloud 90→24 s, recon_check
  47→17 s, sam2d 7 s; `depth_video.py` no longer needs torchvision (runs in the kimodo env).
  Known accepted deltas: bilinear-antialias resize (vs PIL bicubic) and float z-order splat
  ties — both below the NVENC noise floor.
- **Torch mesh rendering** (`gpurender.sample_mesh`/`mesh_splat`): pyrender replaced by
  surface splatting — one-time area-weighted barycentric sampling of the constant-topology
  MHR mesh (~300–900k points by density), then per frame: barycentric gather, face-normal
  two-sided headlight Lambertian, perspective project, painter-ordered z-splat. 6.6 ms/frame
  at density 8 vs ~400–460 ms pyrender. Ports verified frame-matched: `mesh_world_video.py`
  80→13.5 s, `sam3d_mesh_video.py` 169→9.7 s on vera_short (figure full-rate ~110 s vs
  ~40 min). Both mesh renderers now run in the kimodo env (no pyrender/cv2); accepted deltas:
  splat stipple vs smooth Phong, hard edges vs AA. Every `figure/` video renderer is now
  torch-CUDA end-to-end (decode → composite → NVENC).

### 7.10 Pipeline speed: ≤8× real time on one RTX 3090

Benchmark: 1 minute of figure.mp4 (1438 frames, 1280×720 @ 23.976), full MotionRecon
(SAM-3D-Body → VGGT-Omega → lift → fit) on a single RTX 3090. Baseline ~25× real time;
optimized **6.3×** (goal ≤8×):

| Stage | Baseline | Fast (default) | Wall (1-min clip) |
|---|---|---|---|
| SAM-3D-Body | `full` sequential, 582 ms/f | `SAM_INFER=body` + batch 16: 48–73 ms/f | **1:45** (`full` batched: 4:46) |
| VGGT-Omega | 512/stride 120, 328 ms/f eff. | stride 150 + `VGGT_FSTRIDE=2` + NN fill-in | **3:59** |
| lift_skeleton | — | — | 0:11 |
| fit_pose | — | — | 0:22 |
| **total** | ~25 min | | **6:17** (with hands: 9:18) |

- **Two SAM modes, both cross-frame batched** (`SAM_BATCH`, default 16):
  - `SAM_INFER=body` (default, 6.3× total): body decoder only. Hand kps still produced by the
    body decode but coarser (~5.6 cm; wrists up to ~13 cm on occasional frames). Body joints
    deviate ≤3.3 cm / 6.6 px vs full — the two per-hand passes are 74 % of `full`'s cost.
  - `SAM_INFER=full` (body+hands, 9.3× total; 199 vs 582 ms/f sequential): adds the per-hand
    refinement passes. To fit ≤8× with hands, drop `VGGT_FSTRIDE=3` (untested quality).
- **Cross-frame batching** rides the model's person dimension (`prepare_batch` carries a full
  image per entry, so entries can be different frames; one shared `cam_int` — same video
  resolution). Batch invariance vs the sequential path: 1 px / 6.7 mm (body mode). For `full`,
  hand crops are made frame-aware by patching `prepare_batch` in the meta_arch module —
  upstream crops all "persons" from ONE image; the patch substitutes each entry's own frame
  (flipped left-hand calls detected via negative stride). A `[pose] WARN` prints if the patch
  ever fails to engage.
- **Refined-hand gate is bistable on this robot**: upstream's wrist-angle gate
  (`thresh_wrist_angle=1.4`) flips refined↔body-decoded hands between consecutive frames in
  ~2/3 of transitions even in the sequential path (robot grippers sit at the threshold), so
  batched-vs-sequential hand kps differ up to ~20 cm on gate-flip frames while body kps stay
  ≤6 cm — the fitter's temporal smoothing absorbs this. `SAM_WRIST_THRESH` overrides the gate
  (raise → always trust refinement).
- **VGGT knobs**: `VGGT_STRIDE` 120→150 (10 shared frames are plenty for the robust-affine +
  sim(3) stitch); `VGGT_FSTRIDE=2` halves inference — downstream consumes ONE body-level depth
  per frame through a k=19 temporal median, so half-rate depth is near-lossless; skipped
  frames get nearest-neighbor fill-in npz so all consumers see a dense directory. `VGGT_RES`
  stays 512 (untested quality below).
- **bf16: measured, NOT adopted.** VGGT already autocasts bf16 internally (aggregator bf16,
  heads fp32 — upstream design). SAM with the jit MHR rig pinned fp32 (its sparse matmul has
  no bf16 kernel) runs but is no faster at batch 16 (50 vs 48 ms/f; CPU-side decode/transform
  dominates and TF32 already covers the fp32 matmuls) and adds 6 px / 8.8 mm deviation.
- **Quality gate** (fast pipeline vs the full-quality reference fit on the same 1438 frames):
  local inter-pipeline MPJPE **1.02 cm** (p95 1.87) — 4× below the retarget's own 4.18 cm
  floor (§7.8); kappa 1.258 vs 1.257; hip-line yaw diff 1.75°. Trajectory dev 7 cm mean reflects the
  two runs' independent VGGT normalization chains (60 s vs 216 s), not pose degradation.
