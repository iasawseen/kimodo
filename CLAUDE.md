# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

NVIDIA's **Kimodo** (kinematic motion diffusion model, `kimodo/` package) plus local working
layers built on top of it:

- `design/` — **the living empirical documentation.** `gait.md` (how to control G1 gait with
  constraints: speed, step length, turns, EE keyframes, runtime gotchas), `figure.md` (the
  Figure "Helix 02" video reproduction + the MotionRecon perception pipeline), and
  `humanoid_motion_recon.md` (alternatives survey + design considerations for the
  MotionRecon package). These are
  maintained findings-first documents: when you learn something new that contradicts or extends
  them, update them in the same change. Read the relevant sections before generating motion —
  most "why is the robot doing X" questions are already answered there.
- `figure/` — the Helix-02 reproduction package: constraint-driven segment generators + SE(2)
  chain-stitching (`gen_helix_kitchen.py`), scene/camera builders, collision gate, and the
  G1 retarget layer (`soma_retarget.py`, `eval_retarget.py`, `render_replay.py`).
  `design/figure.md` §0 is the file map and run order.
- **MotionRecon** (mono video → world skeletons + meshes: SAM-3D-Body + VGGT-Omega + rigid
  fitter + smoother, plus all its torch-CUDA visualization renderers) now lives in the
  standalone sibling repo `../humanoid-motion-reconstruction` — `pip install -e`'d into both conda
  envs; invoke as `python -m humanoid_motion_recon.<tool>` with the same env vars as before.
  Its README/CLAUDE.md are the authoritative pipeline docs; `design/figure.md` §7 keeps the
  Figure-scenario empirical findings.
- `tools/` — standalone G1 rendering (`render_g1.py`) and long-trajectory stitching (`stitch.py`).
- `outputs/` — generated artifacts (qpos CSVs, scene XMLs, videos, pose npz). Large; mostly
  uncommitted.
- `MotionCorrection/` — C++/pybind post-processing extension (not needed for the workflows above).

## Environments & invocation (hard-won; do not rediscover)

Two conda envs:
- **`kimodo`** — generation + rendering + all numpy/PIL/imageio work. Has mujoco, modelscope,
  matplotlib; **no cv2**.
- **`sam_3d_body`** — torch 2.12+cu126 with cv2; used to run SAM-3D-Body and VGGT-Omega
  (both live in sibling checkouts `../sam-3d-body`, `../vggt-omega`).

Generation must be run as:

```bash
CUDA_VISIBLE_DEVICES=0 TEXT_ENCODERS_DIR=$HOME/.cache/kimodo/text_encoders \
  HF_HOME=$HOME/.cache/kimodo/text_encoders/hf_home \
  PYTHONPATH=. python -m figure.gen_helix_kitchen [segment ...]   # or kimodo.scripts.generate
```

- `CUDA_VISIBLE_DEVICES=0` is **mandatory** (LLM2Vec multiprocessing crashes with >1 visible GPU).
- The Llama-3-8B text-encoder base is gated on HF; `TEXT_ENCODERS_DIR`/`HF_HOME` redirect to a
  local mirror. Gated checkpoints generally (SAM-3D-Body, VGGT-Omega, Llama) are fetched from
  **ModelScope** mirrors instead of HF.
- Headless rendering: `MUJOCO_GL=egl`. The `EGLError` at interpreter exit is harmless.
- GPU convention in this setup: GPU 0 for Kimodo generation, GPU 1 for perception (SAM/VGGT).
- `ffmpeg` in loops needs `-nostdin`. NVENC (`h264_nvenc`) is available and is the fast path for
  encoding point-cloud/noise-like video content (`humanoid_motion_recon.fastvid`); NVENC max frame dimension
  is 4096 px.

## Figure pipeline run order

```bash
PYTHONPATH=. python -m figure.gen_helix_kitchen        # GPU; per-segment cache in outputs/figure/segs/
PYTHONPATH=. python -m figure.build_scene              # scene + camera, seconds, no GPU
MUJOCO_GL=egl PYTHONPATH=. python figure/check_collisions.py outputs/figure/helix_kitchen.csv \
    outputs/figure/kitchen_g1.xml                      # QA gate: fix/re-roll seeds until PASS
PYTHONPATH=. python figure/ground_clamp.py             # foot-floor clamp
MUJOCO_GL=egl python figure/render_scene.py outputs/figure/helix_kitchen.csv \
    outputs/figure/kitchen_g1.xml outputs/figure/helix_kitchen.mp4
bash figure/build_verify.sh                            # video-vs-render verify clips
```

Gotchas: **`rm -f outputs/figure/helix_kitchen_raw.csv` before every re-chain** (ground_clamp
keeps a raw snapshot); delete `outputs/figure/segs/<name>*.npz` to force-regenerate a segment;
the scene re-anchors itself to the measured work pocket, so scene coordinates move when the
chain changes — everything downstream is pocket-relative by design.

## Architecture: what needs multi-file reading to see

- **Constraints are the control interface; text is styling only.** The figure pipeline runs
  `cfg_type="separated", cfg_weight=[0.0, 2.0]` — the prompt is a dead placeholder and all
  content comes from `FullBodyConstraintSet` / `Root2DConstraintSet` / EE keyframes
  (`kimodo/constraints.py`). Constraint tensors must be `.to(device)`'d.
- **Coordinate frames** (source of most past bugs): Kimodo motion space is Y-up, +Z forward,
  X-left; MuJoCo qpos world is Z-up, +X forward (36 cols: root xyz + wxyz quat + 29 hinges).
  `global_root_heading` rows are the **hip-line** direction `(cos θ, −sin θ)` for a facing of
  `roty(θ)`. Getting this sign wrong historically produced 2× yaw overshoots.
- **Segments are stand-bookended and chained** by SE(2) alignment + crossfade
  (`gen_helix_kitchen.chain()`); the first segment keeps its authored frame; Kimodo always
  generates segment roots starting at the origin, so measured paths are authored start-relative.
- **MotionRecon** (design/figure.md §7): SAM-3D-Body gives per-frame skeletons whose kp2d is in
  the **processed-frame pixel space** (auto-detected from `frames/f00000.jpg`); VGGT-Omega gives
  per-frame depth in large overlapping windows stitched by shared-frame affines (works for
  moving cameras). Depth over thin/moving subjects is context-inpainted — never read per-joint
  depth from it; the rigid fitter (`humanoid_motion_recon.fit_pose`) uses one body-level depth per frame + SAM's
  rigid articulation. World alignment is calibrated from the subject itself (torso-up, heels =
  floor, standing pelvis height = scale), not from scene-plane fits, which measurably fail.
  Scenario-specific settings (fps, calibration windows, pelvis height, output dir) are env vars
  (`MR_FPS`, `MR_TMIN`, `MR_STAND0/1`, `MR_YAW0/1`, `MR_UP0/1`, `MR_PELVIS_H`, `MR_OUT`).
- **Frame caches** (extracted video frames, depth npz, mhr npz) are large and live outside the
  repo in the session scratch dir; `outputs/figure/pose*/` holds only the distilled npz + mp4s.

## Repo etiquette

- This repo is a submodule of a private umbrella. Never reference umbrella content (paths,
  docs, strategy) from files committed here — the repo must stand alone.
- Commit only when explicitly asked.
- `outputs/` artifacts are heavyweight; don't `git add` them wholesale.
