# Modulating G1 gait in Kimodo

Practical guide for controlling the **Unitree G1** locomotion produced by `kimodo-g1-rp`.
Everything here is empirical — measured from generated trajectories, not from docs.

> **One-line mental model:** the **text prompt** is a *coarse, semantic* lever that saturates and
> silently substitutes off-distribution requests; **kinematic constraints** are the *exact* lever.
> Use text for *what* the motion is, constraints for *precise numbers* (speed, step length, path).

---

## 0. Decision table — "I want X"

| Goal | Lever | Accuracy | Notes |
|---|---|---|---|
| A named action (walk, turn around, start/stop) | **prompt** | good | densely-represented actions are faithful |
| Relative speed ("slowly", "very slowly") | **prompt** | coarse, **floored at ~0.4 m/s** | saturates; can't go slower |
| Exact **forward speed** (any value) | **Root2D path constraint** | exact on Kimodo root; **calibrate for G1 root** | no floor; down to ≤0.05 m/s |
| Exact **step length** | **foot end-effector constraint** | ~1 cm (steady state), 10–40 cm | pins footfall positions |
| **Torso lean / forward incline** | **FullBodyConstraintSet** (positions) | ~2–3° up to 30° | pitch upper body forward; pins whole body |
| Exact **path / turn radius / waypoints** | **Root2D path constraint** | exact | (x,z) ground trajectory |
| A specific speed *and* step length independently | **Root2D + foot constraints** together | — | decouple via footfall timing |
| A number like "0.1 m/s", "40 cm steps" in the **prompt** | ❌ **ignored** | — | model reads meaning, not units → use constraints |

---

## 1. Prompting (the text lever)

Kimodo mean-pools the whole prompt into **one 4096-d vector** (bidirectional LLM2Vec over Llama-3-8B),
so word order/composition is weak and numbers are meaningless. Empirical rules:

### What the prompt controls well
- **Named, common actions.** `"turn around"` → **180° ± 1°** rotation in place (net translation ~0.2 m).
  Discrete canonical actions are reproduced faithfully.
- **Relative speed via adverbs**, but only *coarsely and monotonically down to a floor*:

  | prompt | cruise speed | peak |
  |---|--:|--:|
  | `go forward` | 1.37 m/s | 1.75 |
  | `go forward slowly` | 0.95 | 1.19 |
  | `go forward very slowly` | 0.46 | 0.66 |
  | `go forward extremely slowly` | 0.43 | 0.62 |

  `very slowly` ≈ `extremely slowly` — **saturated at a ~0.4 m/s walk floor.** Below that the model
  won't produce a sustained slow glide (it's off-distribution); it either shortens the walk or ignores you.
- **Start/stop *envelopes*** from a *single* sentence. `"start walking slowly go forward stop slowly"`
  yields a clean accelerate→cruise→decelerate **bell curve** (e.g. 0.08→1.2→0.17 m/s) — the diffusion
  decoder lays "start" at the beginning and "stop" at the end without being told *when*.
  - Intensifiers lower the peak: cruise-phase `slowly`→1.11, `very slowly`→0.86, `extremely slowly`→0.65 m/s.
  - More "slowly" → *fuller* stop (end speed 0.17→0.05→0.02) and a temporally *shorter* walk.
  - **A single sentence beats a multi-prompt timeline for this.** `multi_prompt=True` with
    `["start walking slowly","go forward","stop slowly"]` came out *worse* — transition blending
    smears the boundaries so it never actually stops. Multi-prompt is for stitching genuinely
    *different* motions (walk→wave→sit), not for shaping one action's velocity.

### What the prompt does NOT control (silently substituted → a plain walk)
- **Numeric targets.** `"go forward at 0.1 m/s"` → ~1 m/s. Units are ignored entirely.
- **Alternative gaits.** `"shuffle forward"` → an ordinary walk (foot-lift 14 cm, identical to normal).
  No shuffle/slide/drag in the distribution for G1.
- **Intermittent structure.** `"go forward with pauses"` → one continuous walk, no mid-motion pauses.
- **Sub-floor cruise.** Anything asking for < ~0.4 m/s sustained.

### Compositional prompts *average*, they don't stack
Two style words blend toward the **midpoint** of their individual effects:
`very slowly` (0.43) + `short steps` (0.91) → `very slowly with short steps` = **0.72** ≈ mean(0.43, 0.91).
So "very slowly **and** short steps" won't give you the slowest *and* shortest — it compromises.

### Prompt gotcha
The CLI (`kimodo_gen` / `scripts.generate`) **splits the prompt on `.`** and re-joins with periods,
so `"...0.1 m/s"` becomes two garbage sub-prompts. **For any prompt containing a decimal/period,
bypass the CLI** and call `model([prompt], [num_frames], ...)` with the string as a single element.

---

## 2. Exact forward speed — `Root2DConstraintSet`

Pin the **smoothed root's (x, z) ground trajectory** to a constant-velocity line. No speed floor;
works down to ≤ 0.05 m/s, and the gait *adapts* (foot-lift shrinks to ~3 cm — the near-shuffle text can't summon).

```python
import torch
from kimodo.constraints import Root2DConstraintSet
# Kimodo frame: Y up, +Z forward, X lateral.  Heading (cos,sin)=(1,0) faces +Z.
def root_path(skel, num_frames, speed_mps, fps, device):
    f = torch.arange(num_frames, device=device)
    z = speed_mps * f.float() / fps                       # constant velocity, +Z
    x = torch.zeros(num_frames, device=device)
    root2d  = torch.stack([x, z], dim=1)                  # [N,2] = (x_lateral, z_forward), METERS
    heading = torch.tensor([[1.0, 0.0]], device=device).repeat(num_frames, 1)
    return Root2DConstraintSet(skel, f, root2d, global_root_heading=heading).to(device)  # NOTE .to(device)

out = model(["go forward"], [num_frames], constraint_lst=[root_path(model.skeleton, num_frames, 0.25, fps, device)],
            num_denoising_steps=100, num_samples=1, multi_prompt=False, post_processing=False,
            cfg_type="separated", cfg_weight=[2.0, 2.0], return_numpy=True)
```

Measured: Kimodo root tracks target near-perfectly (0.20→0.199, 0.10→0.098, 0.05→0.050 m/s).

### The retarget gap — calibrate for the **actual G1 (MuJoCo qpos) root**
The Kimodo root hits target, but the qpos/G1 root (post-`MujocoQposConverter`, G1 post-processing off)
runs **short**, and the gap is **speed-dependent**:

| desired **G1** speed | Kimodo-path target to use | compensation | uncalibrated G1 result |
|--:|--:|--:|--:|
| 0.30 m/s | 0.304 | +1.3% | 0.296 |
| 0.25 m/s | 0.252 | +0.8% | 0.248 |
| 0.20 m/s | 0.2176 | +8.8% | 0.182 |
| 0.10 m/s | (large) | ~+30% | 0.076 (−24%) |
| 0.05 m/s | (large) | ~+25% | 0.040 (−20%) |

There is a **knee between 0.25 and 0.20 m/s**: above ~0.25 the retarget is ~1:1; below it the model
shifts to a tiny-step gait that retargets poorly. **Do not reuse a compensation factor** — re-calibrate
per target. Secant loop (1–3 iterations) that measures the qpos root and adjusts:

```python
# measure(T): build root_path at Kimodo speed T, generate, qpos = conv.dict_to_qpos(out, device),
#             return |qpos[:, :2][-1] - qpos[:, :2][0]| / duration   (MuJoCo ground plane, +X forward)
T = target                                   # e.g. 0.25
for _ in range(6):
    a = measure(T)
    if abs(a - target) < 0.0015: break
    T = T * target / a                       # linear extrapolation; secant if you keep 2 points
# -> converges to the Kimodo target that makes the G1 root hit `target` exactly
```

If you need the *physical* G1 speed exact, either calibrate as above or add foot constraints to
tighten the retarget.

---

## 3. Exact step length — `LeftFootConstraintSet` / `RightFootConstraintSet`

Pin **footfall positions** on a forward grid. `"LeftFoot"` expands to constrain the **positions of
`left_ankle_roll_skel` + `left_toe_base`** and the **rotation of `left_ankle_roll_skel`**, plus root_y
and heading (all read from the positions array — so hips `[8,1]` and root `[0]` must be sensible).

Hand-building a valid body array is fragile. **Robust recipe: reuse real foot poses from an existing
walk and translate them onto the target grid** (preserves natural foot geometry + rotation + hips):

```python
import numpy as np, torch
from kimodo.constraints import LeftFootConstraintSet, RightFootConstraintSet

ref  = np.load("outputs/go_forward.npz")                      # any prior G1 walk
pj   = torch.tensor(ref["posed_joints"],  dtype=torch.float32, device=device)   # [T,34,3] Kimodo frame
grot = torch.tensor(ref["global_rot_mats"], dtype=torch.float32, device=device) # [T,34,3,3]
fc   = ref["foot_contacts"].astype(bool)                      # [T,4] = L-heel,L-toe,R-heel,R-toe
skel = model.skeleton
# a fully-planted stance frame per foot (source pose for that foot's footfalls)
Lidx, Ridx = np.flatnonzero(fc[:,0]&fc[:,1]), np.flatnonzero(fc[:,2]&fc[:,3])
Lref, Rref = int(Lidx[len(Lidx)//2]), int(Ridx[len(Ridx)//2])   # a mid stance frame

def foot_constraint(cls, refframe, ankle_name, entries, STEP):   # entries: [(frame, step_index), ...]
    base_p, base_r = pj[refframe].clone(), grot[refframe].clone()
    z0 = float(base_p[skel.bone_index[ankle_name], 2])
    frames, poss, rots, sr2d = [], [], [], []
    for f, si in entries:
        pose = base_p.clone(); pose[:, 2] += si*STEP - z0        # translate whole body in +Z onto grid
        frames.append(f); poss.append(pose); rots.append(base_r)
        sr2d.append(pose[skel.root_idx][[0, 2]])
    return cls(skel, torch.tensor(frames, device=device),
               torch.stack(poss), torch.stack(rots), torch.stack(sr2d)).to(device)

STEP = 0.30                                                      # target step length (m)
SCHED = [(15,"L",0),(45,"R",1),(75,"L",2),(105,"R",3),(135,"L",4)]   # ~1 Hz cadence, alternating
cons = [foot_constraint(LeftFootConstraintSet,  Lref, "left_ankle_roll_skel",
                        [(f,si) for f,s,si in SCHED if s=="L"], STEP),
        foot_constraint(RightFootConstraintSet, Rref, "right_ankle_roll_skel",
                        [(f,si) for f,s,si in SCHED if s=="R"], STEP)]
out = model(["go forward"], [num_frames], constraint_lst=cons, num_denoising_steps=100,
            num_samples=1, multi_prompt=False, post_processing=False,
            cfg_type="separated", cfg_weight=[2.0,2.0], return_numpy=True)
```

Measured **steady-state accuracy ≈ 1 cm** over a 4× range:

| target | steady-state (last 2 steps) | mean of 5 steps |
|--:|--:|--:|
| 40 cm | 39.5 | 39.4 |
| 30 cm | 31.0 | 31.2 |
| 25 cm | 26.2 | 26.7 |
| 20 cm | 21.4 | 22.1 |
| 15 cm | 16.2 | 17.4 |
| 10 cm | 10.6 | 12.2 |

Caveats:
- **First step from standstill overshoots**, then converges within 2–3 steps (10 cm: `15.1→12.6→10.7→10.5`).
  The model ramps *into* the short-step regime — crop/ignore the first step, or start from a moving pose.
- **Soft floor below ~15 cm** (small growing positive bias) — same distribution edge as the speed floor.
- The constraint is **sparse** (footfall frames only); the model fills transitions and picks the between-step pose.

---

## 4. Torso lean / forward incline — `FullBodyConstraintSet`

There is **no spine-angle constraint** and no way to constrain a single torso joint: `expand_joint_names`
only knows the 5 end-effector groups (feet/hands/hips). The only lever that reaches the torso is
`FullBodyConstraintSet`, which pins joint **positions** (its `update_constraints` explicitly drops the
rotations — *"global rotations are not used here"*). So express a lean **geometrically**: put the
upper-body joints forward of the pelvis.

G1 joint groups: `0` pelvis · `1–14` legs · `15–17` waist (`yaw/roll/pitch`) · `18–25` left arm ·
`26–33` right arm. **Upper body = 15–33.** The heading joints `[8,1]` are leg hips, so a torso lean
leaves heading untouched.

**Recipe:** pitch the upper-body joints forward about the pelvis by θ (rotation about the lateral X
axis), then pin the leaned pose with `FullBodyConstraintSet` at keyframes:

```python
import numpy as np, torch
from kimodo.constraints import FullBodyConstraintSet

ref  = np.load("outputs/go_forward.npz")                       # the gait to lean (legs come from here)
pj   = torch.tensor(ref["posed_joints"],  dtype=torch.float32, device=device)   # [T,34,3]
grot = torch.tensor(ref["global_rot_mats"], dtype=torch.float32, device=device) # [T,34,3,3] (required, unused)
UPPER = list(range(15, 34))                                    # waist + both arms
th = np.radians(30); c, s = np.cos(th), np.sin(th)             # +deg forward lean, -deg backward lean
Rx = torch.tensor([[1,0,0],[0,c,-s],[0,s,c]], dtype=torch.float32, device=device)  # pitch about lateral X
pivot  = pj[:, 0:1]                                             # pelvis
leaned = pj.clone()
leaned[:, UPPER] = pivot + torch.einsum("ij,tkj->tki", Rx, pj[:, UPPER] - pivot)
KF   = torch.arange(0, num_frames, 10, device=device)          # keyframe every 10 frames
cons = [FullBodyConstraintSet(skel, KF, leaned[KF], grot[KF]).to(device)]
out  = model(["go forward"], [num_frames], constraint_lst=cons, num_denoising_steps=100,
             num_samples=1, multi_prompt=False, post_processing=False,
             cfg_type="separated", cfg_weight=[2.0,2.0], return_numpy=True)
```

Measured torso pitch (pelvis→`[waist_pitch, shoulders]` vector, sagittal plane) over a 4° upright baseline.
**θ > 0 leans forward, θ < 0 leans backward**, and the two directions track symmetrically — the model
relaxes toward upright by the same amount either way (~1.5° / 3.5° / 7° at 15 / 30 / 45°), giving a usable
range of about **−34° to +42°** torso pitch:

| target θ | achieved | note |
|--:|--:|---|
| −45° | −33.9° | ~7° relax — resists the extreme |
| −30° | −22.2° | tracks well |
| −15° | −9.3° | tracks well |
| +15° | +17.6° | tracks well |
| +30° | +30.6° | near-exact |
| +45° | +41.8° | ~7° relax — resists the extreme |

Walk speed is unchanged (~1.07 m/s) in either direction — the lean rides on top of the reference gait.

Caveats:
- **Kinematic only — no balance check.** Kimodo has no physics/CoM model, so it will happily produce a
  pose (e.g. leaning back 34° while walking forward) that shifts the CoM off the support polygon. The
  qpos retarget doesn't enforce stability either — whether such a pose is trackable on hardware is a
  separate question for the downstream controller.
- **Rigid "bow from the hips"** — the whole upper body pitches about the pelvis. For a specific spine
  *curvature* (bend only at `waist_pitch` vs. lean the whole torso), change the pivot / joint subset you
  rotate. Other axes reuse the same trick: rotate about Z for a **lateral lean**, about Y for a **twist**.
- **`FullBody` pins *all* joints**, so it also locks the legs to the reference walk you leaned. The lean
  rides on that gait — to lean a slow / short-step walk, build `leaned` from *that* reference. There is
  **no "torso-only, legs free"** path in the API (only the 5 EE groups or the full body).
- Keyframe density trades faithfulness vs. freedom: every ~10 frames holds the lean tightly; sparser lets
  the model relax toward its natural posture between keyframes.

---

## 5. How speed, cadence, and step length relate

`speed = step_length × cadence`. The model **self-selects cadence ≈ 1 Hz** at slow constrained speeds.
Consequences:
- Fixing speed (§2) with the natural ~1 Hz cadence gives you **step_length (cm) ≈ speed (m/s) × 100**
  (0.30 m/s → ~30 cm, 0.25 → ~25 cm). So the speed constraint *also* sets step length for free — but *coupled*.
- To **decouple** (e.g. 40 cm strides at 0.15 m/s, or 20 cm mincing steps at 0.4 m/s): use foot
  constraints (§3) and change the **footfall frame spacing** (cadence) independently of the grid spacing
  (step length). Combine with a Root2D path if you also want the root velocity pinned.

---

## 6. Runtime setup & gotchas (read before running anything)

**Environment** (see also the machine-specific run notes):
- conda env `kimodo` (torch 2.12 cu13, transformers 5.1.0, mujoco, modelscope). Run from repo root with
  `PYTHONPATH=<repo>` (package isn't pip-installed; `python -m kimodo.scripts.generate` also works).
- Text encoder base (Llama-3-8B) is redirected off the gated HF repo via `TEXT_ENCODERS_DIR` +
  patched adapter configs, and `HF_HOME` points at a user-owned cache. (Machine-specific; already set up.)
- **`CUDA_VISIBLE_DEVICES=0` is mandatory.** `LLM2Vec.encode` spawns a multiprocessing pool when >1 GPU
  is visible and the workers crash under `python -`. One 24 GB GPU holds the 16 GB encoder + G1 diffusion.
- Rendering: headless MuJoCo, `MUJOCO_GL=egl`. Load `kimodo/assets/skeletons/g1skel34/xml/g1.xml`
  (ships a floor+skybox), set `model.vis.global_.offwidth/offheight` before `mujoco.Renderer` (default FB
  is 640×480). The `EGLError` at interpreter exit is harmless GC cleanup.

**Coordinate frames (critical):**
- **Kimodo** (posed_joints, constraints, smooth_root): **Y up, +Z forward, X lateral.** Forward heading
  `(cos,sin) = (1, 0)`.
- **MuJoCo qpos** (G1 CSV, `MujocoQposConverter`): **Z up, +X forward, Y lateral.** qpos = `[x,y,z, quat(wxyz), 29 hinges]` (36 cols).
- Constraints are specified in the **Kimodo** frame.

**Model call defaults that matter:**
- `multi_prompt=False` for a single continuous action; `post_processing=False` (G1 post-proc is disabled anyway).
- `cfg_type="separated", cfg_weight=[2.0, 2.0]`, `num_denoising_steps=100`, `seed_everything(seed)` for determinism.
- Constraints **must** be `.to(device)`'d (joint-index tensors are built on CPU in `__init__`).

**Constraint construction quick-ref:**
- `Root2DConstraintSet(skel, frame_indices[N], smooth_root_2d[N,2]=(x,z), global_root_heading=[N,2] | None)`
- `LeftFootConstraintSet | RightFootConstraintSet(skel, frame_indices[N], global_joints_positions[N,34,3], global_joints_rots[N,34,3,3], smooth_root_2d[N,2])`
- `FullBodyConstraintSet(skel, frame_indices, global_joints_positions, global_joints_rots, smooth_root_2d=None)` — pins *all* joint positions + root heading at keyframes (positions+heading only, not per-joint rotation). Used for torso lean (§4).
- Save/load JSON via `constraints.save_constraints_lst` / `load_constraints_lst(path, skel)`.

---

## 7. Measurement recipes (for verifying your control)

```python
# forward speed (G1 / MuJoCo qpos): planar displacement / duration
xy = qpos_csv[:, :2];  speed = np.linalg.norm(xy[-1]-xy[0]) / duration        # +X forward, Y lateral

# forward speed (Kimodo root): smooth_root_pos is [T,3], +Z forward
speed = abs(npz["smooth_root_pos"][-1,2] - npz["smooth_root_pos"][0,2]) / duration

# heading change (turn): yaw from qpos quat [w,x,y,z] about MuJoCo Z
yaw = np.degrees(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z)));  total = np.unwrap(np.radians(yaw))[-1]...

# foot-lift (shuffle detector): posed_joints [T,34,3], Y up; feet = 4 lowest-mean-Y joints
lift = (Y[:,feet].max(0) - Y[:,feet].min(0)).mean()   # ~0.14 normal walk, ~0.03 near-shuffle

# step length: ankle z at footfall (contact onset) frames, diff consecutive footfalls (both feet, sorted)
aL, aR = skel.bone_index["left_ankle_roll_skel"], skel.bone_index["right_ankle_roll_skel"]
# steady-state = mean of last 2 steps (first step from standstill overshoots)

# cadence / step count: rising edges of (heel & toe) contact per foot, summed

# torso lean / forward incline: pitch of pelvis->upper-torso vector in the sagittal (Y-Z) plane
p = pj[:,0]; up = pj[:,[17,18,26]].mean(1)                    # pelvis -> waist_pitch + both shoulders
pitch = np.degrees(np.arctan2((up-p)[:,2], (up-p)[:,1]))      # +Z forward vs +Y up; ~4° upright, +θ leaned
```

---

## 8. Summary of the boundary (what text can and can't reach)

| Property | Text prompt | Constraint |
|---|---|---|
| Named action (walk/turn/start-stop) | ✅ faithful | — |
| Relative speed | ✅ but floored ~0.4 m/s, saturating | — |
| Exact speed (any value) | ❌ (numbers ignored) | ✅ Root2D, no floor (calibrate for G1 root) |
| Exact step length | ❌ | ✅ foot EE, ~1 cm |
| Torso lean / posture | ❌ (posture words ignored) | ✅ FullBody positions, ~2–3° |
| Alternative gait (shuffle/pauses) | ❌ substituted to a walk | ✅ indirectly (slow constraint → low foot-lift; foot pins → arbitrary) |
| Exact path / turn / waypoints | ❌ | ✅ Root2D (x,z) |
| Long / continuous (>10 s) | ❌ single gen degrades & stops | ✅ stitch constant-velocity segments (§10) |

**Rule of thumb:** the model faithfully renders text toward motions that are *densely represented in the
mocap*, and silently substitutes the nearest walk for anything off-distribution (sub-0.4 m/s cruise,
shuffles, pauses, numeric targets). For exact kinematics, drive the **constraints**, not the prompt.

---

## 9. Data-generation plan

The dataset below is the **authoritative generation spec**. Three phases, **32 h of motion each (96 h total)**.
Everything is **constraint-driven** (the prompt can't hit any of these numeric targets — §1). Every generator
recipe is in §2–§4; this section is the *what to generate* plus the *gotchas that must be honored or the
labels will be wrong*.

**Global rules**
- **≤ 8 s per trajectory** = 240 frames @ 30 fps — safely inside the model's validated ≤10 s range (demo
  `MAX_DURATION`) and far under its 5000-frame positional-encoding cap (~167 s). Budget the 8 s to fit
  ramp + cruise + ramp when ramps are included.
- **Fill the hour budget with seeds.** A phase has only tens of distinct parameter configs; 32 h ÷ 8 s ≈
  **14,400 clips/phase**, so replicate each config across **~200 random seeds** for natural gait diversity.
  Put `{phase, param, seed}` in the filename/metadata — reproducibility + de-dup.
- **Throughput** (measured, one RTX 3090 Ti): ~2.5 motion-s per wall-s at 100 denoising steps → **~38 wall-h
  for 96 h**. Batching saturates at B≈4 (compute-bound); the real lever is denoising steps — 20 steps ≈ 11
  motion-s/wall-s (~9 h) *after* eyeballing quality. Embarrassingly parallel → divide by GPU count.
- **Storage**: ~30 GB (full `.npz`) or ~15 GB (qpos CSV only) for all ~43k clips. Decide format up front.

### ⚠ Two hard limits that break the plan as written — do not ignore
1. **Step length has a kinematic ceiling that scales with the robot's leg length** — it is *per-skeleton*, not
   absolute. **On G1** (~0.7 m leg): faithful to ~1 cm up to **0.4 m**; **0.7 m** tracks ~95 % with a late fade;
   **> 0.7 m collapses** — commanding **1.0 m yields ~0.44 m**, an irregular scramble with extra inserted steps.
   The **1.0 m end of the grid is meant for a taller skeleton** (e.g. SOMA-77 / SMPL-X human-scale, longer legs),
   where it is reachable. **So for G1, configs above ~0.7 m will be mislabeled if saved at face value.** Per robot:
   (a) cap the grid near ~1× that skeleton's leg length, or (b) generate past it but store the **measured** step
   length as the label, flagged `degraded`. Also (all skeletons): first step from a standstill overshoots; soft
   floor (small +bias) below ~0.15 m on G1.
2. **Speed retarget is not 1:1 below ~0.25 m/s.** The G1 qpos root runs short of the commanded Kimodo path,
   speed-dependent, with a **knee at 0.25 → 0.20 m/s** (~1 % gap above, 9–24 % below). The grids start at
   0.25 m/s, which sits *on* the knee — **calibrate per target** (secant loop, §2) so 0.25 lands at 0.25, not ~0.18.

### Phase 0 — forward, speed & step-length grids (32 h)
| Sweep | Values | Lever | Key caveat |
|---|---|---|---|
| Forward **velocity** | 0.25 → 1.00 m/s, step 0.05 → **16** | Root2D constant-velocity path (§2) | calibrate retarget per speed; 0.25 is on the knee |
| Forward **step length** | 0.10 → 1.00 m, step 0.05 → **19** | foot EE constraints (§3) | **only ≤0.7 m is faithful** — see limit ①|

Each config saved in **two variants** → ~35 base × 2 = **70 configs**:
- **without ramp** — steady cruise the whole clip (constant Root2D velocity / uniform footfall grid).
- **with ramp** — very-slow **start** (ramp 0→cruise) + very-slow **stop** (cruise→0), ending in a **clean square
  stand**. Use a Root2D *velocity profile* for the ramp (the prompt's start/stop envelope is inexact, §1), and
  append the **settle + FullBody stand-pin** — hold the final velocity/heading ~1–2 s, then pin a square
  standing pose (`go_forward[0]` rotated to the final facing) over the last ~8 frames via FullBody — so the
  clip doesn't freeze mid-stride (raw stops end ~20 cm staggered and still moving; the pin squares feet to 0 cm).

### Phase 1 — curves & turns (32 h)
| Sweep | Values | Lever | Key caveat |
|---|---|---|---|
| **Left curve** radius | 2.0 → 10.0 m × {velocity or step length} | Root2D circular path (§2) | **pin heading to tangent** or it crabs |
| **Right curve** radius | 2.0 → 10.0 m × {velocity or step length} | Root2D circular path, mirrored | negate lateral / `dsign`; same heading pin |
| **Turn-around** angular vel | 0.5 → 2.0 rad/s | Root2D: heading rotates at ω, root pinned in place | no ceiling; settle + stand-pin to end square |

- **Heading MUST be pinned to the path tangent** (`θ = atan2(f_x, f_z)`) on every curve — unpinned, the body
  strafes sideways (crab ≈ 90°), worst on gentle/large-radius curves.
- **Radius calibration**: tight curves come out ~5 % *small* (2.0→1.91), gentle ones ~1.5 % *large* (10.0→10.15) —
  command a corrected radius if you need it exact. Circle-fit residual is ~2 cm regardless.
- **Turn rate**: 0.5–2.0 rad/s all land within a few % (0.54/1.03/1.48/2.03), root stays in place (~10 cm), **no
  saturation** even at 2.0 rad/s. Duration scales as 180°/ω (0.5 rad/s ≈ 6.3 s — still ≤8 s).
- Pick the radius sub-step (e.g. 1 m → 9 radii) and the ω sub-step (e.g. 0.25 → 7 rates); grids × seeds fill 32 h.

### Phase 2 — backward, speed & step-length grids (32 h)
| Sweep | Values | Lever | Key caveat |
|---|---|---|---|
| Backward **velocity** | 0.25 → 1.00 m/s, step 0.05 → **16** | Root2D root −Z, **heading fixed +Z** (§2) | face forward, walk backward (validated) |
| Backward **step length** | 0.10 → 1.00 m, step 0.05 → **19** | foot EE constraints on a −Z grid (§3) | same ~0.7 m ceiling — see limit ①|

Same **two ramp variants** (with / without very-slow start+stop + settle-stand) as Phase 0.
Backward locomotion works with a **fixed forward heading + reverse root path** (confirmed in the diversity
batch); expect the step-length ceiling to apply identically.

### Seed diversity at a fixed config — replication is worth it (measured)

The plan replicates each config across ~200 seeds. **That's justified:** even with the *speed and root path
fully pinned* (go forward, 0.5 m/s, 16 seeds), the gaits vary meaningfully — all in *execution style*, not
trajectory:

| descriptor | range across 16 seeds | CV |
|---|---|--:|
| step length | 28 – 44 cm | 13 % |
| cadence | 1.2 – 2.0 footfalls/s | 13 % |
| foot lift | 12 – 17 cm | 10 % |
| arm swing | 6.5 – 34 cm | **44 %** |
| torso pitch | 4 – 29° | **64 %** |
| lead foot | 6 L / 10 R | — |

Because `speed = step × cadence` is fixed, seeds sample the **step↔cadence tradeoff** (mincing 28 cm @ 2 Hz ↔
striding 44 cm @ 1.2 Hz) *along the constant-speed curve*, **plus** largely-independent spread in **arm swing**
and **torso posture** (the biggest movers) and both **gait phases** (which foot leads). So per-config seed
replication yields a genuine distribution of natural gait styles, not near-duplicates. (The root *path* is
identical by construction — this is the intra-condition style diversity a per-config sample should provide.)

---

## 10. Long trajectories — stitching past the length limit

**Single-shot length is capped.** One generation is trained/validated to **≤ 10 s** (demo `MAX_DURATION`);
past that, quality degrades — a 10 s "go forward" already *decelerates to a stop* at the end (training clips
end at rest). The hard ceiling is **5000 frames ≈ 167 s** (sinusoidal positional-encoding buffer,
`backbone.py` `PositionalEncoding.max_len`); beyond it the model errors. So for anything long, **don't
generate one big clip — tile and stitch.**

**Recipe** (validated: continuous **62 s** "go forward" at a flat **1.00 m/s**, no stops, 2–7 cm seams):
1. Generate N segments (~8 s each) as **constant-velocity Root2D cruises** — no start/stop ramp, so each
   segment holds speed to its final frame (no mid-clip deceleration).
2. **Pin the first ~8 frames of segment i to the last ~8 frames of segment i−1** via `FullBodyConstraintSet`
   (positions) — makes the body pose *and foot phase* continuous across the seam.
3. Translate each segment into one global timeline (root keeps advancing) and **drop the overlap** frames,
   or linearly **crossfade** them (`blend`) for near-zero seams (2–7 cm → ~2 cm).

Result: one continuous walk at full per-segment quality, arbitrarily long — the same tiling that scales the
dataset. A reference **`stitch()` helper** (`tools/stitch.py`) implements it:

```python
from tools.stitch import stitch
stitch("go forward", speed=1.0, duration=60, out_stem="go_forward_60s", blend=True)  # -> qpos [T,36] + CSV
# CLI: python tools/stitch.py "go forward" 1.0 60 --out go_forward_60s --render --blend
```

**Limitation:** the join is **translation-only → straight-line locomotion**. Curved long paths need per-seam
heading rotation (rotate the pinned tail + the continuing root path by the accumulated heading) — not yet
implemented. Speed is constant across the whole trajectory.
