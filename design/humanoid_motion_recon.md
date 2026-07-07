# humanoid_motion_recon vs the field: alternatives survey & design considerations

Survey date: 2026-07-07 (multi-agent web sweep over papers/repos, adversarially gap-checked).
Reference system throughout ("MR"): the standalone `../humanoid_motion_recon` package —
SAM-3D-Body (bbox-prompted per-frame MHR-70 keypoints + mesh + per-joint global rotations) +
VGGT-Omega windowed depth (robust-affine stitched, moving cameras) + subject-calibrated world
lift (torso-up, heels=floor, standing pelvis height = scale) + rigid fit + temporal smoothing.
Measured profile: 6.3× real time on one RTX 3090 (9.3× with hand refinement), works on
faceless humanoid robots, chirality corrected downstream (Viterbi), foot skate 0.03 m/s
residual, no physics. Package docs: `../humanoid_motion_recon/README.md`; Figure-scenario
findings: [figure.md](figure.md) §7.

## 1. The five families

**A. World-grounded regression HMR** — the direct alternatives. WHAM (CVPR 24), GVHMR
(SIGGRAPH Asia 24), TRAM (ECCV 24), WHAC (ECCV 24), PromptHMR-Vid (CVPR 25), GEM/GENMO
(NVIDIA ICCV 25), Human3R (ICLR 26), OnlineHMR (CVPR 26). All regress SMPL/SMPL-X + world
trajectory. Three world-grounding philosophies, none of which is MR's subject-calibration:

1. *Learned root velocity + human-prior scale* (WHAM, GVHMR): no depth anywhere; trajectory
   is integrated network velocity, scale from the AMASS-trained prior. Immune to
   depth-inpainting-over-subject, but open-loop — ~2% relative trajectory drift, no scene
   anchoring.
2. *SLAM camera + background-depth scale* (TRAM, PromptHMR-Vid, VideoMimic): masked
   DROID-SLAM + ZoeDepth/MegaSaM. Metric scale from the **scene** (MR: from the **subject**).
   Wins on texture-rich scenes, fails on texture-poor; heavy.
3. *Unified feed-forward scene+human* (Human3R on CUT3R): one streaming network → humans +
   point cloud + camera, ~15 fps / 8 GB. Architecturally the most credible replacement for
   MR's three-stage stack; human readout is BEDLAM-trained (robot-blind), but retraining is
   ~1 GPU-day.

**B. Per-frame HMR foundations** (HMR2.0/4D-Humans, TokenHMR→CameraHMR, SMPLer-X/SMPLest-X,
Multi-HMR, NLF, Sapiens): camera-frame only — competitors to SAM-3D-Body, not to the
pipeline. Only PromptHMR copies SAM-3D-Body's detector-free bbox/mask-prompt interface.

**C. Optimization pipelines** (SLAHMR CVPR 23, PACE 3DV 24): joint camera+human+prior
optimization. Runtime disqualifying: SLAHMR ~15–24 s/frame (≈500× RT); PACE ~14× RT, code
never released.

**D. Video→humanoid real-to-sim** — **VideoMimic (Berkeley, CoRL 25 Best Student Paper) is
the closest end-to-end competitor** to MR + the G1 retarget: video → SMPL + dense scene mesh
(MegaSaM→NKSR) → contact-aware G1 retarget → RL tracking policy. Better than MR: scene
geometry as output, GeoCalib gravity (no calibration windows), BSTRO contact labels,
physics absorbs foot skate. Worse: ~1–2 orders slower (un-batchified front end), and the
perception stack (Grounded-SAM2 prompted `"person."` + ViTPose + VIMO) is human-locked —
cannot ingest robot-subject video at all.

**E. Downstream layers to consume, not compete with**: physics cleanup — PHC (ICCV 23,
30 Hz RL tracking, 98.9% imitation on AMASS, real-time), PhysPT (CVPR 24, feed-forward
torque/contact refinement), PhysHMR (SIGGRAPH Asia 25); retargeting — GMR (Stanford 25,
MIT, 18 humanoids incl. G1, 35–70 fps on CPU), OmniRetarget (Amazon FAR 25, hard-constraint
SQP: ~0 penetration, 82% downstream-RL success on interactions vs GMR ~55%, needs
terrain/object meshes).

## 2. Head-to-head on MR's axes

| System | World frame | Speed | Robot subject? | Hands | License |
|---|---|---|---|---|---|
| **MR (ours)** | subject-calibrated + VGGT depth | 6.3× RT (3090) | **yes** (bbox-prompted, chirality Viterbi) | optional (9.3×) | components: Meta/ModelScope checkpoints |
| WHAM | integrated velocity, SLAM gyro | ~9 fps online w/ preproc (A100) | no (YOLO+ViTPose face kps, SMPL/AMASS priors) | no | code MIT; SMPL NC |
| GVHMR | gravity-view, yaw-only drift | net 0.28 s/45 s clip; ~1× RT e2e (4090) | no (same stack) | no | ZJU NC |
| TRAM | masked DROID-SLAM + ZoeDepth scale | ≫RT (detect stage ~1.4 s/f) | no (YOLOv7-gated) | no | code MIT; SMPL NC |
| WHAC | SMPLest-X + DPVO + velocity prior | unpublished; offline | no (YOLOv8-gated) | yes (SMPL-X) | S-Lab NC |
| PromptHMR-Vid | DROID-SLAM + metric depth | unpublished; offline | **partial** — bbox-promptable like ours, human-trained | param only | Meshcapade NC |
| GEM/GEM-X | gravity-view + root velocity; SLAM camera | ONNX demo near-RT; diffusion offline | no (YOLOX+ViTPose front) | yes (77-joint SOMA) | **Apache-2.0** (GEM-X) |
| Human3R | native (CUT3R state), streaming | ~15 fps, 8 GB | no (BEDLAM-trained readout; no prompt) | param only | code MIT; CUT3R lineage CC-SA |
| VideoMimic | GeoCalib gravity + SMPL height; MegaSaM | ~6–8 min /100 frames | no (`"person."`-prompted front end) | no | MIT |
| SLAHMR | DROID-SLAM + HuMoR prior | ~500× RT | no | no | code MIT; deps NC |
| NLF | camera-frame (+bolt-on SLAM) | 109–410 fps crops (3090) | no (canonical human volume) | SMPL-X fit | code MIT; weights NC |

(Benchmark caveat from the gap-check: published EMDB/RICH numbers mix GT-gyro vs
estimated-camera settings when papers quote each other — e.g. WHAM 135.6 mm WA-MPJPE
(own paper, gyro) vs 166.1 mm (TRAM's re-eval, estimated cameras). Don't treat cross-paper
tables as commensurable.)

## 3. What the field does better than MR

- **Zero-calibration gravity** — GVHMR's gravity-view frame and VideoMimic's GeoCalib both
  remove per-scenario `MR_YAW0/1`, `MR_UP0/1`, `MR_STAND0/1` windows. Our calibration is the
  most manual step in the pipeline.
- **Contact modeling** — WHAM/GVHMR predict per-joint stationary probabilities and refine
  trajectories to them (GVHMR ~3 mm foot sliding; MR: 0.03 m/s residual, no contact notion).
- **Scene-metric scale** — TRAM/VideoMimic scale from background geometry; independent of
  the `MR_PELVIS_H` subject-height assumption, and yields a first-class camera trajectory.
- **Benchmark discipline** — everyone reports EMDB/RICH world metrics; MR has only internal
  evals. One EMDB run would make our claims commensurable.
- **Multi-person** (PHALP, PromptHMR, Human3R) and **streaming** (WHAM causal, Human3R,
  OnlineHMR) — MR is single-subject, offline by construction (windowed VGGT stitching).
- **Physics** — PHC/PhysPT/PhysHMR guarantee contact consistency; our smoother cannot.
- **Validated retargeting** — GMR/OmniRetarget are graded by downstream RL success;
  our direction-transfer + IK has only our own eval (8.4 cm MPJPE-local floor).

## 4. What MR does better — the moat and the real edges

- **Robot subjects (the moat, structural not incidental).** Every surveyed alternative
  front-ends a COCO person detector and/or ViTPose keypoints (5/17 facial), plus an SMPL(-X)
  shape space that cannot represent G1 proportions, plus an AMASS motion prior that fights
  robot gait. PromptHMR alone shares the bbox-prompt interface, with unproven robot
  generalization. **No surveyed system handles front/back chirality of faceless subjects**;
  MR's raw-hypothesis + Viterbi pass is unique.
- **Speed with a world frame.** 6.3× RT beats everything world-grounded except
  GVHMR+SimpleVO (~1× RT, human-only) and Human3R (~2× RT, human-only). The SLAM family is
  10–500× RT.
- **Model-agnostic output.** MHR per-joint global rotations + subject-calibrated scale
  instead of SMPL betas — exactly why the G1 retarget works on non-human proportions.
- **Scene depth by-product** (stitched VGGT depth) — velocity-integration methods produce no
  scene at all; MR gets the kitchen fit / camera solve from it for free.
- **Licensing.** The SMPL ecosystem is research-only almost throughout (MPI, ZJU, S-Lab,
  NAVER, Meshcapade NC terms). Commercially clean exceptions: GEM-X (Apache-2.0), GMR and
  VideoMimic code (MIT).

## 5. Honest gaps (from the adversarial gap-check)

1. **Known-robot pose tracking is the missing baseline for the moat.** With a known G1
   URDF/CAD, render-and-compare methods (RoboPose CVPR 21 lineage; holistic robot pose
   CVPR 24) estimate joint angles with no detector, no shape prior, and **zero chirality
   ambiguity** (mesh known). For robot videos of a *known platform* this family could beat
   MR outright — evaluate before claiming generality of the robot-subject advantage.
2. **No external benchmark.** Run EMDB once; internal 8.4 cm claims are self-referential.

## 6. Import queue (cheapest first)

1. **GeoCalib gravity** → drop `MR_YAW/UP` calibration env vars (VideoMimic-proven drop-in).
2. **Contact/stationary detection** on fit3d (GVHMR-style probabilities or simple velocity
   gating) → kill the residual skate at the fit stage instead of in the retarget.
3. **GMR as alternative retarget backend** (MIT, 18 robots, CPU-real-time) — A/B against
   `figure/soma_retarget.py` with `eval_retarget`.
4. **PHC-style RL tracking** as the eventual physics layer (also VideoMimic's answer);
   OmniRetarget only if terrain/object meshes enter the picture.
5. **Human3R fine-tune watch**: if a robot-subject variant becomes feasible (~1 GPU-day on
   synthetic robot renders), the one-network architecture obsoletes our three-stage stack.

## 7. Key links

WHAM <https://github.com/yohanshin/WHAM> · GVHMR <https://github.com/zju3dv/GVHMR> ·
TRAM <https://github.com/yufu-wang/tram> · PromptHMR <https://github.com/yufu-wang/PromptHMR> ·
WHAC <https://github.com/SMPLCap/WHAC> · GEM-X <https://github.com/NVlabs/GENMO> ·
Human3R <https://github.com/fanegg/Human3R> · VideoMimic <https://www.videomimic.net/> ·
PHC <https://github.com/ZhengyiLuo/PHC> · GMR (Stanford, arXiv:2510.02252) ·
OmniRetarget (arXiv:2509.26633, in amazon-far/holosoma) · NLF (Sárándi & Pons-Moll,
NeurIPS 24) · SLAHMR <https://github.com/vye16/slahmr> · Sapiens (Meta, ECCV 24)
