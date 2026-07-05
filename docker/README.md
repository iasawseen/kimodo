# Work containers

Two images for the figure / MotionRecon workflow (the repo-root `Dockerfile` + `docker-compose.yaml`
are upstream's demo setup and currently don't build in this checkout — `kimodo-viser` is absent):

| service | image | GPU | purpose |
|---|---|---|---|
| `gen` | `kimodo-gen` | 0 | Kimodo generation, MuJoCo EGL rendering, scene/verify/collate tooling |
| `percept` | `kimodo-percept` | 1 | SAM-3D-Body + VGGT-Omega inference (sibling repos volume-mounted) |

## Build

```bash
docker compose -f docker/compose.yaml build
```

## Run

```bash
# generation (env redirects for the gated text encoder are baked into the service)
docker compose -f docker/compose.yaml run --rm gen \
  python -m figure.gen_helix_kitchen walk_in

# perception: frame caches live under /scratch (host: $SCRATCH_DIR, default /tmp/kimodo-scratch)
docker compose -f docker/compose.yaml run --rm percept \
  bash -c 'DEPTH_WORK=/scratch/pose_x python figure/depth_video.py \
           /root/.cache/modelscope/facebook/VGGT-Omega/vggt_omega_1b_512.pt'

# interactive shell
docker compose -f docker/compose.yaml run --rm gen bash
```

## Notes

- Host caches are mounted: `~/.cache/kimodo` (text encoders), `~/.cache/modelscope` (gated
  checkpoints via ModelScope mirror), `~/.cache/huggingface`. Nothing re-downloads.
- `NVIDIA_DRIVER_CAPABILITIES` includes `graphics` (EGL headless MuJoCo) and `video` (NVENC for
  `figure/fastvid.py`); dropping either silently breaks rendering/encoding.
- The repo is bind-mounted read-write at `/work/kimodo`; `figure/*` scripts find the model repos
  at `/work/sam-3d-body` and `/work/vggt-omega` exactly like the `../..` layout on the host.
- Set `SCRATCH_DIR` to a big disk before extracting frames for long videos.
- Containers run as root by default; generated files under `outputs/` will be root-owned unless
  you add `--user $(id -u):$(id -g)` to `run` (both images also ship `gosu`).
