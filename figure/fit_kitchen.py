#!/usr/bin/env python
"""Fit the kitchen layout + solve the video camera from the VGGT-Omega reconstruction.

Input: <workdir>/lift3d.npz (scene cloud, lifted skeletons, camera transform - all in the
gravity-aligned kitchen frame: floor z=0, +X = the robot's dishwasher-work facing, G1 scale).

Fit:
  - refine the frame yaw so the two counter runs are axis-aligned (orientation histogram of
    counter-top points),
  - sink run   = counter-top band furthest along +X (front face x = const),
  - back run   = counter-top band along y = const,
  - corner, pocket (median work-pelvis), walk start/trajectory,
  - camera pose -> MuJoCo azimuth/elevation/distance/lookat/fovy (roll reported).

Output: outputs/figure/pose/kitchen_fit.json + fit_overlay.png (bird's-eye + fitted lines +
camera frustum + trajectory).
Usage: DEPTH_WORK=<pose_full dir> python figure/fit_kitchen.py
"""
import json
import os

import numpy as np

WORK = os.environ["DEPTH_WORK"]
_PKG = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_PKG)
OUTD = os.path.join(REPO, "outputs", "figure", "pose")

z = np.load(os.path.join(WORK, "lift3d.npz"))
P = z["scene"].astype(np.float64)
J = z["joints_w"].astype(np.float64)
t = z["t"]
ok = z["ok"]
cam_pos = z["cam_pos"]
Rcw = z["R_cam2world"]
fov_h = float(z["fov_h"])

KPN = ["nose", "lsho", "rsho", "lelb", "relb", "lhip", "rhip", "lkne", "rkne",
       "lank", "rank", "lbtoe", "lheel", "rbtoe", "rheel", "rwri", "lwri", "neck"]
LH, RH = KPN.index("lhip"), KPN.index("rhip")

# ---------------- counter-top points
ct = P[(P[:, 2] > 0.55) & (P[:, 2] < 0.75)]

# refine yaw: dominant orientation of local 2D structure (PCA on neighborhoods is overkill;
# use the orientation histogram of pairwise deltas among nearby points)
sub = ct[np.random.default_rng(0).choice(len(ct), min(4000, len(ct)), replace=False)]
d = sub[None, :, :2] - sub[:, None, :2]
dist = np.linalg.norm(d, axis=2)
m = (dist > 0.05) & (dist < 0.5)
ang = np.arctan2(d[..., 1][m], d[..., 0][m]) % (np.pi / 2)
hist, edges = np.histogram(ang, bins=90)
th = float(0.5 * (edges[np.argmax(hist)] + edges[np.argmax(hist) + 1]))
if th > np.pi / 4:
    th -= np.pi / 2
c, s = np.cos(-th), np.sin(-th)
Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
P = P @ Rz.T
ct2 = ct @ Rz.T
J = J @ Rz.T
cam_pos = Rz @ cam_pos
Rcw = Rz @ Rcw
print(f"[fit] yaw refine {np.degrees(th):+.1f} deg")

# ---------------- runs: histogram peaks of counter-top footprint
# sink run: band in x (front face toward -x); back run: band in y
up_pts = P[(P[:, 2] > 1.2) & (P[:, 2] < 2.2)]                        # uppers/walls only
hx, ex = np.histogram(up_pts[:, 0], bins=100)
hy, ey = np.histogram(up_pts[:, 1], bins=100)
x_wall = float(0.5 * (ex[np.argmax(hx)] + ex[np.argmax(hx) + 1]))    # sink wall x
y_wall = float(0.5 * (ey[np.argmax(hy)] + ey[np.argmax(hy) + 1]))    # back wall y
# uppers protrude ~0.3 from the wall plane, and the counter front face sits ~0.1 in front of
# the uppers' face; run center = front face - CD/2
x_run = x_wall - 0.10 + 0.225
y_run = y_wall + 0.10 - 0.225
# extents
sinkband = ct2[np.abs(ct2[:, 0] - x_run) < 0.30]
backband = ct2[np.abs(ct2[:, 1] - y_run) < 0.30]
sink_y0, sink_y1 = np.percentile(sinkband[:, 1], [2, 98])
back_x0, back_x1 = np.percentile(backband[:, 0], [2, 98])

# ---------------- robot anchors
mid = 0.5 * (J[:, LH] + J[:, RH])
work = ok & (t >= 8.0) & (t <= 200.0)
pocket = np.median(mid[work], 0)
walk = ok & (t >= 2.4) & (t <= 6.8)
traj = mid[walk]
start = np.median(traj[:6], 0)

# ---------------- camera -> MuJoCo free-camera parameters
fwd = Rcw @ np.array([0.0, 0.0, 1.0])
upc = Rcw @ np.array([0.0, -1.0, 0.0])
right = Rcw @ np.array([1.0, 0.0, 0.0])
az = float(np.degrees(np.arctan2(fwd[1], fwd[0])))
el = float(np.degrees(np.arcsin(np.clip(fwd[2], -1, 1))))
roll = float(np.degrees(np.arctan2(right[2], np.linalg.norm(right[:2]))))
DIST = 3.0
look = cam_pos + fwd * DIST
fovy = float(np.degrees(fov_h))
fit = dict(x_run=x_run, y_run=y_run, sink_y=[float(sink_y0), float(sink_y1)],
           back_x=[float(back_x0), float(back_x1)],
           pocket=[float(pocket[0]), float(pocket[1])],
           start=[float(start[0]), float(start[1])],
           cam_pos=[float(v) for v in cam_pos], lookat=[float(v) for v in look],
           azimuth=az, elevation=el, roll=roll, distance=DIST, fovy=fovy,
           cam_height=float(cam_pos[2]))
json.dump(fit, open(os.path.join(OUTD, "kitchen_fit.json"), "w"), indent=1)
print(f"[fit] sink run x={x_run:.2f} y[{sink_y0:.2f},{sink_y1:.2f}]  "
      f"back run y={y_run:.2f} x[{back_x0:.2f},{back_x1:.2f}]")
print(f"[fit] pocket {pocket[:2].round(2)}  walk start {start[:2].round(2)}")
print(f"[fit] camera pos {cam_pos.round(2)} az {az:.1f} el {el:.1f} roll {roll:.1f} fovy {fovy:.1f}")

# ---------------- overlay plot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(11, 11))
low = P[(P[:, 2] > 0.05) & (P[:, 2] < 0.5)]
ax.scatter(low[:, 0], low[:, 1], s=0.3, c="0.85", zorder=0, label="bases/floor clutter")
ax.scatter(ct2[:, 0], ct2[:, 1], s=0.5, c="tab:blue", zorder=1, alpha=0.4, label="counter tops")
hi = P[P[:, 2] > 1.0]
ax.scatter(hi[:, 0], hi[:, 1], s=0.3, c="tab:red", alpha=0.25, zorder=1, label="uppers/walls")
ax.axvline(x_run, color="navy", lw=2, label="sink run")
ax.plot([back_x0, back_x1], [y_run, y_run], color="darkgreen", lw=2, label="back run")
g = mid[ok & (t > 2.0) & (t < 210.0)]
ax.plot(g[:, 0], g[:, 1], "-", color="orange", lw=1.5, label="pelvis trajectory")
ax.plot(*pocket[:2], "r*", ms=18, label="work pocket")
ax.plot(*start[:2], "g^", ms=12, label="walk start")
ax.plot(*cam_pos[:2], "ks", ms=10, label="camera")
for sgn in (-1, 1):
    e = cam_pos[:2] + 6.0 * np.array([np.cos(np.radians(az + sgn * fovy * 0.85)),
                                      np.sin(np.radians(az + sgn * fovy * 0.85))])
    ax.plot([cam_pos[0], e[0]], [cam_pos[1], e[1]], "k--", lw=1)
ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(loc="lower left", fontsize=9)
ax.set_title("kitchen fit from VGGT-Omega reconstruction (bird's-eye)")
fig.savefig(os.path.join(OUTD, "fit_overlay.png"), dpi=110, bbox_inches="tight")
print("[fit] fit_overlay.png saved")
