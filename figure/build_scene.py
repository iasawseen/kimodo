#!/usr/bin/env python
"""Build the Helix-02 kitchen scene XML around the measured key poses of the chained motion.
v2 - layout mirrored to match the video composition (sink/dishwasher wall receding image-LEFT,
range wall frontal, dishwasher at the L-corner, dining table foreground-right), dishwasher door as
a HINGED body + bottom rack on a SLIDE joint (animated by render_scene.py in sync with the segment
bounds), camera matched to the video's fixed viewpoint.

Standalone (no GPU): reads outputs/figure/helix_kitchen.csv + _bounds.json, writes
outputs/figure/kitchen_g1.xml + outputs/figure/helix_kitchen_scene.json.
See design/figure.md."""
import json
import os

import numpy as np

_PKG = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_PKG)
OUT = os.path.join(REPO, "outputs", "figure")
G1_XML = os.path.join(REPO, "kimodo/assets/skeletons/g1skel34/xml/g1.xml")
KITCHEN_XML = os.path.join(OUT, "kitchen_g1.xml")
STEM = "helix_kitchen"

BLACK = "0.10 0.10 0.11 1"; WHITE = "0.93 0.92 0.90 1"; STEEL = "0.62 0.63 0.65 1"
WOODT = "0.16 0.13 0.11 1"; RACK = "0.80 0.82 0.85 1"

# G1-scaled kitchen (~0.75x human)
CH, CD = 0.65, 0.45                            # counter height / depth
DW_W, DOOR_L = 0.56, 0.41                      # dishwasher width, door length (fold-down)


def yaw_of(q):
    w, x, y, z = q[3], q[4], q[5], q[6]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def build():
    qpos = np.loadtxt(os.path.join(OUT, STEM + ".csv"), delimiter=",")
    bounds = json.load(open(os.path.join(OUT, STEM + "_bounds.json")))["bounds"]
    bidx = {n: i for n, i in bounds}

    # the planted working spot + facing (all gestures happen here, facing the dishwasher wall)
    qw = qpos[bidx["open_door"]]
    pW = qw[:2].copy()
    # kitchen frame = the CHAIN's design frame (+X): segments are authored/seamed at yaw 0.
    # (Snapping to the measured arrival yaw is wrong now that video anchors give the arrival an
    # intentional ~+60deg angle inside the fixed kitchen, like the real robot.)
    hW = np.array([1.0, 0.0])
    aW = np.array([-hW[1], hW[0]])                       # along the dishwasher wall (robot's left)
    pS = qpos[0, :2].copy()                              # walk-in start

    # gestures aim at the robot's front-LEFT (LAT=+0.55): robot in the corner pocket, dw at its left
    dw_c = pW + 0.58 * hW + 0.55 * aW                    # dishwasher FRONT-line point (door hinge)
    dw_mid = dw_c + hW * CD / 2                          # dishwasher on the counter CENTER line
    front = pW + 0.58 * hW                               # counter front line through the work spot

    geoms, bodies = [], []

    def box(name, c, sx, sy, sz, zc, rgba):
        geoms.append(f'    <geom name="{name}" type="box" size="{sx/2:.3f} {sy/2:.3f} {sz/2:.3f}" '
                     f'pos="{c[0]:.3f} {c[1]:.3f} {zc:.3f}" rgba="{rgba}" contype="0" conaffinity="0"/>')

    def run_box(name, center, along, length, depth, height, zc, rgba, normal=None):
        normal = hW if normal is None else normal
        sz = np.abs(along) * length + np.abs(normal) * depth
        box(name, center, max(sz[0], .02), max(sz[1], .02), height, zc, rgba)

    # ---------------- dishwasher wall (video-left, receding): counter, sink, window, uppers
    # counter run along aW, from the corner (behind the dw, -aW side) outward (+aW side);
    # everything in this run is centered on the counter CENTER line (front + CD/2)
    for s, e, nm in [(-1.22, -(DW_W / 2 + 0.02), "ctrA_corner"), ((DW_W / 2 + 0.02), 2.2, "ctrA_sink")]:
        c = dw_mid + aW * (s + e) / 2
        run_box(nm, c, aW, abs(e - s), CD, CH - 0.04, (CH - 0.04) / 2, BLACK)
    run_box("ctrA_top", dw_mid + aW * 0.30, aW, 4.15, CD + 0.06, 0.04, CH - 0.02, WHITE)
    # dishwasher cavity (door + rack are jointed bodies below), embedded flush in the run
    run_box("dw_cavity", dw_mid, aW, DW_W, CD - 0.02, CH - 0.10, (CH - 0.10) / 2, "0.22 0.23 0.25 1")
    run_box("dw_rack_top", dw_mid + hW * 0.03, aW, DW_W - 0.10, CD - 0.16, 0.05, 0.40, RACK)
    # sink + faucet on the +aW side
    sink = dw_mid + aW * 0.58
    box("sink", sink, abs(aW[0]) * 0.50 + abs(hW[0]) * 0.32 + .01,
        abs(aW[1]) * 0.50 + abs(hW[1]) * 0.32 + .01, 0.02, CH + 0.001, "0.15 0.15 0.16 1")
    box("faucet_v", sink + hW * 0.14, 0.03, 0.03, 0.35, CH + 0.175, BLACK)
    box("faucet_h", sink + hW * 0.05, abs(hW[0]) * 0.20 + .03, abs(hW[1]) * 0.20 + .03, 0.03, CH + 0.34, BLACK)
    # wall, window above the sink, black uppers beyond, tall cabinet at the far end
    wallA_c = dw_mid + hW * (CD / 2 + 0.03) + aW * 0.9
    run_box("wallA", wallA_c, aW, 5.4, 0.06, 2.3, 1.15, "0.90 0.89 0.87 1")
    run_box("windowA", sink + aW * 0.50 + hW * (CD / 2 - 0.01), aW, 0.85, 0.05, 0.85, 1.42, "0.97 0.97 0.96 1")
    run_box("upperA", dw_mid + aW * 2.0 + hW * (CD / 2 - 0.05), aW, 1.1, 0.32, 0.95, 1.52, BLACK)
    # reach-target cabinet directly above the work area (bottom 1.48: the reach-up ends just under it)
    run_box("upperA2", pW + (0.58 + CD / 2) * hW - 0.12 * aW, aW, 0.70, 0.32, 0.58, 1.77, BLACK)
    run_box("tallA", dw_mid + aW * 2.85 + hW * (CD / 2 - 0.12), aW, 0.6, 0.40, 2.2, 1.1, BLACK)

    # ---------------- range wall (frontal): corner section, range + hood, dino counter, column
    hB = aW                                              # back wall normal (points into the room:
    aB = hW                                              # the corner is on the robot's -aW side)
    cornerB = dw_c - aW * (DW_W / 2 + 0.92)              # wall junction reference
    ctrB_line = cornerB - aW * (CD / 2)                  # back counter center line
    # base run split around the range (the range sits proud, like the video's pro range)
    for s, e, nm in [(-0.45, 0.95, "ctrB_base"), (1.78, 3.0, "ctrB_base2")]:
        run_box(nm, ctrB_line - aB * (s + e) / 2, aB, abs(e - s), CD, CH - 0.04, (CH - 0.04) / 2,
                BLACK, normal=hB)
    for s, e, nm in [(0.02, 0.97, "ctrB_top"), (1.76, 3.05, "ctrB_top2")]:
        run_box(nm, ctrB_line - aB * (s + e) / 2, aB, abs(e - s), CD + 0.06, 0.04, CH - 0.02,
                WHITE, normal=hB)
    wallB_c = ctrB_line - hB * (CD / 2 + 0.03) - aB * 0.6
    run_box("wallB", wallB_c, aB, 5.0, 0.06, 2.3, 1.15, "0.90 0.89 0.87 1", normal=hB)
    # range + hood (center-right of frame), proud of the cabinet line
    rng = ctrB_line - aB * 1.37 + hB * 0.04
    run_box("range", rng, aB, 0.78, CD + 0.06, CH + 0.02, (CH + 0.02) / 2, STEEL, normal=hB)
    run_box("range_top", rng, aB, 0.74, CD - 0.02, 0.015, CH + 0.03, "0.08 0.08 0.09 1", normal=hB)
    run_box("range_oven", rng + hB * 0.04, aB, 0.60, CD + 0.02, 0.30, 0.38, "0.30 0.31 0.33 1",
            normal=hB)
    # chimney hood: wide canopy + narrow duct (video's pyramid silhouette)
    run_box("hood", rng - hB * 0.10, aB, 0.78, 0.34, 0.22, 1.56, STEEL, normal=hB)
    run_box("hood_duct", rng - hB * 0.12, aB, 0.30, 0.28, 0.62, 1.98, STEEL, normal=hB)
    # black uppers flanking the hood + corner uppers
    run_box("upperB1", ctrB_line - aB * 2.40 - hB * 0.06, aB, 0.9, 0.32, 0.95, 1.52, BLACK, normal=hB)
    run_box("upperB2", ctrB_line - aB * 0.60 - hB * 0.06, aB, 0.85, 0.32, 0.95, 1.52, BLACK, normal=hB)
    # tall column at the frame-right end + the toy dinosaur on the counter
    run_box("tallB", ctrB_line - aB * 3.05 - hB * 0.10, aB, 0.6, 0.42, 2.2, 1.1, BLACK, normal=hB)
    dino = ctrB_line - aB * 2.25
    box("dino_body", dino, 0.16, 0.05, 0.06, CH + 0.03, "0.06 0.06 0.07 1")
    box("dino_neck", dino + aB * 0.06, 0.04, 0.04, 0.10, CH + 0.10, "0.06 0.06 0.07 1")

    # camera geometry (needed to park the table in the video's foreground-right)
    # camera from the VGGT-Omega reconstruction solve (figure/fit_kitchen.py) refined against
    # 50/50 video blends: fovy 30 and azimuth are the solve's; elevation/height re-estimated
    # from the video framing because the reconstruction's global tilt corrupts them
    CAM_AZ = np.radians(-74.5)
    CAM_EL = np.radians(-5.0)
    CAM_D = 4.6
    CAM_H = 1.10
    fwd3 = np.array([np.cos(CAM_EL) * np.cos(CAM_AZ), np.cos(CAM_EL) * np.sin(CAM_AZ),
                     np.sin(CAM_EL)])
    look = pW + 0.32 * hW + 0.12 * aW
    cam_pos = look[:2] - fwd3[:2] / np.linalg.norm(fwd3[:2]) * CAM_D
    dxy = look - cam_pos
    dhat = dxy / np.linalg.norm(dxy)
    rhat = np.array([dhat[1], -dhat[0]])                 # screen-right in world

    # ---------------- dining table + chairs: SCENE-relative (between camera and corridor,
    # frame-right like the video's table edge) - camera-relative placement once landed the table
    # on the walk corridor when the camera moved
    tb = pW - hW * 1.5 + aW * 1.1
    box("table_top", tb, 0.95, 0.62, 0.035, 0.55, WOODT)
    for ix in (-0.40, 0.40):
        for iy in (-0.24, 0.24):
            box(f"tleg_{ix:+.2f}_{iy:+.2f}".replace(".", "p"), tb + np.array([ix, iy]), 0.05, 0.05,
                0.53, 0.27, WOODT)
    for k, off in enumerate((aW * -0.55, hW * 0.55)):
        cc = tb + off
        box(f"chair{k}_seat", cc, 0.36, 0.36, 0.30, 0.15, BLACK)
        box(f"chair{k}_back", cc + off / np.linalg.norm(off) * 0.16, 0.05 + abs(off[1]) * 0.2,
            0.05 + abs(off[0]) * 0.2, 0.50, 0.55, BLACK)

    # ---------------- dishes (props)
    for i, (p, zz) in enumerate([(dw_mid + aW * 1.35, CH + 0.015), (ctrB_line - aB * 0.15, CH + 0.015)]):
        geoms.append(f'    <geom name="dish{i}" type="cylinder" size="0.075 0.012" '
                     f'pos="{p[0]:.3f} {p[1]:.3f} {zz:.3f}" rgba="0.95 0.95 0.97 1" contype="0" conaffinity="0"/>')
    geoms.append('    <light pos="1.2 0.5 2.8" dir="0 0 -1" diffuse="0.5 0.5 0.48"/>')

    # ---------------- jointed dishwasher door + sliding bottom rack (animated by the renderer)
    hinge = dw_c + hW * 0.01                  # bottom-front edge of the dw opening
    ax = aW / (np.linalg.norm(aW) + 1e-9)
    # hinge axis = aW; NEGATIVE angle swings the top toward -hW... sign depends on aW orientation:
    # rotation about axis a by th moves local +Z by sin(th) * (a x z). We want the top to fall
    # toward the room (-hW). a x z for a=aW: (aW_y*1, -aW_x*1, 0)... choose sign at runtime meta.
    crossx, crossy = ax[1], -ax[0]                       # (a x z)_xy
    open_sign = -1.0 if (crossx * hW[0] + crossy * hW[1]) > 0 else 1.0
    bodies.append(
        f'    <body name="dw_door_body" pos="{hinge[0]:.3f} {hinge[1]:.3f} 0.055">\n'
        f'      <joint name="dw_door_hinge" type="hinge" axis="{ax[0]:.3f} {ax[1]:.3f} 0" '
        f'range="-1.6 1.6" damping="1"/>\n'
        f'      <geom name="dw_door" type="box" size="{abs(ax[0])*DW_W/2+0.015:.3f} '
        f'{abs(ax[1])*DW_W/2+0.015:.3f} 0.265" pos="0 0 0.265" rgba="{STEEL}" '
        f'contype="0" conaffinity="0"/>\n'
        f'    </body>')
    rack0 = dw_c + hW * 0.24                             # parked inside the cavity
    rsx = abs(ax[0]) * (DW_W - 0.12) / 2 + abs(hW[0]) * 0.14
    rsy = abs(ax[1]) * (DW_W - 0.12) / 2 + abs(hW[1]) * 0.14
    bodies.append(
        f'    <body name="dw_rack_body" pos="{rack0[0]:.3f} {rack0[1]:.3f} 0.16">\n'
        f'      <joint name="dw_rack_slide" type="slide" axis="{-hW[0]:.0f} {-hW[1]:.0f} 0" '
        f'range="0 0.6" damping="1"/>\n'
        f'      <geom name="dw_rack" type="box" size="{rsx:.3f} {rsy:.3f} 0.05" '
        f'rgba="{RACK}" contype="0" conaffinity="0"/>\n'
        f'      <geom name="dw_rack_dish" type="cylinder" size="0.07 0.010" pos="0 0 0.065" '
        f'rgba="0.95 0.95 0.97 1" contype="0" conaffinity="0"/>\n'
        f'    </body>')

    xml = open(G1_XML).read()
    meshdir = os.path.abspath(os.path.join(os.path.dirname(G1_XML), "../meshes/g1"))
    xml = xml.replace('meshdir="../meshes/g1"', f'meshdir="{meshdir}"')
    xml = xml.replace('rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"', 'rgb1="0.66 0.58 0.48" rgb2="0.61 0.53 0.44"')
    xml = xml.replace('markrgb="0.8 0.8 0.8"', 'markrgb="0.64 0.56 0.47"')
    xml = xml.replace('reflectance="0.2"', 'reflectance="0.04"')
    xml = xml.replace('texrepeat="5 5"', 'texrepeat="3 16"')   # plank-ish floor stripes
    xml = xml.replace('rgb1="0.3 0.5 0.7" rgb2="0 0 0"', 'rgb1="0.92 0.91 0.89" rgb2="0.55 0.55 0.55"')
    xml = xml.replace("</mujoco>",
                      "    <worldbody>\n" + "\n".join(geoms + bodies) + "\n    </worldbody>\n</mujoco>")
    open(KITCHEN_XML, "w").write(xml)

    # ---------------- camera matched to the video: ~1.5m high, ~-10deg pitch, in the dining area
    dist = float(np.linalg.norm(dxy))
    look3 = np.array([cam_pos[0], cam_pos[1], CAM_H]) + fwd3 * CAM_D
    meta = dict(lookat=[float(v) for v in look3],
                azimuth=float(np.degrees(CAM_AZ)),
                elevation=float(np.degrees(CAM_EL)), distance=CAM_D, fovy=30.0,
                door_open_sign=float(open_sign),
                door_open_angle=float(1.31 * open_sign), rack_out=0.38,
                solids=["ctrA_corner", "ctrA_sink", "ctrA_top", "wallA", "upperA", "upperA2",
                        "tallA", "ctrB_base", "ctrB_base2", "ctrB_top", "ctrB_top2", "wallB",
                        "range", "range_oven", "hood", "upperB1", "upperB2", "tallB",
                        "table_top", "chair0_seat", "chair1_seat"])
    json.dump(meta, open(os.path.join(OUT, STEM + "_scene.json"), "w"), indent=1)
    print(f"[scene] XML -> {KITCHEN_XML}")
    print(f"[scene] work={pW.round(2)} facing={hW.round(1)} dw={dw_c.round(2)} start={pS.round(2)} "
          f"cam={cam_pos.round(2)} az={meta['azimuth']:.0f}")


if __name__ == "__main__":
    build()
