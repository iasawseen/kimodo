"""Reproduction of the Figure 'Introducing Helix 02' kitchen demo with G1 in MuJoCo,
using Kimodo as motion generator + stitcher. See design/figure.md for the full write-up.

Scripts:
    gen_gestures.py      - gesture prompt/constraint probe (contact sheets + numeric report)
    gen_helix_kitchen.py - generate the 15 segments, SE(2)-chain them, save qpos + bounds
    build_scene.py       - build the kitchen XML around the measured key poses (no GPU)
    render_scene.py      - fixed-camera renderer (mp4 or 8-frame contact sheet)
"""
