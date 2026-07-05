#!/bin/bash
# Rebuild the verify clips: TOP = figure.mp4 with the SAM-3D anchor-skeleton overlay
# (figure_sam3d.mp4, same timeline), time-stretched to the segment duration;
# BOTTOM = MuJoCo render (with constraint-skeleton overlay). Both panes carry skeletons.
cd "$(dirname "$0")/.."
while read name _ s e _ r0 r1; do
  t0=$(python3 -c "print($s/30)"); dur=$(python3 -c "print(($e-$s)/30)")
  rdur=$(python3 -c "print($r1-$r0)"); ratio=$(python3 -c "print(($e-$s)/30/($r1-$r0))")
  ffmpeg -nostdin -loglevel error -y \
    -ss $r0 -t $rdur -i outputs/figure/figure_sam3d.mp4 \
    -ss $t0 -t $dur -i outputs/figure/helix_kitchen.mp4 \
    -filter_complex "[0:v]setpts=${ratio}*PTS,fps=30,scale=960:540[a];[1:v]fps=30,scale=960:720[b];[a][b]vstack=shortest=1" \
    outputs/figure/verify/$name.mp4
done < outputs/figure/verify/chunks.txt
echo "verify clips rebuilt (both panes with skeletons)"
