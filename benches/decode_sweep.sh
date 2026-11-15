#!/bin/bash
# Decode-speed sweep: generates 120-frame 1080p / 2160p streams with a
# black-box libvpx-vp9 encode through ffmpeg (single-tile and
# multi-tile-column, 8- and 10-bit), checks the crate's output against
# the black-box decode of the same stream, and prints frames per second
# per thread budget (median of RUNS).
#
# Two kinds of stream are generated, because the frame header decides
# how far the decoder can overlap frames:
#   * frame_parallel_decoding_mode = 1 (no backward adaptation): a
#     frame's entropy decode does not need the previous frame's symbol
#     counts, so consecutive frames' entropy decodes overlap;
#   * `_adapt`, frame_parallel_decoding_mode = 0: every frame's
#     probabilities are adapted from the previous frame's counts, so the
#     entropy decodes form a serial chain (tile columns still split it).
#
#   benches/decode_sweep.sh [scratch-dir] [runs]
#
# Nothing is written into the repository; the streams are several MiB
# each and stay in the scratch directory.
set -euo pipefail
DIR=${1:-/tmp/oxideav-vp9-decode-sweep}
RUNS=${2:-3}
HERE=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$DIR"

gen() { # name size pix_fmt tile-columns(log2) bitrate frame-parallel(0|1)
  [ -f "$DIR/$1.ivf" ] && return
  ffmpeg -hide_banner -loglevel error -y -f lavfi \
    -i "mandelbrot=size=$2:rate=30:end_scale=0.05,noise=alls=6:allf=t" \
    -frames:v 120 -pix_fmt "$3" -c:v libvpx-vp9 -b:v "$5" \
    -tile-columns "$4" -row-mt 1 -threads 16 -cpu-used 5 -g 60 \
    -frame-parallel "$6" "$DIR/$1.ivf"
}
gen 1080p_8b_1tile 1920x1080 yuv420p 0 5M 1
gen 1080p_8b_4tile 1920x1080 yuv420p 2 5M 1
gen 1080p_10b_4tile 1920x1080 yuv420p10le 2 5M 1
gen 2160p_8b_1tile 3840x2160 yuv420p 0 16M 1
gen 2160p_8b_8tile 3840x2160 yuv420p 3 16M 1
gen 2160p_10b_8tile 3840x2160 yuv420p10le 3 16M 1
gen 1080p_8b_1tile_adapt 1920x1080 yuv420p 0 5M 0
gen 1080p_8b_4tile_adapt 1920x1080 yuv420p 2 5M 0
gen 2160p_8b_1tile_adapt 3840x2160 yuv420p 0 16M 0

(cd "$HERE" && cargo build --release --example decode_ivf)
BIN=${CARGO_TARGET_DIR:-$HERE/target}/release/examples/decode_ivf

for s in 1080p_8b_1tile 1080p_8b_4tile 1080p_10b_4tile 2160p_8b_1tile 2160p_8b_8tile 2160p_10b_8tile \
  1080p_8b_1tile_adapt 1080p_8b_4tile_adapt 2160p_8b_1tile_adapt; do
  want=$(ffmpeg -hide_banner -loglevel error -threads 8 -i "$DIR/$s.ivf" -f rawvideo - | cksum)
  line="$s"
  for t in 1 2 4 8 16; do
    got=$("$BIN" "$DIR/$s.ivf" "$t" /dev/stdout 2>/dev/null | cksum)
    [ "$got" = "$want" ] || { echo "$s: output at $t threads differs from the black-box decode" >&2; exit 1; }
    fps=$(for _ in $(seq "$RUNS"); do
      "$BIN" "$DIR/$s.ivf" "$t" 2>&1 | sed -E 's/.*\(([0-9.]+) fps\).*/\1/'
    done | sort -n | sed -n "$(((RUNS + 1) / 2))p")
    line="$line  ${t}t=$fps"
  done
  echo "$line"
done
