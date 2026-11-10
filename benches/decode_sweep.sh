#!/bin/bash
# Decode-speed sweep: generates 120-frame 1080p / 2160p streams with a
# black-box encoder (ffmpeg's VP9 encoder — single-tile and
# multi-tile-column, 8- and 10-bit), checks the crate's output against
# the black-box decode of the same stream, and prints frames per second
# per thread budget (median of RUNS).
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

gen() { # name size pix_fmt tile-columns(log2) bitrate
  [ -f "$DIR/$1.ivf" ] && return
  ffmpeg -hide_banner -loglevel error -y -f lavfi \
    -i "mandelbrot=size=$2:rate=30:end_scale=0.05,noise=alls=6:allf=t" \
    -frames:v 120 -pix_fmt "$3" -c:v libvpx-vp9 -b:v "$5" \
    -tile-columns "$4" -row-mt 1 -threads 16 -cpu-used 5 -g 60 "$DIR/$1.ivf"
}
gen 1080p_8b_1tile 1920x1080 yuv420p 0 5M
gen 1080p_8b_4tile 1920x1080 yuv420p 2 5M
gen 1080p_10b_4tile 1920x1080 yuv420p10le 2 5M
gen 2160p_8b_1tile 3840x2160 yuv420p 0 16M
gen 2160p_8b_8tile 3840x2160 yuv420p 3 16M
gen 2160p_10b_8tile 3840x2160 yuv420p10le 3 16M

(cd "$HERE" && cargo build --release --example decode_ivf)
BIN=${CARGO_TARGET_DIR:-$HERE/target}/release/examples/decode_ivf

for s in 1080p_8b_1tile 1080p_8b_4tile 1080p_10b_4tile 2160p_8b_1tile 2160p_8b_8tile 2160p_10b_8tile; do
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
