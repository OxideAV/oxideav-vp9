#!/bin/sh
# Regenerates the output_params fixtures: tiny black-box libvpx-vp9
# encodes (2 frames each) across the layouts VP9 can carry, plus
# `ffprobe.txt`, the black-box ffprobe -show_streams answer per file
# (name width height pix_fmt color_range color_space) that
# tests/output_params.rs checks the decoder's output_params against.
#
# Usage: sh generate.sh   (from this directory; needs ffmpeg + ffprobe)
set -eu
cd "$(dirname "$0")"

enc() { # name size pix_fmt [extra ffmpeg args...]
    name=$1 size=$2 fmt=$3
    shift 3
    ffmpeg -nostdin -loglevel error -y -f lavfi -i "testsrc2=s=$size:r=25,format=yuv444p" \
        -frames:v 2 -c:v libvpx-vp9 -b:v 100k -g 999 -lag-in-frames 0 \
        -deadline good -cpu-used 4 -pix_fmt "$fmt" "$@" "$name.ivf"
}

enc p0-420-8          48x32 yuv420p
enc p1-422-8          48x32 yuv422p
enc p1-440-8          48x32 yuv440p
enc p1-444-8          48x32 yuv444p
enc p2-420-10         48x32 yuv420p10le
enc p2-420-12         48x32 yuv420p12le
enc p3-422-10         48x32 yuv422p10le
enc p3-444-12         48x32 yuv444p12le
enc p1-rgb-8          48x32 gbrp
enc p3-rgb-10         48x32 gbrp10le
enc odd-35x19-420     35x19 yuv420p
enc bt709-full        48x32 yuv420p -colorspace bt709 -color_range pc
enc bt2020-10         48x32 yuv420p10le -colorspace bt2020nc -color_range tv
# Mid-stream change: concatenated (packet by packet in the test) with
# p0-420-8 — a keyframe at a new size and depth.
enc seg-32x24-444-10  32x24 yuv444p10le

: > ffprobe.txt
for f in *.ivf; do
    printf '%s ' "${f%.ivf}" >> ffprobe.txt
    ffprobe -v error -select_streams v:0 \
        -show_entries stream=width,height,pix_fmt,color_range,color_space \
        -of default=nw=1:nk=1 "$f" | tr '\n' ' ' | sed 's/ $//' >> ffprobe.txt
    echo >> ffprobe.txt
done
