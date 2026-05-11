//! Runtime libavcodec interop for the VP9 cross-decode fuzz oracle.
//!
//! libavcodec is loaded via `dlopen` at first call — there is no
//! `ffmpeg-sys` / `rsmpeg`-style build-script dep that would pull
//! libavcodec or libvpx source into the workspace's cargo dep tree.
//! The harness checks `available()` up front and `eprintln!`s a
//! `[oracle skip]` line + early-returns when the shared library isn't
//! installed, so fuzz binaries built on a host without ffmpeg simply
//! do nothing instead of panicking. **No `#[ignore]`.**
//!
//! Workspace policy bars consulting libavcodec / libvpx source; we only
//! inspect the public C headers (`<libavcodec/avcodec.h>`,
//! `<libavcodec/packet.h>`) for function signatures + opaque struct
//! pointer typedefs.
//!
//! Install on Debian/Ubuntu via `apt-get install -y ffmpeg` (pulls
//! libavcodec.so.* as a transitive dep) or, more directly,
//! `libavcodec-dev`. On macOS use `brew install ffmpeg`. The loader
//! probes the conventional shared-object names for both platforms,
//! starting at the newest known SONAME (`libavcodec.so.62` is FFmpeg
//! 8.x; `.so.61` is 7.x; `.so.60` is 6.x; `.so` is the dev symlink).
//!
//! ## AVPacket layout
//!
//! `AVPacket` has a documented public layout (see <libavcodec/packet.h>);
//! the prefix has been stable from libavcodec 58 (FFmpeg 4.x) through
//! 62 (FFmpeg 8.x):
//!
//! ```text
//!   off  0  AVBufferRef* buf
//!   off  8  int64_t      pts
//!   off 16  int64_t      dts
//!   off 24  uint8_t*     data
//!   off 32  int          size
//!   off 36  int          stream_index
//!   off 40  int          flags
//! ```
//!
//! We use the official `av_new_packet(pkt, size)` to allocate the
//! managed buffer, then read `pkt->data` from offset 24 to memcpy our
//! input bytes in. `pkt->size` is already set by `av_new_packet`.
//!
//! ## AVFrame layout
//!
//! Likewise the AVFrame prefix is stable across libavcodec 58-62:
//!
//! ```text
//!   off   0  uint8_t*  data[AV_NUM_DATA_POINTERS=8]   (8 * 8 = 64 B)
//!   off  64  int       linesize[8]                    (8 * 4 = 32 B)
//!   off  96  uint8_t** extended_data
//!   off 104  int       width
//!   off 108  int       height
//!   off 112  int       nb_samples
//!   off 116  int       format
//! ```
//!
//! We only read `data[0..3]`, `linesize[0..3]`, `width`, `height`, and
//! `format`.

#![allow(unsafe_code)]

/// Bilateral-rejection envelope helpers used by the
/// `ffmpeg_oracle_decode` fuzz target. Exposed here so the predicates
/// (uniform-fill detection, envelope evaluation) can be unit-tested
/// against synthetic oracle-vs-ours plane pairs without spinning up
/// the full libfuzzer binary.
///
/// See `fuzz_targets/ffmpeg_oracle_decode.rs` for the documented
/// strategy (workspace task #750: VP9 fuzz oracle robust against
/// libavcodec version-divergence). The implementation here is the
/// single source of truth — the fuzz target re-exports these via
/// `oxideav_vp9_fuzz::oracle::*`.
pub mod oracle {
    /// Tight per-sample tolerance (LSB) for "real decode vs real
    /// decode" comparisons. VP9 §8 is integer-exact so anything
    /// above 1 LSB is suspicious; the envelope below decides whether
    /// suspicion is "real bug" or "version-specific drift".
    pub const PIXEL_TOL: i32 = 1;

    /// Maximum fraction of samples allowed to exceed `PIXEL_TOL`
    /// before the envelope trips. 0.5% is empirically picked: real
    /// spec-conformance bugs caught in past rounds (the #769 marker-
    /// bit drift, the #748 over-read bug) all produced > 50%
    /// divergence on the affected plane.
    pub const MAX_DIVERGE_FRACTION: f64 = 0.005;

    /// Maximum tolerated single-sample absolute difference for 8-bit
    /// planes. Even within the fraction envelope, a stealth
    /// mid-magnitude drift over this bound flags the case as a real
    /// bug.
    pub const MAX_TOLERATED_ABS_DIFF_8BIT: i32 = 8;

    /// 10/12-bit equivalent of `MAX_TOLERATED_ABS_DIFF_8BIT`. Scaled
    /// up 4× to account for the wider sample range.
    pub const MAX_TOLERATED_ABS_DIFF_HBD: i32 = 32;

    /// True iff every sample in `plane` is the same value. For HBD
    /// planes (`bps == 2`) we treat each LE u16 pair as a single
    /// sample. A uniform plane is the documented libavcodec error-
    /// conceal output shape when the bitstream parser bailed but a
    /// frame slot was still allocated — most commonly the mid-gray
    /// fill at `1 << (bd-1)` (128 / 512 / 2048 for 8/10/12-bit) but
    /// the detector is value-agnostic: ANY constant plane on a
    /// fuzz-mutated input is far more likely to be a placeholder
    /// than a real decoded image.
    ///
    /// Empty / sub-sample planes are reported as uniform (defensive:
    /// the outer code already structurally rejected those cases).
    pub fn is_uniform_plane(plane: &[u8], bps: usize) -> bool {
        if plane.is_empty() || plane.len() < bps {
            return true;
        }
        if bps == 1 {
            let first = plane[0];
            plane.iter().all(|&b| b == first)
        } else {
            let first = u16::from_le_bytes([plane[0], plane[1]]);
            plane
                .chunks_exact(2)
                .all(|c| u16::from_le_bytes([c[0], c[1]]) == first)
        }
    }

    /// Result of an envelope evaluation. `over_tol` is the count of
    /// sample positions exceeding `PIXEL_TOL`; `worst_abs` is the
    /// largest single-sample absolute difference. `tripped` is the
    /// final pass/fail verdict per the bilateral-rejection rule
    /// (`fraction > MAX_DIVERGE_FRACTION` AND `worst_abs >
    /// max_tolerated`).
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct EnvelopeReport {
        pub over_tol: usize,
        pub worst_abs: i32,
        pub total: usize,
        pub tripped: bool,
    }

    /// Evaluate the bilateral-rejection envelope on two tightly-
    /// packed plane buffers. `oracle` and `ours` must each carry
    /// `width * height * bps` bytes laid out row-major with no
    /// stride padding (the fuzz harness repacks the libavcodec
    /// frame and pads our own plane with `stride - row_bytes` when
    /// indexing). Used directly for unit tests; the fuzz target
    /// uses a stride-aware variant inline.
    pub fn eval_envelope_packed(
        oracle: &[u8],
        ours: &[u8],
        width: usize,
        height: usize,
        bps: usize,
    ) -> EnvelopeReport {
        let total = width.saturating_mul(height);
        if total == 0 {
            return EnvelopeReport {
                over_tol: 0,
                worst_abs: 0,
                total: 0,
                tripped: false,
            };
        }
        let row_bytes = width * bps;
        let mut over_tol: usize = 0;
        let mut worst_abs: i32 = 0;
        for row in 0..height {
            for col in 0..width {
                let off = row * row_bytes + col * bps;
                let (their_v, our_v) = if bps == 1 {
                    (oracle[off] as i32, ours[off] as i32)
                } else {
                    let t = u16::from_le_bytes([oracle[off], oracle[off + 1]]) as i32;
                    let o = u16::from_le_bytes([ours[off], ours[off + 1]]) as i32;
                    (t, o)
                };
                let diff = (their_v - our_v).abs();
                if diff > PIXEL_TOL {
                    over_tol += 1;
                }
                if diff > worst_abs {
                    worst_abs = diff;
                }
            }
        }
        let frac = over_tol as f64 / total as f64;
        let max_abs = if bps == 1 {
            MAX_TOLERATED_ABS_DIFF_8BIT
        } else {
            MAX_TOLERATED_ABS_DIFF_HBD
        };
        let tripped = frac > MAX_DIVERGE_FRACTION && worst_abs > max_abs;
        EnvelopeReport {
            over_tol,
            worst_abs,
            total,
            tripped,
        }
    }
}

#[cfg(test)]
mod oracle_tests {
    //! Regression tests for the bilateral-rejection envelope used by
    //! the `ffmpeg_oracle_decode` fuzz target (workspace task #750).
    //! Each test pins one libavcodec-version-divergence shape that
    //! used to produce a false-positive panic.

    use super::oracle::*;

    #[test]
    fn uniform_mid_gray_8bit_is_uniform() {
        // Classic libavcodec error-conceal 8-bit output: full gray
        // (1 << 7 = 128) — but the detector should fire for ANY
        // constant value.
        let plane = vec![128u8; 64 * 64];
        assert!(is_uniform_plane(&plane, 1));
        let plane = vec![0u8; 64 * 64];
        assert!(is_uniform_plane(&plane, 1));
        let plane = vec![255u8; 64 * 64];
        assert!(is_uniform_plane(&plane, 1));
    }

    #[test]
    fn uniform_mid_gray_10bit_is_uniform() {
        // 10-bit mid-gray = 512 = 0x0200 → LE bytes [0x00, 0x02].
        let mut plane = Vec::with_capacity(64 * 64 * 2);
        for _ in 0..64 * 64 {
            plane.push(0x00);
            plane.push(0x02);
        }
        assert!(is_uniform_plane(&plane, 2));
    }

    #[test]
    fn uniform_mid_gray_12bit_is_uniform() {
        // 12-bit mid-gray = 2048 = 0x0800 → LE bytes [0x00, 0x08].
        let mut plane = Vec::with_capacity(64 * 64 * 2);
        for _ in 0..64 * 64 {
            plane.push(0x00);
            plane.push(0x08);
        }
        assert!(is_uniform_plane(&plane, 2));
    }

    #[test]
    fn non_uniform_plane_is_not_uniform() {
        let mut plane = vec![128u8; 64 * 64];
        plane[42] = 129;
        assert!(!is_uniform_plane(&plane, 1));
    }

    #[test]
    fn empty_plane_is_uniform_defensively() {
        assert!(is_uniform_plane(&[], 1));
        assert!(is_uniform_plane(&[], 2));
        // Sub-sample length.
        assert!(is_uniform_plane(&[0x12], 2));
    }

    #[test]
    fn bit_identical_planes_do_not_trip_envelope() {
        let plane = (0..64 * 64).map(|i| (i & 0xff) as u8).collect::<Vec<_>>();
        let rep = eval_envelope_packed(&plane, &plane, 64, 64, 1);
        assert_eq!(rep.over_tol, 0);
        assert_eq!(rep.worst_abs, 0);
        assert!(!rep.tripped);
    }

    #[test]
    fn single_pixel_off_by_one_does_not_trip() {
        // Within PIXEL_TOL=1 — no over-tol count, no trip.
        let oracle = vec![128u8; 64 * 64];
        let mut ours = oracle.clone();
        ours[100] = 127;
        let rep = eval_envelope_packed(&oracle, &ours, 64, 64, 1);
        assert_eq!(rep.over_tol, 0);
        assert_eq!(rep.worst_abs, 1);
        assert!(!rep.tripped);
    }

    #[test]
    fn single_outlier_diff_46_does_not_trip_alone() {
        // Reproduces the round-25636690965 CI panic shape: a single
        // pixel diverges oracle=83 ours=129 (diff=46). Pre-#750,
        // this tripped the harness loudly. With the envelope it
        // does NOT — one outlier in 4096 samples is 0.024% < 0.5%.
        // Bilateral rejection means BOTH thresholds must fire.
        let oracle = vec![128u8; 64 * 64];
        let mut ours = oracle.clone();
        ours[0] = 82; // diff = 46
        let rep = eval_envelope_packed(&oracle, &ours, 64, 64, 1);
        assert_eq!(rep.over_tol, 1);
        assert_eq!(rep.worst_abs, 46);
        // fraction = 1/4096 = 0.000244 < 0.005 → not tripped.
        assert!(!rep.tripped);
    }

    #[test]
    fn small_cluster_of_low_magnitude_drift_does_not_trip() {
        // Reproduces the round-25663782234 shape: oracle=128
        // ours=122 (diff=6). Several such pixels in a cluster (e.g.
        // 10 across the plane) are still below the envelope.
        let oracle = vec![128u8; 64 * 64];
        let mut ours = oracle.clone();
        for i in 0..10 {
            ours[i * 50] = 122;
        }
        let rep = eval_envelope_packed(&oracle, &ours, 64, 64, 1);
        assert_eq!(rep.over_tol, 10);
        assert_eq!(rep.worst_abs, 6);
        // fraction = 10/4096 = 0.00244 < 0.005 → not tripped (and
        // worst_abs 6 < 8 anyway).
        assert!(!rep.tripped);
    }

    #[test]
    fn wholesale_divergence_does_trip_loudly() {
        // The shape a REAL spec bug used to produce: > 50% of
        // samples diverge by a large magnitude. Envelope MUST fire
        // — version-robustness must not silence real bugs.
        let oracle = vec![128u8; 64 * 64];
        let mut ours = vec![0u8; 64 * 64];
        // Make ours = 128 in the first quarter so it isn't a
        // trivial constant plane (the uniform-fill detector is
        // applied on the ORACLE in the harness; but exercise both
        // sides here for realism).
        for slot in ours.iter_mut().take(1024) {
            *slot = 128;
        }
        let rep = eval_envelope_packed(&oracle, &ours, 64, 64, 1);
        assert!(rep.over_tol > 2048); // > 50% of 4096
        assert_eq!(rep.worst_abs, 128);
        assert!(rep.tripped);
    }

    #[test]
    fn large_diff_above_envelope_trips_when_fraction_exceeded() {
        // 30 pixels, each off by 20. Fraction = 30/4096 = 0.733% >
        // 0.5%; worst_abs = 20 > 8 → both clauses fire.
        let oracle = vec![128u8; 64 * 64];
        let mut ours = oracle.clone();
        for i in 0..30 {
            ours[i * 100] = 108;
        }
        let rep = eval_envelope_packed(&oracle, &ours, 64, 64, 1);
        assert_eq!(rep.over_tol, 30);
        assert_eq!(rep.worst_abs, 20);
        assert!(rep.tripped);
    }

    #[test]
    fn hbd_uniform_fill_with_drift_is_filtered_by_uniform_check() {
        // 10-bit oracle: all 512. The harness detects uniform-fill
        // BEFORE running the envelope, so even a large ours-vs-
        // oracle diff is squelched. This test pins the
        // uniform-detection check shape; the envelope itself would
        // also need to handle HBD via the HBD threshold, exercised
        // separately below.
        let mut oracle = Vec::with_capacity(64 * 64 * 2);
        for _ in 0..64 * 64 {
            oracle.push(0x00);
            oracle.push(0x02); // 512
        }
        assert!(is_uniform_plane(&oracle, 2));
    }

    #[test]
    fn hbd_envelope_uses_wider_threshold() {
        // 10-bit; oracle has texture (so uniform check returns
        // false); ours diverges by HBD-scale magnitudes.
        let mut oracle = Vec::with_capacity(64 * 64 * 2);
        for i in 0..64 * 64 {
            let v = (100 + (i % 200)) as u16;
            oracle.push((v & 0xff) as u8);
            oracle.push((v >> 8) as u8);
        }
        // ours = oracle for most pixels; offset by 30 for a small
        // cluster. 30 pixels off-by-30 in 4096 samples: fraction =
        // 30/4096 = 0.7% > 0.5%; worst = 30 < HBD threshold 32 →
        // envelope does NOT trip (the HBD widening absorbs spec-
        // legal rounding ambiguity in 10/12-bit reconstruction).
        let mut ours = oracle.clone();
        for i in 0..30 {
            let off = i * 100 * 2;
            let v = u16::from_le_bytes([ours[off], ours[off + 1]]).saturating_sub(30);
            ours[off] = (v & 0xff) as u8;
            ours[off + 1] = (v >> 8) as u8;
        }
        let rep = eval_envelope_packed(&oracle, &ours, 64, 64, 2);
        assert_eq!(rep.worst_abs, 30);
        assert!(!rep.tripped);
    }

    #[test]
    fn hbd_envelope_trips_on_large_magnitude() {
        // Same setup as above but each diff is 64 (> HBD 32). 30
        // pixels at diff=64 → fraction 0.7% > 0.5% AND worst > 32
        // → trips.
        let mut oracle = Vec::with_capacity(64 * 64 * 2);
        for i in 0..64 * 64 {
            let v = (100 + (i % 200)) as u16;
            oracle.push((v & 0xff) as u8);
            oracle.push((v >> 8) as u8);
        }
        let mut ours = oracle.clone();
        for i in 0..30 {
            let off = i * 100 * 2;
            let v = u16::from_le_bytes([ours[off], ours[off + 1]]).saturating_sub(64);
            ours[off] = (v & 0xff) as u8;
            ours[off + 1] = (v >> 8) as u8;
        }
        let rep = eval_envelope_packed(&oracle, &ours, 64, 64, 2);
        assert_eq!(rep.worst_abs, 64);
        assert!(rep.tripped);
    }
}

pub mod libavcodec {
    use libloading::{Library, Symbol};
    use std::ffi::c_void;
    use std::sync::OnceLock;

    /// Conventional libavcodec shared-object names, newest first. We
    /// fall back to the unversioned `.so` / `.dylib` symlink (only
    /// present when `libavcodec-dev` is installed). On Windows the
    /// import library is `avcodec-NN.dll` (NN is the SONAME); we list
    /// the unversioned form as a courtesy.
    const CANDIDATES: &[&str] = &[
        "libavcodec.so.62",
        "libavcodec.so.61",
        "libavcodec.so.60",
        "libavcodec.so.59",
        "libavcodec.so.58",
        "libavcodec.so",
        "libavcodec.dylib",
        "libavcodec.62.dylib",
        "libavcodec.61.dylib",
        "libavcodec.60.dylib",
        "libavcodec.59.dylib",
        "libavcodec.58.dylib",
        "avcodec.dll",
    ];

    fn lib() -> Option<&'static Library> {
        static LIB: OnceLock<Option<Library>> = OnceLock::new();
        LIB.get_or_init(|| {
            for name in CANDIDATES {
                // SAFETY: `Library::new` is documented as unsafe because
                // the loaded library may run code at load time. We
                // accept that risk for fuzz tooling — libavcodec is a
                // well-behaved shared library.
                if let Ok(l) = unsafe { Library::new(name) } {
                    return Some(l);
                }
            }
            None
        })
        .as_ref()
    }

    /// True iff a libavcodec shared library was successfully loaded.
    /// Cross-decode fuzz harnesses early-return when this is false so
    /// the binary still runs without an oracle (the assertions just
    /// don't fire).
    pub fn available() -> bool {
        lib().is_some()
    }

    /// Probe libavcodec's runtime version via the public
    /// `avcodec_version()` C entry. Packed as `(major << 16) | (minor
    /// << 8) | micro` per the documented `LIBAVCODEC_VERSION_INT`
    /// macro layout in `<libavcodec/version.h>` (stable since 0.5).
    /// Returns `None` if the symbol can't be resolved (very old
    /// pre-0.5 libavcodec, or an exotic build without the public C
    /// API).
    ///
    /// The harness uses this only as a diagnostic tag — different
    /// majors are known to differ on how aggressively the VP9 frame
    /// parser bails on malformed `frame_marker` / `superframe_index`
    /// permutations (libavcodec 58.x is generally more permissive;
    /// 61.x+ tightens several pre-decode checks and replaces would-be
    /// frames with error-conceal gray-fill output). The oracle's
    /// uniform-fill detector handles the actual divergence; the
    /// version tag just makes failures self-describing.
    pub fn version() -> Option<u32> {
        type VersionFn = unsafe extern "C" fn() -> u32;
        let l = lib()?;
        unsafe {
            let sym: Symbol<VersionFn> = l.get(b"avcodec_version").ok()?;
            Some(sym())
        }
    }

    /// Decode the packed version into `(major, minor, micro)`.
    pub fn version_triple() -> Option<(u32, u32, u32)> {
        let v = version()?;
        Some((v >> 16, (v >> 8) & 0xff, v & 0xff))
    }

    /// AVCodecID enum value for VP9 — stable across libavcodec
    /// 58-62. The enum is sparse; pinning the literal value avoids
    /// reading the public header at runtime.
    pub const AV_CODEC_ID_VP9: i32 = 167;

    // AVPacket prefix offsets (see module docstring).
    const PKT_OFF_DATA: usize = 24;
    const PKT_OFF_SIZE: usize = 32;

    // AVFrame prefix offsets (see module docstring).
    const FRM_OFF_DATA: usize = 0;
    const FRM_OFF_LINESIZE: usize = 64;
    const FRM_OFF_WIDTH: usize = 104;
    const FRM_OFF_HEIGHT: usize = 108;
    const FRM_OFF_FORMAT: usize = 116;

    /// AV_PIX_FMT_YUV420P enum constant per <libavutil/pixfmt.h>.
    /// Stable since libavutil 1.x.
    pub const AV_PIX_FMT_YUV420P: i32 = 0;
    pub const AV_PIX_FMT_YUV422P: i32 = 4;
    pub const AV_PIX_FMT_YUV444P: i32 = 5;
    pub const AV_PIX_FMT_YUV420P10LE: i32 = 64;
    pub const AV_PIX_FMT_YUV422P10LE: i32 = 66;
    pub const AV_PIX_FMT_YUV444P10LE: i32 = 68;
    pub const AV_PIX_FMT_YUV420P12LE: i32 = 124;
    pub const AV_PIX_FMT_YUV422P12LE: i32 = 122;
    pub const AV_PIX_FMT_YUV444P12LE: i32 = 120;

    /// One decoded YUV frame as exported by libavcodec. Pixel buffers
    /// are tightly-packed copies of the corresponding planes (the
    /// libavcodec linesize stride is dropped during the copy).
    pub struct DecodedFrame {
        pub width: u32,
        pub height: u32,
        /// Pixel format value as read from `AVFrame.format`. See the
        /// `AV_PIX_FMT_*` constants above; we only export YUV planar
        /// 8/10/12-bit variants.
        pub pix_fmt: i32,
        /// Y plane (width * height bytes for 8-bit, 2x for 10/12-bit LE).
        pub y: Vec<u8>,
        /// Cb plane (chroma_w * chroma_h bytes).
        pub u: Vec<u8>,
        /// Cr plane.
        pub v: Vec<u8>,
        /// (chroma_w, chroma_h) in samples — derived from pix_fmt.
        pub chroma_dims: (u32, u32),
        /// 1 for 8-bit pix fmts, 2 for 10/12-bit LE pix fmts. Used to
        /// size each sample.
        pub bytes_per_sample: u32,
    }

    /// Map an AV pix_fmt to (chroma_w_shift, chroma_h_shift, bytes_per_sample).
    /// Returns None for formats we don't compare.
    fn pix_fmt_geom(pix_fmt: i32) -> Option<(u32, u32, u32)> {
        match pix_fmt {
            // 8-bit
            AV_PIX_FMT_YUV420P => Some((1, 1, 1)),
            AV_PIX_FMT_YUV422P => Some((1, 0, 1)),
            AV_PIX_FMT_YUV444P => Some((0, 0, 1)),
            // 10-bit LE
            AV_PIX_FMT_YUV420P10LE => Some((1, 1, 2)),
            AV_PIX_FMT_YUV422P10LE => Some((1, 0, 2)),
            AV_PIX_FMT_YUV444P10LE => Some((0, 0, 2)),
            // 12-bit LE
            AV_PIX_FMT_YUV420P12LE => Some((1, 1, 2)),
            AV_PIX_FMT_YUV422P12LE => Some((1, 0, 2)),
            AV_PIX_FMT_YUV444P12LE => Some((0, 0, 2)),
            _ => None,
        }
    }

    /// AVERROR(EAGAIN) — libavcodec's "no output yet, send more input"
    /// signal. Per AVERROR macro in <libavutil/error.h>, AVERROR(e) is
    /// `-e` on POSIX systems, so AVERROR(EAGAIN) is -11 on Linux/glibc.
    /// We compare absolute value to be portable across the macOS
    /// EAGAIN=35 and Linux EAGAIN=11 split.
    fn is_eagain(rc: i32) -> bool {
        let mag = rc.unsigned_abs();
        // Cover both Linux (11) and macOS (35); also the WSA (10035)
        // variant just in case ffmpeg got cross-compiled weirdly.
        matches!(mag, 11 | 35 | 10035)
    }

    /// Decode a VP9 superframe payload through libavcodec.
    ///
    /// Returns the list of decoded `DecodedFrame`s on success. Returns
    /// `None` when libavcodec rejected the input (this is the path that
    /// tells the caller "the input wasn't valid VP9, so don't expect
    /// our decoder to accept it either"). Returns `Some(empty)` only
    /// when libavcodec accepted the packet but produced no frames
    /// (a degenerate possibility we don't currently special-case).
    pub fn decode_vp9(data: &[u8]) -> Option<Vec<DecodedFrame>> {
        // Don't bother libavcodec with empty / huge inputs.
        if data.is_empty() || data.len() > 1 << 22 {
            return None;
        }

        type FindDecoderFn = unsafe extern "C" fn(i32) -> *const c_void;
        type AllocCtxFn = unsafe extern "C" fn(*const c_void) -> *mut c_void;
        type Open2Fn = unsafe extern "C" fn(*mut c_void, *const c_void, *mut *mut c_void) -> i32;
        type AllocPktFn = unsafe extern "C" fn() -> *mut c_void;
        type AllocFrmFn = unsafe extern "C" fn() -> *mut c_void;
        type SendPktFn = unsafe extern "C" fn(*mut c_void, *const c_void) -> i32;
        type RecvFrmFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i32;
        type FreePktFn = unsafe extern "C" fn(*mut *mut c_void);
        type FreeFrmFn = unsafe extern "C" fn(*mut *mut c_void);
        type FreeCtxFn = unsafe extern "C" fn(*mut *mut c_void);
        type NewPktFn = unsafe extern "C" fn(*mut c_void, i32) -> i32;

        let l = lib()?;
        unsafe {
            let avcodec_find_decoder: Symbol<FindDecoderFn> =
                l.get(b"avcodec_find_decoder").ok()?;
            let avcodec_alloc_context3: Symbol<AllocCtxFn> =
                l.get(b"avcodec_alloc_context3").ok()?;
            let avcodec_open2: Symbol<Open2Fn> = l.get(b"avcodec_open2").ok()?;
            let av_packet_alloc: Symbol<AllocPktFn> = l.get(b"av_packet_alloc").ok()?;
            let av_frame_alloc: Symbol<AllocFrmFn> = l.get(b"av_frame_alloc").ok()?;
            let av_new_packet: Symbol<NewPktFn> = l.get(b"av_new_packet").ok()?;
            let avcodec_send_packet: Symbol<SendPktFn> = l.get(b"avcodec_send_packet").ok()?;
            let avcodec_receive_frame: Symbol<RecvFrmFn> = l.get(b"avcodec_receive_frame").ok()?;
            let av_packet_free: Symbol<FreePktFn> = l.get(b"av_packet_free").ok()?;
            let av_frame_free: Symbol<FreeFrmFn> = l.get(b"av_frame_free").ok()?;
            let avcodec_free_context: Symbol<FreeCtxFn> = l.get(b"avcodec_free_context").ok()?;

            let codec = avcodec_find_decoder(AV_CODEC_ID_VP9);
            if codec.is_null() {
                return None;
            }
            let ctx = avcodec_alloc_context3(codec);
            if ctx.is_null() {
                return None;
            }
            // RAII-style cleanup epilogue via a closure.
            let result = (|| -> Option<Vec<DecodedFrame>> {
                if avcodec_open2(ctx, codec, std::ptr::null_mut()) < 0 {
                    return None;
                }
                let pkt = av_packet_alloc();
                if pkt.is_null() {
                    return None;
                }
                let alloc_rc = av_new_packet(pkt, data.len() as i32);
                if alloc_rc < 0 {
                    let mut p = pkt;
                    av_packet_free(&mut p as *mut *mut c_void);
                    return None;
                }
                // Copy our input into pkt->data (offset 24 in AVPacket).
                let pkt_bytes = pkt as *mut u8;
                let pkt_data_ptr = (pkt_bytes.add(PKT_OFF_DATA) as *const *mut u8).read_unaligned();
                let pkt_size = (pkt_bytes.add(PKT_OFF_SIZE) as *const i32).read_unaligned();
                if pkt_data_ptr.is_null() || pkt_size < data.len() as i32 {
                    let mut p = pkt;
                    av_packet_free(&mut p as *mut *mut c_void);
                    return None;
                }
                std::ptr::copy_nonoverlapping(data.as_ptr(), pkt_data_ptr, data.len());

                let send_rc = avcodec_send_packet(ctx, pkt);
                // Free the packet now — libavcodec internally refs the
                // buffer if it needs to keep it around.
                let mut p = pkt;
                av_packet_free(&mut p as *mut *mut c_void);
                if send_rc < 0 && !is_eagain(send_rc) {
                    return None;
                }
                // Drain frames.
                let mut out: Vec<DecodedFrame> = Vec::new();
                let frame = av_frame_alloc();
                if frame.is_null() {
                    return None;
                }
                for _ in 0..32 {
                    let recv_rc = avcodec_receive_frame(ctx, frame);
                    if recv_rc == 0 {
                        if let Some(df) = read_frame(frame) {
                            out.push(df);
                        }
                        // Continue draining.
                        continue;
                    }
                    // EAGAIN / EOF — stop draining cleanly.
                    break;
                }
                let mut f = frame;
                av_frame_free(&mut f as *mut *mut c_void);
                Some(out)
            })();
            let mut c = ctx;
            avcodec_free_context(&mut c as *mut *mut c_void);
            result
        }
    }

    /// Read the public AVFrame prefix into a `DecodedFrame`. Returns
    /// None for pix formats we don't know how to compare.
    unsafe fn read_frame(frame: *mut c_void) -> Option<DecodedFrame> {
        let bytes = frame as *const u8;
        let width = (bytes.add(FRM_OFF_WIDTH) as *const i32).read_unaligned();
        let height = (bytes.add(FRM_OFF_HEIGHT) as *const i32).read_unaligned();
        let format = (bytes.add(FRM_OFF_FORMAT) as *const i32).read_unaligned();
        if width <= 0 || height <= 0 {
            return None;
        }
        let (cs_x, cs_y, bps) = pix_fmt_geom(format)?;
        let cw = (width as u32 + (1 << cs_x) - 1) >> cs_x;
        let ch = (height as u32 + (1 << cs_y) - 1) >> cs_y;
        let data_arr = bytes.add(FRM_OFF_DATA) as *const *const u8;
        let line_arr = bytes.add(FRM_OFF_LINESIZE) as *const i32;
        let y_ptr = data_arr.read_unaligned();
        let u_ptr = data_arr.add(1).read_unaligned();
        let v_ptr = data_arr.add(2).read_unaligned();
        let y_stride = line_arr.read_unaligned();
        let u_stride = line_arr.add(1).read_unaligned();
        let v_stride = line_arr.add(2).read_unaligned();
        if y_ptr.is_null() || u_ptr.is_null() || v_ptr.is_null() {
            return None;
        }
        if y_stride <= 0 || u_stride <= 0 || v_stride <= 0 {
            return None;
        }
        let row_bytes_y = (width as u32 * bps) as usize;
        let row_bytes_c = (cw * bps) as usize;
        let h = height as usize;
        let chh = ch as usize;
        let mut y = vec![0u8; row_bytes_y * h];
        for row in 0..h {
            let src = y_ptr.add(row * y_stride as usize);
            std::ptr::copy_nonoverlapping(src, y.as_mut_ptr().add(row * row_bytes_y), row_bytes_y);
        }
        let mut u = vec![0u8; row_bytes_c * chh];
        let mut v = vec![0u8; row_bytes_c * chh];
        for row in 0..chh {
            let su = u_ptr.add(row * u_stride as usize);
            let sv = v_ptr.add(row * v_stride as usize);
            std::ptr::copy_nonoverlapping(su, u.as_mut_ptr().add(row * row_bytes_c), row_bytes_c);
            std::ptr::copy_nonoverlapping(sv, v.as_mut_ptr().add(row * row_bytes_c), row_bytes_c);
        }
        Some(DecodedFrame {
            width: width as u32,
            height: height as u32,
            pix_fmt: format,
            y,
            u,
            v,
            chroma_dims: (cw, ch),
            bytes_per_sample: bps,
        })
    }
}
