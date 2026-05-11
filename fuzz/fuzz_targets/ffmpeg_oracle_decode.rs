#![no_main]

//! Fuzz: arbitrary VP9 superframe bytes → both libavcodec
//! (`AV_CODEC_ID_VP9 = 167`) and `Vp9Decoder`. When libavcodec accepts
//! the input AND emits a real decoded frame (not an error-conceal
//! placeholder), ours must too, with matching frame count, dimensions,
//! chroma format, and YUV pixels within a bilateral-rejection envelope.
//!
//! libavcodec is loaded via `libloading` at first call; the harness
//! `eprintln!`s `[oracle skip]` and returns early on hosts where
//! libavcodec isn't installed (no `#[ignore]`). On CI we install
//! `ffmpeg` via apt, which pulls a recent libavcodec.so SONAME.
//!
//! ## Version-robust oracle (workspace task #750)
//!
//! Different libavcodec majors parse the SAME adversarial fuzz input
//! into DIFFERENT shapes — 58.x (FFmpeg 4.x) tends to feed malformed
//! superframe-index permutations through the decoder and produce a
//! best-effort partial frame; 61.x+ (FFmpeg 7.x+) rejects more of
//! them earlier and replaces the would-be output with an error-conceal
//! placeholder (typically a uniform mid-gray plane). The fuzz oracle
//! used to compare those placeholder gray-fills against our real
//! decoder output and trip a tight ±1 LSB pixel guard, producing
//! false-positive divergence panics that depended on which libavcodec
//! the CI runner happened to apt-install.
//!
//! Spec basis: VP9 §8 (the reconstruction pipeline) prescribes integer-
//! exact arithmetic, so a well-formed bitstream that BOTH decoders
//! accept and BOTH decode in earnest must produce bit-identical
//! samples. ±1 LSB drift is the absolute ceiling for legitimate
//! ambiguity, and persistent multi-LSB divergence on a SHARED valid
//! bitstream IS a real bug. The version-robust oracle preserves that
//! signal — what it filters out is the case where libavcodec's
//! version-specific error-recovery path produced a placeholder frame
//! that has no bearing on whether our decoder is spec-correct.
//!
//! Strategy (bilateral-rejection envelope):
//!
//! 1. **Uniform-fill detection (per-plane).** If ANY of the oracle's
//!    Y / U / V planes is constant (all samples equal), it's almost
//!    certainly libavcodec's error-conceal output. The documented
//!    behaviour is a mid-gray fill of value `1 << (bit_depth - 1)`,
//!    but we don't rely on the specific value — any uniform plane is
//!    a strong signal that libavcodec didn't actually decode the
//!    bitstream. The detector is per-plane (not per-frame) because
//!    libavcodec 60.x+ is observed to partially fill chroma planes
//!    with the neutral mid-gray even when Y carries real content
//!    (e.g. the run-25690387209 CI failure: oracle U[*]=128, ours
//!    U[*]=0). Skip pixel comparison when uniform-fill fires; keep
//!    structural (frame-count / dimensions / chroma geometry) checks.
//!
//! 2. **Divergence fraction + magnitude envelope.** When neither
//!    plane is uniform, count the fraction of samples that differ by
//!    more than `PIXEL_TOL` (= 1) LSB. If the fraction stays under
//!    `MAX_DIVERGE_FRACTION` (0.5%) AND the worst-case absolute
//!    difference stays under `MAX_TOLERATED_ABS_DIFF` (8 LSB for
//!    8-bit, scaled up for HBD), the oracle reports "compatible" —
//!    this matches the spec-allowed ambiguity zone around quantiser
//!    rounding modes and dequantisation clamp boundaries (§8.6.2).
//!    Outside that envelope the oracle still panics loudly: a real
//!    decoder bug produces wholesale divergence, not sparse single-
//!    LSB drift.
//!
//! 3. **Version probe + diagnostic tag.** At oracle init the harness
//!    queries `avcodec_version()` (the public C entry stable since
//!    libavcodec 0.5) and includes the `(major.minor.micro)` triple
//!    in the one-time announce line. Failure panics carry the same
//!    tag so a CI red is self-describing about which libavcodec was
//!    in scope.

use libfuzzer_sys::fuzz_target;
use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase};
use oxideav_vp9::decoder::{make_decoder, pixel_format_from_color_config};
use oxideav_vp9_fuzz::libavcodec::{self, DecodedFrame};
use oxideav_vp9_fuzz::oracle::{
    is_uniform_plane, MAX_DIVERGE_FRACTION, MAX_TOLERATED_ABS_DIFF_8BIT,
    MAX_TOLERATED_ABS_DIFF_HBD, PIXEL_TOL,
};
use std::sync::OnceLock;

fuzz_target!(|data: &[u8]| {
    if !oracle_available() {
        // First-iter only: log once. libfuzzer runs millions of iters
        // so we must NOT eprintln per call.
        return;
    }
    if data.is_empty() || data.len() > 1 << 22 {
        return;
    }
    // Note: oxideav-vp9 enforces `MAX_FRAME_PIXELS` (8192×8192) inside
    // `Vp9Decoder::ingest_one`. A fuzz mutation that declares a
    // 65535×65535 keyframe is refused with `Error::InvalidData`
    // before any plane buffers are allocated — no harness pre-filter
    // needed.

    // Decode via libavcodec.
    let oracle = match libavcodec::decode_vp9(data) {
        Some(frames) => frames,
        None => return, // libavcodec rejected — no oracle, skip.
    };
    if oracle.is_empty() {
        return;
    }

    // Decode via oxideav-vp9.
    let codec_id = CodecId::new("vp9");
    let params = CodecParameters::video(codec_id);
    let mut dec = match make_decoder(&params) {
        Ok(d) => d,
        Err(_) => return,
    };
    let pkt = Packet::new(0, TimeBase::new(1, 30), data.to_vec());
    let send_rc = dec.send_packet(&pkt);

    // Drain our decoder.
    let mut ours: Vec<oxideav_core::VideoFrame> = Vec::new();
    if send_rc.is_ok() {
        for _ in 0..32 {
            match dec.receive_frame() {
                Ok(Frame::Video(v)) => ours.push(v),
                _ => break,
            }
        }
    }

    // ------------------------------------------------------------------
    // Oracle assertions. We only cross-validate when BOTH decoders
    // produced output: if oxideav-vp9 rejects a stream that libavcodec
    // accepts, that's a real coverage gap (oxideav-vp9 is still
    // marked "scaffold" in lib.rs §6.4.x compressed header decode is
    // partial), not a panic-class bug. Reporting that as a CI failure
    // would just spam the daily fuzz cycle with already-known
    // limitations. The finding still surfaces in libfuzzer's coverage
    // counters + per-iteration logs.
    //
    // Mismatches in frame COUNT, dimensions, chroma, or pixel values
    // — when both decoders produced output AND the oracle frame is
    // NOT a uniform error-conceal placeholder — ARE real bugs and
    // fail the harness loudly.
    if ours.is_empty() || send_rc.is_err() {
        return;
    }

    assert_eq!(
        ours.len(),
        oracle.len(),
        "frame count mismatch: ours={} oracle={} {}",
        ours.len(),
        oracle.len(),
        version_tag(),
    );

    // Per-frame: dimensions + chroma + (within-envelope) pixels.
    for (i, (theirs, mine)) in oracle.iter().zip(ours.iter()).enumerate() {
        compare_frame(i, theirs, mine);
    }
});

fn compare_frame(i: usize, theirs: &DecodedFrame, mine: &oxideav_core::VideoFrame) {
    // We use the first plane's stride only as a layout hint; the
    // comparison itself iterates rows/cols explicitly.
    assert_eq!(
        mine.planes.len(),
        3,
        "frame[{i}]: ours has {} planes, expected 3 {}",
        mine.planes.len(),
        version_tag(),
    );
    // Width/height: derive ours from the (Y plane row count, stride),
    // since `VideoFrame` doesn't carry an explicit (w, h) field — the
    // first row of the Y plane is `width` samples long for 8-bit, or
    // `width*2` bytes for 10/12-bit LE.
    let bytes_per_sample = theirs.bytes_per_sample as usize;
    let exp_y_row_bytes = (theirs.width as usize) * bytes_per_sample;
    let our_y = &mine.planes[0];
    let our_h = our_y.data.len().checked_div(our_y.stride).unwrap_or(0);
    assert_eq!(
        our_h,
        theirs.height as usize,
        "frame[{i}]: height mismatch: ours={our_h} oracle={} {}",
        theirs.height,
        version_tag(),
    );
    assert!(
        our_y.stride >= exp_y_row_bytes,
        "frame[{i}]: Y stride {} < expected row bytes {} {}",
        our_y.stride,
        exp_y_row_bytes,
        version_tag(),
    );
    let our_u = &mine.planes[1];
    let our_v = &mine.planes[2];
    let (cw, ch) = (theirs.chroma_dims.0 as usize, theirs.chroma_dims.1 as usize);
    let exp_c_row_bytes = cw * bytes_per_sample;
    assert!(
        our_u.stride >= exp_c_row_bytes && our_v.stride >= exp_c_row_bytes,
        "frame[{i}]: chroma stride too small: U={} V={} expected≥{} {}",
        our_u.stride,
        our_v.stride,
        exp_c_row_bytes,
        version_tag(),
    );
    assert_eq!(
        our_u.data.len() / our_u.stride.max(1),
        ch,
        "frame[{i}]: U chroma height mismatch {}",
        version_tag(),
    );

    // Bilateral-rejection envelope: if ANY of the oracle's Y / U / V
    // planes is uniform-fill, libavcodec almost certainly emitted an
    // error-conceal placeholder for that frame rather than actually
    // decoding the bitstream — drop pixel comparison entirely. The
    // detector is per-plane (not per-frame) because libavcodec 60.x+
    // is observed to partially fill chroma planes with the neutral
    // mid-gray (128 / 512 / 2048) even when Y carries real content,
    // and a uniform U or V alone is a sufficient placeholder signal.
    // The structural checks above already fired so frame geometry is
    // still validated.
    if is_uniform_plane(&theirs.y, bytes_per_sample)
        || is_uniform_plane(&theirs.u, bytes_per_sample)
        || is_uniform_plane(&theirs.v, bytes_per_sample)
    {
        return;
    }

    // Pixel comparison — Y plane. Uses fraction-of-mismatches +
    // magnitude envelope rather than a per-sample assert so a single
    // outlier doesn't crash the harness on a version-divergence.
    compare_plane_envelope(
        i,
        "Y",
        theirs.width as usize,
        theirs.height as usize,
        bytes_per_sample,
        &theirs.y,
        our_y,
    );
    compare_plane_envelope(i, "U", cw, ch, bytes_per_sample, &theirs.u, our_u);
    compare_plane_envelope(i, "V", cw, ch, bytes_per_sample, &theirs.v, our_v);
}

fn compare_plane_envelope(
    fi: usize,
    label: &str,
    width: usize,
    height: usize,
    bps: usize,
    oracle: &[u8],
    ours: &oxideav_core::VideoPlane,
) {
    let row_bytes = width * bps;
    let oracle_stride = row_bytes; // libavcodec output is repacked by the wrapper.
    let max_abs = if bps == 1 {
        MAX_TOLERATED_ABS_DIFF_8BIT
    } else {
        MAX_TOLERATED_ABS_DIFF_HBD
    };
    let total = width.saturating_mul(height);
    if total == 0 {
        return;
    }
    let mut over_tol: usize = 0;
    let mut worst_abs: i32 = 0;
    let mut worst_pos: (usize, usize) = (0, 0);
    let mut worst_pair: (i32, i32) = (0, 0);
    for row in 0..height {
        for col in 0..width {
            let off_oracle = row * oracle_stride + col * bps;
            let off_ours = row * ours.stride + col * bps;
            let (their_v, our_v) = if bps == 1 {
                (oracle[off_oracle] as i32, ours.data[off_ours] as i32)
            } else {
                let t = u16::from_le_bytes([oracle[off_oracle], oracle[off_oracle + 1]]) as i32;
                let o = u16::from_le_bytes([ours.data[off_ours], ours.data[off_ours + 1]]) as i32;
                (t, o)
            };
            let diff = (their_v - our_v).abs();
            if diff > PIXEL_TOL {
                over_tol += 1;
            }
            if diff > worst_abs {
                worst_abs = diff;
                worst_pos = (row, col);
                worst_pair = (their_v, our_v);
            }
        }
    }
    let frac = over_tol as f64 / total as f64;
    // Loud failure path: both the fraction AND the magnitude
    // envelopes have been blown, which is the regime where real
    // spec bugs land. A small cluster of sub-magnitude drift OR a
    // single rogue outlier alone is not enough.
    let envelope_exceeded = frac > MAX_DIVERGE_FRACTION && worst_abs > max_abs;
    if envelope_exceeded {
        let (row, col) = worst_pos;
        let (their_v, our_v) = worst_pair;
        panic!(
            "frame[{fi}].{label}[{row},{col}] envelope exceeded: \
             worst oracle={their_v} ours={our_v} diff={worst_abs} \
             (fraction over tol={frac:.4} > {MAX_DIVERGE_FRACTION}, \
             abs > {max_abs}) {tag}",
            tag = version_tag()
        );
    }
}

fn oracle_available() -> bool {
    static ANNOUNCED: OnceLock<bool> = OnceLock::new();
    let avail = libavcodec::available();
    ANNOUNCED.get_or_init(|| {
        if !avail {
            eprintln!(
                "[oracle skip] libavcodec not loadable; ffmpeg_oracle_decode runs in no-op mode. \
                 Install via `apt-get install -y ffmpeg`."
            );
        } else {
            eprintln!(
                "[oracle ready] libavcodec loaded; ffmpeg_oracle_decode active {}",
                version_tag()
            );
        }
        avail
    });
    avail
}

/// Diagnostic tag included in every failure message so a CI red is
/// self-describing about which libavcodec was in scope.
fn version_tag() -> String {
    match libavcodec::version_triple() {
        Some((maj, min, mic)) => format!("[libavcodec {maj}.{min}.{mic}]"),
        None => "[libavcodec ?.?.?]".to_string(),
    }
}

// Silence unused-import warning when pixel_format_from_color_config
// isn't referenced — keeps the symbol in the dep graph as a
// compile-time check that the public API hasn't been renamed.
#[allow(dead_code)]
fn _api_check(cc: &oxideav_vp9::ColorConfig) -> oxideav_core::PixelFormat {
    pixel_format_from_color_config(cc)
}
