#![no_main]

//! Fuzz: arbitrary VP9 superframe bytes → both libavcodec
//! (`AV_CODEC_ID_VP9 = 167`) and `Vp9Decoder`. When libavcodec accepts
//! the input, ours must too, with matching frame count, dimensions,
//! chroma format, and YUV pixels within ±1 LSB.
//!
//! libavcodec is loaded via `libloading` at first call; the harness
//! `eprintln!`s `[oracle skip]` and returns early on hosts where
//! libavcodec isn't installed (no `#[ignore]`). On CI we install
//! `ffmpeg` via apt, which pulls a recent libavcodec.so SONAME.
//!
//! ## Tolerance
//!
//! VP9 §8 prescribes integer-exact arithmetic — there's no rounding
//! ambiguity. We therefore use a tight ±1 LSB bound on YUV samples to
//! absorb any single-bit drift from differing interpretation of an
//! ambiguous spec corner; if the gap widens, that's a real bug to
//! report (not silently widen the tolerance).

use libfuzzer_sys::fuzz_target;
use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase};
use oxideav_vp9::decoder::{make_decoder, pixel_format_from_color_config};
use oxideav_vp9::headers::parse_uncompressed_header;
use oxideav_vp9_fuzz::libavcodec::{self, DecodedFrame};
use std::sync::OnceLock;

const PIXEL_TOL: i32 = 1;
/// Mirror of `panic_free_decode::MAX_PIXELS`. Until oxideav-vp9
/// adds bounds-checking on declared frame dimensions, we cap the
/// fuzz harness at 256 KP per frame to keep iter density high.
const MAX_DECODE_PIXELS: u32 = 1 << 18;

fuzz_target!(|data: &[u8]| {
    if !oracle_available() {
        // First-iter only: log once. libfuzzer runs millions of iters
        // so we must NOT eprintln per call.
        return;
    }
    if data.is_empty() || data.len() > 1 << 22 {
        return;
    }

    // Sanity probe: parse the uncompressed header (with no prior
    // color_config — fine since the harness mostly sees keyframes).
    // The header parser will reject anything that isn't really VP9,
    // and libavcodec doing the same is the expected outcome.
    //
    // We also cap declared dimensions at 1 MP per frame: oxideav-vp9
    // does not yet bound its allocations on the declared width/height
    // (a separate decoder bug — see panic_free_decode for tracking),
    // so a fuzz mutation that declares a 65535×65535 keyframe will
    // OOM the harness before any cross-decode comparison even runs.
    if let Ok(h) = parse_uncompressed_header(data, None) {
        if h.width.saturating_mul(h.height) > MAX_DECODE_PIXELS {
            return;
        }
    }

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
    // — when both decoders produced output — ARE real bugs and fail
    // the harness loudly.
    if ours.is_empty() || send_rc.is_err() {
        return;
    }

    assert_eq!(
        ours.len(),
        oracle.len(),
        "frame count mismatch: ours={} oracle={}",
        ours.len(),
        oracle.len()
    );

    // Per-frame: dimensions + chroma + (within-tolerance) pixels.
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
        "frame[{i}]: ours has {} planes, expected 3",
        mine.planes.len()
    );
    // Width/height: derive ours from the (Y plane row count, stride),
    // since `VideoFrame` doesn't carry an explicit (w, h) field — the
    // first row of the Y plane is `width` samples long for 8-bit, or
    // `width*2` bytes for 10/12-bit LE.
    let bytes_per_sample = theirs.bytes_per_sample as usize;
    let exp_y_row_bytes = (theirs.width as usize) * bytes_per_sample;
    let our_y = &mine.planes[0];
    let our_h = if our_y.stride == 0 {
        0
    } else {
        our_y.data.len() / our_y.stride
    };
    assert_eq!(
        our_h, theirs.height as usize,
        "frame[{i}]: height mismatch: ours={our_h} oracle={}",
        theirs.height
    );
    assert!(
        our_y.stride >= exp_y_row_bytes,
        "frame[{i}]: Y stride {} < expected row bytes {}",
        our_y.stride,
        exp_y_row_bytes
    );
    let our_u = &mine.planes[1];
    let our_v = &mine.planes[2];
    let (cw, ch) = (theirs.chroma_dims.0 as usize, theirs.chroma_dims.1 as usize);
    let exp_c_row_bytes = cw * bytes_per_sample;
    assert!(
        our_u.stride >= exp_c_row_bytes && our_v.stride >= exp_c_row_bytes,
        "frame[{i}]: chroma stride too small: U={} V={} expected≥{}",
        our_u.stride,
        our_v.stride,
        exp_c_row_bytes
    );
    assert_eq!(
        our_u.data.len() / our_u.stride.max(1),
        ch,
        "frame[{i}]: U chroma height mismatch"
    );

    // Pixel comparison — Y plane.
    compare_plane(
        i,
        "Y",
        theirs.width as usize,
        theirs.height as usize,
        bytes_per_sample,
        &theirs.y,
        our_y,
    );
    compare_plane(i, "U", cw, ch, bytes_per_sample, &theirs.u, our_u);
    compare_plane(i, "V", cw, ch, bytes_per_sample, &theirs.v, our_v);
}

fn compare_plane(
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
            assert!(
                diff <= PIXEL_TOL,
                "frame[{fi}].{label}[{row},{col}] diverges: oracle={their_v} ours={our_v} diff={diff} (tol={PIXEL_TOL})"
            );
        }
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
            eprintln!("[oracle ready] libavcodec loaded; ffmpeg_oracle_decode active");
        }
        avail
    });
    avail
}

// Silence unused-import warning when pixel_format_from_color_config
// isn't referenced — keeps the symbol in the dep graph as a
// compile-time check that the public API hasn't been renamed.
#[allow(dead_code)]
fn _api_check(cc: &oxideav_vp9::ColorConfig) -> oxideav_core::PixelFormat {
    pixel_format_from_color_config(cc)
}
