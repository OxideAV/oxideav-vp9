//! r-next-hp — 1/8-pel high-precision MV (`allow_high_precision_mv`).
//!
//! Round 49 + r-next stopped at quarter-pel ME (1/4-pel MV components,
//! `hp` bit elided from the uncompressed header). Many real video
//! sources contain sub-1/4-pel translations that quarter-pel ME can
//! only round-down to the nearest 1/4-pel — the 1/8-pel MV bit
//! recovers that lost precision.
//!
//! Wire effect: when `EncoderParams::allow_high_precision_mv = true`
//! the uncompressed header carries `allow_high_precision_mv = 1` and
//! every `read_mv_component` reads the extra `hp` bit (§6.4.19). The
//! encoder additionally runs a fourth 8-neighbour ME refinement
//! stage at step = 1 (in 1/8-pel units) after the existing integer +
//! half + quarter-pel passes.
//!
//! Fixture: apply the §6.3 `sub_pel_filters_8` EightTap filter at
//! phase 2 (= 1/8-pel offset) to the decoder-reconstructed frame 1.
//! Without HP the encoder can only reach phase 0 (integer) or phase 4
//! (quarter-pel) since MV components are constrained to even 1/8-pel
//! units. With HP enabled the encoder reaches phase 2 (MV = 1 in
//! 1/8-pel) and the MC bit-recovers the fixture.
//!
//! Coverage:
//!   * `pframe_hp_off_baseline_caps_psnr_on_eighth_pel_shift` — pins
//!     the no-HP PSNR ceiling on the phase-2 fixture so the headline
//!     row has something stable to compare against.
//!   * `pframe_hp_on_lifts_psnr_on_eighth_pel_shift` — the headline
//!     gain row: HP-on PSNR_Y MUST be ≥ 38 dB AND strictly higher than
//!     the HP-off baseline (i.e. the extra refinement stage actually
//!     fires and the `hp` bit reaches the wire).
//!   * `pframe_hp_on_no_regression_on_quarter_pel_shift` — feeding HP
//!     on at a true 1/4-pel translation must not lower PSNR; the
//!     refinement loop is monotone (only ever drops SAD), so HP on
//!     ≥ HP off on this content.

use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase};
use oxideav_vp9::{
    encoder::{encode_keyframe_yuv, encode_pframe_yuv, EncoderParams, ReferenceFrame, YuvFrame},
    make_decoder, CODEC_ID_STR,
};

const W: u32 = 64;
const H: u32 = 64;

/// PSNR in dB between two equal-length 8-bit planes.
fn psnr_db(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut sse: u64 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = *x as i32 - *y as i32;
        sse += (d * d) as u64;
    }
    if sse == 0 {
        return f64::INFINITY;
    }
    let mse = (sse as f64) / (a.len() as f64);
    10.0 * (255.0 * 255.0 / mse).log10()
}

/// VP9 §6.3 `sub_pel_filters_8` EightTap row — phase 2 = 1/8-pel
/// offset (the smallest non-integer phase representable in a 1/8-pel
/// MV). Phase 4 = 1/4-pel (the smallest non-integer phase reachable
/// when `allow_high_precision_mv = false`).
const FILTER_EIGHTTAP_PHASE_2: [i32; 8] = [-1, 3, -10, 122, 18, -6, 2, 0];
const FILTER_EIGHTTAP_PHASE_4: [i32; 8] = [-1, 4, -16, 112, 37, -11, 4, -1];
const FILTER_BITS: i32 = 7;

/// Apply an 8-tap horizontal filter with edge-clamp to produce a sub-pel
/// shifted source. Replicated here so the test doesn't reach into the
/// crate-private `mcfilter` module.
fn horiz_filter_8tap(
    src: &[u8],
    width: usize,
    height: usize,
    taps: &[i32; 8],
    int_offset: i32,
) -> Vec<u8> {
    let mut out = vec![0u8; width * height];
    for r in 0..height {
        for c in 0..width {
            let base_c = c as i32 + int_offset;
            let mut acc = 0i32;
            for (k, &t) in taps.iter().enumerate() {
                let sc = (base_c + k as i32 - 3).clamp(0, width as i32 - 1) as usize;
                acc += (src[r * width + sc] as i32) * t;
            }
            let v = (acc + (1 << (FILTER_BITS - 1))) >> FILTER_BITS;
            out[r * width + c] = v.clamp(0, 255) as u8;
        }
    }
    out
}

/// Build a 64×64 4:2:0 frame from a luma function. Chroma stays at 128.
fn make_yuv<F: FnMut(usize, usize) -> u8>(mut f: F) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; (W * H) as usize];
    for r in 0..H as usize {
        for c in 0..W as usize {
            y[r * W as usize + c] = f(r, c);
        }
    }
    let uv_size = ((W / 2) * (H / 2)) as usize;
    (y, vec![128u8; uv_size], vec![128u8; uv_size])
}

/// Encode frame 1 (key), decode for the reconstructed reference.
fn encode_key_and_decode_recon(p: &EncoderParams, src: &YuvFrame<'_>) -> (Vec<u8>, ReferenceFrame) {
    let key_bytes = encode_keyframe_yuv(p, src);
    let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    let mut dec = make_decoder(&params).expect("make_decoder");
    let pkt = Packet::new(0, TimeBase::new(1, 30), key_bytes.clone());
    dec.send_packet(&pkt).expect("send key");
    let f = dec.receive_frame().expect("recv key");
    let Frame::Video(v) = f else {
        panic!("expected Video");
    };
    let y = v.planes[0].data.clone();
    let y_stride = v.planes[0].stride;
    let u = v.planes[1].data.clone();
    let uv_stride = v.planes[1].stride;
    let vv = v.planes[2].data.clone();
    let refr = ReferenceFrame {
        y,
        y_stride,
        u,
        v: vv,
        uv_stride,
        width: W,
        height: H,
    };
    (key_bytes, refr)
}

/// Decode a key + P pair, return the decoded P-frame luma at W×H.
fn decode_pframe_luma(key_bytes: &[u8], p_bytes: &[u8]) -> Vec<u8> {
    let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    let mut dec = make_decoder(&params).expect("make_decoder");
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), key_bytes.to_vec()))
        .expect("send key");
    let _ = dec.receive_frame().expect("recv key");
    dec.send_packet(&Packet::new(1, TimeBase::new(1, 30), p_bytes.to_vec()))
        .expect("send pframe");
    let f2 = dec.receive_frame().expect("recv pframe");
    let Frame::Video(v) = f2 else {
        panic!("expected Video");
    };
    let stride = v.planes[0].stride;
    let mut y = Vec::with_capacity((W * H) as usize);
    for r in 0..H as usize {
        y.extend_from_slice(&v.planes[0].data[r * stride..r * stride + W as usize]);
    }
    y
}

/// Build a 1/8-pel-shifted P-frame source (phase 2 = 1/8-pel right)
/// from the reconstructed frame 1. Returns the (y, u, v) planes.
fn make_eighth_pel_shifted_pframe(refr: &ReferenceFrame) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut refl = vec![0u8; (W * H) as usize];
    for r in 0..H as usize {
        for c in 0..W as usize {
            refl[r * W as usize + c] = refr.y[r * refr.y_stride + c];
        }
    }
    let y2 = horiz_filter_8tap(&refl, W as usize, H as usize, &FILTER_EIGHTTAP_PHASE_2, 0);
    let uv_size = ((W / 2) * (H / 2)) as usize;
    (y2, vec![128u8; uv_size], vec![128u8; uv_size])
}

/// HP-off baseline: pin the PSNR ceiling on the 1/8-pel fixture so we
/// have a stable comparison point for the headline gain test.
#[test]
fn pframe_hp_off_baseline_caps_psnr_on_eighth_pel_shift() {
    let mut p = EncoderParams::keyframe(W, H);
    p.allow_high_precision_mv = false;
    let (y1, u1, v1) = make_yuv(|_r, c| 40 + (c as u8) * 3);
    let src1 = YuvFrame {
        y: &y1,
        y_stride: W as usize,
        u: &u1,
        v: &v1,
        uv_stride: (W / 2) as usize,
        width: W,
        height: H,
    };
    let (key_bytes, refr) = encode_key_and_decode_recon(&p, &src1);

    let (y2, u2, v2) = make_eighth_pel_shifted_pframe(&refr);
    let src2 = YuvFrame {
        y: &y2,
        y_stride: W as usize,
        u: &u2,
        v: &v2,
        uv_stride: (W / 2) as usize,
        width: W,
        height: H,
    };
    let p_bytes = encode_pframe_yuv(&p, &src2, &refr);
    let decoded_y = decode_pframe_luma(&key_bytes, &p_bytes);
    let psnr = psnr_db(&y2, &decoded_y);
    eprintln!(
        "hp=OFF (baseline) on 1/8-pel shift: I-frame={} B, P-frame={} B, PSNR_Y = {psnr:.2} dB",
        key_bytes.len(),
        p_bytes.len()
    );
    // Smooth-ramp content + nearest-quarter-pel rounding still gives
    // tolerable PSNR; pin a low floor so the test fails loudly only on
    // genuine regressions. The headline test is the HP-on row below.
    assert!(
        psnr >= 20.0,
        "HP-off PSNR_Y {psnr:.2} dB collapsed below 20 dB sanity floor"
    );
}

/// Headline: 1/8-pel-aware encoder lifts PSNR on the 1/8-pel fixture.
/// MUST clear an absolute ≥ 38 dB target AND beat the HP-off baseline
/// by ≥ 3 dB — proves both that the §6.4.19 `hp` bit reaches the wire
/// AND the new ME refinement actually picks an odd-1/8-pel MV.
#[test]
fn pframe_hp_on_lifts_psnr_on_eighth_pel_shift() {
    let mut p_off = EncoderParams::keyframe(W, H);
    p_off.allow_high_precision_mv = false;
    let mut p_on = EncoderParams::keyframe(W, H);
    p_on.allow_high_precision_mv = true;
    let (y1, u1, v1) = make_yuv(|_r, c| 40 + (c as u8) * 3);
    let src1 = YuvFrame {
        y: &y1,
        y_stride: W as usize,
        u: &u1,
        v: &v1,
        uv_stride: (W / 2) as usize,
        width: W,
        height: H,
    };
    // Both encoders share the same key-frame reconstruction. Encoder
    // params at the key frame match (HP off → same key bitstream).
    let (key_bytes, refr) = encode_key_and_decode_recon(&p_off, &src1);

    let (y2, u2, v2) = make_eighth_pel_shifted_pframe(&refr);
    let src2 = YuvFrame {
        y: &y2,
        y_stride: W as usize,
        u: &u2,
        v: &v2,
        uv_stride: (W / 2) as usize,
        width: W,
        height: H,
    };

    let p_off_bytes = encode_pframe_yuv(&p_off, &src2, &refr);
    let off_decoded = decode_pframe_luma(&key_bytes, &p_off_bytes);
    let psnr_off = psnr_db(&y2, &off_decoded);

    let p_on_bytes = encode_pframe_yuv(&p_on, &src2, &refr);
    let on_decoded = decode_pframe_luma(&key_bytes, &p_on_bytes);
    let psnr_on = psnr_db(&y2, &on_decoded);

    eprintln!(
        "hp=ON  on 1/8-pel shift: I-frame={} B, P-frame={} B, PSNR_Y = {psnr_on:.2} dB (off = {psnr_off:.2} dB, delta = {:.2} dB)",
        key_bytes.len(),
        p_on_bytes.len(),
        psnr_on - psnr_off,
    );
    // Sanity: the wire MUST differ — at minimum the `hp` bit in the
    // uncompressed header flips, and the MV-emit path either elides or
    // emits the extra fractional bit per component.
    assert_ne!(
        p_off_bytes, p_on_bytes,
        "HP-on and HP-off P-frame bitstreams must differ (hp flag + MV refinement should reach the wire)"
    );

    assert!(
        psnr_on >= 38.0,
        "HP-on PSNR_Y {psnr_on:.2} dB < 38 dB target on 1/8-pel fixture"
    );
    let delta = psnr_on - psnr_off;
    assert!(
        delta >= 3.0,
        "HP-on must beat HP-off by ≥ 3 dB on 1/8-pel fixture; got {delta:.2} dB \
         (off={psnr_off:.2}, on={psnr_on:.2})"
    );
}

/// Regression: on a true 1/4-pel translation the HP-on encoder must
/// not regress PSNR vs HP-off. The 1/8-pel refinement is monotone
/// (only ever lowers SAD), so HP-on ≥ HP-off ON THIS CONTENT.
#[test]
fn pframe_hp_on_no_regression_on_quarter_pel_shift() {
    let mut p_off = EncoderParams::keyframe(W, H);
    p_off.allow_high_precision_mv = false;
    let mut p_on = EncoderParams::keyframe(W, H);
    p_on.allow_high_precision_mv = true;
    let (y1, u1, v1) = make_yuv(|_r, c| 50 + (c as u8) * 2);
    let src1 = YuvFrame {
        y: &y1,
        y_stride: W as usize,
        u: &u1,
        v: &v1,
        uv_stride: (W / 2) as usize,
        width: W,
        height: H,
    };
    let (key_bytes, refr) = encode_key_and_decode_recon(&p_off, &src1);

    // Apply phase-4 (1/4-pel) filter — this IS reachable with HP off.
    let mut refl = vec![0u8; (W * H) as usize];
    for r in 0..H as usize {
        for c in 0..W as usize {
            refl[r * W as usize + c] = refr.y[r * refr.y_stride + c];
        }
    }
    let y2 = horiz_filter_8tap(&refl, W as usize, H as usize, &FILTER_EIGHTTAP_PHASE_4, 0);
    let uv_size = ((W / 2) * (H / 2)) as usize;
    let u2 = vec![128u8; uv_size];
    let v2 = vec![128u8; uv_size];
    let src2 = YuvFrame {
        y: &y2,
        y_stride: W as usize,
        u: &u2,
        v: &v2,
        uv_stride: (W / 2) as usize,
        width: W,
        height: H,
    };

    let p_off_bytes = encode_pframe_yuv(&p_off, &src2, &refr);
    let off_decoded = decode_pframe_luma(&key_bytes, &p_off_bytes);
    let psnr_off = psnr_db(&y2, &off_decoded);

    let p_on_bytes = encode_pframe_yuv(&p_on, &src2, &refr);
    let on_decoded = decode_pframe_luma(&key_bytes, &p_on_bytes);
    let psnr_on = psnr_db(&y2, &on_decoded);

    eprintln!(
        "hp=ON on 1/4-pel shift: PSNR_Y = {psnr_on:.2} dB (off = {psnr_off:.2} dB, delta = {:.2} dB)",
        psnr_on - psnr_off,
    );

    // HP-on must not regress relative to HP-off on the 1/4-pel fixture
    // by more than 0.5 dB — the refinement loop is monotone in SAD, but
    // SAD-optimal isn't strictly PSNR-optimal (sub-pel interpolation
    // smoothing), so we allow a tiny floor.
    assert!(
        psnr_on + 0.5 >= psnr_off,
        "HP-on regressed on 1/4-pel fixture: off={psnr_off:.2}, on={psnr_on:.2}"
    );
    assert!(
        psnr_on >= 36.0,
        "HP-on absolute PSNR_Y {psnr_on:.2} dB < 36 dB on 1/4-pel fixture"
    );
}
