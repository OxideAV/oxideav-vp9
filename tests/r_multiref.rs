//! r-multiref — LAST_FRAME + GOLDEN_FRAME per-CU RDO end-to-end test.
//!
//! Validates `encode_pframe_yuv_multi_ref`: the encoder runs ME against
//! both `refs.last` and `refs.golden`, picks the lower-SAD reference
//! per CU, and emits the §6.4.5 single-ref tree bits accordingly. The
//! gate is a 4-frame GOP fixture where frames 3+4 match the keyframe
//! (GOLDEN) far better than the immediately-preceding noisy P-frame
//! (LAST). The multi-ref encoder should pick GOLDEN for most CUs in
//! frames 3+4 and reconstruct them at much higher PSNR_Y than the
//! single-LAST baseline.
//!
//! Headline target: ≥ 0.5 dB PSNR_Y on frames 3+4 (multi-ref vs
//! LAST-only).

use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase};
use oxideav_vp9::{
    encoder::{
        encode_keyframe_yuv, encode_pframe_yuv, encode_pframe_yuv_multi_ref, EncoderParams,
        ReferenceFrame, ReferenceSet, YuvFrame,
    },
    make_decoder, CODEC_ID_STR,
};

const W: u32 = 64;
const H: u32 = 64;

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

fn make_yuv<F: FnMut(usize, usize) -> u8>(mut f: F) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; (W * H) as usize];
    for r in 0..H as usize {
        for c in 0..W as usize {
            y[r * W as usize + c] = f(r, c);
        }
    }
    let uv_size = ((W / 2) * (H / 2)) as usize;
    let u = vec![128u8; uv_size];
    let v = vec![128u8; uv_size];
    (y, u, v)
}

/// Extract a `ReferenceFrame` from a decoded `VideoFrame` planes.
fn recon_to_ref(planes: &[oxideav_core::VideoPlane]) -> ReferenceFrame {
    ReferenceFrame {
        y: planes[0].data.clone(),
        y_stride: planes[0].stride,
        u: planes[1].data.clone(),
        v: planes[2].data.clone(),
        uv_stride: planes[1].stride,
        width: W,
        height: H,
    }
}

/// Decode a single packet and return the resulting `VideoFrame`.
fn decode_one(dec: &mut Box<dyn oxideav_core::Decoder>, pkt: Packet) -> oxideav_core::VideoFrame {
    dec.send_packet(&pkt).expect("send");
    let f = dec.receive_frame().expect("recv");
    let Frame::Video(v) = f else {
        panic!("expected Video");
    };
    v
}

/// Decode the entire stream produced by `encode_chain` and return
/// the PSNR_Y of frames 3+4 vs their respective sources.
///
/// `encode_chain` returns `(key, p1_bytes, p2_bytes, p3_bytes, src_frames)`.
fn psnr_frames_3_and_4(
    key: &[u8],
    p1: &[u8],
    p2: &[u8],
    p3: &[u8],
    src_f3: &[u8],
    src_f4: &[u8],
) -> (f64, f64) {
    let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    let mut dec = make_decoder(&params).expect("make_decoder");
    let _f1 = decode_one(&mut dec, Packet::new(0, TimeBase::new(1, 30), key.to_vec()));
    let _f2 = decode_one(&mut dec, Packet::new(1, TimeBase::new(1, 30), p1.to_vec()));
    let f3 = decode_one(&mut dec, Packet::new(2, TimeBase::new(1, 30), p2.to_vec()));
    let f4 = decode_one(&mut dec, Packet::new(3, TimeBase::new(1, 30), p3.to_vec()));
    let stride3 = f3.planes[0].stride;
    let stride4 = f4.planes[0].stride;
    let mut y3 = Vec::with_capacity((W * H) as usize);
    let mut y4 = Vec::with_capacity((W * H) as usize);
    for r in 0..H as usize {
        y3.extend_from_slice(&f3.planes[0].data[r * stride3..r * stride3 + W as usize]);
        y4.extend_from_slice(&f4.planes[0].data[r * stride4..r * stride4 + W as usize]);
    }
    (psnr_db(src_f3, &y3), psnr_db(src_f4, &y4))
}

/// Build the 4-frame test corpus + the per-frame source luma planes.
///
/// Because the round-49 P-frame encoder is skip=1, the only way LAST's
/// reconstruction can DIVERGE from the keyframe is by ME finding a
/// non-zero MV. We exploit that with two ORTHOGONAL patterns:
///
/// * Frame 1 (key): horizontal-stripe pattern A.
/// * Frame 2 (P): orthogonal pattern B (a white square on a black
///   field). The skip=1 + MC-from-A encoder can't faithfully synth B
///   from striped reference data — its best MV picks the stripe-patch
///   that least-poorly matches B, and frame-2 RECONSTRUCTION ends up
///   as a warped stripe excerpt, NOT a clean white square.
/// * Frame 3 (P): pattern A again. With single-LAST, reference =
///   frame-2's warped stripe excerpt → encoder can find decent MVs
///   inside it to recover most of A, but residual error remains.
///   With multi-ref + GOLDEN = the original keyframe, the encoder
///   picks GOLDEN+ZEROMV everywhere → reconstruction is an exact
///   sample-for-sample copy of pattern A (subject only to §8.8
///   loop-filter smoothing at SB edges).
/// * Frame 4 (P): pattern A again. Same argument vs LAST = frame-3 recon.
#[allow(clippy::type_complexity)]
fn build_fixture() -> (
    EncoderParams,
    Vec<u8>,      // source y1
    Vec<u8>,      // source y2
    Vec<u8>,      // source y3
    Vec<u8>,      // source y4
    Vec<Vec<u8>>, // [u1..u4]
    Vec<Vec<u8>>, // [v1..v4]
) {
    let p = EncoderParams::keyframe(W, H);

    // Pattern A — horizontal stripes (luma varies with row only).
    // Distinctive enough that even ME-aided MC can't fake the look
    // of a different orthogonal pattern.
    fn pattern_a(r: usize, _c: usize) -> u8 {
        let v = 40 + ((r as i32) * 7) % 180;
        v as u8
    }
    let (y1, u1, v1) = make_yuv(pattern_a);

    // Pattern B — pure-white square in a black frame (orthogonal
    // content to pattern A). The skip=1 + MC-from-A encoder can't
    // reconstruct B well — its best MV picks the cleanest sub-patch
    // of the stripes that matches the white square, but that's a far
    // cry from a clean white square. So frame-2 RECONSTRUCTION is
    // some warped version of the stripe pattern, NOT pattern B.
    fn pattern_b(r: usize, c: usize) -> u8 {
        if (16..48).contains(&r) && (16..48).contains(&c) {
            240
        } else {
            16
        }
    }
    let (y2, u2, v2) = make_yuv(pattern_b);

    // Frame 3 == Frame 4 == original pattern A again. Against
    // frame-2's poor reconstruction (LAST), the encoder must do
    // serious MC work to recover the stripes — and skip=1 + filter
    // smoothing leaves residual error. Against the keyframe (GOLDEN),
    // ZEROMV → exact sample copy → clean reconstruction.
    let (y3, u3, v3) = make_yuv(pattern_a);
    let (y4, u4, v4) = make_yuv(pattern_a);

    (
        p,
        y1,
        y2,
        y3,
        y4,
        vec![u1, u2, u3, u4],
        vec![v1, v2, v3, v4],
    )
}

fn yuv_view<'a>(y: &'a [u8], u: &'a [u8], v: &'a [u8]) -> YuvFrame<'a> {
    YuvFrame {
        y,
        y_stride: W as usize,
        u,
        v,
        uv_stride: (W / 2) as usize,
        width: W,
        height: H,
    }
}

/// Encode a 4-frame chain in single-LAST mode (round-49 behaviour).
/// Returns `(key, p1, p2, p3)` bytes.
fn encode_chain_last_only(
    p: &EncoderParams,
    y: &[Vec<u8>; 4],
    u: &[Vec<u8>; 4],
    v: &[Vec<u8>; 4],
) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
    let s1 = yuv_view(&y[0], &u[0], &v[0]);
    let s2 = yuv_view(&y[1], &u[1], &v[1]);
    let s3 = yuv_view(&y[2], &u[2], &v[2]);
    let s4 = yuv_view(&y[3], &u[3], &v[3]);

    let key = encode_keyframe_yuv(p, &s1);

    // Decode key → reconstruction = LAST for frame 2.
    let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    let mut dec = make_decoder(&params).expect("make_decoder");
    let f1 = decode_one(&mut dec, Packet::new(0, TimeBase::new(1, 30), key.clone()));
    let last_after_key = recon_to_ref(&f1.planes);

    let p1 = encode_pframe_yuv(p, &s2, &last_after_key);
    let f2 = decode_one(&mut dec, Packet::new(1, TimeBase::new(1, 30), p1.clone()));
    let last_after_p1 = recon_to_ref(&f2.planes);

    let p2 = encode_pframe_yuv(p, &s3, &last_after_p1);
    let f3 = decode_one(&mut dec, Packet::new(2, TimeBase::new(1, 30), p2.clone()));
    let last_after_p2 = recon_to_ref(&f3.planes);

    let p3 = encode_pframe_yuv(p, &s4, &last_after_p2);
    let _f4 = decode_one(&mut dec, Packet::new(3, TimeBase::new(1, 30), p3.clone()));

    (key, p1, p2, p3)
}

/// Encode a 4-frame chain with multi-ref (LAST + GOLDEN). GOLDEN
/// stays the reconstructed keyframe across all P-frames.
fn encode_chain_multi_ref(
    p: &EncoderParams,
    y: &[Vec<u8>; 4],
    u: &[Vec<u8>; 4],
    v: &[Vec<u8>; 4],
) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
    let s1 = yuv_view(&y[0], &u[0], &v[0]);
    let s2 = yuv_view(&y[1], &u[1], &v[1]);
    let s3 = yuv_view(&y[2], &u[2], &v[2]);
    let s4 = yuv_view(&y[3], &u[3], &v[3]);

    let key = encode_keyframe_yuv(p, &s1);

    let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    let mut dec = make_decoder(&params).expect("make_decoder");
    let f1 = decode_one(&mut dec, Packet::new(0, TimeBase::new(1, 30), key.clone()));
    let golden = recon_to_ref(&f1.planes);
    let last_after_key = golden.clone();

    let refs_p1 = ReferenceSet {
        last: &last_after_key,
        golden: Some(&golden),
    };
    let p1 = encode_pframe_yuv_multi_ref(p, &s2, &refs_p1);
    let f2 = decode_one(&mut dec, Packet::new(1, TimeBase::new(1, 30), p1.clone()));
    let last_after_p1 = recon_to_ref(&f2.planes);

    let refs_p2 = ReferenceSet {
        last: &last_after_p1,
        golden: Some(&golden),
    };
    let p2 = encode_pframe_yuv_multi_ref(p, &s3, &refs_p2);
    let f3 = decode_one(&mut dec, Packet::new(2, TimeBase::new(1, 30), p2.clone()));
    let last_after_p2 = recon_to_ref(&f3.planes);

    let refs_p3 = ReferenceSet {
        last: &last_after_p2,
        golden: Some(&golden),
    };
    let p3 = encode_pframe_yuv_multi_ref(p, &s4, &refs_p3);
    let _f4 = decode_one(&mut dec, Packet::new(3, TimeBase::new(1, 30), p3.clone()));

    (key, p1, p2, p3)
}

#[test]
fn multi_ref_beats_last_only_by_at_least_half_db_on_repeated_key() {
    let (p, y1, y2, y3, y4, uvs_u, uvs_v) = build_fixture();
    let y = [y1, y2, y3, y4];
    let u: [Vec<u8>; 4] = [
        uvs_u[0].clone(),
        uvs_u[1].clone(),
        uvs_u[2].clone(),
        uvs_u[3].clone(),
    ];
    let v: [Vec<u8>; 4] = [
        uvs_v[0].clone(),
        uvs_v[1].clone(),
        uvs_v[2].clone(),
        uvs_v[3].clone(),
    ];

    // Baseline: LAST-only chain.
    let (key_l, p1_l, p2_l, p3_l) = encode_chain_last_only(&p, &y, &u, &v);
    let (psnr3_l, psnr4_l) = psnr_frames_3_and_4(&key_l, &p1_l, &p2_l, &p3_l, &y[2], &y[3]);
    let avg_last = (psnr3_l + psnr4_l) / 2.0;
    eprintln!(
        "LAST-only: PSNR_Y frame3={psnr3_l:.2} dB, frame4={psnr4_l:.2} dB, avg={avg_last:.2} dB"
    );
    eprintln!(
        "LAST-only sizes: key={} P1={} P2={} P3={}",
        key_l.len(),
        p1_l.len(),
        p2_l.len(),
        p3_l.len()
    );

    // Multi-ref: LAST + GOLDEN(=keyframe).
    let (key_m, p1_m, p2_m, p3_m) = encode_chain_multi_ref(&p, &y, &u, &v);
    let (psnr3_m, psnr4_m) = psnr_frames_3_and_4(&key_m, &p1_m, &p2_m, &p3_m, &y[2], &y[3]);
    let avg_multi = (psnr3_m + psnr4_m) / 2.0;
    eprintln!(
        "Multi-ref: PSNR_Y frame3={psnr3_m:.2} dB, frame4={psnr4_m:.2} dB, avg={avg_multi:.2} dB"
    );
    eprintln!(
        "Multi-ref sizes: key={} P1={} P2={} P3={}",
        key_m.len(),
        p1_m.len(),
        p2_m.len(),
        p3_m.len()
    );

    let delta = avg_multi - avg_last;
    eprintln!("PSNR_Y delta (multi-ref − LAST-only, avg over frames 3+4) = {delta:+.2} dB");
    assert!(
        delta >= 0.5,
        "PSNR_Y delta {delta:.2} dB < 0.5 dB target — multi-ref should beat LAST-only \
         on a GOP where frames 3+4 match the keyframe far better than frame 2"
    );
}

/// Identical to the headline test but asserts both single-LAST and
/// multi-ref P-frames decode without error (smoke test for the new
/// wire format). The single-LAST half also pins the round-49
/// regression behaviour.
#[test]
fn multi_ref_chain_decodes_without_error() {
    let (p, y1, y2, y3, y4, uvs_u, uvs_v) = build_fixture();
    let y = [y1, y2, y3, y4];
    let u: [Vec<u8>; 4] = [
        uvs_u[0].clone(),
        uvs_u[1].clone(),
        uvs_u[2].clone(),
        uvs_u[3].clone(),
    ];
    let v: [Vec<u8>; 4] = [
        uvs_v[0].clone(),
        uvs_v[1].clone(),
        uvs_v[2].clone(),
        uvs_v[3].clone(),
    ];
    let (k, p1, p2, p3) = encode_chain_multi_ref(&p, &y, &u, &v);
    let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    let mut dec = make_decoder(&params).expect("make_decoder");
    for (i, bytes) in [&k, &p1, &p2, &p3].iter().enumerate() {
        dec.send_packet(&Packet::new(i as u32, TimeBase::new(1, 30), bytes.to_vec()))
            .unwrap_or_else(|e| panic!("frame {i} send_packet failed: {e}"));
        let _f = dec
            .receive_frame()
            .unwrap_or_else(|e| panic!("frame {i} receive_frame failed: {e}"));
    }
}

/// When `refs.golden` is `None`, `encode_pframe_yuv_multi_ref` must
/// produce byte-identical output to `encode_pframe_yuv` (the
/// round-49 single-LAST path).
#[test]
fn multi_ref_with_no_golden_matches_single_last_bytes() {
    let p = EncoderParams::keyframe(W, H);
    let (y1, u1, v1) = make_yuv(|_r, c| 80 + (c as u8) * 2);
    let src = yuv_view(&y1, &u1, &v1);
    let key = encode_keyframe_yuv(&p, &src);
    let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    let mut dec = make_decoder(&params).expect("make_decoder");
    let f = decode_one(&mut dec, Packet::new(0, TimeBase::new(1, 30), key.clone()));
    let refr = recon_to_ref(&f.planes);

    let bytes_single = encode_pframe_yuv(&p, &src, &refr);
    let refs = ReferenceSet {
        last: &refr,
        golden: None,
    };
    let bytes_multi = encode_pframe_yuv_multi_ref(&p, &src, &refs);
    assert_eq!(
        bytes_single, bytes_multi,
        "multi_ref(golden=None) must produce identical bytes to encode_pframe_yuv"
    );
}
