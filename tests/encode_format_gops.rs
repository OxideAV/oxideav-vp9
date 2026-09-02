//! Round-455 public-surface pins of the **format-matrix wrappers** of
//! the structured-GOP (`encode_vp9_lossy_sequence_with_*`) and
//! resized (`encode_vp9_lossy_sequence_resized_*`) entries: 8-bit
//! 4:4:4 / 4:2:2 / 4:4:0 (profile 1) and 10 / 12-bit 4:2:0 / 4:4:4 /
//! 4:2:2 / 4:4:0 (profiles 2 / 3), each decoded back through the crate
//! at the declared format, byte-deterministic, distortion bounded.

use oxideav_vp9::{
    decode_vp9_sequence, encode_vp9_lossy_sequence_resized_422,
    encode_vp9_lossy_sequence_resized_440, encode_vp9_lossy_sequence_resized_444,
    encode_vp9_lossy_sequence_resized_hbd, encode_vp9_lossy_sequence_resized_hbd_422,
    encode_vp9_lossy_sequence_resized_hbd_440, encode_vp9_lossy_sequence_with_422,
    encode_vp9_lossy_sequence_with_440, encode_vp9_lossy_sequence_with_444,
    encode_vp9_lossy_sequence_with_hbd, encode_vp9_lossy_sequence_with_hbd_422,
    encode_vp9_lossy_sequence_with_hbd_440, Vp9DecodedFrame, Vp9GopConfig,
};

/// Chroma dims for a subsampling pair.
fn chroma(w: usize, h: usize, ssx: bool, ssy: bool) -> (usize, usize) {
    (
        if ssx { w.div_ceil(2) } else { w },
        if ssy { h.div_ceil(2) } else { h },
    )
}

/// Translating textured content, planar, at `bit_depth`.
fn scene16(w: usize, h: usize, ssx: bool, ssy: bool, bit_depth: u32, k: usize) -> Vec<u16> {
    let (cw, ch) = chroma(w, h, ssx, ssy);
    let shift = bit_depth - 8;
    let mut px = Vec::with_capacity(w * h + 2 * cw * ch);
    for y in 0..h {
        for x in 0..w {
            let mut v = ((x + 2 * k) * 3 + y * 2) % 200 + 20;
            let px_x = x as i64 - 2 * k as i64;
            if (8..24).contains(&px_x) && (8..24).contains(&y) {
                v = (px_x as usize * 37 + y * 53) % 255;
            }
            px.push(
                ((v as u32) << shift) as u16 | (if shift > 0 { (x + y) as u16 & 3 } else { 0 }),
            );
        }
    }
    for plane in 0..2usize {
        for y in 0..ch {
            for x in 0..cw {
                let v = ((x + k) * 5 + y * 3 + plane * 90) % 256;
                px.push(((v as u32) << shift) as u16);
            }
        }
    }
    px
}

fn to_u8(v: &[u16]) -> Vec<u8> {
    v.iter().map(|&s| s as u8).collect()
}

/// PSNR (dB) of decoded frames against 16-bit planar sources, at the
/// source bit depth.
fn psnr(frames: &[Vp9DecodedFrame], sources: &[Vec<u16>], bit_depth: u32) -> f64 {
    let max = f64::from((1u32 << bit_depth) - 1);
    let mut sse = 0f64;
    let mut n = 0usize;
    for (f, src) in frames.iter().zip(sources) {
        let dec: Vec<u16> = f.y.iter().chain(&f.u).chain(&f.v).copied().collect();
        assert_eq!(dec.len(), src.len(), "sample count");
        for (&a, &b) in dec.iter().zip(src) {
            let d = f64::from(a) - f64::from(b);
            sse += d * d;
        }
        n += dec.len();
    }
    let mse = sse / n as f64;
    if mse == 0.0 {
        99.0
    } else {
        10.0 * (max * max / mse).log10()
    }
}

fn check(
    name: &str,
    packets: &[Vec<u8>],
    sources: &[Vec<u16>],
    bit_depth: u32,
    ssx: bool,
    ssy: bool,
    dims: &[(u32, u32)],
) {
    let refs: Vec<&[u8]> = packets.iter().map(Vec::as_slice).collect();
    let decoded = decode_vp9_sequence(&refs).expect("decodes");
    assert_eq!(decoded.len(), sources.len(), "{name}: shown frames");
    for (i, f) in decoded.iter().enumerate() {
        assert_eq!(u32::from(f.bit_depth), bit_depth, "{name}: bit depth");
        assert_eq!(
            (f.subsampling_x, f.subsampling_y),
            (ssx, ssy),
            "{name}: subsampling"
        );
        assert_eq!(
            (f.width, f.height),
            dims[i.min(dims.len() - 1)],
            "{name}: frame {i} dims"
        );
    }
    let p = psnr(&decoded, sources, bit_depth);
    assert!(p > 27.0, "{name}: PSNR {p:.2} dB");
    eprintln!(
        "{name}: {} bytes @ {p:.2} dB",
        packets.iter().map(Vec::len).sum::<usize>()
    );
}

type Stream = (
    String,
    Vec<Vec<u8>>,
    Vec<Vec<u16>>,
    u32,
    bool,
    bool,
    Vec<(u32, u32)>,
);

/// Every wrapper's stream: `(name, packets, sources, bit_depth, ssx,
/// ssy, per-frame dims)`.
fn streams() -> Vec<Stream> {
    let (w, h) = (64usize, 48usize);
    let mut cfg = Vp9GopConfig::new(100);
    cfg.altref_interval = 2;
    let n = 4usize;
    let mut out: Vec<Stream> = Vec::new();
    // Structured 8-bit.
    for (name, ssx, ssy, f) in [
        (
            "with-444",
            false,
            false,
            encode_vp9_lossy_sequence_with_444
                as fn(
                    &[&[u8]],
                    u32,
                    u32,
                    &Vp9GopConfig,
                ) -> Result<Vec<Vec<u8>>, oxideav_vp9::Error>,
        ),
        ("with-422", true, false, encode_vp9_lossy_sequence_with_422),
        ("with-440", false, true, encode_vp9_lossy_sequence_with_440),
    ] {
        let src: Vec<Vec<u16>> = (0..n).map(|k| scene16(w, h, ssx, ssy, 8, k)).collect();
        let src8: Vec<Vec<u8>> = src.iter().map(|s| to_u8(s)).collect();
        let refs: Vec<&[u8]> = src8.iter().map(Vec::as_slice).collect();
        let packets = f(&refs, w as u32, h as u32, &cfg).expect(name);
        out.push((
            name.into(),
            packets,
            src,
            8,
            ssx,
            ssy,
            vec![(w as u32, h as u32)],
        ));
    }
    // Structured HBD.
    for (name, bd, ssx, ssy) in [
        ("with-hbd10-420", 10u32, true, true),
        ("with-hbd12-444", 12, false, false),
        ("with-hbd12-422", 12, true, false),
        ("with-hbd10-440", 10, false, true),
    ] {
        let src: Vec<Vec<u16>> = (0..n).map(|k| scene16(w, h, ssx, ssy, bd, k)).collect();
        let refs: Vec<&[u16]> = src.iter().map(Vec::as_slice).collect();
        let packets = match (ssx, ssy) {
            (true, false) => {
                encode_vp9_lossy_sequence_with_hbd_422(&refs, w as u32, h as u32, bd as u8, &cfg)
            }
            (false, true) => {
                encode_vp9_lossy_sequence_with_hbd_440(&refs, w as u32, h as u32, bd as u8, &cfg)
            }
            (s, _) => {
                encode_vp9_lossy_sequence_with_hbd(&refs, w as u32, h as u32, bd as u8, s, &cfg)
            }
        }
        .expect(name);
        out.push((
            name.into(),
            packets,
            src,
            bd,
            ssx,
            ssy,
            vec![(w as u32, h as u32)],
        ));
    }
    // Resized.
    let sizes: Vec<(u32, u32)> = vec![(64, 48), (48, 32), (64, 48)];
    for (name, ssx, ssy, f) in [
        (
            "resized-444",
            false,
            false,
            encode_vp9_lossy_sequence_resized_444
                as fn(&[&[u8]], &[(u32, u32)], u8) -> Result<Vec<Vec<u8>>, oxideav_vp9::Error>,
        ),
        (
            "resized-422",
            true,
            false,
            encode_vp9_lossy_sequence_resized_422,
        ),
        (
            "resized-440",
            false,
            true,
            encode_vp9_lossy_sequence_resized_440,
        ),
    ] {
        let src: Vec<Vec<u16>> = sizes
            .iter()
            .enumerate()
            .map(|(k, &(sw, sh))| scene16(sw as usize, sh as usize, ssx, ssy, 8, k))
            .collect();
        let src8: Vec<Vec<u8>> = src.iter().map(|s| to_u8(s)).collect();
        let refs: Vec<&[u8]> = src8.iter().map(Vec::as_slice).collect();
        let packets = f(&refs, &sizes, 100).expect(name);
        out.push((name.into(), packets, src, 8, ssx, ssy, sizes.clone()));
    }
    for (name, bd, ssx, ssy) in [
        ("resized-hbd10-420", 10u32, true, true),
        ("resized-hbd12-444", 12, false, false),
        ("resized-hbd12-422", 12, true, false),
        ("resized-hbd10-440", 10, false, true),
    ] {
        let src: Vec<Vec<u16>> = sizes
            .iter()
            .enumerate()
            .map(|(k, &(sw, sh))| scene16(sw as usize, sh as usize, ssx, ssy, bd, k))
            .collect();
        let refs: Vec<&[u16]> = src.iter().map(Vec::as_slice).collect();
        let packets = match (ssx, ssy) {
            (true, false) => {
                encode_vp9_lossy_sequence_resized_hbd_422(&refs, &sizes, bd as u8, 100)
            }
            (false, true) => {
                encode_vp9_lossy_sequence_resized_hbd_440(&refs, &sizes, bd as u8, 100)
            }
            (s, _) => encode_vp9_lossy_sequence_resized_hbd(&refs, &sizes, bd as u8, s, 100),
        }
        .expect(name);
        out.push((name.into(), packets, src, bd, ssx, ssy, sizes.clone()));
    }
    out
}

#[test]
fn format_matrix_wrappers_decode_at_their_declared_format() {
    for (name, packets, src, bd, ssx, ssy, dims) in streams() {
        check(&name, &packets, &src, bd, ssx, ssy, &dims);
    }
}

#[test]
fn format_matrix_wrappers_are_deterministic_and_reject_bad_depths() {
    let a = streams();
    let b = streams();
    for (x, y) in a.iter().zip(&b) {
        assert_eq!(x.1, y.1, "{}: not deterministic", x.0);
    }
    let src = scene16(32, 32, true, true, 10, 0);
    let refs: Vec<&[u16]> = vec![src.as_slice()];
    assert!(
        encode_vp9_lossy_sequence_with_hbd(&refs, 32, 32, 9, true, &Vp9GopConfig::new(100))
            .is_err()
    );
    assert!(encode_vp9_lossy_sequence_resized_hbd(&refs, &[(32, 32)], 8, true, 100).is_err());
    // Out-of-range samples for the declared depth reject.
    let hot = vec![4095u16; src.len()];
    assert!(encode_vp9_lossy_sequence_with_hbd(
        &[hot.as_slice()],
        32,
        32,
        10,
        true,
        &Vp9GopConfig::new(100)
    )
    .is_err());
}

fn ivf_wrap(packets: &[Vec<u8>], w: u16, h: u16) -> Vec<u8> {
    let mut ivf = Vec::new();
    ivf.extend_from_slice(b"DKIF");
    ivf.extend_from_slice(&0u16.to_le_bytes());
    ivf.extend_from_slice(&32u16.to_le_bytes());
    ivf.extend_from_slice(b"VP90");
    ivf.extend_from_slice(&w.to_le_bytes());
    ivf.extend_from_slice(&h.to_le_bytes());
    ivf.extend_from_slice(&25u32.to_le_bytes());
    ivf.extend_from_slice(&1u32.to_le_bytes());
    ivf.extend_from_slice(&(packets.len() as u32).to_le_bytes());
    ivf.extend_from_slice(&0u32.to_le_bytes());
    for (i, f) in packets.iter().enumerate() {
        ivf.extend_from_slice(&(f.len() as u32).to_le_bytes());
        ivf.extend_from_slice(&(i as u64).to_le_bytes());
        ivf.extend_from_slice(f);
    }
    ivf
}

/// Black-box validation dump under `OXIDEAV_VP9_FORMAT_DUMP_DIR`:
/// `<name>/input.ivf`, `<name>/crate-decode.yuv` (the crate's planar
/// packing, little-endian `u16` above 8 bits) and `<name>/pixfmt.txt`.
/// No-op unless the env var is set.
#[test]
fn dump_format_matrix_streams_when_requested() {
    let Some(dir) = std::env::var_os("OXIDEAV_VP9_FORMAT_DUMP_DIR") else {
        return;
    };
    for (name, packets, _src, bd, ssx, ssy, dims) in streams() {
        let refs: Vec<&[u8]> = packets.iter().map(Vec::as_slice).collect();
        let decoded = decode_vp9_sequence(&refs).expect("decodes");
        let mut yuv = Vec::new();
        for f in &decoded {
            yuv.extend_from_slice(&f.to_planar_bytes());
        }
        let sub = std::path::Path::new(&dir).join(&name);
        std::fs::create_dir_all(&sub).expect("create dump dir");
        std::fs::write(
            sub.join("input.ivf"),
            ivf_wrap(&packets, dims[0].0 as u16, dims[0].1 as u16),
        )
        .unwrap();
        std::fs::write(sub.join("crate-decode.yuv"), yuv).unwrap();
        let geom = match (ssx, ssy) {
            (true, true) => "420",
            (false, false) => "444",
            (true, false) => "422",
            (false, true) => "440",
        };
        let pix = if bd == 8 {
            format!("yuv{geom}p")
        } else {
            format!("yuv{geom}p{bd}le")
        };
        std::fs::write(sub.join("pixfmt.txt"), pix).unwrap();
    }
}
