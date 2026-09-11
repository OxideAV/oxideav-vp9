//! Round-458 encoder benchmarks (Criterion): the default structured
//! GOP entry on a small moving scene, the plain adaptive chain, and a
//! lossy keyframe — the profiling targets of the encoder's
//! byte-identical speedups. Run with `cargo bench --bench encode`.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use oxideav_vp9::{
    encode_vp9_lossy, encode_vp9_lossy_sequence, encode_vp9_lossy_sequence_with, Vp9GopConfig,
};

/// Translating textured 4:2:0 content (the rate-control tests' scene).
fn scene(w: usize, h: usize, k: usize) -> Vec<u8> {
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    let mut px = Vec::with_capacity(w * h + 2 * cw * ch);
    for y in 0..h {
        for x in 0..w {
            let mut v = ((x + 2 * k) * 3 + y * 2) % 200 + 20;
            let px_x = x as i64 - 2 * k as i64;
            if (8..24).contains(&px_x) && (8..24).contains(&y) {
                v = (px_x as usize * 37 + y * 53) % 255;
            }
            if (x * 7 + y * 11 + k) % 29 == 0 {
                v = (v + 40) % 256;
            }
            px.push(v as u8);
        }
    }
    for plane in 0..2usize {
        for y in 0..ch {
            for x in 0..cw {
                px.push(((x + k) * 5 + y * 3 + plane * 90) as u8);
            }
        }
    }
    px
}

fn bench_encoder(c: &mut Criterion) {
    let (w, h, n) = (96usize, 64usize, 6usize);
    let frames: Vec<Vec<u8>> = (0..n).map(|k| scene(w, h, k)).collect();
    let refs: Vec<&[u8]> = frames.iter().map(Vec::as_slice).collect();

    let mut g = c.benchmark_group("encode");
    g.sample_size(10);
    g.throughput(Throughput::Elements(n as u64));
    let mut cfg = Vp9GopConfig::new(110);
    cfg.altref_interval = 3;
    g.bench_function("structured_gop_96x64_6f", |b| {
        b.iter(|| {
            encode_vp9_lossy_sequence_with(black_box(&refs), w as u32, h as u32, &cfg).unwrap()
        })
    });
    g.bench_function("chain_96x64_6f", |b| {
        b.iter(|| encode_vp9_lossy_sequence(black_box(&refs), w as u32, h as u32, 110).unwrap())
    });
    g.finish();

    let kf = scene(256, 160, 0);
    let mut k = c.benchmark_group("keyframe");
    k.sample_size(10);
    k.throughput(Throughput::Elements(1));
    k.bench_function("lossy_256x160", |b| {
        b.iter(|| encode_vp9_lossy(black_box(&kf), 256, 160, 110).unwrap())
    });
    k.finish();
}

criterion_group!(benches, bench_encoder);
criterion_main!(benches);
