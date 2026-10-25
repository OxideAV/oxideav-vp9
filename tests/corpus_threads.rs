//! Whole-corpus thread-budget invariance: every staged fixture under
//! `docs/video/vp9/fixtures/` decodes through the framework
//! [`oxideav_vp9::Vp9Decoder`] to the same bytes at every thread budget
//! (serial, 2, 4, 8, 18 and `auto`), and — where the fixture stages a
//! full-sequence `expected.yuv` — to exactly that ground truth.
//! Docs-gated (standalone CI has no docs tree).

use oxideav_core::{Decoder, ExecutionContext, Frame, Packet, TimeBase};

fn ivf_packets(ivf: &[u8]) -> Vec<Packet> {
    assert!(ivf.len() >= 32 && &ivf[..4] == b"DKIF", "IVF signature");
    let hdr = u16::from_le_bytes([ivf[6], ivf[7]]) as usize;
    let mut out = Vec::new();
    let mut at = hdr;
    while at + 12 <= ivf.len() {
        let size = u32::from_le_bytes([ivf[at], ivf[at + 1], ivf[at + 2], ivf[at + 3]]) as usize;
        at += 12;
        if at + size > ivf.len() {
            break;
        }
        out.push(Packet::new(
            0,
            TimeBase::MILLIS,
            ivf[at..at + size].to_vec(),
        ));
        at += size;
    }
    out
}

/// Concatenated planar bytes of every shown frame.
fn decode_bytes(packets: &[Packet], exec: &ExecutionContext) -> Vec<u8> {
    let mut dec = oxideav_vp9::Vp9Decoder::new();
    dec.set_execution_context(exec);
    let mut out = Vec::new();
    let mut drain = |dec: &mut oxideav_vp9::Vp9Decoder| {
        while let Ok(Frame::Video(v)) = dec.receive_frame() {
            for plane in &v.planes {
                out.extend_from_slice(&plane.data);
            }
        }
    };
    for p in packets {
        dec.send_packet(p).expect("decode");
        drain(&mut dec);
    }
    dec.flush().expect("flush");
    drain(&mut dec);
    out
}

#[test]
fn corpus_is_thread_budget_invariant_and_matches_ground_truth() {
    let root = std::path::Path::new("../../docs/video/vp9/fixtures");
    if !root.is_dir() {
        eprintln!("docs corpus not present; docs-gated");
        return;
    }
    let budgets = [
        ExecutionContext::with_threads(2),
        ExecutionContext::with_threads(4),
        ExecutionContext::with_threads(8),
        ExecutionContext::with_threads(18),
        ExecutionContext::auto(),
    ];
    let mut swept = 0usize;
    let mut exact = 0usize;
    let mut dirs: Vec<_> = std::fs::read_dir(root)
        .expect("fixtures dir")
        .map(|e| e.expect("entry").path())
        .collect();
    dirs.sort();
    for dir in dirs {
        let ivf_path = dir.join("input.ivf");
        if !ivf_path.is_file() {
            continue;
        }
        let name = dir.file_name().unwrap().to_string_lossy().into_owned();
        let packets = ivf_packets(&std::fs::read(&ivf_path).expect("input.ivf"));
        let serial = decode_bytes(&packets, &ExecutionContext::serial());
        for exec in &budgets {
            let got = decode_bytes(&packets, exec);
            assert!(
                got == serial,
                "{name}: output differs from serial at {} threads",
                exec.threads
            );
        }
        if let Ok(expected) = std::fs::read(dir.join("expected.yuv"))
            && expected.len() == serial.len()
        {
            assert!(expected == serial, "{name}: differs from expected.yuv");
            exact += 1;
        }
        swept += 1;
    }
    eprintln!("swept {swept} fixtures, {exact} matched a full-sequence expected.yuv");
    assert!(swept >= 40, "corpus sweep covered {swept} fixtures");
    assert!(exact >= 40, "only {exact} fixtures matched expected.yuv");
}
