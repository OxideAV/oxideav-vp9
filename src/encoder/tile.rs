//! VP9 keyframe tile / partition / block emitter — mirror of §6.4.
//!
//! Emits the tile payload that the decoder's `IntraTile::decode` walks.
//! Scope (MVP): single tile, full frame, one `PARTITION_NONE` leaf per
//! 64×64 superblock (or edge-clipped smaller block at the bottom/right
//! of non-multiple-of-64 frames), `skip=1` (no residual), `DC_PRED`
//! luma + chroma intra mode.
//!
//! `skip=1` means no coefficients are encoded — the reconstructed
//! output is just the intra predictor. With `DC_PRED` and no decoded
//! neighbours, every sample is mid-grey (128). This yields a valid
//! VP9 keyframe that ffmpeg accepts and this crate's decoder
//! round-trips.

use crate::compressed_header::TxMode;
use crate::encoder::bool_encoder::BoolEncoder;
use crate::encoder::params::EncoderParams;
use crate::probs::KF_PARTITION_PROBS;
use crate::tables::{KF_UV_MODE_PROBS, KF_Y_MODE_PROBS};

/// Index of `DC_PRED` in the intra-mode tree (§7.4.5 Table 7-5).
const MODE_DC: usize = 0;
/// Skip probability used by the decoder (hard-coded context-0 approx).
const SKIP_PROB: u8 = 192;

/// Emit the tile payload for a keyframe at the given frame size using
/// the MVP scheme: every block is 64×64 PARTITION_NONE, skip=1,
/// DC_PRED luma + chroma.
pub fn emit_keyframe_tile(p: &EncoderParams, tx_mode: TxMode) -> Vec<u8> {
    let mut be = BoolEncoder::new();
    let mi_cols = (p.width as usize).div_ceil(8);
    let mi_rows = (p.height as usize).div_ceil(8);
    let mut ctx = PartitionCtx {
        above: vec![0u8; mi_cols],
        left: vec![0u8; mi_rows],
    };
    let sb_cols = p.width.div_ceil(64);
    let sb_rows = p.height.div_ceil(64);
    for sby in 0..sb_rows {
        for sbx in 0..sb_cols {
            let col = sbx * 64;
            let row = sby * 64;
            emit_partition(&mut be, &mut ctx, row, col, 64, p.width, p.height, tx_mode);
        }
    }
    be.finish()
}

/// Mirror of the decoder's §7.4.6 partition context so the bits we emit
/// resolve to the same context bin on the decode side.
struct PartitionCtx {
    above: Vec<u8>,
    left: Vec<u8>,
}

impl PartitionCtx {
    fn lookup(&self, bsize: u32, mi_row: usize, mi_col: usize) -> [u8; 3] {
        let bsl = match bsize {
            8 => 0usize,
            16 => 1,
            32 => 2,
            64 => 3,
            _ => 3,
        };
        let num8x8 = (bsize as usize) / 8;
        let boffset = 3 - bsl;
        let mut above = 0u8;
        let mut left = 0u8;
        for i in 0..num8x8 {
            let c = mi_col + i;
            if c < self.above.len() {
                above |= self.above[c];
            }
            let r = mi_row + i;
            if r < self.left.len() {
                left |= self.left[r];
            }
        }
        let above_bit = ((above >> boffset) & 1) as usize;
        let left_bit = ((left >> boffset) & 1) as usize;
        // Mirror `block::read_partition` exactly: tbl_bsl = 3 - bsl so
        // bsize=64 maps to row 0..3 (KF_PARTITION_PROBS is 64×64-first
        // in the decoder's internal layout).
        let tbl_bsl = 3 - bsl;
        let ctx = tbl_bsl * 4 + left_bit * 2 + above_bit;
        KF_PARTITION_PROBS[ctx]
    }

    fn update(&mut self, bsize_px: u32, sub_w: u32, sub_h: u32, mi_row: usize, mi_col: usize) {
        let num8x8 = (bsize_px as usize) / 8;
        let bsl = match bsize_px {
            8 => 0usize,
            16 => 1,
            32 => 2,
            64 => 3,
            _ => 3,
        };
        let boffset = 3 - bsl;
        let above_fill = if sub_w >= bsize_px {
            (1u8 << boffset) - 1 + (1u8 << boffset)
        } else {
            0
        };
        let left_fill = if sub_h >= bsize_px {
            (1u8 << boffset) - 1 + (1u8 << boffset)
        } else {
            0
        };
        for i in 0..num8x8 {
            let c = mi_col + i;
            if c < self.above.len() {
                self.above[c] = above_fill;
            }
            let r = mi_row + i;
            if r < self.left.len() {
                self.left[r] = left_fill;
            }
        }
    }
}

fn emit_partition(
    be: &mut BoolEncoder,
    ctx: &mut PartitionCtx,
    row: u32,
    col: u32,
    bsize: u32,
    frame_w: u32,
    frame_h: u32,
    tx_mode: TxMode,
) {
    debug_assert!(matches!(bsize, 64 | 32 | 16 | 8));
    if row >= frame_h || col >= frame_w {
        return;
    }
    let on_right = col + bsize > frame_w;
    let on_bottom = row + bsize > frame_h;
    let mi_row = (row as usize) / 8;
    let mi_col = (col as usize) / 8;
    let probs = ctx.lookup(bsize, mi_row, mi_col);
    let half = bsize / 2;

    if on_right && on_bottom {
        // §6.4.2 last paragraph: SPLIT forced, no bit read.
        if bsize == 8 {
            // 8×8 can't split per MVP path. Fall back to emitting the
            // block as NONE (this only happens when both width and
            // height are multiples of 8, so bsize==8 edge-clip is
            // actually an exact-fit).
            emit_block(be, bsize, bsize, false);
            ctx.update(bsize, bsize, bsize, mi_row, mi_col);
            return;
        }
        emit_partition(be, ctx, row, col, half, frame_w, frame_h, tx_mode);
        emit_partition(be, ctx, row, col + half, half, frame_w, frame_h, tx_mode);
        emit_partition(be, ctx, row + half, col, half, frame_w, frame_h, tx_mode);
        emit_partition(
            be,
            ctx,
            row + half,
            col + half,
            half,
            frame_w,
            frame_h,
            tx_mode,
        );
        return;
    }
    if on_right {
        // Only VERT or SPLIT readable — one bit with probs[2].
        // We choose SPLIT (bit=1) to simplify edge handling: each
        // half recurses.
        be.write(1, probs[2]);
        if bsize == 8 {
            emit_block(be, bsize, bsize, false);
            ctx.update(bsize, bsize, bsize, mi_row, mi_col);
            return;
        }
        emit_partition(be, ctx, row, col, half, frame_w, frame_h, tx_mode);
        emit_partition(be, ctx, row, col + half, half, frame_w, frame_h, tx_mode);
        emit_partition(be, ctx, row + half, col, half, frame_w, frame_h, tx_mode);
        emit_partition(
            be,
            ctx,
            row + half,
            col + half,
            half,
            frame_w,
            frame_h,
            tx_mode,
        );
        return;
    }
    if on_bottom {
        // Only HORZ or SPLIT readable — one bit with probs[1].
        be.write(1, probs[1]);
        if bsize == 8 {
            emit_block(be, bsize, bsize, false);
            ctx.update(bsize, bsize, bsize, mi_row, mi_col);
            return;
        }
        emit_partition(be, ctx, row, col, half, frame_w, frame_h, tx_mode);
        emit_partition(be, ctx, row, col + half, half, frame_w, frame_h, tx_mode);
        emit_partition(be, ctx, row + half, col, half, frame_w, frame_h, tx_mode);
        emit_partition(
            be,
            ctx,
            row + half,
            col + half,
            half,
            frame_w,
            frame_h,
            tx_mode,
        );
        return;
    }
    // Interior — emit PARTITION_NONE. Tree reads: bit0=0 -> NONE.
    if bsize == 8 {
        // 8×8 branch reads bit0 -> bit1 -> bit2 (no SPLIT). NONE is
        // still bit0=0.
        be.write(0, probs[0]);
    } else {
        be.write(0, probs[0]);
    }
    emit_block(be, bsize, bsize, false /* is_8x8_branch */);
    ctx.update(bsize, bsize, bsize, mi_row, mi_col);
    let _ = tx_mode;
}

/// Emit one block's symbols: skip=1, DC_PRED luma, DC_PRED chroma,
/// no coefficients.
fn emit_block(be: &mut BoolEncoder, _w: u32, _h: u32, _edge_split_descent: bool) {
    // Skip bit.
    be.write(1, SKIP_PROB);
    // tx_size: ONLY_4X4 so no symbol is written.
    // Luma intra mode tree (KF_Y_MODE_PROBS[DC][DC] = row 0, col 0).
    let p = &KF_Y_MODE_PROBS[MODE_DC][MODE_DC];
    emit_intra_mode_tree(be, p, MODE_DC);
    // UV intra mode tree keyed by luma mode DC.
    let puv = &KF_UV_MODE_PROBS[MODE_DC];
    emit_intra_mode_tree(be, puv, MODE_DC);
}

/// Emit an intra mode symbol `m` against a 9-prob tree. Mirrors
/// `block::read_intra_mode_tree`.
fn emit_intra_mode_tree(be: &mut BoolEncoder, p: &[u8; 9], m: usize) {
    // Tree shape (libvpx `vp9_intra_mode_tree`):
    //   -DC,   2, -TM, 4, -V, 6, 8, 12, -H, 10, -D135, -D117, -D45, 14, -D63, 16, -D153, -D207
    // Walk the tree to `m` and emit the branch bits.
    match m {
        0 => be.write(0, p[0]), // DC
        9 => {
            be.write(1, p[0]);
            be.write(0, p[1]);
        } // TM
        1 => {
            be.write(1, p[0]);
            be.write(1, p[1]);
            be.write(0, p[2]);
        } // V
        2 => {
            // H
            be.write(1, p[0]);
            be.write(1, p[1]);
            be.write(1, p[2]);
            be.write(0, p[3]);
            be.write(0, p[4]);
        }
        4 => {
            // D135
            be.write(1, p[0]);
            be.write(1, p[1]);
            be.write(1, p[2]);
            be.write(0, p[3]);
            be.write(1, p[4]);
            be.write(0, p[5]);
        }
        5 => {
            // D117
            be.write(1, p[0]);
            be.write(1, p[1]);
            be.write(1, p[2]);
            be.write(0, p[3]);
            be.write(1, p[4]);
            be.write(1, p[5]);
        }
        3 => {
            // D45
            be.write(1, p[0]);
            be.write(1, p[1]);
            be.write(1, p[2]);
            be.write(1, p[3]);
            be.write(0, p[6]);
        }
        8 => {
            // D63
            be.write(1, p[0]);
            be.write(1, p[1]);
            be.write(1, p[2]);
            be.write(1, p[3]);
            be.write(1, p[6]);
            be.write(0, p[7]);
        }
        6 => {
            // D153
            be.write(1, p[0]);
            be.write(1, p[1]);
            be.write(1, p[2]);
            be.write(1, p[3]);
            be.write(1, p[6]);
            be.write(1, p[7]);
            be.write(0, p[8]);
        }
        7 => {
            // D207
            be.write(1, p[0]);
            be.write(1, p[1]);
            be.write(1, p[2]);
            be.write(1, p[3]);
            be.write(1, p[6]);
            be.write(1, p[7]);
            be.write(1, p[8]);
        }
        _ => unreachable!("invalid intra mode index {m}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::compressed_header::emit_compressed_header;
    use crate::encoder::uncompressed_header::emit_uncompressed_header;

    /// Assemble a full keyframe and decode it with our own decoder.
    fn assemble_and_decode(width: u32, height: u32) -> Vec<u8> {
        let p = EncoderParams::keyframe(width, height);
        let ch = emit_compressed_header(TxMode::Only4x4, false);
        let tile = emit_keyframe_tile(&p, TxMode::Only4x4);
        let uh = emit_uncompressed_header(&p, ch.len() as u16);
        let mut out = uh;
        out.extend_from_slice(&ch);
        out.extend_from_slice(&tile);
        out
    }

    #[test]
    fn self_roundtrip_64x64_dc_midgrey() {
        use crate::block::IntraTile;
        use crate::bool_decoder::BoolDecoder;
        use crate::compressed_header::parse_compressed_header;
        use crate::headers::parse_uncompressed_header;

        let frame = assemble_and_decode(64, 64);
        let h = parse_uncompressed_header(&frame, None).unwrap();
        let cmp_start = h.uncompressed_header_size;
        let cmp_end = cmp_start + h.header_size as usize;
        let ch = parse_compressed_header(&frame[cmp_start..cmp_end], &h).unwrap();
        let mut tile = IntraTile::new(&h, &ch);
        let mut bd = BoolDecoder::new(&frame[cmp_end..]).unwrap();
        tile.decode(&mut bd).unwrap();
        // All samples should be 128 (DC_PRED midgrey for the first
        // block at the top-left where there are no neighbours). Later
        // blocks have neighbours (all 128), so still 128.
        assert_eq!(tile.y.len(), 64 * 64);
        for &v in &tile.y {
            assert_eq!(v, 128);
        }
        for &v in &tile.u {
            assert_eq!(v, 128);
        }
        for &v in &tile.v {
            assert_eq!(v, 128);
        }
    }

    #[test]
    fn debug_symbols_128() {
        // Manually check what symbols we write for a 2-SB stream using
        // the same probs the 64×64 encoder uses.
        let mut be = BoolEncoder::new();
        // bsize=64 -> bsl=3 -> tbl_bsl=0 -> KF_PARTITION_PROBS[0]=158.
        let p_part = KF_PARTITION_PROBS[0][0];
        for _ in 0..2 {
            be.write(0, p_part);
            be.write(1, SKIP_PROB);
            be.write(0, KF_Y_MODE_PROBS[0][0][0]);
            be.write(0, KF_UV_MODE_PROBS[0][0]);
        }
        let buf = be.finish();

        use crate::bool_decoder::BoolDecoder;
        let mut bd = BoolDecoder::new(&buf).unwrap();
        for i in 0..2 {
            let part = bd.read(p_part).unwrap();
            assert_eq!(part, 0, "partition sb{i}");
            let skip = bd.read(SKIP_PROB).unwrap();
            assert_eq!(skip, 1, "skip sb{i}");
            let y_first_bit = bd.read(KF_Y_MODE_PROBS[0][0][0]).unwrap();
            assert_eq!(y_first_bit, 0, "y mode sb{i}");
            let uv_first_bit = bd.read(KF_UV_MODE_PROBS[0][0]).unwrap();
            assert_eq!(uv_first_bit, 0, "uv mode sb{i}");
        }
    }

    #[test]
    fn self_roundtrip_through_vp9_decoder() {
        // Exercise the top-level `Vp9Decoder` facade so we know the
        // bitstream survives ingest / receive_frame.
        use crate::decoder::make_decoder;
        use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase};

        let frame = assemble_and_decode(64, 64);
        let codec_id = CodecId::new(crate::CODEC_ID_STR);
        let params = CodecParameters::video(codec_id);
        let mut d = make_decoder(&params).unwrap();
        let pkt = Packet::new(0, TimeBase::new(1, 30), frame);
        d.send_packet(&pkt).unwrap();
        let f = d.receive_frame().unwrap();
        match f {
            Frame::Video(v) => {
                assert_eq!(v.width, 64);
                assert_eq!(v.height, 64);
                // Luma plane should be all-128 (DC-chain midgrey).
                let luma = &v.planes[0].data;
                for &s in luma {
                    assert_eq!(s, 128, "all luma samples should be 128");
                }
            }
            _ => panic!("expected Video frame"),
        }
    }

    #[test]
    fn self_roundtrip_128x128() {
        use crate::block::IntraTile;
        use crate::bool_decoder::BoolDecoder;
        use crate::compressed_header::parse_compressed_header;
        use crate::headers::parse_uncompressed_header;

        let frame = assemble_and_decode(128, 128);
        let h = parse_uncompressed_header(&frame, None).unwrap();
        let cmp_start = h.uncompressed_header_size;
        let cmp_end = cmp_start + h.header_size as usize;
        let ch = parse_compressed_header(&frame[cmp_start..cmp_end], &h).unwrap();
        let mut tile = IntraTile::new(&h, &ch);
        let mut bd = BoolDecoder::new(&frame[cmp_end..]).unwrap();
        tile.decode(&mut bd).unwrap();
        // All samples should be exactly 128 — DC_PRED neighbour chain
        // propagates 128 everywhere.
        for &v in &tile.y {
            assert_eq!(v, 128);
        }
        for &v in &tile.u {
            assert_eq!(v, 128);
        }
        for &v in &tile.v {
            assert_eq!(v, 128);
        }
    }
}
