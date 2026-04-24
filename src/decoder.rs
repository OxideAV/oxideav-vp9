//! VP9 decoder facade.
//!
//! Status: parses the §6.2 uncompressed header for every packet, populates
//! `CodecParameters` (width/height/pixel_format), and surfaces a clean
//! `Error::Unsupported` for the actual pixel-reconstruction step. This is
//! enough for higher layers to probe a VP9 stream, mux/remux it, and
//! enumerate frames.

use std::collections::VecDeque;

use oxideav_codec::Decoder;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, Packet, PixelFormat, Rational, Result, TimeBase,
    VideoFrame, VideoPlane,
};

use crate::block::IntraTile;
use crate::bool_decoder::BoolDecoder;
use crate::compressed_header::parse_compressed_header_with_seed;
use crate::dpb::{Dpb, RefFrame};
use crate::frame_ctx::FrameContext;
use crate::headers::{parse_uncompressed_header, ColorConfig, FrameType, UncompressedHeader};
use crate::inter::InterTile;

/// Build a `CodecParameters` from a parsed uncompressed header.
pub fn codec_parameters_from_header(h: &UncompressedHeader) -> CodecParameters {
    let mut params = CodecParameters::video(CodecId::new(crate::CODEC_ID_STR));
    params.width = Some(h.width);
    params.height = Some(h.height);
    params.pixel_format = Some(pixel_format_from_color_config(&h.color_config));
    params
}

/// Map VP9 color_config (subsampling + bit depth) to the closest oxideav
/// `PixelFormat`. We only have unsubsampled / 4:2:0 in core today, so
/// 4:2:2 / 4:4:4 / 10-bit / 12-bit fall back to `Yuv420P` until core
/// gains the missing variants.
pub fn pixel_format_from_color_config(cc: &ColorConfig) -> PixelFormat {
    // Only 8-bit 4:2:0 maps cleanly today.
    let _ = cc;
    PixelFormat::Yuv420P
}

/// Factory used by the codec registry.
pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    Ok(Box::new(Vp9Decoder::new(params.codec_id.clone())))
}

pub struct Vp9Decoder {
    codec_id: CodecId,
    /// Last-seen color_config — needed for non-key frames (§6.2.1).
    last_color_config: Option<ColorConfig>,
    /// Last-parsed header, kept for inspection.
    last_header: Option<UncompressedHeader>,
    /// Decoded frames waiting to be drained.
    ready_frames: VecDeque<VideoFrame>,
    /// PTS of the packet currently being ingested — attached to the
    /// next produced video frame.
    pending_pts: Option<i64>,
    /// Time base of the current stream (container-supplied).
    current_time_base: TimeBase,
    /// 8-slot reference picture buffer (§6.2).
    dpb: Dpb,
    /// Four saved frame-context slots (§8.10). Each frame picks its
    /// seed slot via `frame_context_idx` (0..=3) and, if
    /// `refresh_frame_context` was set, writes its post-compressed-header
    /// context back to the same slot at end-of-frame. Keyframes /
    /// intra_only / `reset_frame_context >= 2` reset all four slots to
    /// the §10.5 defaults.
    saved_ctx: [FrameContext; 4],
    eof: bool,
}

impl Vp9Decoder {
    pub fn new(codec_id: CodecId) -> Self {
        Self {
            codec_id,
            last_color_config: None,
            last_header: None,
            ready_frames: VecDeque::new(),
            pending_pts: None,
            current_time_base: TimeBase::new(1, 90_000),
            dpb: Dpb::new(),
            saved_ctx: [
                FrameContext::new_default(),
                FrameContext::new_default(),
                FrameContext::new_default(),
                FrameContext::new_default(),
            ],
            eof: false,
        }
    }

    pub fn last_header(&self) -> Option<&UncompressedHeader> {
        self.last_header.as_ref()
    }

    /// Parse one packet and update internal state. Returns `Ok(())` on
    /// successful decode. Keyframe / intra-only frames run the intra
    /// pipeline; non-key / non-intra-only frames run the inter pipeline
    /// against the persistent DPB.
    fn ingest(&mut self, packet: &Packet) -> Result<()> {
        self.pending_pts = packet.pts;
        self.current_time_base = packet.time_base;
        // VP9 Annex B: packets may carry a "superframe" trailer that
        // concatenates several frames into one access unit. Split them
        // and decode each sub-frame individually.
        let frames = split_superframe(&packet.data);
        for (i, frame_data) in frames.iter().enumerate() {
            // Only attach the packet's PTS to the first shown frame of a
            // superframe — the rest are typically altref (hidden).
            if i > 0 {
                self.pending_pts = None;
            } else {
                self.pending_pts = packet.pts;
            }
            self.ingest_one(frame_data)?;
        }
        Ok(())
    }

    fn ingest_one(&mut self, frame_data: &[u8]) -> Result<()> {
        let mut h = parse_uncompressed_header(frame_data, self.last_color_config)?;
        if h.show_existing_frame {
            self.last_header = Some(h);
            return Ok(());
        }
        if h.frame_type == FrameType::Key || h.intra_only {
            self.last_color_config = Some(h.color_config);
        }
        let is_intra = h.frame_type == FrameType::Key || h.intra_only;
        // Inter frames using `frame_size_with_refs`'s "found_ref" path
        // inherit their dimensions from the selected reference. The
        // parser returns 0×0 in that case — patch it up here.
        if !is_intra && (h.width == 0 || h.height == 0) {
            if let Some(r) = self.dpb.get(h.ref_frame_idx[0]) {
                h.width = r.width as u32;
                h.height = r.height as u32;
            } else {
                return Err(Error::invalid("vp9 §6.2.2.1: found_ref but slot is empty"));
            }
        }

        // Parse compressed header (§6.3). Seed the per-frame context
        // from either the §10.5 defaults (on keyframe / intra_only /
        // reset_frame_context >= 2) or from the saved slot indicated by
        // `frame_context_idx` (§8.10).
        let cmp_start = h.uncompressed_header_size;
        let cmp_end = cmp_start.saturating_add(h.header_size as usize);
        if cmp_end > frame_data.len() || h.header_size == 0 {
            return Err(Error::invalid("vp9 compressed header missing or truncated"));
        }
        let reset_all = is_intra || h.reset_frame_context >= 2;
        if reset_all {
            for slot in self.saved_ctx.iter_mut() {
                *slot = FrameContext::new_default();
            }
        }
        let slot_idx = (h.frame_context_idx as usize).min(3);
        let seed = if reset_all {
            FrameContext::new_default()
        } else {
            self.saved_ctx[slot_idx].clone()
        };
        let ch = parse_compressed_header_with_seed(&frame_data[cmp_start..cmp_end], &h, seed)?;
        // §8.10 reference_frame_update: if refresh_frame_context is set
        // and frame_parallel_decoding_mode is 0, save our post-§6.3
        // context back into the slot. (With fpdm=1 the spec technically
        // uses backward adaptation; we approximate by saving the
        // compressed-header-only state — still better than starting
        // every inter frame from defaults.)
        if h.refresh_frame_context {
            self.saved_ctx[slot_idx] = ch.ctx.clone();
        }
        let tile_payload = &frame_data[cmp_end..];
        if tile_payload.is_empty() {
            return Err(Error::invalid("vp9 tile payload empty"));
        }

        let tile_cols = 1u32 << h.tile_info.log2_tile_cols as u32;
        let tile_rows = 1u32 << h.tile_info.log2_tile_rows as u32;

        // Split tile_payload into `tile_rows * tile_cols` sub-slices. All
        // tiles except the last are prefixed with a 4-byte big-endian
        // length (§6.4 decode_tiles).
        let tile_slices = split_tile_payload(tile_payload, tile_cols, tile_rows)?;

        let (y, y_stride, u, v, uv_stride, uv_w, uv_h, seg_map) = if is_intra {
            let mut tile = IntraTile::new(&h, &ch);
            for tr in 0..tile_rows {
                for tc in 0..tile_cols {
                    let slice = tile_slices[(tr * tile_cols + tc) as usize];
                    let mut bd = BoolDecoder::new(slice)?;
                    let (col_s, col_e, row_s, row_e) = tile_pixel_bounds(&h, tc, tr);
                    tile.decode_rect(&mut bd, col_s, col_e, row_s, row_e)?;
                }
            }
            tile.finalize();
            let seg_map = Some(tile.segment_ids.clone());
            (
                tile.y,
                tile.y_stride,
                tile.u,
                tile.v,
                tile.uv_stride,
                tile.uv_w,
                tile.uv_h,
                seg_map,
            )
        } else {
            let refs = [
                self.dpb.get(h.ref_frame_idx[0]),
                self.dpb.get(h.ref_frame_idx[1]),
                self.dpb.get(h.ref_frame_idx[2]),
            ];
            let mut tile = InterTile::new(&h, &ch, h.width as usize, h.height as usize, refs);
            // §6.4.14 PrevSegmentIds: inherit from the primary reference
            // when the previous frame saved one. When absent (or sizes
            // mismatch), the InterTile keeps its all-zero default —
            // matches §8.2 setup_past_independence.
            if let Some(rf) = self.dpb.get(h.ref_frame_idx[0]) {
                tile.set_prev_segment_ids(rf.segment_ids.as_ref());
            }
            for tr in 0..tile_rows {
                for tc in 0..tile_cols {
                    let slice = tile_slices[(tr * tile_cols + tc) as usize];
                    let mut bd = BoolDecoder::new(slice)?;
                    let (col_s, col_e, row_s, row_e) = tile_pixel_bounds(&h, tc, tr);
                    tile.decode_rect(&mut bd, col_s, col_e, row_s, row_e)?;
                }
            }
            tile.finalize();
            // §8.1 step 3: save SegmentIds → PrevSegmentIds iff
            // update_map was set for the frame we just decoded.
            let seg_map = if h.segmentation.enabled && h.segmentation.update_map {
                Some(tile.segment_ids.clone())
            } else {
                // Carry the prior map forward so stream resumption works.
                Some(tile.prev_segment_ids.clone())
            };
            (
                tile.y,
                tile.y_stride,
                tile.u,
                tile.v,
                tile.uv_stride,
                tile.uv_w,
                tile.uv_h,
                seg_map,
            )
        };

        // Update DPB according to refresh_frame_flags.
        let sub_x = h.color_config.subsampling_x as u8;
        let sub_y = h.color_config.subsampling_y as u8;
        let rf = RefFrame {
            y: y.clone(),
            y_stride,
            u: u.clone(),
            v: v.clone(),
            uv_stride,
            width: h.width as usize,
            height: h.height as usize,
            uv_width: uv_w,
            uv_height: uv_h,
            subsampling_x: sub_x,
            subsampling_y: sub_y,
            segment_ids: seg_map,
        };
        self.dpb.refresh(h.refresh_frame_flags, &rf);

        if h.show_frame {
            let frame = VideoFrame {
                format: pixel_format_from_color_config(&h.color_config),
                width: h.width,
                height: h.height,
                pts: self.pending_pts.take(),
                time_base: self.current_time_base,
                planes: vec![
                    VideoPlane {
                        stride: y_stride,
                        data: y,
                    },
                    VideoPlane {
                        stride: uv_stride,
                        data: u,
                    },
                    VideoPlane {
                        stride: uv_stride,
                        data: v,
                    },
                ],
            };
            self.ready_frames.push_back(frame);
        }
        self.last_header = Some(h);
        Ok(())
    }
}

impl Decoder for Vp9Decoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        self.ingest(packet)
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        if let Some(f) = self.ready_frames.pop_front() {
            return Ok(Frame::Video(f));
        }
        if self.eof {
            return Err(Error::Eof);
        }
        Err(Error::NeedMore)
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.last_color_config = None;
        self.last_header = None;
        self.ready_frames.clear();
        self.pending_pts = None;
        self.dpb = Dpb::new();
        self.saved_ctx = [
            FrameContext::new_default(),
            FrameContext::new_default(),
            FrameContext::new_default(),
            FrameContext::new_default(),
        ];
        self.eof = false;
        Ok(())
    }
}

/// Split a VP9 Annex B superframe into one or more sub-frames. Most
/// packets are single-frame and this returns a one-element vec. When
/// the trailing marker byte's top 3 bits are `110`, the tail carries
/// `num_frames` size fields (each 1..=4 bytes) that slice the rest of
/// the payload into individual frames.
pub fn split_superframe(data: &[u8]) -> Vec<&[u8]> {
    if data.is_empty() {
        return vec![data];
    }
    let last = data[data.len() - 1];
    if last & 0xe0 != 0xc0 {
        return vec![data];
    }
    let bytes_per_size = (((last >> 3) & 0x3) + 1) as usize;
    let num_frames = ((last & 0x7) + 1) as usize;
    let index_bytes = 2 + num_frames * bytes_per_size;
    if data.len() < index_bytes {
        return vec![data];
    }
    let index_start = data.len() - index_bytes;
    // The byte immediately before the size fields must match the trailer.
    if data[index_start] != last {
        return vec![data];
    }
    let mut sizes = Vec::with_capacity(num_frames);
    let size_table_start = index_start + 1;
    for i in 0..num_frames {
        let off = size_table_start + i * bytes_per_size;
        let mut sz = 0usize;
        // Little-endian per the Annex B.
        for b in 0..bytes_per_size {
            sz |= (data[off + b] as usize) << (8 * b);
        }
        sizes.push(sz);
    }
    let mut out = Vec::with_capacity(num_frames);
    let mut cursor = 0usize;
    for sz in sizes {
        if cursor + sz > index_start {
            // Malformed — fall back to the whole packet as a single frame.
            return vec![data];
        }
        out.push(&data[cursor..cursor + sz]);
        cursor += sz;
    }
    out
}

/// Split the tile payload into `tile_rows * tile_cols` byte slices per
/// §6.4 decode_tiles: every tile except the last is prefixed with a
/// 4-byte big-endian length; the final tile consumes the remainder.
fn split_tile_payload<'a>(
    payload: &'a [u8],
    tile_cols: u32,
    tile_rows: u32,
) -> Result<Vec<&'a [u8]>> {
    let total = (tile_cols as usize) * (tile_rows as usize);
    let mut out: Vec<&'a [u8]> = Vec::with_capacity(total);
    let mut cursor = 0usize;
    for i in 0..total {
        let is_last = i + 1 == total;
        let (slice, next) = if is_last {
            (&payload[cursor..], payload.len())
        } else {
            if cursor + 4 > payload.len() {
                return Err(Error::invalid("vp9 multi-tile: truncated tile_size prefix"));
            }
            let sz = u32::from_be_bytes([
                payload[cursor],
                payload[cursor + 1],
                payload[cursor + 2],
                payload[cursor + 3],
            ]) as usize;
            let start = cursor + 4;
            let end = start + sz;
            if end > payload.len() {
                return Err(Error::invalid("vp9 multi-tile: tile_size overruns payload"));
            }
            (&payload[start..end], end)
        };
        out.push(slice);
        cursor = next;
    }
    Ok(out)
}

/// Pixel-space bounds `(col_start, col_end, row_start, row_end)` for
/// tile (`tile_col`, `tile_row`) per §6.4.1 `get_tile_offset`. The
/// returned bounds are clamped to the frame dimensions.
fn tile_pixel_bounds(
    h: &UncompressedHeader,
    tile_col: u32,
    tile_row: u32,
) -> (u32, u32, u32, u32) {
    let mi_cols = h.width.div_ceil(8);
    let mi_rows = h.height.div_ceil(8);
    let col_s = get_tile_offset(tile_col, mi_cols, h.tile_info.log2_tile_cols as u32) * 8;
    let col_e = get_tile_offset(tile_col + 1, mi_cols, h.tile_info.log2_tile_cols as u32) * 8;
    let row_s = get_tile_offset(tile_row, mi_rows, h.tile_info.log2_tile_rows as u32) * 8;
    let row_e = get_tile_offset(tile_row + 1, mi_rows, h.tile_info.log2_tile_rows as u32) * 8;
    (
        col_s.min(h.width),
        col_e.min(h.width),
        row_s.min(h.height),
        row_e.min(h.height),
    )
}

/// §6.4.1 get_tile_offset: offset = ((tileNum * sbs) >> tileSzLog2) << 3;
/// return min(offset, mis).
fn get_tile_offset(tile_num: u32, mis: u32, tile_sz_log2: u32) -> u32 {
    let sbs = mis.div_ceil(8);
    let offset = ((tile_num * sbs) >> tile_sz_log2) << 3;
    offset.min(mis)
}

/// Helper that returns frame_rate from container-supplied stream timing
/// when available — VP9 itself doesn't carry frame_rate in-band.
pub fn frame_rate_from_container(num: i64, den: i64) -> Option<Rational> {
    if num > 0 && den > 0 {
        Some(Rational::new(num, den))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::TimeBase;

    /// Build the same synthetic 64×64 key-frame header as the parser unit
    /// test, manually here so the decoder's test doesn't have to reach
    /// into the headers test module.
    fn synth_key_frame_header() -> Vec<u8> {
        let mut bw = BitWriter::new();
        bw.write(2, 2);
        bw.write(0, 1);
        bw.write(0, 1);
        bw.write(0, 1);
        bw.write(0, 1);
        bw.write(1, 1);
        bw.write(0, 1);
        bw.write(0x49, 8);
        bw.write(0x83, 8);
        bw.write(0x42, 8);
        bw.write(1, 3);
        bw.write(0, 1);
        bw.write(63, 16);
        bw.write(63, 16);
        bw.write(0, 1);
        bw.write(1, 1);
        bw.write(0, 1);
        bw.write(0, 2);
        bw.write(0, 6);
        bw.write(0, 3);
        bw.write(0, 1);
        bw.write(60, 8);
        bw.write(0, 1);
        bw.write(0, 1);
        bw.write(0, 1);
        bw.write(0, 1);
        bw.write(0, 1);
        bw.write(0, 16);
        bw.finish()
    }

    struct BitWriter {
        out: Vec<u8>,
        cur: u8,
        bits: u32,
    }

    impl BitWriter {
        fn new() -> Self {
            Self {
                out: Vec::new(),
                cur: 0,
                bits: 0,
            }
        }
        fn write(&mut self, value: u32, n: u32) {
            for i in (0..n).rev() {
                let b = ((value >> i) & 1) as u8;
                self.cur = (self.cur << 1) | b;
                self.bits += 1;
                if self.bits == 8 {
                    self.out.push(self.cur);
                    self.cur = 0;
                    self.bits = 0;
                }
            }
        }
        fn finish(mut self) -> Vec<u8> {
            if self.bits > 0 {
                self.cur <<= 8 - self.bits;
                self.out.push(self.cur);
            }
            self.out.extend_from_slice(&[0u8; 4]);
            self.out
        }
    }

    #[test]
    fn split_superframe_single_frame_passthrough() {
        let buf = [0x01u8, 0x02, 0x03, 0x04];
        let out = split_superframe(&buf);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0], &buf[..]);
    }

    #[test]
    fn split_superframe_two_frames() {
        // A manually crafted two-frame superframe: frame A (3 bytes),
        // frame B (5 bytes), then the size table (2 * 1-byte) and the
        // marker duplicated.
        //   bytes_per_size = 1, num_frames = 2
        //   marker = 0b110_00_001 = 0xc1 (nf-1=1, bps-1=0)
        let frame_a = [0x11u8, 0x22, 0x33];
        let frame_b = [0x44u8, 0x55, 0x66, 0x77, 0x88];
        let marker = 0xc1;
        let mut buf = Vec::new();
        buf.extend_from_slice(&frame_a);
        buf.extend_from_slice(&frame_b);
        buf.push(marker);
        buf.push(frame_a.len() as u8);
        buf.push(frame_b.len() as u8);
        buf.push(marker);
        let out = split_superframe(&buf);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0], &frame_a[..]);
        assert_eq!(out[1], &frame_b[..]);
    }

    /// Synthetic keyframe with an empty compressed header triggers the
    /// "compressed header missing" guard path and surfaces InvalidData.
    #[test]
    fn empty_compressed_header_surfaces_error() {
        let codec_id = CodecId::new(crate::CODEC_ID_STR);
        let params = CodecParameters::video(codec_id);
        let mut d = make_decoder(&params).unwrap();
        let buf = synth_key_frame_header();
        let pkt = Packet::new(0, TimeBase::new(1, 90_000), buf);
        // send_packet returns the ingest error directly now that
        // ingest runs the full pipeline.
        let r = d.send_packet(&pkt);
        match r {
            Err(Error::InvalidData(_)) | Err(Error::Unsupported(_)) => {}
            other => panic!("expected error, got {other:?}"),
        }
    }
}
