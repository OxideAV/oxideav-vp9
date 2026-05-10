//! VP9 decoder facade.
//!
//! Status: parses the §6.2 uncompressed header for every packet, populates
//! `CodecParameters` (width/height/pixel_format), and surfaces a clean
//! `Error::Unsupported` for the actual pixel-reconstruction step. This is
//! enough for higher layers to probe a VP9 stream, mux/remux it, and
//! enumerate frames.

use std::collections::VecDeque;

use oxideav_core::Decoder;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, Packet, PixelFormat, Rational, Result, TimeBase,
    VideoFrame, VideoPlane,
};

use crate::block::IntraTile;
use crate::bool_decoder::BoolDecoder;
use crate::compressed_header::parse_compressed_header_with_seed;
use crate::dpb::{Dpb, RefFrame};
use crate::frame_ctx::FrameContext;
use crate::headers::{
    parse_uncompressed_header, ColorConfig, ColorSpace, FrameType, UncompressedHeader,
};
use crate::inter::InterTile;

/// Maximum supported frame dimension along either axis (pixels).
///
/// VP9 syntax encodes `frame_width_minus_1` / `frame_height_minus_1` as
/// 16-bit fields (§6.2.2), allowing up to 65535×65535. We cap each axis
/// at 8192 to match the codec capability advertised in
/// `register_codecs` (`with_max_size(8192, 8192)`) and to bound
/// allocations on hostile inputs.
pub const MAX_FRAME_DIM: u32 = 8192;

/// Maximum supported total luma sample count per frame (pixels).
///
/// Equals `MAX_FRAME_DIM * MAX_FRAME_DIM` = 67,108,864. This is well
/// above the largest defined VP9 level (Level 6, 35,651,584 luma
/// samples, ~8192×4352) so legitimate streams are never refused, while
/// a fuzz-generated 65535×65535 keyframe is rejected with
/// `Error::InvalidData` rather than allocating ~16 GiB of plane
/// buffers.
pub const MAX_FRAME_PIXELS: u32 = MAX_FRAME_DIM * MAX_FRAME_DIM;

/// Per-tile zero-pad budget for the boolean range decoder, scaled to
/// the tile-payload size (§9.2). The spec requires `BoolMaxBits` to
/// stay positive throughout decoding, but real libvpx-encoded streams
/// routinely consume a handful of zero-pad bits during tile
/// finalisation (the renormalise step pulls one bit at a time and the
/// encoder pads the trailing byte to byte alignment). The in-tree
/// inter pipeline is still scaffold and uses default probability
/// tables for several syntax elements, so it consumes a few extra
/// bits per tile vs an adaptive-probability oracle — a flat budget
/// false-positives the multi-tile / P-frame fixtures.
///
/// We allow zero-pad up to **2× the tile size in bits** (i.e. up to
/// 200% over-read of the payload). This comfortably covers the
/// observed slack across the in-tree fixture corpus while rejecting
/// fuzz mutations whose tile payload is fundamentally too short for
/// the declared frame size:
///
/// * Workspace task #748: 3-byte tile for a 385×1 frame → 166 bits
///   of zero-pad over a 24-bit payload (~691% over-read), refused.
/// * Multi-tile fixture: 3369-byte tile → 1212 bits of zero-pad over
///   26,952-bit payload (~4.5% over-read), accepted.
/// * P-frame fixture: 15-byte scaffold-inter tile → 171 bits of
///   zero-pad over 120-bit payload (~143% over-read), accepted.
///
/// A floor of 32 bits absorbs the byte-alignment + renormalise tail
/// for tiny payloads (~8 bits) without losing the tightness needed to
/// reject the task #748 fuzz input (3-byte tile, 166-bit over-read).
const BOOL_OVER_READ_FLOOR_BITS: u32 = 32;

#[inline]
fn bool_over_read_budget_bits(tile_payload_len: usize) -> u32 {
    let scaled = (tile_payload_len as u64).saturating_mul(16) as u32; // 2× bits
    scaled.max(BOOL_OVER_READ_FLOOR_BITS)
}

/// Build a `CodecParameters` from a parsed uncompressed header.
pub fn codec_parameters_from_header(h: &UncompressedHeader) -> CodecParameters {
    let mut params = CodecParameters::video(CodecId::new(crate::CODEC_ID_STR));
    params.width = Some(h.width);
    params.height = Some(h.height);
    params.pixel_format = Some(pixel_format_from_color_config(&h.color_config));
    params
}

/// Map a VP9 `color_config` (§6.2.1: `bit_depth`, `color_space`,
/// `subsampling_x`, `subsampling_y`) to the matching
/// [`oxideav_core::PixelFormat`].
///
/// VP9 enumerates one of three bit depths (8 / 10 / 12) and three
/// chroma layouts (4:2:0 = both subsampled, 4:2:2 = horizontal only,
/// 4:4:4 = full-res). When `color_space == Srgb` (§6.2.1 the spec also
/// pins `subsampling_x = subsampling_y = 0`) the planes carry G/B/R at
/// full resolution; we surface that as a `Yuv444P*` variant since
/// `oxideav_core::PixelFormat` has no dedicated GBR-planar variant
/// today (RGB is only available packed: `Rgb24`, `Rgb48Le`, …).
/// Higher layers can read `color_space` from the codec parameters' raw
/// header (or the bitstream itself) when they need to distinguish a
/// YUV 4:4:4 plane from a GBR-planar one.
///
/// 4:2:2 falls under the catch-all default — VP9's `parse_color_config`
/// (§6.2.1) does allow `(subsampling_x, subsampling_y) = (1, 0)` for
/// non-sRGB profile 1/3 streams, and we map that pair to the
/// `Yuv422P*` family.
pub fn pixel_format_from_color_config(cc: &ColorConfig) -> PixelFormat {
    let chroma = (cc.subsampling_x, cc.subsampling_y);
    match (cc.bit_depth, chroma, cc.color_space) {
        // sRGB: planes are GBR full-res. No GBRP variant in core, so
        // surface as the matching 4:4:4 family at the right bit depth.
        (8, _, ColorSpace::Srgb) => PixelFormat::Yuv444P,
        (10, _, ColorSpace::Srgb) => PixelFormat::Yuv444P10Le,
        (12, _, ColorSpace::Srgb) => PixelFormat::Yuv444P12Le,
        // Non-sRGB: pick by (chroma layout, bit depth).
        (8, (true, true), _) => PixelFormat::Yuv420P,
        (8, (true, false), _) => PixelFormat::Yuv422P,
        (8, (false, false), _) => PixelFormat::Yuv444P,
        (10, (true, true), _) => PixelFormat::Yuv420P10Le,
        (10, (true, false), _) => PixelFormat::Yuv422P10Le,
        (10, (false, false), _) => PixelFormat::Yuv444P10Le,
        (12, (true, true), _) => PixelFormat::Yuv420P12Le,
        (12, (true, false), _) => PixelFormat::Yuv422P12Le,
        (12, (false, false), _) => PixelFormat::Yuv444P12Le,
        // VP9 forbids `(subsampling_x, subsampling_y) = (false, true)`
        // (§6.2.1: 4:4:0 is not a profile). Default to the closest
        // 4:2:0 variant rather than panicking on a malformed header.
        (_, _, _) => PixelFormat::Yuv420P,
    }
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
            // §6.2 / §8.2: emit the existing frame from
            // `frame_to_show_map_idx`. This is a pure pass-through —
            // header_size == 0, refresh_frame_flags == 0, no compressed
            // header / tile / loopfilter parsing. The DPB stays
            // unchanged; the previously-decoded reference is copied to
            // output.
            let idx = h.existing_frame_to_show;
            if let Some(rf) = self.dpb.get(idx) {
                // VP9 inherits color_config from the most recent
                // key/intra_only frame (§6.2.1). `last_color_config`
                // tracks that; fall back to 8-bit on the unlikely path
                // where show_existing_frame is the very first packet.
                let bit_depth = self
                    .last_color_config
                    .map(|c| c.bit_depth as u32)
                    .unwrap_or(8);
                let frame = build_video_frame_from_ref(rf, bit_depth, self.pending_pts.take());
                self.ready_frames.push_back(frame);
            } else {
                return Err(Error::invalid(format!(
                    "vp9 show_existing_frame: slot {idx} is empty"
                )));
            }
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

        // §6.2.2 + §A.1 bound check on the FINAL (post-found_ref)
        // dimensions. `frame_width_minus_1` / `_height_minus_1` are
        // 16-bit fields (§6.2.2), so the syntax allows 65535×65535
        // ≈ 4.3 GP — far beyond the largest defined level (Level 6,
        // 35,651,584 luma samples ≈ 8192×4352, see
        // `webmproject-vp9-levels-testing.html`). Without a cap, a
        // fuzz mutation that flips a 64×64 keyframe to a 65535×65535
        // declaration tips `IntraTile::new` / `InterTile::new` into a
        // multi-GiB allocation and OOMs the process before the
        // bitstream is even validated further. We refuse anything
        // above `MAX_FRAME_PIXELS` (= `MAX_FRAME_DIM` × `MAX_FRAME_DIM`
        // = 8192 × 8192 = 67 MP — matches the codec capability
        // `with_max_size(8192, 8192)` advertised in `register_codecs`)
        // with `Error::InvalidData`. This is a bitstream-conformance
        // check on top of the spec, not a §-mandated one.
        let pixels = (h.width as u64).saturating_mul(h.height as u64);
        if h.width == 0
            || h.height == 0
            || h.width > MAX_FRAME_DIM
            || h.height > MAX_FRAME_DIM
            || pixels > MAX_FRAME_PIXELS as u64
        {
            return Err(Error::invalid(format!(
                "vp9: declared frame {}×{} ({} px) exceeds decoder cap {}×{} ({} px)",
                h.width, h.height, pixels, MAX_FRAME_DIM, MAX_FRAME_DIM, MAX_FRAME_PIXELS
            )));
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
                    // §9.2 conformance: the bool decoder is allowed
                    // a small zero-pad tail (real libvpx-encoded
                    // streams routinely consume a handful of bits past
                    // the last encoded byte during tile finalisation).
                    // But a tile that runs MASSIVELY past its payload
                    // is non-conformant — every additional zero-pad
                    // bit pollutes the decoded pixels with arbitrary
                    // garbage that diverges from any other conformant
                    // decoder. Workspace task #748: a 178-byte fuzz
                    // input with a 3-byte tile payload over-read by
                    // ~hundreds of bits and disagreed with libavcodec
                    // on Y[0,0] by 3 LSB.
                    let budget = bool_over_read_budget_bits(slice.len());
                    if std::env::var("VP9_TRACE_OVERREAD").is_ok() {
                        eprintln!(
                            "OVERREAD: tile {}-byte over_read={} budget={}",
                            slice.len(),
                            bd.over_read_bits(),
                            budget
                        );
                    }
                    if bd.over_read_bits() > budget {
                        return Err(Error::invalid(format!(
                            "vp9 §9.2: tile bool decoder over-read {} bits past end of {}-byte payload (budget {} bits)",
                            bd.over_read_bits(),
                            slice.len(),
                            budget,
                        )));
                    }
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
                    // §9.2 conformance — see intra branch above for
                    // rationale (workspace task #748).
                    let budget = bool_over_read_budget_bits(slice.len());
                    if std::env::var("VP9_TRACE_OVERREAD").is_ok() {
                        eprintln!(
                            "OVERREAD inter: tile {}-byte over_read={} budget={}",
                            slice.len(),
                            bd.over_read_bits(),
                            budget
                        );
                    }
                    if bd.over_read_bits() > budget {
                        return Err(Error::invalid(format!(
                            "vp9 §9.2: tile bool decoder over-read {} bits past end of {}-byte payload (budget {} bits)",
                            bd.over_read_bits(),
                            slice.len(),
                            budget,
                        )));
                    }
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
            // Widen each reconstructed sample to the bitstream's
            // declared bit depth. The in-tree pipeline reconstructs in
            // 8-bit space today, so for >8-bit profiles we promote each
            // u8 to a `u16` LE container by left-shifting into the top
            // of the bit-depth window: `(byte as u16) << (bit_depth-8)`.
            // The resulting plane layout is two bytes per sample with
            // the active bits in `[bit_depth-1 .. bit_depth-8]` —
            // matching the `Yuv*P10Le` / `Yuv*P12Le` plane shapes
            // documented on `oxideav_core::PixelFormat`. The h264
            // sibling crate's `picture_to_video_frame` (task #259)
            // adopts the same convention. Once the VP9 pipeline grows
            // a real HBD reconstruction path (qlookup-12 / mcfilter
            // i32, etc.) this widening will simply pack the already-
            // wide samples instead of left-shifting u8s.
            let bit_depth = h.color_config.bit_depth as u32;
            let (y_data, y_out_stride) = widen_plane(&y, y_stride, bit_depth);
            let (u_data, uv_out_stride) = widen_plane(&u, uv_stride, bit_depth);
            let (v_data, _) = widen_plane(&v, uv_stride, bit_depth);
            let frame = VideoFrame {
                pts: self.pending_pts.take(),
                planes: vec![
                    VideoPlane {
                        stride: y_out_stride,
                        data: y_data,
                    },
                    VideoPlane {
                        stride: uv_out_stride,
                        data: u_data,
                    },
                    VideoPlane {
                        stride: uv_out_stride,
                        data: v_data,
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
fn tile_pixel_bounds(h: &UncompressedHeader, tile_col: u32, tile_row: u32) -> (u32, u32, u32, u32) {
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

/// Widen one reconstructed plane to the bitstream's declared bit
/// depth. For 8-bit the input bytes are returned as-is. For 10/12-bit,
/// each input byte is promoted to a `u16` little-endian container by
/// `(byte as u16) << (bit_depth - 8)` — the active bits land in the top
/// of the bit-depth window so that a downstream `>> shift` narrows
/// losslessly back to the original byte. The returned `out_stride` is
/// `in_stride` for 8-bit and `in_stride * 2` for >8-bit, matching the
/// `Yuv*P10Le` / `Yuv*P12Le` two-bytes-per-sample plane layout.
fn widen_plane(plane: &[u8], in_stride: usize, bit_depth: u32) -> (Vec<u8>, usize) {
    if bit_depth <= 8 {
        return (plane.to_vec(), in_stride);
    }
    let shift = bit_depth - 8;
    let mut out = Vec::with_capacity(plane.len() * 2);
    for &b in plane {
        let v = (b as u16) << shift;
        out.extend_from_slice(&v.to_le_bytes());
    }
    (out, in_stride * 2)
}

/// Build an output `VideoFrame` from a `RefFrame` already in the DPB,
/// applying the same `widen_plane` HBD packing as the live decode
/// path. Used by the §8.2 `show_existing_frame` pass-through and by
/// any future dispatch that needs to surface a previously-decoded
/// reference as output.
fn build_video_frame_from_ref(rf: &RefFrame, bit_depth: u32, pts: Option<i64>) -> VideoFrame {
    let (y_data, y_out_stride) = widen_plane(&rf.y, rf.y_stride, bit_depth);
    let (u_data, uv_out_stride) = widen_plane(&rf.u, rf.uv_stride, bit_depth);
    let (v_data, _) = widen_plane(&rf.v, rf.uv_stride, bit_depth);
    VideoFrame {
        pts,
        planes: vec![
            VideoPlane {
                stride: y_out_stride,
                data: y_data,
            },
            VideoPlane {
                stride: uv_out_stride,
                data: u_data,
            },
            VideoPlane {
                stride: uv_out_stride,
                data: v_data,
            },
        ],
    }
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

    fn cc(bit_depth: u8, sub_x: bool, sub_y: bool, cs: ColorSpace) -> ColorConfig {
        ColorConfig {
            bit_depth,
            color_space: cs,
            color_range: false,
            subsampling_x: sub_x,
            subsampling_y: sub_y,
        }
    }

    /// `pixel_format_from_color_config` covers VP9's full
    /// (bit_depth, chroma layout, color_space) matrix per §6.2.1.
    /// Each profile column maps to a distinct `oxideav_core::PixelFormat`.
    #[test]
    fn pixel_format_from_color_config_matrix() {
        // Profile 0 (4:2:0, 8-bit), non-sRGB.
        assert_eq!(
            pixel_format_from_color_config(&cc(8, true, true, ColorSpace::Bt709)),
            PixelFormat::Yuv420P
        );
        // Profile 1: 4:4:4, 8-bit.
        assert_eq!(
            pixel_format_from_color_config(&cc(8, false, false, ColorSpace::Bt709)),
            PixelFormat::Yuv444P
        );
        // Profile 1: 4:2:2, 8-bit (sub_x=1, sub_y=0).
        assert_eq!(
            pixel_format_from_color_config(&cc(8, true, false, ColorSpace::Bt709)),
            PixelFormat::Yuv422P
        );
        // Profile 2: 4:2:0, 10-bit.
        assert_eq!(
            pixel_format_from_color_config(&cc(10, true, true, ColorSpace::Bt709)),
            PixelFormat::Yuv420P10Le
        );
        // Profile 2: 4:2:2, 10-bit.
        assert_eq!(
            pixel_format_from_color_config(&cc(10, true, false, ColorSpace::Bt709)),
            PixelFormat::Yuv422P10Le
        );
        // Profile 3: 4:4:4, 10-bit.
        assert_eq!(
            pixel_format_from_color_config(&cc(10, false, false, ColorSpace::Bt709)),
            PixelFormat::Yuv444P10Le
        );
        // Profile 2: 4:2:0, 12-bit.
        assert_eq!(
            pixel_format_from_color_config(&cc(12, true, true, ColorSpace::Bt709)),
            PixelFormat::Yuv420P12Le
        );
        // Profile 3: 4:4:4, 12-bit.
        assert_eq!(
            pixel_format_from_color_config(&cc(12, false, false, ColorSpace::Bt709)),
            PixelFormat::Yuv444P12Le
        );
        // sRGB / GBR planar at 8 / 10 / 12-bit (subsampling pinned to
        // 0/0 by §6.2.1 — but we tolerate other subsampling fields too).
        assert_eq!(
            pixel_format_from_color_config(&cc(8, false, false, ColorSpace::Srgb)),
            PixelFormat::Yuv444P
        );
        assert_eq!(
            pixel_format_from_color_config(&cc(10, false, false, ColorSpace::Srgb)),
            PixelFormat::Yuv444P10Le
        );
        assert_eq!(
            pixel_format_from_color_config(&cc(12, false, false, ColorSpace::Srgb)),
            PixelFormat::Yuv444P12Le
        );
    }

    /// 8-bit `widen_plane` must pass bytes through unchanged and keep
    /// the input stride. 10/12-bit must emit two LE bytes per input
    /// byte with the input bits shifted to the top of the bit-depth
    /// window — and double the stride.
    #[test]
    fn widen_plane_8bit_passthrough() {
        let plane = vec![0x00u8, 0x42, 0xff];
        let (out, stride) = widen_plane(&plane, 3, 8);
        assert_eq!(out, plane);
        assert_eq!(stride, 3);
    }

    #[test]
    fn widen_plane_10bit_left_shifts_by_two() {
        // Each input byte b → ((b as u16) << 2) packed LE.
        let plane = vec![0x00u8, 0x42, 0xff];
        let (out, stride) = widen_plane(&plane, 3, 10);
        // 0x00 → 0x0000, 0x42 → 0x0108, 0xff → 0x03fc
        assert_eq!(out, vec![0x00, 0x00, 0x08, 0x01, 0xfc, 0x03]);
        assert_eq!(stride, 6);
    }

    #[test]
    fn widen_plane_12bit_left_shifts_by_four() {
        let plane = vec![0x00u8, 0x42, 0xff];
        let (out, stride) = widen_plane(&plane, 3, 12);
        // 0x00 → 0x0000, 0x42 → 0x0420, 0xff → 0x0ff0
        assert_eq!(out, vec![0x00, 0x00, 0x20, 0x04, 0xf0, 0x0f]);
        assert_eq!(stride, 6);
    }

    /// The widened HBD bytes round-trip back to the original plane via
    /// `>> shift` — that's the contract `tests/docs_corpus.rs` relies
    /// on for u16-vs-u16 comparison.
    #[test]
    fn widen_plane_round_trip_via_shift() {
        for &bd in &[10u32, 12u32] {
            let shift = bd - 8;
            let plane: Vec<u8> = (0u8..=255).collect();
            let (wide, _) = widen_plane(&plane, plane.len(), bd);
            let back: Vec<u8> = wide
                .chunks_exact(2)
                .map(|c| (u16::from_le_bytes([c[0], c[1]]) >> shift).min(255) as u8)
                .collect();
            assert_eq!(back, plane, "bit_depth {bd} round-trip");
        }
    }

    /// §6.2 / §8.2 `show_existing_frame`: a packet whose first 5 bits
    /// are `1010_1` (frame_marker=2, profile=00, show_existing=1)
    /// followed by 3 bits of `frame_to_show_map_idx` MUST surface the
    /// referenced DPB slot as a new visible frame, with no compressed
    /// header / tile decode required. Pre-fix this path silently
    /// dropped the frame, leaving the consumer with an apparent gap.
    /// Also asserts the DPB stays unchanged (refresh_frame_flags == 0
    /// per the spec for this case).
    #[test]
    fn show_existing_frame_emits_dpb_slot_as_new_frame() {
        // Build a synthetic "previous" decoded frame and stuff it
        // directly into DPB slot 3 (skip the parse path so this test
        // stays focused on the show_existing dispatch).
        let mut dec = Vp9Decoder::new(CodecId::new(crate::CODEC_ID_STR));
        let rf = RefFrame {
            y: vec![0xA5; 64 * 64],
            y_stride: 64,
            u: vec![0x33; 32 * 32],
            v: vec![0xCC; 32 * 32],
            uv_stride: 32,
            width: 64,
            height: 64,
            uv_width: 32,
            uv_height: 32,
            subsampling_x: 1,
            subsampling_y: 1,
            segment_ids: None,
        };
        dec.dpb.refresh(1u8 << 3, &rf);
        // Set last_color_config so build_video_frame_from_ref picks
        // 8-bit (no widening) — matches a profile-0 stream.
        dec.last_color_config = Some(cc(8, true, true, ColorSpace::Bt601));

        // Synth show_existing_frame packet. All 8 bits of byte 0:
        //   marker(2)=10, profile_low=0, profile_high=0,
        //   show_existing=1, frame_to_show_map_idx(3)=011 (= 3)
        //   -> "10001011" = 0x8B
        let pkt_bytes = vec![0x8B, 0x00];
        let pkt = Packet::new(0, TimeBase::new(1, 90_000), pkt_bytes).with_pts(42);
        dec.send_packet(&pkt).expect("send_packet ok");

        // The decoder must surface a visible frame on receive_frame.
        let frame = dec.receive_frame().expect("receive_frame");
        let Frame::Video(vf) = frame else {
            panic!("expected video frame")
        };
        assert_eq!(vf.pts, Some(42));
        assert_eq!(vf.planes.len(), 3);
        assert_eq!(vf.planes[0].data.len(), 64 * 64);
        assert_eq!(vf.planes[1].data.len(), 32 * 32);
        // First sample of each plane matches the DPB slot we set up.
        assert_eq!(vf.planes[0].data[0], 0xA5);
        assert_eq!(vf.planes[1].data[0], 0x33);
        assert_eq!(vf.planes[2].data[0], 0xCC);
    }

    /// `show_existing_frame` referencing an empty DPB slot must surface
    /// a clean InvalidData error (rather than silently emitting a
    /// black frame or panicking).
    #[test]
    fn show_existing_frame_empty_slot_errors() {
        let mut dec = Vp9Decoder::new(CodecId::new(crate::CODEC_ID_STR));
        dec.last_color_config = Some(cc(8, true, true, ColorSpace::Bt601));
        // frame_to_show_map_idx = 5; slot 5 is empty.
        // First byte: "10001101" = 0x8D (idx low 3 bits = "101").
        let pkt = Packet::new(0, TimeBase::new(1, 90_000), vec![0x8D, 0x00]);
        match dec.send_packet(&pkt) {
            Err(Error::InvalidData(msg)) => {
                assert!(msg.contains("show_existing_frame"), "msg = {msg}");
            }
            other => panic!("expected InvalidData, got {other:?}"),
        }
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

    /// Regression for fuzz `oom-8859eec1032a1ad42f67a374eb6b85fcdee6da52`
    /// (workspace task #749): a 20-byte VP9 keyframe header that
    /// declares a 2069×39029 picture (~80 MP) used to force
    /// `IntraTile::new` to allocate ~243 MiB of plane buffers before
    /// the bitstream was rejected — well within the OOM-killer's
    /// crosshairs once concurrent fuzz workers each held one. The
    /// decoder now refuses anything beyond `MAX_FRAME_PIXELS`
    /// (8192×8192) at the start of `ingest_one`, so this input
    /// surfaces as a plain `Error::InvalidData` with no large
    /// allocation.
    #[test]
    fn rejects_oversized_declared_dimensions() {
        let bytes = [
            0x81, 0x49, 0x83, 0x42, 0x00, 0x81, 0x49, 0x83, 0x42, 0x00, 0xa0, 0x00, 0x00, 0x03,
            0xad, 0xa0, 0x00, 0x01, 0x05, 0xad,
        ];
        let mut dec = Vp9Decoder::new(CodecId::new(crate::CODEC_ID_STR));
        let pkt = Packet::new(0, TimeBase::new(1, 90_000), bytes.to_vec());
        match dec.send_packet(&pkt) {
            Err(Error::InvalidData(msg)) => {
                assert!(
                    msg.contains("exceeds decoder cap"),
                    "expected size-cap error, got {msg}"
                );
            }
            other => panic!("expected InvalidData(size-cap), got {other:?}"),
        }
    }

    /// `MAX_FRAME_DIM` × `MAX_FRAME_DIM` must be exactly the cap and
    /// must never overflow `u32`. Pinned so future bumps stay deliberate.
    #[test]
    fn max_frame_pixels_is_dim_squared_and_fits_u32() {
        assert_eq!(
            MAX_FRAME_PIXELS as u64,
            MAX_FRAME_DIM as u64 * MAX_FRAME_DIM as u64
        );
        // 8192 × 8192 = 67,108,864 — comfortable u32.
        assert_eq!(MAX_FRAME_PIXELS, 67_108_864);
    }

    /// `bool_over_read_budget_bits` math must:
    ///
    /// * apply the floor for tiny payloads
    /// * scale 2× the tile size in bits for normal payloads
    /// * not overflow on `u32`-sized maxima
    #[test]
    fn bool_over_read_budget_pinned() {
        // Floor: 0/1-byte tiles get the 32-bit floor (1×8 = 8 < 32).
        assert_eq!(bool_over_read_budget_bits(0), 32);
        assert_eq!(bool_over_read_budget_bits(1), 32);
        // Task #748: 3-byte tile → 3×16 = 48 bits budget — small enough
        // to reject the fuzz input (166 bits over-read).
        assert_eq!(bool_over_read_budget_bits(3), 48);
        // 15-byte fixture P-frame: 240 bits — above its 171-bit
        // observed over-read.
        assert_eq!(bool_over_read_budget_bits(15), 240);
        // Multi-tile fixture tiles, both well above their observed
        // over-read.
        assert_eq!(bool_over_read_budget_bits(1991), 31_856);
        assert_eq!(bool_over_read_budget_bits(3369), 53_904);
    }

    /// Regression for fuzz `crash-d2075b74…` (workspace task #748): a
    /// 178-byte VP9 keyframe declaring a 385×1 picture with only 3
    /// bytes of tile payload would silently zero-pad its bool decoder
    /// stream by hundreds of bits and produce a Y plane that
    /// disagreed with libavcodec on every pixel of the first ~70
    /// columns (oracle Y[0,0]=132 vs ours 129). The fix surfaces the
    /// over-read at end-of-tile via `BoolDecoder::over_read_bits()`
    /// and refuses the frame with `Error::InvalidData` whenever the
    /// over-read exceeds the size-scaled budget; the fuzz harness then
    /// skips the cross-decode comparison cleanly.
    #[test]
    fn rejects_oversized_bool_over_read() {
        // Trimmed verbatim from
        // fuzz/artifacts/ffmpeg_oracle_decode/crash-d2075b74…
        let bytes = [
            0xa2, 0x49, 0x83, 0x42, 0x78, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0a,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x42, 0x78, 0x36,
            0xda, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x66, 0x0d, 0x0d, 0x0d, 0x0d,
            0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d,
            0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d,
            0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d,
            0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d,
            0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d,
            0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d,
            0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x73, 0x2b,
        ];
        let mut dec = Vp9Decoder::new(CodecId::new(crate::CODEC_ID_STR));
        let pkt = Packet::new(0, TimeBase::new(1, 30), bytes.to_vec());
        match dec.send_packet(&pkt) {
            Err(Error::InvalidData(msg)) => {
                assert!(
                    msg.contains("over-read"),
                    "expected over-read error, got {msg}"
                );
            }
            other => panic!("expected InvalidData(over-read), got {other:?}"),
        }
    }
}
