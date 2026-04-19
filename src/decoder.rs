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
use crate::compressed_header::parse_compressed_header;
use crate::dpb::{Dpb, RefFrame};
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
        let mut h = parse_uncompressed_header(&packet.data, self.last_color_config)?;
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

        // Parse compressed header (§6.3).
        let cmp_start = h.uncompressed_header_size;
        let cmp_end = cmp_start.saturating_add(h.header_size as usize);
        if cmp_end > packet.data.len() || h.header_size == 0 {
            return Err(Error::invalid("vp9 compressed header missing or truncated"));
        }
        let ch = parse_compressed_header(&packet.data[cmp_start..cmp_end], &h)?;
        let tile_payload = &packet.data[cmp_end..];
        if tile_payload.is_empty() {
            return Err(Error::invalid("vp9 tile payload empty"));
        }
        if h.tile_info.log2_tile_cols != 0 || h.tile_info.log2_tile_rows != 0 {
            self.last_header = Some(h);
            return Err(Error::unsupported(
                "vp9 multi-tile frames pending (log2_tile > 0)",
            ));
        }

        let (y, y_stride, u, v, uv_stride, uv_w, uv_h) = if is_intra {
            let mut tile = IntraTile::new(&h, &ch);
            let mut bd = BoolDecoder::new(tile_payload)?;
            tile.decode(&mut bd)?;
            (
                tile.y,
                tile.y_stride,
                tile.u,
                tile.v,
                tile.uv_stride,
                tile.uv_w,
                tile.uv_h,
            )
        } else {
            let refs = [
                self.dpb.get(h.ref_frame_idx[0]),
                self.dpb.get(h.ref_frame_idx[1]),
                self.dpb.get(h.ref_frame_idx[2]),
            ];
            let mut tile = InterTile::new(&h, &ch, h.width as usize, h.height as usize, refs);
            let mut bd = BoolDecoder::new(tile_payload)?;
            tile.decode(&mut bd)?;
            (
                tile.y,
                tile.y_stride,
                tile.u,
                tile.v,
                tile.uv_stride,
                tile.uv_w,
                tile.uv_h,
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
        self.eof = false;
        Ok(())
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
