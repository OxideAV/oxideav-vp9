//! oxideav-core registry integration.
//!
//! The framework-facing [`Vp9Decoder`] / [`Vp9Encoder`] pair (the
//! [`oxideav_core::registry::Decoder`] / [`Encoder`] trait objects the
//! registry hands out), the direct [`make_decoder`] / [`make_encoder`]
//! factories, and the [`register`] entry that installs the codec into a
//! [`RuntimeContext`].
//!
//! Both sides ride the crate's sequence engines:
//!
//! * the decoder streams packets through [`Vp9SequenceDecoder`] — the
//!   incremental form of [`crate::decode_vp9_sequence`], threading the
//!   §8.10 reference buffers, the §6.5 previous-frame motion field, the
//!   §6.1.2 / §7.2 `FrameContext[ 4 ]` entropy banks, and the §7.2.8 /
//!   §7.2.10 persistent header state across packets — with the §B.2
//!   Annex B superframe split applied per packet ([`split_superframe`]),
//!   so hidden alt-ref frames and `show_existing_frame` re-displays
//!   work exactly as in the batch API;
//! * the encoder streams frames through the **§7.2.6 chain-framed
//!   default GOP path** (the same engine behind
//!   [`crate::encode_vp9_lossy_sequence`] /
//!   [`crate::encode_vp9_lossless_sequence`]): the first frame codes a
//!   keyframe, every later frame a shown non-error-resilient P-frame
//!   with prev-frame-MV modeling, multi-reference + compound election,
//!   and the per-frame §8.8 filter (+ §6.2.8 delta) elections.

use std::collections::VecDeque;

use oxideav_core::registry::{CodecInfo, Decoder, Encoder};
use oxideav_core::{
    parse_options, CodecCapabilities, CodecId, CodecOptionsStruct, CodecParameters, CodecTag,
    Error as CoreError, ExecutionContext, Frame, OptionField, OptionKind, OptionValue, Packet,
    PixelFormat, Result as CoreResult, RuntimeContext, TimeBase, VideoFrame, VideoPlane,
};

use crate::decode_frame::{Vp9DecodedFrame, Vp9SequenceDecoder};
use crate::pixel_encoder::{
    padded_targets_from_u16, padded_targets_from_u8, ChainModel, LosslessGopEncoder420,
    LossyFormat, LossyGopEncoder,
};
use crate::superframe::split_superframe;

/// Registry codec id this crate answers to.
const VP9_CODEC_ID: &str = "vp9";

/// Map this crate's [`crate::Error`] onto the framework error space:
/// [`crate::Error::Unsupported`] (valid but out of scope) keeps its
/// meaning, everything else is malformed input.
fn map_err(e: crate::Error) -> CoreError {
    match e {
        crate::Error::Unsupported | crate::Error::NotImplemented => {
            CoreError::unsupported(e.to_string())
        }
        crate::Error::UnexpectedEof | crate::Error::InvalidBitstream => {
            CoreError::invalid(e.to_string())
        }
    }
}

/// The §7.2.2 format triple `(BitDepth, subsampling_x, subsampling_y)`
/// for a framework pixel format, if VP9 can carry it.
fn format_triple(fmt: PixelFormat) -> Option<(u8, bool, bool)> {
    Some(match fmt {
        PixelFormat::Yuv420P => (8, true, true),
        PixelFormat::Yuv422P => (8, true, false),
        PixelFormat::Yuv444P => (8, false, false),
        PixelFormat::Yuv420P10Le => (10, true, true),
        PixelFormat::Yuv422P10Le => (10, true, false),
        PixelFormat::Yuv444P10Le => (10, false, false),
        PixelFormat::Yuv420P12Le => (12, true, true),
        PixelFormat::Yuv422P12Le => (12, true, false),
        PixelFormat::Yuv444P12Le => (12, false, false),
        PixelFormat::Yuv440P => (8, false, true),
        PixelFormat::Yuv440P10Le => (10, false, true),
        PixelFormat::Yuv440P12Le => (12, false, true),
        _ => return None,
    })
}

/// The framework pixel format for a §7.2.2 triple. Every one of the
/// twelve `(BitDepth, subsampling_x, subsampling_y)` combinations the
/// §6.2.2 `color_config( )` syntax can signal has a label — the 4:4:0
/// geometry (`ssx = 0, ssy = 1`) maps onto the framework's `Yuv440P`
/// family (full-width, half-height chroma). `None` only for a
/// bit-depth outside the §7.2.2 set, which the header parser already
/// rejects.
pub fn pixel_format_for_triple(bit_depth: u8, ssx: bool, ssy: bool) -> Option<PixelFormat> {
    Some(match (bit_depth, ssx, ssy) {
        (8, true, true) => PixelFormat::Yuv420P,
        (8, true, false) => PixelFormat::Yuv422P,
        (8, false, false) => PixelFormat::Yuv444P,
        (10, true, true) => PixelFormat::Yuv420P10Le,
        (10, true, false) => PixelFormat::Yuv422P10Le,
        (10, false, false) => PixelFormat::Yuv444P10Le,
        (12, true, true) => PixelFormat::Yuv420P12Le,
        (12, true, false) => PixelFormat::Yuv422P12Le,
        (12, false, false) => PixelFormat::Yuv444P12Le,
        (8, false, true) => PixelFormat::Yuv440P,
        (10, false, true) => PixelFormat::Yuv440P10Le,
        (12, false, true) => PixelFormat::Yuv440P12Le,
        _ => return None,
    })
}

/// Chroma plane extent per §8.10: `ceil` division on each subsampled
/// axis.
fn chroma_dims(width: u32, height: u32, ssx: bool, ssy: bool) -> (usize, usize) {
    let cw = if ssx { width.div_ceil(2) } else { width } as usize;
    let ch = if ssy { height.div_ceil(2) } else { height } as usize;
    (cw, ch)
}

/// Pack a decoded plane (`u16` samples) into framework plane bytes:
/// one byte per sample at 8-bit, §8.10 little-endian pairs at 10/12.
fn pack_plane(samples: &[u16], width: usize, bit_depth: u8) -> VideoPlane {
    if bit_depth == 8 {
        VideoPlane {
            stride: width,
            data: samples.iter().map(|&s| s as u8).collect(),
        }
    } else {
        let mut data = Vec::with_capacity(samples.len() * 2);
        for &s in samples {
            data.extend_from_slice(&s.to_le_bytes());
        }
        VideoPlane {
            stride: width * 2,
            data,
        }
    }
}

/// Convert one decoded frame into a framework [`Frame`], carrying the
/// packet's presentation timestamp through, and report the frame's
/// framework pixel format. The plane geometry is the framework's own
/// `PixelFormat::plane_dimensions` rule for that label (§8.10 `ceil`
/// on each subsampled axis — for 4:4:0, full-width, half-height
/// chroma).
fn decoded_to_frame(f: &Vp9DecodedFrame, pts: Option<i64>) -> CoreResult<(PixelFormat, Frame)> {
    let Some(fmt) = pixel_format_for_triple(f.bit_depth, f.subsampling_x, f.subsampling_y) else {
        return Err(CoreError::unsupported(format!(
            "vp9: no framework pixel-format label for bit depth {} (§7.2.2 allows 8/10/12)",
            f.bit_depth
        )));
    };
    let (cw, ch) = chroma_dims(f.width, f.height, f.subsampling_x, f.subsampling_y);
    debug_assert_eq!(
        fmt.plane_dimensions(1, f.width, f.height),
        Some((cw as u32, ch as u32)),
        "framework chroma geometry must agree with §8.10"
    );
    debug_assert_eq!(f.u.len(), cw * ch);
    Ok((
        fmt,
        Frame::Video(VideoFrame {
            pts,
            planes: vec![
                pack_plane(&f.y, f.width as usize, f.bit_depth),
                pack_plane(&f.u, cw, f.bit_depth),
                pack_plane(&f.v, cw, f.bit_depth),
            ],
        }),
    ))
}

// ───────────────────────── decoder ─────────────────────────

/// Framework [`Decoder`] over the incremental [`Vp9SequenceDecoder`].
///
/// Each packet is one temporal unit: it is Annex-B-split
/// ([`split_superframe`]) and every enclosed coded frame is pushed in
/// decode order, so a packet carrying a hidden alt-ref plus a
/// `show_existing_frame` behaves exactly like the corpus streams do
/// under [`crate::decode_vp9_sequence`]. Shown frames queue for
/// [`Self::receive_frame`] with the packet's `pts` attached.
#[derive(Debug)]
pub struct Vp9Decoder {
    id: CodecId,
    seq: Vp9SequenceDecoder,
    pending: VecDeque<Frame>,
    flushed: bool,
    /// The framework label of the most recently decoded frame's §7.2.2
    /// format triple (`None` before the first decoded frame or after
    /// [`Decoder::reset`]).
    format: Option<PixelFormat>,
    /// The caller-granted threading budget (framework contract: serial
    /// until `set_execution_context` says otherwise); forwarded to the
    /// sequence decoder's §6.4 tile-column fan-out and preserved across
    /// [`Decoder::reset`].
    exec: ExecutionContext,
}

impl Vp9Decoder {
    /// Fresh decoder state (the stream must start with a keyframe).
    pub fn new() -> Self {
        Self {
            id: CodecId::new(VP9_CODEC_ID),
            seq: Vp9SequenceDecoder::new(),
            pending: VecDeque::new(),
            flushed: false,
            format: None,
            exec: ExecutionContext::serial(),
        }
    }

    /// The framework pixel format the decoded stream is labelled with —
    /// the §7.2.2 `(BitDepth, subsampling_x, subsampling_y)` triple of
    /// the most recently decoded frame mapped onto the framework's
    /// planar YUV families (4:2:0 / 4:2:2 / 4:4:4 / 4:4:0 at 8, 10 and
    /// 12 bits). `None` until a frame has been decoded, and again after
    /// [`Decoder::reset`].
    pub fn pixel_format(&self) -> Option<PixelFormat> {
        self.format
    }
}

impl Default for Vp9Decoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Decoder for Vp9Decoder {
    fn codec_id(&self) -> &CodecId {
        &self.id
    }

    fn set_execution_context(&mut self, ctx: &ExecutionContext) {
        // The single threading authority: multi-tile-column frames
        // fan their §6.4 tile columns out over
        // `ctx.effective_workers( tileCols )` workers (the §9.2.4
        // multi-coder path), byte-identical to the serial walk.
        self.exec = ctx.clone();
        self.seq.set_execution_context(ctx);
    }

    fn send_packet(&mut self, packet: &Packet) -> CoreResult<()> {
        // §B.2 / §B.4: split the packet into its coded frames (a chunk
        // with no valid superframe index is a single coded frame — the
        // split is total, per the §B.4 fallback).
        let frames = split_superframe(&packet.data);
        for payload in frames {
            if let Some(decoded) = self.seq.push_frame(payload).map_err(map_err)? {
                let (fmt, frame) = decoded_to_frame(&decoded, packet.pts)?;
                self.format = Some(fmt);
                self.pending.push_back(frame);
            }
        }
        Ok(())
    }

    fn receive_frame(&mut self) -> CoreResult<Frame> {
        match self.pending.pop_front() {
            Some(f) => Ok(f),
            None if self.flushed => Err(CoreError::Eof),
            None => Err(CoreError::NeedMore),
        }
    }

    fn flush(&mut self) -> CoreResult<()> {
        // VP9 decode is packet-synchronous (every shown frame is
        // emitted by the send_packet that carried it), so flushing only
        // marks end-of-stream for the receive loop.
        self.flushed = true;
        Ok(())
    }

    fn reset(&mut self) -> CoreResult<()> {
        // A container seek lands on a keyframe; §7.2
        // setup_past_independence( ) makes a keyframe self-contained,
        // so fresh cross-frame state (empty §8.10 buffers, default
        // probability banks) is exactly the decoder state a conforming
        // stream expects there. The granted threading budget is stream
        // configuration, not stream state — it survives the reset.
        self.seq = Vp9SequenceDecoder::new();
        self.seq.set_execution_context(&self.exec);
        self.pending.clear();
        self.flushed = false;
        self.format = None;
        Ok(())
    }
}

// ───────────────────────── encoder ─────────────────────────

/// Typed options for [`Vp9Encoder`] (`CodecParameters::options`).
#[derive(Debug, Clone)]
pub struct Vp9EncoderOptions {
    /// §7.2.9 `base_q_idx` for every frame of the GOP (`1..=255`,
    /// smaller = higher quality). Ignored when `lossless` is set.
    pub q: u32,
    /// Lossless encode (§7.2.9 `Lossless == 1`: `base_q_idx == 0` with
    /// zero deltas — the §8.7.2 WHT path). Requires 8-bit 4:2:0 input.
    pub lossless: bool,
}

impl Default for Vp9EncoderOptions {
    fn default() -> Self {
        Self {
            q: 110,
            lossless: false,
        }
    }
}

impl CodecOptionsStruct for Vp9EncoderOptions {
    const SCHEMA: &'static [OptionField] = &[
        OptionField {
            name: "q",
            kind: OptionKind::U32,
            default: OptionValue::U32(110),
            help: "quantizer index (base_q_idx) for every frame, 1..=255; smaller = higher quality",
        },
        OptionField {
            name: "lossless",
            kind: OptionKind::Bool,
            default: OptionValue::Bool(false),
            help: "lossless encode (WHT residual path); requires yuv420p input",
        },
    ];

    fn apply(&mut self, key: &str, value: &OptionValue) -> CoreResult<()> {
        match key {
            "q" => self.q = value.as_u32()?,
            "lossless" => self.lossless = value.as_bool()?,
            _ => unreachable!("guarded by SCHEMA"),
        }
        Ok(())
    }
}

/// The two chain-framed GOP engines behind [`Vp9Encoder`].
enum GopBackend {
    /// 8-bit 4:2:0 lossless chain ([`LosslessGopEncoder420`]).
    Lossless(LosslessGopEncoder420),
    /// Lossy chain at any §7.2 matrix format ([`LossyGopEncoder`]).
    Lossy(Box<LossyGopEncoder>),
}

/// Framework [`Encoder`] riding the §7.2.6 **chain-framed default GOP
/// path**: frame 0 codes a keyframe, every later frame a shown
/// non-error-resilient P-frame (prev-frame-MV modeling, LAST/GOLDEN +
/// `[ LAST, ALTREF ]` compound election, per-frame §8.8 filter and
/// §6.2.8 delta elections — the exact engine behind
/// [`crate::encode_vp9_lossy_sequence`] /
/// [`crate::encode_vp9_lossless_sequence`], one frame per
/// [`Self::send_frame`]).
pub struct Vp9Encoder {
    id: CodecId,
    output: CodecParameters,
    backend: GopBackend,
    width: u32,
    height: u32,
    bit_depth: u8,
    ssx: bool,
    ssy: bool,
    pending: VecDeque<Packet>,
    frames_sent: u64,
    flushed: bool,
}

impl std::fmt::Debug for Vp9Encoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The backend holds megabyte-scale chain state; summarise.
        f.debug_struct("Vp9Encoder")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("bit_depth", &self.bit_depth)
            .field("lossless", &matches!(self.backend, GopBackend::Lossless(_)))
            .field("frames_sent", &self.frames_sent)
            .finish_non_exhaustive()
    }
}

impl Vp9Encoder {
    /// Build an encoder from stream parameters: `width` / `height` /
    /// `pixel_format` are required; `options` carries
    /// [`Vp9EncoderOptions`].
    pub fn new(params: &CodecParameters) -> CoreResult<Self> {
        let width = params
            .width
            .ok_or_else(|| CoreError::invalid("vp9 encoder: width is required"))?;
        let height = params
            .height
            .ok_or_else(|| CoreError::invalid("vp9 encoder: height is required"))?;
        let fmt = params
            .pixel_format
            .ok_or_else(|| CoreError::invalid("vp9 encoder: pixel_format is required"))?;
        let opts: Vp9EncoderOptions = parse_options(&params.options)?;

        let Some((bit_depth, ssx, ssy)) = format_triple(fmt) else {
            return Err(CoreError::unsupported(format!(
                "vp9 encoder: pixel format {fmt:?} has no §7.2.2 encoder mapping"
            )));
        };

        let backend = if opts.lossless {
            if fmt != PixelFormat::Yuv420P {
                return Err(CoreError::unsupported(
                    "vp9 encoder: the lossless GOP path is 8-bit 4:2:0 (yuv420p) only",
                ));
            }
            GopBackend::Lossless(LosslessGopEncoder420::new(width, height).map_err(map_err)?)
        } else {
            let q: u8 = u8::try_from(opts.q)
                .ok()
                .filter(|&q| q >= 1)
                .ok_or_else(|| CoreError::invalid("vp9 encoder: q must be in 1..=255"))?;
            let lossy_fmt = LossyFormat::new(bit_depth, ssx, ssy).map_err(map_err)?;
            GopBackend::Lossy(Box::new(
                LossyGopEncoder::new(width, height, q, lossy_fmt, ChainModel::Adaptive)
                    .map_err(map_err)?,
            ))
        };

        let mut output = CodecParameters::video(CodecId::new(VP9_CODEC_ID));
        output.width = Some(width);
        output.height = Some(height);
        output.pixel_format = Some(fmt);
        output.frame_rate = params.frame_rate;

        Ok(Self {
            id: CodecId::new(VP9_CODEC_ID),
            output,
            backend,
            width,
            height,
            bit_depth,
            ssx,
            ssy,
            pending: VecDeque::new(),
            frames_sent: 0,
            flushed: false,
        })
    }

    /// De-stride the frame's three planes into one packed planar byte
    /// buffer (`Y` then `U` then `V`, visible extents — the layout
    /// every public encode entry consumes).
    fn packed_planar(&self, v: &VideoFrame) -> CoreResult<Vec<u8>> {
        let planes = v.image_planes();
        if planes.len() != 3 {
            return Err(CoreError::invalid(format!(
                "vp9 encoder: expected 3 image planes, got {}",
                planes.len()
            )));
        }
        let bps = if self.bit_depth == 8 { 1usize } else { 2usize };
        let (cw, ch) = chroma_dims(self.width, self.height, self.ssx, self.ssy);
        let dims = [
            (self.width as usize, self.height as usize),
            (cw, ch),
            (cw, ch),
        ];
        let mut out = Vec::with_capacity((dims[0].0 * dims[0].1 + 2 * cw * ch) * bps);
        for (plane, &(w, h)) in planes.iter().zip(&dims) {
            let row_bytes = w * bps;
            if plane.stride < row_bytes || plane.data.len() < (h - 1) * plane.stride + row_bytes {
                return Err(CoreError::invalid(
                    "vp9 encoder: plane buffer too short for the declared geometry",
                ));
            }
            for row in 0..h {
                let at = row * plane.stride;
                out.extend_from_slice(&plane.data[at..at + row_bytes]);
            }
        }
        Ok(out)
    }
}

impl Encoder for Vp9Encoder {
    fn codec_id(&self) -> &CodecId {
        &self.id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.output
    }

    fn send_frame(&mut self, frame: &Frame) -> CoreResult<()> {
        let Frame::Video(v) = frame else {
            return Err(CoreError::invalid("vp9 encoder: expected a video frame"));
        };
        let planar = self.packed_planar(v)?;
        let bytes = match &mut self.backend {
            GopBackend::Lossless(enc) => enc.push(&planar).map_err(map_err)?,
            GopBackend::Lossy(enc) => {
                let fmt = LossyFormat::new(self.bit_depth, self.ssx, self.ssy).map_err(map_err)?;
                let targets = if self.bit_depth == 8 {
                    padded_targets_from_u8(&planar, self.width, self.height, fmt)
                } else {
                    // §8.10 output convention: little-endian pairs per
                    // sample — reassemble the native u16 planes.
                    let samples: Vec<u16> = planar
                        .chunks_exact(2)
                        .map(|c| u16::from_le_bytes([c[0], c[1]]))
                        .collect();
                    let max = (1u16 << self.bit_depth) - 1;
                    if samples.iter().any(|&s| s > max) {
                        return Err(CoreError::invalid(format!(
                            "vp9 encoder: sample exceeds the {}-bit range",
                            self.bit_depth
                        )));
                    }
                    padded_targets_from_u16(&samples, self.width, self.height, fmt)
                };
                enc.push(&targets).map_err(map_err)?
            }
        };
        let keyframe = self.frames_sent == 0;
        self.frames_sent += 1;
        let mut packet = Packet::new(0, TimeBase::MILLIS, bytes).with_keyframe(keyframe);
        packet.pts = v.pts;
        packet.dts = v.pts;
        self.pending.push_back(packet);
        Ok(())
    }

    fn receive_packet(&mut self) -> CoreResult<Packet> {
        match self.pending.pop_front() {
            Some(p) => Ok(p),
            None if self.flushed => Err(CoreError::Eof),
            None => Err(CoreError::NeedMore),
        }
    }

    fn flush(&mut self) -> CoreResult<()> {
        // The chain codes one packet per frame with no lookahead, so
        // flushing only marks end-of-stream for the receive loop.
        self.flushed = true;
        Ok(())
    }
}

// ───────────────────────── factories + registration ─────────────────────────

/// Direct decoder factory (the registry's `DecoderFactory`): build a
/// fresh [`Vp9Decoder`] for the given stream parameters.
pub fn make_decoder(_params: &CodecParameters) -> CoreResult<Box<dyn Decoder>> {
    Ok(Box::new(Vp9Decoder::new()))
}

/// Direct encoder factory (the registry's `EncoderFactory`): build a
/// [`Vp9Encoder`] from `width` / `height` / `pixel_format` +
/// [`Vp9EncoderOptions`].
pub fn make_encoder(params: &CodecParameters) -> CoreResult<Box<dyn Encoder>> {
    Ok(Box::new(Vp9Encoder::new(params)?))
}

/// Install this codec into the runtime context: decode + encode
/// factories, the §7.2 format-matrix pixel formats, and the container
/// tag claims.
pub fn register(ctx: &mut RuntimeContext) {
    ctx.codecs.register(
        CodecInfo::new(CodecId::new(VP9_CODEC_ID))
            .capabilities(
                CodecCapabilities::video("oxideav-vp9")
                    .with_decode()
                    .with_encode()
                    .with_lossy(true)
                    .with_lossless(true)
                    .with_pixel_formats(vec![
                        PixelFormat::Yuv420P,
                        PixelFormat::Yuv422P,
                        PixelFormat::Yuv444P,
                        PixelFormat::Yuv420P10Le,
                        PixelFormat::Yuv422P10Le,
                        PixelFormat::Yuv444P10Le,
                        PixelFormat::Yuv420P12Le,
                        PixelFormat::Yuv422P12Le,
                        PixelFormat::Yuv444P12Le,
                        PixelFormat::Yuv440P,
                        PixelFormat::Yuv440P10Le,
                        PixelFormat::Yuv440P12Le,
                    ]),
            )
            .decoder(make_decoder)
            .encoder(make_encoder)
            .encoder_options::<Vp9EncoderOptions>()
            .tags([
                CodecTag::fourcc(b"VP90"),
                CodecTag::fourcc(b"VP09"),
                CodecTag::matroska("V_VP9"),
            ]),
    );
}
