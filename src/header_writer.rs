//! VP9 §6.2 uncompressed-header **writer** — the inverse of
//! `header::parse_uncompressed_header`.
//!
//! The encoder bootstrap needs to *emit* the §6.2 uncompressed header
//! (frame marker / profile / frame-type branch / color config / sizes /
//! loop-filter / quantization / segmentation / tile info /
//! `header_size_in_bytes`) plus the §6.1.1 `trailing_bits()` zero-fill.
//! This module provides a minimal MSB-first [`BitWriter`] (`f(n)` /
//! `s(n)` / byte-alignment) and [`write_uncompressed_header`], which
//! walks the same field order the parser reads so a written header parses
//! back to an equivalent [`Vp9FrameHeader`].
//!
//! Scope: the **key-frame** branch (and the `show_existing_frame`
//! sentinel) plus the **inter** (non-intra-only, shown) branch —
//! `reset_frame_context` / `refresh_frame_flags` / `ref_frame_idx` +
//! `ref_frame_sign_bias`, the §6.2.5 explicit-`frame_size` path
//! (`found_ref = 0` for all references), `allow_high_precision_mv`, and
//! the §6.2.7 `interpolation_filter`. The intra-only inter branch
//! (`show_frame == 0` ⇒ `intra_only` flag) is not yet written; such a
//! header returns [`Error::Unsupported`].
//!
//! Provenance: VP9 Bitstream & Decoding Process Specification v0.7
//! (`docs/video/vp9/vp9-spec.txt`) §6.2; the field order mirrors the
//! in-crate parser exactly.

use crate::header::{
    ColorConfig, FrameType, LoopFilterParams, QuantizationParams, SegmentationParams, TileInfo,
    Vp9FrameHeader, MAX_SEGMENTS, SEGMENTATION_FEATURE_BITS, SEGMENTATION_FEATURE_SIGNED,
    SEG_LVL_MAX,
};
use crate::Error;

/// MSB-first bit writer producing a byte buffer. The `f(n)` / `s(n)`
/// primitives mirror the `BitReader` parsing process bit-for-bit.
#[derive(Debug, Default)]
pub(crate) struct BitWriter {
    out: Vec<u8>,
    /// Bits buffered in the partial trailing byte, MSB-first.
    cur: u8,
    /// Number of bits currently held in `cur` (`0..=7`).
    nbits: u8,
}

impl BitWriter {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Current bit position (number of bits written so far).
    pub(crate) fn position(&self) -> usize {
        self.out.len() * 8 + self.nbits as usize
    }

    /// `f(n)`: write the low `n` bits of `value`, MSB-first.
    pub(crate) fn write_bits(&mut self, value: u32, n: u32) {
        debug_assert!(n <= 32);
        for i in (0..n).rev() {
            let bit = ((value >> i) & 1) as u8;
            self.cur = (self.cur << 1) | bit;
            self.nbits += 1;
            if self.nbits == 8 {
                self.out.push(self.cur);
                self.cur = 0;
                self.nbits = 0;
            }
        }
    }

    /// `f(1)` of a boolean flag.
    pub(crate) fn write_flag(&mut self, b: bool) {
        self.write_bits(u32::from(b), 1);
    }

    /// `s(n)`: an `n`-bit magnitude followed by a 1-bit sign (1 =
    /// negative), inverting `BitReader::read_signed`.
    pub(crate) fn write_signed(&mut self, value: i32, n: u32) {
        debug_assert!(n <= 31);
        let mag = value.unsigned_abs();
        self.write_bits(mag, n);
        self.write_flag(value < 0);
    }

    /// `trailing_bits()` (§6.1.1): zero-fill to the next byte boundary.
    pub(crate) fn trailing_bits(&mut self) {
        if self.nbits != 0 {
            self.cur <<= 8 - self.nbits;
            self.out.push(self.cur);
            self.cur = 0;
            self.nbits = 0;
        }
    }

    /// Consume the writer, returning the byte buffer. Requires the
    /// writer to be byte-aligned (call `trailing_bits` first).
    pub(crate) fn into_bytes(self) -> Vec<u8> {
        debug_assert_eq!(self.nbits, 0, "into_bytes on a non-byte-aligned writer");
        self.out
    }
}

/// §7.2 frame-sync code (`0x49 0x83 0x42`).
fn write_frame_sync_code(w: &mut BitWriter) {
    w.write_bits(0x49, 8);
    w.write_bits(0x83, 8);
    w.write_bits(0x42, 8);
}

/// §6.2.2 `color_config()` — inverse of `read_color_config`.
fn write_color_config(w: &mut BitWriter, profile: u8, cc: &ColorConfig) -> Result<(), Error> {
    if profile >= 2 {
        // ten_or_twelve_bit: 1 => 12, 0 => 10.
        match cc.bit_depth {
            10 => w.write_bits(0, 1),
            12 => w.write_bits(1, 1),
            _ => return Err(Error::Unsupported),
        }
    } else if cc.bit_depth != 8 {
        return Err(Error::Unsupported);
    }

    w.write_bits(cc.color_space.to_bits(), 3);
    if cc.color_space.to_bits() != 7 {
        // Non-RGB: color_range, then (profile 1/3) subsampling + reserved.
        w.write_flag(cc.color_range_full);
        if profile == 1 || profile == 3 {
            // §7.2.2: profile_low_bit==1 forbids 4:2:0.
            if (profile & 1) == 1 && cc.subsampling_x && cc.subsampling_y {
                return Err(Error::InvalidBitstream);
            }
            w.write_flag(cc.subsampling_x);
            w.write_flag(cc.subsampling_y);
            w.write_bits(0, 1); // reserved_zero
        }
        // Profiles 0/2 carry implicit 4:2:0 — nothing more to write.
    } else {
        // CS_RGB: §7.2.2 forces full range + 4:4:4 (not coded), legal
        // only for profiles 1/3, where a reserved_zero bit follows.
        if profile == 0 || profile == 2 {
            return Err(Error::InvalidBitstream);
        }
        w.write_bits(0, 1); // reserved_zero
    }
    Ok(())
}

/// §6.2.4 `frame_size()` — 16-bit `frame_*_minus_1` pair.
fn write_frame_size(w: &mut BitWriter, width: u32, height: u32) -> Result<(), Error> {
    if width == 0 || height == 0 || width > (1 << 16) || height > (1 << 16) {
        return Err(Error::Unsupported);
    }
    w.write_bits(width - 1, 16);
    w.write_bits(height - 1, 16);
    Ok(())
}

/// §6.2.7 `read_interpolation_filter( )` inverse. `is_filter_switchable
/// f(1)` set ⇒ `SWITCHABLE`; otherwise `raw_interpolation_filter f(2)`
/// where the §6.2.7 `literal_to_type[ ] = { SMOOTH, EIGHTTAP, SHARP,
/// BILINEAR }` maps the raw value to the filter type — so the writer
/// inverts that table to recover the 2-bit raw value.
fn write_interpolation_filter(w: &mut BitWriter, interpolation_filter: u8) -> Result<(), Error> {
    const SWITCHABLE: u8 = 4;
    // type_to_literal: inverse of literal_to_type = {1, 0, 2, 3}.
    // EIGHTTAP(0)->1, EIGHTTAP_SMOOTH(1)->0, EIGHTTAP_SHARP(2)->2,
    // BILINEAR(3)->3.
    if interpolation_filter == SWITCHABLE {
        w.write_flag(true);
        return Ok(());
    }
    w.write_flag(false);
    let raw = match interpolation_filter {
        0 => 1, // EIGHTTAP
        1 => 0, // EIGHTTAP_SMOOTH
        2 => 2, // EIGHTTAP_SHARP
        3 => 3, // BILINEAR
        _ => return Err(Error::Unsupported),
    };
    w.write_bits(raw, 2);
    Ok(())
}

/// §6.2.3 `render_size()` — inverse of `read_render_size`.
fn write_render_size(
    w: &mut BitWriter,
    frame_w: u32,
    frame_h: u32,
    render_w: u32,
    render_h: u32,
) -> Result<(), Error> {
    if render_w == frame_w && render_h == frame_h {
        w.write_flag(false);
    } else {
        w.write_flag(true);
        if render_w == 0 || render_h == 0 || render_w > (1 << 16) || render_h > (1 << 16) {
            return Err(Error::Unsupported);
        }
        w.write_bits(render_w - 1, 16);
        w.write_bits(render_h - 1, 16);
    }
    Ok(())
}

/// §6.2.8 `loop_filter_params()` — inverse of `read_loop_filter_params`.
fn write_loop_filter_params(w: &mut BitWriter, lf: &LoopFilterParams) {
    w.write_bits(u32::from(lf.level), 6);
    w.write_bits(u32::from(lf.sharpness), 3);
    w.write_flag(lf.delta_enabled);
    if lf.delta_enabled {
        w.write_flag(lf.delta_update);
        if lf.delta_update {
            for slot in lf.ref_deltas.iter() {
                match slot {
                    Some(v) => {
                        w.write_flag(true);
                        w.write_signed(i32::from(*v), 6);
                    }
                    None => w.write_flag(false),
                }
            }
            for slot in lf.mode_deltas.iter() {
                match slot {
                    Some(v) => {
                        w.write_flag(true);
                        w.write_signed(i32::from(*v), 6);
                    }
                    None => w.write_flag(false),
                }
            }
        }
    }
}

/// §6.2.10 `read_delta_q()` inverse: a `delta_coded` flag then `s(4)`.
fn write_delta_q(w: &mut BitWriter, v: i8) {
    if v == 0 {
        w.write_flag(false);
    } else {
        w.write_flag(true);
        w.write_signed(i32::from(v), 4);
    }
}

/// §6.2.9 `quantization_params()` — inverse of `read_quantization_params`.
fn write_quantization_params(w: &mut BitWriter, q: &QuantizationParams) {
    w.write_bits(u32::from(q.base_q_idx), 8);
    write_delta_q(w, q.delta_q_y_dc);
    write_delta_q(w, q.delta_q_uv_dc);
    write_delta_q(w, q.delta_q_uv_ac);
}

/// §6.2.12 `read_prob()` inverse: emit the prob explicitly unless it is
/// the `255` "not coded" sentinel.
fn write_prob(w: &mut BitWriter, p: u8) {
    if p == 255 {
        w.write_flag(false);
    } else {
        w.write_flag(true);
        w.write_bits(u32::from(p), 8);
    }
}

/// §6.2.11 `segmentation_params()` — inverse of `read_segmentation_params`.
fn write_segmentation_params(w: &mut BitWriter, seg: &SegmentationParams) -> Result<(), Error> {
    w.write_flag(seg.enabled);
    if !seg.enabled {
        return Ok(());
    }
    w.write_flag(seg.update_map);
    if seg.update_map {
        let tree = seg.tree_probs.ok_or(Error::InvalidBitstream)?;
        for &p in tree.iter() {
            write_prob(w, p);
        }
        w.write_flag(seg.temporal_update);
        if seg.temporal_update {
            let pred = seg.pred_prob.ok_or(Error::InvalidBitstream)?;
            for &p in pred.iter() {
                write_prob(w, p);
            }
        }
    }
    w.write_flag(seg.update_data);
    if seg.update_data {
        w.write_flag(seg.abs_or_delta_update);
        for i in 0..MAX_SEGMENTS {
            for j in 0..SEG_LVL_MAX {
                let enabled = seg.feature_enabled[i][j];
                w.write_flag(enabled);
                if enabled {
                    let bits = SEGMENTATION_FEATURE_BITS[j];
                    let data = seg.feature_data[i][j];
                    if bits > 0 {
                        let mag = (data as i32).unsigned_abs();
                        if mag >= (1u32 << bits) {
                            return Err(Error::Unsupported);
                        }
                        w.write_bits(mag, bits);
                    }
                    if SEGMENTATION_FEATURE_SIGNED[j] {
                        w.write_flag(data < 0);
                    }
                }
            }
        }
    }
    Ok(())
}

/// §6.2.14 `calc_min_log2_tile_cols` (MAX_TILE_WIDTH_B64 = 64).
fn calc_min_log2_tile_cols(sb64_cols: u32) -> u8 {
    let mut min_log2 = 0u8;
    while (64u32 << min_log2) < sb64_cols {
        min_log2 += 1;
    }
    min_log2
}

/// §6.2.14 `calc_max_log2_tile_cols` (MIN_TILE_WIDTH_B64 = 4).
fn calc_max_log2_tile_cols(sb64_cols: u32) -> u8 {
    let mut max_log2 = 1u8;
    while (sb64_cols >> max_log2) >= 4 {
        max_log2 += 1;
    }
    max_log2 - 1
}

/// §6.2.13 `tile_info()` — inverse of `read_tile_info`. The `tile_cols_log2`
/// increment chain emits one `1` per step above `min_log2`, then a `0`
/// terminator iff still below `max_log2`.
fn write_tile_info(w: &mut BitWriter, frame_width: u32, ti: &TileInfo) -> Result<(), Error> {
    let mi_cols = (frame_width + 7) >> 3;
    let sb64_cols = (mi_cols + 7) >> 3;
    let min_log2 = calc_min_log2_tile_cols(sb64_cols);
    let max_log2 = calc_max_log2_tile_cols(sb64_cols);
    if ti.tile_cols_log2 < min_log2 || ti.tile_cols_log2 > max_log2 || ti.tile_cols_log2 > 6 {
        return Err(Error::Unsupported);
    }
    let mut t = min_log2;
    while t < max_log2 {
        if t < ti.tile_cols_log2 {
            w.write_flag(true);
            t += 1;
        } else {
            w.write_flag(false);
            break;
        }
    }
    // tile_rows_log2 (0..=2): bit0, then an increment bit when bit0 set.
    if ti.tile_rows_log2 > 2 {
        return Err(Error::Unsupported);
    }
    if ti.tile_rows_log2 == 0 {
        w.write_flag(false);
    } else {
        w.write_flag(true);
        w.write_flag(ti.tile_rows_log2 == 2);
    }
    Ok(())
}

/// Write the §6.2 uncompressed header for `hdr` (key-frame branch /
/// `show_existing_frame` sentinel), including the §6.1.1
/// `trailing_bits()` byte-alignment.
///
/// Returns [`Error::Unsupported`] for the not-yet-written inter-frame
/// branch and for field values outside the writable range.
pub(crate) fn write_uncompressed_header(hdr: &Vp9FrameHeader) -> Result<Vec<u8>, Error> {
    let mut w = BitWriter::new();

    // frame_marker f(2) = 2.
    w.write_bits(2, 2);

    // Profile: profile_low_bit f(1), profile_high_bit f(1), then a
    // reserved_zero bit for Profile == 3.
    if hdr.profile > 3 {
        return Err(Error::Unsupported);
    }
    w.write_bits(u32::from(hdr.profile & 1), 1);
    w.write_bits(u32::from((hdr.profile >> 1) & 1), 1);
    if hdr.profile == 3 {
        w.write_bits(0, 1); // reserved_zero
    }

    // show_existing_frame f(1).
    w.write_flag(hdr.show_existing_frame);
    if hdr.show_existing_frame {
        let idx = hdr.frame_to_show_map_idx.ok_or(Error::InvalidBitstream)?;
        w.write_bits(u32::from(idx), 3);
        w.trailing_bits();
        return Ok(w.into_bytes());
    }

    // Intra-only inter frames are not yet written.
    if hdr.frame_type == FrameType::NonKeyFrame && hdr.intra_only {
        return Err(Error::Unsupported);
    }

    let frame_is_intra = hdr.frame_type == FrameType::KeyFrame;

    // frame_type f(1) (0 = key / 1 = non-key), show_frame f(1),
    // error_resilient_mode f(1).
    w.write_bits(u32::from(!frame_is_intra), 1);
    w.write_flag(hdr.show_frame);
    w.write_flag(hdr.error_resilient_mode);

    if frame_is_intra {
        // Key-frame body: frame_sync_code, color_config, frame_size,
        // render_size. (refresh_frame_flags = 0xFF, reset_frame_context =
        // 0, FrameIsIntra = 1 are all implicit — not coded.)
        write_frame_sync_code(&mut w);
        write_color_config(&mut w, hdr.profile, &hdr.color_config)?;
        write_frame_size(&mut w, hdr.frame_width, hdr.frame_height)?;
        write_render_size(
            &mut w,
            hdr.frame_width,
            hdr.frame_height,
            hdr.render_width,
            hdr.render_height,
        )?;
    } else {
        // §6.2 inter (non-intra-only) branch. show_frame == 0 would read
        // an intra_only flag the writer doesn't support yet; require a
        // shown inter frame so the flag is inferred 0.
        if !hdr.show_frame {
            return Err(Error::Unsupported);
        }
        // reset_frame_context f(2) — only present when not error-resilient.
        if !hdr.error_resilient_mode {
            w.write_bits(u32::from(hdr.reset_frame_context & 3), 2);
        }
        // refresh_frame_flags f(8).
        w.write_bits(u32::from(hdr.refresh_frame_flags), 8);
        // for i in 0..3 { ref_frame_idx[i] f(3); ref_frame_sign_bias f(1) }
        let ref_idx = hdr.ref_frame_idx.ok_or(Error::Unsupported)?;
        for (&idx, &bias) in ref_idx.iter().zip(hdr.ref_frame_sign_bias.iter()) {
            if idx > 7 {
                return Err(Error::Unsupported);
            }
            w.write_bits(u32::from(idx), 3);
            w.write_flag(bias);
        }
        // §6.2.5 frame_size_with_refs( ): the writer always codes the
        // explicit-size path (found_ref = 0 for all three references), so
        // every found_ref flag is 0 followed by an explicit frame_size.
        for _ in 0..3 {
            w.write_flag(false);
        }
        write_frame_size(&mut w, hdr.frame_width, hdr.frame_height)?;
        write_render_size(
            &mut w,
            hdr.frame_width,
            hdr.frame_height,
            hdr.render_width,
            hdr.render_height,
        )?;
        // allow_high_precision_mv f(1); read_interpolation_filter().
        w.write_flag(hdr.allow_high_precision_mv);
        write_interpolation_filter(&mut w, hdr.interpolation_filter)?;
    }

    // §6.2 tail. For a key frame error_resilient_mode gates
    // refresh_frame_context + frame_parallel_decoding_mode, and
    // frame_context_idx is coded then reset to 0 by the parser.
    if !hdr.error_resilient_mode {
        w.write_flag(hdr.refresh_frame_context);
        w.write_flag(hdr.frame_parallel_decoding_mode);
    }
    // frame_context_idx f(2): the parser forces it to 0 for intra
    // frames regardless, so any value round-trips; emit the stored one.
    w.write_bits(u32::from(hdr.frame_context_idx), 2);

    write_loop_filter_params(&mut w, &hdr.loop_filter);
    write_quantization_params(&mut w, &hdr.quantization);
    write_segmentation_params(&mut w, &hdr.segmentation)?;
    write_tile_info(&mut w, hdr.frame_width, &hdr.tile_info)?;

    // header_size_in_bytes f(16).
    w.write_bits(u32::from(hdr.header_size_in_bytes), 16);

    // §6.1.1 trailing_bits() zero-fill to byte boundary.
    w.trailing_bits();
    Ok(w.into_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::header::{parse_uncompressed_header, ColorSpace};

    #[test]
    fn bitwriter_roundtrips_through_bit_layout() {
        let mut w = BitWriter::new();
        w.write_bits(0b101, 3);
        w.write_flag(true);
        w.write_signed(-5, 4);
        w.write_signed(7, 4);
        w.trailing_bits();
        let bytes = w.into_bytes();
        // 3 + 1 + 5 + 5 = 14 bits -> 2 bytes.
        assert_eq!(bytes.len(), 2);
    }

    fn minimal_keyframe_header() -> Vp9FrameHeader {
        Vp9FrameHeader {
            profile: 0,
            show_existing_frame: false,
            frame_to_show_map_idx: None,
            frame_type: FrameType::KeyFrame,
            show_frame: true,
            error_resilient_mode: false,
            intra_only: false,
            color_config: ColorConfig {
                bit_depth: 8,
                color_space: ColorSpace::Bt601,
                color_range_full: false,
                subsampling_x: true,
                subsampling_y: true,
            },
            frame_width: 64,
            frame_height: 64,
            render_width: 64,
            render_height: 64,
            reset_frame_context: 0,
            refresh_frame_flags: 0xFF,
            refresh_frame_context: true,
            frame_parallel_decoding_mode: false,
            frame_context_idx: 0,
            loop_filter: LoopFilterParams {
                level: 10,
                sharpness: 0,
                delta_enabled: true,
                delta_update: false,
                ref_deltas: [None; 4],
                mode_deltas: [None; 2],
            },
            quantization: QuantizationParams {
                base_q_idx: 64,
                delta_q_y_dc: 0,
                delta_q_uv_dc: 0,
                delta_q_uv_ac: 0,
                lossless: false,
            },
            segmentation: SegmentationParams::default_disabled(),
            tile_info: TileInfo {
                tile_cols_log2: 0,
                tile_rows_log2: 0,
            },
            header_size_in_bytes: 5,
            uncompressed_header_size_bytes: 0,
            ref_frame_idx: None,
            ref_frame_sign_bias: [false; 3],
            allow_high_precision_mv: false,
            interpolation_filter: 0,
        }
    }

    /// A written key-frame header parses back to the same fields.
    fn assert_roundtrips(hdr: &Vp9FrameHeader) {
        let bytes = write_uncompressed_header(hdr).expect("write header");
        let parsed = parse_uncompressed_header(&bytes).expect("parse header");
        assert_eq!(parsed.profile, hdr.profile);
        assert_eq!(parsed.frame_type, hdr.frame_type);
        assert_eq!(parsed.show_frame, hdr.show_frame);
        assert_eq!(parsed.error_resilient_mode, hdr.error_resilient_mode);
        assert_eq!(parsed.color_config, hdr.color_config);
        assert_eq!(parsed.frame_width, hdr.frame_width);
        assert_eq!(parsed.frame_height, hdr.frame_height);
        assert_eq!(parsed.render_width, hdr.render_width);
        assert_eq!(parsed.render_height, hdr.render_height);
        assert_eq!(parsed.refresh_frame_flags, 0xFF);
        assert_eq!(parsed.refresh_frame_context, hdr.refresh_frame_context);
        assert_eq!(
            parsed.frame_parallel_decoding_mode,
            hdr.frame_parallel_decoding_mode
        );
        assert_eq!(parsed.loop_filter, hdr.loop_filter);
        assert_eq!(parsed.quantization.base_q_idx, hdr.quantization.base_q_idx);
        assert_eq!(parsed.segmentation, hdr.segmentation);
        assert_eq!(parsed.tile_info, hdr.tile_info);
        assert_eq!(parsed.header_size_in_bytes, hdr.header_size_in_bytes);
        // The header is byte-aligned by trailing_bits.
        assert_eq!(parsed.uncompressed_header_size_bytes, bytes.len());
    }

    #[test]
    fn minimal_keyframe_roundtrips() {
        assert_roundtrips(&minimal_keyframe_header());
    }

    #[test]
    fn keyframe_with_render_override_roundtrips() {
        let mut h = minimal_keyframe_header();
        h.frame_width = 1920;
        h.frame_height = 1080;
        h.render_width = 1280;
        h.render_height = 720;
        assert_roundtrips(&h);
    }

    #[test]
    fn keyframe_error_resilient_roundtrips() {
        let mut h = minimal_keyframe_header();
        h.error_resilient_mode = true;
        // With error_resilient_mode the tail flags are not coded; the
        // parser yields refresh_frame_context = false, parallel = true.
        h.refresh_frame_context = false;
        h.frame_parallel_decoding_mode = true;
        assert_roundtrips(&h);
    }

    #[test]
    fn keyframe_with_loop_filter_deltas_roundtrips() {
        let mut h = minimal_keyframe_header();
        h.loop_filter = LoopFilterParams {
            level: 32,
            sharpness: 3,
            delta_enabled: true,
            delta_update: true,
            ref_deltas: [Some(1), None, Some(-2), None],
            mode_deltas: [Some(0), Some(-1)],
        };
        assert_roundtrips(&h);
    }

    #[test]
    fn keyframe_with_quant_deltas_roundtrips() {
        let mut h = minimal_keyframe_header();
        h.quantization = QuantizationParams {
            base_q_idx: 0,
            delta_q_y_dc: -3,
            delta_q_uv_dc: 2,
            delta_q_uv_ac: -1,
            lossless: false,
        };
        assert_roundtrips(&h);
    }

    #[test]
    fn keyframe_with_segmentation_roundtrips() {
        let mut h = minimal_keyframe_header();
        let mut seg = SegmentationParams::default_disabled();
        seg.enabled = true;
        seg.update_map = true;
        seg.tree_probs = Some([200, 100, 255, 32, 64, 128, 255]);
        seg.temporal_update = false;
        // The parser populates pred_prob with the 255 "not coded"
        // sentinel whenever update_map is set, even with
        // temporal_update == 0 (no bits read); mirror that so the
        // round-trip compares equal.
        seg.pred_prob = Some([255; 3]);
        seg.update_data = true;
        seg.abs_or_delta_update = true;
        // SEG_LVL_ALT_Q (j=0, signed 8-bit), SEG_LVL_ALT_L (j=1, signed 6),
        // SEG_LVL_REF_FRAME (j=2, 2-bit unsigned), SEG_LVL_SKIP (j=3, flag).
        seg.feature_enabled[0][0] = true;
        seg.feature_data[0][0] = -50;
        seg.feature_enabled[1][1] = true;
        seg.feature_data[1][1] = 12;
        seg.feature_enabled[2][2] = true;
        seg.feature_data[2][2] = 3;
        seg.feature_enabled[3][3] = true;
        seg.feature_data[3][3] = 0;
        h.segmentation = seg;
        assert_roundtrips(&h);
    }

    #[test]
    fn keyframe_with_temporal_segmentation_roundtrips() {
        let mut h = minimal_keyframe_header();
        let mut seg = SegmentationParams::default_disabled();
        seg.enabled = true;
        seg.update_map = true;
        seg.tree_probs = Some([10, 20, 30, 40, 50, 60, 70]);
        seg.temporal_update = true;
        seg.pred_prob = Some([128, 200, 255]);
        seg.update_data = false;
        h.segmentation = seg;
        assert_roundtrips(&h);
    }

    #[test]
    fn keyframe_4k_tile_cols_roundtrips() {
        let mut h = minimal_keyframe_header();
        h.frame_width = 3840;
        h.frame_height = 2160;
        h.render_width = 3840;
        h.render_height = 2160;
        h.tile_info = TileInfo {
            tile_cols_log2: 2,
            tile_rows_log2: 1,
        };
        assert_roundtrips(&h);
    }

    #[test]
    fn profile2_10bit_roundtrips() {
        let mut h = minimal_keyframe_header();
        h.profile = 2;
        h.color_config = ColorConfig {
            bit_depth: 10,
            color_space: ColorSpace::Bt709,
            color_range_full: false,
            subsampling_x: true,
            subsampling_y: true,
        };
        assert_roundtrips(&h);
    }

    #[test]
    fn profile3_rgb_444_roundtrips() {
        let mut h = minimal_keyframe_header();
        h.profile = 3;
        h.color_config = ColorConfig {
            bit_depth: 10,
            color_space: ColorSpace::Rgb,
            color_range_full: true,
            subsampling_x: false,
            subsampling_y: false,
        };
        assert_roundtrips(&h);
    }

    #[test]
    fn show_existing_frame_sentinel_roundtrips() {
        let mut h = minimal_keyframe_header();
        h.show_existing_frame = true;
        h.frame_to_show_map_idx = Some(5);
        let bytes = write_uncompressed_header(&h).expect("write");
        let parsed = parse_uncompressed_header(&bytes).expect("parse");
        assert!(parsed.show_existing_frame);
        assert_eq!(parsed.frame_to_show_map_idx, Some(5));
    }

    fn minimal_inter_header() -> Vp9FrameHeader {
        let mut h = minimal_keyframe_header();
        h.frame_type = FrameType::NonKeyFrame;
        h.refresh_frame_flags = 0x01;
        h.ref_frame_idx = Some([0, 1, 2]);
        h.ref_frame_sign_bias = [false, false, true];
        h.allow_high_precision_mv = false;
        h.interpolation_filter = 0; // EIGHTTAP
        h
    }

    /// Round-trip an inter header through the `_with_refs` parser. The
    /// writer codes the explicit-frame-size path, so `ref_dims` is unused,
    /// but the parser still requires a `ref_state` to be present.
    fn assert_inter_roundtrips(h: &Vp9FrameHeader) {
        use crate::header::{parse_uncompressed_header_with_refs, RefFrameState};
        let bytes = write_uncompressed_header(h).expect("write inter");
        let dims = [(h.frame_width, h.frame_height); 8];
        let state = RefFrameState {
            ref_dims: &dims,
            color_config: h.color_config,
        };
        let parsed = parse_uncompressed_header_with_refs(&bytes, Some(state)).expect("parse inter");
        assert_eq!(parsed.frame_type, FrameType::NonKeyFrame);
        assert_eq!(parsed.frame_width, h.frame_width);
        assert_eq!(parsed.frame_height, h.frame_height);
        assert_eq!(parsed.refresh_frame_flags, h.refresh_frame_flags);
        assert_eq!(parsed.ref_frame_idx, h.ref_frame_idx);
        assert_eq!(parsed.ref_frame_sign_bias, h.ref_frame_sign_bias);
        assert_eq!(parsed.allow_high_precision_mv, h.allow_high_precision_mv);
        assert_eq!(parsed.interpolation_filter, h.interpolation_filter);
        assert_eq!(parsed.reset_frame_context, h.reset_frame_context);
        assert_eq!(parsed.frame_context_idx, h.frame_context_idx);
    }

    #[test]
    fn inter_frame_minimal_roundtrips() {
        assert_inter_roundtrips(&minimal_inter_header());
    }

    #[test]
    fn inter_frame_high_precision_mv_roundtrips() {
        let mut h = minimal_inter_header();
        h.allow_high_precision_mv = true;
        assert_inter_roundtrips(&h);
    }

    #[test]
    fn inter_frame_switchable_filter_roundtrips() {
        let mut h = minimal_inter_header();
        h.interpolation_filter = 4; // SWITCHABLE
        assert_inter_roundtrips(&h);
    }

    #[test]
    fn inter_frame_all_filters_roundtrip() {
        for filt in [0u8, 1, 2, 3, 4] {
            let mut h = minimal_inter_header();
            h.interpolation_filter = filt;
            assert_inter_roundtrips(&h);
        }
    }

    #[test]
    fn inter_frame_reset_context_roundtrips() {
        let mut h = minimal_inter_header();
        h.reset_frame_context = 2;
        h.frame_context_idx = 1;
        assert_inter_roundtrips(&h);
    }

    #[test]
    fn inter_frame_error_resilient_roundtrips() {
        let mut h = minimal_inter_header();
        h.error_resilient_mode = true;
        // error_resilient forces frame_context_idx to 0 in the parser.
        h.frame_context_idx = 0;
        assert_inter_roundtrips(&h);
    }

    #[test]
    fn inter_intra_only_is_unsupported() {
        let mut h = minimal_inter_header();
        h.intra_only = true;
        h.show_frame = false;
        assert_eq!(
            write_uncompressed_header(&h).unwrap_err(),
            Error::Unsupported
        );
    }
}
