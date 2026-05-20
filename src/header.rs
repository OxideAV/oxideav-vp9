//! VP9 uncompressed-header structural walker.
//!
//! Implements the syntax tree of VP9 spec v0.7 §6.2 only far enough to
//! land the fields enumerated in round-1 scope:
//!
//! * `profile`, `show_existing_frame` / `frame_to_show_map_idx`,
//!   `frame_type`, `show_frame`, `error_resilient_mode`,
//! * `color_config` (bit depth, color space, color range, chroma
//!   subsampling),
//! * `frame_size` and `render_size`.
//!
//! Inter-frame paths (`frame_size_with_refs`, motion-vector flags,
//! interpolation filter) and everything past the post-color-config
//! section of §6.2 (loop_filter_params, quantization_params, segmentation,
//! tile_info, header_size_in_bytes, trailing bits, compressed header)
//! are intentionally NOT walked yet.

use crate::bitreader::BitReader;
use crate::Error;

/// Frame type per spec §7.2 table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameType {
    /// `KEY_FRAME` (`frame_type == 0`).
    KeyFrame,
    /// `NON_KEY_FRAME` (`frame_type == 1`). May further qualify as
    /// intra-only via [`Vp9FrameHeader::intra_only`].
    NonKeyFrame,
}

/// `color_space` enumeration per spec §7.2.2 (3-bit field).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorSpace {
    /// `0` — `CS_UNKNOWN`: color space signalled out-of-band.
    Unknown,
    /// `1` — `CS_BT_601`.
    Bt601,
    /// `2` — `CS_BT_709`.
    Bt709,
    /// `3` — `CS_SMPTE_170`.
    Smpte170,
    /// `4` — `CS_SMPTE_240`.
    Smpte240,
    /// `5` — `CS_BT_2020`.
    Bt2020,
    /// `6` — `CS_RESERVED`.
    Reserved,
    /// `7` — `CS_RGB` (sRGB). Only legal when `Profile >= 1`.
    Rgb,
}

impl ColorSpace {
    fn from_bits(bits: u32) -> Self {
        match bits & 0b111 {
            0 => Self::Unknown,
            1 => Self::Bt601,
            2 => Self::Bt709,
            3 => Self::Smpte170,
            4 => Self::Smpte240,
            5 => Self::Bt2020,
            6 => Self::Reserved,
            7 => Self::Rgb,
            _ => unreachable!(),
        }
    }
}

/// Color configuration walked out of `color_config()` (spec §6.2.2 /
/// §7.2.2). Defaults match the intra-only `Profile == 0` fall-through
/// (`CS_BT_601`, 4:2:0, 8-bit).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColorConfig {
    /// Bit depth per sample (8, 10 or 12).
    pub bit_depth: u8,
    /// Decoded `color_space` field.
    pub color_space: ColorSpace,
    /// `color_range` field. `true` = full swing; `false` = studio
    /// swing. Forced to full swing for RGB (spec §6.2.2).
    pub color_range_full: bool,
    /// `subsampling_x`. Defaults to `1` for profiles 0/2.
    pub subsampling_x: bool,
    /// `subsampling_y`. Defaults to `1` for profiles 0/2.
    pub subsampling_y: bool,
}

impl ColorConfig {
    /// Default color config the spec installs in the inter-frame
    /// `intra_only && Profile == 0` branch (§6.2): BT.601, 4:2:0, 8-bit.
    const fn default_intra_only_profile0() -> Self {
        Self {
            bit_depth: 8,
            color_space: ColorSpace::Bt601,
            color_range_full: false,
            subsampling_x: true,
            subsampling_y: true,
        }
    }
}

/// Round-1 view of the VP9 uncompressed header.
///
/// Fields populated correspond to spec §6.2 entries the round-1 walker
/// reaches. Inter-frame motion-vector flags, the post-color-config
/// section (loop filter / quantizer / segmentation / tile info /
/// header_size_in_bytes), trailing bits, and the compressed header are
/// out of scope and intentionally NOT exposed here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Vp9FrameHeader {
    /// `Profile` per §7.2 (0..=3).
    pub profile: u8,
    /// `show_existing_frame` flag.
    pub show_existing_frame: bool,
    /// Reference-buffer index of the frame to display, when
    /// `show_existing_frame == 1`. `None` otherwise.
    pub frame_to_show_map_idx: Option<u8>,
    /// `frame_type`. Only meaningful when `show_existing_frame == 0`.
    pub frame_type: FrameType,
    /// `show_frame` flag.
    pub show_frame: bool,
    /// `error_resilient_mode` flag.
    pub error_resilient_mode: bool,
    /// `intra_only` (only set in the inter-frame branch where
    /// `show_frame == 0`; otherwise `false` per §6.2 fall-through).
    pub intra_only: bool,
    /// Color configuration (bit depth / color space / range /
    /// subsampling), per `color_config()` walk.
    pub color_config: ColorConfig,
    /// `FrameWidth = frame_width_minus_1 + 1`.
    pub frame_width: u32,
    /// `FrameHeight = frame_height_minus_1 + 1`.
    pub frame_height: u32,
    /// Decoded `renderWidth` (= `FrameWidth` if
    /// `render_and_frame_size_different == 0`).
    pub render_width: u32,
    /// Decoded `renderHeight`.
    pub render_height: u32,
}

/// Parse a VP9 uncompressed header from `data`.
///
/// The walker stops after `render_size()` (or after
/// `frame_to_show_map_idx` for the `show_existing_frame == 1` early
/// return) — the round-1 scope. Returns [`Error::UnexpectedEof`] on
/// truncated input and [`Error::InvalidBitstream`] when a "shall be
/// equal to" constraint from §7.2 is violated.
///
/// For the round-1 cut, inter-frame headers — i.e. `frame_type ==
/// NON_KEY_FRAME` with `show_frame == 1` — are not yet walked: that
/// path needs `frame_size_with_refs` plus reference-frame state, which
/// lands in a later round. Such inputs return
/// [`Error::Unsupported`].
pub fn parse_uncompressed_header(data: &[u8]) -> Result<Vp9FrameHeader, Error> {
    let mut br = BitReader::new(data);

    // frame_marker — spec §7.2: "shall be equal to 2".
    let frame_marker = br.read_bits(2)?;
    if frame_marker != 2 {
        return Err(Error::InvalidBitstream);
    }

    // Profile = (profile_high_bit << 1) | profile_low_bit, then a
    // reserved_zero bit only for Profile == 3.
    let profile_low_bit = br.read_bits(1)?;
    let profile_high_bit = br.read_bits(1)?;
    let profile = ((profile_high_bit << 1) | profile_low_bit) as u8;
    if profile == 3 {
        let reserved_zero = br.read_bits(1)?;
        if reserved_zero != 0 {
            return Err(Error::InvalidBitstream);
        }
    }

    let show_existing_frame = br.read_flag()?;
    if show_existing_frame {
        let frame_to_show_map_idx = br.read_bits(3)? as u8;
        return Ok(Vp9FrameHeader {
            profile,
            show_existing_frame: true,
            frame_to_show_map_idx: Some(frame_to_show_map_idx),
            // The remaining fields are unused on the
            // show_existing_frame == 1 path; populate with safe
            // defaults rather than expose Option<…> across the board.
            frame_type: FrameType::NonKeyFrame,
            show_frame: true,
            error_resilient_mode: false,
            intra_only: false,
            color_config: ColorConfig::default_intra_only_profile0(),
            frame_width: 0,
            frame_height: 0,
            render_width: 0,
            render_height: 0,
        });
    }

    let frame_type = if br.read_bits(1)? == 0 {
        FrameType::KeyFrame
    } else {
        FrameType::NonKeyFrame
    };
    let show_frame = br.read_flag()?;
    let error_resilient_mode = br.read_flag()?;

    let (intra_only, color_config, frame_width, frame_height, render_width, render_height) =
        match frame_type {
            FrameType::KeyFrame => {
                // frame_sync_code(): three required bytes.
                read_frame_sync_code(&mut br)?;
                let color_config = read_color_config(&mut br, profile)?;
                let (frame_width, frame_height) = read_frame_size(&mut br)?;
                let (render_width, render_height) =
                    read_render_size(&mut br, frame_width, frame_height)?;
                (
                    false,
                    color_config,
                    frame_width,
                    frame_height,
                    render_width,
                    render_height,
                )
            }
            FrameType::NonKeyFrame => {
                // §6.2 inter-frame branch: intra_only is read only
                // when show_frame == 0, otherwise inferred as 0.
                let intra_only = if !show_frame { br.read_flag()? } else { false };
                if !intra_only {
                    // Inter (non-intra-only) frames need
                    // frame_size_with_refs + a reference-frame buffer,
                    // which is out of round-1 scope.
                    return Err(Error::Unsupported);
                }

                // reset_frame_context is consumed-but-not-exposed
                // when error_resilient_mode == 0.
                if !error_resilient_mode {
                    let _reset_frame_context = br.read_bits(2)?;
                }

                // Intra-only branch: frame_sync_code, then
                // color_config() only if Profile > 0 — for Profile 0
                // the spec installs CS_BT_601 / 4:2:0 / 8-bit
                // defaults.
                read_frame_sync_code(&mut br)?;
                let color_config = if profile > 0 {
                    read_color_config(&mut br, profile)?
                } else {
                    ColorConfig::default_intra_only_profile0()
                };

                // refresh_frame_flags — consumed but not exposed in
                // round 1 (its consumers, the reference-buffer
                // refresh logic, land later).
                let _refresh_frame_flags = br.read_bits(8)?;
                let (frame_width, frame_height) = read_frame_size(&mut br)?;
                let (render_width, render_height) =
                    read_render_size(&mut br, frame_width, frame_height)?;
                (
                    true,
                    color_config,
                    frame_width,
                    frame_height,
                    render_width,
                    render_height,
                )
            }
        };

    Ok(Vp9FrameHeader {
        profile,
        show_existing_frame: false,
        frame_to_show_map_idx: None,
        frame_type,
        show_frame,
        error_resilient_mode,
        intra_only,
        color_config,
        frame_width,
        frame_height,
        render_width,
        render_height,
    })
}

fn read_frame_sync_code(br: &mut BitReader<'_>) -> Result<(), Error> {
    // §7.2.1: bytes shall be 0x49 / 0x83 / 0x42.
    let b0 = br.read_bits(8)?;
    let b1 = br.read_bits(8)?;
    let b2 = br.read_bits(8)?;
    if b0 != 0x49 || b1 != 0x83 || b2 != 0x42 {
        return Err(Error::InvalidBitstream);
    }
    Ok(())
}

fn read_color_config(br: &mut BitReader<'_>, profile: u8) -> Result<ColorConfig, Error> {
    let bit_depth = if profile >= 2 {
        let ten_or_twelve_bit = br.read_bits(1)?;
        if ten_or_twelve_bit == 1 {
            12
        } else {
            10
        }
    } else {
        8
    };

    let color_space = ColorSpace::from_bits(br.read_bits(3)?);
    let color_range_full;
    let mut subsampling_x = true;
    let mut subsampling_y = true;

    if !matches!(color_space, ColorSpace::Rgb) {
        color_range_full = br.read_flag()?;
        if profile == 1 || profile == 3 {
            subsampling_x = br.read_flag()?;
            subsampling_y = br.read_flag()?;
            // Profile 1/3 4:2:0 is prohibited by §7.2.2: "either
            // subsampling_x is equal to 0 or subsampling_y is equal
            // to 0 when profile_low_bit is equal to 1".
            if (profile & 1) == 1 && subsampling_x && subsampling_y {
                return Err(Error::InvalidBitstream);
            }
            let reserved_zero = br.read_bits(1)?;
            if reserved_zero != 0 {
                return Err(Error::InvalidBitstream);
            }
        }
        // Profiles 0 / 2: defaults already set to 4:2:0.
    } else {
        // CS_RGB: §7.2.2 forces color_range = full, 4:4:4 chroma.
        color_range_full = true;
        subsampling_x = false;
        subsampling_y = false;
        // §7.2.2: "It is a requirement of bitstream conformance that
        // color_space is not equal to CS_RGB when profile_low_bit is
        // equal to 0." profile_low_bit == 0 means profiles 0 and 2.
        if profile == 0 || profile == 2 {
            return Err(Error::InvalidBitstream);
        }
        if profile == 1 || profile == 3 {
            let reserved_zero = br.read_bits(1)?;
            if reserved_zero != 0 {
                return Err(Error::InvalidBitstream);
            }
        }
    }

    Ok(ColorConfig {
        bit_depth,
        color_space,
        color_range_full,
        subsampling_x,
        subsampling_y,
    })
}

fn read_frame_size(br: &mut BitReader<'_>) -> Result<(u32, u32), Error> {
    let frame_width_minus_1 = br.read_bits(16)?;
    let frame_height_minus_1 = br.read_bits(16)?;
    Ok((frame_width_minus_1 + 1, frame_height_minus_1 + 1))
}

fn read_render_size(
    br: &mut BitReader<'_>,
    frame_width: u32,
    frame_height: u32,
) -> Result<(u32, u32), Error> {
    let render_and_frame_size_different = br.read_flag()?;
    if render_and_frame_size_different {
        let render_width_minus_1 = br.read_bits(16)?;
        let render_height_minus_1 = br.read_bits(16)?;
        Ok((render_width_minus_1 + 1, render_height_minus_1 + 1))
    } else {
        Ok((frame_width, frame_height))
    }
}
