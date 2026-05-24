//! VP9 residual driver per spec v0.7 §6.4.21 (intra path).
//!
//! Round 14 lands the §6.4.21 `residual( )` outer loop — the per-plane,
//! per-4x4-sub-block walk that owns the `AboveNonzeroContext` /
//! `LeftNonzeroContext` write-back across a whole MI block, drives the
//! round-13 §6.4.24 [`crate::tokens::tokens`] per-block decode, and
//! feeds the round-11 §8.6.2 [`crate::reconstruct::reconstruct_block`]
//! with real per-block `Tokens` arrays, availability flags and
//! plane/quantizer state.
//!
//! Scope: the **intra** path only. The §6.4.21 listing branches on
//! `is_inter` at the top: when `is_inter == 1` it calls
//! `predict_inter( )` for the whole block (or per-4x4 sub-block) before
//! running the per-block residual loop. `predict_inter( )` requires
//! reference-buffer state (the §6.3.9+ inter-only syntax and the §8.5.2
//! inter prediction process), neither of which has landed yet — so the
//! inter path is deferred to a later round. The intra side, which the
//! round-10 [`crate::intra::predict_intra`] already supports, is the
//! immediate increment this round closes.
//!
//! The pieces this module exposes:
//!
//! * §10.2 [`BLOCK_SIZES`] / [`BLOCK_INVALID`] and the
//!   `num_4x4_blocks_wide_lookup` / `num_4x4_blocks_high_lookup` /
//!   `max_txsize_lookup` lookup tables, plus the §6.4.23
//!   `ss_size_lookup` and [`get_plane_block_size`] / [`get_uv_tx_size`]
//!   helpers — all transcribed verbatim from §10.2 and §6.4.22 /
//!   §6.4.23 of the spec.
//! * [`ResidualBlockCtx`] — the per-MI-block / per-frame state the
//!   driver consumes (`MiCol`, `MiRow`, `MiSize`, `tx_size`,
//!   `subsampling_x` / `y`, `MiCols` / `MiRows`, `skip`, `lossless`,
//!   `bit_depth`, the per-block `PredMode` and the per-plane DC/AC
//!   quantizers).
//! * [`PlaneBuffers`] — the three `CurrFrame[ plane ]` buffers the
//!   residual driver writes into.
//! * [`residual_intra`] — the §6.4.21 driver proper for an intra block:
//!   per plane, walks the `num4x4w × num4x4h` 4x4 grid with `step = 1
//!   << txSz`, calls [`crate::intra::predict_intra`] for the block
//!   (within the plane bounds), runs the round-13
//!   [`crate::tokens::tokens`] driver (when `!skip`), and feeds the
//!   per-block `Tokens` array into the round-11
//!   [`crate::reconstruct::reconstruct_block`] — then updates the
//!   `AboveNonzeroContext` / `LeftNonzeroContext` strips per
//!   §6.4.21's trailing write-back.
//!
//! `residual_intra` is wired against a per-block `TokenSource`
//! callback (rather than a fixed `Vec<Tokens>` bag) so the test suite
//! can supply hand-crafted token buffers without yet pulling in the
//! per-block mode-info decode (which is owned by the deferred §6.4 /
//! §7.4 mode-info round). A production caller threads the real
//! per-block `BoolCoder` token decode through the same hook.
//!
//! Provenance: VP9 Bitstream & Decoding Process Specification v0.7
//! (`docs/video/vp9/vp9-spec.txt` §6.4.21 / §6.4.22 / §6.4.23 / §10.2
//! / §3 `BLOCK_SIZES`). No external library source was consulted;
//! every formula is transcribed directly from the spec listings.

// Several helpers in this module are exercised exclusively from
// `#[cfg(test)]` until the per-frame public decode path lands (the
// per-block mode-info decode is the missing piece).
#![allow(dead_code)]

use crate::intra::{predict_intra, Plane, PredMode};
use crate::reconstruct::reconstruct_block;
use crate::tokens::NonzeroContext;

// ----- §3 BLOCK_SIZES constants -----

/// `BLOCK_4X4` — `subsize == 0` per §7.4.3.
pub(crate) const BLOCK_4X4: u8 = 0;
/// `BLOCK_4X8` — `subsize == 1`.
pub(crate) const BLOCK_4X8: u8 = 1;
/// `BLOCK_8X4` — `subsize == 2`.
pub(crate) const BLOCK_8X4: u8 = 2;
/// `BLOCK_8X8` — `subsize == 3`.
pub(crate) const BLOCK_8X8: u8 = 3;
/// `BLOCK_8X16` — `subsize == 4`.
pub(crate) const BLOCK_8X16: u8 = 4;
/// `BLOCK_16X8` — `subsize == 5`.
pub(crate) const BLOCK_16X8: u8 = 5;
/// `BLOCK_16X16` — `subsize == 6`.
pub(crate) const BLOCK_16X16: u8 = 6;
/// `BLOCK_16X32` — `subsize == 7`.
pub(crate) const BLOCK_16X32: u8 = 7;
/// `BLOCK_32X16` — `subsize == 8`.
pub(crate) const BLOCK_32X16: u8 = 8;
/// `BLOCK_32X32` — `subsize == 9`.
pub(crate) const BLOCK_32X32: u8 = 9;
/// `BLOCK_32X64` — `subsize == 10`.
pub(crate) const BLOCK_32X64: u8 = 10;
/// `BLOCK_64X32` — `subsize == 11`.
pub(crate) const BLOCK_64X32: u8 = 11;
/// `BLOCK_64X64` — `subsize == 12`.
pub(crate) const BLOCK_64X64: u8 = 12;

/// `BLOCK_INVALID = 14` — sentinel marking illegal partition choices,
/// per §3.
pub(crate) const BLOCK_INVALID: u8 = 14;

/// `BLOCK_SIZES = 13` per §3 — the count of valid `subsize` values.
pub(crate) const BLOCK_SIZES: usize = 13;

// ----- §10.2 lookup tables (verbatim transcription) -----

/// `num_4x4_blocks_wide_lookup[ BLOCK_SIZES ]` per §10.2 — width of each
/// block size in 4-sample units.
pub(crate) const NUM_4X4_BLOCKS_WIDE_LOOKUP: [u32; BLOCK_SIZES] =
    [1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16];

/// `num_4x4_blocks_high_lookup[ BLOCK_SIZES ]` per §10.2 — height of
/// each block size in 4-sample units.
pub(crate) const NUM_4X4_BLOCKS_HIGH_LOOKUP: [u32; BLOCK_SIZES] =
    [1, 2, 1, 2, 4, 2, 4, 8, 4, 8, 16, 8, 16];

/// `max_txsize_lookup[ BLOCK_SIZES ]` per §6.4.10 — maximum `tx_size`
/// permitted for each block size. Values are §3 `TX_*` indices
/// (`TX_4X4 = 0`, `TX_8X8 = 1`, `TX_16X16 = 2`, `TX_32X32 = 3`).
pub(crate) const MAX_TXSIZE_LOOKUP: [u32; BLOCK_SIZES] = [
    0, 0, 0, // BLOCK_4X4, BLOCK_4X8, BLOCK_8X4 -> TX_4X4
    1, 1, 1, // BLOCK_8X8, BLOCK_8X16, BLOCK_16X8 -> TX_8X8
    2, 2, 2, // BLOCK_16X16, BLOCK_16X32, BLOCK_32X16 -> TX_16X16
    3, 3, 3, 3, // BLOCK_32X32, BLOCK_32X64, BLOCK_64X32, BLOCK_64X64 -> TX_32X32
];

/// `ss_size_lookup[ BLOCK_SIZES ][ 2 ][ 2 ]` per §6.4.23 — maps a
/// `(subsize, subsampling_x, subsampling_y)` triple to the chroma plane
/// block size. `BLOCK_INVALID` marks combinations that violate the
/// §7.4.3 conformance rule.
pub(crate) const SS_SIZE_LOOKUP: [[[u8; 2]; 2]; BLOCK_SIZES] = [
    // BLOCK_4X4
    [[BLOCK_4X4, BLOCK_INVALID], [BLOCK_INVALID, BLOCK_INVALID]],
    // BLOCK_4X8
    [[BLOCK_4X8, BLOCK_4X4], [BLOCK_INVALID, BLOCK_INVALID]],
    // BLOCK_8X4
    [[BLOCK_8X4, BLOCK_INVALID], [BLOCK_4X4, BLOCK_INVALID]],
    // BLOCK_8X8
    [[BLOCK_8X8, BLOCK_8X4], [BLOCK_4X8, BLOCK_4X4]],
    // BLOCK_8X16
    [[BLOCK_8X16, BLOCK_8X8], [BLOCK_INVALID, BLOCK_4X8]],
    // BLOCK_16X8
    [[BLOCK_16X8, BLOCK_INVALID], [BLOCK_8X8, BLOCK_8X4]],
    // BLOCK_16X16
    [[BLOCK_16X16, BLOCK_16X8], [BLOCK_8X16, BLOCK_8X8]],
    // BLOCK_16X32
    [[BLOCK_16X32, BLOCK_16X16], [BLOCK_INVALID, BLOCK_8X16]],
    // BLOCK_32X16
    [[BLOCK_32X16, BLOCK_INVALID], [BLOCK_16X16, BLOCK_16X8]],
    // BLOCK_32X32
    [[BLOCK_32X32, BLOCK_32X16], [BLOCK_16X32, BLOCK_16X16]],
    // BLOCK_32X64
    [[BLOCK_32X64, BLOCK_32X32], [BLOCK_INVALID, BLOCK_16X32]],
    // BLOCK_64X32
    [[BLOCK_64X32, BLOCK_INVALID], [BLOCK_32X32, BLOCK_32X16]],
    // BLOCK_64X64
    [[BLOCK_64X64, BLOCK_64X32], [BLOCK_32X64, BLOCK_32X32]],
];

/// `get_plane_block_size( subsize, plane )` per §6.4.23.
///
/// `subsize` is the §7.4.3 MI block size; `subsampling_x` /
/// `subsampling_y` come from the color config. Returns the chroma plane's
/// block size or [`BLOCK_INVALID`] when the §7.4.3 conformance rule is
/// violated (luma always returns `subsize` since `ss_size_lookup[*][0][0]
/// == subsize`).
pub(crate) fn get_plane_block_size(
    subsize: u8,
    plane: usize,
    subsampling_x: bool,
    subsampling_y: bool,
) -> u8 {
    let subx = if plane > 0 && subsampling_x { 1 } else { 0 };
    let suby = if plane > 0 && subsampling_y { 1 } else { 0 };
    SS_SIZE_LOOKUP[subsize as usize][subx][suby]
}

/// `get_uv_tx_size( )` per §6.4.22.
///
/// Returns the chroma `txSz` for an MI block, capped by the chroma
/// plane's `max_txsize_lookup` and forced to `TX_4X4` for sub-8x8
/// blocks. `tx_size` is the luma `txSz`; `mi_size` the §7.4.3 MI block
/// size; `subsampling_x` / `subsampling_y` come from the color config.
pub(crate) fn get_uv_tx_size(
    tx_size: u32,
    mi_size: u8,
    subsampling_x: bool,
    subsampling_y: bool,
) -> u32 {
    // §6.4.22: if (MiSize < BLOCK_8X8) return TX_4X4.
    if mi_size < BLOCK_8X8 {
        return 0;
    }
    // §6.4.22: Min(tx_size, max_txsize_lookup[ get_plane_block_size(MiSize, 1) ]).
    let plane_sz = get_plane_block_size(mi_size, 1, subsampling_x, subsampling_y);
    debug_assert_ne!(
        plane_sz, BLOCK_INVALID,
        "chroma plane size invalid for this (mi_size, subsampling) combination"
    );
    tx_size.min(MAX_TXSIZE_LOOKUP[plane_sz as usize])
}

// ----- Per-MI-block / per-frame state -----

/// `CurrFrame[ Y ]` / `CurrFrame[ U ]` / `CurrFrame[ V ]` — the three
/// plane buffers the §6.4.21 driver writes into.
///
/// Each plane is sized for the full frame area (or larger). The driver
/// only writes the transform-block rectangles dictated by `MiSize` /
/// `tx_size` / `subsampling_*`.
#[derive(Debug)]
pub(crate) struct PlaneBuffers<'a> {
    /// `CurrFrame[ 0 ]` — luma.
    pub y: &'a mut Plane,
    /// `CurrFrame[ 1 ]` — U chroma.
    pub u: &'a mut Plane,
    /// `CurrFrame[ 2 ]` — V chroma.
    pub v: &'a mut Plane,
}

/// Per-MI-block / per-frame state consumed by [`residual_intra`].
///
/// Mirrors the §6.4.21 variables one-for-one — the names match the spec
/// where reasonable. Currently all blocks within the MI block use the
/// same [`PredMode`] (the `mode2txfm_map` lookup keys the §6.4.25
/// `TxType` off this), the same `tx_size`, the same `skip` flag and the
/// same per-plane quantizers, which mirrors the spec's per-MI-block
/// uniformity for the intra mode-info path (`y_mode` per MI block plus
/// `uv_mode` per MI block; sub-8x8 luma may carry `sub_modes[4]` but
/// that's deferred to the mode-info round).
#[derive(Debug, Clone, Copy)]
pub(crate) struct ResidualBlockCtx {
    /// `MiCol` — horizontal block location in units of 8x8 pixels.
    pub mi_col: u32,
    /// `MiRow` — vertical block location in units of 8x8 pixels.
    pub mi_row: u32,
    /// `MiCols` — width of the coded area in 8x8 units (§6.2.6).
    pub mi_cols: u32,
    /// `MiRows` — height of the coded area in 8x8 units (§6.2.6).
    pub mi_rows: u32,
    /// `MiSize` — §7.4.3 block size. Must be one of the valid `BLOCK_*`
    /// values listed in [`BLOCK_SIZES`].
    pub mi_size: u8,
    /// Luma `tx_size`. Must satisfy `tx_size <= max_txsize_lookup[
    /// MiSize ]` per §6.4.10. Chroma `txSz` is derived via
    /// [`get_uv_tx_size`].
    pub tx_size: u32,
    /// `subsampling_x` from the color config (§7.2.2). When `true`,
    /// chroma horizontal resolution is halved.
    pub subsampling_x: bool,
    /// `subsampling_y` from the color config.
    pub subsampling_y: bool,
    /// `skip` flag (§7.4.7) — when `true`, the residual loop skips both
    /// the per-block token decode and the [`reconstruct_block`] call,
    /// and the `AboveNonzeroContext` / `LeftNonzeroContext` strips are
    /// written 0 for every covered 4-sample column / row.
    pub skip: bool,
    /// `Lossless` flag (§6.2.9) — selects the WHT inverse transform.
    pub lossless: bool,
    /// `BitDepth` (8, 10 or 12) — drives `Clip1` and the §6.4.26 high-bit
    /// extra-bits loop.
    pub bit_depth: u32,
    /// Luma `PredMode` (§7.4.5). The §6.4.25 `TxType` is derived from
    /// this for the luma intra path; chroma is forced to `DCT_DCT`.
    pub y_mode: PredMode,
    /// Chroma `PredMode` (§7.4.5). The §6.4.25 `TxType` for chroma is
    /// forced to `DCT_DCT` regardless of this mode, but `predict_intra`
    /// still consumes it.
    pub uv_mode: PredMode,
    /// `get_dc_quant( 0 )` — luma DC quantizer (§8.6.1).
    pub dc_quant_y: i32,
    /// `get_ac_quant( 0 )` — luma AC quantizer.
    pub ac_quant_y: i32,
    /// `get_dc_quant( 1 )` — chroma DC quantizer (shared U / V).
    pub dc_quant_uv: i32,
    /// `get_ac_quant( 1 )` — chroma AC quantizer (shared U / V).
    pub ac_quant_uv: i32,
}

/// Per-block availability flags resolved at the §6.4.21 call site.
///
/// `AvailL` / `AvailU` are the MI-level neighbour-availability flags
/// from §7.4.4 (the MI block has a left / above neighbour in the same
/// tile). They combine with the per-sub-block `x > 0` / `y > 0`
/// position to form the `predict_intra` `have_left` / `have_above`
/// flags per the §6.4.21 listing:
///
/// ```text
/// have_left  = AvailL || x > 0
/// have_above = AvailU || y > 0
/// not_on_right = x + step < num4x4w
/// ```
#[derive(Debug, Clone, Copy)]
pub(crate) struct AvailFlags {
    /// `AvailL` — MI block has a usable left neighbour.
    pub avail_l: bool,
    /// `AvailU` — MI block has a usable above neighbour.
    pub avail_u: bool,
}

/// `Tokens[ ]` source for one transform block.
///
/// Called by [`residual_intra`] for each transform block that survives
/// the `startX < maxx && startY < maxy` bounds check and the `!skip`
/// gate. The closure receives `(plane, block_idx, tx_sz, segEob)` and
/// returns a `(tokens, nonzero)` pair — `tokens` must have exactly
/// `segEob = 16 << (tx_sz << 1)` entries (the full raster layout the
/// inverse transform expects), and `nonzero` mirrors the §6.4.24 return
/// value the driver writes back into `AboveNonzeroContext` /
/// `LeftNonzeroContext`.
///
/// A production caller backs this with the round-13 `tokens( )` driver
/// wired to a `BoolCoder` (and to the per-block `coef_probs` /
/// `NonzeroContext` / `TokenCache` state). The tests in this module
/// supply hand-crafted token buffers so the §6.4.21 outer-loop
/// semantics can be verified independently of the bit-exact token
/// stream — that lockstep land lands once the per-block mode-info
/// decode connects the residual loop to a real `BoolCoder`.
pub(crate) type TokenSource<'a> = dyn FnMut(usize, usize, u32) -> (Vec<i64>, bool) + 'a;

/// The §6.4.21 `residual( )` driver — intra path.
///
/// For an intra MI block at `(MiRow, MiCol)` of size `MiSize`, walks all
/// three planes:
///
/// 1. Computes `bsize = max(MiSize, BLOCK_8X8)`, the per-plane
///    `planeSz = get_plane_block_size(bsize, plane)` and its
///    `num_4x4_blocks_wide_lookup` / `_high_lookup` dimensions; the
///    chroma `txSz` comes from [`get_uv_tx_size`].
/// 2. For each transform block `(y, x)` in `0..num4x4h × 0..num4x4w`
///    stepping by `step = 1 << txSz`:
///    * If `startX < maxx && startY < maxy`: calls
///      [`crate::intra::predict_intra`] (with `have_left = AvailL || x >
///      0`, `have_above = AvailU || y > 0`, `not_on_right = x + step <
///      num4x4w`), then — when `!skip` — pulls a `Tokens[ ]` array from
///      the `token_source` callback and feeds it to
///      [`crate::reconstruct::reconstruct_block`] (with the §6.4.25
///      `TxType` resolved from the per-plane mode).
///    * Writes `AboveNonzeroContext[ plane ][ x4 + i ] =
///      LeftNonzeroContext[ plane ][ y4 + i ] = nonzero` for `i ∈ 0..step`
///      per the §6.4.21 trailing write-back.
///
/// `planes` carries the three `CurrFrame[ plane ]` buffers; `nz` carries
/// the three `(AboveNonzeroContext, LeftNonzeroContext)` strip pairs (a
/// fresh frame supplies a zero context). `block` carries the per-MI
/// state; `avail` the neighbour-availability flags.
///
/// This is the intra-only specialisation — the `is_inter` branch of the
/// §6.4.21 listing (which calls `predict_inter( )` before the per-block
/// loop) is deferred until reference-buffer state lands. A
/// `block.is_inter = true` caller would route through a separate
/// `residual_inter` once the §8.5.2 inter prediction process is in
/// place.
pub(crate) fn residual_intra(
    planes: &mut PlaneBuffers<'_>,
    nz: &mut [NonzeroContext; 3],
    block: &ResidualBlockCtx,
    avail: &AvailFlags,
    mut token_source: Box<TokenSource<'_>>,
) {
    // §6.4.21: bsize = MiSize < BLOCK_8X8 ? BLOCK_8X8 : MiSize.
    let bsize = if block.mi_size < BLOCK_8X8 {
        BLOCK_8X8
    } else {
        block.mi_size
    };
    let mi_col_8 = block.mi_col * 8;
    let mi_row_8 = block.mi_row * 8;

    for (plane, plane_nz) in nz.iter_mut().enumerate() {
        let tx_sz = if plane == 0 {
            block.tx_size
        } else {
            get_uv_tx_size(
                block.tx_size,
                block.mi_size,
                block.subsampling_x,
                block.subsampling_y,
            )
        };
        let step = 1u32 << tx_sz;

        let plane_sz = get_plane_block_size(bsize, plane, block.subsampling_x, block.subsampling_y);
        if plane_sz == BLOCK_INVALID {
            // Per §7.4.3 conformance such a combination is forbidden; the
            // caller is expected to never construct one. Skip rather than
            // panic so a malformed in-memory ctx degrades gracefully in
            // release builds.
            debug_assert!(false, "BLOCK_INVALID for plane {plane}");
            continue;
        }
        let num4x4w = NUM_4X4_BLOCKS_WIDE_LOOKUP[plane_sz as usize];
        let num4x4h = NUM_4X4_BLOCKS_HIGH_LOOKUP[plane_sz as usize];

        let sub_x = plane > 0 && block.subsampling_x;
        let sub_y = plane > 0 && block.subsampling_y;
        let base_x = mi_col_8 >> u32::from(sub_x);
        let base_y = mi_row_8 >> u32::from(sub_y);

        let maxx = (block.mi_cols * 8) >> u32::from(sub_x);
        let maxy = (block.mi_rows * 8) >> u32::from(sub_y);

        let mode = if plane == 0 {
            block.y_mode
        } else {
            block.uv_mode
        };
        let dc_quant = if plane == 0 {
            block.dc_quant_y
        } else {
            block.dc_quant_uv
        };
        let ac_quant = if plane == 0 {
            block.ac_quant_y
        } else {
            block.ac_quant_uv
        };

        // n0 = block side in samples; n0*n0 = segEob entries per block.
        let n0 = 1usize << (tx_sz + 2);

        let mut block_idx: usize = 0;
        let mut y = 0u32;
        while y < num4x4h {
            let mut x = 0u32;
            while x < num4x4w {
                let start_x = base_x + 4 * x;
                let start_y = base_y + 4 * y;
                let mut nonzero: u8 = 0;

                if start_x < maxx && start_y < maxy {
                    // §6.4.21: intra path predicts each transform block
                    // up-front so the next block's left/above neighbours
                    // see the reconstructed (predicted) samples.
                    let have_left = avail.avail_l || x > 0;
                    let have_above = avail.avail_u || y > 0;
                    let not_on_right = x + step < num4x4w;
                    // §8.5.1 plane-edge clamps: max_x / max_y are the
                    // last valid plane sample index. The plane's actual
                    // width / height bound is provided by the buffer
                    // itself; we use the §6.4.21 `maxx`/`maxy` (sample
                    // count) minus 1.
                    let max_x = (maxx - 1) as usize;
                    let max_y = (maxy - 1) as usize;

                    let plane_buf: &mut Plane = match plane {
                        0 => planes.y,
                        1 => planes.u,
                        _ => planes.v,
                    };

                    predict_intra(
                        plane_buf,
                        start_x as usize,
                        start_y as usize,
                        have_left,
                        have_above,
                        not_on_right,
                        tx_sz,
                        mode,
                        max_x,
                        max_y,
                        block.bit_depth,
                    );

                    if !block.skip {
                        let seg_eob = (n0 * n0) as u32; // 16 << (tx_sz << 1)
                        let (tokens, nz_flag) = token_source(plane, block_idx, tx_sz);
                        debug_assert_eq!(
                            tokens.len(),
                            seg_eob as usize,
                            "TokenSource returned {} entries for plane {plane} block {block_idx}, expected {seg_eob}",
                            tokens.len()
                        );
                        // §6.4.25 TxType selection: chroma forces
                        // DCT_DCT; TX_32X32 forces DCT_DCT; lossless
                        // ignores tx_type (WHT path). Luma intra uses
                        // mode2txfm_map[ y_mode ].
                        let tx_type = if plane > 0 || tx_sz == 3 || block.lossless {
                            crate::idct::DCT_DCT
                        } else {
                            crate::reconstruct::tx_type_for_intra(mode)
                        };

                        reconstruct_block(
                            plane_buf,
                            start_x as usize,
                            start_y as usize,
                            tx_sz,
                            &tokens,
                            dc_quant,
                            ac_quant,
                            tx_type,
                            block.lossless,
                            block.bit_depth,
                        );

                        if nz_flag {
                            nonzero = 1;
                        }
                    }
                }

                // §6.4.21: write the nonzero flag into the per-4-sample
                // above / left strips for every column / row the block
                // covers (`step` 4-sample units in each direction).
                let x4 = (start_x >> 2) as usize;
                let y4 = (start_y >> 2) as usize;
                for i in 0..step as usize {
                    if x4 + i < plane_nz.above.len() {
                        plane_nz.above[x4 + i] = nonzero;
                    }
                    if y4 + i < plane_nz.left.len() {
                        plane_nz.left[y4 + i] = nonzero;
                    }
                }

                block_idx += 1;
                x += step;
            }
            y += step;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::idct::{inverse_transform_2d, DCT_DCT};
    use crate::intra::PredMode;

    // ----- Static tables -----

    #[test]
    fn num_4x4_lookups_match_spec() {
        // §10.2 transcription anchors.
        assert_eq!(
            NUM_4X4_BLOCKS_WIDE_LOOKUP,
            [1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16]
        );
        assert_eq!(
            NUM_4X4_BLOCKS_HIGH_LOOKUP,
            [1, 2, 1, 2, 4, 2, 4, 8, 4, 8, 16, 8, 16]
        );
    }

    #[test]
    fn max_txsize_lookup_matches_spec() {
        // §6.4.10 listing: TX_4X4 (0) for 4x4..8x4, TX_8X8 (1) for
        // 8x8..16x8, TX_16X16 (2) for 16x16..32x16, TX_32X32 (3) for
        // 32x32..64x64.
        assert_eq!(MAX_TXSIZE_LOOKUP[BLOCK_4X4 as usize], 0);
        assert_eq!(MAX_TXSIZE_LOOKUP[BLOCK_8X4 as usize], 0);
        assert_eq!(MAX_TXSIZE_LOOKUP[BLOCK_8X8 as usize], 1);
        assert_eq!(MAX_TXSIZE_LOOKUP[BLOCK_16X8 as usize], 1);
        assert_eq!(MAX_TXSIZE_LOOKUP[BLOCK_16X16 as usize], 2);
        assert_eq!(MAX_TXSIZE_LOOKUP[BLOCK_32X16 as usize], 2);
        assert_eq!(MAX_TXSIZE_LOOKUP[BLOCK_32X32 as usize], 3);
        assert_eq!(MAX_TXSIZE_LOOKUP[BLOCK_64X64 as usize], 3);
    }

    #[test]
    fn ss_size_lookup_anchors_match_spec() {
        // Luma path: (subsampling_x, subsampling_y) = (false, false) ->
        // ss_size_lookup[subsize][0][0] == subsize for every subsize.
        for (sz, row) in SS_SIZE_LOOKUP.iter().enumerate() {
            assert_eq!(row[0][0], sz as u8);
        }
        // §6.4.23 listing anchors:
        assert_eq!(SS_SIZE_LOOKUP[BLOCK_4X4 as usize][1][1], BLOCK_INVALID);
        assert_eq!(SS_SIZE_LOOKUP[BLOCK_8X8 as usize][1][1], BLOCK_4X4);
        assert_eq!(SS_SIZE_LOOKUP[BLOCK_16X16 as usize][1][1], BLOCK_8X8);
        assert_eq!(SS_SIZE_LOOKUP[BLOCK_32X32 as usize][1][1], BLOCK_16X16);
        assert_eq!(SS_SIZE_LOOKUP[BLOCK_64X64 as usize][1][1], BLOCK_32X32);
        // Asymmetric subsampling:
        assert_eq!(SS_SIZE_LOOKUP[BLOCK_8X8 as usize][1][0], BLOCK_4X8);
        assert_eq!(SS_SIZE_LOOKUP[BLOCK_8X8 as usize][0][1], BLOCK_8X4);
    }

    // ----- get_plane_block_size / get_uv_tx_size -----

    #[test]
    fn get_plane_block_size_luma_returns_subsize() {
        for sz in 0..BLOCK_SIZES {
            assert_eq!(get_plane_block_size(sz as u8, 0, true, true), sz as u8);
            assert_eq!(get_plane_block_size(sz as u8, 0, false, false), sz as u8);
        }
    }

    #[test]
    fn get_plane_block_size_chroma_subsamples_4_2_0() {
        // 4:2:0 (subsampling_x=true, subsampling_y=true).
        assert_eq!(get_plane_block_size(BLOCK_8X8, 1, true, true), BLOCK_4X4);
        assert_eq!(get_plane_block_size(BLOCK_16X16, 1, true, true), BLOCK_8X8);
        assert_eq!(
            get_plane_block_size(BLOCK_64X64, 1, true, true),
            BLOCK_32X32
        );
    }

    #[test]
    fn get_uv_tx_size_caps_at_chroma_max() {
        // 4:2:0, MiSize = BLOCK_16X16 (chroma plane = BLOCK_8X8, max
        // txsz = TX_8X8 (1)). Luma tx_size=2 -> chroma tx_size=1.
        assert_eq!(get_uv_tx_size(2, BLOCK_16X16, true, true), 1);
        // 4:4:4 (subsampling false/false): chroma plane = MiSize,
        // chroma max = luma max.
        assert_eq!(get_uv_tx_size(2, BLOCK_16X16, false, false), 2);
        // MiSize < BLOCK_8X8 forces TX_4X4 (0).
        assert_eq!(get_uv_tx_size(3, BLOCK_4X4, true, true), 0);
    }

    // ----- residual_intra outer loop -----

    fn make_ctx(mi_size: u8, tx_size: u32, skip: bool) -> ResidualBlockCtx {
        ResidualBlockCtx {
            mi_col: 0,
            mi_row: 0,
            mi_cols: 2, // 16-pixel-wide coded area
            mi_rows: 2,
            mi_size,
            tx_size,
            subsampling_x: true,
            subsampling_y: true,
            skip,
            lossless: false,
            bit_depth: 8,
            y_mode: PredMode::DcPred,
            uv_mode: PredMode::DcPred,
            dc_quant_y: 16,
            ac_quant_y: 16,
            dc_quant_uv: 16,
            ac_quant_uv: 16,
        }
    }

    fn make_nz() -> [NonzeroContext; 3] {
        // For an MI block at (0,0) with mi_cols=mi_rows=2 the per-plane
        // strip widths are 4 (luma) / 2 (chroma 4:2:0).
        [
            NonzeroContext::new(4, 4),
            NonzeroContext::new(2, 2),
            NonzeroContext::new(2, 2),
        ]
    }

    /// `skip = true` skips all token decode / reconstruct calls and
    /// writes 0 into every `AboveNonzero` / `LeftNonzero` cell the block
    /// covers.
    #[test]
    fn residual_intra_skip_writes_zero_nonzero_context() {
        let mut y_plane = Plane::new(16, 16);
        let mut u_plane = Plane::new(8, 8);
        let mut v_plane = Plane::new(8, 8);
        let mut planes = PlaneBuffers {
            y: &mut y_plane,
            u: &mut u_plane,
            v: &mut v_plane,
        };
        let mut nz = make_nz();
        // Pre-seed a 1 to confirm `skip` overwrites it.
        nz[0].above[0] = 1;
        nz[0].left[0] = 1;
        let ctx = make_ctx(BLOCK_16X16, /* tx_size = */ 1, /* skip = */ true);
        let avail = AvailFlags {
            avail_l: false,
            avail_u: false,
        };

        let calls = std::cell::RefCell::new(0u32);
        let source: Box<TokenSource<'_>> = Box::new(|_plane, _idx, _ts| {
            *calls.borrow_mut() += 1;
            (vec![], false)
        });
        residual_intra(&mut planes, &mut nz, &ctx, &avail, source);
        assert_eq!(
            *calls.borrow(),
            0,
            "token source must not fire when skip=true"
        );
        for plane_nz in &nz {
            assert!(plane_nz.above.iter().all(|&b| b == 0));
            assert!(plane_nz.left.iter().all(|&b| b == 0));
        }
    }

    /// `skip = false` with an all-zero token source predicts every block
    /// and leaves the prediction in the plane (the residual is 0).
    /// Token source returns `nonzero = false` so the strips stay 0.
    #[test]
    fn residual_intra_runs_predict_and_reconstruct_for_each_block() {
        let mut y_plane = Plane::new(16, 16);
        // Seed the row above and column left of the MI block (which sits
        // at (0,0)) — but predict_intra at (0,0) with !avail_l !avail_u
        // and y==x==0 falls back to the BitDepth midpoint anyway, so the
        // seeds aren't read.
        let mut u_plane = Plane::new(8, 8);
        let mut v_plane = Plane::new(8, 8);
        let mut planes = PlaneBuffers {
            y: &mut y_plane,
            u: &mut u_plane,
            v: &mut v_plane,
        };
        let mut nz = make_nz();
        // tx_size = TX_4X4 (0), 16x16 -> 16 luma blocks + chroma is
        // 8x8 with tx_size = 0 -> 4 blocks each.
        let ctx = make_ctx(BLOCK_16X16, 0, false);
        let avail = AvailFlags {
            avail_l: false,
            avail_u: false,
        };

        let calls = std::cell::RefCell::new(Vec::<(usize, usize, u32)>::new());
        let source: Box<TokenSource<'_>> = Box::new(|plane, idx, tx_sz| {
            calls.borrow_mut().push((plane, idx, tx_sz));
            (vec![0i64; 16], false)
        });

        residual_intra(&mut planes, &mut nz, &ctx, &avail, source);

        let recorded = calls.borrow();
        // luma: 16 blocks; chroma: 4 + 4.
        assert_eq!(recorded.len(), 16 + 4 + 4);
        // First 16 entries are luma block 0..15 with tx_sz=0.
        for (i, &(p, b, ts)) in recorded[0..16].iter().enumerate() {
            assert_eq!(p, 0);
            assert_eq!(b, i);
            assert_eq!(ts, 0);
        }
        // U plane: 4 entries.
        for (i, &(p, b, ts)) in recorded[16..20].iter().enumerate() {
            assert_eq!(p, 1);
            assert_eq!(b, i);
            assert_eq!(ts, 0);
        }
        // V plane: 4 entries.
        for (i, &(p, b, ts)) in recorded[20..24].iter().enumerate() {
            assert_eq!(p, 2);
            assert_eq!(b, i);
            assert_eq!(ts, 0);
        }

        // Every strip cell stays 0 (token source reports nonzero=false).
        for plane_nz in &nz {
            assert!(plane_nz.above.iter().all(|&b| b == 0));
            assert!(plane_nz.left.iter().all(|&b| b == 0));
        }

        // The plane buffers carry the predicted samples. DC_PRED with
        // no neighbours uses the BitDepth midpoint = 128. The
        // reconstruct step adds a zero residual so 128 survives. Every
        // sample within the 16x16 luma region should equal 128.
        for i in 0..16 {
            for j in 0..16 {
                assert_eq!(planes.y.get(j, i), 128, "luma at ({j},{i})");
            }
        }
        // Chroma planes equally land at midpoint 128.
        for i in 0..8 {
            for j in 0..8 {
                assert_eq!(planes.u.get(j, i), 128, "u at ({j},{i})");
            }
        }
    }

    /// `nonzero = true` from the token source writes 1 into the strips
    /// for every 4-sample column / row the block covers (`step` units).
    #[test]
    fn residual_intra_writes_nonzero_strips_per_step() {
        let mut y_plane = Plane::new(16, 16);
        let mut u_plane = Plane::new(8, 8);
        let mut v_plane = Plane::new(8, 8);
        let mut planes = PlaneBuffers {
            y: &mut y_plane,
            u: &mut u_plane,
            v: &mut v_plane,
        };
        let mut nz = make_nz();
        // 16x16 MI, tx_size = TX_8X8 (1) -> step = 2, num4x4w=num4x4h=4
        // -> 2x2 = 4 luma blocks per MI block. Chroma is 8x8 with
        // tx_size = 1 (8x8) -> 1 block, step = 2.
        let ctx = make_ctx(BLOCK_16X16, 1, false);
        let avail = AvailFlags {
            avail_l: false,
            avail_u: false,
        };

        let source: Box<TokenSource<'_>> = Box::new(|_p, _b, _ts| {
            // 8x8 -> 64 entries; nonzero = true.
            (vec![0i64; 64], true)
        });

        residual_intra(&mut planes, &mut nz, &ctx, &avail, source);
        // Luma above/left strips: 4 cells each (mi_cols=2 -> 4 luma
        // 4-sample units), all should be 1 (every block covers 2 cells).
        assert_eq!(nz[0].above, vec![1, 1, 1, 1]);
        assert_eq!(nz[0].left, vec![1, 1, 1, 1]);
        // Chroma above/left strips: 2 cells each (8/4 = 2 4-sample
        // units), all 1.
        assert_eq!(nz[1].above, vec![1, 1]);
        assert_eq!(nz[1].left, vec![1, 1]);
        assert_eq!(nz[2].above, vec![1, 1]);
        assert_eq!(nz[2].left, vec![1, 1]);
    }

    /// A block whose `startX` lands outside the frame's `maxx` is skipped
    /// entirely (no predict, no reconstruct), but the strip write-back
    /// still runs with `nonzero = 0` so the §6.4.21 trailing context
    /// invariant holds.
    #[test]
    fn residual_intra_skips_out_of_bounds_block() {
        // MI block at (mi_col=1, mi_row=0) on a frame with mi_cols=2
        // (16 luma samples wide). The block covers samples 8..16, which
        // is in-bounds: pick a smaller frame to force the OOB path.
        // mi_cols=1 means the frame is 8 samples wide; placing the MI
        // block at (1,0) puts startX=8 already at maxx=8.
        let mut y_plane = Plane::new(16, 16);
        let mut u_plane = Plane::new(8, 8);
        let mut v_plane = Plane::new(8, 8);
        let mut planes = PlaneBuffers {
            y: &mut y_plane,
            u: &mut u_plane,
            v: &mut v_plane,
        };
        let mut nz = [
            NonzeroContext::new(4, 4),
            NonzeroContext::new(2, 2),
            NonzeroContext::new(2, 2),
        ];
        let ctx = ResidualBlockCtx {
            mi_col: 1,
            mi_row: 0,
            mi_cols: 1, // maxx_luma = 8
            mi_rows: 1,
            mi_size: BLOCK_8X8,
            tx_size: 0,
            subsampling_x: true,
            subsampling_y: true,
            skip: false,
            lossless: false,
            bit_depth: 8,
            y_mode: PredMode::DcPred,
            uv_mode: PredMode::DcPred,
            dc_quant_y: 16,
            ac_quant_y: 16,
            dc_quant_uv: 16,
            ac_quant_uv: 16,
        };
        let avail = AvailFlags {
            avail_l: false,
            avail_u: false,
        };

        let token_calls = std::cell::RefCell::new(0u32);
        let source: Box<TokenSource<'_>> = Box::new(|_p, _b, _ts| {
            *token_calls.borrow_mut() += 1;
            (vec![0i64; 16], true)
        });

        residual_intra(&mut planes, &mut nz, &ctx, &avail, source);
        // start_x = 8 >= maxx = 8 -> block predict / reconstruct skipped
        // for every transform block in the MI block.
        assert_eq!(*token_calls.borrow(), 0);
        // Strips covered (luma x4 ∈ {2,3}, y4 ∈ {0,1}) stay 0.
        assert_eq!(nz[0].above, vec![0, 0, 0, 0]);
        assert_eq!(nz[0].left, vec![0, 0, 0, 0]);
    }

    /// A DC-only luma block reconstructs to the prediction plus a flat
    /// residual, identical to the round-11 `reconstruct_block` output.
    /// Threads one transform block through the residual driver and
    /// cross-checks against an independent probe. The MI block is placed
    /// at (mi_col=1, mi_row=1) so both `AvailL` and `AvailU` correspond
    /// to in-bounds neighbour samples.
    #[test]
    fn residual_intra_dc_residual_matches_reconstruct_block() {
        let mut y_plane = Plane::new(32, 32);
        // Seed the above row (y = 7, the last row before the MI block
        // at MiRow=1 / start_y=8) and left column (x = 7) so DC_PRED
        // averages a known value.
        for j in 0..32 {
            y_plane.set(j, 7, 50);
        }
        for i in 0..32 {
            y_plane.set(7, i, 50);
        }
        let mut u_plane = Plane::new(16, 16);
        let mut v_plane = Plane::new(16, 16);
        let mut planes = PlaneBuffers {
            y: &mut y_plane,
            u: &mut u_plane,
            v: &mut v_plane,
        };
        let mut nz = [
            NonzeroContext::new(8, 8),
            NonzeroContext::new(4, 4),
            NonzeroContext::new(4, 4),
        ];
        // 16x16 MI block, tx_size = TX_16X16 (2) -> one luma 16x16 block
        // at MI cell (1,1); step = 4.
        let mut ctx = make_ctx(BLOCK_16X16, 2, false);
        ctx.mi_col = 1;
        ctx.mi_row = 1;
        ctx.mi_cols = 4;
        ctx.mi_rows = 4;
        let avail = AvailFlags {
            avail_l: true,
            avail_u: true,
        };

        // Token buffer: DC-only Tokens[0] = 2 on luma; zero on chroma.
        let source: Box<TokenSource<'_>> = Box::new(|plane, _b, tx_sz| {
            let n0 = 1usize << (tx_sz + 2);
            let mut t = vec![0i64; n0 * n0];
            if plane == 0 {
                t[0] = 2;
            }
            (t, plane == 0)
        });

        residual_intra(&mut planes, &mut nz, &ctx, &avail, source);

        // Independent probe: rerun predict + reconstruct in isolation
        // and compare the luma 16x16 region.
        let mut probe_plane = Plane::new(32, 32);
        for j in 0..32 {
            probe_plane.set(j, 7, 50);
        }
        for i in 0..32 {
            probe_plane.set(7, i, 50);
        }
        // start_x = (1 * 8) >> 0 = 8; start_y = 8.
        // maxX = (4 * 8) - 1 = 31; maxY = 31. not_on_right: x + step
        // (0+4) == num4x4w (4) -> false.
        predict_intra(
            &mut probe_plane,
            8,
            8,
            true,
            true,
            false,
            2,
            PredMode::DcPred,
            31,
            31,
            8,
        );
        // All tokens are 0 except [0]; the AC scaling collapses to 0.
        let mut probe_dequant = vec![0i64; 16 * 16];
        probe_dequant[0] = 2 * ctx.dc_quant_y as i64;
        inverse_transform_2d(&mut probe_dequant, 4, DCT_DCT, false);
        for i in 0..16 {
            for j in 0..16 {
                let pred = probe_plane.get(8 + j, 8 + i) as i64;
                let expected = (pred + probe_dequant[i * 16 + j]).clamp(0, 255) as i32;
                assert_eq!(
                    planes.y.get(8 + j, 8 + i),
                    expected,
                    "luma at ({},{}) mismatch",
                    8 + j,
                    8 + i
                );
            }
        }

        // Strip layout: luma block at x4 = (8>>2) = 2 covers
        // x4 ∈ {2, 3, 4, 5}; left[y4 ∈ {2,3,4,5}] = 1.
        for x4 in 2..6 {
            assert_eq!(nz[0].above[x4], 1, "above[{x4}]");
        }
        for y4 in 2..6 {
            assert_eq!(nz[0].left[y4], 1, "left[{y4}]");
        }
        // Untouched cells remain 0.
        assert_eq!(nz[0].above[0], 0);
        assert_eq!(nz[0].above[1], 0);
    }

    /// Sub-8x8 MI block (BLOCK_4X4) — §6.4.21 widens bsize to BLOCK_8X8
    /// so the residual loop still walks an 8x8 plane region (luma:
    /// 2x2 blocks of TX_4X4; chroma 4x4 -> 1 TX_4X4 block).
    #[test]
    fn residual_intra_block_4x4_widens_bsize_to_8x8() {
        let mut y_plane = Plane::new(16, 16);
        let mut u_plane = Plane::new(8, 8);
        let mut v_plane = Plane::new(8, 8);
        let mut planes = PlaneBuffers {
            y: &mut y_plane,
            u: &mut u_plane,
            v: &mut v_plane,
        };
        let mut nz = make_nz();
        let ctx = make_ctx(BLOCK_4X4, 0, false);
        let avail = AvailFlags {
            avail_l: false,
            avail_u: false,
        };

        let block_counts = std::cell::RefCell::new([0u32; 3]);
        let source: Box<TokenSource<'_>> = Box::new(|plane, _b, _ts| {
            block_counts.borrow_mut()[plane] += 1;
            (vec![0i64; 16], false)
        });

        residual_intra(&mut planes, &mut nz, &ctx, &avail, source);

        // bsize widened to BLOCK_8X8: luma plane size = BLOCK_8X8 ->
        // num4x4w = num4x4h = 2 -> 4 blocks. Chroma 4:2:0 ->
        // get_plane_block_size(BLOCK_8X8, chroma, true, true) = BLOCK_4X4
        // -> num4x4w = num4x4h = 1 -> 1 block.
        assert_eq!(*block_counts.borrow(), [4, 1, 1]);
    }
}
