//! VP9 compressed header (§6.3).
//!
//! The compressed header lives between the uncompressed header and the
//! first tile partition. Its length is `header_size` (read at the very end
//! of the uncompressed header). It is decoded with the boolean engine
//! (§9.2). The decoded values steer downstream tile decode.
//!
//! Subsections implemented:
//! * §6.3.1 read_tx_mode           — 2..3 bits, picks one of {ONLY_4x4,
//!   ALLOW_8x8, ALLOW_16x16, ALLOW_32x32, TX_MODE_SELECT}.
//! * §6.3.2 tx_mode_probs          — only when tx_mode == TX_MODE_SELECT.
//! * §6.3.3 diff_update_prob       — core subtree-delta machinery.
//! * §6.3.4 decode_term_subexp     — 4/5/7/8-bit variable-length delta.
//! * §6.3.5 inv_remap_prob         — the §10.5 remap table.
//! * §6.3.6 inv_recenter_nonneg    — signed → unsigned remap.
//! * §6.3.7 read_coef_probs        — per-tx_size coefficient updates.
//! * §6.3.8 read_skip_prob         — 3 context-conditioned probs.
//! * §6.3.9 read_inter_mode_probs  — INTER_MODE_CONTEXTS × (INTER_MODES-1).
//! * §6.3.10 read_interp_filter_probs — only if interpolation_filter == SWITCHABLE.
//! * §6.3.11 read_is_inter_probs.
//! * §6.3.12 frame_reference_mode  — derives SINGLE / COMPOUND / SELECT.
//! * §6.3.13 frame_reference_mode_probs — comp_mode / single_ref / comp_ref.
//! * §6.3.14 read_y_mode_probs.
//! * §6.3.15 read_partition_probs.
//! * §6.3.16 mv_probs              — joints + per-component {sign, class, bits, fr, hp}.
//! * §6.3.18 setup_compound_reference_mode.
//!
//! The decoded probabilities live in the embedded [`FrameContext`]
//! which the tile decoder consumes for coefficient / mode entropy
//! decode. On keyframes / intra_only frames the spec starts from the
//! §10.5 defaults; for inter frames we do the same for now (saved
//! per-slot contexts from `frame_context_idx` are not yet carried
//! across frames — that's an §8.10 follow-up).

use oxideav_core::{Error, Result};

use crate::bool_decoder::BoolDecoder;
use crate::frame_ctx::{
    diff_update_prob, update_mv_prob, FrameContext, BLOCK_SIZE_GROUPS, BLOCK_TYPES, COEF_BANDS,
    COEF_CTX, COEF_NODES, COMP_MODE_CONTEXTS, INTERP_FILTER_CONTEXTS, INTERP_FILTER_PROBS,
    INTER_MODE_CONTEXTS, INTER_MODE_PROBS, INTRA_MODES, INTRA_MODES_M1, IS_INTER_CONTEXTS,
    PARTITION_CONTEXTS, PARTITION_TYPES_M1, REF_CONTEXTS, REF_TYPES, SKIP_CONTEXTS, TX_SIZES,
    TX_SIZE_CONTEXTS,
};
use crate::headers::{FrameType, UncompressedHeader};

/// Interpolation filter sentinel used by §6.3.10. The uncompressed
/// header carries `interpolation_filter` as a u8; the `SWITCHABLE` code
/// value is 4 per §7.2.7.
pub const SWITCHABLE_INTERP_FILTER: u8 = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TxMode {
    Only4x4 = 0,
    Allow8x8 = 1,
    Allow16x16 = 2,
    Allow32x32 = 3,
    Select = 4,
}

impl TxMode {
    pub fn biggest_tx_size(self) -> usize {
        match self {
            TxMode::Only4x4 => 0,
            TxMode::Allow8x8 => 1,
            TxMode::Allow16x16 => 2,
            TxMode::Allow32x32 | TxMode::Select => 3,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReferenceMode {
    SingleReference = 0,
    CompoundReference = 1,
    ReferenceModeSelect = 2,
}

/// Reference-frame wiring derived by §6.3.18 `setup_compound_reference_mode`.
/// Defaults match the spec's `ref_frame_sign_bias[LAST]==ref_frame_sign_bias[ALTREF]`
/// fallback so keyframes / intra-only frames get a well-defined value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompoundRefs {
    /// The fixed reference frame used in every compound block.
    pub fixed: u8,
    /// Two variable reference choices (indexed by `comp_ref` bit).
    pub var: [u8; 2],
}

impl Default for CompoundRefs {
    fn default() -> Self {
        Self {
            // §6.3.18 fallback — LAST / GOLDEN / ALTREF sign-bias equal.
            fixed: 1,    // LAST_FRAME
            var: [2, 3], // GOLDEN_FRAME / ALTREF_FRAME
        }
    }
}

#[derive(Clone, Debug)]
pub struct CompressedHeader {
    pub tx_mode: Option<TxMode>,
    pub reference_mode: Option<ReferenceMode>,
    pub compound_refs: CompoundRefs,
    /// Per-frame entropy state — default §10.5 tables, then updated
    /// by the §6.3 diff_update_prob stream.
    pub ctx: FrameContext,
}

impl Default for CompressedHeader {
    fn default() -> Self {
        Self {
            tx_mode: None,
            reference_mode: None,
            compound_refs: CompoundRefs::default(),
            ctx: FrameContext::new_default(),
        }
    }
}

/// Parse the §6.3 compressed header in full. All probability deltas are
/// folded into the embedded `FrameContext`; tile decode should use it
/// for all entropy decisions. The seed FrameContext is built from the
/// §10.5 default tables — callers wanting per-slot saved contexts
/// (§8.10) should use [`parse_compressed_header_with_seed`].
pub fn parse_compressed_header(
    payload: &[u8],
    hdr: &UncompressedHeader,
) -> Result<CompressedHeader> {
    parse_compressed_header_with_seed(payload, hdr, FrameContext::new_default())
}

/// Like [`parse_compressed_header`] but seeds the frame context with
/// `seed` instead of the default. Used by the top-level decoder to
/// carry probability state across frames via the four `frame_context_idx`
/// slots (§8.10).
pub fn parse_compressed_header_with_seed(
    payload: &[u8],
    hdr: &UncompressedHeader,
    seed: FrameContext,
) -> Result<CompressedHeader> {
    if payload.is_empty() {
        return Err(Error::invalid("vp9 §6.3: compressed header missing"));
    }
    let mut bd = BoolDecoder::new(payload)?;
    let mut out = CompressedHeader {
        tx_mode: None,
        reference_mode: None,
        compound_refs: CompoundRefs::default(),
        ctx: seed,
    };

    // §6.3.1 read_tx_mode.
    let tx_mode = if hdr.quantization.lossless {
        TxMode::Only4x4
    } else {
        read_tx_mode(&mut bd)?
    };
    out.tx_mode = Some(tx_mode);

    // §6.3.2 tx_mode_probs.
    if tx_mode == TxMode::Select {
        read_tx_mode_probs(&mut bd, &mut out.ctx)?;
    }

    // §6.3.7 read_coef_probs.
    read_coef_probs(&mut bd, &mut out.ctx, tx_mode)?;

    // §6.3.8 read_skip_prob.
    for i in 0..SKIP_CONTEXTS {
        out.ctx.skip_probs[i] = diff_update_prob(&mut bd, out.ctx.skip_probs[i])?;
    }

    let frame_is_intra = hdr.frame_type == FrameType::Key || hdr.intra_only;
    if !frame_is_intra {
        // §6.3.9 read_inter_mode_probs.
        read_inter_mode_probs(&mut bd, &mut out.ctx)?;

        // §6.3.10 read_interp_filter_probs (only SWITCHABLE).
        if hdr.interpolation_filter == SWITCHABLE_INTERP_FILTER {
            read_interp_filter_probs(&mut bd, &mut out.ctx)?;
        }

        // §6.3.11 read_is_inter_probs.
        for i in 0..IS_INTER_CONTEXTS {
            out.ctx.is_inter_prob[i] = diff_update_prob(&mut bd, out.ctx.is_inter_prob[i])?;
        }

        // §6.3.12 frame_reference_mode — decides
        // SINGLE/COMPOUND/REFERENCE_MODE_SELECT.
        let reference_mode = read_reference_mode(&mut bd, hdr)?;
        out.reference_mode = Some(reference_mode);
        // §6.3.18 setup_compound_reference_mode.
        if reference_mode != ReferenceMode::SingleReference {
            out.compound_refs = setup_compound_reference_mode(hdr);
        }

        // §6.3.13 frame_reference_mode_probs.
        read_frame_reference_mode_probs(&mut bd, &mut out.ctx, reference_mode)?;

        // §6.3.14 read_y_mode_probs.
        for i in 0..BLOCK_SIZE_GROUPS {
            for j in 0..INTRA_MODES_M1 {
                out.ctx.y_mode_probs[i][j] = diff_update_prob(&mut bd, out.ctx.y_mode_probs[i][j])?;
            }
        }

        // §6.3.15 read_partition_probs.
        for i in 0..PARTITION_CONTEXTS {
            for j in 0..PARTITION_TYPES_M1 {
                out.ctx.partition_probs[i][j] =
                    diff_update_prob(&mut bd, out.ctx.partition_probs[i][j])?;
            }
        }

        // §6.3.16 mv_probs.
        read_mv_probs(&mut bd, &mut out.ctx, hdr.allow_high_precision_mv)?;
    } else {
        out.reference_mode = Some(ReferenceMode::SingleReference);
    }

    Ok(out)
}

fn read_tx_mode(bd: &mut BoolDecoder<'_>) -> Result<TxMode> {
    let tx_mode = bd.read_literal(2)?;
    let tx_mode = if tx_mode == 3 {
        3 + bd.read_literal(1)?
    } else {
        tx_mode
    };
    Ok(match tx_mode {
        0 => TxMode::Only4x4,
        1 => TxMode::Allow8x8,
        2 => TxMode::Allow16x16,
        3 => TxMode::Allow32x32,
        _ => TxMode::Select,
    })
}

/// §6.3.2 tx_mode_probs — only when tx_mode == TX_MODE_SELECT.
fn read_tx_mode_probs(bd: &mut BoolDecoder<'_>, ctx: &mut FrameContext) -> Result<()> {
    // 8×8 probs: TX_SIZE_CONTEXTS × 1 = TX_SIZES-3 probs per context.
    for i in 0..TX_SIZE_CONTEXTS {
        for j in 0..1 {
            ctx.tx_probs_8x8[i][j] = diff_update_prob(bd, ctx.tx_probs_8x8[i][j])?;
        }
    }
    // 16×16 probs: × 2 (TX_SIZES-2).
    for i in 0..TX_SIZE_CONTEXTS {
        for j in 0..2 {
            ctx.tx_probs_16x16[i][j] = diff_update_prob(bd, ctx.tx_probs_16x16[i][j])?;
        }
    }
    // 32×32 probs: × 3 (TX_SIZES-1).
    for i in 0..TX_SIZE_CONTEXTS {
        for j in 0..3 {
            ctx.tx_probs_32x32[i][j] = diff_update_prob(bd, ctx.tx_probs_32x32[i][j])?;
        }
    }
    Ok(())
}

/// §6.3.7 read_coef_probs — the bulk of the compressed header. For
/// each tx_size up to `maxTxSize`, an outer `L(1)` flag gates whether
/// that tx_size's 2×2×6×(3|6)×3 coefficient table receives updates.
fn read_coef_probs(
    bd: &mut BoolDecoder<'_>,
    ctx: &mut FrameContext,
    tx_mode: TxMode,
) -> Result<()> {
    let max_tx = tx_mode.biggest_tx_size();
    for tx_sz in 0..=max_tx {
        let update = bd.read_literal(1)?;
        if update == 0 {
            continue;
        }
        for i in 0..BLOCK_TYPES {
            for j in 0..REF_TYPES {
                for k in 0..COEF_BANDS {
                    // §6.3.7 — band 0 has only 3 contexts.
                    let max_l = if k == 0 { 3 } else { COEF_CTX };
                    for l in 0..max_l {
                        for m in 0..COEF_NODES {
                            let cur = ctx.coef_probs[tx_sz][i][j][k][l][m];
                            ctx.coef_probs[tx_sz][i][j][k][l][m] = diff_update_prob(bd, cur)?;
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// §6.3.9 read_inter_mode_probs.
fn read_inter_mode_probs(bd: &mut BoolDecoder<'_>, ctx: &mut FrameContext) -> Result<()> {
    for i in 0..INTER_MODE_CONTEXTS {
        for j in 0..INTER_MODE_PROBS {
            ctx.inter_mode_probs[i][j] = diff_update_prob(bd, ctx.inter_mode_probs[i][j])?;
        }
    }
    Ok(())
}

/// §6.3.10 read_interp_filter_probs.
fn read_interp_filter_probs(bd: &mut BoolDecoder<'_>, ctx: &mut FrameContext) -> Result<()> {
    for j in 0..INTERP_FILTER_CONTEXTS {
        for i in 0..INTERP_FILTER_PROBS {
            ctx.interp_filter_probs[j][i] = diff_update_prob(bd, ctx.interp_filter_probs[j][i])?;
        }
    }
    Ok(())
}

/// §6.3.12 frame_reference_mode — derives `reference_mode`.
fn read_reference_mode(
    bd: &mut BoolDecoder<'_>,
    hdr: &UncompressedHeader,
) -> Result<ReferenceMode> {
    // §6.3.12 compoundReferenceAllowed: true iff any of ref_frame_sign_bias
    // for LAST/GOLDEN/ALTREF differ from ref_frame_sign_bias[1].
    let bias = hdr.ref_frame_sign_bias;
    let mut compound_allowed = false;
    for i in 1..3 {
        if bias[i + 1] as u8 != bias[1] as u8 {
            compound_allowed = true;
            break;
        }
    }
    if !compound_allowed {
        return Ok(ReferenceMode::SingleReference);
    }
    let non_single = bd.read_literal(1)?;
    if non_single == 0 {
        return Ok(ReferenceMode::SingleReference);
    }
    let select = bd.read_literal(1)?;
    if select == 1 {
        Ok(ReferenceMode::ReferenceModeSelect)
    } else {
        Ok(ReferenceMode::CompoundReference)
    }
}

/// §6.3.13 frame_reference_mode_probs.
fn read_frame_reference_mode_probs(
    bd: &mut BoolDecoder<'_>,
    ctx: &mut FrameContext,
    reference_mode: ReferenceMode,
) -> Result<()> {
    if reference_mode == ReferenceMode::ReferenceModeSelect {
        for i in 0..COMP_MODE_CONTEXTS {
            ctx.comp_mode_prob[i] = diff_update_prob(bd, ctx.comp_mode_prob[i])?;
        }
    }
    if reference_mode != ReferenceMode::CompoundReference {
        for i in 0..REF_CONTEXTS {
            for j in 0..2 {
                ctx.single_ref_prob[i][j] = diff_update_prob(bd, ctx.single_ref_prob[i][j])?;
            }
        }
    }
    if reference_mode != ReferenceMode::SingleReference {
        for i in 0..REF_CONTEXTS {
            ctx.comp_ref_prob[i] = diff_update_prob(bd, ctx.comp_ref_prob[i])?;
        }
    }
    Ok(())
}

/// §6.3.18 setup_compound_reference_mode — pick {CompFixedRef, CompVarRef[0..1]}
/// from the sign-bias pattern over {LAST, GOLDEN, ALTREF}.
fn setup_compound_reference_mode(hdr: &UncompressedHeader) -> CompoundRefs {
    const LAST: u8 = 1;
    const GOLDEN: u8 = 2;
    const ALTREF: u8 = 3;
    let bias: [u8; 4] = [
        hdr.ref_frame_sign_bias[0] as u8,
        hdr.ref_frame_sign_bias[1] as u8,
        hdr.ref_frame_sign_bias[2] as u8,
        hdr.ref_frame_sign_bias[3] as u8,
    ];
    if bias[LAST as usize] == bias[GOLDEN as usize] {
        CompoundRefs {
            fixed: ALTREF,
            var: [LAST, GOLDEN],
        }
    } else if bias[LAST as usize] == bias[ALTREF as usize] {
        CompoundRefs {
            fixed: GOLDEN,
            var: [LAST, ALTREF],
        }
    } else {
        CompoundRefs {
            fixed: LAST,
            var: [GOLDEN, ALTREF],
        }
    }
}

/// §6.3.16 mv_probs — joints + per-component fields.
fn read_mv_probs(
    bd: &mut BoolDecoder<'_>,
    ctx: &mut FrameContext,
    allow_high_precision_mv: bool,
) -> Result<()> {
    for j in 0..3 {
        ctx.mv_probs.joints[j] = update_mv_prob(bd, ctx.mv_probs.joints[j])?;
    }
    for i in 0..2 {
        let c = &mut ctx.mv_probs.comps[i];
        c.sign = update_mv_prob(bd, c.sign)?;
        for j in 0..c.classes.len() {
            c.classes[j] = update_mv_prob(bd, c.classes[j])?;
        }
        c.class0_bit = update_mv_prob(bd, c.class0_bit)?;
        for j in 0..c.bits.len() {
            c.bits[j] = update_mv_prob(bd, c.bits[j])?;
        }
    }
    for i in 0..2 {
        let c = &mut ctx.mv_probs.comps[i];
        for j in 0..c.class0_fr.len() {
            for k in 0..c.class0_fr[j].len() {
                c.class0_fr[j][k] = update_mv_prob(bd, c.class0_fr[j][k])?;
            }
        }
        for k in 0..c.fr.len() {
            c.fr[k] = update_mv_prob(bd, c.fr[k])?;
        }
    }
    if allow_high_precision_mv {
        for i in 0..2 {
            let c = &mut ctx.mv_probs.comps[i];
            c.class0_hp = update_mv_prob(bd, c.class0_hp)?;
            c.hp = update_mv_prob(bd, c.hp)?;
        }
    }
    Ok(())
}

/// Silence unused-import warnings if the feature set is narrowed in
/// the future. `INTRA_MODES` is wired through so UV-mode updates would
/// hang off the same table; not yet invoked.
#[allow(dead_code)]
fn _unused_imports() {
    let _ = INTRA_MODES;
    let _ = TX_SIZES;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_header_has_seeded_context() {
        let ch = CompressedHeader::default();
        assert_eq!(ch.ctx.skip_probs, [192, 128, 64]);
        // coef probs seeded from spec tables.
        assert_ne!(ch.ctx.coef_probs[0][0][0][0][0][0], 0);
    }

    #[test]
    fn all_zero_payload_leaves_probs_untouched() {
        // A bool decoder fed all-zero bytes reads zero bits; `diff_update_prob`
        // sees the B(252) gate return 0 every time and leaves probs alone.
        // That means after a compressed-header parse with an all-zero tail,
        // ctx.skip_probs should equal the §10.5 defaults.
        //
        // We can't easily run the full parser without a matching header,
        // but `diff_update_prob` itself should round-trip a prob through
        // an all-zero bool stream.
        use crate::frame_ctx::diff_update_prob;
        let buf = [0u8; 16];
        let mut bd = BoolDecoder::new(&buf).unwrap();
        let p = diff_update_prob(&mut bd, 192).unwrap();
        assert_eq!(p, 192);
    }
}
