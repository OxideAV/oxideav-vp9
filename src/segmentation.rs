//! VP9 per-block segmentation map (§6.4.7, §6.4.9, §6.4.12, §6.4.14).
//!
//! VP9 lets an encoder partition the frame into up to 8 segments. For
//! each segment the encoder can enable one or more of four features:
//! `SEG_LVL_ALT_Q`, `SEG_LVL_ALT_L`, `SEG_LVL_REF_FRAME`, `SEG_LVL_SKIP`
//! (§6.2.11). The header parser handles the frame-level bits; this
//! module handles the *per-block* segment_id assignment used during
//! block decode.
//!
//! The flow per the spec:
//!
//! * §6.4.7 `intra_segment_id()` — on a key / intra-only frame, if
//!   `segmentation_update_map == 1` read `segment_id` from the boolean
//!   decoder using `segment_tree` (§9.3.2); otherwise `segment_id = 0`.
//! * §6.4.12 `inter_segment_id()` — on an inter frame, look up the
//!   predicted segment id from the previous frame's segment map
//!   (`get_segment_id()` §6.4.14), then either (a) use it directly when
//!   `update_map == 0`, (b) combine with a per-block `seg_id_predicted`
//!   bit when `update_map == 1 && temporal_update == 1`, or (c) always
//!   read a fresh tree-coded segment_id when `update_map == 1 &&
//!   temporal_update == 0`.
//! * §6.4.14 `get_segment_id()` — the predicted id is the min over the
//!   covered region of `PrevSegmentIds`.
//! * §6.4.9 `seg_feature_active(feature)` — lives on `SegmentationParams`.
//!
//! After the frame is decoded the current `SegmentIds` map replaces
//! `PrevSegmentIds` — but only when `update_map == 1` (§8.1 step 3).
//!
//! All probability selection follows §9.3.2 / §6.4.14:
//!
//!   segment_id   -> prob = segmentation_tree_probs[node]
//!   seg_id_pred  -> prob = segmentation_pred_prob[ctx]  (ctx = above+left)

use oxideav_core::Result;

use crate::bool_decoder::BoolDecoder;
use crate::headers::SegmentationParams;

/// Maximum segments allowed by VP9 (§3 table "MAX_SEGMENTS = 8").
pub const MAX_SEGMENTS: usize = 8;

/// A `MiCols × MiRows` 8×8-mi segment-id map. One byte per 8×8 unit.
#[derive(Clone, Debug)]
pub struct SegmentIdMap {
    pub mi_cols: usize,
    pub mi_rows: usize,
    cells: Vec<u8>,
}

impl SegmentIdMap {
    /// New map initialised to `segment_id = 0` — matches the spec's
    /// `setup_past_independence` rule for `PrevSegmentIds` (§8.2).
    pub fn zeros(mi_cols: usize, mi_rows: usize) -> Self {
        Self {
            mi_cols,
            mi_rows,
            cells: vec![0u8; mi_cols.max(1) * mi_rows.max(1)],
        }
    }

    pub fn get(&self, mi_row: usize, mi_col: usize) -> u8 {
        if self.mi_cols == 0 || self.mi_rows == 0 {
            return 0;
        }
        let r = mi_row.min(self.mi_rows - 1);
        let c = mi_col.min(self.mi_cols - 1);
        self.cells[r * self.mi_cols + c]
    }

    /// §6.4.4: stamp `seg` into every 8×8 cell covered by the prediction
    /// block at `(mi_row, mi_col)` with size `bw × bh` in 8×8 units,
    /// clipping against the frame boundary (spec uses `xmis / ymis`).
    pub fn fill(&mut self, mi_row: usize, mi_col: usize, bw: usize, bh: usize, seg: u8) {
        let xmis = bw.min(self.mi_cols.saturating_sub(mi_col));
        let ymis = bh.min(self.mi_rows.saturating_sub(mi_row));
        for dy in 0..ymis {
            let r = mi_row + dy;
            for dx in 0..xmis {
                let c = mi_col + dx;
                self.cells[r * self.mi_cols + c] = seg;
            }
        }
    }

    /// §6.4.14 `get_segment_id()`: predicted id = min over the covered
    /// region of the prior frame's `SegmentIds`. Starts at 7 (spec uses
    /// `seg = 7`), reduces to whatever sits below.
    pub fn predicted_segment_id(
        &self,
        mi_row: usize,
        mi_col: usize,
        bw: usize,
        bh: usize,
    ) -> u8 {
        let xmis = bw.min(self.mi_cols.saturating_sub(mi_col));
        let ymis = bh.min(self.mi_rows.saturating_sub(mi_row));
        let mut seg: u8 = 7;
        for dy in 0..ymis {
            let r = mi_row + dy;
            for dx in 0..xmis {
                let c = mi_col + dx;
                let v = self.cells[r * self.mi_cols + c];
                if v < seg {
                    seg = v;
                }
            }
        }
        seg
    }
}

/// Per-tile above/left seg-id prediction contexts (§6.4.12 /
/// §7.4.1 / §7.4.2). They track per-8x8 the `seg_id_predicted` bit of
/// the most recently decoded neighbour. Cleared per tile.
#[derive(Clone, Debug)]
pub struct SegPredContext {
    /// Above row: one byte per 8×8 mi column (0 or 1).
    pub above: Vec<u8>,
    /// Left column: one byte per 8×8 mi row (0 or 1).
    pub left: Vec<u8>,
}

impl SegPredContext {
    pub fn zeros(mi_cols: usize, mi_rows: usize) -> Self {
        Self {
            above: vec![0u8; mi_cols.max(1)],
            left: vec![0u8; mi_rows.max(1)],
        }
    }

    /// §9.3.2 `seg_id_predicted` ctx = `LeftSegPredContext[MiRow]` +
    /// `AboveSegPredContext[MiCol]`. Out-of-bounds indices default to 0.
    pub fn ctx(&self, mi_row: usize, mi_col: usize) -> usize {
        let a = self.above.get(mi_col).copied().unwrap_or(0) as usize;
        let l = self.left.get(mi_row).copied().unwrap_or(0) as usize;
        a + l
    }

    /// §6.4.12: after reading `seg_id_predicted` stamp it across the
    /// block's above row (`MiCol..MiCol+bw`) and left column
    /// (`MiRow..MiRow+bh`).
    pub fn stamp(&mut self, mi_row: usize, mi_col: usize, bw: usize, bh: usize, v: u8) {
        for dx in 0..bw {
            let c = mi_col + dx;
            if c < self.above.len() {
                self.above[c] = v;
            }
        }
        for dy in 0..bh {
            let r = mi_row + dy;
            if r < self.left.len() {
                self.left[r] = v;
            }
        }
    }
}

/// Decode a tree-coded `segment_id` symbol per §9.3.2 (`segment_tree`
/// §6.4.14). The tree is balanced 3-level — 7 internal nodes, 8 leaves:
///
/// ```text
/// segment_tree[14] = { 2, 4, 6, 8, 10, 12,
///                      0, -1, -2, -3, -4, -5, -6, -7 }
/// ```
///
/// Read bit i uses `tree_probs[i]`. With 7 probs we get 3-level binary
/// dispatch: first decide 0..3 vs 4..7, then the two halves.
pub fn read_segment_id(bd: &mut BoolDecoder<'_>, tree_probs: &[u8; 7]) -> Result<u8> {
    // Emulate the tree walk. The libvpx tree is laid out so that each
    // internal node splits evenly — traverse from the root following
    // bit reads.
    let mut idx: usize = 0; // current tree index, in pairs of 2
    loop {
        let bit = bd.read(tree_probs[idx / 2])? as usize;
        let next = SEGMENT_TREE[idx + bit];
        if next <= 0 {
            // Leaf — segment index is (-next).
            return Ok((-next) as u8);
        }
        idx = next as usize;
    }
}

/// `segment_tree[14]` from §6.4.14 — the 7-internal-node binary tree
/// that maps 3 bit reads to a segment id 0..7.
///
/// Pairs of `(left, right)` entries: non-positive values are leaf
/// segment ids (as negatives of the id), positive values are the index
/// of the next pair to read.
const SEGMENT_TREE: [i8; 14] = [2, 4, 6, 8, 10, 12, 0, -1, -2, -3, -4, -5, -6, -7];

/// §6.4.12 `inter_segment_id()` — pick the segment_id for an inter block.
///
/// * `prev_map` is `PrevSegmentIds` — the previous frame's segment map.
///   When unavailable the spec zeroes it in `setup_past_independence`.
/// * `seg_ctx` tracks the above/left `seg_id_predicted` markers.
/// * `(mi_row, mi_col, bw, bh)` are the block's 8×8 coordinates.
///
/// Returns `segment_id` to stamp into the current-frame map.
#[allow(clippy::too_many_arguments)]
pub fn read_inter_segment_id(
    bd: &mut BoolDecoder<'_>,
    seg: &SegmentationParams,
    prev_map: &SegmentIdMap,
    seg_ctx: &mut SegPredContext,
    mi_row: usize,
    mi_col: usize,
    bw: usize,
    bh: usize,
) -> Result<u8> {
    if !seg.enabled {
        return Ok(0);
    }
    let predicted = prev_map.predicted_segment_id(mi_row, mi_col, bw, bh);
    if !seg.update_map {
        // §6.4.12 else branch: segment_id = predictedSegmentId.
        return Ok(predicted);
    }
    if seg.temporal_update {
        // §6.4.12 temporal_update == 1: read a `seg_id_predicted` bit
        // with ctx = left + above pred context; if set, use predicted,
        // else read a tree-coded id. Then stamp the pred bit.
        let ctx = seg_ctx.ctx(mi_row, mi_col);
        let prob = seg.pred_probs[ctx.min(2)];
        let pred_bit = bd.read(prob)? as u8;
        seg_ctx.stamp(mi_row, mi_col, bw, bh, pred_bit);
        if pred_bit == 1 {
            Ok(predicted)
        } else {
            read_segment_id(bd, &seg.tree_probs)
        }
    } else {
        // §6.4.12 temporal_update == 0: always read a fresh tree id.
        read_segment_id(bd, &seg.tree_probs)
    }
}

/// §6.4.7 `intra_segment_id()` — key / intra-only frames either read a
/// tree-coded segment_id (`update_map == 1`) or default to 0.
pub fn read_intra_segment_id(
    bd: &mut BoolDecoder<'_>,
    seg: &SegmentationParams,
) -> Result<u8> {
    if seg.enabled && seg.update_map {
        read_segment_id(bd, &seg.tree_probs)
    } else {
        Ok(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predicted_is_min_over_region() {
        let mut m = SegmentIdMap::zeros(4, 4);
        m.fill(0, 0, 4, 4, 5);
        m.fill(1, 1, 1, 1, 2);
        assert_eq!(m.predicted_segment_id(0, 0, 4, 4), 2);
        assert_eq!(m.predicted_segment_id(2, 2, 2, 2), 5);
    }

    #[test]
    fn predicted_is_seven_on_empty_overlap() {
        let m = SegmentIdMap::zeros(2, 2);
        // Out-of-bounds starting point: xmis=ymis=0 so the loop runs
        // zero iterations; initial `seg = 7` falls through.
        assert_eq!(m.predicted_segment_id(10, 10, 4, 4), 7);
    }

    #[test]
    fn seg_pred_ctx_sums_neighbours() {
        let mut ctx = SegPredContext::zeros(4, 4);
        ctx.stamp(0, 0, 1, 1, 1); // stamps above[0]=1, left[0]=1
        assert_eq!(ctx.ctx(0, 0), 2);
        assert_eq!(ctx.ctx(0, 1), 1);
        assert_eq!(ctx.ctx(1, 0), 1);
        assert_eq!(ctx.ctx(2, 2), 0);
    }
}
