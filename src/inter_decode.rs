//! Inter-frame decode adapters — the bridge between the frame-wide
//! §6.4.4 state ([`crate::decode_block::Vp9FrameState`]) + the §8.10
//! reference buffers ([`crate::ref_buffer::RefBuffers`]) and the inter
//! decode machinery (the §6.5 `find_mv_refs( )` candidate scan and the
//! §8.5.2 [`crate::inter_pred::predict_inter`] driver).
//!
//! Two adapters land here:
//!
//! 1. [`FrameStateMvSource`] — a [`crate::mv_ref::MvCandidateSource`]
//!    over the current-frame `RefFrames` / `Mvs` / `SubMvs` / `YModes`
//!    arrays plus an optional previous-frame `PrevRefFrames` / `PrevMvs`
//!    snapshot. This is what the §6.4.16 `inter_block_mode_info( )` /
//!    §6.5.1 `find_mv_refs( )` scan reads its neighbour candidates from.
//! 2. [`build_ref_planes`] — resolves a per-block `ref_frame[ refList ]`
//!    pair through the §6.2 `ref_frame_idx[ ]` map to §8.10 `FrameStore`
//!    slots and packages the per-plane reference samples + geometry into
//!    the [`crate::inter_pred::RefPlanes`] the §8.5.2 driver consumes.
//!
//! Neither adapter decodes anything itself; they translate the stored
//! frame state into the shapes the §6.5 / §8.5.2 primitives expect, so
//! the full inter-frame decode path can compose
//! `inter_frame_mode_info( )` → `predict_inter( )` without either
//! primitive needing to know about `Vp9FrameState` / `RefBuffers`
//! layout.
//!
//! Single source of truth: `docs/video/vp9/vp9-spec.txt` §6.5.10-11
//! (candidate loads), §8.5.2.3 / §8.5.2.4 (refIdx resolution), §8.10.

// The full inter-frame partition walk that drives these adapters lands
// on top; until then they are reachable only from the unit tests, so
// the crate-internal `dead_code` lint is silenced module-wide.
#![allow(dead_code)]

use crate::decode_block::Vp9FrameState;
use crate::inter_pred::{RefPlane, RefPlanes};
use crate::mode_info::LAST_FRAME;
use crate::mv_ref::MvCandidateSource;
use crate::ref_buffer::{resolve_ref_idx, RefBuffers, REFS_PER_FRAME};

/// Optional previous-frame motion-field snapshot the §6.5.10 candidate
/// scan reads when `UsePrevFrameMvs` is set.
///
/// `prev_ref_frames` is `PrevRefFrames[ row ][ col ][ list ]` stored
/// `[(row * mi_cols + col) * 2 + list]`; `prev_mvs` is
/// `PrevMvs[ row ][ col ][ list ]` in the same layout. Both span
/// `MiRows × MiCols × 2`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PrevFrameMvs<'a> {
    pub prev_ref_frames: &'a [i32],
    pub prev_mvs: &'a [(i16, i16)],
}

/// A [`MvCandidateSource`] view over a [`Vp9FrameState`] (current frame)
/// plus an optional [`PrevFrameMvs`] (previous frame).
///
/// Out-of-frame reads return the §6.5 "intra / zero" sentinels
/// (`ref_frame = INTRA_FRAME = 0`, `mv = [0, 0]`); the §6.5.2
/// `is_inside( )` gate already excludes those positions before the scan
/// dereferences them, but the defensive default keeps a malformed call
/// from panicking.
pub(crate) struct FrameStateMvSource<'a> {
    state: &'a Vp9FrameState,
    prev: Option<PrevFrameMvs<'a>>,
}

impl<'a> FrameStateMvSource<'a> {
    /// Build the source over the current frame state and an optional
    /// previous-frame motion field.
    pub fn new(state: &'a Vp9FrameState, prev: Option<PrevFrameMvs<'a>>) -> Self {
        Self { state, prev }
    }

    /// Row-major MI index for `(row, col)`, or `None` when out of frame.
    fn idx(&self, r: i32, c: i32) -> Option<usize> {
        if r < 0 || c < 0 {
            return None;
        }
        let (r, c) = (r as u32, c as u32);
        if r < self.state.mi_rows && c < self.state.mi_cols {
            Some((r as usize) * (self.state.mi_cols as usize) + (c as usize))
        } else {
            None
        }
    }
}

impl MvCandidateSource for FrameStateMvSource<'_> {
    fn y_mode(&self, r: i32, c: i32) -> u8 {
        self.idx(r, c).map(|i| self.state.y_modes[i]).unwrap_or(0)
    }

    fn ref_frame(&self, r: i32, c: i32, ref_list: usize) -> i32 {
        self.idx(r, c)
            .map(|i| self.state.ref_frames[i * 2 + ref_list])
            .unwrap_or(0)
    }

    fn mv(&self, r: i32, c: i32, ref_list: usize) -> [i32; 2] {
        match self.idx(r, c) {
            Some(i) => {
                let (row, col) = self.state.mvs[i * 2 + ref_list];
                [row as i32, col as i32]
            }
            None => [0, 0],
        }
    }

    fn sub_mv(&self, r: i32, c: i32, ref_list: usize, idx: usize) -> [i32; 2] {
        match self.idx(r, c) {
            Some(i) => {
                let (row, col) = self.state.sub_mvs[(i * 2 + ref_list) * 4 + idx];
                [row as i32, col as i32]
            }
            None => [0, 0],
        }
    }

    fn prev_ref_frame(&self, r: i32, c: i32, ref_list: usize) -> i32 {
        match (self.prev, self.idx(r, c)) {
            (Some(p), Some(i)) => p.prev_ref_frames[i * 2 + ref_list],
            _ => 0,
        }
    }

    fn prev_mv(&self, r: i32, c: i32, ref_list: usize) -> [i32; 2] {
        match (self.prev, self.idx(r, c)) {
            (Some(p), Some(i)) => {
                let (row, col) = p.prev_mvs[i * 2 + ref_list];
                [row as i32, col as i32]
            }
            _ => [0, 0],
        }
    }
}

/// Resolve the per-block `ref_frame[ refList ]` pair to §8.10
/// `FrameStore` reference planes for one plane, building the
/// [`RefPlanes`] the §8.5.2 [`crate::inter_pred::predict_inter`] driver
/// consumes.
///
/// `ref_frame` is the §6.4 per-block reference pair (`ref_frame[ 0 ]`
/// always `>= LAST_FRAME` on inter blocks; `ref_frame[ 1 ] >= LAST_FRAME`
/// for compound, `< LAST_FRAME` / `NONE` for single reference).
/// `ref_frame_idx` is the frame's §6.2 slot map; `plane` selects which
/// stored plane (0 = luma, 1 = U, 2 = V) the returned [`RefPlane`]
/// borrows.
///
/// Returns `None` for `refs.list[ 1 ]` when the block is single
/// reference (`ref_frame[ 1 ] < LAST_FRAME`).
pub(crate) fn build_ref_planes<'a>(
    bufs: &'a RefBuffers,
    ref_frame: [i32; 2],
    ref_frame_idx: &[u8; REFS_PER_FRAME],
    plane: usize,
) -> RefPlanes<'a> {
    let mut list: [Option<RefPlane<'a>>; 2] = [None, None];
    for (ref_list, slot) in list.iter_mut().enumerate() {
        let rf = ref_frame[ref_list];
        if rf < LAST_FRAME {
            // INTRA_FRAME / NONE — no reference plane for this list.
            continue;
        }
        let ref_idx = resolve_ref_idx(rf, ref_frame_idx);
        let s = bufs.slot(ref_idx);
        let (samples, stride) = match plane {
            0 => (s.y.as_slice(), s.frame_width as usize),
            1 => (s.u.as_slice(), s.chroma_width()),
            _ => (s.v.as_slice(), s.chroma_width()),
        };
        *slot = Some(RefPlane {
            samples,
            stride,
            ref_frame_width: s.frame_width as i32,
            ref_frame_height: s.frame_height as i32,
        });
    }
    RefPlanes { list }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mode_info::{ALTREF_FRAME, GOLDEN_FRAME, INTRA_FRAME, LAST_FRAME, NONE_REF_FRAME};
    use crate::ref_buffer::CurrFramePlanes;

    fn make_state() -> Vp9FrameState {
        let mut state = Vp9FrameState::new(4, 4);
        // Mark MI (1, 1) as an inter block using LAST_FRAME with a known
        // MV, and (0, 1) using GOLDEN_FRAME.
        // MI index = row * mi_cols + col, mi_cols = 4.
        let i11 = 4 + 1;
        state.ref_frames[i11 * 2] = LAST_FRAME;
        state.ref_frames[i11 * 2 + 1] = NONE_REF_FRAME;
        state.mvs[i11 * 2] = (7, -3);
        state.sub_mvs[(i11 * 2) * 4 + 2] = (9, 9);
        state.y_modes[i11] = 13; // NEWMV-ish
        let i01 = 1;
        state.ref_frames[i01 * 2] = GOLDEN_FRAME;
        state.mvs[i01 * 2] = (1, 2);
        state
    }

    #[test]
    fn mv_source_reads_current_frame_arrays() {
        let state = make_state();
        let src = FrameStateMvSource::new(&state, None);
        assert_eq!(src.ref_frame(1, 1, 0), LAST_FRAME);
        assert_eq!(src.ref_frame(1, 1, 1), NONE_REF_FRAME);
        assert_eq!(src.mv(1, 1, 0), [7, -3]);
        assert_eq!(src.sub_mv(1, 1, 0, 2), [9, 9]);
        assert_eq!(src.y_mode(1, 1), 13);
        assert_eq!(src.ref_frame(0, 1, 0), GOLDEN_FRAME);
        assert_eq!(src.mv(0, 1, 0), [1, 2]);
    }

    #[test]
    fn mv_source_out_of_frame_returns_sentinels() {
        let state = make_state();
        let src = FrameStateMvSource::new(&state, None);
        // Negative + past-edge reads return the intra / zero sentinels.
        assert_eq!(src.ref_frame(-1, 0, 0), INTRA_FRAME);
        assert_eq!(src.mv(-1, 0, 0), [0, 0]);
        assert_eq!(src.ref_frame(0, 99, 0), INTRA_FRAME);
        assert_eq!(src.mv(99, 0, 1), [0, 0]);
    }

    #[test]
    fn mv_source_prev_frame_snapshot() {
        let state = make_state();
        let prev_ref = vec![LAST_FRAME; 4 * 4 * 2];
        let mut prev_mvs = vec![(0i16, 0i16); 4 * 4 * 2];
        prev_mvs[(4 + 1) * 2] = (5, 6);
        let src = FrameStateMvSource::new(
            &state,
            Some(PrevFrameMvs {
                prev_ref_frames: &prev_ref,
                prev_mvs: &prev_mvs,
            }),
        );
        assert_eq!(src.prev_ref_frame(1, 1, 0), LAST_FRAME);
        assert_eq!(src.prev_mv(1, 1, 0), [5, 6]);
        // Without prev, the no-snapshot source returns zero.
        let src2 = FrameStateMvSource::new(&state, None);
        assert_eq!(src2.prev_ref_frame(1, 1, 0), 0);
        assert_eq!(src2.prev_mv(1, 1, 0), [0, 0]);
    }

    fn make_bufs() -> RefBuffers {
        let mut bufs = RefBuffers::new();
        // Fill slot 0 with luma 10 / U 20 / V 30; slot 1 with luma 40.
        let y = vec![10i32; 64 * 64];
        let u = vec![20i32; 32 * 32];
        let v = vec![30i32; 32 * 32];
        bufs.update(
            0x01,
            &CurrFramePlanes {
                y: &y,
                y_stride: 64,
                u: &u,
                v: &v,
                uv_stride: 32,
                frame_width: 64,
                frame_height: 64,
                subsampling_x: true,
                subsampling_y: true,
                bit_depth: 8,
            },
        );
        let y1 = vec![40i32; 64 * 64];
        let u1 = vec![41i32; 32 * 32];
        let v1 = vec![42i32; 32 * 32];
        bufs.update(
            0x02,
            &CurrFramePlanes {
                y: &y1,
                y_stride: 64,
                u: &u1,
                v: &v1,
                uv_stride: 32,
                frame_width: 64,
                frame_height: 64,
                subsampling_x: true,
                subsampling_y: true,
                bit_depth: 8,
            },
        );
        bufs
    }

    #[test]
    fn build_ref_planes_single_reference_luma() {
        let bufs = make_bufs();
        // ref_frame_idx maps LAST->slot 0, GOLDEN->slot 1, ALTREF->slot 2.
        let map = [0u8, 1, 2];
        let refs = build_ref_planes(&bufs, [LAST_FRAME, NONE_REF_FRAME], &map, 0);
        let p0 = refs.list[0].expect("luma ref plane");
        assert_eq!(p0.samples[0], 10);
        assert_eq!(p0.stride, 64);
        assert_eq!(p0.ref_frame_width, 64);
        assert_eq!(p0.ref_frame_height, 64);
        assert!(refs.list[1].is_none());
    }

    #[test]
    fn build_ref_planes_chroma_uses_subsampled_plane() {
        let bufs = make_bufs();
        let map = [0u8, 1, 2];
        let refs_u = build_ref_planes(&bufs, [LAST_FRAME, NONE_REF_FRAME], &map, 1);
        let p = refs_u.list[0].expect("U ref plane");
        assert_eq!(p.samples[0], 20);
        assert_eq!(p.stride, 32);
        let refs_v = build_ref_planes(&bufs, [LAST_FRAME, NONE_REF_FRAME], &map, 2);
        assert_eq!(refs_v.list[0].expect("V ref plane").samples[0], 30);
    }

    #[test]
    fn build_ref_planes_compound_resolves_both_lists() {
        let bufs = make_bufs();
        let map = [0u8, 1, 2];
        // Compound: LAST (slot 0, luma 10) + GOLDEN (slot 1, luma 40).
        let refs = build_ref_planes(&bufs, [LAST_FRAME, GOLDEN_FRAME], &map, 0);
        assert_eq!(refs.list[0].expect("list0").samples[0], 10);
        assert_eq!(refs.list[1].expect("list1").samples[0], 40);
    }

    #[test]
    fn build_ref_planes_intra_list_is_none() {
        let bufs = make_bufs();
        let map = [0u8, 1, 2];
        // ref_frame[0] = INTRA_FRAME -> no plane (the §8.5.2 driver is
        // never invoked on an intra block, but the resolver tolerates it).
        let refs = build_ref_planes(&bufs, [INTRA_FRAME, NONE_REF_FRAME], &map, 0);
        assert!(refs.list[0].is_none());
        assert!(refs.list[1].is_none());
        let _ = ALTREF_FRAME; // referenced for the constant import
    }
}
