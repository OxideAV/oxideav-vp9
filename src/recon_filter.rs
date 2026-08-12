//! Encode-side §8.8 loop-filter mirror + frame filter-level election.
//!
//! ## Why the encoder must filter its own reconstruction
//!
//! §8.1 step 2 runs the §8.8 loop filter process on every decoded
//! frame with `loop_filter_level != 0`, and §8.10 stores the
//! **filtered** planes into `FrameStore[ ]` — so the reference frames
//! a decoder predicts from are post-filter. An encoder that codes a
//! non-zero `loop_filter_level` but motion-compensates against its
//! *unfiltered* reconstruction drifts from every conforming decoder on
//! the very next inter frame. [`filter_reconstruction`] therefore
//! replays the identical §8.8 chain the decode path runs — the §8.8.1
//! `loop_filter_frame_init( )` over the §7.2-resolved deltas, then the
//! §8.8 superblock raster ([`frame_loop_filter`]) over the working
//! planes — on the encoder's [`ReconState`], keyed by the writer's own
//! §6.4.4 [`Vp9FrameState`] per-MI arrays (`MiSizes` / `TxSizes` /
//! `Skips` / `YModes` / `SegmentIds` / `RefFrames`), which are
//! bit-identical to the arrays the decoder reconstructs from the coded
//! bytes.
//!
//! ## Filter-level election
//!
//! The frame's `loop_filter_level` is free encoder choice (any value
//! `0..=63` yields a conforming stream; the level is coded as a §6.2.8
//! fixed-width field, so the choice never changes the stream length).
//! [`elect_filter_level`] picks the level that minimises the
//! reconstruction's sample-squared error against the **source** frame
//! over the visible extents: it replays the §8.8 chain on a scratch
//! copy of the reconstruction at every candidate level and keeps the
//! argmin (ties resolve to the smaller level, so flat / lossless-clean
//! content elects 0 and stays untouched). At quantizers that leave
//! visible block-edge discontinuities the elected level is non-zero —
//! the smoothed edges land closer to the source than the raw quantized
//! reconstruction — while content whose detail the filter would smear
//! elects low levels or 0. This is a pure encoder-side search: the
//! decode side just runs §8.8 at whatever level the header carries.
//!
//! ## Provenance
//!
//! VP9 Bitstream & Decoding Process Specification v0.7
//! (`docs/video/vp9/vp9-spec.txt`): §8.1 step 2 (filter gate), §8.8 /
//! §8.8.1 / §8.8.2 (the filter chain), §8.10 (post-filter reference
//! store), §7.2 `setup_past_independence( )` delta defaults, §6.2.8
//! `loop_filter_params( )` fixed-width `filter_level` field.

use crate::decode_block::Vp9FrameState;
use crate::frame_loop_filter::{frame_loop_filter, CurrFrame};
use crate::header::Vp9FrameHeader;
use crate::intra::Plane;
use crate::loop_filter::{loop_filter_frame_init, resolve_lf_deltas, MAX_LOOP_FILTER};
use crate::pixel_encoder::ReconState;
use crate::superblock_loop_filter::{SuperblockFilterFrame, SuperblockFilterPlane};

/// Borrow-free per-MI copies of the [`Vp9FrameState`] columns the §8.8
/// processes read, in the exact shapes [`SuperblockFilterFrame`] wants
/// (`skips` as `bool`, `ref_frames` reduced to `RefFrames[ ][ ][ 0 ]`).
struct FilterMiArrays {
    skips: Vec<bool>,
    ref_frames_0: Vec<i32>,
}

impl FilterMiArrays {
    fn from_state(state: &Vp9FrameState) -> Self {
        Self {
            skips: state.skips.iter().map(|&s| s != 0).collect(),
            ref_frames_0: state.ref_frames.chunks_exact(2).map(|p| p[0]).collect(),
        }
    }
}

/// Run the three-plane §8.8 raster over `planes` in place at the given
/// `LvlLookup` — the shared kernel under [`filter_reconstruction`] and
/// the election's per-candidate probes.
fn run_filter(
    planes: &mut [Plane; 3],
    arrays: &FilterMiArrays,
    state: &Vp9FrameState,
    hdr: &Vp9FrameHeader,
    lvl_lookup: &crate::loop_filter::LvlLookup,
) {
    let [y, u, v] = planes;
    let (y_w, y_h) = (y.width(), y.height());
    let (u_w, u_h) = (u.width(), u.height());
    let (v_w, v_h) = (v.width(), v.height());
    let filter_frame = SuperblockFilterFrame {
        mi_sizes: &state.mi_sizes,
        tx_sizes: &state.tx_sizes,
        skips: &arrays.skips,
        ref_frames_0: &arrays.ref_frames_0,
        y_modes: &state.y_modes,
        segment_ids: &state.segment_ids,
        mi_cols: state.mi_cols,
        mi_rows: state.mi_rows,
        subsampling_x: u8::from(hdr.color_config.subsampling_x),
        subsampling_y: u8::from(hdr.color_config.subsampling_y),
        loop_filter_sharpness: hdr.loop_filter.sharpness,
        bit_depth: hdr.color_config.bit_depth,
        lvl_lookup,
    };
    let mut curr = CurrFrame {
        planes: [
            SuperblockFilterPlane {
                data: y.samples_mut(),
                stride: y_w,
                width: y_w,
                height: y_h,
            },
            SuperblockFilterPlane {
                data: u.samples_mut(),
                stride: u_w,
                width: u_w,
                height: u_h,
            },
            SuperblockFilterPlane {
                data: v.samples_mut(),
                stride: v_w,
                width: v_w,
                height: v_h,
            },
        ],
    };
    frame_loop_filter(&mut curr, &filter_frame);
}

/// Apply the decode path's §8.1 step 2 / §8.8 loop filter to the
/// encoder's reconstruction, in place — the exact transformation every
/// conforming decoder applies to these coded bytes before §8.10 stores
/// the frame as a reference.
///
/// * `recon` — the encoder's decoder-mirror reconstruction (the §6.4
///   working planes at the MI-padded extents), modified in place.
/// * `state` — the writer's final §6.4.4 per-MI arrays for the frame
///   (from the `_with_state` assemblers), bit-identical to the
///   decoder's.
/// * `hdr` — the frame header actually coded; `hdr.loop_filter.level`,
///   `.sharpness`, the §6.2.8 delta slots and `hdr.segmentation` drive
///   the §8.8.1 init exactly as on the decode side. Per §8.1 step 2 a
///   zero `loop_filter_level` skips the process entirely (the §8.8.1
///   deltas can NOT lift a zero frame level into filtering).
///
/// The §7.2 delta resolution here is defaults + this header's coded
/// updates — correct wherever no *earlier* frame's `delta_update`
/// persists (keyframes, error-resilient chains, and the first update
/// of a chain). Chain-framing callers whose previous frames coded
/// updates must use [`filter_reconstruction_with_deltas`] with the
/// §7.2.8-folded values instead.
pub(crate) fn filter_reconstruction(
    recon: &mut ReconState,
    state: &Vp9FrameState,
    hdr: &Vp9FrameHeader,
) {
    let (ref_deltas, mode_deltas) = resolve_lf_deltas(&hdr.loop_filter);
    filter_reconstruction_with_deltas(recon, state, hdr, ref_deltas, mode_deltas);
}

/// [`filter_reconstruction`] with the §7.2.8 **resolved** delta arrays
/// supplied by the caller — the form the chain-framing sequence
/// encoders use, where a previous frame's coded `delta_update` leaves
/// persistent values a bare header resolve (defaults + this frame's
/// updates) would miss. The values must equal what the decoder's
/// persistent-delta fold yields for this frame.
pub(crate) fn filter_reconstruction_with_deltas(
    recon: &mut ReconState,
    state: &Vp9FrameState,
    hdr: &Vp9FrameHeader,
    ref_deltas: [i8; 4],
    mode_deltas: [i8; 2],
) {
    if hdr.loop_filter.level == 0 {
        return;
    }
    let lvl_lookup =
        loop_filter_frame_init(&hdr.loop_filter, &hdr.segmentation, ref_deltas, mode_deltas);
    let arrays = FilterMiArrays::from_state(state);
    run_filter(&mut recon.planes, &arrays, state, hdr, &lvl_lookup);
}

/// Sample-squared error of `planes` against `targets` over the visible
/// `vis_w x vis_h` luma window (chroma windows at the header's §8.10
/// subsampled extents). The MI-padding overhang is excluded: it never
/// reaches the display output and only its clamped reads feed inter
/// prediction.
fn visible_sse(
    planes: &[Plane; 3],
    targets: &[Plane; 3],
    vis_w: usize,
    vis_h: usize,
    ssx: bool,
    ssy: bool,
) -> u64 {
    let mut sse = 0u64;
    for (plane_idx, (p, t)) in planes.iter().zip(targets.iter()).enumerate() {
        let (w, h) = if plane_idx == 0 {
            (vis_w, vis_h)
        } else {
            (
                (vis_w + usize::from(ssx)) >> usize::from(ssx),
                (vis_h + usize::from(ssy)) >> usize::from(ssy),
            )
        };
        for y in 0..h {
            for x in 0..w {
                let d = i64::from(p.get(x, y)) - i64::from(t.get(x, y));
                sse += (d * d) as u64;
            }
        }
    }
    sse
}

/// Elect the frame `loop_filter_level` (`0..=MAX_LOOP_FILTER`) that
/// minimises the filtered reconstruction's sample-squared error against
/// the source over the visible extents — pure encoder freedom (§6.2.8
/// codes the level as a fixed-width field; every choice is conforming
/// and the stream length is unchanged).
///
/// Sweeps every candidate level, replaying the full §8.8 chain (the
/// §8.8.1 init at the candidate level with the same resolved deltas /
/// segmentation the coded header will carry, then the §8.8 raster) on
/// a scratch copy of `recon`. Ties resolve to the **smaller** level, so
/// content the filter cannot improve — flat areas, lossless-clean
/// reconstructions — elects 0.
///
/// * `recon` — the frame's unfiltered reconstruction (not modified).
/// * `state` — the writer's final §6.4.4 per-MI arrays.
/// * `hdr` — the header to be coded; its `loop_filter.level` is ignored
///   (each candidate is substituted), everything else — sharpness,
///   delta slots, segmentation — is scored exactly as it will decode.
/// * `targets` — the MI-padded source planes the encoder quantized.
/// * `vis_w` / `vis_h` — visible luma extents (`FrameWidth` /
///   `FrameHeight`).
// Level-only convenience over [`elect_filter_params`]; the non-test
// encoders elect the full pair, so only tests call this.
#[allow(dead_code)]
pub(crate) fn elect_filter_level(
    recon: &ReconState,
    state: &Vp9FrameState,
    hdr: &Vp9FrameHeader,
    targets: &[Plane; 3],
    vis_w: usize,
    vis_h: usize,
) -> u8 {
    elect_filter_params(recon, state, hdr, targets, vis_w, vis_h).0
}

/// [`elect_filter_level`] extended over the second free §6.2.8 axis:
/// elect the `(loop_filter_level, loop_filter_sharpness)` pair (both
/// fixed-width fields — the whole election is rate-free).
///
/// Two-stage search keeping the probe count bounded at 64 + 7 full
/// §8.8 replays: sweep every level `0..=63` at the header's own
/// sharpness first (ties toward the smaller level, so unimprovable
/// content elects `(0, hdr.sharpness)` and stays untouched), then —
/// when a non-zero level won — sweep the remaining sharpness values
/// `0..=7` at that level (ties toward the header's own sharpness). The
/// §8.8.4 sharpness derivation only tightens the per-edge `limit`
/// (`limit = Clip3( 1, 9 - sharpness, lvl >> shift )` instead of
/// `Max( 1, lvl )`), so it trades smoothing strength for detail
/// retention along strong edges; the stage-2 sweep picks whichever
/// trade lands closer to the source.
pub(crate) fn elect_filter_params(
    recon: &ReconState,
    state: &Vp9FrameState,
    hdr: &Vp9FrameHeader,
    targets: &[Plane; 3],
    vis_w: usize,
    vis_h: usize,
) -> (u8, u8) {
    let (ref_deltas, mode_deltas) = resolve_lf_deltas(&hdr.loop_filter);
    elect_filter_params_at(
        recon,
        state,
        hdr,
        targets,
        vis_w,
        vis_h,
        ref_deltas,
        mode_deltas,
    )
}

/// [`elect_filter_params`] scored at caller-supplied **resolved**
/// §7.2.8 delta arrays (the chain-framing baseline: a previous frame's
/// coded updates persist, so the probes must run at the values the
/// decoder will actually hold for this frame).
#[allow(clippy::too_many_arguments)]
pub(crate) fn elect_filter_params_at(
    recon: &ReconState,
    state: &Vp9FrameState,
    hdr: &Vp9FrameHeader,
    targets: &[Plane; 3],
    vis_w: usize,
    vis_h: usize,
    ref_deltas: [i8; 4],
    mode_deltas: [i8; 2],
) -> (u8, u8) {
    let ssx = hdr.color_config.subsampling_x;
    let ssy = hdr.color_config.subsampling_y;
    let arrays = FilterMiArrays::from_state(state);

    let probe_sse = |cand_hdr: &Vp9FrameHeader| -> u64 {
        let lvl_lookup = loop_filter_frame_init(
            &cand_hdr.loop_filter,
            &cand_hdr.segmentation,
            ref_deltas,
            mode_deltas,
        );
        let mut probe = recon.planes.clone();
        run_filter(&mut probe, &arrays, state, cand_hdr, &lvl_lookup);
        visible_sse(&probe, targets, vis_w, vis_h, ssx, ssy)
    };

    // Stage 1 — level sweep at the header's sharpness. Level 0: §8.1
    // step 2 skips the filter — the unfiltered baseline.
    let mut best_level = 0u8;
    let mut best_sse = visible_sse(&recon.planes, targets, vis_w, vis_h, ssx, ssy);
    let mut cand_hdr = *hdr;
    for level in 1..=MAX_LOOP_FILTER as u8 {
        cand_hdr.loop_filter.level = level;
        let sse = probe_sse(&cand_hdr);
        if sse < best_sse {
            best_sse = sse;
            best_level = level;
        }
    }
    if best_level == 0 {
        // No level improves at any strength-limit trade the header's
        // sharpness allows; keep the header's own sharpness (the field
        // is coded either way).
        return (0, hdr.loop_filter.sharpness);
    }

    // Stage 2 — sharpness sweep at the winning level.
    let mut best_sharpness = hdr.loop_filter.sharpness;
    cand_hdr.loop_filter.level = best_level;
    for sharpness in 0..=7u8 {
        if sharpness == hdr.loop_filter.sharpness {
            continue;
        }
        cand_hdr.loop_filter.sharpness = sharpness;
        let sse = probe_sse(&cand_hdr);
        if sse < best_sse {
            best_sse = sse;
            best_sharpness = sharpness;
        }
    }
    (best_level, best_sharpness)
}

/// Per-delta candidate values the §6.2.8 delta election probes on each
/// axis — a coarse-to-fine ladder over the s(6)-codeable `-63..=63`
/// range keeping the search bounded (6 axes x ≤11 probes of a full
/// §8.8 replay).
const LF_DELTA_CANDIDATES: [i8; 11] = [-16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16];

/// Elect the §6.2.8 / §8.8.1 **loop-filter delta** arrays — per
/// reference frame (`loop_filter_ref_deltas[ 4 ]`: INTRA / LAST /
/// GOLDEN / ALTREF) and per mode class (`loop_filter_mode_deltas[ 2 ]`:
/// `ZEROMV` vs other inter modes) — that minimise the filtered
/// reconstruction's SSE against the source at the already-elected
/// `(level, sharpness)` in `hdr`.
///
/// §8.8.1 derives each block's filter strength as `lvlSeg +
/// (ref_deltas[ ref ] << nShift) + (mode_deltas[ mode ] << nShift)`
/// (intra blocks take only the INTRA ref delta), so on a frame whose
/// block classes want *different* strengths — static `ZEROMV` regions
/// that any smoothing degrades next to moving `NEWMV` regions the
/// filter helps — the deltas reach per-class strengths the single
/// frame level cannot express. Unlike the level/sharpness fields the
/// deltas are **not** rate-free: a moved slot codes a §6.2.8 update
/// (1 + 7 bits), so the caller only codes slots that changed and the
/// election accepts only strict SSE wins.
///
/// One pass of per-axis coordinate descent from `baseline` (the §7.2.8
/// resolved persistent values for this frame), each axis sweeping
/// [`LF_DELTA_CANDIDATES`]; ties keep the baseline.
#[allow(clippy::too_many_arguments)]
pub(crate) fn elect_lf_deltas(
    recon: &ReconState,
    state: &Vp9FrameState,
    hdr: &Vp9FrameHeader,
    targets: &[Plane; 3],
    vis_w: usize,
    vis_h: usize,
    baseline_ref: [i8; 4],
    baseline_mode: [i8; 2],
) -> ([i8; 4], [i8; 2]) {
    debug_assert!(hdr.loop_filter.level > 0, "deltas are dead at level 0");
    let ssx = hdr.color_config.subsampling_x;
    let ssy = hdr.color_config.subsampling_y;
    let arrays = FilterMiArrays::from_state(state);

    let probe_sse = |ref_deltas: [i8; 4], mode_deltas: [i8; 2]| -> u64 {
        let lvl_lookup =
            loop_filter_frame_init(&hdr.loop_filter, &hdr.segmentation, ref_deltas, mode_deltas);
        let mut probe = recon.planes.clone();
        run_filter(&mut probe, &arrays, state, hdr, &lvl_lookup);
        visible_sse(&probe, targets, vis_w, vis_h, ssx, ssy)
    };

    let mut best_ref = baseline_ref;
    let mut best_mode = baseline_mode;
    let mut best_sse = probe_sse(best_ref, best_mode);

    for axis in 0..6usize {
        for &cand in &LF_DELTA_CANDIDATES {
            let mut ref_d = best_ref;
            let mut mode_d = best_mode;
            let slot = if axis < 4 {
                &mut ref_d[axis]
            } else {
                &mut mode_d[axis - 4]
            };
            if *slot == cand {
                continue;
            }
            *slot = cand;
            let sse = probe_sse(ref_d, mode_d);
            if sse < best_sse {
                best_sse = sse;
                best_ref = ref_d;
                best_mode = mode_d;
            }
        }
    }
    (best_ref, best_mode)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode_frame::decode_vp9_sequence;
    use crate::header::QuantizationParams;
    use crate::pixel_encoder::{
        encode_keyframe_lossy_tree_with_state, encode_pframe_lossy_tree_motion_with_state,
        lossless_pframe_header, lossy_keyframe_header_420, padded_plane_from_bytes,
        plan_keyframe_tree, PFRAME_SEARCH_RANGE,
    };

    /// Deterministic textured 4:2:0 planar frame — enough spatial
    /// structure that a mid-range quantizer leaves visible block-edge
    /// steps for the filter to work on.
    fn textured_planar(w: usize, h: usize, seed: i64) -> Vec<u8> {
        let cw = w.div_ceil(2);
        let ch = h.div_ceil(2);
        let mut px = Vec::with_capacity(w * h + 2 * cw * ch);
        let f = |x: i64, y: i64| -> u8 {
            ((((x + seed) * 7 + y * 13) % 61) * 4 + ((x + seed) * y) % 19) as u8
        };
        for y in 0..h as i64 {
            for x in 0..w as i64 {
                px.push(f(x, y));
            }
        }
        for y in 0..ch as i64 {
            for x in 0..cw as i64 {
                px.push(f(x + 40, y + seed));
            }
        }
        for y in 0..ch as i64 {
            for x in 0..cw as i64 {
                px.push(f(x + seed, y + 70));
            }
        }
        px
    }

    fn padded_targets(px: &[u8], w: usize, h: usize) -> [Plane; 3] {
        let cw = w.div_ceil(2);
        let ch = h.div_ceil(2);
        let y_w = w.div_ceil(8) * 8;
        let y_h = h.div_ceil(8) * 8;
        [
            padded_plane_from_bytes(&px[..w * h], w, h, y_w, y_h),
            padded_plane_from_bytes(&px[w * h..w * h + cw * ch], cw, ch, y_w / 2, y_h / 2),
            padded_plane_from_bytes(&px[w * h + cw * ch..], cw, ch, y_w / 2, y_h / 2),
        ]
    }

    /// Encode a lossy keyframe at `q` with the given coded filter
    /// `level` / `sharpness`, returning (bytes, recon, state).
    fn encode_kf_sharp(
        px: &[u8],
        w: u32,
        h: u32,
        q: u8,
        level: u8,
        sharpness: u8,
    ) -> (Vec<u8>, ReconState, Vp9FrameState) {
        let targets = padded_targets(px, w as usize, h as usize);
        let mut hdr = lossy_keyframe_header_420(w, h, q);
        hdr.loop_filter.level = level;
        hdr.loop_filter.sharpness = sharpness;
        let plan = plan_keyframe_tree(&targets, (h + 7) >> 3, (w + 7) >> 3, true, true, 8, q);
        encode_keyframe_lossy_tree_with_state(&hdr, &targets, &plan).expect("kf encode")
    }

    /// [`encode_kf_sharp`] at sharpness 0.
    fn encode_kf(
        px: &[u8],
        w: u32,
        h: u32,
        q: u8,
        level: u8,
    ) -> (Vec<u8>, ReconState, Vp9FrameState) {
        encode_kf_sharp(px, w, h, q, level, 0)
    }

    /// §8.1 step 2: a zero coded level skips the whole §8.8 process —
    /// [`filter_reconstruction`] must leave every sample untouched.
    #[test]
    fn level_zero_is_a_no_op() {
        let (w, h) = (48u32, 32u32);
        let px = textured_planar(w as usize, h as usize, 3);
        let (_, mut recon, state) = encode_kf(&px, w, h, 80, 0);
        let before: Vec<Vec<i32>> = recon.planes.iter().map(|p| p.samples().to_vec()).collect();
        let hdr = lossy_keyframe_header_420(w, h, 80);
        filter_reconstruction(&mut recon, &state, &hdr);
        for (p, b) in recon.planes.iter().zip(&before) {
            assert_eq!(p.samples(), b.as_slice());
        }
    }

    /// Gentle low-amplitude 4:2:0 content: a slow diagonal ramp whose
    /// coarse-q reconstruction carries small block-edge steps — inside
    /// the §8.8.5.1 filterMask thresholds, so the filter engages even
    /// at low levels.
    fn gentle_planar(w: usize, h: usize) -> Vec<u8> {
        let cw = w.div_ceil(2);
        let ch = h.div_ceil(2);
        let mut px = Vec::with_capacity(w * h + 2 * cw * ch);
        for y in 0..h as i64 {
            for x in 0..w as i64 {
                px.push((110 + (x * 3 + y * 2) / 4 % 40) as u8);
            }
        }
        for y in 0..ch as i64 {
            for x in 0..cw as i64 {
                px.push((90 + (x + y * 3) / 3 % 30) as u8);
            }
        }
        for y in 0..ch as i64 {
            for x in 0..cw as i64 {
                px.push((140 + (x * 2 + y) / 5 % 24) as u8);
            }
        }
        px
    }

    /// Keyframe mirror: a lossy keyframe coded with a non-zero
    /// `loop_filter_level`, filtered through the encode-side §8.8
    /// chain, equals the decoder's output sample-exactly on every
    /// plane — including the §8.8.1 INTRA ref-delta (`+1 << nShift`)
    /// path every intra block takes.
    #[test]
    fn keyframe_filtered_recon_equals_decoder_output() {
        let (w, h) = (48u32, 40u32);
        let px = gentle_planar(w as usize, h as usize);
        let mut any_level_moved_samples = false;
        for level in [8u8, 32, 63] {
            let (bytes, mut recon, state) = encode_kf(&px, w, h, 140, level);
            let (_, recon_raw, _) = encode_kf(&px, w, h, 140, level);
            let mut hdr = lossy_keyframe_header_420(w, h, 140);
            hdr.loop_filter.level = level;
            filter_reconstruction(&mut recon, &state, &hdr);
            any_level_moved_samples |= recon.planes[0].samples() != recon_raw.planes[0].samples();

            let decoded = crate::decode_intra_frame(&bytes).expect("decode");
            let (vis_w, vis_h) = (w as usize, h as usize);
            for row in 0..vis_h {
                for col in 0..vis_w {
                    assert_eq!(
                        i32::from(decoded.y[row * vis_w + col]),
                        recon.planes[0].get(col, row),
                        "level {level}: luma mirror ({col},{row})"
                    );
                }
            }
            let (cw, ch) = (vis_w.div_ceil(2), vis_h.div_ceil(2));
            for (plane, samples) in [(1usize, &decoded.u), (2usize, &decoded.v)] {
                for row in 0..ch {
                    for col in 0..cw {
                        assert_eq!(
                            i32::from(samples[row * cw + col]),
                            recon.planes[plane].get(col, row),
                            "level {level}: chroma {plane} mirror ({col},{row})"
                        );
                    }
                }
            }
        }
        // The filter must actually have moved samples somewhere in the
        // tested level set (otherwise the mirror is vacuous).
        assert!(
            any_level_moved_samples,
            "expected at least one tested level to change the luma plane"
        );
    }

    /// P-frame mirror: a lossy P GOP whose inter frame codes a non-zero
    /// level — the filtered recon equals the sequence decoder's output,
    /// pinning the §8.8.4 inter path (LAST ref-delta 0, GOLDEN/ALTREF
    /// -1, mode deltas) through the encode-side chain.
    #[test]
    fn pframe_filtered_recon_equals_sequence_decoder_output() {
        let (w, h) = (48u32, 32u32);
        let cw = w.div_ceil(2) as usize;
        let ch = h.div_ceil(2) as usize;
        let f0 = textured_planar(w as usize, h as usize, 0);
        let f1 = textured_planar(w as usize, h as usize, 2);

        // Lossless keyframe: the reference is the exact texture.
        let kf = crate::encode_vp9(&f0, w, h).expect("lossless kf");

        let targets = padded_targets(&f1, w as usize, h as usize);
        let ref_y: Vec<i32> = f0[..(w * h) as usize]
            .iter()
            .map(|&b| i32::from(b))
            .collect();
        let ref_u: Vec<i32> = f0[(w * h) as usize..(w * h) as usize + cw * ch]
            .iter()
            .map(|&b| i32::from(b))
            .collect();
        let ref_v: Vec<i32> = f0[(w * h) as usize + cw * ch..]
            .iter()
            .map(|&b| i32::from(b))
            .collect();
        let reference: [(&[i32], usize); 3] = [
            (ref_y.as_slice(), w as usize),
            (ref_u.as_slice(), cw),
            (ref_v.as_slice(), cw),
        ];

        for level in [10u8, 40] {
            let mut hdr = lossless_pframe_header(w, h);
            hdr.loop_filter.level = level;
            hdr.quantization = QuantizationParams {
                base_q_idx: 90,
                delta_q_y_dc: 0,
                delta_q_uv_dc: 0,
                delta_q_uv_ac: 0,
                lossless: false,
            };
            let (p1, mut recon, state) = encode_pframe_lossy_tree_motion_with_state(
                &hdr,
                &targets,
                &reference,
                None,
                w,
                h,
                PFRAME_SEARCH_RANGE,
                true,
                true,
                None,
            )
            .expect("p1 encode");
            filter_reconstruction(&mut recon, &state, &hdr);

            let decoded = decode_vp9_sequence(&[kf.as_slice(), p1.as_slice()]).expect("decode");
            assert_eq!(decoded.len(), 2);
            let out = &decoded[1];
            for row in 0..h as usize {
                for col in 0..w as usize {
                    assert_eq!(
                        i32::from(out.y[row * w as usize + col]),
                        recon.planes[0].get(col, row),
                        "level {level}: luma mirror ({col},{row})"
                    );
                }
            }
            for (plane, samples) in [(1usize, &out.u), (2usize, &out.v)] {
                for row in 0..ch {
                    for col in 0..cw {
                        assert_eq!(
                            i32::from(samples[row * cw + col]),
                            recon.planes[plane].get(col, row),
                            "level {level}: chroma {plane} mirror ({col},{row})"
                        );
                    }
                }
            }
        }
    }

    /// A lossless-clean reconstruction elects level 0: filtering exact
    /// content can only move it away from the source, and ties resolve
    /// downward.
    #[test]
    fn election_on_exact_recon_is_zero() {
        let (w, h) = (48u32, 32u32);
        let px = textured_planar(w as usize, h as usize, 1);
        // q=1 quantizes very finely; the recon is near-exact and every
        // non-zero level's smoothing strictly loses.
        let (_, recon, state) = encode_kf(&px, w, h, 1, 0);
        let hdr = lossy_keyframe_header_420(w, h, 1);
        let targets = padded_targets(&px, w as usize, h as usize);
        let level = elect_filter_level(&recon, &state, &hdr, &targets, w as usize, h as usize);
        assert_eq!(level, 0);
    }

    /// At a coarse quantizer on textured content the election picks a
    /// non-zero level and the filtered reconstruction lands strictly
    /// closer to the source than the raw one.
    #[test]
    fn election_improves_coarse_quantized_content() {
        let (w, h) = (64u32, 48u32);
        let px = textured_planar(w as usize, h as usize, 7);
        let q = 160u8;
        let (_, recon, state) = encode_kf(&px, w, h, q, 0);
        let hdr = lossy_keyframe_header_420(w, h, q);
        let targets = padded_targets(&px, w as usize, h as usize);
        let level = elect_filter_level(&recon, &state, &hdr, &targets, w as usize, h as usize);
        assert!(
            level > 0,
            "coarse-q textured content should elect filtering"
        );

        let sse_raw = visible_sse(&recon.planes, &targets, w as usize, h as usize, true, true);
        let mut hdr2 = hdr;
        hdr2.loop_filter.level = level;
        let mut filtered = recon;
        filter_reconstruction(&mut filtered, &state, &hdr2);
        let sse_filtered = visible_sse(
            &filtered.planes,
            &targets,
            w as usize,
            h as usize,
            true,
            true,
        );
        assert!(
            sse_filtered < sse_raw,
            "elected level {level} must strictly reduce SSE ({sse_filtered} vs {sse_raw})"
        );
    }

    /// The election is a pure function of its inputs (the corpus
    /// staging and the byte-determinism pins rely on it).
    #[test]
    fn election_is_deterministic() {
        let (w, h) = (48u32, 32u32);
        let px = textured_planar(w as usize, h as usize, 9);
        let (_, recon, state) = encode_kf(&px, w, h, 120, 0);
        let hdr = lossy_keyframe_header_420(w, h, 120);
        let targets = padded_targets(&px, w as usize, h as usize);
        let a = elect_filter_params(&recon, &state, &hdr, &targets, w as usize, h as usize);
        let b = elect_filter_params(&recon, &state, &hdr, &targets, w as usize, h as usize);
        assert_eq!(a, b);
    }

    /// Keyframe mirror at non-zero `loop_filter_sharpness`: the §8.8.4
    /// `shift` / `Clip3( 1, 9 - sharpness, … )` limit derivation runs
    /// through the encode-side chain exactly as through the decoder —
    /// and a sharpness change demonstrably alters the filtered output
    /// (the mirror is not vacuous in the sharpness axis).
    #[test]
    fn keyframe_filtered_recon_mirrors_decoder_at_nonzero_sharpness() {
        let (w, h) = (48u32, 40u32);
        let px = gentle_planar(w as usize, h as usize);
        let mut outputs: Vec<Vec<i32>> = Vec::new();
        for sharpness in [0u8, 3, 7] {
            let (bytes, mut recon, state) = encode_kf_sharp(&px, w, h, 140, 32, sharpness);
            let mut hdr = lossy_keyframe_header_420(w, h, 140);
            hdr.loop_filter.level = 32;
            hdr.loop_filter.sharpness = sharpness;
            filter_reconstruction(&mut recon, &state, &hdr);

            let decoded = crate::decode_intra_frame(&bytes).expect("decode");
            let (vis_w, vis_h) = (w as usize, h as usize);
            for row in 0..vis_h {
                for col in 0..vis_w {
                    assert_eq!(
                        i32::from(decoded.y[row * vis_w + col]),
                        recon.planes[0].get(col, row),
                        "sharpness {sharpness}: luma mirror ({col},{row})"
                    );
                }
            }
            outputs.push(recon.planes[0].samples().to_vec());
        }
        assert_ne!(
            outputs[0], outputs[2],
            "sharpness 0 vs 7 must alter the filtered plane on this content"
        );
    }

    /// The two-stage `(level, sharpness)` election never lands worse
    /// than the level-only election at the header's sharpness — stage
    /// 2 only moves on a strict SSE win.
    #[test]
    fn sharpness_election_never_regresses_the_level_election() {
        let (w, h) = (64u32, 48u32);
        let px = textured_planar(w as usize, h as usize, 11);
        let q = 160u8;
        let (_, recon, state) = encode_kf(&px, w, h, q, 0);
        let hdr = lossy_keyframe_header_420(w, h, q);
        let targets = padded_targets(&px, w as usize, h as usize);
        let (level, sharpness) =
            elect_filter_params(&recon, &state, &hdr, &targets, w as usize, h as usize);
        assert!(
            level > 0,
            "coarse-q textured content should elect filtering"
        );

        let sse_at = |lv: u8, sh: u8| -> u64 {
            let mut hdr2 = hdr;
            hdr2.loop_filter.level = lv;
            hdr2.loop_filter.sharpness = sh;
            let mut probe = ReconState {
                planes: recon.planes.clone(),
                mi_cols: recon.mi_cols,
                mi_rows: recon.mi_rows,
                subsampling_x: recon.subsampling_x,
                subsampling_y: recon.subsampling_y,
                bit_depth: recon.bit_depth,
            };
            filter_reconstruction(&mut probe, &state, &hdr2);
            visible_sse(&probe.planes, &targets, w as usize, h as usize, true, true)
        };
        assert!(
            sse_at(level, sharpness) <= sse_at(level, hdr.loop_filter.sharpness),
            "stage-2 sharpness must not regress the stage-1 winner"
        );
    }
}
