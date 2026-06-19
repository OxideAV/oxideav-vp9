# Changelog

All notable changes to `oxideav-vp9` are recorded here.

## [Unreleased]

### Other

- end-to-end inter-frame (P-frame) decode — the §6.4.4 `decode_block( )` inter arm is wired into the §6.4.3 partition walk: the §6.4.11 `inter_frame_mode_info( )` prelude + §6.4.5 dispatch (§6.4.16 `inter_block_mode_info( )` / §6.4.15 `intra_block_mode_info( )`), the §6.4.21 `residual( )` inter arm that runs §8.5.2 `predict_inter( )` per plane (single + compound; sub-8x8 per-4x4 + `>= BLOCK_8X8` whole-plane) before the §6.4.24 token / §8.6.2 reconstruct loop, and the §6.4.4 fan-out of `RefFrames` / `Mvs` / `SubMvs` / `InterpFilters` into the frame-wide arrays. The new public `decode_vp9_sequence( )` multi-frame driver threads the §8.10 `RefBuffers` update (`refresh_frame_flags`-masked, post-loop-filter), the inherited color config, the §6.5 previous-frame motion field, the §7.2.6 `UsePrevFrameMvs` derivation, the §6.4.14 previous-segment map, and `show_existing_frame` re-emit between frames. Validated byte-exact against the `i-frame-then-p-frame-64x64` corpus fixture — both the keyframe and the single-reference-LAST high-precision-MV P-frame reconstruct to `expected.yuv` with 0 differing bytes across all planes (round 343)
- inter-frame decode adapters — `FrameStateMvSource` (a §6.5 `MvCandidateSource` over the current-frame `RefFrames` / `Mvs` / `SubMvs` / `YModes` arrays plus an optional previous-frame `PrevRefFrames` / `PrevMvs` snapshot) and `build_ref_planes` (resolves a per-block `ref_frame[ refList ]` pair through the §6.2 `ref_frame_idx[ ]` map to §8.10 `FrameStore` slots and packages the per-plane samples + geometry into the §8.5.2 `RefPlanes`). These bridge the frame-wide §6.4.4 state + §8.10 buffers to the §6.5 MV-reference scan and the §8.5.2 driver so the full inter path can compose `inter_frame_mode_info( )` → `predict_inter( )` (round 339)
- §6.2 inter (non-intra-only) uncompressed-header parse — `parse_uncompressed_header_with_refs( )` walks the inter-frame branch the parser previously rejected: `ref_frame_idx[ 3 ]` / `ref_frame_sign_bias`, §6.2.5 `frame_size_with_refs( )` (resolved against caller-supplied §8.10 reference dimensions through the new `RefFrameState`), `allow_high_precision_mv`, and §6.2.7 `read_interpolation_filter( )` (`literal_to_type[ ]`). Pinned byte-exact against the `i-frame-then-p-frame-64x64` corpus P-frame (frame_size_with_refs → 64x64, ref_frame_idx=[0,1,2], header_size=3, uncompressed_header_size=10). `parse_uncompressed_header( )` stays the intra-only entry point (round 339)
- §8.10 reference frame-buffer state — `RefBuffers`, the `NUM_REF_FRAMES = 8` `FrameStore[ ]` slots with per-slot `RefFrameWidth` / `RefFrameHeight` / sub-sampling / bit-depth geometry; the §8.10 reference frame update process (`refresh_frame_flags`-masked per-slot copy with visible-extent crop); and the §8.5.2.3 / §8.5.2.4 `resolve_ref_idx` (`ref_frame_idx[ ref_frame[ refList ] - LAST_FRAME ]`). The persistent buffer half of the §8.5.2 inter-prediction wiring (round 339)
- §8.5.2 inter prediction driver — `predict_inter` chains the §8.5.2.1 `select_mv` / §8.5.2.2 `clamp_mv` / §8.5.2.3 `scale_mv` / §8.5.2.4 `block_inter_predict` steps into a region prediction written into `CurrFrame`, for both single reference (`preds[ 0 ]`) and compound (`Round2( preds[0] + preds[1], 1 )`). Derives the §8.5.2.4 `lastX` / `lastY` per-plane from the reference frame's luma dimensions; pure function of caller-supplied geometry + `RefPlanes` (round 339)
- §8.5.2.1 / §8.5.2.2 / §8.5.2.3 motion-vector selection, clamping, and scaling — the three steps that turn a decoded `BlockMvs[ ]` entry into the `startX` / `startY` / `stepX` / `stepY` sampling inputs the §8.5.2.4 leaf consumes. `select_mv` picks the per-`blockIdx` vector for luma / `>= BLOCK_8X8` blocks and rounds-averages the covered luma vectors for sub-sampled sub-8x8 chroma (`round_mv_comp_q2` / `round_mv_comp_q4`); `clamp_mv` converts to plane precision (`(2 * mv) >> s`) and clamps against the `INTERP_EXTEND` border; `scale_mv` folds in the reference-frame size ratio (`xScale` / `yScale`) to produce the 1/16 th-sample start position and per-sample step, deriving the fractional part from the luma block location even for chroma. Pure functions of caller-supplied geometry; the §8.5.2 driver that chains them into the §8.5.2.4 leaf and writes `CurrFrame` lands on top (round 336)
- §8.5.2.4 block inter prediction process — the two-pass 8-tap sub-pixel convolution leaf of the §8.5.2 inter prediction process. Transcribes the `subpel_filters[4][16][8]` kernels (EIGHTTAP / EIGHTTAP_SMOOTH / EIGHTTAP_SHARP / BILINEAR) and runs the horizontal-then-vertical interpolation over a caller-supplied edge-clamped reference plane, producing a `w × h` prediction block. Pure function of the §8.5.2.3 scaled sampling inputs (`startX` / `startY` / `xStep` / `yStep`); the §8.5.2.1-3 selection / clamping / scaling steps and the driver that writes into `CurrFrame` land on top (round 331)
- §6.4.11 inter_frame_mode_info( ) driver — ties the per-block inter-frame mode-info prelude into one pass: §6.4.12 `inter_segment_id( )`, §6.4.8 `read_skip( )`, §6.4.13 `read_is_inter( )`, and §6.4.10 `read_tx_size( !skip || !is_inter )`, resolving the §6.4.9 `seg_feature_active( )` predicates (`SEG_LVL_SKIP` / `SEG_LVL_REF_FRAME`) against the just-decoded `segment_id`, then dispatching the §6.4.5 arm to either §6.4.16 `inter_block_mode_info( )` or §6.4.15 `intra_block_mode_info( )` and surfacing `segment_id` / `skip` / `is_inter` / `tx_size` plus the arm-specific block products (round 325)
- §6.4.16 inter_block_mode_info( ) driver — ties §6.4.17 `read_ref_frames( )`, §6.5.1 `find_mv_refs( )` + §6.5.12 `find_best_ref_mvs( )`, the §9.3.1 `inter_mode_tree` / `interp_filter_tree` tree decodes (with §9.3.2 `ModeContext[ ref_frame[ 0 ] ]` / switchable interp-filter contexts), §6.5.14 `append_sub8x8_mvs( )` and §6.4.18 `assign_mv( )` into the full per-block inter mode-info decode, producing `ref_frame[ ]` / `y_mode` / `interp_filter` / `BlockMvs[ ][ ]` for both `MiSize >= BLOCK_8X8` and the sub-8x8 `(idy, idx)` walk (round 321)
- §6.4.17 read_ref_frames( ) driver — threads the §9.3.2 `comp_mode` / `comp_ref` / `single_ref_p1` / `single_ref_p2` context derivations against the per-block neighbour ref-frame pairs, resolving `ref_frame[ 0 ]` / `ref_frame[ 1 ]` for single + compound + segment-override paths (round 317)
- §9.3.2 comp_mode_context — `comp_mode` probability-context derivation (5-context, neighbour ref-frame `*Single` / `*Intra` / `CompFixedRef` predicates) toward the §6.4.17 read_ref_frames driver (round 314)

## [0.0.12](https://github.com/OxideAV/oxideav-vp9/compare/v0.0.11...v0.0.12) - 2026-06-15

### Other

- §6.4.18 assign_mv( ) per-reference-list motion-vector resolver
- §6.5.14 append_sub8x8_mvs — sub-8x8 inter motion-vector predictor builder (round 305)
- §6.5.1 find_mv_refs candidate scan + §6.5.6-6.5.11 helpers
- §6.5.2/§6.5.3/§6.5.4/§6.5.5/§6.5.12 motion-vector reference geometry — clamp + find_best_ref_mvs primitives (round 293)
- §6.4.19/§6.4.20 motion-vector residual syntax — read_mv + read_mv_component leaf primitives (round 288)
- §6.4/§6.4.4/§8.8 top-level intra decode wiring — decode_vp9 decodes keyframes end-to-end
- stand cargo-fuzz back up — frame_header + compressed_header panic-surface targets + tracked seed corpus (round 282)
- §8.8 loop filter process — frame-level raster driver + 3-plane CurrFrame (round 281)
- §8.8.2 superblock loop filter — full per-plane per-pass driver (steps 1-17)
- §8.8.2 superblock loop filter per-edge predicate derivation (steps 1-14)
- §8.8.5 sample_filtering outer driver (round 267)
- Round 259: §8.8.5.3 wide_filter( ) — per-edge low-pass leaf primitive
- Round 255: §8.8.5.2 narrow_filter( ) — per-edge sample-mutation leaf primitive
- Round 253: §8.8.5.1 filter_mask( ) — per-edge filter-mask leaf primitive
- Round 250: §8.8.4 adaptive_filter_strength( ) — per-edge filter-strength leaf primitive
- drop release-plz.toml — use release-plz defaults across the workspace
- Round 244: §8.8.3 filter_size( ) — per-edge filter-size leaf primitive
- Round 37: §8.8.1 loop_filter_frame_init( ) — public LvlLookup builder
- Round 36: §6.4 lines 2306-2311 byte-walk lifted to a public primitive — tile_payload_sizes
- Round 35: §6.3 parse_compressed_header_inter integration-test coverage
- Round 34: §6.3 if (FrameIsIntra == 0) outer dispatch — parse_compressed_header_inter entry point
- Round 33: §6.4 decode_tiles( ) outer driver — frame-level tile walk
- Round 32: §6.4.1 get_tile_offset + §6.4.2 decode_tile — tile-driver primitive layer
- Round 31: §6.4.4 decode_block( ) driver — pure-state fan-out primitive
- Round 30: §6.3.16 mv_probs() compressed-header outer sweep
- Round 29: §6.3.12 frame_reference_mode() compressed-header outer driver
- Round 28: §6.3.18 setup_compound_reference_mode() pure-compute leaf

### Added

* **Round 309: §6.4.18 `assign_mv( isCompound )` — the per-reference-list
  motion-vector resolver, added to the `mv` module as `assign_mv` plus
  the `MvPredictors` predictor bundle.** Threads the round-288
  `read_mv( )` leaf together with the round-293 / round-305 predictors
  to produce the final `Mv[ 0 ]` / `Mv[ 1 ]` an inter block uses.
  * §6.4.18 listing implemented verbatim: `Mv[ 1 ]` is pre-set to the
    §3 `ZeroMv` before the loop; for each active reference list `i` in
    `0..1 + isCompound`, `y_mode == NEWMV` reads one §6.4.19
    `read_mv( i )` difference onto `BestMv[ i ]`, `NEARESTMV` /
    `NEARMV` copy `NearestMv[ i ]` / `NearMv[ i ]`, and the remaining
    arm (`ZEROMV`, including the §6.4.16 `SEG_LVL_SKIP` forced case)
    yields `ZeroMv`.
  * `MvPredictors { nearest, near, best }` carries the §6.5.12
    `find_best_ref_mvs( )` (or §6.5.14 `append_sub8x8_mvs( )`) outputs
    for one list slot, so `assign_mv( )` stays a pure function of the
    bool coder, the §6.3.16 `MvProbs` bundle, the predictor pair, and
    `allow_high_precision_mv` — no frame-wide state to thread until the
    §6.4.16 `inter_block_mode_info( )` driver lands.
  * 7 new unit tests (lib 641 -> 648); intra decode unchanged (still
    byte-exact on the 13-fixture corpus).

* **Round 305: §6.5.14 `append_sub8x8_mvs( )` — the sub-8x8 inter
  motion-vector predictor builder, added to the `mv_ref` module as
  `MvRefGeometry::append_sub8x8_mvs`.** Composes round 299's §6.5.1
  `find_mv_refs( )`: for one sub-block of a sub-8x8 inter block it
  derives the `[NearestMv, NearMv]` pair the §6.4.18 `assign_mv( )`
  driver consumes when `y_mode` is `NEARESTMV` / `NEARMV`.
  * §6.5.14 listing implemented verbatim: `block == 0` takes both
    `RefListMv[ ]` candidates; `block <= 2` seeds slot 0 from
    `BlockMvs[ refList ][ 0 ]`; the `block == 3` arm seeds from
    `BlockMvs[ refList ][ 2 ]` then walks sub-blocks 1 then 0 adding
    any that differ from slot 0; remaining slots fill from the
    `RefListMv[ ]` candidates (skipping a slot-0 duplicate), then a §3
    `ZeroMv` backfills.
  * The §6.5.14 `refList` index is resolved by the caller — `block_mvs`
    is the `BlockMvs[ refList ]` row and the return value is the
    per-`refList` `[NearestMv, NearMv]` pair, so the predictor stays a
    pure function of geometry plus candidate data.
  * +10 unit tests (lib 631 -> 641): the three `block` arms, the
    `RefListMv` / `BlockMvs` dedup against slot 0, the `ZeroMv`
    backfill, and that the `block` index threads through to the
    `find_mv_refs( )` sub-block selection. Intra decode unchanged
    (still byte-exact on the 13-fixture corpus).
* **Round 299: §6.5.1 `find_mv_refs( )` candidate scan plus the
  §6.5.6-6.5.11 helpers (`add_mv_ref_list( )` /
  `if_same_ref_frame_add_mv( )` / `if_diff_ref_frame_add_mv( )` /
  `scale_mv( )` / `get_block_mv( )` / `get_sub_block_mv( )`), added to
  the `mv_ref` module.** The layer above round 293's clamps:
  `find_mv_refs( refFrame, block )` walks the
  `mv_ref_blocks[ MiSize ]` neighbour positions to build the two
  `RefListMv[ ]` predictors and the `ModeContext[ refFrame ]` value
  that `find_best_ref_mvs( )` and the §6.4.16 inter driver consume.
  * §6.5.1 three-pass scan: positions 0..2 read the candidate's
    sub-block vector (§6.5.11 `get_sub_block_mv( )` via the
    `idx_n_column_to_subblock[ ]` table) on a same-reference match;
    positions 2..`MVREF_NEIGHBOURS` use the §6.5.7
    `if_same_ref_frame_add_mv( )` whole-block helper; then, when any
    in-frame neighbour was seen, a §6.5.8 `if_diff_ref_frame_add_mv( )`
    pass fills any remaining slot with a sign-scaled
    different-reference vector (§6.5.9 `scale_mv( )`).
  * `UsePrevFrameMvs` adds the previous frame's vector at
    `(MiRow, MiCol)` through the `usePrev` arm of §6.5.10
    `get_block_mv( )`.
  * `contextCounter` accumulates `mode_2_counter[ YModes[ ][ ] ]` over
    the pass-1 neighbours; `counter_to_context[ contextCounter ]`
    yields `ModeContext[ refFrame ]`. The `mv_ref_blocks` /
    `mode_2_counter` / `counter_to_context` / `idx_n_column_to_subblock`
    tables are transcribed verbatim from the §6.5.1 listing.
  * §6.5.6 `add_mv_ref_list( )` dedup (cap 2, drop a duplicate of
    `RefListMv[ 0 ]`) and the §6.5.3 `clamp_mv_ref( )` of both final
    candidates.
  * Neighbour mode-info is read through a new `MvCandidateSource` trait
    so the scan stays a pure function of geometry plus candidate data,
    directly testable against a synthetic frame; the §6.4.4
    `decode_block` fan-out's per-MI arrays will back it once the inter
    decode driver threads them.
  * +11 unit tests (lib 620 -> 631): empty neighbourhood, pass-1
    same-reference match, `add_mv_ref_list` dedup, two distinct
    candidates, the different-reference fill pass, `scale_mv`
    sign-bias negation, `UsePrevFrameMvs`, the `contextCounter`
    classification, `is_inside` out-of-frame exclusion, the final
    clamp, and the §6.5.11 sub-block index selection. Intra decode
    unchanged (still byte-exact on the 13-fixture corpus).

* **Round 293: §6.5.2 / §6.5.3 / §6.5.4 / §6.5.5 / §6.5.12
  motion-vector reference geometry — `is_inside( )` /
  `clamp_mv_ref( )` / `clamp_mv_row( )` / `clamp_mv_col( )` /
  `find_best_ref_mvs( )` primitives (new module `mv_ref`).** The layer
  above round 288's `read_mv( )` — `find_best_ref_mvs( )` produces the
  `BestMv[ ref ]` predictor `read_mv( )` consumes:
  * §6.5.4 `clamp_mv_row( mvec, border )` / §6.5.5 `clamp_mv_col(
    mvec, border )` — `Clip3` of a component into
    `[mbToTopEdge - border, mbToBottomEdge + border]` (and the
    left/right analogue), with the edges derived from `MiRow` /
    `MiCol` / `MiRows` / `MiCols` / `MiSize` in eighth-pel units.
  * §6.5.3 `clamp_mv_ref( i )` — both component clamps with the §3
    `MV_BORDER = 128` border.
  * §6.5.2 `is_inside( candidateR, candidateC )` — candidate-position
    gate: whole-frame rows (`0 .. MiRows`) but per-*tile* columns
    (`MiColStart .. MiColEnd`), since tile column edges are not
    crossable.
  * §6.5.12 `find_best_ref_mvs( refList )` — odd eighth-pel components
    rounded toward zero when `!allow_high_precision_mv ||
    !use_mv_hp( )`, then the wide
    `(BORDERINPIXELS - INTERP_EXTEND) << 3 == 1248` clamp, yielding
    `NearestMv` / `NearMv` / `BestMv`.
  All clamps are pure functions of a per-block `MvRefGeometry`. 10 new
  unit tests (lib 610 -> 620). Still deferred for end-to-end inter:
  §6.5.1 `find_mv_refs( )` candidate scan, §6.4.16-18
  `inter_block_mode_info( )` / `read_ref_frames( )` / `assign_mv( )`,
  and §8.5.2 inter prediction.

* **Round 288: §6.4.19 / §6.4.20 motion-vector residual syntax —
  `read_mv( )` / `read_mv_component( )` leaf primitives.** The first
  step of the inter-block motion-vector decode (new module `mv`):
  * §6.5.13 `use_mv_hp( deltaMv )` — the high-precision predicate
    `(Abs(deltaMv[0]) >> 3) < COMPANDED_MVREF_THRESH && (Abs(deltaMv[1])
    >> 3) < COMPANDED_MVREF_THRESH` that, combined with the frame-level
    `allow_high_precision_mv`, derives the local `UseHp`.
  * §6.4.20 `read_mv_component( comp )` — `mv_sign` + `mv_class`
    (`mv_class_tree`) and the two magnitude arms: the class-0 path
    (`mv_class0_bit` / `mv_class0_fr` over `mv_fr_tree` /
    `mv_class0_hp`) with `mag = ((bit<<3)|(fr<<1)|hp)+1`, and the
    class-`n` path (`mv_bit` offset loop, `mag = CLASS0_SIZE <<
    (mv_class+2)`, then `mv_fr` / `mv_hp`). The §9.3.3 rule that
    `mv_class0_hp` / `mv_hp` are absent and read as `1` when `UseHp ==
    0` is honoured.
  * §6.4.19 `read_mv( ref )` — `mv_joint` (`mv_joint_tree`) selects
    which of the (row, col) components are read, then the decoded
    difference is added to the §6.5.3 `BestMv[ ref ]` predictor to
    yield the final `Mv[ ref ]`. The §9.3.2 per-component probability
    selection drives every read against the `MvProbs` bundle the
    §6.3.16 compressed header parsed.
  * 9 unit tests over constructed bool-coder buffers cover the
    `use_mv_hp` threshold boundary, the joint-zero fast path, the
    class-0 minimum-magnitude formula, the no-hp fixed bit, and the
    `mv_class_tree` / `mv_joint_tree` leaf coverage.

* **Round 284: top-level intra decode wiring — `decode_vp9` /
  `decode_intra_frame` decode whole keyframes end-to-end.** The
  §6.4 / §6.4.4 / §8.8 composition round: every previously-landed
  primitive is now driven by a real frame walk (new module
  `decode_frame`):
  * §6.2 + §6.3 headers → §6.4 `decode_tiles( )` (per-tile
    `init_bool` / `exit_bool`, §7.4.1 `clear_above_context` once per
    frame, §7.4.2 `clear_left_context` per superblock row) → §6.4.2
    superblock raster → §6.4.3 `decode_partition( )` → §6.4.4
    `decode_block( )` at every leaf, with the block syntax decoded
    inline from the same §9.2 coder as the partition syntax.
  * §6.4.4 block composition: §6.4.6 `intra_frame_mode_info( )`
    (with the §6.4.7 segment-id decode hoisted so the §6.4.9
    `SEG_LVL_SKIP` gate sees the decoded segment), the §6.4.21
    `residual( )` walk decoding §6.4.24 `tokens( )` inline and
    reconstructing each transform block (§8.5.1 `predict_intra` with
    per-`blockIdx` sub-8x8 modes → §8.6 dequant → §8.7 inverse
    transform → `Clip1` add), then the §6.4.4 fan-out via
    `decode_block_apply`.
  * §8.8 loop filter wired over the reconstructed frame:
    `loop_filter_frame_init` from the §6.2.8 deltas resolved against
    the §7.2 `setup_past_independence` defaults, then
    `frame_loop_filter` with the §6.4.4 frame-wide arrays.
  * §8.10 output crop: `Vp9DecodedFrame` carries the planar `u16`
    samples plus geometry; `to_planar_bytes( )` packs 8-bit content
    one byte per sample and 10/12-bit as little-endian pairs.
  * `decode_partition` / `decode_tile` now invoke a `LeafSink` at
    every §6.4.4 call site (a `Vec<LeafBlock>` sink preserves the
    old leaf-log behaviour for partition-only streams).
  * Validation (+7 integration tests in `tests/decode_vp9.rs`):
    byte-exact decodes of the staged corpus fixtures
    `tiny-i-only-16x16`, `lossless-i-only` (§8.7.1.10 WHT path) and
    `q-low` (embedded verbatim so standalone CI runs them), plus a
    workspace-checkout sweep decoding the leading keyframe of all 13
    intra-leading `docs/video/vp9/fixtures/` entries byte-exactly
    against `expected.yuv` (4:2:0 + 4:4:4, 8/10/12-bit, RGB,
    2-tile-column, segmentation-AQ, lossless, q extremes), and
    truncation / `show_existing_frame` error-path checks.

* **Round 282: cargo-fuzz scaffold — `fuzz/` stood back up so the
  scheduled Fuzz workflow runs again.** Two panic-surface targets,
  auto-discovered by the org reusable fuzz workflow:
  * `frame_header` — `parse_uncompressed_header` plus the frame
    walk that hangs off it (§6.3 compressed-header slice via
    `header_size_in_bytes` + the header-derived `Lossless` flag,
    then the §6.4 tile-size prefix chain via `tile_payload_sizes`).
  * `compressed_header` — the §9.2 Boolean-decoder walkers
    (`parse_compressed_header` / `parse_compressed_header_inter`)
    with a leading control byte steering the caller-supplied flags
    (`Lossless`, intra vs. inter, `SWITCHABLE` interpolation
    filter, `allow_high_precision_mv`, §6.2.5
    `ref_frame_sign_bias`).
  * 9-entry tracked seed corpus (`fuzz/corpus/*/seed-*`) derived
    from the crate's synthetic test vectors.
  * Local soak: 320 s per target under AddressSanitizer — 96.4 M +
    43.6 M execs, zero findings.
  * `fuzz.yml` preamble rewritten to describe the new target set
    (the old preamble described pre-rebuild harnesses).

* **Round 281: §8.8 `loop filter process` — the frame-level driver —
  [`frame_loop_filter`] + the 3-plane [`CurrFrame`] container.**
  Lands the outermost layer of the §8.8 loop-filter arc per
  `vp9-spec.txt` lines 5436-5455:
  * `frame_loop_filter(curr: &mut CurrFrame, frame:
    &SuperblockFilterFrame)` walks the §8.8 four-deep raster — `row`
    over `0, 8, .. < MiRows`, `col` over `0, 8, .. < MiCols`,
    `plane` over `0..2`, `pass` over `0..1` — invoking the
    round-278 §8.8.2 [`superblock_loop_filter`] driver at each step
    (lines 5451-5455), in exactly the listing's nesting order per
    the §8.8 ordering NOTE (lines 5458-5460: many samples filter
    more than once, so each call mutates the plane in place before
    the next call reads it).
  * The §8.8 first step — the §8.8.1 frame init (line 5441) — is
    the caller's [`loop_filter_frame_init`] invocation; its
    `LvlLookup` output arrives via
    `SuperblockFilterFrame::lvl_lookup`.
  * New public type `CurrFrame` — the §8.8 input/output (lines
    5437-5438): three [`SuperblockFilterPlane`] views, Y at
    `FrameWidth x FrameHeight` and U / V at the §8.10 subsampled
    extents (lines 5944-5948).
  * Up-front consistency panics: luma extent vs. the §7.2.6
    `MiCols = (FrameWidth + 7) >> 3` / `MiRows = (FrameHeight + 7)
    >> 3` grid (lines 1760-1761); chroma extents vs. the §8.10
    subsampled extents.
  * Validation (+10 lib tests, 591 -> 601; +6 integration tests in
    `tests/frame_loop_filter.rs`): flat-frame identity on all three
    planes; the step-17 `lvl > 0` gate at frame level; sample-exact
    equivalence against the §8.8 raster transcribed directly from
    the listing over individual §8.8.2 calls on order-sensitive
    noise frames (full 2x2-superblock 4:2:0, partial-superblock
    `MiCols = MiRows = 12`, non-MI-aligned 52x36 / 26x18 extents,
    and 10-bit); a cross-superblock vertical boundary at `x = 64`
    filtering via the `col = 8` call's edge 0; chroma routing
    through the subsampled raster; extent-mismatch panics.

* **Round 278: §8.8.2 `superblock loop filter process` — the full
  per-plane, per-pass driver landed as a public entry point —
  [`superblock_loop_filter`].** Composes every previously-landed
  loop-filter primitive into the complete §8.8.2 process per
  `vp9-spec.txt` lines 5491-5586, modifying a `CurrFrame[ plane ]`
  sample plane in place:
  * `superblock_loop_filter(plane_buf: &mut SuperblockFilterPlane,
    frame: &SuperblockFilterFrame, plane: u8, pass: u8, row: u32,
    col: u32)` walks the §8.8.2 `edge ∈ 0..(16 >> sub) - 1` /
    `i ∈ 0..edgeLen - 1` raster (lines 5524-5525) using the
    round-274 [`superblock_filter_geometry`] header, runs the
    round-274 steps 1-14 predicate bundle
    ([`superblock_filter_edge`]), then threads steps 15-17: §8.8.3
    [`filter_size`] (lines 5579-5581), §8.8.4
    [`adaptive_filter_strength`] at `(loopRow, loopCol)` (lines
    5582-5583), and — when `applyFilter == 1 && lvl > 0` — §8.8.5
    [`sample_filtering`] at `(x >> subX, y >> subY)` along
    `(dx, dy)` (lines 5584-5586), including the §8.8.5.1 16-sample
    stencil gather / write-back (lines 5703-5727).
  * Step 6's chroma `txSz` is resolved through the §6.4.22
    `get_uv_tx_size( )` helper (lines 2871-2876) from the `MiSize` /
    `tx_size` read at `(loopRow, loopCol)` — the first caller to
    thread it outside the §6.4 residual path.
  * New public type `SuperblockFilterPlane` — a mutable
    `data / stride / width / height` view of one `CurrFrame[ plane ]`
    sample plane (`i32` samples, matching the §8.8.5 working type).
  * New public type `SuperblockFilterFrame` — the per-frame decode
    state: the six row-major `MiRows x MiCols` per-MI arrays
    (`MiSizes` / `TxSizes` / `Skips` / `RefFrames[..][..][0]` /
    `YModes` / `SegmentIds`), the frame scalars (`mi_cols` /
    `mi_rows` / `subsampling_x` / `subsampling_y` /
    `loop_filter_sharpness` / `bit_depth`), and the §8.8.1
    [`LvlLookup`].
  * Right / bottom off-screen raster positions are short-circuited
    *before* the steps 4-9 per-MI reads: step 13 forces
    `applyFilter = 0` there, so the reads are dead, and the
    short-circuit keeps every array access inside `MiRows x MiCols`.
    Out-of-plane stencil reads (possible only for the unused outer
    ring per the §8.8.5.1 NOTE) are edge-clamped; write-back drops
    positions whose true coordinate is outside the plane.
  * `pub use superblock_loop_filter::{superblock_loop_filter,
    SuperblockFilterFrame, SuperblockFilterPlane};` on the crate
    root.
  * Validation: +13 lib tests (lib total 578 → 591) + 8 integration
    tests in `tests/superblock_loop_filter.rs`. Covers flat-plane
    identity on both passes, the step-17 `lvl > 0` gate
    (`loop_filter_level == 0` no-op), vertical / horizontal /
    4:2:0-chroma step responses cross-checked against the §8.8.4 +
    §8.8.5 primitives invoked directly on the same stencil, the
    step-14 skip / block-edge / tx-edge gating threaded end-to-end,
    the step-13 left / top frame-edge exclusions, the per-segment
    `SEG_LVL_ALT_L` lvl partition (step-16 indexing at
    `(loopRow, loopCol)`), a mid-superblock frame end (MiCols =
    MiRows = 6) with no out-of-bounds access, the 10-bit path, and
    the up-front plane-view / per-MI-array consistency panics.

* **Round 274: §8.8.2 `superblock loop filter process` per-edge
  predicate derivation (steps 1-14) lifted to a public leaf primitive
  — [`superblock_filter_edge`] + [`superblock_filter_geometry`].** The
  §8.8.2 driver's per-edge book-keeping that turns the raster position
  `(pass, row, col, edge, i)` plus the per-MI decode state at the
  resolved `(loopRow, loopCol)` into the
  `(x, y, loopRow, loopCol, isBlockEdge, isTxEdge, is32Edge, onScreen,
  applyFilter)` bundle the §8.8.2 steps 15-17 hand-off consumes, per
  `vp9-spec.txt` §8.8.2 lines 5491-5586.
  * `superblock_filter_geometry(pass, sub_x, sub_y) ->
    SuperblockFilterGeometry` lifts the §8.8.2 `dx` / `dy` / `sub` /
    `edgeLen` header (lines 5510-5519): vertical edges (`pass == 0`)
    give `dx=1, dy=0, sub=subX, edgeLen=64>>subY`; horizontal edges
    give `dx=0, dy=1, sub=subY, edgeLen=64>>subX`. The driver iterates
    `edge ∈ 0..(16>>sub)-1` and `i ∈ 0..edgeLen-1`.
  * `superblock_filter_edge(pass, row, col, edge, i, sub_x, sub_y,
    mi_cols, mi_rows, &SuperblockFilterMi) -> SuperblockFilterEdge`
    runs steps 1-14: the §8.8.2 step-1 `x`/`y` luma coordinates, the
    step-2/3 `loopCol`/`loopRow` sub-sampling align-down, the step-7
    `sbSize = sub==0 ? MiSize : Max(BLOCK_16X16, MiSize)`, step-10
    `isBlockEdge` against `8*num_8x8_blocks_{wide,high}_lookup[sbSize]`,
    step-11 `isTxEdge` (including the chroma horizontal-boundary
    right-image-edge suppression), step-12 `is32Edge` (the §8.8.3
    filter-size input), step-13 `onScreen` (right/bottom + implicit
    left/top frame-edge exclusion), and the step-14 `applyFilter`
    gate `onScreen && (isBlockEdge || (isTxEdge && (isIntra ||
    !skip)))`.
  * `txSz` is supplied already-resolved by the caller (step 6's
    `(plane>0) ? get_uv_tx_size( ) : tx_size`), exactly as the §8.8.3
    [`filter_size`] caller does; `SuperblockFilterMi` carries
    `(mi_size, tx_sz, skip, ref_frame_0)` read at `(loopRow,
    loopCol)`.
  * `pub use superblock_filter::{superblock_filter_edge,
    superblock_filter_geometry, SuperblockFilterEdge,
    SuperblockFilterGeometry, SuperblockFilterMi};` on the crate root.
  * Validation: +30 lib tests (lib total 548 → 578) + 8 integration
    tests in `tests/superblock_filter.rs`. Covers the luma / 4:2:0
    chroma geometry, the step-1 `x`/`y` and step-2/3 align-down, the
    `isBlockEdge` block-size scaling (incl. the chroma `sbSize`
    promotion to `BLOCK_16X16`), the `isTxEdge` tx-size multiples plus
    the chroma right-edge suppression, the `is32Edge` multiple-of-8
    rule, the `onScreen` frame-edge exclusions, and every `applyFilter`
    arm (block edge / intra-tx-edge-even-when-skip / inter-tx-edge skip
    vs non-skip).

* **Round 267: §8.8.5 `sample filtering process` outer driver lifted
  to a public leaf primitive — [`sample_filtering`].** New per-edge
  dispatcher composing the three §8.8.5 sub-processes (§8.8.5.1
  [`filter_mask`], §8.8.5.2 [`narrow_filter`], §8.8.5.3
  [`wide_filter`]). Signature: `sample_filtering(samples:
  &SampleFilterSamples, limit: u8, blimit: u8, thresh: u8,
  filter_size: u8, bit_depth: u8) -> SampleFilterOutput` per
  `vp9-spec.txt` §8.8.5 lines 5662-5684.
  * Runs §8.8.5.1 `filter_mask` on the 16-sample stencil first
    (lines 5672-5674), then dispatches per the §8.8.5 table (lines
    5678-5684): `filterMask == 0` → no filter; `filterSize ==
    TX_4X4 || flatMask == 0` → §8.8.5.2 narrow (fed `hevMask`);
    `filterSize == TX_8X8 || flatMask2 == 0` → §8.8.5.3 wide
    `log2Size = 3`; otherwise → wide `log2Size = 4`.
  * The `flatMask` / `flatMask2` reads sit behind `filterSize`
    short-circuits, so the §8.8.5.1 `None` returns are never
    dereferenced.
  * `SampleFilterSamples` carries the `p7..p0` / `q0..q7` stencil;
    `SampleFilterOutput` carries the full 16-sample post-filter
    stencil with positions outside the mutation window echoed
    through, so the caller writes the whole stencil back to
    `CurrFrame` unconditionally.
  * `pub use sample_filtering::{sample_filtering,
    SampleFilterOutput, SampleFilterSamples};` on the crate root.
  * Validation: +8 lib tests (lib total 546 → 554) + 8 integration
    tests in `tests/sample_filtering.rs`. Covers the flat-region
    identity at BitDepth 8/10/12, the `filterMask == 0` no-op, and
    all four dispatch arms (narrow / wide-log2-3 / wide-log2-4 /
    `flatMask2 == 0` drop-back) cross-checked against the three
    sub-process primitives run directly.

* **Round 259: §8.8.5.3 `wide filter process` lifted to a public
  leaf primitive — [`wide_filter`].** New per-edge low-pass
  the §8.8.5 outer driver will call after the round-253 §8.8.5.1
  [`filter_mask`] step picks the wide branch via the §8.8.5
  dispatch table at `vp9-spec.txt` lines 5681-5684. Signature:
  `wide_filter(samples: &WideFilterSamples, log2_size: u32,
  bit_depth: u8) -> WideFilterOutput` per
  `vp9-spec.txt` §8.8.5.3 lines 5855-5888.
  * `log2_size == 3` (8-tap kernel, `n == 3`) per lines
    5681-5682: loop walks `i ∈ [-3, 2]` producing six mutated
    outputs at positions `p2..p0`, `q0..q2`. The remaining
    `op6..op3`, `oq3..oq6` fields echo the corresponding input
    through unchanged so the caller can write all 14 fields
    without branching.
  * `log2_size == 4` (16-tap kernel, `n == 7`) per lines
    5683-5684: loop walks `i ∈ [-7, 6]` producing fourteen
    mutated outputs at positions `p6..p0`, `q0..q6`.
  * Kernel verbatim from lines 5868-5885:
    `F[ i ] = Round2( CurrFrame[i] + sum_{j=-n..n}
    CurrFrame[Clip3(-(n+1), n, i+j)], log2Size )` with
    `n = (1 << (log2Size - 1)) - 1` (lines 5864-5865) and
    `Round2( t, k ) = (t + (1 << (k - 1))) >> k` (§3). Total
    samples summed per output: `2n + 2`.
  * `Clip3( -(n+1), n, i+j )` edge-replication (line 5879) pulls
    in duplicates of the outermost in-range sample when the
    index walk overshoots either side of the stencil window.
  * Unlike §8.8.5.2 the wide filter operates directly on
    unsigned pixel values — no `0x80 << (BitDepth - 8)` working-
    range offset, no `filter4_clamp` BitDepth scaling. The
    `bit_depth` parameter is carried for API symmetry with
    [`narrow_filter`] only.
  * `WideFilterSamples` carries the 16-sample stencil
    (`p7..p0`, `q0..q7`) the §8.8.5 outer driver assembles from
    `CurrFrame[ plane ][ y +/- dy*k ][ x +/- dx*k ]` per
    §8.8.5.1 lines 5703-5727.
  * `WideFilterOutput` carries up to 14 mutated samples
    (`op6..op0`, `oq0..oq6`) the caller writes back to
    `CurrFrame` at the matching `(y + i*dy, x + i*dx)` for
    `i ∈ [-n, n-1]` per lines 5884-5885.
  * `log2_size ∉ {3, 4}` panics with `§8.8.5.3: log2_size must
    be 3 or 4 per §8.8.5 dispatch table` — the §8.8.5 outer
    driver's two-arm dispatch (lines 5682 / 5684) is the only
    producer of `log2_size` values.
  * Public surface: `wide_filter` + `WideFilterSamples` +
    `WideFilterOutput` exposed at the crate root.
  * +12 lib tests and +9 integration tests in
    `tests/wide_filter.rs`: unity-gain on flat stencils at
    `(3, 8)`, `(4, 8)`, `(3, 10)`, `(4, 12)`; outer-field echo
    on `log2_size == 3`; hand-traced step response producing
    exact `(op2, op1, op0, oq0, oq1, oq2) = (13, 25, 38, 63,
    75, 88)` for a 0→100 step on the 8-tap kernel; `Clip3` edge-
    replication isolating `p3 = 80` driving `op2 = 30`; 16-tap
    boundary `op0 = 56` for the (0 → 128) step; `Round2` half-up
    rounding verified at the boundary; `log2_size` precondition
    panics on `2`, `5`, `7`.

* **Round 255: §8.8.5.2 `narrow filter process` lifted to a public
  leaf primitive — [`narrow_filter`].** New per-edge sample-mutation
  the §8.8.5 outer driver will call after the round-253 §8.8.5.1
  [`filter_mask`] step picks the narrow branch per spec
  `vp9-spec.txt` §8.8.5.2 lines 5795-5853. Signature:
  `narrow_filter(samples: &NarrowFilterSamples, hev_mask: bool,
  bit_depth: u8) -> NarrowFilterOutput`.
  * `hev_mask == 1` (high edge variance) per lines 5809-5811:
    modifies only `op0` and `oq0`, leaves `op1` / `oq1` equal to
    the input. The filter is derived from all four input samples
    via `filter = filter4_clamp(ps1 - qs1)` (line 5838) → `filter
    = filter4_clamp(filter + 3 * (qs0 - ps0))` (line 5839).
  * `hev_mask == 0` (smooth / low variance) per lines 5806-5808
    and 5846-5852: modifies all four samples. The filter's
    `ps1 - qs1` term drops out so `filter` starts at 0, and a
    half-strength pass via `Round2(filter1, 1)` is added to
    `op1` / `oq1`.
  * `filter4_clamp` (lines 5824-5826) clips into the signed range
    `[-(1 << (BitDepth - 1)), (1 << (BitDepth - 1)) - 1]` per
    `Clip3` (§3); the `0x80 << (BitDepth - 8)` offset (lines
    5834-5837) is applied and undone verbatim per
    `BitDepth ∈ {8, 10, 12}` (§6.2.2).
  * `filter1 = filter4_clamp(filter + 4) >> 3` and
    `filter2 = filter4_clamp(filter + 3) >> 3` (lines 5840-5841)
    bias the rounding for `oq0` vs `op0` asymmetrically.
  * `NarrowFilterSamples` carries the 4-sample stencil
    (`p1`, `p0`, `q0`, `q1`) the §8.8.5 outer driver assembles
    from `CurrFrame[ plane ][ y +/- dy*k ][ x +/- dx*k ]` per
    lines 5830-5833.
  * `NarrowFilterOutput` carries the four mutated samples
    (`op1`, `op0`, `oq0`, `oq1`) the caller writes back to
    `CurrFrame` at the matching `(y +/- dy*k, x +/- dx*k)` per
    lines 5844-5851.
  * Public surface: `narrow_filter` + `NarrowFilterSamples` +
    `NarrowFilterOutput` exposed at the crate root.
  * +12 lib tests and +9 integration tests in
    `tests/narrow_filter.rs`: baseline flat-stencil no-op at
    8/10/12-bit on both hev / smooth branches, hev branch
    outer-pair preservation, smooth branch outer-pair mutation
    via `Round2(filter1, 1)`, `filter4_clamp` saturation at the
    8-bit and 10-bit signed-range edges, `filter1` / `filter2`
    asymmetric rounding for `qs0` vs `ps0` outputs, matched
    outer samples collapsing the hev `ps1 - qs1` term, and a
    `op0 + oq0` symmetry property over a 5×5 stencil grid.

* **Round 253: §8.8.5.1 `filter mask process` lifted to a public
  leaf primitive — [`filter_mask`].** New per-edge mask derivation
  the §8.8.5 outer driver will call before dispatching to the
  §8.8.5.2 narrow filter or the §8.8.5.3 wide filter per spec
  `vp9-spec.txt` §8.8.5.1 lines 5685-5792. Signature:
  `filter_mask(samples: &FilterMaskSamples, limit: u8, blimit: u8,
  thresh: u8, filter_size: u8, bit_depth: u8) -> FilterMask`.
  * `hevMask` per lines 5730-5734: `(Abs(p1 - p0) > threshBd) ||
    (Abs(q1 - q0) > threshBd)` with `threshBd = thresh <<
    (BitDepth - 8)`.
  * `filterMask` per lines 5737-5750: seven inner abs-diff pair
    tests against `limitBd = limit << (BitDepth - 8)` plus the
    boundary term `Abs(p0 - q0) * 2 + Abs(p1 - q1) / 2 >
    blimitBd`. `filterMask = (mask == 0)`.
  * `flatMask` per lines 5753-5774: six abs-diff tests over the
    inner four samples on each side relative to `p0` / `q0`,
    gated by `filterSize >= TX_8X8` (returned as `None`
    otherwise per line 5697).
  * `flatMask2` per lines 5777-5792: eight abs-diff tests over
    the outer four samples on each side relative to `p0` / `q0`,
    gated by `filterSize >= TX_16X16` (returned as `None`
    otherwise per line 5698).
  * `FilterMaskSamples` carries the 16-sample stencil `p7..p0` /
    `q0..q7` the §8.8.5 outer driver assembles from `CurrFrame[
    plane ][ y +/- dy*k ][ x +/- dx*k ]` per lines 5703-5727.
  * Public surface: `filter_mask` + `FilterMask` +
    `FilterMaskSamples` exposed at the crate root.
  * +15 lib tests and +9 integration tests in
    `tests/filter_mask.rs`: baseline, lead-paragraph gating
    (`TX_4X4` / `TX_8X8` / `TX_16X16`), per-side `hevMask`
    triggers, equality-vs-strict-`>` cutoff, every `filterMask`
    reset path including the integer-division floor on `/ 2`,
    inner-region `flatMask` reset, outer-ring `flatMask2`
    reset, rising-slope mixed stencil, and BitDepth scaling at
    10-bit and 12-bit.

* **Round 250: §8.8.4 `adaptive_filter_strength( )` lifted to a
  public leaf primitive — [`adaptive_filter_strength`].** New per-
  `(loopRow, loopCol)` filter-strength derivation built from the
  `(lvl_lookup, segment_id, ref_frame, y_mode, loop_filter_sharpness)`
  inputs the §8.8.2 superblock raster walk will supply per spec
  `vp9-spec.txt` §8.8.4 lines 5626-5661. Signature:
  `adaptive_filter_strength(lvl_lookup: &LvlLookup, segment_id:
  usize, ref_frame: i32, y_mode: u8, loop_filter_sharpness: u8) ->
  Option<FilterStrength>`. Returns `None` for an out-of-range axis
  (`segment_id >= MAX_SEGMENTS` or `ref_frame` outside `0..=3`).
  * Step 1 `lvl` derivation per lines 5632-5639: reads the §8.8.1
    [`LvlLookup`] at `(segment_id, ref_frame, modeType)`, where
    `modeType = 1` for `NEARESTMV` / `NEARMV` / `NEWMV` and
    `modeType = 0` for intra modes (0..=9) or `ZEROMV` per lines
    5637-5638.
  * Step 2 `shift` derivation per lines 5642-5645: `shift = 2` when
    `loop_filter_sharpness > 4`, `shift = 1` when
    `loop_filter_sharpness > 0`, and `shift = 0` otherwise.
  * Step 3 `limit` derivation per lines 5648-5651: sharpness > 0
    → `limit = Clip3( 1, 9 - loop_filter_sharpness, lvl >> shift
    )`; sharpness = 0 → `limit = Max( 1, lvl >> shift )`. Both
    branches enforce `limit >= 1`.
  * Step 4 `blimit` per line 5660: `blimit = 2 * (lvl + 2) +
    limit`. The §8.8.1 `Clip3( 0, MAX_LOOP_FILTER, … )` ceiling
    bounds `blimit <= 2 * 65 + 63 = 193 < u8::MAX`.
  * Step 5 `thresh` per line 5661: `thresh = lvl >> 4`, the §8.8.5.1
    high-edge-variance threshold.
  * New public constants verbatim from §7.4.11 (`vp9-spec.txt`
    lines 3957-3961): `NEARESTMV: u8 = 10`, `NEARMV: u8 = 11`,
    `ZEROMV: u8 = 12`, `NEWMV: u8 = 13`.
  * New public helper `mode_to_mode_type(mode: u8) -> usize` —
    exposes the §8.8.4 step-1 classification so a future §8.8.2
    raster walker can derive `modeType` directly from `YModes[ ][
    ]` without re-reading the lookup.
  * 11 new lib-side `adaptive_filter_strength::tests` (lib total
    496 -> 507): mode→modeType classification across the §7.4.11
    inter modes and the §7.4.5 intra modes, sharpness = 0 baseline
    `(lvl=16, limit=16, blimit=52, thresh=1)`, sharpness = 5 shift =
    2 + Clip3, sharpness = 1 shift = 1 + Clip3, sharpness = 0 Max
    lower clip at `lvl = 0`, sharpness > 0 Clip3 lower clip at `lvl
    = 0`, sharpness = 7 Clip3 cap at `limit = 2`, blimit high-water
    mark at `lvl = 63` / sharpness = 1, modeType dispatch picks the
    correct `LvlLookup[s][ref][m]` column under non-zero mode-
    delta, segment override propagates into the lookup, out-of-
    range axes return `None` without panic.
  * 7 new integration tests in `tests/adaptive_filter_strength.rs`:
    end-to-end `level = 25 / sharpness = 0 / intra` returns
    `(lvl=25, limit=25, blimit=79, thresh=1)`; end-to-end
    NEARESTMV / LAST_FRAME with `delta_enabled = 1`; modeType
    column routing under non-zero mode-delta; full `0..=7`
    sharpness sweep at `lvl = 40` against an independent
    re-derivation of the §8.8.4 formulas; `thresh` partition over
    `lvl ∈ {15, 16, 31, 32, 47, 48, 63}`; public `mode_to_mode_type`
    surface; `MAX_LOOP_FILTER` caps `thresh` at 3.
  * Out of scope: §8.8.2 superblock raster walk, §8.8.5 sample
    filtering, §6.2.5 frame_size_with_refs.
* **Round 244: §8.8.3 `filter_size( )` lifted to a public leaf
  primitive — [`filter_size`].** New per-edge filter-size derivation
  built from the `(tx_sz, is_32_edge, pass, x, y, sub_x, sub_y,
  mi_cols, mi_rows)` inputs the §8.8.2 superblock raster walk will
  supply per spec `vp9-spec.txt` §8.8.3 lines 5587-5625. Signature:
  `filter_size(tx_sz: u8, is_32_edge: bool, pass: u8, x: u32, y:
  u32, sub_x: u8, sub_y: u8, mi_cols: u32, mi_rows: u32) -> u8`.
  * Step 1 `baseSize` derivation per lines 5609-5611: the `txSz ==
    TX_4X4 && is32Edge == 1 → baseSize = TX_8X8` promotion (the
    §8.8.3 lead paragraph "minimum size of TX_8X8 for boundaries
    on a multiple of 32 samples" rule) and the otherwise-branch
    `baseSize = Min(TX_16X16, txSz)` clip (cap below `TX_32X32`).
  * Step 2 vertical chroma right-edge clip per lines 5615-5619:
    `pass == 0 && sub_x == 1 && baseSize == TX_16X16 && (x >> 3)
    == MiCols - 1 → TX_8X8`. Realises the §8.8.3 lead paragraph
    "reduce the width of chroma filters" rule.
  * Step 2 horizontal chroma bottom-edge clip per lines 5620-5624:
    mirror gate on `pass == 1 && sub_y == 1 && baseSize ==
    TX_16X16 && (y >> 3) == MiRows - 1 → TX_8X8`.
  * Otherwise `filterSize = baseSize` per line 5625.
  * New public constants verbatim from §7.4.8 (`vp9-spec.txt`
    lines 3937-3940): `TX_4X4: u8 = 0`, `TX_8X8: u8 = 1`,
    `TX_16X16: u8 = 2`, `TX_32X32: u8 = 3`. Two §8.8.3 pass-
    direction integers: `PASS_VERTICAL: u8 = 0` and
    `PASS_HORIZONTAL: u8 = 1`.
  * 14 new lib-side `filter_size::tests` (lib total 482 -> 496):
    `Min` clip on `TX_8X8` interior, `Min` cap of `TX_32X32` at
    `TX_16X16`, `TX_4X4` interior keeps `TX_4X4`, `TX_4X4 +
    is_32_edge` promotion to `TX_8X8`, vertical chroma right-edge
    clip fires, vertical clip skipped on horizontal pass, vertical
    clip skipped when `sub_x = 0`, vertical clip skipped when
    `baseSize < TX_16X16`, vertical clip skipped on interior edge,
    horizontal chroma bottom-edge clip fires, horizontal clip
    skipped on vertical pass, horizontal clip skipped when `sub_y
    = 0`, `is32Edge` promotion doesn't fire the chroma clip
    (`baseSize != TX_16X16` after promotion), and the
    `mi_cols == 0` / `mi_rows == 0` no-clip edge case.
  * 8 new integration tests in `tests/filter_size.rs`: `Min` clip
    via public API, `is_32_edge` promotion via public API,
    vertical chroma-clip sweep over `mi_cols ∈ [1, 8]`, horizontal
    chroma-clip sweep over `mi_rows ∈ [1, 8]`, luma 8x8 grid
    stays unclipped on both passes, `TX_*` constants match §7.4.8
    values, pass-direction constants verified, and the `TX_32X32`
    right-edge chroma clip via the intermediate `TX_16X16`
    `baseSize`.

* **Round 37: §8.8.1 `loop_filter_frame_init( )` lifted to a public
  primitive — [`loop_filter_frame_init`].** New per-frame book-keeping
  function building the `LvlLookup[ MAX_SEGMENTS ][ MAX_REF_FRAMES ][
  MAX_MODE_LF_DELTAS ]` filter-strength lookup table per spec
  `vp9-spec.txt` §8.8.1 lines 5465-5488. Signature:
  `loop_filter_frame_init(lf: &LoopFilterParams, seg:
  &SegmentationParams, ref_deltas: [i8; 4], mode_deltas: [i8; 2]) ->
  LvlLookup`.
  * Covers all four §8.8.1 steps: step 1 `lvlSeg = loop_filter_level`,
    step 2 `seg_feature_active( SEG_LVL_ALT_L )` segment override
    (with §6.2.11 abs/delta mode handling + §8.8.1 step 2.c
    `Clip3( 0, MAX_LOOP_FILTER, lvlSeg )` saturation), step 3
    `delta_update == 0` per-segment broadcast, and step 4
    `delta_enabled == 1` per-(ref, mode) delta-apply walk (with the
    §8.8.1 line 5481 / 5482-5487 split: `INTRA_FRAME / mode 0` writes
    line 5481, `LAST_FRAME..ALTREF_FRAME / 0..MAX_MODE_LF_DELTAS - 1`
    writes lines 5482-5487; the `INTRA_FRAME / mode 1` cell is never
    touched by step 4).
  * `nShift = loop_filter_level >> 5` line 5468: the deltas scale by
    `<< nShift` so a `level >= 32` doubles every `±1` delta into
    `±2` (and `level >= 64` would 4x, but `MAX_LOOP_FILTER = 63`
    caps the input).
  * Caller supplies resolved `ref_deltas[ 4 ]` / `mode_deltas[ 2 ]`
    (post-`Option::unwrap_or(prev)`) per §7.2's "previous value"
    rule. The §7.2 `setup_past_independence` defaults are
    `loop_filter_ref_deltas = [1, 0, -1, -1]` and
    `loop_filter_mode_deltas = [0, 0]`.
  * New `pub struct LvlLookup { pub levels: [[[u8; 2]; 4]; 8] }` with
    `LvlLookup::zeros()` no-filter identity constructor and a
    bounds-checked `get(segment_id: usize, ref_frame: i32, mode:
    usize) -> Option<u8>` read-back.
  * New public constants: `MAX_MODE_LF_DELTAS: usize = 2` (§3
    `vp9-spec.txt` line 513), `MAX_LOOP_FILTER: i32 = 63` (§3 line
    515). Crate-local `SEG_LVL_ALT_L: usize = 1` (§3 line 476).
  * 13 new lib-side `loop_filter::tests` (lib total 469 -> 482): the
    all-disabled zero base case, the step-3 broadcast, the
    `delta_update + delta_enabled` step-4 cover excluding `INTRA / 1`,
    the `nShift = 1` threshold at level 32, both `Clip3` saturations
    (0 and 63), step 2.a abs-mode replacement, step 2.b delta-mode
    addition, step 2.c underflow + overflow clips, the §6.4.9
    `segmentation_enabled == 0` gate that makes step 2 a no-op even
    when `feature_enabled[ ][ SEG_LVL_ALT_L ] == 1`, the
    `INTRA_FRAME / 1` cell retention when step 3 broadcasts then step
    4 partial-overwrites, and the `LvlLookup::get` bounds-check.
  * 5 new integration tests in `tests/loop_filter.rs` (suite total
    499 -> 517): step-3 broadcast via public API, `nShift` threshold
    sweep at levels 31 / 32 / 63 with `Clip3` saturation, step 3 +
    step 4 composition leaving `INTRA_FRAME / 1` at broadcast value,
    step 2 segment-specific override (only the configured segment
    sees the alt level), and `LvlLookup::zeros()` identity.

* **Round 36: §6.4 lines 2306-2311 byte-walk lifted to a public
  primitive — [`tile_payload_sizes`].** Factors the pure
  byte-arithmetic prefix walk out of the round-33 `decode_tiles`
  outer driver into a new public function:
  `tile_payload_sizes(data, sz, tile_rows_log2, tile_cols_log2) ->
  Result<Vec<u32>, Error>`. The helper walks the `(1 <<
  tile_rows_log2) x (1 << tile_cols_log2)` grid in row-major order
  per §6.4 lines 2304-2305, reads the `f(32)` length prefix per line
  2310 for every tile except the last, applies the
  `sz -= tile_size + 4` running subtraction per line 2311 with
  checked arithmetic, assigns `tile_size = sz` per line 2308 to the
  last tile, and range-checks every declared body against `data`.
  The §9.2 bool coder and the §6.4.2 `decode_tile( )` body are not
  invoked — this is the demuxer slice a caller needs to split a
  frame's tile payload into per-tile bool-coder sub-streams without
  decoding any block content.
  * `decode_tiles` is refactored to call `tile_payload_sizes` for
    the prefix walk; the per-tile slice fetch can then trust that
    every tile body is in-bounds, removing the duplicate
    range-check from the per-tile loop.
  * `Error::UnexpectedEof` — non-last tile's 4-byte prefix runs
    past the end of `data`, or a declared `tile_size` extends past
    the available byte slice.
  * `Error::InvalidBitstream` — declared `tile_size + 4` would
    underflow the running `sz` budget per §6.4 line 2311.
  * 5 new partition::tests cases (lib total 464 -> 469): single-tile
    pass-through returning `[sz]`; two-tile horizontal layout
    matching the `docs/video/vp9/fixtures/tile-cols-2` per-frame
    trace (`tile_size` 662 + 635 totalling 1301 bytes with one
    4-byte prefix); 2x2 grid emitting four distinguishable sizes in
    row-major order (would catch a transpose); 3-byte input
    rejected with `UnexpectedEof` at the first `f(32)` prefix;
    declared `tile_size = u32::MAX` rejected with
    `InvalidBitstream` at the §6.4 line 2311 underflow.
  * `pub use partition::tile_payload_sizes;` exposes the helper on
    the crate root.

* **Round 35: §6.3 `parse_compressed_header_inter` integration-test
  coverage.** Pins the round-34 inter outer-dispatch entry point at
  the public-API boundary in `tests/compressed_header.rs`. Ten new
  integration tests cover the zero-buffer default-table pass-through
  across the full §6.3.1..§6.3.16 chain, the §6.3.10
  `interpolation_filter == SWITCHABLE` gate, the §6.3.12
  `compoundReferenceAllowed` short-circuit vs. bool-coder-reading
  arm, the §6.3.16 `allow_high_precision_mv` tail gate, lossless and
  non-lossless intra-prefix parity with `parse_compressed_header`,
  shared `init_bool` error surface (empty buffer + non-zero marker)
  against the intra walker, and the
  `RefFrameSignBias::from_inter_biases` / `get` public-surface
  round-trip across all eight §6.2.5 sign-bias tuples (the §3
  `INTRA_FRAME` slot stays 0). Anchors §10.5 default values
  transcribed verbatim from the spec listing: `default_is_inter_prob
  = {9, 102, 187, 225}`, `default_inter_mode_probs[ 0 ] = {2, 173,
  34}`, `default_mv_class0_hp_prob = {160, 160}`, `default_mv_hp_prob
  = {128, 128}`, plus the §6.3.2 / §6.3.7 / §6.3.8 anchors reused
  from the existing intra tests. Suite total 484 → 494 (+10
  integration tests).

* **Round 34: §6.3 `if ( FrameIsIntra == 0 )` outer dispatch —
  [`parse_compressed_header_inter`] entry point.** Wires the
  inter-frame arm of the §6.3 compressed-header listing
  (`vp9-spec.txt` lines 1964-1974) into a new public entry point
  composing the round-22..30 inter-only primitives in spec order.
  * `parse_compressed_header_inter(data, lossless, inputs) ->
    Vp9CompressedHeaderInter` per §6.3. Runs the intra-shared prefix
    (§6.3.1 / §6.3.2 / §6.3.7 / §6.3.8) via an extracted crate-local
    helper, then walks the inter-only tail: §6.3.9
    `read_inter_mode_probs( )` → §6.3.10 `read_interp_filter_probs( )`
    (gated on `interpolation_filter == SWITCHABLE`) → §6.3.11
    `read_is_inter_probs( )` → §6.3.12 `frame_reference_mode( )`
    (which also fires §6.3.18 `setup_compound_reference_mode( )` on
    non-`SingleReference` arms) → §6.3.13
    `frame_reference_mode_probs( )` → §6.3.14 `read_y_mode_probs( )` →
    §6.3.15 `read_partition_probs( )` → §6.3.16 `mv_probs( )` (with
    §6.3.17 `update_mv_prob( )` per cell, plus the conditional
    high-precision tail).
  * `Vp9CompressedHeaderInterInputs { interpolation_filter_is_switchable,
    ref_frame_sign_bias, allow_high_precision_mv }`: bundles the
    three §6.2-derived flags the inter tail needs from the
    uncompressed-header walker (§6.2.7 `read_interpolation_filter( )`
    + §6.2.5 `ref_frame_sign_bias[ ]` + §6.2.7
    `allow_high_precision_mv`).
  * `Vp9CompressedHeaderInter { intra, inter_mode_probs[7][3],
    interp_filter_probs[4][2], is_inter_prob[4], reference_mode,
    compound_reference_config, comp_mode_prob[5],
    single_ref_prob[5][2], comp_ref_prob[5], y_mode_probs[4][9],
    partition_probs[16][3], mv_probs }`: post-§6.3.16 state of every
    inter-only probability table plus the §6.3.12 frame-level
    `reference_mode` decision and (when compound is active) the
    §6.3.18 fixed-vs-variable ref-frame partition. `intra` is a
    `Vp9CompressedHeader` matching what [`parse_compressed_header`]
    returns on the same intra-shared prefix bit-for-bit.
  * `RefFrameSignBias`, `ReferenceMode`, `CompoundReferenceConfig`,
    `MvProbs` promoted from `pub(crate)` to `pub` (they surface in
    `Vp9CompressedHeaderInter` / `Vp9CompressedHeaderInterInputs`).
  * +11 lib tests (lib 453 → 464; suite 473 → 484): zero-buffer
    preserves all §10 / §10.5 defaults; intra-shared prefix matches
    intra-only walker (non-lossless + lossless); §6.3.10 gate skips
    walker on `interpolation_filter != SWITCHABLE`; §6.3.12
    `compoundReferenceAllowed == 0` short-circuit (sign-bias
    `(0,0,0)`) vs. mixed-bias path; §6.3.16 high-precision tail
    gating on `allow_high_precision_mv`; empty buffer surfaces same
    `InvalidBitstream` error as intra walker; full composed walk
    bit-identical to explicit independent hand-walk against every
    §6.3.x primitive in spec order; `RefFrameSignBias::from_inter_biases`
    / `get` round-trip across all eight sign-bias tuples; inputs
    bundle is `Copy`.

* **Round 33: §6.4 `decode_tiles( )` outer driver — frame-level tile
  walk.** Composes the round-32 §6.4.1 / §6.4.2 primitives into the
  full `(1 << tile_rows_log2) × (1 << tile_cols_log2)` frame walk per
  `vp9-spec.txt` lines 2300-2331.
  * `decode_tiles( data, sz, tile_rows_log2, tile_cols_log2, mi_rows,
    mi_cols, ctx_state, probs_kind )` per §6.4: derives `tileCols` /
    `tileRows` from the §7.2.11 log2 fields, fires
    `PartitionContextState::clear_above( )` (the §7.4.1
    `clear_above_context( )` reset) once before the tile walk, then
    iterates `(tileRow, tileCol)` in row-major order. For every tile
    except the last it reads `tile_size  f(32)` (big-endian) from the
    byte stream and runs `sz -= tile_size + 4`; on the last tile
    `tile_size = sz`. Per tile it derives the four MI extents via the
    §6.4.1 helper, brackets a fresh `BoolCoder` with `init_bool(
    tile_size ) / exit_bool( )` per §9.2.1 / §9.2.3, and invokes the
    §6.4.2 `decode_tile( )` primitive. Returns `Vec<DecodedTile>`
    where each entry carries `(tile_row, tile_col, mi_row_start,
    mi_row_end, mi_col_start, mi_col_end, tile_size, leaves)` for
    downstream replay.
  * `PartitionContextState::clear_above( )` per §7.4.1: the dual of
    the round-32 `clear_left( )` reset, zeroes
    `AbovePartitionContext[ ]` once per `decode_tiles( )` invocation.
  * `DecodedTile { tile_row, tile_col, mi_row_start, mi_row_end,
    mi_col_start, mi_col_end, tile_size, leaves }`: per-tile record
    bundling the §6.4 listing's four `get_tile_offset( )` outputs,
    the `tile_size` byte budget, and the per-tile §6.4.2 leaf log.
  * Bitstream-error surface: §6.4 line 2310 underflow on the f(32)
    read → `Error::UnexpectedEof`; declared `tile_size` whose `(+ 4)`
    addend would exceed remaining `sz` → `Error::InvalidBitstream`;
    declared `tile_size` larger than the available byte stream →
    `Error::UnexpectedEof`; per-tile `init_bool( )` marker rejection
    or `exit_bool( )` non-zero-padding → `Error::InvalidBitstream`.
  * +13 lib tests (lib 440 → 453; suite 460 → 473): single-tile
    `lastTile = true` consuming the full payload; §6.4 line 2303
    `clear_above_context( )` zeroing a pre-poisoned strip before the
    first tile; two-tile horizontal split reading one `f(32)` prefix
    and `MiColStart` / `MiColEnd` matching the §6.4.1 split at
    `(0, 8, 16)`; 2×2 grid iterating `(0,0) → (0,1) → (1,0) →
    (1,1)`; last-tile skipping `f(32)` prefix (back-to-back bodies
    with one prefix only); output `Vec<DecodedTile>` length matching
    `tileRows * tileCols`; truncated 3-byte stream raising
    `UnexpectedEof` at the `f(32)` fetch; oversized declared
    `tile_size = u32::MAX` raising `InvalidBitstream`; truncated tile
    body (declared 8, supplied 6) raising `UnexpectedEof`;
    nonzero-marker first byte (`0x80`) raising `InvalidBitstream`
    from `init_bool( )`; 1×2 vertical split partitioning MI rows at
    `(0, 8, 16)`; full 2×2 grid invariant that consecutive tiles are
    contiguous within rows AND within columns; and
    `PartitionContextState::clear_above( )` zero-strip dual to the
    round-32 `clear_left` invariant.

* **Round 32: §6.4.1 `get_tile_offset( )` + §6.4.2 `decode_tile( )` —
  tile-driver primitive layer.** Lifts the §6.4.3 recursive partition
  driver landed in round 19 into the §6.4.2 superblock-row driver and
  the §6.4.1 per-tile-axis offset arithmetic that §6.4
  `decode_tiles( )` composes them with.
  * `get_tile_offset( tile_num, mis, tile_sz_log2 )` per §6.4.1
    (`vp9-spec.txt` lines 2335-2338): three-line pure-u32 helper —
    `sbs = (mis + 7) >> 3`, `offset = ((tile_num * sbs) >>
    tile_sz_log2) << 3`, `Min( offset, mis )`. Used by the §6.4
    outer driver four times per tile to derive `MiRowStart` /
    `MiRowEnd` / `MiColStart` / `MiColEnd`.
  * `decode_tile( coder, mi_row_start, mi_row_end, mi_col_start,
    mi_col_end, mi_rows, mi_cols, ctx_state, probs_kind, leaves )` per
    §6.4.2 (`vp9-spec.txt` lines 2343-2349): outer `r ∈ [Start, End)`
    step-8 loop firing `PartitionContextState::clear_left( )` (the
    §7.4.2 `clear_left_context( )` reset) once per superblock-row
    start, inner `c ∈ [Start, End)` step-8 loop calling
    `decode_partition( r, c, BLOCK_64X64, ... )` once per superblock
    origin.
  * +12 lib tests covering: §6.4.1 single-tile (`tile_sz_log2 == 0`)
    cases for sb64-aligned and non-aligned `mis`, the past-end clamp;
    two-tile case (`tile_sz_log2 == 1`, `mis = 16`) producing
    `(0, 8, 16)`; the `Min` clamp on `tile_sz_log2 == 2`,
    `mis = 8`; consecutive-pair cover proof; an 8-alignment sweep
    across `mis ∈ {8, 16, 32, 64, 256}` and `tile_sz_log2 ∈
    {0, 1, 2, 3}`; §6.4.2 empty-window early-return; single-sb64
    tile producing one leaf; two-sb-wide row producing leaves in
    `c` order; two-sb-tall column producing leaves in `r` order plus
    sentinel-proof that `clear_left_context( )` fires at the START
    of the second row; 2×2 sb64 row-major traversal order; sub-tile
    MI window starting at `(8, 8)`; §6.4.1 + §6.4.2 composition
    splitting a 16-MI-wide frame into two tiles.
  * Surface stays internal-only (`pub(crate)` with
    `#[allow(dead_code)]` on `get_tile_offset` / `decode_tile`).
    Public API still exposes `parse_uncompressed_header`,
    `parse_compressed_header` and their result types exclusively.

* **Round 31: §6.4.4 `decode_block( r, c, subsize )` driver — pure-state
  fan-out primitive.** Lands the §6.4.4 per-leaf driver as a standalone
  book-keeping primitive that consumes the per-MI outputs of
  `mode_info( )` and `residual( )` (decoded by the §6.4.5 / §6.4.6 /
  §6.4.15 / §6.4.21 primitives landed in earlier rounds) and fans them
  into the frame-wide §6.4.4 arrays at every `(r + y, c + x)` cell for
  `y ∈ 0..num_8x8_blocks_high_lookup[ subsize ]`,
  `x ∈ 0..num_8x8_blocks_wide_lookup[ subsize ]`.
  * `decode_block_apply( state, r, c, subsize, result )` per §6.4.4
    (`vp9-spec.txt` lines 2395-2437): two phases — the `skip = 1`
    rewrite under `is_inter ∧ subsize ≥ BLOCK_8X8 ∧ EobTotal = 0`
    (lines 2405-2407), then the `num_8x8_blocks_*_lookup[ subsize ]`
    fan-out (lines 2408-2436) writing ten cells per `(y, x)` step
    (`Skips` / `TxSizes` / `MiSizes` / `YModes` / `SegmentIds` /
    `RefFrames[ 0..2 ]` + `InterpFilters` / `Mvs[ 0..2 ]` /
    `SubMvs[ 0..2 ][ 0..4 ]` on `is_inter`, `SubModes[ 0..4 ]` on
    `!is_inter`). Returns the rewritten `skip` value.
  * `DecodedBlockResult { skip, tx_size, y_mode, segment_id,
    ref_frame[ 2 ], is_inter, eob_total, interp_filter,
    block_mvs[ 2 ][ 4 ], sub_modes[ 4 ] }` bundles the per-MI values
    upstream §6.4.5 / §6.4.6 / §6.4.15 / §6.4.21 produce. `Default`
    seeds `ref_frame = [INTRA_FRAME = 0, NONE = -1]` per §6.4.6
    lines 2469-2470 (intra-block init).
  * `Vp9FrameState { mi_cols, mi_rows, skips, tx_sizes, mi_sizes,
    y_modes, segment_ids, ref_frames, interp_filters, mvs, sub_mvs,
    sub_modes }` owns the `MiRows × MiCols` (× 2 / × 4 for the
    per-`refList` / per-sub-block strides) §6.4.4 write-back arrays
    in row-major order. Accessors return `Option<T>` for out-of-frame
    coordinates per §7.4.3 defensive bounds.
  * +13 lib tests covering: §6.4.4 intra-default top-left write;
    BLOCK_8X8 single-cell write with `sub_modes` propagation;
    BLOCK_16X16 2×2 fan-out; BLOCK_64X64 full-8×8-MI-frame fan-out;
    the §6.4.4 `skip = 1` rewrite firing under all three preconditions
    and the three non-firing cases (sub-8×8, `EobTotal > 0`,
    intra block); inter-branch `Mvs` / `SubMvs` / `InterpFilters`
    write-back; `ref_frame[ 0..2 ]` writes on both branches; §7.4.3
    out-of-frame clip on a 32×32 block at the edge of an 8×8 MI frame;
    `skip = 1` propagating into every cell of the fan-out; and §10.2
    `num_8x8_blocks_*_lookup[ ]` table pinning.
  * Lib total 415 → 428; suite total 435 → 448.
  * Out of scope: wiring `decode_block_apply` into the §6.4.3
    [`partition::decode_partition`] driver — the swap is mechanical
    (the existing leaf log carries `(r, c, subsize)`) but requires a
    frame-state allocator + per-leaf §6.4.5 `mode_info( )` invocation
    which sits outside the §6.4.4 scope.

* **Round 30: §6.3.16 `mv_probs( )` compressed-header outer sweep.**
  Closes the §6.3.x primitives chain by landing the final 65/69-cell
  MV-probability walk that drives the §6.3.17 [`update_mv_prob`]
  per-cell primitive across nine `mv_*_prob[ ]` arrays:
  * `mv_probs( coder, probs, allow_high_precision_mv )` per §6.3.16
    (`vp9-spec.txt` lines 2234-2259). Three unconditional phases —
    joint probs (3 cells), per-component bulk (2 × 22 = 44 cells:
    sign + class + class0_bit + bits) and per-component fractional (2
    × 9 = 18 cells: class0_fr + fr) — plus one conditional tail
    (high-precision: 2 × 2 = 4 cells, gated on
    `allow_high_precision_mv`). Totals: **65 cells** (no HP), **69
    cells** (HP). Every cell consumes one `B(252)` `update_mv_prob`
    flag plus, on flag-set, seven extra `L(7)` literal bits.
  * `MvProbs { joint_probs, sign_prob, class_probs, class0_bit_prob,
    bits_prob, class0_fr_probs, fr_probs, class0_hp_prob, hp_prob }`
    bundles the nine arrays as a single mutable target; the
    `MvProbs::defaults()` constructor seeds every slot from the §10.5
    listings (single source of truth in `mode_info.rs`).
  * §3 MV-constants transcribed into `mode_info.rs`: `MV_JOINTS = 4`
    (line 508), `MV_CLASSES = 11` (line 509), `CLASS0_SIZE = 2`
    (line 510), `MV_OFFSET_BITS = 10` (line 511), `MV_FR_SIZE = 4`
    (line 458). Nine §10.5 default tables transcribed verbatim:
    `DEFAULT_MV_JOINT_PROBS = [32, 64, 96]`,
    `DEFAULT_MV_SIGN_PROB = [128, 128]`,
    `DEFAULT_MV_CLASS_PROBS` (2 × 10),
    `DEFAULT_MV_CLASS0_BIT_PROB = [216, 208]`,
    `DEFAULT_MV_BITS_PROB` (2 × 10),
    `DEFAULT_MV_CLASS0_FR_PROBS` (2 × 2 × 3),
    `DEFAULT_MV_FR_PROBS` (2 × 3),
    `DEFAULT_MV_CLASS0_HP_PROB = [160, 160]`,
    `DEFAULT_MV_HP_PROB = [128, 128]`.
  * +13 lib tests covering: cell-count constants (3 + 44 + 18 = 65,
    +4 HP); §10.5 default-table transcription cross-check; zero-buffer
    pass-through with `hp ∈ {false, true}` on both defaulting bundle
    and custom-starts path; cursor-equivalence proofs at 65 and 69
    cells; four-flag-catch-up cursor proof that the HP tail
    contributes exactly 4 bool-coder reads; explicit phase-walk
    equivalence against a hand-coded §6.3.16 listing walker (two
    starts × `hp ∈ {false, true}`); HP-field preservation under
    `allow_high_precision_mv == false`; §3 constant pinning; and a
    defaults-vs-`mode_info` single-source-of-truth audit.
  * Lib test count 402 → 415; suite total 422 → 435. §6.3.x
    primitives chain (§6.3.1 → §6.3.18 inclusive) is now complete
    modulo wiring into the outer dispatch.

* **Round 29: §6.3.12 `frame_reference_mode( )` compressed-header
  outer driver.** Two-`L(1)` walker that gates the §6.3.18
  [`setup_compound_reference_mode`] caller and decides the frame-
  level `reference_mode`:
  * `frame_reference_mode( coder, ref_frame_sign_bias )` per §6.3.12
    (`vp9-spec.txt` lines 2170-2191). Computes
    `compoundReferenceAllowed` via the §3 loop
    `for ( i = 1; i < REFS_PER_FRAME; i++ )` against
    `ref_frame_sign_bias[ 1 ]` (`LAST_FRAME`). All-agree sign-bias
    tuples `(0, 0, 0)` / `(1, 1, 1)` short-circuit to
    `SingleReference` with zero bool-coder reads; other six tuples
    read `L(1) non_single_reference` and, on 1, `L(1)
    reference_select` then invoke §6.3.18
    [`setup_compound_reference_mode`].
  * Returns `(ReferenceMode, Option<CompoundReferenceConfig>)`:
    `SingleReference` arms return `None`; `CompoundReference` and
    `ReferenceModeSelect` arms return the §6.3.18 partition of
    `{LAST_FRAME, GOLDEN_FRAME, ALTREF_FRAME}` into
    `(CompFixedRef, CompVarRef[ 2 ])`.
  * `REFS_PER_FRAME = 3` constant transcribed from §3
    (`vp9-spec.txt` line 457) into `mode_info.rs`.
  * +11 lib tests covering: all-agree short-circuit on `(0,0,0)` /
    `(1,1,1)`, zero-bool-read cursor proof; the six allowed tuples'
    one-`L(1)` SingleReference path with cursor-equivalence;
    brute-forced 16-bit prefix-space searches for `(L(1)=1, L(1)=0)`
    CompoundReference and `(L(1)=1, L(1)=1)` ReferenceModeSelect
    buffers; two-`L(1)` cursor-equivalence proofs on each compound
    arm; cross-check that the returned CompoundReferenceConfig
    matches §6.3.18 directly; exhaustive 8-tuple allowed-vs-not
    predicate match against the inline §6.3.12 loop; step-walk
    equivalence between production code and a re-derived listing
    walker across all 32 (sign-bias × buffer) combinations.
  * Lib test count 391 → 402; suite total 411 → 422.

* **Round 28: §6.3.18 `setup_compound_reference_mode( )`
  compressed-header pure-compute leaf.** Closes the §6.3.x primitives
  chain modulo the still-deferred §6.3.12 `frame_reference_mode( )`
  and §6.3.16 `mv_probs( )` outer drivers:
  * `setup_compound_reference_mode( ref_frame_sign_bias )` per §6.3.18
    (`vp9-spec.txt` lines 2279-2296). Pure compute — no bool-coder
    reads. Partitions the three §3 inter reference frames
    (`LAST_FRAME`, `GOLDEN_FRAME`, `ALTREF_FRAME`) into a
    `CompFixedRef` plus `CompVarRef[ 2 ]` pair based on the §6.2.5
    `ref_frame_sign_bias[ ]` `f(1)` flags via the three-arm if/else
    chain: branch 1 (`LAST == GOLDEN` => fixed = `ALTREF`); branch 2
    (`LAST != GOLDEN AND LAST == ALTREF` => fixed = `GOLDEN`); branch
    3 (else => fixed = `LAST`).
  * §3 ref-frame enumeration transcribed into `mode_info.rs` alongside
    the existing `INTRA_FRAME = 0`: `LAST_FRAME = 1`, `GOLDEN_FRAME =
    2`, `ALTREF_FRAME = 3`, `MAX_REF_FRAMES = 4` (spec line 470).
  * `RefFrameSignBias` newtype + `from_inter_biases(last, golden,
    altref)` constructor enforces the §6.2.5 "inter slots only"
    invariant; `CompoundReferenceConfig { fixed_ref, var_ref }`
    bundles the §6.3.18 output for downstream §6.4.16 `comp_ref` /
    §6.5 MV-reference consumption.
  * +9 lib tests covering §3 sentinel pinning, each of the three
    branches, exhaustive 8-tuple truth-table sweep, branch-1
    precedence on `(0,0,0)` / `(1,1,1)`, pairwise-distinct-and-
    permutation-of-inter-set invariant, inter-only population on
    `RefFrameSignBias`, and pure-compute / type-level signature pin.
  * Lib test count 382 → 391; suite total 402 → 411.

## [0.0.11](https://github.com/OxideAV/oxideav-vp9/compare/v0.0.10...v0.0.11) - 2026-05-29

### Other

- Round 27: §6.3.17 update_mv_prob() compressed-header primitive
- Round 26: §6.3.15 read_partition_probs() compressed-header sweep
- Round 25: §6.3.13 frame_reference_mode_probs( ) compressed-header sweep
- Round 24: §6.3.14 read_y_mode_probs compressed-header sweep
- Round 23: §6.3.9 read_inter_mode_probs + §6.3.10 read_interp_filter_probs
- Round 22: §6.3.11 read_is_inter_probs() compressed-header sweep
- Round 21: §6.4.13 read_is_inter + §9.3.2 is_inter ctx + §10.5 default_is_inter_prob
- Round 20: §6.4.12 inter_segment_id + §6.4.14 get_segment_id + §7.4 seg-pred ctx
- Round 19: §6.4.3 recursive decode_partition() driver
- Round 18: §6.4.3 decode_partition_type() per-call partition reader
- §6.4.15 intra_block_mode_info() inter-frame intra-block reader
- roadmap reflects round 17 landing intra_frame_mode_info()
- Round 17: §6.4.6 intra_frame_mode_info() keyframe driver
- Round 16: §6.4.7 intra_segment_id + §9.3.1 segment_tree
- vp9 round 15: §6.4.8 read_skip + §6.4.10 read_tx_size + §9.3.3 tree_decode
- vp9 round 14: §6.4.21 residual() intra driver
- vp9 round 13: §6.4.24 tokens() per-block coefficient driver
- vp9 round 12: §6.4.25 get_scan scan-order selection
- round 11 — §8.6.2 reconstruct driver (reconstruct module)
- round 10 — §8.5.1 intra prediction process (intra module)
- round 9: §8.7 inverse transform process (idct module)
- round 8: §8.6.1 dequantization functions (dequant module)
- §6.4.24 / §6.4.26 coefficient-token decoder (round 7)
- round 6: §6.3.7 read_coef_probs 6D sweep + default_coef_probs
- Round 5: §6.3.2 tx_mode_probs + §6.3.8 read_skip_prob sweeps
- round 4: §6.3.3 diff_update_prob chain + §6.3.4..§6.3.6 helpers + INV_MAP_TABLE
- round 3: §9.2 Boolean decoder + §6.3.1 read_tx_mode walk
- round 2: full §6.2 uncompressed-header walk + §6.1.1 trailing_bits
- round 1: uncompressed-header walker per VP9 spec v0.7 §6.2
- orphan rebuild: clean-room scaffold post 2026-05-20 audit

### Added

* **Round 27: §6.3.17 `update_mv_prob( prob )` compressed-header
  per-cell primitive.** Lands the per-call MV-probability-update
  helper consumed by every cell of the still-deferred §6.3.16
  `mv_probs( )` sweep:
  * `update_mv_prob( coder, prob )` per §6.3.17 (`vp9-spec.txt` lines
    2261-2275). Two-stage primitive — read one `B(252)`
    `update_mv_prob` flag and, on 1, pull a 7-bit `L(7)` `mv_prob`
    literal and rewrite `prob = (mv_prob << 1) | 1`. Otherwise the
    caller's `prob` is returned unchanged.
  * Distinct from §6.3.3 `read_diff_update_prob`: the diff-update
    primitive uses `decode_term_subexp` + `inv_remap_prob` and the
    output depends on the previous probability; the MV-update
    primitive ignores the previous probability entirely on the
    flag-set branch and produces a fresh value purely from the 7-bit
    literal. The `<< 1 | 1` rewrite forces odd parity and the
    `[1, 255]` step-2 range (MV probabilities can't be 0 because the
    §6.5.x MV tree decode treats 0 as an unconditional branch).
  * 8 new unit tests covering: zero-buffer pass-through (each base in
    `{0, 1, 7, 64, 127, 128, 129, 200, 254, 255}` returned unchanged);
    cursor-equivalence on the zero-buffer fast path (one `B(252)`
    consumed); brute-forced flag-set buffer producing a deterministic
    output independent of input base; cursor-equivalence on the
    flag-set branch (one `B(252)` + one `L(7)` consumed); a parity +
    range invariant sweep across every L(7) value 0..=127; baseline
    cross-check confirming the flag-set output is input-independent;
    a direct distinction test against §6.3.3 `read_diff_update_prob`
    proving the two primitives are not aliases (different output on
    the same flag-set buffer with the same base prob); and an explicit
    step-walk equivalence against a hand-coded §6.3.17 listing walker
    (zero buffer + flag-set buffer × 6 base values).
  * Surface stays internal-only (`pub(crate)` with
    `#[allow(dead_code)]`). Wiring into `parse_compressed_header`
    waits on §6.3.16 `mv_probs( )` and §6.3.12 `frame_reference_mode( )`,
    which need reference-buffer + `ref_frame_sign_bias[ ]` state the
    uncompressed-header walker still rejects with `Error::Unsupported`.

* **Round 26: §6.3.15 `read_partition_probs( )` compressed-header
  sweep.** Lands the unconditional `PARTITION_CONTEXTS = 16` ×
  `PARTITION_TYPES - 1 = 3` = 48-cell `diff_update_prob` walk that
  populates the running inter-frame `partition_probs[ ][ ]` table:
  * `read_partition_probs( coder, partition_probs )` per §6.3.15
    (`vp9-spec.txt` lines 2227-2232). 48 sequential
    `read_diff_update_prob` calls — one `B(252)` `update_prob` flag
    per cell and, on 1, a `decode_term_subexp( )` + `inv_remap_prob( )`
    cascade — updating `partition_probs[0..PARTITION_CONTEXTS][0..PARTITION_TYPES - 1]`
    in place.
  * §3 constants `PARTITION_CONTEXTS = 16` (line 463) and
    `PARTITION_TYPES = 4` (line 497) reused from the round-18
    `partition` module transcription. `DEFAULT_PARTITION_PROBS`
    (§10.5 lines 7623-7651) reused as the round-18 single source of
    truth.
  * `DEFAULT_PARTITION_PROBS_TABLE` re-export in `compressed.rs` keeps
    `partition::DEFAULT_PARTITION_PROBS` as the single source of truth
    (mirroring the round-22..25 staging pattern of one re-export per
    sweep). Same constant feeds the §6.4.3 `decode_partition_type( )`
    per-call partition decoder on inter frames via the §9.3.2
    `partition_plane_context( )` ctx.
  * 9 new unit tests covering: §3 `PARTITION_CONTEXTS = 16` and
    `PARTITION_TYPES = 4` pinning; verbatim §10.5 transcription of
    `default_partition_probs` (16-row × 3-col table with the four
    block-size-group / (above, left) split annotations preserved);
    the zero-buffer `update_prob = 0` pass-through preserving the
    starting table; all-cells-visited check with a non-uniform custom
    starting table; cursor-equivalence proof that the sweep consumes
    exactly 48 `B(252)` flags against a parallel-coder walker;
    row-major walk equivalence against a parallel coder for two
    distinct starting tables; tuple-sweep across distinct starting
    probabilities (`0, 1, 7, 64, 127, 128, 129, 200, 254, 255`)
    surviving zero-buffer pass-through; and a single-source-of-truth
    check that the `compressed.rs` re-export equals the
    `partition::DEFAULT_PARTITION_PROBS` constant.
  * Surface stays internal-only (`pub(crate)` with
    `#[allow(dead_code)]` on the function + re-export const). Wiring
    into `parse_compressed_header` waits on §6.3.12
    `frame_reference_mode( )` (and §6.3.16 `mv_probs( )` / §6.3.17),
    which need reference-buffer + `ref_frame_sign_bias[ ]` state the
    uncompressed-header walker still rejects with `Error::Unsupported`.

* **Round 25: §6.3.13 `frame_reference_mode_probs( )` compressed-header
  sweep.** Extends the §6.3 inter-arm primitives chain by the three
  reference-mode-gated sweeps over `comp_mode_prob`, `single_ref_prob`,
  `comp_ref_prob`, alongside the round-22..24 §6.3.9 / §6.3.10 / §6.3.11
  / §6.3.14 primitives:
  * `read_frame_reference_mode_probs( coder, reference_mode,
    comp_mode_prob, single_ref_prob, comp_ref_prob )` per §6.3.13
    (`vp9-spec.txt` lines 2195-2210). Conditional dispatch gated by
    the §3 sentinels:
    * `REFERENCE_MODE_SELECT` fires the `COMP_MODE_CONTEXTS = 5` cell
      `comp_mode_prob` sweep;
    * `!= COMPOUND_REFERENCE` fires the `REF_CONTEXTS × 2 = 10` cell
      `single_ref_prob` sweep;
    * `!= SINGLE_REFERENCE` fires the `REF_CONTEXTS = 5` cell
      `comp_ref_prob` sweep.
    Each cell consumes one `B(252)` `update_prob` flag and, on 1, a
    `decode_term_subexp( )` + `inv_remap_prob( )` cascade.
  * `pub enum ReferenceMode` mirrors §3 / §6.3.12 sentinels
    (`SingleReference = 0`, `CompoundReference = 1`,
    `ReferenceModeSelect = 2`).
  * §3 constants `COMP_MODE_CONTEXTS = 5`, `REF_CONTEXTS = 5`
    transcribed verbatim from `vp9-spec.txt` lines 472-473.
  * `DEFAULT_COMP_MODE_PROB`, `DEFAULT_COMP_REF_PROB`,
    `DEFAULT_SINGLE_REF_PROB` (`mode_info`) transcribed verbatim from
    §10.5 lines 7694-7710. `DEFAULT_COMP_MODE_PROB_TABLE` /
    `DEFAULT_COMP_REF_PROB_TABLE` / `DEFAULT_SINGLE_REF_PROB_TABLE`
    re-exports in `compressed.rs` keep `mode_info` as the single source
    of truth (mirroring round-22..24 staging patterns).
  * 12 new unit tests covering: §3 `COMP_MODE_CONTEXTS = 5` and
    `REF_CONTEXTS = 5` pinning; verbatim §10.5 transcription of each
    default table; `SingleReference` only touching `single_ref_prob`;
    `CompoundReference` only touching `comp_ref_prob`;
    `ReferenceModeSelect` firing all three sweeps; cursor-equivalence
    proofs that each branch consumes exactly its 5 / 10 / 20 `B(252)`
    flags against a parallel walker; row-major walk equivalence against
    a parallel coder for `ReferenceModeSelect` with two starting-table
    triples; and single-source-of-truth checks tying each
    `compressed.rs` re-export back to its `mode_info` constant.
  * Surface stays `pub(crate) + #[allow(dead_code)]` on the function +
    re-export consts; only `ReferenceMode` is `pub` so the §6.3.12
    walker can land it. Wiring into `parse_compressed_header` waits on
    §6.3.12 `frame_reference_mode( )` which needs `ref_frame_sign_bias[ ]`
    state the uncompressed-header walker still rejects with
    `Error::Unsupported`.

* **Round 24: §6.3.14 `read_y_mode_probs( )` compressed-header
  sweep.** Extends the §6.3 inter-arm primitives chain by one cell
  alongside the round-22 §6.3.11 `read_is_inter_probs( )` and
  round-23 §6.3.9 / §6.3.10 sweeps:
  * `read_y_mode_probs( coder, y_mode_probs )` per §6.3.14
    (`vp9-spec.txt` lines 2220-2225) — `BLOCK_SIZE_GROUPS = 4`
    (§3 line 460) × `INTRA_MODES - 1 = 9` (§3 line 505) = 36-cell
    row-major `read_diff_update_prob` sweep against the §9.3 / §10.5
    `default_y_mode_probs[ BLOCK_SIZE_GROUPS ][ INTRA_MODES - 1 ]`
    table.
  * `DEFAULT_Y_MODE_PROBS_TABLE` re-export in `compressed.rs`
    preserves `mode_info::DEFAULT_Y_MODE_PROBS` as the single source
    of truth (same constant feeds the (still-deferred) §7.4.5
    intra-mode tree decoder of `inter_block_mode_info( )`).
  * 8 new unit tests covering: §3 `BLOCK_SIZE_GROUPS = 4` and
    `INTRA_MODES = 10` constant pinning; verbatim §9.3 default-table
    transcription (row annotations preserved 0 = block_size < 8x8 …
    3 = block_size >= 32x32); zero-buffer `update_prob = 0`
    pass-through; all-cells-visited check with a non-uniform custom
    starting table; cursor-equivalence proof that the sweep consumes
    exactly 36 `B(252)` flags; explicit row-major walk equivalence
    against a parallel-coder reference (two starting tables); and
    a single-source-of-truth check tying
    `DEFAULT_Y_MODE_PROBS_TABLE` back to `mode_info::DEFAULT_Y_MODE_PROBS`.
  * Surface stays `pub(crate) + #[allow(dead_code)]` — wiring into
    `parse_compressed_header` waits on §6.3.12 (`frame_reference_mode( )`)
    and §6.3.13 (`frame_reference_mode_probs( )`) which need
    `ref_frame_sign_bias[ ]` state the uncompressed-header walker
    still rejects with `Error::Unsupported`.

* **Round 23: §6.3.9 `read_inter_mode_probs( )` + §6.3.10
  `read_interp_filter_probs( )` compressed-header sweeps.** Extends
  the §6.3 inter-arm primitives chain alongside the round-22 §6.3.11
  `read_is_inter_probs( )`:
  * `read_inter_mode_probs( coder, inter_mode_probs )` per §6.3.9
    (`vp9-spec.txt` lines 2138-2143) — `INTER_MODE_CONTEXTS = 7`
    × `INTER_MODES - 1 = 3` = 21-cell row-major
    `read_diff_update_prob` sweep against the §10.5
    `default_inter_mode_probs` table.
  * `read_interp_filter_probs( coder, interp_filter_probs )` per
    §6.3.10 (`vp9-spec.txt` lines 2146-2151) —
    `INTERP_FILTER_CONTEXTS = 4` × `SWITCHABLE_FILTERS - 1 = 2`
    = 8-cell row-major `read_diff_update_prob` sweep against the
    §10.5 `default_interp_filter_probs` table. The §6.3.10 listing
    swaps the loop-index names (outer `j`, inner `i`) — visit order
    matches the `[INTERP_FILTER_CONTEXTS][SWITCHABLE_FILTERS - 1]`
    layout.
  * §3 constants `INTER_MODES = 4`, `INTER_MODE_CONTEXTS = 7`,
    `SWITCHABLE_FILTERS = 3`, `INTERP_FILTER_CONTEXTS = 4`
    transcribed verbatim from `vp9-spec.txt` lines 506-507 / 487 / 495.
  * `DEFAULT_INTER_MODE_PROBS` (`mode_info`) transcribed verbatim
    from §10.5 lines 7758-7766 — row annotations preserved
    (0 = both zero mv … 6 = two intra neighbors).
  * `DEFAULT_INTERP_FILTER_PROBS` (`mode_info`) transcribed verbatim
    from §10.5 lines 7769-7775.
  * `DEFAULT_INTER_MODE_PROBS_TABLE` and `DEFAULT_INTERP_FILTER_PROBS_TABLE`
    re-exports in `compressed.rs` preserve `mode_info` as the single
    source of truth.
  * 16 new unit tests (+ 5 for inter-mode, 5 for interp-filter, plus
    layout / constants / re-export equality) covering: §3
    constant pinning; verbatim §10.5 default-table transcription;
    zero-buffer `update_prob = 0` pass-through; all-cells visited
    with custom starts; cursor-equivalence proofs that each sweep
    consumes exactly its prescribed cell count of `B(252)` flags
    (21 / 8); explicit row-major walk equivalence against a
    parallel-coder reference; single-source-of-truth checks tying
    the `compressed.rs` re-exports back to `mode_info`.
* **Round 22: §6.3.11 `read_is_inter_probs( )` compressed-header
  sweep.** Unconditional `IS_INTER_CONTEXTS = 4` `diff_update_prob`
  walk over the §10.5 `default_is_inter_prob[ IS_INTER_CONTEXTS ] =
  {9, 102, 187, 225}` initials, populating the running
  `is_inter_prob[ ]` table the round-21 §6.4.13 `read_is_inter( )`
  per-block decoder consumes via the §9.3.2 ctx:
  * `read_is_inter_probs( coder, is_inter_prob )` per §6.3.11
    (`vp9-spec.txt` lines 2154-2167) — four sequential
    `read_diff_update_prob` calls, one `B(252)` `update_prob` flag per
    slot and on 1 a `decode_term_subexp` + `inv_remap_prob` cascade
    updating `is_inter_prob[ ]` in place.
  * `DEFAULT_IS_INTER_PROB_TABLE` re-export of the round-21
    `mode_info::DEFAULT_IS_INTER_PROB` constant — single source of
    truth for the §10.5 default table across both the §6.4.13
    per-block decoder and the §6.3.11 compressed-header sweep.
  * 7 unit tests covering the §10.5 default re-export equality, the
    zero-buffer `update_prob = 0` path passing every cell through
    unchanged, four-context cell-count visiting with a custom prob
    array, equivalence between the sweep and an explicit
    `read_diff_update_prob` × 4 call sequence (probs + cursor parity),
    the §3 `IS_INTER_CONTEXTS = 4` constant pin, an exhaustive
    starting-tuple round-trip on the zero buffer, and a
    cursor-equivalence check against four explicit `B(252)` reads.
* **Round 21: §6.4.13 `read_is_inter( )` + §9.3.2 `is_inter` ctx +
  §10.5 `default_is_inter_prob`.** Adds the per-block inter/intra
  decoder the §6.4.11 `inter_frame_mode_info( )` driver fires after
  `read_skip( )`:
  * `SEG_LVL_REF_FRAME = 2` / `IS_INTER_CONTEXTS = 4` constants from §3.
  * `default_is_inter_prob[IS_INTER_CONTEXTS] = {9, 102, 187, 225}`
    transcribed verbatim from §10.5.
  * `IsInterNeighbours { above: Option<i32>, left: Option<i32> }` —
    the §6.4.11 above/left `RefFrames[ ][ ][ 0 ]` view (`None` = §6.4.11
    "unavailable → force to `INTRA_FRAME`" rule).
  * `is_inter_context( nb )` (§9.3.2) — the four-branch ctx derivation:
    both available + both intra → 3; both available + one intra → 1;
    both available + neither intra → 0; only above / only left →
    `2 * intra_flag`; neither → 0. Returns `0..=3` indexing
    `is_inter_prob[ ]`.
  * `read_is_inter( coder, seg_feature_ref_frame_active,
    segment_ref_frame_data, is_inter_prob, nb )` (§6.4.13) — the
    two-path reader: when `seg_feature_active( SEG_LVL_REF_FRAME )` is
    set, `is_inter = FeatureData[ segment_id ][ SEG_LVL_REF_FRAME ] !=
    INTRA_FRAME` without consuming any coder bits; otherwise the §9.3.3
    `BINARY_TREE` walk under `is_inter_prob[ ctx ]`.
  * 15 unit tests pinning the §10.5 default constants, the §3
    `SEG_LVL_REF_FRAME` value, every branch of `is_inter_context` (both
    unavailable / both-intra / one-intra / neither-intra / only-above
    intra-and-inter / only-left intra-and-inter / `NONE` ref-frame
    sentinel treated as intra), the §6.4.13 seg-feature path for both
    `INTRA_FRAME` and each of `LAST/GOLDEN/ALTREF_FRAME` overrides, the
    zero-coder bit=0 path, the bias-coder bit=1 path with both-intra
    neighbours, the ctx-indexes-into-`is_inter_prob` sweep across all
    four ctxs, and the path-1 short-circuit ignoring both neighbours
    and the coder.
* **Round 20: §6.4.12 `inter_segment_id( )` + §6.4.14 `get_segment_id( )`
  + §7.4 segmentation-prediction context strips.** Lands the inter-frame
  companion to the round-16 `intra_segment_id` primitive — the per-block
  segment-id reader the §6.4.11 `inter_frame_mode_info( )` driver fires
  before `read_skip( )` / `read_is_inter( )` / `read_tx_size( )`:
  * §10.2 `num_8x8_blocks_high_lookup[ BLOCK_SIZES ]` =
    `{1, 1, 1, 1, 2, 1, 2, 4, 2, 4, 8, 4, 8}` transcribed verbatim into
    the `partition` module alongside the existing `_WIDE_LOOKUP`.
  * `PrevSegmentIds<'a>` — borrowed row-major `MiRows × MiCols` view of
    the previous frame's segment-id plane.
  * `get_segment_id( prev, mi_row, mi_col, mi_size )` (§6.4.14) — the
    `bw` / `bh` clamp via `Min( MiCols - MiCol, bw )` /
    `Min( MiRows - MiRow, bh )` and the `seg = 7; seg = Min( seg,
    PrevSegmentIds[ … ] )` spatial-min sweep.
  * `SegPredContextState { above[MiCols], left[MiRows] }` — the §7.4.1
    / §7.4.2 strip storage with `new( )` zero-init, `clear_left( )`
    per-superblock-row reset, and `above( )` / `left( )` ctx accessors.
  * `read_seg_id_predicted( )` — the §9.3.2 `ctx =
    LeftSegPredContext[ MiRow ] + AboveSegPredContext[ MiCol ]`
    derivation + §9.3.1 `binary_tree` one-bit decode under
    `segmentation_pred_prob[ ctx ]`.
  * `inter_segment_id( )` (§6.4.12) — the four-path orchestrator:
    `!enabled` → 0; `enabled && !update_map` → `predictedSegmentId`;
    `update_map && !temporal_update` → `read_segment_id`;
    `update_map && temporal_update` → `read_seg_id_predicted` then
    either `predictedSegmentId` or a fresh `read_segment_id`, followed
    by the trailing write-back of `seg_id_predicted` into
    `AboveSegPredContext[ MiCol + i ]` / `LeftSegPredContext[ MiRow +
    i ]` over the `num_8x8_blocks_*_lookup` sub-strips.
  * 12 unit tests: `get_segment_id` (interior 2x2 min, partial-edge
    clamp, all-7 fallback), the §7.4 zero-init contract, `clear_left`
    not touching Above, the §9.3.2 ctx wiring of
    `read_seg_id_predicted`, each of the four §6.4.12 paths,
    `Error::InvalidBitstream` on missing `tree_probs` / `pred_prob`,
    and a partial-edge `BLOCK_32X32` write-back clamp.
  * Provenance: VP9 Bitstream & Decoding Process Specification v0.7
    (`docs/video/vp9/vp9-spec.txt` §6.4.4 lines 2395-2437, §6.4.7 lines
    2480-2494, §6.4.12 lines 2562-2586, §6.4.14 lines 2607-2620, §7.4.1
    lines 3824-3830, §7.4.2 lines 3831-3838, §9.3.2 lines 6313-6314,
    §10.2 line 7117).

* **Round 19: §6.4.3 recursive `decode_partition( )` driver (extending
  the crate-local `partition` module).** Composes the round-18
  `decode_partition_type( )` primitive into the recursive §6.4.3
  partition driver — the per-superblock walker the §6.4.2
  `decode_tile( )` outer loop fires at every `(r, c, BLOCK_64X64)`
  cell:
  * `decode_partition( coder, r, c, bsize, mi_rows, mi_cols,
    ctx_state, probs_kind, leaves )` — walks the §6.4.3 listing
    line-for-line: the `(r >= MiRows || c >= MiCols)` quadrant
    short-circuit, the `num8x8 = num_8x8_blocks_wide_lookup[bsize]`
    / `halfBlock8x8 = num8x8 >> 1` / `hasRows = (r + halfBlock8x8) <
    MiRows` / `hasCols = (c + halfBlock8x8) < MiCols` derivation, the
    `partition` decode via [`decode_partition_type`] (using the
    §9.3.2 `partition_plane_context` ctx + the per-frame probability
    source), the four-way dispatch on the decoded `PARTITION_*` value
    (with the HORZ second-leaf gated by `hasRows`, the VERT
    second-leaf gated by `hasCols`, and SPLIT recursing in spec
    order TL → TR → BL → BR per spec lines 2381-2384), and the
    §6.4.3 tail write-back into the partition-context strips
    (gated by `bsize == BLOCK_8X8 || partition != PARTITION_SPLIT`,
    writing `15 >> b_width_log2_lookup[ subsize ]` into
    `AbovePartitionContext[ c + i ]` and `15 >>
    b_height_log2_lookup[ subsize ]` into `LeftPartitionContext[ r +
    i ]` for `i ∈ 0..num8x8`).
  * `PartitionContextState` — the `AbovePartitionContext[ ]` /
    `LeftPartitionContext[ ]` strips (sized `Sb64Cols * 8` /
    `Sb64Rows * 8` per the §7.4 listing). Exposes
    `new( mi_cols, mi_rows )` with the §7.4 zero-reset, and
    `clear_left( )` for the §6.4.2 per-superblock-row reset
    invoked by the §6.4.2 tile driver.
  * `PartitionProbsKind` — the per-frame probability source enum:
    `Keyframe` indexes [`KF_PARTITION_PROBS`] directly per the
    §9.3.2 `FrameIsIntra == 1` arm; `Inter(&[[u8; 3]; 16])` indexes
    the caller's running `partition_probs` table (typically
    initialised from [`DEFAULT_PARTITION_PROBS`] and conditionally
    updated by the §6.3 `read_partition_probs( )` sweep — still
    pending in a later round).
  * `LeafBlock { r, c, subsize }` log records — emitted in §6.4.3
    traversal order in lieu of the §6.4.4 `decode_block( r, c,
    subsize )` call site (the per-block `mode_info` / `residual`
    decode is downstream of this driver and not yet wired). The
    deferred-leaf log is the validation surface for the recursion
    layout this round.
  * Test-only minimal range encoder (`RangeEncoder` in the
    `partition::tests` module) — a forward-simulation bounded
    brute-force search over `BoolValue ∈ 0..128` plus a DFS over
    per-renorm refill bits. For each `(bool_value, stream)`
    candidate, the §9.2.2 decoder is forward-simulated against the
    target `(bit, p)` sequence; the first candidate that produces
    every target bit wins. The trailing tail is zero-padded so any
    further renorm reads past the strictly-required bits resolve to
    0. The search loop walks the §9.2.2 listing verbatim.
  * 8 new unit tests covering the recursive driver: a `RangeEncoder`
    roundtrip across an arbitrary 8-element `(bit, p)` sequence; a
    roundtrip with extreme probabilities (`p ∈ { 1, 128, 255 }`); a
    single-leaf 64x64 `PARTITION_NONE` hand-built bitstream (one
    leaf at `{ 0, 0, BLOCK_64X64 }` + §6.4.3 tail `15 >> 4 = 0`
    write-back); a four-leaf SPLIT-into-32x32-NONE hand-built
    bitstream (four leaves in TL → TR → BL → BR order at
    `{ (0,0), (0,4), (4,0), (4,4), BLOCK_32X32 }` + §6.4.3 tail
    `15 >> 3 = 1` write-back per child but not the parent SPLIT); a
    mixed HORZ/VERT quadrant hand-built bitstream (8 leaves: TL
    HORZ → 2 at BLOCK_32X16, TR VERT → 2 at BLOCK_16X32, BL HORZ,
    BR VERT, exercising both the HORZ / VERT second-leaf paths and
    the §9.3.2 ctx-derivation lockstep against the successively
    populated strip state); the `(r >= mi_rows || c >= mi_cols)`
    short-circuit invariant (no leaves emitted, strips untouched);
    the `PartitionContextState::clear_left( )` zero-the-left-strip
    invariant (above strip unchanged); and the
    `PartitionProbsKind::Inter` table dispatch matching the
    caller's row across `ctx ∈ 0..16`.

  Out of scope for round 19 (deferred):
  * The §6.3 `read_partition_probs( )` compressed-header sweep
    (`PARTITION_CONTEXTS × (PARTITION_TYPES - 1) = 16 × 3 = 48`
    `diff_update_prob` cells against `DEFAULT_PARTITION_PROBS`) —
    the driver consumes the `Inter` running table, but constructing
    it lands in a later round.
  * The §6.4.4 `decode_block( )` mode-info + residual decode that
    `LeafBlock` stands in for — wiring it into this driver is
    downstream of all the §6.4 mode-info readers landing first.
  * The §6.4.2 `decode_tile( )` outer loop (the `r += 8, c += 8`
    superblock walk + per-row `clear_left_context( )`) — composes
    this driver but is a separate round.
  * The §8.4 `counts_partition` probability-adaption accumulator
    (§9.3.4 bookkeeping) for inter-frame `partition_probs[ ]`
    adaption.

  The round-19 surface stays internal-only (`pub(crate)`); the
  public API still exposes `parse_uncompressed_header`,
  `parse_compressed_header` and their result types exclusively.

* **§6.4.3 `decode_partition_type( )` per-call partition reader (new
  crate-local `partition` module).** The single-call decoder the
  recursive §6.4.3 `decode_partition( r, c, bsize )` driver (later
  round) fires once per `(r, c, bsize)` quadrant inside a tile:
  * §9.3.1 partition trees `PARTITION_TREE[6]`, `COLS_PARTITION_TREE[2]`
    and `ROWS_PARTITION_TREE[2]` transcribed verbatim from
    `docs/video/vp9/vp9-spec.txt`.
  * §3 partition enumeration: `PARTITION_NONE = 0`, `PARTITION_HORZ = 1`,
    `PARTITION_VERT = 2`, `PARTITION_SPLIT = 3`, plus dimensions
    `PARTITION_TYPES = 4` and `PARTITION_CONTEXTS = 16`.
  * §10.2 lookups transcribed verbatim: `B_WIDTH_LOG2_LOOKUP` /
    `B_HEIGHT_LOG2_LOOKUP` (the §6.4.3 tail `15 >>
    b_*_log2_lookup[subsize]` write-back inputs),
    `MI_WIDTH_LOG2_LOOKUP` (the §9.3.2 `bsl` derivation input),
    `NUM_8X8_BLOCKS_WIDE_LOOKUP` (the §6.4.3 `num8x8` input).
  * §10.2 `SUBSIZE_LOOKUP[4][13]` (`PARTITION → child block size`)
    transcribed verbatim, with `BLOCK_INVALID = 14` for the
    horizontal / vertical / split combinations that have no legal
    child at non-square parents.
  * §10.4 `KF_PARTITION_PROBS[16][3]` (keyframe / intra-only fixed
    probabilities) and §10.5 `DEFAULT_PARTITION_PROBS[16][3]` (inter
    frame initial probabilities, prior to the §6.3
    `read_partition_probs( )` sweep) transcribed verbatim. Each
    table has a shape + listing-anchor + §9.2-minimum-prob test.
  * `partition_plane_context( bsize, above_ctx, left_ctx )` — the
    §9.3.2 `ctx = bsl * 4 + left * 2 + above` derivation, with
    `bsl = mi_width_log2_lookup[bsize]`, `boffset = 3 - bsl`, and
    an OR-fold of the `AbovePartitionContext[ ]` /
    `LeftPartitionContext[ ]` strips across `num8x8` cells.
  * `decode_partition_type( coder, has_rows, has_cols, probs )` —
    the §6.4.3 reader proper: dispatches on `(has_rows, has_cols)`
    per the §9.3.1 tree-selection rule (interior → 6-entry tree,
    right-edge → 2-entry `cols_partition_tree`, bottom-edge →
    2-entry `rows_partition_tree`, corner → return
    `PARTITION_SPLIT` without consuming bits) and remaps the §9.3.3
    walker's node index per the §9.3.2 `node2` rule (interior:
    `node2 = node`, right-edge: `node2 = 1`, bottom-edge:
    `node2 = 2`). Returns one of the four `PARTITION_*` constants.
  * 37 new unit tests covering: every §3 constant (4 partition
    values + 2 dimensions); the four §10.2 lookups against the
    spec listings; `SUBSIZE_LOOKUP` `PARTITION_NONE` identity /
    `PARTITION_SPLIT` superblock anchors / `PARTITION_HORZ` +
    `PARTITION_VERT` superblock anchors / `BLOCK_INVALID` at
    non-square parents; all three §9.3.1 trees verbatim;
    `KF_PARTITION_PROBS` + `DEFAULT_PARTITION_PROBS` shape + four
    listing anchors each + §9.2 min-prob sanity;
    `partition_plane_context` zero-strip + above-only + left-only +
    both-bits-set cases across each of the four superblock sizes,
    the OR-fold across the strip, unrelated-bit masking, the
    panic-on-invalid-bsize / mismatched-strip guards, and an
    exhaustive sweep proving the 16 ctx values 0..=15 are all
    reachable; `decode_partition_type` against the zero coder
    (every arm picks its first leaf), the all-ones coder (interior
    walks every right-branch → `PARTITION_SPLIT`), the
    one-then-zero coder (each arm's first-right-then-left walk),
    the corner case (consumes zero bits + leaves bool-coder
    untouched), and an exhaustive arm × buffer × probability
    smoke-test confirming every output stays in `0..=3`.

  The recursive §6.4.3 `decode_partition( )` driver itself (which
  threads `SUBSIZE_LOOKUP[partition][bsize]` into four recursive
  calls when `PARTITION_SPLIT` and writes back the
  `AbovePartitionContext[ ]` / `LeftPartitionContext[ ]` strips with
  `15 >> b_*_log2_lookup[subsize]`) and the §6.3
  `read_partition_probs( )` compressed-header sweep both land in a
  later round; the round-18 surface is internal-only, `pub(crate)`.

* **§6.4.15 `intra_block_mode_info( )` inter-frame intra-block reader
  (extending the crate-local `mode_info` module).** The companion to
  the §6.4.6 keyframe driver, for intra blocks within non-keyframe
  frames:
  * §9.3.2 `SIZE_GROUP_LOOKUP[BLOCK_SIZES]`
    (`{0,0,0,1,1,1,2,2,2,3,3,3,3}`) plus §9.3
    `DEFAULT_Y_MODE_PROBS[BLOCK_SIZE_GROUPS][INTRA_MODES - 1]` (4 × 9)
    and `DEFAULT_UV_MODE_PROBS[INTRA_MODES][INTRA_MODES - 1]` (10 × 9)
    transcribed verbatim from `docs/video/vp9/vp9-spec.txt` — the
    compressed-header `y_mode_probs` / `uv_mode_probs` defaults
    (distinct from the §10.5 keyframe `kf_*_mode_probs`).
  * `intra_mode( coder, y_mode_probs, mi_size )` (§9.3.2 ctx =
    `size_group_lookup[MiSize]`), `sub_intra_mode( coder, y_mode_probs )`
    (ctx = 0), and `uv_mode( coder, uv_mode_probs, y_mode )` (ctx =
    `y_mode`) — §9.3.3 walks over `INTRA_MODE_TREE` with the §9.3
    compressed-header rows.
  * `intra_block_mode_info( )` (§6.4.15) returning
    `Vp9IntraBlockModeInfo { ref_frame_0, ref_frame_1, y_mode,
    sub_modes[4], uv_mode }`. Sets `ref_frame[0] = INTRA_FRAME`,
    `ref_frame[1] = NONE`; the `MiSize >= BLOCK_8X8` arm decodes one
    `intra_mode` replicated across `sub_modes[ ]`, the sub-8x8 arm
    walks the `(idy, idx)` grid decoding one `sub_intra_mode` per cell
    (`y_mode` = last decoded). Reads modes only — `segment_id` / `skip`
    / `tx_size` are decoded by the §6.4.11 driver beforehand.
  * §6.4.5 `mode_info( )` dispatch: a `Vp9ModeInfo` enum
    (`IntraFrame` / `InterFrameIntraBlock`) plus
    `inter_frame_intra_block_mode_info( )` wiring the §6.4.15 path
    alongside the existing §6.4.6 keyframe path.
  * Per-table shape + anchor + §9.2 min-prob tests for
    `SIZE_GROUP_LOOKUP` / `DEFAULT_Y_MODE_PROBS` / `DEFAULT_UV_MODE_PROBS`,
    instrumented-callback ctx-row tests for each reader, hand-traced
    bias-buffer decodes, and a per-block decode scenario (BLOCK_8X8
    bias buffer → `y_mode = D207_PRED`, `uv_mode = D153_PRED`). The
    surface stays crate-internal (`pub(crate)`).
* **Round 17: §6.4.6 `intra_frame_mode_info( )` keyframe driver
  (extending the crate-local `mode_info` module).** Wires the rounds
  15 / 16 primitives into the top-level §6.4.6 per-block mode-info
  reader for keyframe (and intra-only) frames:
  * §9.3.1 `intra_mode_tree[18]` constant
    `{ -DC_PRED, 2, -TM_PRED, 4, -V_PRED, 6, 8, 12, -H_PRED, 10,
    -D135_PRED, -D117_PRED, -D45_PRED, 14, -D63_PRED, 16, -D153_PRED,
    -D207_PRED }` transcribed verbatim — the 18-entry / 10-leaf tree
    shared by `default_intra_mode` / `default_uv_mode` / `intra_mode`
    / `sub_intra_mode` / `uv_mode`.
  * §10.5 `KF_Y_MODE_PROBS[10][10][9]` (a 900-byte 3D table indexed by
    `[abovemode][leftmode][node]` per the §9.3.2 `default_intra_mode`
    listing) transcribed verbatim from the spec listing
    (lines 7463–7599).
  * §10.5 `KF_UV_MODE_PROBS[10][9]` (a 90-byte 2D table indexed by
    `[y_mode][node]` per the §9.3.2 `default_uv_mode` listing)
    transcribed verbatim from the spec listing (lines 7602–7613).
  * `default_intra_mode( coder, abovemode, leftmode )` and
    `default_uv_mode( coder, y_mode )` — §9.3.3 walks over
    `INTRA_MODE_TREE` with the respective `kf_*_mode_probs` row.
  * `intra_frame_mode_info()` (§6.4.6) — the orchestrator threading
    `intra_segment_id( )` + `read_skip( )` + `read_tx_size( 1 )` +
    `default_intra_mode` + `default_uv_mode` into a
    `Vp9IntraMiBlock { segment_id, skip, tx_size, ref_frame_0,
    ref_frame_1, is_inter, y_mode, sub_modes[4], uv_mode }`. The
    §6.4.6 `ref_frame[0] = INTRA_FRAME = 0` / `ref_frame[1] = NONE =
    -1` / `is_inter = false` triple is hardwired per the spec
    listing. Handles both the `MiSize >= BLOCK_8X8` single-mode
    partition (one `default_intra_mode` decode replicated into all
    four `sub_modes[ ]` cells) and the `MiSize < BLOCK_8X8` sub-mode
    walk (the §6.4.6 `(idy, idx)` grid stepped by
    `num_4x4_blocks_high_lookup[MiSize]` /
    `num_4x4_blocks_wide_lookup[MiSize]` — 4 reads for BLOCK_4X4, 2
    for BLOCK_4X8 / BLOCK_8X4 — with each cell receiving its own
    decoded mode replicated across the (num4x4h × num4x4w)
    `sub_modes[ ]` sub-grid; `y_mode` set to the *last* decoded
    `default_intra_mode`).
  * `IntraFrameNeighbours` bundle — per-MI-block neighbour state a
    tile driver builds from its frame-wide `SubModes[ ][ ][ ]` array
    (positions {2, 3} of the above neighbour, positions {1, 3} of the
    left neighbour, plus the §7.4.4 `AvailU` / `AvailL` flags). The
    §9.3.2 listing reads only those four cells; `DC_PRED` is
    substituted when the corresponding `avail_*` flag is false.
* **Round 16: §6.4.7 `intra_segment_id( )` + §9.3.1 `segment_tree[14]`
  (extending the crate-local `mode_info` module).** Lands the next
  slice of the §6.4.6 `intra_frame_mode_info()` orchestrator's
  primitives that round 15 left deferred:
  * The §9.3.1 `segment_tree[14]` constant
    `{ 2, 4, 6, 8, 10, 12, 0, -1, -2, -3, -4, -5, -6, -7 }` transcribed
    verbatim — the 7-leaf binary tree used by every `segment_id`
    decode site (intra §6.4.7 + inter §6.4.12).
  * `read_segment_id( coder, tree_probs )` — the §9.3.3 walk over the
    new `SEGMENT_TREE` with per-node probability
    `segmentation_tree_probs[node]` per the §9.3.2 listing's
    `segment_id` entry. Returns the decoded segment id in `0..=7`.
  * `intra_segment_id( coder, segmentation_enabled,
    segmentation_update_map, tree_probs )` (§6.4.7) — the
    `segmentation_enabled && segmentation_update_map` gate around
    `read_segment_id`, falling through to `segment_id = 0` otherwise
    (the intra-only path has no `segmentation_temporal_update` /
    `seg_id_predicted` machinery — that's inter-only and lands when
    the §6.4.12 syntax does).
* **Round 15: §6.4.8 `read_skip` + §6.4.10 `read_tx_size` + §9.3.3
  `tree_decode` (crate-local `mode_info` module).** The first slice of
  the §6.4 per-block mode-info decode that the round-14
  `residual_intra` driver currently consumes via a caller-supplied
  bundle — unblocks the per-block `BoolCoder`-driven mode-info wiring
  the §6.4.6 `intra_frame_mode_info()` orchestrator will need.
  * `tree_decode( coder, tree, prob )` — the §9.3.3 generic tree
    decoding loop `do { n = T[n + read_bool(P(n >> 1))] } while (n >
    0)` that every tree-coded syntax element routes through. The
    probability callback is a `FnMut(usize) -> u8` so call-sites can
    splice the right §9.3.2 row in without the helper needing to know
    which syntax element it's decoding.
  * §9.3.1 trees `tx_size_8_tree[2]` / `tx_size_16_tree[4]` /
    `tx_size_32_tree[6]` and `binary_tree[2]` transcribed verbatim
    from the spec listing.
  * `skip_context( nb )` (§9.3.2) — the `Skips[MiRow-1][MiCol] +
    Skips[MiRow][MiCol-1]` ctx derivation with `AvailU` / `AvailL`
    gating; `tx_size_context( nb, max_tx_size )` (§9.3.2) — the
    `(above + left) > maxTxSize` ctx derivation that consults
    neighbour `TxSizes[ ]` only on unskipped MI blocks (and mirrors
    the side when a neighbour is unavailable).
  * `read_skip( coder, seg_feature_skip_active, skip_prob, nb )`
    (§6.4.8) — the §6.4.9 `seg_feature_active(SEG_LVL_SKIP)`
    early-return rule plus the §9.3.2 binary tree decode under
    `skip_prob[skip_context(nb)]`.
  * `read_tx_size( coder, allow_select, tx_mode, mi_size, tx_probs,
    nb )` (§6.4.10) — the `allow_select && tx_mode == TX_MODE_SELECT
    && MiSize >= BLOCK_8X8` path picking the §9.3.1 tree by
    `max_txsize_lookup[MiSize]` and the §9.3.2 ctx, falling through
    to `Min(maxTxSize, tx_mode_to_biggest_tx_size[tx_mode])` per the
    spec's `else` branch.
  * `NeighbourSkips` / `NeighbourTxSizes` — the per-MI-block
    neighbour-state bundles a tile driver builds from its
    frame-wide `Skips[ ][ ]` / `TxSizes[ ][ ]` arrays.
  * 22 unit tests: the §9.3.1 tree-listing anchors (verbatim
    transcription check), the §9.3.3 walker with a zero-coder
    (every read yields bit=0, every tree picks its first leaf),
    a "bias buffer" prefix `[0x7F, 0x00, ...]` whose post-marker
    coder state lets the first `read_bool(255)` flip to 1 (one
    right-branch step in any tree), the node-index argument-order
    invariant for `tree_decode`'s prob callback, exhaustive
    `skip_context` cases (no neighbours, single-side, both-sides
    mixed and matching), `tx_size_context` against the §9.3.2
    listing across max_tx_size 0..=3 plus the skipped-neighbour
    fallback and the `!AvailL` / `!AvailU` mirroring, the §6.4.8
    `seg_feature_active` early-return path, `read_skip` against
    both the zero coder (always false) and the bias coder + p=255
    (true), the §6.4.10 `else`-branch `Min(max, biggest)` fallback
    for `allow_select == false` / non-SELECT `tx_mode` / sub-8x8
    `MiSize`, the §9.3.1 tree-dispatch for `BLOCK_8X8` /
    `BLOCK_16X16` / `BLOCK_32X32`, and the row-by-ctx selection
    correctness via the spec-derived ctx derivation lockstep
    (both ctx=0 and ctx=1 evaluated against `tx_size_context`).
    Every test consumes the §9.2 BoolCoder through valid byte
    buffers (marker-bit conformant) hand-derived by walking the
    §9.2 listing.
  * Out of scope this round: the §6.4.6 `intra_frame_mode_info()`
    orchestrator (which wires `read_skip` + `read_tx_size` + the
    deferred §6.4.7 `intra_segment_id` + §6.4.15
    `intra_block_mode_info` into a single `Vp9IntraMiBlock`); the
    `Skips[ ][ ]` / `TxSizes[ ][ ]` frame-wide array write-back
    (left to the §6.4.6 driver); inter-frame mode info (§6.4.11+,
    needs reference-buffer state); and the §8.4 `counts_skip` /
    `counts_tx_size` probability-adaption accumulators. The
    round-15 surface is internal-only; the public API still
    exposes `parse_uncompressed_header`, `parse_compressed_header`
    and their result types exclusively.
* **Round 14: §6.4.21 `residual( )` intra driver (crate-local `residual`
  module).** The §6.4.21 outer loop for the intra path — the per-plane,
  per-4x4-sub-block walk that owns the `AboveNonzeroContext` /
  `LeftNonzeroContext` write-back across a whole MI block, drives the
  round-13 §6.4.24 `tokens( )` per-block decode, and feeds the round-11
  §8.6.2 `reconstruct_block` with real per-block `Tokens` arrays,
  availability flags and plane/quantizer state.
  * §10.2 `num_4x4_blocks_wide_lookup[13]` / `num_4x4_blocks_high_lookup[13]`,
    §6.4.10 `max_txsize_lookup[13]`, and §6.4.23 `ss_size_lookup[13][2][2]`
    tables transcribed verbatim, alongside the `BLOCK_4X4 .. BLOCK_64X64`
    / `BLOCK_INVALID` `subsize` constants from §3.
  * `get_plane_block_size( subsize, plane, subsampling_x, subsampling_y )`
    (§6.4.23) and `get_uv_tx_size( tx_size, mi_size, subsampling_x,
    subsampling_y )` (§6.4.22) — the chroma-plane block-size /
    transform-size derivations that key the per-plane loop.
  * `ResidualBlockCtx` — the per-MI-block / per-frame bundle (`MiCol` /
    `MiRow` / `MiCols` / `MiRows`, `MiSize`, `tx_size`, `subsampling_x` /
    `y`, `skip`, `Lossless`, `BitDepth`, the per-block `PredMode` for
    luma and chroma, and the per-plane DC/AC quantizers from round 8);
    plus `AvailFlags` for §7.4.4 `AvailL` / `AvailU` and a
    `PlaneBuffers` wrapper for the three `CurrFrame[ plane ]` planes.
  * `residual_intra( planes, nz, block, avail, token_source )` — the
    §6.4.21 driver proper: per plane, computes `bsize = MiSize <
    BLOCK_8X8 ? BLOCK_8X8 : MiSize`, the per-plane `planeSz` +
    `num4x4w` / `num4x4h` dimensions and chroma `txSz`, then walks the
    `(y, x)` 4x4 grid stepping by `step = 1 << txSz`. For each in-bounds
    transform block (`startX < maxx && startY < maxy`) it calls the
    round-10 `predict_intra` with the resolved `have_left` /
    `have_above` / `not_on_right` flags, pulls `Tokens[ ]` from a
    per-block `TokenSource` callback (when `!skip`), derives the §6.4.25
    `TxType` (chroma / `TX_32X32` / lossless force `DCT_DCT`; luma intra
    uses round-11 `tx_type_for_intra`), runs the round-11
    `reconstruct_block`, and writes
    `AboveNonzeroContext[ plane ][ x4 + i ] = LeftNonzeroContext[
    plane ][ y4 + i ] = nonzero` for `i ∈ 0..step` per the §6.4.21
    trailing write-back.
  * 12 unit tests: the §10.2 / §6.4.10 / §6.4.23 table-anchor checks
    (luma-identity invariant + 4:2:0 / asymmetric subsampling
    anchors), the §6.4.22 `get_uv_tx_size` chroma cap and the sub-8x8
    short-circuit, the `skip = true` path leaving every strip cell at 0
    (no token decode), the full `skip = false` walk firing
    `token_source` exactly 16 luma + 4 U + 4 V times for a `BLOCK_16X16`
    MI block at `tx_size = TX_4X4` (each call recorded with `(plane,
    block_idx, tx_sz)`), the `nonzero = true` strip write-back over
    `step = 1 << tx_sz` 4-sample units for `tx_size = TX_8X8`, the
    out-of-bounds block skip with intact zero context, a DC-only luma
    block at MI (1,1) lockstep against an independent `predict_intra` +
    `reconstruct_block` probe, and the `bsize = max(MiSize, BLOCK_8X8)`
    widening for a `BLOCK_4X4` MI block. Every formula and table is
    transcribed directly from the §6.4.21 / §6.4.22 / §6.4.23 / §10.2
    listings.
  * The `is_inter` branch of §6.4.21 (which calls `predict_inter( )`
    before the per-block loop) is deferred until the §8.5.2 inter
    prediction process and reference-buffer state land; the per-block
    mode-info decode (`y_mode` / `sub_modes` / `tx_size` / `skip` /
    `segment_id` from §6.4) that the residual loop reads is also a
    later-round increment. The round-14 surface is internal-only.

* **Round 13: §6.4.24 `tokens( )` per-block coefficient driver
  (crate-local `tokens` module).** Walks the round-12 §6.4.25 scan
  order and feeds each scan position through the round-7
  `read_coef_token` pipeline, recovering one transform block's
  quantised coefficients into a `Tokens[ ]` array.
  * The §10 band tables — `coefband_4x4[ 16 ]` transcribed verbatim and
    `coefband_8x8plus[ 1024 ]` built from the verbatim 21-entry prefix
    plus the all-`5` tail — selected by `coef_band( c, txSz )` per the
    §6.4.24 `(txSz == TX_4X4) ? coefband_4x4 : coefband_8x8plus` rule.
  * `token_cache_neighbours( c, pos, txSz, txType )` — the §9.3.2
    neighbour pair (`nb[ 0 ]` / `nb[ 1 ]`): `(0, 0)` for the DC
    coefficient, and for `c > 0` the above (`(i-1)*n + j`) / left
    (`i*n + j-1`) raster cells with the `DCT_ADST` (double above) /
    `ADST_DCT` (double left) / first-row / first-column variants
    (`n = 4 << txSz`).
  * `build_token_probs( cell )` — the §9.3.2 10-node probability array:
    node 0 → `cell[1]`, node 1 → `cell[2]`, node `2..=9` →
    `pareto( node, cell[2] )`.
  * `NonzeroContext` (the per-plane `AboveNonzeroContext` /
    `LeftNonzeroContext` 4-sample strips) and `TokenBlockCtx` (the
    per-block / per-frame state `tokens( )` reads — `plane`,
    `is_inter`, resolved `TxType`, `BitDepth`, `x4` / `y4`, `maxX` /
    `maxY`).
  * `tokens( coder, block, txSz, scan, coef_probs, nz, token_cache,
    tokens )` — the §6.4.24 driver: `segEob = 16 << (txSz << 1)`, the
    `checkEob` gating, the §9.3.2 per-coefficient `ctx` (DC from the
    non-zero strips, `c > 0` from `TokenCache`), the
    `coef_probs[txSz][plane>0][is_inter][band][ctx]` cell pick, the
    `more_coefs` / `token` / `read_coef` / `sign_bit` decode, the
    `TokenCache[ pos ] = energy_class[ token ]` write, the
    `ZERO_TOKEN`-clears-`checkEob` rule, the trailing `Tokens[ scan[ i
    ] ] = 0` zero-fill, and the `nonzero = c > 0` return.
  * 15 unit tests: the band tables vs the §10 listing + the
    `coef_band` dispatch, the §9.3.2 neighbour derivation (DC, interior
    DCT_DCT / ADST variants, first row / column, 8x8 width scaling),
    `build_token_probs` node mapping, and the `tokens( )` driver
    (zero-buffer immediate EOB + all-zero fill, the
    `ZERO_TOKEN`-clears-`checkEob` block fill, a lockstep match against
    an independent `read_coef_token` walk, the DC non-zero-context cell
    routing, the trailing zero-fill, and the `NonzeroContext::new`
    all-zero invariant). The band tables and every formula are
    transcribed directly from the §6.4.24 / §9.3.2 / §10 listings. The §6.4.21 residual loop that
    threads `NonzeroContext` across the frame and feeds the round-11
    reconstruct driver lands in a later round; the round-13 surface is
    internal-only.

* **Round 12: §6.4.25 `get_scan` scan-order selection (crate-local
  `scan` module).** The first step of the §6.4.24 `tokens( )` per-block
  driver — it picks the scan order (the sequence of raster positions
  `pos = scan[ c ]` the coefficient loop visits) for a transform block.
  * The §10.1 scan tables transcribed verbatim: `default_scan_4x4` /
    `col_scan_4x4` / `row_scan_4x4` (16 entries), the 8x8 trio (64),
    the 16x16 trio (256), and `default_scan_32x32` (1024). Element
    type `u16` so the 32x32 table's `0..=1023` range fits.
  * `get_scan( plane, txSz, txType )` — the §6.4.25 selection:
    `ADST_DCT` → `row_scan`, `DCT_ADST` → `col_scan`, else
    (`DCT_DCT` / `ADST_ADST`) → `default`, applying the §6.4.25 first
    half (a chroma plane or a `TX_32X32` block forces `TxType =
    DCT_DCT`). The mode-info-dependent `mode2txfm_map[ y_mode ]`
    `TxType` derivation already lives in
    [`reconstruct::tx_type_for_intra`]; the per-block mode-info state
    is owned by the (deferred) §6.4.21 residual driver.
  * `TX_4X4` / `TX_8X8` / `TX_16X16` / `TX_32X32` `txSz` index
    constants (§3).
  * 10 unit tests: every table's spec length, the permutation
    invariant (each raster position appears exactly once), the
    DC-first invariant, §10.1 listing anchors (first four + last
    entry of each table), the §6.4.25 `txType` → table selection for
    4x4 / 8x8 / 16x16, the `TX_32X32`-always-default and
    chroma-forces-default first-half overrides, and the
    `16 << (txSz << 1)` `segEob` length match. The tables are
    transcribed directly from the §10.1 listing.

* **Round 11: §8.6.2 reconstruct driver (crate-local `reconstruct`
  module).** Ties the rounds 7-10 pieces together at the conceptual
  `reconstruct( plane, startX, startY, txSz )` call site of the
  §6.4.21 residual syntax.
  * `tx_type_for_intra( mode )` — the §6.4.25 `mode2txfm_map[ y_mode ]`
    lookup selecting the `TxType` (`DCT_DCT` / `ADST_DCT` / `DCT_ADST`
    / `ADST_ADST`) for a luma intra block from its `PredMode`. The
    10-entry intra prefix of `mode2txfm_map` (§10.5) is transcribed
    verbatim.
  * `reconstruct_block( plane_buf, x, y, tx_sz, tokens, dc_quant,
    ac_quant, tx_type, lossless, bit_depth )` (§8.6.2) — sets
    `dqDenom = 2` for `txSz == TX_32X32` else `1`, `n = 2 + txSz`,
    `n0 = 1 << n`; step 1 `Dequant[i][j] = (Tokens[i*n0+j] *
    get_ac_quant) / dqDenom`; step 2 the `Dequant[0][0] = (Tokens[0] *
    get_dc_quant) / dqDenom` DC override; step 3 the §8.7.2
    `inverse_transform_2d`; step 4 `CurrFrame[plane][y+i][x+j] =
    Clip1( CurrFrame[...] + Dequant[i][j] )`. Integer division
    truncates toward zero per §4.1 (Rust's `i64 /` matches).
  * `reconstruct_intra_block( … )` — the end-to-end one-block driver:
    predicts via §8.5.1 `predict_intra` (round 10), derives the
    `TxType` with the §6.4.25 `TX_32X32` / lossless `DCT_DCT`
    overrides, then runs `reconstruct_block`. The shape the deferred
    §6.4.21 residual loop will call once it threads real availability
    and quantizer state.
  * Crate-local `clip1` (§3) helper operating in `i64` so the
    high-precision residual sum does not overflow before clamping.
  * 10 unit tests: the `mode2txfm_map` intra prefix vs the §10.5
    listing, `clip1` range clamping, an all-zero `Tokens` block
    leaving the prediction unchanged, a DC-only `Tokens` block adding
    a flat residual to a flat prediction, step-4 clipping at both the
    bit-depth max and zero, the `TX_32X32` `dqDenom = 2` halving
    through the real driver, `reconstruct_intra_block` with DC_PRED +
    zero tokens equalling the pure DC prediction, DC_PRED + a known DC
    residual reconstructing the expected pixels, and the lossless WHT
    path driven via `reconstruct_intra_block`.

* **Round 10: §8.5.1 intra prediction process (crate-local `intra`
  module).**
  * `PredMode` — the 10 §7.4.5 intra prediction modes with
    discriminants matching the spec numbering exactly (`DC_PRED` = 0,
    `V_PRED` = 1, `H_PRED` = 2, `D45_PRED` = 3, `D135_PRED` = 4,
    `D117_PRED` = 5, `D153_PRED` = 6, `D207_PRED` = 7, `D63_PRED` = 8,
    `TM_PRED` = 9) plus `from_raw` for the (deferred) mode-info decode.
  * `Plane` — a minimal row-major `i32` plane buffer standing in for
    `CurrFrame[ plane ]`; the §8.5.1 process reads neighbour samples
    from it and writes the prediction back.
  * `predict_intra( plane, x, y, have_left, have_above, not_on_right,
    tx_sz, mode, max_x, max_y, bit_depth )` (§8.5.1) — builds
    `aboveRow[-1 .. 2*size-1]` and `leftCol[0 .. size-1]` per the
    `haveAbove` / `haveLeft` / `notOnRight` availability rules (the
    upper-right extension fires only for `txSz == 0`; missing
    neighbours fill `(1<<(BitDepth-1)) ± 1`), forms the `pred` block
    for the selected mode — `V`/`H` copies; the four `DC` neighbour
    cases (`avg` = `(sum + size) >> (log2Size+1)`, `leftAvg` /
    `aboveAvg` = `(sum + (1<<(log2Size-1))) >> log2Size`, and the
    `1<<(BitDepth-1)` no-neighbour fill); the `D45`/`D63`/`D117`/
    `D135`/`D153`/`D207` directional `Round2` recurrences (including
    the reverse-order `D207` step 5); and `TM` with `Clip1` — then
    stores it back. Neighbour reads clamp with `Min(maxX, .)` /
    `Min(maxY, .)`. Crate-local `round2` (§3) and `clip1` (§3)
    helpers.
  * 16 unit tests: `PredMode::from_raw` round-tripping the §7.4.5
    numbering (and the discriminant values), `round2` / `clip1`
    against their §3 definitions, `V_PRED` / `H_PRED` copy semantics,
    all four `DC_PRED` neighbour cases, the `TM_PRED` formula plus
    out-of-range clipping, a constant-neighbour invariant collapsing
    every directional mode to the neighbour constant across all four
    transform sizes, the `D207_PRED` bottom-row / step-2 formulas, the
    `notOnRight`-gated upper-right `aboveRow` extension (via
    `D45_PRED`, enabled only for `txSz == 0`), and the `Min(maxX, .)`
    plane-edge clamping of neighbour reads. Every formula is
    transcribed directly from the spec §8.5.1 listing. The §8.6.2 reconstruct driver that
    supplies the real availability flags and adds the round-9
    inverse-transformed residual to this prediction remains deferred
    to a future round; the round-10 surface is internal-only.

* **Round 9: §8.7 inverse transform process (crate-local `idct`
  module).**
  * The §8.7.1.1 butterfly primitives — `B` (butterfly rotation,
    including the `16 + 32*k` two-multiply fast path), `H` (Hadamard
    rotation), `SB` (butterfly into the high-precision `S` array) and
    `SH` (Hadamard rotation + `Round2(·,14)` out of `S`) — plus the
    `cos64` / `sin64` angle functions backed by the verbatim 33-entry
    `COS64_LOOKUP` quarter-wave table and the `brev` bit-reversal
    helper. All fixed-point intermediates are evaluated in `i64`
    (the spec notes `S` needs `24 + BitDepth` bits).
  * `inverse_dct( t, n )` (§8.7.1.2 + §8.7.1.3) — the inverse-DCT
    array permutation (`T[i] = copyT[ brev(n, i) ]`) followed by the
    recursive inverse DCT process for `2 <= n <= 5`.
  * `inverse_adst( t, n )` (§8.7.1.4 .. §8.7.1.9) — the ADST
    input/output permutations and the ADST4 / ADST8 / ADST16
    processes (the `SINPI_1_9 .. SINPI_4_9` constants transcribed
    verbatim) dispatched by `n` for `2 <= n <= 4`.
  * `inverse_wht( t, shift )` (§8.7.1.10) — the in-place inverse
    Walsh-Hadamard transform with the `shift` pre-scaling argument.
  * `inverse_transform_2d( dequant, n, tx_type, lossless )` (§8.7.2)
    — the 2D driver: per-`TxType` row transform then column
    transform over a `(1<<n)` by `(1<<n)` `Dequant` block, the
    lossless WHT path (`shift = 2` rows / `0` columns), and the
    `Round2( T[i], Min(6, n+2) )` column rounding. `TxType` constants
    `DCT_DCT` / `ADST_DCT` / `DCT_ADST` / `ADST_ADST` are defined per
    §3.
  * 20 unit tests: `cos64` quarter-wave symmetry + periodicity,
    `sin64` shift, `brev`, the Hadamard sum/difference + flip
    semantics, the `16+32*k` butterfly fast-path equivalence, the
    DC-only "flat output" property of the 4/8/16/32-point inverse
    DCT, zero-in/zero-out for all ADST sizes, the §8.7.1.5 output
    permutation indices, and the 2D driver's zero-in/zero-out (all
    four `TxType`s, `n = 2..5`) + DC-only flat-block property (lossy
    and lossless paths). The `cos64_lookup` table and `SINPI_*_9`
    constants are transcribed directly from the spec §8.7.1 listings. The §8.6.2
    reconstruct driver that builds the `Dequant` input (round-7 token
    magnitudes scaled by the round-8 quantizers) and adds the
    residual to the prediction remains deferred to a future round.

* **Round 8: §8.6.1 dequantization functions (crate-local `dequant`
  module).**
  * `dc_q( bit_depth, b )` / `ac_q( bit_depth, b )` (§8.6.1) — index
    the `dc_qlookup[3][256]` / `ac_qlookup[3][256]` tables by the
    `(BitDepth - 8) >> 1` row and the `Clip3(0, 255, b)` column.
    Both 256-entry tables are transcribed verbatim from the §8.6.1
    listing into `DC_QLOOKUP` / `AC_QLOOKUP`.
  * `seg_feature_active( seg, segment_id, feature )` (§6.4.9) —
    `segmentation_enabled && FeatureEnabled[ segment_id ][ feature ]`.
  * `get_qindex( seg, quant, segment_id )` (§8.6.1) — applies the
    `SEG_LVL_ALT_Q` segment feature (absolute update replaces
    `base_q_idx`, delta update offsets it, then `Clip3(0, 255, .)`)
    or returns `base_q_idx` when the feature is inactive.
  * `get_dc_quant( plane, .. )` / `get_ac_quant( plane, .. )`
    (§8.6.1) — combine `get_qindex()` with the plane-specific header
    delta (`delta_q_y_dc` luma DC, `delta_q_uv_dc` chroma DC,
    `delta_q_uv_ac` chroma AC; luma AC has none) and dispatch to
    `dc_q` / `ac_q`.
  * `SEG_LVL_ALT_Q` constant (§3 table of constants) plus a private
    `clip3` helper (§5.1).
  * 13 unit tests including table-shape / spec-anchor checks, the
    `clip3` clamp branches, `dc_q` / `ac_q` index clipping + row
    selection, all three `get_qindex` paths, plane-delta selection
    for `get_dc_quant` / `get_ac_quant`, and the high-bit-depth
    divergence of the same qindex across the three table rows. The
    §8.6.2 reconstruct driver that consumes these helpers (scaling
    the round-7 `Tokens` array, with the `dqDenom = 2` halving for
    `TX_32X32`) remains deferred to a future round. No external
    library / source was consulted; the lookup tables are
    transcribed directly from the spec §8.6.1 listing.

* **Round 7: §6.4.24 / §6.4.26 coefficient-token decoder (crate-local
  `tokens` module).**
  * `read_token( coder, &probs )` (§6.4.24, §9.3.3) — walks the
    20-entry `token_tree` returning one of `ZERO_TOKEN` ..
    `DCT_VAL_CATEGORY6`. `probs[0..=9]` are the 10 internal-node
    `read_bool` probabilities pre-derived from
    `coef_probs[...][1]` / `coef_probs[...][2]` via `pareto`.
  * `pareto( node, prob )` (§9.3.2) — short-circuits to `prob` for
    `node < 2`; otherwise looks up
    `PARETO_TABLE[ (prob - 1) / 2 ][ node - 2 ]` (odd `prob`) or
    interpolates two adjacent rows (even `prob`). The full 128×8
    pareto table is transcribed verbatim from spec §10.3.
  * `read_more_coefs( coder, prob )` (§6.4.24, §9.3.2) — single
    `B(p)` returning `true` (continue scan) / `false` (EOB).
  * `read_coef( coder, token, bit_depth )` (§6.4.26) — extra-bits
    decoder. For tokens `ONE`..`FOUR` no bits are read and the base
    coef is returned. For `DCT_VAL_CATEGORY1..6` the `numExtra`
    `B(p)` reads against `cat_probs[cat]` build the magnitude in
    `EXTRA_BITS[token][2] + (extra_bits_value)`. For `CAT6` at
    `bit_depth ∈ {10, 12}` an additional `BitDepth - 8` `B(255)`
    `high_bit` reads prepend MSBs at shift `5 + BitDepth - e`.
  * `read_coef_token( coder, check_eob, more_coefs_prob,
    &token_probs, bit_depth )` — driver returning
    `CoefStep::Eob | Coef { token, value }`. Folds `read_more_coefs`
    + `read_token` + `read_coef` + `L(1) sign_bit` together. The
    `checkEob` flag itself stays in the caller's residual loop per
    §6.4.24.
  * `EXTRA_BITS[11][3]`, `CAT_PROBS[7][14]`, `ENERGY_CLASS[12]`,
    `TOKEN_TREE[20]`, `PARETO_TABLE[128][8]` constants — all
    transcribed verbatim from spec §6.4.26 / §10.2 / §10.3 / §9.3.
  * 28 unit tests including hand-traced golden buffers for
    `ONE_TOKEN` (`0x40 0x00 …`) and `TWO_TOKEN` (`0x60 0x00 …`)
    derived by stepping the §9.2 decoder by hand. No external
    library / source was consulted; the §6.4.21 `residual( )` driver
    that will consume these helpers remains deferred to a future
    round.

* **Round 6: §6.3.7 `read_coef_probs` 6D coefficient-probability
  sweep wired into `parse_compressed_header`.**
  * `read_coef_probs(&mut coder, tx_mode, &mut coef_probs)` (§6.3.7)
    — walks `txSz ∈ [TX_4X4, maxTxSize]` with `maxTxSize =
    tx_mode_to_biggest_tx_size[ tx_mode ]` (§10.5). Per active
    tx-size, reads an outer `L(1) update_probs` flag and, on 1,
    drives a nested `(i, j, k, l, m)` sweep over `BLOCK_TYPES=2 ×
    REF_TYPES=2 × COEF_BANDS=6 × maxL(k) × UNCONSTRAINED_NODES=3`
    cells, with `maxL = (k == 0) ? 3 : 6` per §6.3.7 (band 0 has
    only 3 valid previous-coef contexts). Each cell becomes
    `read_diff_update_prob( coder, cell )`. Fully-active inner
    walk is 396 cells per tx-size, 1584 for a TX_MODE_SELECT
    frame.
  * `tx_mode_to_biggest_tx_size( tx_mode )` const-fn (§10.5) —
    maps `TxMode` to the spec's biggest-tx-size index, with the
    `ALLOW_32X32` and `TX_MODE_SELECT` rows both mapping to
    `TX_32X32 = 3`.
  * `coef_probs::DEFAULT_COEF_PROBS: CoefProbs` constant in the
    new `src/coef_probs.rs` module — the §10
    `default_coef_probs[TX_SIZES=4][BLOCK_TYPES=2][REF_TYPES=2][
    COEF_BANDS=6][PREV_COEF_CONTEXTS=6][UNCONSTRAINED_NODES=3]`
    table (1728 u8 entries) transcribed verbatim from the spec
    listing. Band-0 trailing `{0, 0, 0} // unused` rows preserved
    as in-table sentinels matching the `maxL = 3` clamp.
  * `CoefProbs` public type alias re-exported from the crate root
    (`pub use coef_probs::CoefProbs;`), naming the 6D array shape.
  * `Vp9CompressedHeader` extended with `pub coef_probs:
    CoefProbs`. `parse_compressed_header` now runs the sweeps in
    spec order: `read_tx_mode` → optional `read_tx_mode_probs` →
    `read_coef_probs` → `read_skip_prob`.
  * `Vp9CompressedHeader` no longer derives `Copy` — the 1728-byte
    `coef_probs` field makes silent copies costly. `Clone` is
    retained.
  * 7 new unit tests: `tx_mode_to_biggest_tx_size` against the
    §10.5 listing, `read_coef_probs` zero-buffer passthrough for
    `ONLY_4X4` (1-slab) and `TX_MODE_SELECT` (4-slab) modes, an
    across-mode sweep, `DEFAULT_COEF_PROBS` shape +
    spec-listing anchors (TX_4X4 / block-type 0 / Intra / band 0
    / ctx 0 = {195, 29, 183}; TX_32X32 / block-type 1 / Inter /
    band 5 / ctx 5 = {1, 16, 6}), the band-0 unused-row sentinel
    invariant, and the inner-sweep `2 × 2 × (3 + 5 × 6) × 3 =
    396` cell-count check.
  * 2 new end-to-end integration tests
    (`tests/compressed_header.rs`):
    `end_to_end_tx_mode_select_runs_coef_probs_sweep` driving the
    full TX_MODE_SELECT → tx_mode_probs → coef_probs → skip_prob
    chain through `parse_uncompressed_header` +
    `parse_compressed_header` and verifying two §10 default
    anchors survive verbatim; and
    `end_to_end_only_4x4_visits_only_first_tx_size_coef_slab`
    confirming the outer-loop tx-size clipping for the ONLY_4X4
    path.

* **Round 5: §6.3.2 `tx_mode_probs` + §6.3.8 `read_skip_prob` sweeps
  wired into `parse_compressed_header`.**
  * `read_tx_mode_probs(&mut coder, &mut tx_probs)` (§6.3.2) —
    three nested sweeps (`TX_SIZE_CONTEXTS * (1+2+3) = 12` cells)
    walking `tx_probs_8x8`, `tx_probs_16x16`, `tx_probs_32x32` via
    `read_diff_update_prob`. Gated on
    `tx_mode == TX_MODE_SELECT` per the §6.3 compressed-header
    syntax dispatch.
  * `read_skip_prob(&mut coder, &mut skip_prob)` (§6.3.8) —
    unconditional `SKIP_CONTEXTS = 3` sweep via
    `read_diff_update_prob`.
  * `DEFAULT_TX_PROBS: [[[u8; 3]; 2]; 4]` and
    `DEFAULT_SKIP_PROB: [u8; 3] = [192, 128, 64]` constants
    transcribed verbatim from the §10 default-tables listing.
  * `Vp9CompressedHeader` extended with `tx_probs` and `skip_prob`
    fields exposing the post-sweep tables to callers.
    `parse_compressed_header` runs the sweeps in spec order:
    `read_tx_mode` → optional `read_tx_mode_probs` →
    `read_skip_prob`.
  * 10 new unit tests covering `DEFAULT_TX_PROBS` shape + value
    spot-check, `DEFAULT_SKIP_PROB` value, `read_tx_mode_probs`
    zero-buffer passthrough, the 12-cell sweep count, the row-0
    (`TX_4X4`) non-modification invariant, `read_skip_prob`
    zero-buffer passthrough across `SKIP_CONTEXTS = 3`, and three
    `parse_compressed_header` integration scenarios (ONLY_4X4
    non-lossless, lossless, and TX_MODE_SELECT). 2 new end-to-end
    integration tests in `tests/compressed_header.rs` driving the
    full uncompressed-header splice plus the new sweeps.
  * `read_diff_update_prob`, `decode_term_subexp`,
    `inv_remap_prob`, `inv_recenter_nonneg`, and `INV_MAP_TABLE`
    drop their round-4 `#[allow(dead_code)]` markers — they are
    now driven live by the §6.3.2 / §6.3.8 sweeps.

* **Round 4: §6.3.3 `diff_update_prob` chain (`decode_term_subexp` +
  `inv_remap_prob` + `inv_recenter_nonneg` + 255-entry
  `inv_map_table`).**
  * `read_diff_update_prob( coder, base_prob )` (§6.3.3) — reads the
    `B(252)` `update_prob` flag, on 1 pulls a `decode_term_subexp`
    value and remaps the previous probability through
    `inv_remap_prob`. On 0, passes the base probability straight
    through.
  * `decode_term_subexp( )` (§6.3.4) — the 5-leg
    `L(1) → L(4)` / `L(1) → L(4)+16` / `L(1) → L(5)+32` /
    `L(7) → +64` / `L(7), L(1) → (v<<1)-1+bit` cascade producing a
    value in `0..=254`.
  * `inv_remap_prob( delta_prob, prob )` (§6.3.5) — low-half /
    high-half piecewise remap calling `inv_recenter_nonneg`.
  * `inv_recenter_nonneg( v, m )` (§6.3.6) — pure arithmetic
    helper covering the `v > 2*m` short-circuit plus the
    odd / even split.
  * `INV_MAP_TABLE: [u8; 255]` — the §6.3.5 listing transcribed
    verbatim (a permutation of 1..=254 with a duplicated trailing
    253).
  * 16 new unit tests covering `inv_recenter_nonneg` (all three
    piecewise branches plus the `v == 2*m` boundary),
    `INV_MAP_TABLE` length + spot-checks, `inv_remap_prob` against
    both low-half / high-half branches and the `v > 2m`
    short-circuit, `decode_term_subexp` against hand-derived §9.2
    buffers (leg-1 zero plus a sweep that confirms the result
    stays in `0..=254`), and `read_diff_update_prob` confirming
    the `update_prob == 0` passthrough across the full
    1..=255 base-probability sweep.
  * The chain is structural; no caller in §6.3.2 / §6.3.7+ uses it
    yet, so each helper carries `#[allow(dead_code)]` until the
    next round wires them into the table sweeps.

* **Round 3: §9.2 Boolean decoder primitives + §6.3.1 `read_tx_mode`
  walk.**
  * New `src/bool_coder.rs` module implementing the four §9.2
    primitives: `init_bool( sz )` (§9.2.1) with the marker-bit
    zero-conformance check, `read_bool( p )` (§9.2.2) with the
    `split = 1 + (((BoolRange-1) * p) >> 8)` narrow and the
    `range < 128` renormalisation refill, `read_literal( n )`
    (§9.2.4) folded over `read_bool(128)`, and `exit_bool( )`
    (§9.2.3) consuming the remaining `BoolMaxBits` with a
    zero-pad conformance check. `BoolMaxBits` underflow raises
    `InvalidBitstream` instead of silently injecting 0.
  * New `src/compressed.rs` module with `Vp9CompressedHeader`,
    `TxMode` (5-variant enum mapping §3 `TX_MODES`), and the
    `parse_compressed_header(payload, lossless)` entry point.
    `read_tx_mode( )` (§6.3.1) short-circuits to `ONLY_4X4` when
    `Lossless == 1` (§6.2.9), otherwise reads `L(2)` and (for the
    `ALLOW_32X32` raw value) an extra `L(1)` `tx_mode_select` to
    distinguish `ALLOW_32X32` from `TX_MODE_SELECT`.
  * 9 `bool_coder` unit tests + 7 `compressed` unit tests + 4
    end-to-end integration tests (`tests/compressed_header.rs`)
    splicing a hand-derived §9.2 payload past
    `Vp9FrameHeader::uncompressed_header_size_bytes`. Every byte
    vector used was derived by stepping the §9.2 decoder, not
    borrowed from any third-party VP9 implementation.
  * The §6.3.2+ syntax (`tx_mode_probs`, `read_coef_probs`,
    `read_skip_prob`, `read_inter_mode_probs`,
    `read_interp_filter_probs`, …) all flow through the §6.3.3
    `diff_update_prob` chain (`decode_term_subexp` (§6.3.4) +
    `inv_remap_prob` (§6.3.5) + `inv_recenter_nonneg` (§6.3.6) +
    the 255-entry `inv_map_table` constant) and have been
    deferred to the next round so this drop lands a verifiable
    Boolean-coder primitive plus the §6.3.1 walk in isolation.

* **Round 2: full §6.2 uncompressed-header walk.** Extends the round-1
  walker through the end of `uncompressed_header()` and the §6.1.1
  `trailing_bits()` zero-fill alignment:
  * `s(n)` signed-integer reader per spec §4.9.2 plus
    `BitReader::trailing_bits()` zero-pad consumer with §7.1.1
    zero-bit conformance check (`src/bitreader.rs`).
  * `read_loop_filter_params()` (§6.2.8) with full `delta_enabled` /
    `delta_update` / per-ref / per-mode `s(6)` delta walk.
  * `read_quantization_params()` (§6.2.9) with `read_delta_q()`
    (§6.2.10) for `delta_q_y_dc` / `_uv_dc` / `_uv_ac` and the
    `Lossless` derivation.
  * `read_segmentation_params()` (§6.2.11) with `read_prob()`
    (§6.2.12), the 7 `tree_probs`, the 3 `pred_prob` (with
    `temporal_update` switching between f(0)-implicit-255 and
    `read_prob()`), and the per-segment / per-feature
    `feature_enabled` / `feature_value` / optional `feature_sign`
    loop driven by `segmentation_feature_bits` /
    `segmentation_feature_signed`.
  * `read_tile_info()` (§6.2.13) with `Sb64Cols` computed from
    `FrameWidth` via §6.2.6 (`MiCols = (W+7)>>3`,
    `Sb64Cols = (MiCols+7)>>3`) and `calc_min_log2_tile_cols` /
    `calc_max_log2_tile_cols` per §6.2.14
    (`MIN_TILE_WIDTH_B64 = 4`, `MAX_TILE_WIDTH_B64 = 64`). The
    §7.2.11 `tile_cols_log2 <= 6` conformance constraint is checked.
  * f(16) `header_size_in_bytes`, then `trailing_bits()` consumed so
    `uncompressed_header_size_bytes` exposes the byte-aligned offset
    at which the §6.3 compressed header starts.
  * `refresh_frame_context`, `frame_parallel_decoding_mode`,
    `frame_context_idx` (with the `FrameIsIntra ||
    error_resilient_mode` reset to 0 per §6.2 `setup_past_independence`),
    `reset_frame_context`, `refresh_frame_flags` (0xFF for key
    frames per spec).
  * New public types: `LoopFilterParams`, `QuantizationParams`,
    `SegmentationParams`, `TileInfo`, plus constants `MAX_SEGMENTS`,
    `SEG_LVL_MAX`, `SEGMENTATION_FEATURE_BITS`,
    `SEGMENTATION_FEATURE_SIGNED`. `Vp9FrameHeader` extended with all
    the new fields plus `uncompressed_header_size_bytes`.
  * 6 additional bit-reader tests (`s(n)` round-trip,
    `trailing_bits` accept/reject/no-op) + 2 `tile_info`
    arithmetic tests + 4 additional integration tests
    (loop_filter delta-update, segmentation full update,
    `tile_info` 4K increment walk, nonzero trailing bit rejection).
    Existing integration tests extended with the new tail.

* **Round 1: uncompressed-header walker.** A clean-room implementation
  of the structural portion of VP9 spec v0.7 §6.2 / §7.2:
  * MSB-first `f(n)` bit reader (`src/bitreader.rs`) per spec §9.1.
  * `parse_uncompressed_header()` walking `frame_marker`, `Profile`
    (including the `profile == 3` reserved bit), `show_existing_frame`
    early-return with `frame_to_show_map_idx`, `frame_type`,
    `show_frame`, `error_resilient_mode`, `frame_sync_code`,
    `color_config()` (with full §7.2.2 constraint checks including
    CS_RGB-on-profile-0/2 rejection and `reserved_zero` enforcement),
    `frame_size()` and `render_size()`.
  * Public `Vp9FrameHeader`, `ColorConfig`, `ColorSpace`, `FrameType`
    types in the crate root.
  * `Error::UnexpectedEof`, `Error::InvalidBitstream`,
    `Error::Unsupported` variants in addition to the existing
    `NotImplemented`.
  * 3 internal bit-reader tests plus 8 integration tests
    (`tests/uncompressed_header.rs`) covering all four profiles,
    studio/full-swing color ranges, render-size overrides, the
    `show_existing_frame` early return, the intra-only inter-frame
    branch, and three failure paths (bad `frame_marker`, bad
    `frame_sync_code`, truncated buffer).

  `decode_vp9()` / `encode_vp9()` continue to return
  `Error::NotImplemented`; their full pipelines land in later rounds.

### Changed

* **Orphan rebuild (2026-05-20).** The crate was reset to a clean-room
  scaffold. The prior implementation contained module-level docstrings
  and inline comments whose provenance could not be defended against
  the workspace clean-room rule. Per the workspace's
  Implementer-Round procedure, such audit failures are unrecoverable
  via incremental cleanup and require an orphan rebuild.

  Every public API path returned `Error::NotImplemented`. A
  clean-room re-implementation against the VP9 Bitstream & Decoding
  Process Specification (v0.7) has now begun (see "Added" above).

  No `old` branch is retained; long-standing audit failures forfeit
  the archive per workspace policy.
