# oxideav-vp9

Pure-Rust VP9 codec — clean-room re-implementation against the VP9
Bitstream & Decoding Process Specification v0.7.

## Status — 2026-06-12 (round 284)

**Round 284: top-level intra decode wiring — [`decode_vp9`] /
[`decode_intra_frame`] decode whole VP9 keyframes end-to-end,
byte-exact on all 13 intra-leading fixtures of the staged corpus.**
The composition round: the §6.2 / §6.3 header walkers, §9.2 bool
decoder, §6.4.3 partition driver, §6.4.6 intra mode-info, §6.4.21 /
§6.4.24 residual + token decode, §8.5.1 intra prediction, §8.6
dequant, §8.7 inverse transforms and the complete §8.8 loop filter
are now driven by one §6.4 frame walk (new module `decode_frame`):

* §6.4 `decode_tiles( )`: per-tile `f(32)` size prefixes
  ([`tile_payload_sizes`]) + §6.4.1 `get_tile_offset( )` extents,
  §9.2 `init_bool / exit_bool` bracketing each tile, §7.4.1
  `clear_above_context( )` once per frame and §7.4.2
  `clear_left_context( )` per superblock row (partition strips +
  `LeftNonzeroContext`).
* §6.4.3 `decode_partition( )` now fires a `LeafSink` at every
  §6.4.4 `decode_block( )` call site, so the per-block syntax
  decodes inline from the same bool coder the partition syntax uses
  (the `Vec<LeafBlock>` sink keeps the round-19 leaf-log shape for
  partition-only streams).
* §6.4.4 `decode_block( )`: `AvailU = r > 0` / `AvailL = c >
  MiColStart`, §6.4.6 `intra_frame_mode_info( )` with the §6.4.7
  segment-id decode hoisted ahead so the §6.4.9 `SEG_LVL_SKIP` gate
  reads the decoded segment (bit order unchanged), then the §6.4.21
  residual walk — §8.5.1 `predict_intra` per transform block (with
  the §8.5.1 `sub_modes[ blockIdx ]` selection for sub-8x8 luma),
  §6.4.24 `tokens( )` against the frame `coef_probs` and the
  above/left nonzero strips, §8.6.2 reconstruct (per-segment §8.6.1
  quantizers, §6.4.25 `TxType`, lossless WHT path) — and finally the
  §6.4.4 fan-out via the round-31 `decode_block_apply`.
* §8.8 loop filter over the reconstructed planes:
  [`loop_filter_frame_init`] with the §6.2.8 deltas resolved against
  the §7.2 `setup_past_independence` defaults, then
  [`frame_loop_filter`] reading the §6.4.4 frame-wide arrays.
* §8.10 output: [`Vp9DecodedFrame`] (planar `u16` samples + geometry
  + `to_planar_bytes( )` packing — bytes for 8-bit, little-endian
  pairs for 10/12-bit), and [`decode_vp9`] returning the packed
  planar frame. Inter frames and `show_existing_frame` still return
  `Error::Unsupported` (reference-buffer state is the next arc).

Validation (+7 integration tests in `tests/decode_vp9.rs`, suite
total 690 -> 697): `tiny-i-only-16x16`, `lossless-i-only` (§8.7.1.10
WHT) and `q-low` are embedded verbatim and decode byte-exactly in
standalone CI; a workspace-checkout sweep decodes the leading
keyframe of every intra-leading fixture under
`docs/video/vp9/fixtures/` byte-for-byte against `expected.yuv` —
all 13 pass: 4:2:0 and 4:4:4, 8/10/12-bit, RGB, two tile columns,
segmentation AQ, lossless, both quantizer extremes and
frame-parallel mode. Truncation at every byte boundary of the tiny
fixture errors cleanly.

Out of scope for round 284 and queued for later rounds:

* Inter frames: reference-buffer state, §6.4.5's inter arm
  (`inter_frame_mode_info( )`), §8.5.2 inter prediction, §8.4.1
  motion-vector prediction, and superframe-index splitting.
* §8.4 probability adaptation / §6.1.2 frame-context refresh
  (single-frame decode does not persist contexts yet).
* §9.2.4 multi-coder tile parallelism (tiles decode sequentially).

## Previously — round 282 (cargo-fuzz scaffold)

**Round 282: cargo-fuzz scaffold — `fuzz/` stood back up so the
scheduled Fuzz workflow runs again.** The post-rebuild tree had no
`fuzz/` directory, leaving the daily Fuzz workflow red. Round 282
lands a self-contained cargo-fuzz harness package
(`fuzz/Cargo.toml` + `fuzz/fuzz_targets/`) with two panic-surface
targets, auto-discovered by the org reusable fuzz workflow:

* `frame_header` — fuzzes [`parse_uncompressed_header`] and, when
  the header parses, walks the rest of the frame: the §6.3
  compressed-header slice (`header_size_in_bytes` bytes plus the
  header-derived `Lossless` flag into [`parse_compressed_header`])
  and the §6.4 tile-size prefix chain ([`tile_payload_sizes`]).
* `compressed_header` — fuzzes the §9.2 Boolean-decoder walkers
  directly: a leading control byte steers `Lossless`, the
  intra-vs-inter entry point, `interpolation_filter == SWITCHABLE`,
  `allow_high_precision_mv` and the §6.2.5 `ref_frame_sign_bias`
  triple; the remaining bytes feed [`parse_compressed_header`] /
  [`parse_compressed_header_inter`] verbatim.

A 9-entry seed corpus (`fuzz/corpus/*/seed-*`, tracked in git) is
derived from the crate's synthetic test vectors: minimal keyframe
headers (1280x720, 64x64 lossless + non-lossless,
`show_existing_frame`), TX_MODE_SELECT golden buffers, and inter
walks exercising the switchable-filter / high-precision-MV /
sign-bias gates. Local soak: 320 s per target under
AddressSanitizer — 96.4 M execs (`frame_header`) + 43.6 M execs
(`compressed_header`), zero panics / overflows / OOMs. The stale
`fuzz.yml` preamble (describing pre-rebuild harnesses) was
rewritten to match the new target set.

## Previously — round 281 (§8.8 frame-level loop-filter driver)

**Round 281: §8.8 `loop filter process` — the frame-level driver —
[`frame_loop_filter`] + the 3-plane [`CurrFrame`] container.** Round
281 lands the outermost layer of the §8.8 loop-filter arc: the raster
walk over every superblock of the frame per `vp9-spec.txt` lines
5436-5455, deblocking a whole reconstructed frame in place:

* `pub fn frame_loop_filter(curr: &mut CurrFrame, frame:
  &SuperblockFilterFrame)` walks the §8.8 four-deep raster — `row`
  over `0, 8, .. < MiRows`, `col` over `0, 8, .. < MiCols`, `plane`
  over `0..2`, `pass` over `0..1` — invoking the round-278 §8.8.2
  [`superblock_loop_filter`] driver at each step (lines 5451-5455),
  in exactly the listing's nesting order per the §8.8 ordering NOTE
  (lines 5458-5460): many samples are filtered more than once, so
  every §8.8.2 call mutates the plane in place before the next call
  reads it.
* The §8.8 first step — the §8.8.1 frame init (line 5441) — is the
  caller's [`loop_filter_frame_init`] invocation; its `LvlLookup`
  output arrives via `SuperblockFilterFrame::lvl_lookup`, keeping
  the driver free of the §6.2.8 / §6.2.11 header state §8.8.1
  consumes.
* [`CurrFrame`] is the §8.8 input/output (lines 5437-5438): three
  [`SuperblockFilterPlane`] views — Y at `FrameWidth x FrameHeight`,
  U / V at the §8.10 subsampled extents
  `((FrameWidth + subsampling_x) >> subsampling_x) x
  ((FrameHeight + subsampling_y) >> subsampling_y)` (lines
  5944-5948).
* Up-front consistency panics tie the luma extent to the §7.2.6
  `MiCols = (FrameWidth + 7) >> 3` / `MiRows = (FrameHeight + 7) >>
  3` grid (lines 1760-1761) and the chroma extents to the §8.10
  subsampling; partial right / bottom superblocks then ride the
  §8.8.2 driver's off-screen short-circuit.

`pub use frame_loop_filter::{frame_loop_filter, CurrFrame};` joins
the §8.8 surface on the crate root.

Validation (+10 lib tests, lib total 591 -> 601; +6 integration
tests in `tests/frame_loop_filter.rs`):

* Flat-frame identity on all three planes and the step-17 `lvl > 0`
  gate threaded through the whole frame raster.
* Sample-exact equivalence against the §8.8 raster transcribed
  directly from the listing over individual §8.8.2 calls, on
  order-sensitive pseudo-random frames: full 2x2-superblock 4:2:0
  (128x128), partial-superblock `MiCols = MiRows = 12` (96x96),
  non-MI-aligned 52x36 luma / 26x18 chroma extents, and 10-bit
  content.
* A 60 -> 68 luma step exactly at `x = 64` — the boundary between
  the `col = 0` and `col = 8` superblocks, reached as the `col = 8`
  call's edge 0 — filters while far-from-edge columns and the flat
  chroma planes stay put; a U-plane step routes through the
  subsampled raster with Y / V untouched.
* Extent-mismatch panics: a luma plane disagreeing with `MiCols` /
  `MiRows`, and an unsubsampled chroma plane under 4:2:0.

Out of scope for round 281 and queued for later rounds:

* Wiring the §6.4.4 `decode_block` fan-out's per-frame `MiSizes` /
  `TxSizes` / `Skips` / `RefFrames` / `YModes` / `SegmentIds`
  arrays into [`SuperblockFilterFrame`] from inside [`decode_vp9`],
  and calling [`frame_loop_filter`] after frame reconstruction.
* §6.2.5 inter-frame header walker (decode side); §6.4.4
  `decode_block_apply` per-block apply driver.

## Previously — round 278 (§8.8.2 superblock loop filter driver)

**Round 278: §8.8.2 `superblock loop filter process` — the full
per-plane, per-pass driver landed as a public entry point —
[`superblock_loop_filter`].** Round 278 closes the §8.8 loop-filter
arc's per-superblock layer: the new driver composes every
previously-landed primitive (§8.8.1 r37, §8.8.3 r244, §8.8.4 r250,
§8.8.5.1/.2/.3 r253/r255/r259, §8.8.5 r267, §8.8.2 steps 1-14 r274)
into the complete §8.8.2 process per `vp9-spec.txt` lines 5491-5586,
modifying a `CurrFrame[ plane ]` sample plane in place:

* `pub fn superblock_loop_filter(plane_buf: &mut
  SuperblockFilterPlane, frame: &SuperblockFilterFrame, plane: u8,
  pass: u8, row: u32, col: u32)` walks the §8.8.2
  `edge ∈ 0..(16 >> sub) - 1` / `i ∈ 0..edgeLen - 1` raster (lines
  5524-5525) using the round-274 [`superblock_filter_geometry`]
  header, runs the round-274 steps 1-14 predicate bundle
  ([`superblock_filter_edge`]), then threads steps 15-17 in spec
  order: §8.8.3 [`filter_size`] (lines 5579-5581), §8.8.4
  [`adaptive_filter_strength`] at `(loopRow, loopCol)` (lines
  5582-5583), and — when `applyFilter == 1 && lvl > 0` — §8.8.5
  [`sample_filtering`] at `(x >> subX, y >> subY)` along `(dx, dy)`
  (lines 5584-5586), gathering and writing back the §8.8.5.1
  16-sample stencil per lines 5703-5727.
* Step 6's chroma `txSz` resolves through the §6.4.22
  `get_uv_tx_size( )` helper (lines 2871-2876) from the `MiSize` /
  `tx_size` read at `(loopRow, loopCol)`.
* [`SuperblockFilterPlane`] is a mutable `data / stride / width /
  height` view of one `CurrFrame[ plane ]` plane (`i32` samples,
  the §8.8.5 working type). [`SuperblockFilterFrame`] carries the
  six row-major `MiRows x MiCols` per-MI arrays (`MiSizes` /
  `TxSizes` / `Skips` / `RefFrames[..][..][0]` / `YModes` /
  `SegmentIds`), the frame scalars (`mi_cols` / `mi_rows` /
  `subsampling_x` / `subsampling_y` / `loop_filter_sharpness` /
  `bit_depth`), and the §8.8.1 [`LvlLookup`].
* Right / bottom off-screen raster positions short-circuit *before*
  the steps 4-9 per-MI reads (step 13 forces `applyFilter = 0`
  there, making the reads dead), keeping every array access inside
  `MiRows x MiCols`. Out-of-plane stencil reads — possible only for
  the unused outer ring per the §8.8.5.1 NOTE — are edge-clamped;
  write-back drops positions whose true coordinate lies outside the
  plane.
* Edges process in the §8.8.2 raster order with in-place
  write-back, so later edges read samples earlier edges already
  filtered — the spec's ordered-steps semantics (and the §8.8 NOTE
  that the edge order must be respected).

`pub use superblock_loop_filter::{superblock_loop_filter,
SuperblockFilterFrame, SuperblockFilterPlane};` joins the §8.8
surface on the crate root.

Validation (+13 lib tests, lib total 578 -> 591; +8 integration
tests in `tests/superblock_loop_filter.rs`):

* Flat-plane identity across both passes (every §8.8.5 branch is the
  identity on flat content) and the step-17 `lvl > 0` gate
  (`loop_filter_level == 0` leaves a sharp step untouched).
* Vertical / horizontal / 4:2:0-chroma step responses cross-checked
  against §8.8.4 + §8.8.5 invoked directly on the same two-level
  stencil — the narrow window (4 samples around the boundary) moves
  to the exact direct-call values; everything else stays put.
* Step-14 gating threaded end-to-end: a pure tx edge on a skipped
  inter block is untouched; clearing `skip` filters it; a BLOCK_8X8
  block edge filters even when skipped inter.
* Step-13 left / top frame-edge exclusions: a discontinuity hugging
  `x == 0` / `y == 0` never changes (it would trip the §8.8.5.2 hev
  branch against clamped reads if the excluded edge ran).
* Step-16 indexing at `(loopRow, loopCol)`: a segment-1
  `SEG_LVL_ALT_L` absolute override of 0 turns off exactly the edge
  whose MI sits in segment 1 while the segment-0 edge still
  filters.
* A frame ending mid-superblock (`MiCols = MiRows = 6`, 48x48 luma)
  walks the full 64x64 raster with no out-of-bounds access.
* 10-bit step response matches the direct §8.8.5 call at
  `BitDepth = 10`.
* Up-front consistency panics: per-MI arrays shorter than
  `MiRows * MiCols`; `stride < width` plane views.

Out of scope for round 278 and queued for later rounds:

* §8.8 frame-level walk — the four-deep
  `row / col / plane / pass` raster over all superblocks (spec lines
  5450-5455) plus a 3-plane `CurrFrame` container; a thin loop once
  the §7.4 frame state lands.
* Wiring the §6.4.4 `decode_block` fan-out's per-frame `MiSizes` /
  `TxSizes` / `Skips` / `RefFrames` / `YModes` / `SegmentIds`
  arrays into [`SuperblockFilterFrame`] from inside [`decode_vp9`].
* §6.2.5 inter-frame header walker (decode side); §6.4.4
  `decode_block_apply` per-block apply driver.

## Previously — round 274 (§8.8.2 steps 1-14 per-edge predicates)

**Round 274: §8.8.2 per-edge predicate derivation (steps 1-14)
lifted to a public leaf primitive — [`superblock_filter_edge`] +
[`superblock_filter_geometry`].** The §8.8.2 driver's per-edge
book-keeping that turns the raster position `(pass, row, col, edge,
i)` plus the per-MI decode state at the resolved `(loopRow,
loopCol)` into the `(x, y, loopRow, loopCol, isBlockEdge, isTxEdge,
is32Edge, onScreen, applyFilter)` bundle the steps 15-17 hand-off
consumes, per `vp9-spec.txt` §8.8.2 lines 5491-5586.
`superblock_filter_geometry` lifts the `dx` / `dy` / `sub` /
`edgeLen` header (lines 5510-5519); `txSz` is supplied
already-resolved by the caller. Validation: +30 lib tests (548 ->
578) + 8 integration tests in `tests/superblock_filter.rs`.

## Previously — round 267 (§8.8.5 `sample filtering process` outer driver)

**Round 267: §8.8.5 `sample filtering process` outer driver lifted
to a public leaf primitive — [`sample_filtering`].** Round 267
composes the three §8.8.5 sub-processes landed in earlier rounds
(§8.8.5.1 [`filter_mask`] r253, §8.8.5.2 [`narrow_filter`] r255,
§8.8.5.3 [`wide_filter`] r259) into the per-edge dispatcher the
§8.8.2 superblock raster walk invokes at every loop-filter edge:

* `pub fn sample_filtering(samples: &SampleFilterSamples, limit: u8,
  blimit: u8, thresh: u8, filter_size: u8, bit_depth: u8) ->
  SampleFilterOutput` per spec `vp9-spec.txt` §8.8.5 lines
  5662-5684. Runs §8.8.5.1 `filter_mask` on the 16-sample stencil
  first (lines 5672-5674), then dispatches per the §8.8.5 table
  (lines 5678-5684).
* Dispatch (verbatim from lines 5678-5684):
  `filterMask == 0` → no filter (stencil echoed through);
  `filterSize == TX_4X4 || flatMask == 0` → §8.8.5.2 narrow
  filter (fed `hevMask`); `filterSize == TX_8X8 || flatMask2 == 0`
  → §8.8.5.3 wide filter, `log2Size = 3`; otherwise → wide filter,
  `log2Size = 4`.
* The `flatMask` / `flatMask2` reads sit behind `filterSize ==
  TX_4X4` / `filterSize == TX_8X8` short-circuits, so the §8.8.5.1
  `None` returns (emitted exactly when the matching `filterSize >=`
  precondition fails) are never dereferenced.
* `SampleFilterSamples` carries the 16-sample stencil
  (`p7..p0` / `q0..q7`) the §8.8.2 raster walk assembles from
  `CurrFrame[ plane ][ y +/- dy*k ][ x +/- dx*k ]` per §8.8.5.1
  lines 5703-5727. `SampleFilterOutput` carries the full 16-sample
  post-filter stencil: positions outside the chosen filter's
  mutation window are echoed unchanged so the caller writes the
  whole stencil back to `CurrFrame` unconditionally (`p7` / `q7`
  are never mutated by any branch — positions `-8` / `+7` lie
  outside the `i ∈ [-n, n-1]` write window even for `log2Size ==
  4`).

`pub use sample_filtering::{sample_filtering, SampleFilterOutput,
SampleFilterSamples};` exposes the §8.8.5 surface on the crate root
alongside the round-253 §8.8.5.1 surface (`filter_mask`), the
round-255 §8.8.5.2 surface (`narrow_filter`), and the round-259
§8.8.5.3 surface (`wide_filter`).

Validation (+8 lib tests, lib total 546 -> 554; +8 integration
tests in `tests/sample_filtering.rs`):

* §8.8.5 baseline: a flat stencil passes `filterMask` but every
  filter branch is the identity on a flat region — verified at
  every `filterSize` and at BitDepth 8 / 10 / 12 (midpoints 128 /
  512 / 2048).
* §8.8.5 line 5678 `filterMask == 0`: a `limit`-tripping inner
  jump resets `filterMask` and the whole stencil echoes through
  untouched.
* §8.8.5 line 5679 narrow dispatch: `filterSize == TX_4X4` routes
  to §8.8.5.2; the 4-sample window matches the `narrow_filter`
  primitive run directly and the rest of the stencil stays put.
* §8.8.5 line 5679 `flatMask == 0`: a non-flat inner four samples
  forces the narrow branch even at `filterSize == TX_8X8`.
* §8.8.5 line 5681 wide `log2Size == 3`: `filterSize == TX_8X8`
  with a flat inner region routes to §8.8.5.3 (`log2Size = 3`);
  `p2..q2` match the wide primitive, `p3` / `q3` and the outer
  ring stay put.
* §8.8.5 lines 5683-5684 wide `log2Size == 4`: a fully flat
  `filterSize == TX_16X16` region routes to the 16-tap kernel;
  only `p7` / `q7` echo through.
* §8.8.5 line 5682 `flatMask2 == 0`: an outer-ring outlier at
  `filterSize == TX_16X16` drops the dispatch back from `log2Size
  == 4` to `log2Size == 3`.

Out of scope for round 267 and queued for later rounds:

* §8.8.2 `superblock_loop_filter( )` — the per-superblock raster
  walk that assembles the stencil from `CurrFrame`, derives
  `(filterSize, limit, blimit, thresh)` via §8.8.3 + §8.8.4, calls
  this primitive for each `(plane, pass, row, col)` edge, and
  writes [`SampleFilterOutput`] back into `CurrFrame`. Needs the
  `MiSizes` / `TxSizes` / `Skips` / `RefFrames` / `YModes` /
  `SegmentIds` arrays the §6.4.4 `decode_block` fan-out produces.
* §6.2.5 inter-frame header walker (decode side); §6.4.4
  `decode_block_apply` per-block apply driver.

## Previously — round 259 (§8.8.5.3 `wide filter process`)

**Round 259: §8.8.5.3 `wide filter process` lifted to a public
leaf primitive — [`wide_filter`].** Round 259 lands the per-edge
low-pass primitive the §8.8.5 outer driver will call after the
round-253 §8.8.5.1 `filter_mask` step picks the wide branch via
the §8.8.5 dispatch table at `vp9-spec.txt` lines 5681-5684:

* `pub fn wide_filter(samples: &WideFilterSamples, log2_size: u32,
  bit_depth: u8) -> WideFilterOutput` per spec
  `vp9-spec.txt` §8.8.5.3 lines 5855-5888.
* `log2_size == 3` (8-tap kernel, `n == 3`): the loop walks
  `i ∈ [-3, 2]` (6 mutated outputs at positions `p2..p0`,
  `q0..q2`). The outer eight fields of [`WideFilterOutput`]
  (`op6..op3`, `oq3..oq6`) echo the corresponding input through
  so the caller can write all 14 fields unconditionally.
* `log2_size == 4` (16-tap kernel, `n == 7`): the loop walks
  `i ∈ [-7, 6]` (14 mutated outputs at positions `p6..p0`,
  `q0..q6`).
* The kernel `F[ i ] = Round2( CurrFrame[i] + sum_{j=-n..n}
  CurrFrame[Clip3(-(n+1), n, i+j)], log2Size )` is implemented
  verbatim with `n = (1 << (log2Size - 1)) - 1` (lines
  5864-5865) and `Clip3` edge-replication (line 5879). Total
  samples summed: `2n + 2`. `Round2( t, log2Size ) = (t + (1 <<
  (log2Size - 1))) >> log2Size` matches the §3 half-up rounding.
* Unlike §8.8.5.2 the wide filter does NOT subtract the `0x80 <<
  (BitDepth - 8)` working-range offset — arithmetic happens in
  the original unsigned-pixel domain per `vp9-spec.txt` lines
  5868-5885 verbatim.
* `WideFilterSamples` carries the 16-sample stencil
  (`p7..p0`, `q0..q7`) the §8.8.5 outer driver assembles from
  `CurrFrame[ plane ][ y +/- dy*k ][ x +/- dx*k ]` per
  §8.8.5.1 lines 5703-5727.

`pub use wide_filter::{wide_filter, WideFilterOutput,
WideFilterSamples};` exposes the §8.8.5.3 surface on the crate
root alongside the round-255 §8.8.5.2 surface (`narrow_filter` /
`NarrowFilterOutput` / `NarrowFilterSamples`) and the round-253
§8.8.5.1 surface (`filter_mask` / `FilterMask` /
`FilterMaskSamples`).

Validation (+12 lib tests, lib total 534 -> 546; +9 integration
tests in `tests/wide_filter.rs`):

* §8.8.5.3 unity-gain on flat stencils at every supported
  `(log2_size, BitDepth)` combination: `(3, 8)`, `(4, 8)`,
  `(3, 10)`, `(4, 12)` all return the input value at every
  output position.
* §8.8.5.3 outer-field echo on `log2_size == 3`: arbitrary
  values in `p7..p4` / `q4..q7` come through unchanged on the
  outer eight output fields; only the inner six are filtered.
* §8.8.5.3 hand-traced step response (`p = 0`, `q = 100`,
  `log2_size = 3`): six exact mutated values
  `(op2, op1, op0, oq0, oq1, oq2) = (13, 25, 38, 63, 75, 88)`
  derived from the listing verbatim.
* §8.8.5.3 line 5879 `Clip3` edge-replication: isolating `p3 =
  80` in an otherwise-zero stencil drives `op2 = 30` (three
  extra copies of `p3` pulled in via the clamp).
* §8.8.5.3 log2_4 (16-tap) step response at the boundary:
  `op0 = 56` for the `(0 → 128)` step.
* §8.8.5.3 line 5882 `Round2` half-up rounding verified at the
  `Round2(8, 3) = 1` boundary.
* §8.8.5 dispatch precondition: `log2_size ∉ {3, 4}` panics
  with the §8.8.5.3 message — `log2_size = 2`, `5`, `7` all
  rejected.

Out of scope for round 259 and queued for later rounds:

* §8.8.5 `sample_filtering( )` — the per-edge outer driver that
  reads the stencil from `CurrFrame`, runs §8.8.5.1, dispatches
  to §8.8.5.2 or §8.8.5.3 (this round), and writes the result
  back.
* §8.8.2 `superblock_loop_filter` — the per-superblock raster
  walk that calls §8.8.3 + §8.8.4 + §8.8.5 for each `(loopRow,
  loopCol)` step.
* §6.2.5 inter-frame header walker (decode side); §6.4.4
  `decode_block_apply` per-block apply driver.

## Previously — round 255 (§8.8.5.2 `narrow filter process`)

Round 255 lands the per-edge sample-mutation primitive the §8.8.5
outer driver calls when the round-253 §8.8.5.1 `filter_mask`
output picks the narrow branch:

* `pub fn narrow_filter(samples: &NarrowFilterSamples, hev_mask:
  bool, bit_depth: u8) -> NarrowFilterOutput` per spec
  `vp9-spec.txt` §8.8.5.2 lines 5795-5853.
* `hev_mask == 1` (lines 5809-5811): modifies only `op0` / `oq0`;
  `op1` / `oq1` are returned equal to the input so the caller can
  write them back unconditionally. The filter draws from all four
  input samples via `filter = filter4_clamp(ps1 - qs1)` →
  `filter = filter4_clamp(filter + 3 * (qs0 - ps0))`.
* `hev_mask == 0` (lines 5806-5808 + 5846-5852): modifies all four
  samples. The `ps1 - qs1` term drops out so `filter` starts at 0,
  and a half-strength pass via `Round2(filter1, 1)` is added to
  `op1` / `oq1`.
* `filter1 = filter4_clamp(filter + 4) >> 3` and
  `filter2 = filter4_clamp(filter + 3) >> 3` (lines 5840-5841)
  bias the rounding for `oq0` vs `op0` asymmetrically.
* `filter4_clamp` (lines 5824-5826) clips into the signed range
  `[-(1 << (BitDepth - 1)), (1 << (BitDepth - 1)) - 1]`; the
  `0x80 << (BitDepth - 8)` offset (lines 5834-5837) is applied and
  undone verbatim for `BitDepth ∈ {8, 10, 12}`.
* `NarrowFilterSamples` carries the 4-sample stencil
  (`p1`, `p0`, `q0`, `q1`) the §8.8.5 outer driver assembles from
  `CurrFrame[ plane ][ y +/- dy*k ][ x +/- dx*k ]` per lines
  5830-5833. `NarrowFilterOutput` carries the four mutated samples
  the caller writes back to `CurrFrame`.

`pub use narrow_filter::{narrow_filter, NarrowFilterOutput,
NarrowFilterSamples};` exposes the §8.8.5.2 surface on the crate
root alongside the round-253 §8.8.5.1 surface (`filter_mask` /
`FilterMask` / `FilterMaskSamples`).

Validation (+12 lib tests, lib total 522 -> 534; +9 integration
tests in `tests/narrow_filter.rs`):

* §8.8.5.2 baseline at `BitDepth ∈ {8, 10, 12}`: a flat stencil at
  the bit-depth midpoint (128 / 512 / 2048) yields no change on
  either `hev_mask` branch.
* §8.8.5.2 lead paragraph (lines 5806-5811): the `hev_mask == 1`
  branch leaves `op1` / `oq1` equal to the input even when the
  inner step is sharp; the `hev_mask == 0` branch mutates all
  four samples via `Round2(filter1, 1)`.
* §8.8.5.2 line 5825 — `filter4_clamp` saturates at the bit-depth
  range: 8-bit `(255, 255, 0, 0)` → `(_, 239, 16, _)`; 10-bit
  `(1023, 1023, 0, 0)` → `(_, 959, 64, _)` (wider working range
  shifts the saturation point).
* §8.8.5.2 lines 5840-5841 — `filter1` / `filter2` asymmetric
  rounding: with `filter == 4`, `filter1 = 1` (shifts `q0`) but
  `filter2 = 0` (leaves `p0`).
* §8.8.5.2 line 5847 — `Round2(filter1, 1)` half-strength pass:
  with `filter1 == 3` the smooth pass shifts `p1` / `q1` by 2;
  with `filter1 == 0` it leaves them alone.
* §8.8.5.2 collapse case: when `ps1 == qs1` (matched outer
  samples), the `filter4_clamp(ps1 - qs1)` term equals 0, so the
  hev and smooth branches agree on `op0` / `oq0` (they still
  differ on `op1` / `oq1`).
* §8.8.5.2 round-trip property: `op0 + oq0` stays within 1 of
  `p0 + q0` over a 5×5 stencil grid (the offset cancels and the
  `+3`/`+4` rounding asymmetry is the only source of drift).

## Previously — round 253 (§8.8.5.1 `filter mask process`)

Round 253 lands the per-edge mask derivation as a pure-state
function the §8.8.5 outer driver will call before dispatching to
the §8.8.5.2 narrow filter or the §8.8.5.3 wide filter at every
loop-filter edge:

* `pub fn filter_mask(samples: &FilterMaskSamples, limit: u8,
  blimit: u8, thresh: u8, filter_size: u8, bit_depth: u8) ->
  FilterMask` per spec `vp9-spec.txt` §8.8.5.1 lines 5685-5792.
* Step 1 (`hevMask`, lines 5730-5734): `hevMask = (Abs(p1 - p0) >
  threshBd) || (Abs(q1 - q0) > threshBd)`, with `threshBd = thresh
  << (BitDepth - 8)`. Strict `>` per the listing.
* Step 2 (`filterMask`, lines 5737-5750): the seven inner abs-diff
  pair tests against `limitBd = limit << (BitDepth - 8)` plus the
  `Abs(p0 - q0) * 2 + Abs(p1 - q1) / 2 > blimitBd` boundary term
  (with `blimitBd = blimit << (BitDepth - 8)`). Integer division
  on the `/ 2` floor is honoured verbatim. `filterMask = (mask ==
  0)` — `true` only when every test stays at false.
* Step 3 (`flatMask`, lines 5753-5774): six abs-diff tests
  relative to `p0` / `q0` over the inner four samples on each
  side, against `thresholdBd = 1 << (BitDepth - 8)`. Gated by
  `filterSize >= TX_8X8` per line 5697; returned as `None`
  otherwise.
* Step 4 (`flatMask2`, lines 5777-5792): eight abs-diff tests
  relative to `p0` / `q0` over the outer four samples on each
  side, against the same `thresholdBd`. Gated by `filterSize >=
  TX_16X16` per line 5698; returned as `None` otherwise.
* `FilterMaskSamples` carries the 16-sample stencil
  (`p7..p0` / `q0..q7`) the §8.8.5 outer driver assembles from
  `CurrFrame[ plane ][ y +/- dy*k ][ x +/- dx*k ]` per lines
  5703-5727. Samples are `i32` so the abs-diff subtractions don't
  underflow at 10-bit / 12-bit pixels.

`pub use filter_mask::{filter_mask, FilterMask, FilterMaskSamples};`
exposes the §8.8.5.1 surface on the crate root alongside the
round-250 §8.8.4 surface (`adaptive_filter_strength` /
`FilterStrength` / `mode_to_mode_type`) and the round-244 §8.8.3
surface (`filter_size` / `TX_4X4` / `TX_8X8` / `TX_16X16` /
`TX_32X32` / `PASS_VERTICAL` / `PASS_HORIZONTAL`).

Validation (+15 lib tests, lib total 507 -> 522; +9 integration
tests in `tests/filter_mask.rs`; suite total 557 -> 581):

* §8.8.5.1 baseline: a flat 16-sample stencil at `BitDepth = 8` /
  `TX_16X16` yields `hev_mask = false`, `filter_mask = true`,
  `flat_mask = Some(true)`, `flat_mask2 = Some(true)`.
* §8.8.5.1 lead paragraph gating (lines 5697-5698): `TX_4X4`
  collapses both flat masks to `None`; `TX_8X8` keeps `flat_mask`
  populated but gates `flat_mask2` to `None`.
* §8.8.5.1 step 1 `hevMask` triggers on both `Abs(p1 - p0)` and
  `Abs(q1 - q0)` terms; equality with `threshBd` keeps the mask at
  `0` (strict `>` per line 5733).
* §8.8.5.1 step 2 `filterMask`: outer-pair resets (`p3 - p2` /
  `q3 - q2`) and the boundary term `|p0 - q0|*2 + |p1 - q1|/2 >
  blimitBd` reset independently.
* §8.8.5.1 step 2 integer-division `/ 2`: a `p1/q1` diff of 3
  floors to 1 (not 1.5), so the boundary term reads `|p0 - q0|*2
  + 1`.
* §8.8.5.1 step 3 `flatMask`: resets when `Abs(p2 - p0) >
  thresholdBd` even with a flat `(p0, q0)` boundary.
* §8.8.5.1 step 4 `flatMask2`: outer-ring `p7 - p0` diff of 2
  resets `flatMask2` while `flatMask` survives. Rising-slope
  stencil (p7..p4 ascending toward p0; q4..q7 descending from q0)
  confirms the same partition.
* §8.8.5.1 BitDepth scaling: at 10-bit, `thresh = 4` scales to
  `threshBd = 16`; at 12-bit, `thresholdBd = 16`. Strict `>` cutoff
  verified at the boundary in both depths.

Out of scope for round 253 and queued for later rounds:

* §8.8.5 `sample_filtering( )` — the per-edge outer driver that
  reads the stencil from `CurrFrame` and dispatches to narrow /
  wide filters based on this round's [`FilterMask`].
* §8.8.5.2 `filter4` / §8.8.5.3 `filter6` / `filter8` / `filter16`
  — the sample-mutating filter primitives that consume the mask.
* §8.8.2 `superblock_loop_filter( )` — the per-superblock raster
  walk that invokes §8.8.3 + §8.8.4 + §8.8.5 in sequence at every
  `(loopRow, loopCol)` step.

## Status — 2026-06-07 (round 250)

**Round 250: §8.8.4 `adaptive_filter_strength( )` lifted to a public
leaf primitive — [`adaptive_filter_strength`].** Round 250 lands
the per-(loopRow, loopCol) filter-strength derivation as a pure-
state function the §8.8.2 superblock raster walk will call at every
luma 8x8 position to produce the `(lvl, limit, blimit, thresh)`
tuple the §8.8.5 sample-filter pass consumes:

* `pub fn adaptive_filter_strength(lvl_lookup: &LvlLookup,
  segment_id: usize, ref_frame: i32, y_mode: u8,
  loop_filter_sharpness: u8) -> Option<FilterStrength>` per spec
  `vp9-spec.txt` §8.8.4 lines 5626-5661. Returns `None` for an
  out-of-range axis (`segment_id >= MAX_SEGMENTS`, `ref_frame`
  outside `0..=3`).
* Step 1 (`lvl` derivation, lines 5632-5639): the §8.8.1
  [`LvlLookup`] from round 37 is indexed by `(segment_id, ref_frame,
  modeType)`, where `modeType = 1` when `y_mode` is one of the
  three §7.4.11 MV-predicting inter modes (`NEARESTMV` = 10,
  `NEARMV` = 11, `NEWMV` = 13) and `modeType = 0` for intra modes
  (0..=9) or `ZEROMV` = 12, per the §8.8.4 step-1 partition (lines
  5637-5638).
* Step 2 (`shift` derivation, lines 5642-5645): `shift = 2` when
  `loop_filter_sharpness > 4`, `shift = 1` when
  `loop_filter_sharpness > 0`, and `shift = 0` otherwise.
* Step 3 (`limit` derivation, lines 5648-5651): sharpness > 0 →
  `limit = Clip3( 1, 9 - loop_filter_sharpness, lvl >> shift )`;
  sharpness = 0 → `limit = Max( 1, lvl >> shift )`. Both branches
  guarantee `limit >= 1` even when `lvl >> shift = 0`.
* Step 4 (`blimit` line 5660): `blimit = 2 * (lvl + 2) + limit`.
  The §8.8.1 `Clip3( 0, MAX_LOOP_FILTER, … )` ceiling bounds `lvl`
  at 63, so the maximum `blimit` is `2 * 65 + 63 = 193` — under
  `u8::MAX`.
* Step 5 (`thresh` line 5661): `thresh = lvl >> 4` — the §8.8.5.1
  high-edge-variance threshold. Bounded by `lvl <= 63 → thresh <=
  3`.
* §7.4.11 inter-mode constants surfaced verbatim (`vp9-spec.txt`
  lines 3957-3961): `pub const NEARESTMV: u8 = 10`, `pub const
  NEARMV: u8 = 11`, `pub const ZEROMV: u8 = 12`, `pub const NEWMV:
  u8 = 13`.
* Helper `pub fn mode_to_mode_type(mode: u8) -> usize` exposes the
  §8.8.4 step-1 classification at module scope so a future §8.8.2
  raster walker can derive `modeType` directly from `YModes[ ][ ]`
  without re-reading the lookup.

`pub use adaptive_filter_strength::{adaptive_filter_strength,
mode_to_mode_type, FilterStrength, NEARESTMV, NEARMV, NEWMV,
ZEROMV};` exposes the §8.8.4 surface on the crate root alongside the
round-37 §8.8.1 surface (`loop_filter_frame_init` / `LvlLookup` /
`MAX_LOOP_FILTER` / `MAX_MODE_LF_DELTAS`) and the round-244 §8.8.3
surface (`filter_size` / `TX_4X4` / `TX_8X8` / `TX_16X16` /
`TX_32X32` / `PASS_VERTICAL` / `PASS_HORIZONTAL`).

Validation (+11 lib tests, lib total 496 -> 507; +7 integration
tests in `tests/adaptive_filter_strength.rs`; suite total 539 ->
557):

* §8.8.4 step 1 modeType classification: every intra mode 0..=9 and
  `ZEROMV` (= 12) maps to `modeType = 0`; `NEARESTMV` / `NEARMV` /
  `NEWMV` map to `modeType = 1`.
* §8.8.4 step 1 lookup dispatch: with non-zero mode-delta = `[0,
  4]`, the mode-0 / mode-1 columns of `LvlLookup[s][LAST][m]` differ
  by 4 and the four inter MV modes route to the mode-1 column while
  `ZEROMV` and the ten intra modes route to the mode-0 column.
* §8.8.4 step 2 `shift` boundaries verified at `sharpness = 0` (=
  0), `sharpness = 1` (= 1), `sharpness = 5` (= 2) per the strict-
  `>` comparisons in the spec.
* §8.8.4 step 3 envelope: full `0..=7` sharpness sweep computes the
  expected `(shift, limit, blimit, thresh)` independently and
  matches the primitive at `lvl = 40`.
* §8.8.4 step 3 lower clip at `lvl = 0`: both sharpness = 0
  (`Max(1, 0) = 1`) and sharpness > 0 (`Clip3(1, 9-sharp, 0) = 1`)
  enforce `limit >= 1`.
* §8.8.4 step 3 upper clip at sharpness = 7: `Clip3(1, 2, ...) <=
  2` even when `lvl >> shift = 15`.
* §8.8.4 steps 1 + 2 + 3 + 4 + 5 end-to-end at `level = 25` with
  §7.2 setup_past_independence defaults: returns
  `FilterStrength { lvl: 25, limit: 25, blimit: 79, thresh: 1 }`.
* §8.8.1 segment override propagates into §8.8.4 step 1 lookup:
  segment 2's `feature_data` override of 50 surfaces as
  `out.lvl = 50` / `blimit = 154` / `thresh = 3`.
* §8.8.4 step 5 `thresh` partition: a sweep at level ∈ {15, 16, 31,
  32, 47, 48, 63} confirms the four `lvl >> 4` bands.
* Out-of-range axes return `None` without panic: `segment_id =
  MAX_SEGMENTS`, `ref_frame = -1` (the `NONE` sentinel from
  §6.4.16), `ref_frame = 4`.

Out of scope for round 250 and queued for later rounds:

* §8.8.2 `superblock_loop_filter( )` — the per-superblock raster
  walk that invokes this primitive at every `(loopRow, loopCol)`
  step. Needs `MiSizes` / `TxSizes` / `Skips` / `RefFrames` /
  `YModes` / `SegmentIds` arrays the §6.4.4 [`decode_block`] fan-
  out produces.
* §8.8.5 `sample_filtering( )` — the actual edge-filter primitives
  (`filter4` / `filter6` / `filter8` / `filter16`) that consume the
  `FilterStrength` tuple this round returns.
* §6.2.5 `frame_size_with_refs` — needed before the inter-frame
  uncompressed-header path returns anything other than
  `Error::Unsupported`.

## Status — 2026-06-07 (round 244)

**Round 244: §8.8.3 `filter_size( )` lifted to a public leaf
primitive — [`filter_size`].** Round 244 lands the per-edge filter-
size derivation as a pure-state function the §8.8.2 superblock
raster walk will call at every loop-filter edge to pick the maximum
filter size between the §8.8.4 strength derivation and the §8.8.5
sample-filter pass:

* `pub fn filter_size(tx_sz: u8, is_32_edge: bool, pass: u8, x: u32,
  y: u32, sub_x: u8, sub_y: u8, mi_cols: u32, mi_rows: u32) -> u8`
  per spec `vp9-spec.txt` §8.8.3 lines 5587-5625. Returns one of
  `TX_4X4` / `TX_8X8` / `TX_16X16` (the §8.8.3 `Min(TX_16X16, txSz)`
  step caps the output below `TX_32X32`).
* Step 1 (`baseSize` derivation, lines 5609-5611): the `txSz ==
  TX_4X4 && is32Edge == 1 → baseSize = TX_8X8` promotion realises
  the §8.8.3 lead paragraph's "minimum size of TX_8X8 for boundaries
  on a multiple of 32 samples" rule; otherwise
  `baseSize = Min(TX_16X16, txSz)`.
* Step 2 (chroma frame-edge clip, lines 5615-5624): the vertical
  pass clip `pass == 0 && sub_x == 1 && baseSize == TX_16X16 && (x
  >> 3) == MiCols - 1 → TX_8X8` realises the §8.8.3 lead paragraph's
  "reduce the width of chroma filters" rule; the mirror horizontal
  pass clip `pass == 1 && sub_y == 1 && baseSize == TX_16X16 && (y
  >> 3) == MiRows - 1 → TX_8X8` handles the bottom edge.
* Constants exposed verbatim per §7.4.8 (`vp9-spec.txt` lines
  3937-3940): `pub const TX_4X4: u8 = 0`, `pub const TX_8X8: u8 =
  1`, `pub const TX_16X16: u8 = 2`, `pub const TX_32X32: u8 = 3`.
  The §8.8.3 pass-direction integers are surfaced as `pub const
  PASS_VERTICAL: u8 = 0` and `pub const PASS_HORIZONTAL: u8 = 1`.

`pub use filter_size::{filter_size, PASS_HORIZONTAL, PASS_VERTICAL,
TX_16X16, TX_32X32, TX_4X4, TX_8X8};` exposes the §8.8.3 surface on
the crate root alongside the round-37 §8.8.1 surface
(`loop_filter_frame_init` / `LvlLookup` / `MAX_LOOP_FILTER` /
`MAX_MODE_LF_DELTAS`).

Validation (+14 lib tests, lib total 482 -> 496; +8 integration
tests in `tests/filter_size.rs`; suite total 517 -> 539):

* §8.8.3 line 5611 `Min(TX_16X16, txSz)` clip: `TX_8X8` stays
  `TX_8X8`; `TX_32X32` caps at `TX_16X16`; `TX_4X4` (without the
  `is32Edge` promotion) stays `TX_4X4`.
* §8.8.3 line 5610 `is32Edge` promotion: `tx_sz = TX_4X4 &&
  is_32_edge = true` lifts `baseSize` to `TX_8X8`.
* §8.8.3 lines 5615-5619 vertical chroma right-edge clip: fires
  only on `pass == 0 && sub_x == 1 && baseSize == TX_16X16 && (x
  >> 3) == MiCols - 1`. Verified by a `mi_cols ∈ [1, 8]` sweep
  comparing the `sub_x = 1` clip with the `sub_x = 0` no-clip path.
* §8.8.3 lines 5620-5624 horizontal chroma bottom-edge clip: mirror
  gate on `pass == 1 && sub_y == 1 && baseSize == TX_16X16 && (y >>
  3) == MiRows - 1`. Verified by the same `mi_rows ∈ [1, 8]` sweep.
* §8.8.3 step 1 + step 2 composition: a `TX_32X32` input on the
  sub-sampled chroma right-edge clips to `TX_8X8` via the
  intermediate `baseSize = TX_16X16`; a `TX_4X4 + is32Edge` input
  on the same edge stays at `TX_8X8` (the step-1 promotion's
  output skips the step-2 gate because `baseSize != TX_16X16`).
* §8.8.3 lead paragraph (lines 5597-5599) purpose check: an 8x8
  grid of edges with `sub_x = sub_y = 0` (luma plane) never sees a
  clip on either pass; both chroma-clip gates require sub-sampling.
* `mi_cols == 0` / `mi_rows == 0` edge case: the on-edge gates
  evaluate to `false` (no integer wrap-around into a spurious clip).

Out of scope for round 244 (each lands in a separate later round):

* §8.8.2 `superblock_loop_filter` — the per-superblock raster walk
  that calls [`filter_size`] at every `(plane, pass, row, col)`
  step. Needs the `MiSizes` / `TxSizes` / `Skips` / `RefFrames`
  arrays the §6.4.4 [`decode_block_apply`] fan-out produces.
* §8.8.4 `adaptive_filter_strength` — reads the round-37
  [`LvlLookup`] and emits `(lvl, limit, blimit, thresh)` per lines
  5626-5661.
* §8.8.5 `sample_filtering` — the actual MB-edge deblocking filter
  primitives (`filter4` / `filter6` / `filter8` / `filter16`).

## Status — 2026-06-06 (round 37)

**Round 37: §8.8.1 `loop_filter_frame_init( )` lifted to a public
primitive — [`loop_filter_frame_init`].** Round 37 lands the
per-frame §8.8 loop-filter init step as a standalone public function
that converts the §6.2.8 [`LoopFilterParams`] and §6.2.11
[`SegmentationParams`] walker outputs (already produced by earlier
rounds) into the `LvlLookup[ MAX_SEGMENTS ][ MAX_REF_FRAMES ][
MAX_MODE_LF_DELTAS ]` filter-strength table the §8.8.4
adaptive-strength consumer reads at every superblock raster step:

* `pub fn loop_filter_frame_init(lf: &LoopFilterParams, seg:
  &SegmentationParams, ref_deltas: [i8; 4], mode_deltas: [i8; 2]) ->
  LvlLookup` per spec `vp9-spec.txt` §8.8.1 lines 5465-5488. Computes
  `nShift = loop_filter_level >> 5` per line 5468; iterates the
  `segment_id = 0..MAX_SEGMENTS - 1` outer loop per line 5469;
  applies the §8.8.1 step 1 `lvlSeg = loop_filter_level` init, the
  step 2 `seg_feature_active( SEG_LVL_ALT_L )` override (§6.4.9 gate
  + §6.2.11 abs/delta mode + `Clip3( 0, MAX_LOOP_FILTER, … )`
  saturation), the step 3 `delta_update == 0` per-segment broadcast,
  and the step 4 `delta_enabled == 1` per-(ref, mode) delta-apply
  walk (with the spec listing's `INTRA_FRAME / 0` line 5481 +
  `LAST..ALTREF / 0..MAX_MODE_LF_DELTAS - 1` lines 5482-5487 split,
  and the final `Clip3( 0, MAX_LOOP_FILTER, … )` saturations on every
  output cell).
* `pub struct LvlLookup { pub levels: [[[u8; 2]; 4]; 8] }` carries
  the §8.8.1 output indexed by `(segment_id, ref_frame, mode)` with
  a `LvlLookup::zeros()` no-filter identity constructor and a
  bounds-checked `get(segment_id, ref_frame, mode) -> Option<u8>`
  read-back surface. Cells fit `u8` because §8.8.1's `Clip3( 0,
  MAX_LOOP_FILTER, … )` (line 5476 / 5481 / 5486) saturates every
  output into `0..=63`.
* Constants exposed: `pub const MAX_MODE_LF_DELTAS: usize = 2` (§3
  `vp9-spec.txt` line 513 — the per-mode delta slot count), `pub
  const MAX_LOOP_FILTER: i32 = 63` (§3 line 515 — the §8.8.1 `Clip3`
  upper bound). The §3 `SEG_LVL_ALT_L = 1` segmentation-feature
  index is the crate-local `pub(crate) const` carrying the §8.8.1
  step 2 gate's feature slot.

Caller-supplied `ref_deltas[ 4 ]` and `mode_deltas[ 2 ]` arrays carry
the resolved (post-`Option::unwrap_or(prev)`) values per §7.2's
"previous value" rule — the §7.2 `setup_past_independence` defaults
are `loop_filter_ref_deltas = [1, 0, -1, -1]` (indexed `INTRA / LAST
/ GOLDEN / ALTREF`) and `loop_filter_mode_deltas = [0, 0]`. Keeping
the resolved arrays as inputs lets §8.8.1 stand alone without a
`Vp9DecoderState` carrying the running deltas across frames — that
state is the §7.2 inter-frame orchestrator's responsibility.

Validation (+13 lib tests, lib total 469 -> 482; +5 integration
tests in `tests/loop_filter.rs`; suite total 499 -> 517):

* §8.8.1 base case: zero `loop_filter_level` with everything disabled
  yields an all-zero `LvlLookup` (line 5468 makes `nShift = 0`, step
  3 broadcasts `lvlSeg = 0` into every cell, step 4 is gated off).
* §8.8.1 step 3 broadcast: `delta_update == 0 && delta_enabled == 0`
  broadcasts `lvlSeg = loop_filter_level` into every `(segment_id,
  ref, mode)` cell.
* §8.8.1 step 4 alone: `delta_update == 1 && delta_enabled == 1`
  skips the step-3 broadcast and step 4 covers every cell except
  `(INTRA_FRAME, 1)` (line 5481 is mode-0-only; lines 5482-5487 skip
  `INTRA_FRAME`); the `INTRA / 1` cell stays at the zero default.
* §8.8.1 line 5468 `nShift = level >> 5` threshold: at level 32 a
  `±1` ref-delta moves the cell by ±2 from `lvlSeg`; at level 31
  the shift is 0 (1:1).
* §8.8.1 line 5481 / 5486 `Clip3( 0, MAX_LOOP_FILTER, … )`
  saturation on both bounds: positive overflow clamps to 63;
  negative underflow clamps to 0 (no `u8` underflow).
* §8.8.1 step 2 segment override: a segment with `SEG_LVL_ALT_L`
  enabled has its `lvlSeg` REPLACED in abs mode (`feature_data` is
  the new level) or ADDED to in delta mode (`feature_data +
  loop_filter_level`).
* §8.8.1 step 2.c `Clip3( 0, MAX_LOOP_FILTER, lvlSeg )` saturation:
  a `feature_data = -100` in delta mode clamps to 0; a `feature_data
  = 200` in abs mode clamps to 63.
* §6.4.9 `seg_feature_active( )` gate: when `segmentation_enabled ==
  0` step 2 is OFF even if `feature_enabled[ ][ SEG_LVL_ALT_L ] ==
  1` (both flags are required per §6.4.9).
* §8.8.1 step 3 + step 4 composition: with `delta_update = 0 &&
  delta_enabled = 1` step 3 broadcasts then step 4 partially
  overwrites, leaving `(INTRA_FRAME, 1)` at the step-3 broadcast
  value (line 5481 + 5482-5487 don't cover that cell).
* `LvlLookup::get` returns `Some(_)` for every in-range
  `(segment_id < 8, ref_frame ∈ [0, 4), mode < 2)` triple and `None`
  for every out-of-range axis (including `ref_frame < 0`).

`pub use loop_filter::{loop_filter_frame_init, LvlLookup,
MAX_LOOP_FILTER, MAX_MODE_LF_DELTAS};` exposes the §8.8.1 surface on
the crate root alongside `parse_uncompressed_header` /
`parse_compressed_header` / `parse_compressed_header_inter` /
`tile_payload_sizes`.

Out of scope for round 37 (each lands in a separate later round):

* §8.8.2 `superblock_loop_filter` — the per-superblock raster walk
  driving §8.8.3 / §8.8.4 / §8.8.5, which needs `MiSizes` / `TxSizes`
  / `Skips` / `RefFrames` arrays the §6.4.4 `decode_block` fan-out
  produces. The round-31 `decode_block_apply` primitive plus the
  round-19 `decode_partition` driver already build that frame
  state; wiring them into §8.8.2 is the natural follow-up.
* §8.8.3 `filter_size` — the `txSz` / `is32Edge` derivation per
  lines 5587-5625.
* §8.8.4 `adaptive_filter_strength` — reads `LvlLookup` produced by
  this round and emits `(lvl, limit, blimit, thresh)` per lines
  5626 onwards.
* §8.8.5 `sample_filtering` — the actual MB-edge deblocking filter
  primitives (`filter4` / `filter6` / `filter8` / `filter16`).

## Status — 2026-06-05 (round 36)

**Round 36: §6.4 lines 2306-2311 byte-walk lifted to a public
primitive — [`tile_payload_sizes`].** Round 36 factors the pure
byte-arithmetic prefix walk out of the round-33 `decode_tiles`
outer driver into a new standalone public function that a caller can
invoke without instantiating any per-tile bool-coder state:

* `pub fn tile_payload_sizes(data, sz, tile_rows_log2,
  tile_cols_log2) -> Result<Vec<u32>, Error>` per spec `vp9-spec.txt`
  §6.4 lines 2306-2311. Walks the `(1 << tile_rows_log2) x (1 <<
  tile_cols_log2)` grid in row-major order per §6.4 lines 2304-2305;
  reads the `f(32)` length prefix per line 2310 for every tile
  except the last; applies the `sz -= tile_size + 4` running
  subtraction per line 2311 with checked arithmetic; assigns
  `tile_size = sz` per line 2308 to the last tile; range-checks
  every declared body against `data`.
* `decode_tiles` is refactored to invoke `tile_payload_sizes` for
  the prefix walk, so the two §6.4 entry points share a single
  byte-walk implementation. The per-tile slice fetch in
  `decode_tiles` can now trust that every tile body is in-bounds,
  eliminating the duplicate range-check from the per-tile loop.
* Error surface mirrors the §6.4 conformance constraints:
  `Error::UnexpectedEof` when a non-last tile's 4-byte prefix runs
  past the end of `data` or when a declared `tile_size` extends past
  the available byte slice; `Error::InvalidBitstream` when a
  declared `tile_size + 4` would underflow the running `sz` budget
  per §6.4 line 2311.

Validation (+5 lib tests, lib total 464 -> 469; suite total 494 ->
499):

* Single-tile pass-through: `tile_rows_log2 = tile_cols_log2 = 0`
  returns `vec![sz]` with no `f(32)` read per §6.4 lines 2306-2308.
* Two-tile horizontal layout matching the
  `docs/video/vp9/fixtures/tile-cols-2` per-frame trace (`tile_size
  = 662` for the first tile, `tile_size = 635` for the last; total
  budget 4 + 662 + 635 = 1301 bytes).
* 2x2 grid (`tile_rows_log2 = tile_cols_log2 = 1`) emits four
  distinguishable sizes in row-major order — a transpose would
  surface here.
* 3-byte input rejected with `UnexpectedEof` at the first `f(32)`
  prefix per §6.4 line 2310.
* Declared `tile_size = u32::MAX` rejected with `InvalidBitstream`
  at the §6.4 line 2311 underflow rather than wrapping.

`pub use partition::tile_payload_sizes;` exposes the helper on the
crate root alongside `parse_uncompressed_header` /
`parse_compressed_header` / `parse_compressed_header_inter`.

Out of scope for round 36: a public `decode_tiles` wrapper that
runs the inner §6.4.2 / §6.4.3 walk against a real
fixture — `decode_partition` currently only emits the partition
tree, so against a real keyframe payload it would consume the
partition bits but leave the mode / tx / coefficient bits
unconsumed; that needs §6.4.4 `decode_block( )` wired into the
recursive driver first.

## Status — 2026-06-04 (round 35)

**Round 35: §6.3 `parse_compressed_header_inter` integration-test
coverage.** Round 35 pins the round-34 inter outer-dispatch entry
point at the public-API boundary by adding ten new integration tests
to `tests/compressed_header.rs` against
[`parse_compressed_header_inter`] per spec `vp9-spec.txt` lines
1957-1975:

* Zero-buffer default-table pass-through across the full §6.3.1
  ..§6.3.16 chain — every §10 / §10.5 default table survives a
  zero-filled compressed-header payload (every `B(252)` flag and
  every §6.3.7 outer `L(1) update_probs` flag decodes to 0, so each
  primitive returns its input unchanged). Anchors: `default_coef_probs`
  TX_4X4/intra/band-0/ctx-0 = `{195, 29, 183}` and
  TX_32X32/inter/band-5/ctx-5 = `{1, 16, 6}`; `default_skip_prob =
  {192, 128, 64}`; `default_tx_probs` rows 1..3 from §10;
  `default_is_inter_prob = {9, 102, 187, 225}`;
  `default_inter_mode_probs[ 0 ] = {2, 173, 34}`;
  `default_mv_class0_hp_prob = {160, 160}`;
  `default_mv_hp_prob = {128, 128}`; `MvProbs::defaults( )`.
* §6.3.10 `interpolation_filter == SWITCHABLE` gate: with the gate
  off, the 8-cell `read_interp_filter_probs( )` sweep is skipped
  entirely; downstream `is_inter_prob` / `mv_probs` results are
  bit-identical across gate states on a zero buffer (cursor shift
  doesn't matter when every flag is 0).
* §6.3.12 `compoundReferenceAllowed` paths: with `LAST = GOLDEN =
  ALTREF = 0` the walker takes the short-circuit arm returning
  `SingleReference` with no bool reads and no compound config; with
  `LAST = 0, GOLDEN = 0, ALTREF = 1` the bool-coder-reading arm runs
  one `L(1) non_single_reference` = 0 on a zero buffer, still
  yielding `SingleReference` (this time via the `L(1)`-driven path,
  not the short-circuit) without firing §6.3.18.
* §6.3.16 `allow_high_precision_mv` tail gate: the 4-cell
  high-precision tail is gated correctly — both `class0_hp_prob` and
  `hp_prob` stay at their §10.5 defaults across gate states on a
  zero buffer.
* Intra-prefix parity: the inter walker's §6.3.1 / §6.3.2 / §6.3.7
  / §6.3.8 prefix is bit-identical to `parse_compressed_header` on
  identical input for both lossless (no `L(2)` read in §6.3.1) and
  non-lossless (`L(2)` reads zero → `ONLY_4X4`) paths.
* Shared-error surface: empty buffer and non-zero §9.2.1 marker bit
  (first byte `0xFF`) both raise the same `Error` variant the intra
  walker raises — `init_bool` is the shared first step.
* `RefFrameSignBias::from_inter_biases` / `get` public-surface
  round-trip across all eight §6.2.5 input tuples (`LAST` /
  `GOLDEN` / `ALTREF` ∈ {0, 1}); the §3 `INTRA_FRAME` slot stays at
  0 (never populated by §6.2.5).

Validation (+10 integration tests, suite total 484 → 494; lib total
unchanged at 464).

Out of scope for round 35: wiring `parse_compressed_header_inter`
into [`decode_vp9`] — the uncompressed-header walker still rejects
inter frames with `Error::Unsupported` (`frame_size_with_refs` +
reference-buffer state are required first). The §6.3 listing is now
fully covered by both unit tests (round 34) and integration tests
(round 35) at the public-API boundary.

## Status — 2026-06-04 (round 34)

**Round 34: §6.3 `if ( FrameIsIntra == 0 )` outer dispatch —
[`parse_compressed_header_inter`] entry point composing the
round-22..30 inter-only primitives.** Round 34 wires the inter-frame
arm of the §6.3 compressed-header listing (`vp9-spec.txt` lines
1964-1974) into a new public entry point that walks the full §6.3
listing on inter frames:

* New public function `parse_compressed_header_inter(data, lossless,
  inputs)` per §6.3 (`vp9-spec.txt` lines 1957-1975). Runs the
  intra-shared prefix (§6.3.1 / §6.3.2 / §6.3.7 / §6.3.8) bit-for-bit
  identically to [`parse_compressed_header`] via a newly-extracted
  crate-local helper, then walks the inter-only tail in spec order:
  §6.3.9 `read_inter_mode_probs( )` →
  §6.3.10 `read_interp_filter_probs( )` gated on
  `interpolation_filter == SWITCHABLE` →
  §6.3.11 `read_is_inter_probs( )` →
  §6.3.12 `frame_reference_mode( )` (which fires
  §6.3.18 `setup_compound_reference_mode( )` on the non-`SingleReference`
  arms) →
  §6.3.13 `frame_reference_mode_probs( )` (the conditional 5 / 10 / 20
  cell sweep keyed by `reference_mode`) →
  §6.3.14 `read_y_mode_probs( )` →
  §6.3.15 `read_partition_probs( )` →
  §6.3.16 `mv_probs( )` (which fires
  §6.3.17 `update_mv_prob( )` per cell and walks the high-precision
  tail when `allow_high_precision_mv == 1`).
* New public type `Vp9CompressedHeaderInterInputs { interpolation_filter_is_switchable,
  ref_frame_sign_bias, allow_high_precision_mv }` bundling the three
  §6.2-derived flags the inter tail needs from the uncompressed-header
  walker.
* New public result `Vp9CompressedHeaderInter { intra,
  inter_mode_probs[ 7 ][ 3 ], interp_filter_probs[ 4 ][ 2 ],
  is_inter_prob[ 4 ], reference_mode, compound_reference_config,
  comp_mode_prob[ 5 ], single_ref_prob[ 5 ][ 2 ], comp_ref_prob[ 5 ],
  y_mode_probs[ 4 ][ 9 ], partition_probs[ 16 ][ 3 ], mv_probs }`
  bundling the post-§6.3.16 state of every inter-only probability
  table plus the §6.3.12 frame-level decision and (when compound is
  active) the §6.3.18 fixed-vs-variable ref-frame partition.
* `RefFrameSignBias`, `ReferenceMode`, `CompoundReferenceConfig`,
  `MvProbs` promoted from `pub(crate)` to `pub` since they surface
  in `Vp9CompressedHeaderInter` / `Vp9CompressedHeaderInterInputs`.

Validation (+11 lib tests, lib total 453 → 464; suite total 473 →
484): every §10 / §10.5 default table survives a zero-buffer walk
(every `B(252)` flag decodes to 0 → all primitives pass-through);
the inter-walker's intra-shared prefix is bit-identical to
[`parse_compressed_header`] on the same buffer for both lossless and
non-lossless paths; the §6.3.10 gate skips the walker when
`interpolation_filter != SWITCHABLE`; the §6.3.12
`compoundReferenceAllowed == 0` short-circuit (sign-bias tuple
`(0,0,0)`) returns `SingleReference` with no compound config and no
bool-coder reads, vs. the mixed-bias path that consumes the
non_single_reference flag; the §6.3.16 high-precision tail is gated
on `allow_high_precision_mv` (a zero-buffer walk leaves the four HP
slots at their §10.5 defaults in both states, isolating gate
behaviour from value-update); empty input surfaces the same
`InvalidBitstream` error as the intra walker (`init_bool` rejects);
the full composed walk is bit-identical to an explicit independent
hand-walk against every §6.3.x primitive in spec order; and
`RefFrameSignBias::from_inter_biases` / `get` round-trips across all
eight sign-bias tuples.

Out of scope for round 34: wiring `parse_compressed_header_inter`
into [`decode_vp9`] — the uncompressed-header walker still rejects
inter frames with `Error::Unsupported` (`frame_size_with_refs` +
reference-buffer state are required first). The round-34 entry
point is callable directly by integrators that have the §6.2-derived
flags from another source. The §6.3.18 `setup_compound_reference_mode( )`
output is surfaced on the non-`SingleReference` arms but the
downstream §6.4.16 / §6.4.18 / §6.5 consumers still need to land.

## Status — 2026-06-03 (round 33)

**Round 33: §6.4 `decode_tiles( )` outer driver — frame-level tile
walk.** Round 33 composes the round-32 §6.4.1 / §6.4.2 primitives
into the full `(1 << tile_rows_log2) × (1 << tile_cols_log2)` frame
walk per `vp9-spec.txt` lines 2300-2331:

* `decode_tiles( data, sz, tile_rows_log2, tile_cols_log2, mi_rows,
  mi_cols, ctx_state, probs_kind )` per §6.4. Phases:
  * **Phase 1 — frame reset** (line 2303): fires
    `PartitionContextState::clear_above( )` (the §7.4.1
    `clear_above_context( )` reset) once before any tile walks.
  * **Phase 2 — tile-grid walk** (lines 2304-2330): iterates
    `(tileRow, tileCol)` in row-major order across `tileRows × tileCols
    = (1 << tile_rows_log2) × (1 << tile_cols_log2)` cells. For
    every tile except `lastTile = (tileRow == tileRows - 1) &&
    (tileCol == tileCols - 1)` it reads `tile_size  f(32)`
    (big-endian) from the byte stream and runs `sz -= tile_size + 4`
    per spec line 2311; the last tile assigns `tile_size = sz`. Per
    tile it derives `MiRowStart` / `MiRowEnd` / `MiColStart` /
    `MiColEnd` via the §6.4.1 helper, brackets a fresh `BoolCoder`
    with `init_bool( tile_size ) / exit_bool( )` per §9.2.1 /
    §9.2.3, and invokes the §6.4.2 `decode_tile( )` primitive.
* `DecodedTile { tile_row, tile_col, mi_row_start, mi_row_end,
  mi_col_start, mi_col_end, tile_size, leaves }` bundles the §6.4
  listing's four `get_tile_offset( )` outputs, the per-tile byte
  budget, and the §6.4.2 leaf log (in §6.4.3 traversal order). The
  `Vec<DecodedTile>` output is sized exactly `tileRows * tileCols`.
* `PartitionContextState::clear_above( )` per §7.4.1: the dual of
  the round-32 `clear_left( )` reset, zeroes
  `AbovePartitionContext[ ]` once per `decode_tiles( )` call. The
  §7.4.1 note observes the canonical span is `0..Sb64Cols * 8 - 1`
  (the array can be read past `MiCols`); callers wanting that span
  size the strip rounded up to the next multiple of 8 at
  `PartitionContextState::new( )`.

Bitstream-error surface (with explicit test coverage): §6.4 line
2310 underflow on the `f(32)` read raises `Error::UnexpectedEof`; a
declared `tile_size` whose `(+ 4)` would exceed the remaining `sz`
raises `Error::InvalidBitstream` (the spec's `sz -= tile_size + 4`
must not underflow); a declared `tile_size` larger than the
available byte stream raises `Error::UnexpectedEof`; per-tile
`init_bool( )` marker-rejection or `exit_bool( )` non-zero-padding
raises `Error::InvalidBitstream`.

Validation (+13 lib tests, lib total 440 → 453; suite total 460 →
473) covers: single-tile (`tile_rows_log2 == 0`, `tile_cols_log2 ==
0`) frame consuming the full payload via `lastTile = true` (no
`f(32)` prefix); §6.4 line 2303 `clear_above_context( )` zeroing a
pre-poisoned `above[ ]` strip BEFORE any `decode_tile( )` runs;
two-tile horizontal split reading exactly one `f(32)` prefix and
deriving `MiColStart` / `MiColEnd` matching the §6.4.1 split at
`(0, 8, 16)`; full 2×2 grid iterating `(0,0) → (0,1) → (1,0) →
(1,1)` with three `f(32)` prefixes and one `lastTile`; last-tile
explicitly skipping the `f(32)` prefix (back-to-back bodies with one
prefix only); output `Vec<DecodedTile>` length matching `tileRows *
tileCols` for a 1×4 strip plus contiguous `MiColEnd`/`MiColStart`
pairs; truncated 3-byte stream raising `UnexpectedEof` at the
`f(32)` fetch; oversized declared `tile_size = u32::MAX` raising
`InvalidBitstream` (the spec's `sz -= tile_size + 4` arithmetic);
truncated tile body (declared 8, supplied 6) raising
`UnexpectedEof`; a non-zero-marker first byte (`0x80`) raising
`InvalidBitstream` via the per-tile `init_bool( )`; 1×2 vertical
split partitioning MI rows at `(0, 8, 16)`; the cross-axis 2×2
invariant that consecutive tiles are contiguous within rows AND
within columns (each row's last `MiColEnd == MiCols`, each column's
last `MiRowEnd == MiRows`); and
`PartitionContextState::clear_above( )` zero-strip dual to the
round-32 `clear_left` invariant.

Out of scope for round 33: wiring `decode_tiles( )` into the
public `decode_vp9( )` entry point — the per-tile leaf log
(`DecodedTile::leaves`) still feeds the §6.4.4 `decode_block_apply`
driver from round 31 rather than the full §6.4.5 `mode_info( )` +
§6.4.6 `residual( )` pipeline; the public API still exposes
`parse_uncompressed_header`, `parse_compressed_header` and their
result types exclusively. The round-33 surface stays internal-only
(`pub(crate)` on `decode_tiles` / `DecodedTile`).

## Status — 2026-06-03 (round 32)

**Round 32: §6.4.1 `get_tile_offset( )` + §6.4.2 `decode_tile( )` —
tile-driver primitive layer.** Round 32 lifts the §6.4.3
[`partition::decode_partition`] driver landed in round 19 into the
§6.4.2 tile-row driver and the §6.4.1 per-tile-offset arithmetic that
§6.4 `decode_tiles( )` will compose them with:

* `get_tile_offset( tile_num, mis, tile_sz_log2 )` per §6.4.1
  (`vp9-spec.txt` lines 2335-2338). Three lines:
  `sbs = (mis + 7) >> 3` (round-up to sb64-cell count),
  `offset = ((tile_num * sbs) >> tile_sz_log2) << 3` (per-tile-axis
  MI-cell offset, 8-aligned to the sb64 boundary),
  `return Min( offset, mis )` (clamp the past-the-end fetch the §6.4
  caller fires for `tileNum = tilesPerAxis` against the frame extent).
  Pure u32 arithmetic — the §6.4 outer driver invokes it four times
  per tile to derive `MiRowStart` / `MiRowEnd` / `MiColStart` /
  `MiColEnd` from `MiRows` + `tile_rows_log2` and `MiCols` +
  `tile_cols_log2`.

* `decode_tile( coder, mi_row_start, mi_row_end, mi_col_start,
  mi_col_end, mi_rows, mi_cols, ctx_state, probs_kind, leaves )` per
  §6.4.2 (`vp9-spec.txt` lines 2343-2349). Two-deep loop:
  * outer `r ∈ [ mi_row_start, mi_row_end )` step 8, firing
    `PartitionContextState::clear_left( )` (the §7.4.2
    `clear_left_context( )` reset) once per superblock-row start;
  * inner `c ∈ [ mi_col_start, mi_col_end )` step 8, firing
    [`decode_partition`]`( r, c, BLOCK_64X64, mi_rows, mi_cols, ... )`
    once per superblock origin.
  The §6.4.3 driver's own `r >= mi_rows || c >= mi_cols`
  short-circuit absorbs tiles whose `End` offsets fall past the frame
  edge with no extra bookkeeping.

Validation (+12 lib tests, lib total 428 → 440; suite total 448 →
460) covers: §6.4.1 single-tile (`tile_sz_log2 == 0`) cases for both
sb64-aligned (`mis = 8`) and non-aligned (`mis = 11`) frame extents,
including the past-end clamp; the §6.4.1 two-tile case
(`tile_sz_log2 == 1`, `mis = 16`) producing `(0, 8, 16)`; the
`Min( offset, mis )` clamp on `tile_sz_log2 == 2` with
`mis = 8`; a consecutive-pair `(i, i+1)` cover proof that
`get_tile_offset( i+1, mis, log2 ) >= get_tile_offset( i, mis, log2 )`
and the last `End` equals `mis` (the §6.4 outer-driver invariant);
an 8-alignment sweep across `mis ∈ {8, 16, 32, 64, 256}` and
`tile_sz_log2 ∈ {0, 1, 2, 3}`; the §6.4.2 empty-window
(`mi_row_start == mi_row_end`) early-return preserving the above
strip; the §6.4.2 single-sb64 tile producing one `(0, 0, BLOCK_64X64)`
leaf; the §6.4.2 two-sb-wide row producing leaves at `(0, 0)` then
`(0, 8)` in order; the §6.4.2 two-sb-tall column producing leaves at
`(0, 0)` then `(8, 0)` and proof that `clear_left_context( )` fires
at the START of the second row (pre-poisoned `left[ ]` sentinel does
NOT leak into the second-row partition_decode ctx); the §6.4.2 2×2
sb64 row-major traversal order `(0,0) → (0,8) → (8,0) → (8,8)`; the
§6.4.2 sub-tile MI window starting at `(mi_row_start = 8,
mi_col_start = 8)` producing one `(8, 8)` leaf; and the §6.4.1 +
§6.4.2 composition splitting a 16-MI-wide frame into two tiles and
decoding each tile's single superblock with the matching `c` offset.

Out of scope for round 32: the §6.4 `decode_tiles( )` outer driver
(reads `tile_size` as `f(32)` between tiles, fires `init_bool( ) /
exit_bool( )` per tile, calls `clear_above_context( )` once per
frame, walks all `(1 << tile_rows_log2) × (1 << tile_cols_log2)`
tiles), and wiring the §6.4.4 [`decode_block::decode_block_apply`]
fan-out into the per-leaf log site inside [`decode_partition`] — the
leaf log already carries `(r, c, subsize)` but the swap needs a
frame-state allocator plus per-leaf §6.4.5 `mode_info( )`
invocation. The round-32 surface stays internal-only (`pub(crate)`
with `#[allow(dead_code)]` on `get_tile_offset` / `decode_tile`);
the public API still exposes `parse_uncompressed_header`,
`parse_compressed_header` and their result types exclusively.

## Status — 2026-06-02 (round 31)

**Round 31: §6.4.4 `decode_block( r, c, subsize )` driver — pure-state
fan-out primitive.** Round 31 lands the §6.4.4 per-leaf driver as a
standalone book-keeping primitive that consumes the per-MI outputs of
`mode_info( )` and `residual( )` (decoded by the §6.4.5 / §6.4.6 /
§6.4.15 / §6.4.21 primitives landed in earlier rounds) and fans them
into the frame-wide §6.4.4 arrays at every `(r + y, c + x)` cell for
`y ∈ 0..num_8x8_blocks_high_lookup[ subsize ]`,
`x ∈ 0..num_8x8_blocks_wide_lookup[ subsize ]`:

* `decode_block_apply( state, r, c, subsize, result )` per §6.4.4
  (`vp9-spec.txt` lines 2395-2437). Two phases:
  * **Phase 1 — `skip` rewrite** (lines 2405-2407): if
    `is_inter && subsize >= BLOCK_8X8 && EobTotal == 0` set
    `skip = 1`. Returns the rewritten `skip` value so a §8.4
    probability-adaption sink can consume it.
  * **Phase 2 — fan-out** (lines 2408-2436): for each `(y, x)` step
    of the `num_8x8_blocks_*_lookup[ subsize ]` grid, write the ten
    cells `Skips`, `TxSizes`, `MiSizes`, `YModes`, `SegmentIds`,
    `RefFrames[ 0..2 ]`, `InterpFilters` (inter only),
    `Mvs[ 0..2 ]` = `BlockMvs[ refList ][ 3 ]` (inter only),
    `SubMvs[ 0..2 ][ 0..4 ]` (inter only), and
    `SubModes[ 0..4 ] = sub_modes[ ]` (intra only).
* `DecodedBlockResult { skip, tx_size, y_mode, segment_id,
  ref_frame[ 2 ], is_inter, eob_total, interp_filter,
  block_mvs[ 2 ][ 4 ], sub_modes[ 4 ] }` bundles the per-MI values
  the §6.4.5 / §6.4.6 / §6.4.15 / §6.4.21 primitives produce. The
  `Default` impl seeds `ref_frame = [INTRA_FRAME = 0, NONE = -1]` per
  §6.4.6 lines 2469-2470 (intra-block init).
* `Vp9FrameState { mi_cols, mi_rows, skips, tx_sizes, mi_sizes,
  y_modes, segment_ids, ref_frames, interp_filters, mvs, sub_mvs,
  sub_modes }` owns the `MiRows × MiCols` (and `× 2` / `× 4` for the
  per-`refList` / per-sub-block strides) §6.4.4 write-back arrays in
  row-major order, with `get_skip` / `get_tx_size` / `get_mi_size`
  / `get_y_mode` / `get_segment_id` / `get_ref_frame` /
  `get_interp_filter` / `get_mv` / `get_sub_mv` / `get_sub_mode`
  accessors returning `Option<T>` for out-of-frame coordinates.

Validation (+13 lib tests, lib total 415 → 428; suite total 435 →
448) covers: §6.4.4 intra-default top-left-cell write (skip = 0,
`ref_frame = [INTRA, NONE]`); §6.4.4 BLOCK_8X8 single-cell write
with `sub_modes[ 4 ]` propagation and untouched neighbours; §6.4.4
BLOCK_16X16 2×2 fan-out propagating `MiSize` / `y_mode` / `segment_id`
/ `tx_size` into all four cells; §6.4.4 BLOCK_64X64 full-8×8-MI-frame
fan-out (64 cells); §6.4.4 lines 2405-2407 `skip = 1` rewrite under
`is_inter ∧ subsize ≥ BLOCK_8X8 ∧ EobTotal = 0`, plus all three
non-firing cases (sub-8×8 / `EobTotal > 0` / `is_inter = 0`); inter
branch writing `InterpFilters` + `Mvs[ refList ] = BlockMvs[ refList
][ 3 ]` + `SubMvs[ refList ][ b ] = BlockMvs[ refList ][ b ]` (all
16 cells of a 16x16 fan-out × 2 refLists × 4 sub-blocks); §6.4.4
line 2416 `ref_frame[ 0..2 ]` write on both branches; §7.4.3
out-of-frame clip on a 32×32 block straddling the edge of an 8×8
MI frame; `skip = 1` rewrite propagating into every cell of the
fan-out; and §10.2 `num_8x8_blocks_*_lookup[ ]` table pinning so the
fan-out cell-counts stay anchored to the spec values.

Out of scope for round 31: wiring `decode_block_apply` into the
§6.4.3 [`partition::decode_partition`] driver — i.e. invoking it at
every `LeafBlock` log site rather than only logging the leaf. The
swap is mechanical (the leaf log already carries `(r, c, subsize)`)
but requires a frame-state allocator + per-leaf §6.4.5 `mode_info( )`
invocation which sits outside the §6.4.4 scope. The round-31 surface
stays internal-only (`pub(crate)` with `#[allow(dead_code)]` on
`DecodedBlockResult` / `Vp9FrameState` / `decode_block_apply`); the
public API still exposes `parse_uncompressed_header`,
`parse_compressed_header` and their result types exclusively.

## Status — 2026-06-01 (round 30)

**Round 30: §6.3.16 `mv_probs( )` compressed-header outer sweep.** Round
30 closes the §6.3.x primitives chain by landing the final outer driver
— the 65/69-cell MV-probability walk that consumes the §6.3.17
[`update_mv_prob`] primitive across nine `mv_*_prob[ ]` arrays:

* `mv_probs( coder, probs, allow_high_precision_mv )` per §6.3.16
  (`vp9-spec.txt` lines 2234-2259). Three unconditional phases plus
  one conditional tail:
  * **Phase 1 — joint probs** (3 cells): walks
    `mv_joint_probs[ MV_JOINTS - 1 = 3 ]`.
  * **Phase 2 — per-component bulk** (44 cells = 2 × 22): per
    `i ∈ {0, 1}`, walks `sign_prob[ i ]` (1) +
    `class_probs[ i ][ MV_CLASSES - 1 = 10 ]` (10) +
    `class0_bit_prob[ i ]` (1) +
    `bits_prob[ i ][ MV_OFFSET_BITS = 10 ]` (10).
  * **Phase 3 — per-component fractional** (18 cells = 2 × 9): per
    `i ∈ {0, 1}`, walks
    `class0_fr_probs[ i ][ CLASS0_SIZE = 2 ][ MV_FR_SIZE - 1 = 3 ]`
    (6) + `fr_probs[ i ][ MV_FR_SIZE - 1 = 3 ]` (3).
  * **Phase 4 (conditional) — high-precision tail** (4 cells, gated on
    `allow_high_precision_mv == 1`): per `i ∈ {0, 1}`, walks
    `class0_hp_prob[ i ]` (1) + `hp_prob[ i ]` (1).

  Total cell count = **65 cells** when `allow_high_precision_mv == 0`,
  **69 cells** when `1`. Every cell consumes one `B(252)`
  `update_mv_prob` flag (and seven extra `L(7)` bits on flag-set).

* `MvProbs { joint_probs, sign_prob, class_probs, class0_bit_prob,
  bits_prob, class0_fr_probs, fr_probs, class0_hp_prob, hp_prob }`
  bundles the nine arrays as a single mutable target for the sweep;
  `MvProbs::defaults()` seeds every slot from the §10.5 listings (the
  newly-transcribed `DEFAULT_MV_JOINT_PROBS` /
  `DEFAULT_MV_SIGN_PROB` / `DEFAULT_MV_CLASS_PROBS` /
  `DEFAULT_MV_CLASS0_BIT_PROB` / `DEFAULT_MV_BITS_PROB` /
  `DEFAULT_MV_CLASS0_FR_PROBS` / `DEFAULT_MV_FR_PROBS` /
  `DEFAULT_MV_CLASS0_HP_PROB` / `DEFAULT_MV_HP_PROB` constants in
  `mode_info.rs`).

* §3 MV-constants transcribed verbatim into `mode_info.rs`:
  `MV_JOINTS = 4` (line 508), `MV_CLASSES = 11` (line 509),
  `CLASS0_SIZE = 2` (line 510), `MV_OFFSET_BITS = 10` (line 511),
  `MV_FR_SIZE = 4` (line 458). These size the §6.5 MV-tree decoders
  as well as the §6.3.16 sweep.

Validation (+13 lib tests, lib total 402 → 415; suite total 422 →
435) covers: cell-count constants (3 + 44 + 18 = 65 unconditional,
+4 HP tail); verbatim transcription of the nine §10.5 default
tables against the spec listing; zero-buffer pass-through on both
`allow_high_precision_mv ∈ {false, true}` defaulting-bundle and
custom-starts paths; cursor-equivalence proofs that the no-HP sweep
consumes exactly 65 `B(252)` flags and the with-HP sweep consumes
exactly 69; a four-flag-catch-up cursor proof that the HP tail
contributes exactly 4 cells of bool-coder delta; explicit phase-walk
equivalence against a hand-coded §6.3.16 listing walker (two starting
bundles × `hp ∈ {false, true}`); HP-field preservation under
`allow_high_precision_mv == false`; §3 constant pinning; and a
defaults-vs-`mode_info` single-source-of-truth audit.

Out of scope for round 30: wiring `mv_probs( )` into the
`parse_compressed_header` outer dispatch — the call site is gated on
`FrameIsIntra == 0` and needs the `allow_high_precision_mv` flag from
§6.2.5, which the uncompressed-header walker still rejects with
`Error::Unsupported`. The round-30 surface stays internal-only
(`pub(crate)` with `#[allow(dead_code)]` on the function + struct);
the public API still exposes `parse_uncompressed_header`,
`parse_compressed_header` and their result types exclusively. The
§6.3.x primitives chain is now complete (§6.3.1 → §6.3.18 inclusive)
modulo wiring into the outer dispatch.

## Status — 2026-06-01 (round 29)

**Round 29: §6.3.12 `frame_reference_mode( )` compressed-header outer
driver.** Round 29 lands the two-`L(1)` outer driver that gates the
§6.3.18 [`setup_compound_reference_mode`] caller and decides the
frame-level `reference_mode` from the §6.2.5
`ref_frame_sign_bias[ ]` array — closing the §6.3.x driver chain
modulo §6.3.16 `mv_probs( )`:

* `frame_reference_mode( coder, ref_frame_sign_bias )` per §6.3.12
  (`vp9-spec.txt` lines 2170-2191). Computes
  `compoundReferenceAllowed` from the §3 loop
  `for ( i = 1; i < REFS_PER_FRAME; i++ ) if (
  ref_frame_sign_bias[ i + 1 ] != ref_frame_sign_bias[ 1 ] )`. With
  `REFS_PER_FRAME = 3` the loop iterates `i = 1, 2` and compares
  `ref_frame_sign_bias[ GOLDEN_FRAME = 2 ]` and
  `ref_frame_sign_bias[ ALTREF_FRAME = 3 ]` against
  `ref_frame_sign_bias[ LAST_FRAME = 1 ]`. The all-agree sign-bias
  tuples `(0, 0, 0)` and `(1, 1, 1)` short-circuit to
  `SingleReference` with zero bool-coder reads; the other six
  tuples enter the compound-allowed arm.
* On the allowed arm: `L(1) non_single_reference`. On 0 →
  `SingleReference` (one bit consumed total). On 1 → `L(1)
  reference_select`; 0 → `CompoundReference`, 1 →
  `ReferenceModeSelect`. Both 1-arms invoke the §6.3.18
  [`setup_compound_reference_mode`] partitioner.
* `REFS_PER_FRAME = 3` constant transcribed verbatim from §3
  (`vp9-spec.txt` line 457) into `mode_info.rs`.
* Returns `(ReferenceMode, Option<CompoundReferenceConfig>)`:
  `SingleReference` → `None` (no compound machinery active);
  `CompoundReference` / `ReferenceModeSelect` → `Some(cfg)` with the
  §6.3.18 partition of `{LAST, GOLDEN, ALTREF}` into
  `(CompFixedRef, CompVarRef[ 2 ])` directly consumable by §6.4.16
  `inter_block_mode_info( )` `comp_ref` per-block decode and §6.5
  MV-reference search.

Validation (+11 lib tests, lib total 391 → 402; suite total 411 →
422) covers: the all-agree short-circuit on `(0, 0, 0)` and
`(1, 1, 1)` (no compound config, no bool-coder reads); a cursor-
equivalence proof that the all-agree arm consumes exactly zero
`L(1)` reads against a parallel-coder walker; the six
compound-allowed tuples reading exactly one `L(1)` on the
`non_single_reference == 0` zero-buffer path (SingleReference with
no compound config); brute-forced 16-bit prefix-space searches for
buffers producing `(L(1)=1, L(1)=0)` and `(L(1)=1, L(1)=1)` to
exercise the CompoundReference and ReferenceModeSelect arms;
cursor-equivalence proofs that each two-`L(1)` arm consumes exactly
two bool-coder reads; a cross-check confirming the returned
`CompoundReferenceConfig` matches the §6.3.18
[`setup_compound_reference_mode`] output on the same bias bundle;
an exhaustive 8-tuple allowed-vs-not predicate match against the
inline §6.3.12 loop; and a step-walk equivalence asserting the
production code matches an independently re-derived listing walker
across all 32 (sign-bias × buffer) combinations.

Out of scope for round 29: wiring `frame_reference_mode( )` into
the `parse_compressed_header` outer dispatch — the call site is
gated on `FrameIsIntra == 0` and needs the
`ref_frame_sign_bias[ ]` array sourced from §6.2.5, which the
uncompressed-header walker still rejects with `Error::Unsupported`.
The round-29 surface stays internal-only (`pub(crate)` with
`#[allow(dead_code)]` on the function); the public API still
exposes `parse_uncompressed_header`, `parse_compressed_header` and
their result types exclusively. §6.3.16 `mv_probs( )` remains
deferred (needs §3 MV constants + §10.5 default MV tables + the
`allow_high_precision_mv` flag from §6.2.5).

## Status — 2026-05-31 (round 28)

**Round 28: §6.3.18 `setup_compound_reference_mode( )` compressed-header
pure-compute leaf.** Round 28 closes the §6.3.x primitives chain modulo
the still-deferred §6.3.12 `frame_reference_mode( )` and §6.3.16
`mv_probs( )` outer drivers, landing the final §6.3.x leaf — a pure
compute function that takes no bool-coder reads:

* `setup_compound_reference_mode( ref_frame_sign_bias )` per §6.3.18
  (`vp9-spec.txt` lines 2279-2296). Partitions the three §3 inter
  reference frames (`LAST_FRAME = 1`, `GOLDEN_FRAME = 2`,
  `ALTREF_FRAME = 3`) into a `CompFixedRef` plus `CompVarRef[ 0 ]` /
  `CompVarRef[ 1 ]` pair, based on the §6.2.5 `ref_frame_sign_bias[ ]`
  `f(1)` flags. The §6.3.18 listing has three branches:
  * Branch 1 (`LAST == GOLDEN`): `fixed = ALTREF`; var = `{LAST,
    GOLDEN}`. Fires on 4 of the 8 sign-bias tuples — including the
    all-agree cases `(0,0,0)` and `(1,1,1)`, where branch 1 takes
    precedence over branch 2 by the listing's if/else order.
  * Branch 2 (`LAST != GOLDEN AND LAST == ALTREF`): `fixed = GOLDEN`;
    var = `{LAST, ALTREF}`. Fires on 2 of 8 tuples.
  * Branch 3 (else: `LAST != GOLDEN AND LAST != ALTREF`, which
    implies `GOLDEN == ALTREF`): `fixed = LAST`; var = `{GOLDEN,
    ALTREF}`. Fires on the remaining 2 of 8 tuples.
* §3 ref-frame enumeration transcribed verbatim into `mode_info.rs`
  alongside the existing `INTRA_FRAME = 0`: `LAST_FRAME = 1`,
  `GOLDEN_FRAME = 2`, `ALTREF_FRAME = 3`, `MAX_REF_FRAMES = 4` (spec
  line 470). The four sentinels match the §7.4.12 `ref_frame[ 0 ]` /
  `ref_frame[ 1 ]` enumeration tables (lines 3990-4006).
* `RefFrameSignBias` newtype around `[u8; MAX_REF_FRAMES]` enforces
  the §6.2.5 "inter slots only" populated-by-`f(1)` invariant via a
  `from_inter_biases(last, golden, altref)` constructor with debug
  assertions on the two-state range. The `INTRA_FRAME` slot is held
  at zero internally (§6.3.18 never reads it).
* `CompoundReferenceConfig { fixed_ref: i32, var_ref: [i32; 2] }`
  bundles the §6.3.18 output for downstream §6.4.16
  `inter_block_mode_info( )` `comp_ref` per-block decode and §6.5
  MV-reference search consumption.

Validation (+9 lib tests, lib total 382 → 391; suite total 402 →
411) covers: §3 sentinel pinning (`LAST_FRAME = 1`, `GOLDEN_FRAME =
2`, `ALTREF_FRAME = 3`, `MAX_REF_FRAMES = 4`, monotonic order via
`const { assert!(..) }`); each of the three §6.3.18 branches against
its prescribed tuple set (branch 1 on 4 tuples, branch 2 on 2,
branch 3 on 2); an exhaustive 8-tuple truth-table sweep
cross-checking every input maps to the expected
`(fixed_ref, var_ref[0], var_ref[1])` triple and branch ID; explicit
precedence pinning that the all-agree `(0,0,0)` / `(1,1,1)` tuples
fire branch 1 (ALTREF fixed) not branch 2 (GOLDEN fixed); a
pairwise-distinct-and-permutation-of-inter-set invariant proving
every output is a permutation of `{LAST_FRAME, GOLDEN_FRAME,
ALTREF_FRAME}` (never collapsing or mixing in INTRA_FRAME); the
inter-only-population invariant on `RefFrameSignBias`; and a
pure-compute / type-level signature pin (no `BoolCoder` parameter,
function is reentrant on the same input).

Out of scope for round 28: §6.3.12 `frame_reference_mode( )` and
§6.3.16 `mv_probs( )` outer drivers — both need `ref_frame_sign_bias[
]` state the §6.2.5 uncompressed-header walker still rejects with
`Error::Unsupported`, plus (for §6.3.16) the `allow_high_precision_mv`
flag. The round-28 surface stays internal-only (`pub(crate)` with
`#[allow(dead_code)]` on the function, struct, and helper); the
public API still exposes `parse_uncompressed_header`,
`parse_compressed_header` and their result types exclusively.

## Status — 2026-05-29 (round 27)

**Round 27: §6.3.17 `update_mv_prob( prob )` compressed-header per-cell
primitive.** Round 27 extends the §6.3 inter-arm primitives chain by
the per-call MV-probability-update helper that the still-deferred
§6.3.16 `mv_probs( )` sweep will call once per cell. The primitive is
a stand-alone leaf (no per-cell cascade like §6.3.3
`read_diff_update_prob`):

* `update_mv_prob( coder, prob )` per §6.3.17 (`vp9-spec.txt` lines
  2261-2275). Reads one `B(252)` `update_mv_prob` flag and, on 1,
  pulls a 7-bit `L(7)` `mv_prob` literal and rewrites
  `prob = (mv_prob << 1) | 1`. Otherwise returns the caller's `prob`
  unchanged. The `<< 1 | 1` rewrite forces odd parity and the
  `[1, 255]` step-2 range — MV probabilities can't be 0 because
  §6.5.x MV tree decode treats 0 as an unconditional branch.
* Distinct from the §6.3.3 `read_diff_update_prob` chain consumed by
  every other §6.3 probability sweep (rounds 4..26). That chain uses
  `decode_term_subexp` (§6.3.4) + `inv_remap_prob` (§6.3.5) and the
  output depends on the previous probability. The §6.3.17 primitive
  ignores the input `prob` entirely on the flag-set branch and
  computes a fresh value purely from the 7-bit literal — the
  `B(252)` + `L(7)` flag-set path is 8 bits of bool-coder state.
* The §6.3.16 caller — still deferred because §6.3.12 needs
  `ref_frame_sign_bias[ ]` state the uncompressed-header walker still
  rejects with `Error::Unsupported` — walks `MV_JOINTS - 1 = 3` joint
  slots, per-component `MV_CLASSES - 1 = 10` class slots + 1
  `class0-bit` slot + `MV_OFFSET_BITS = 10` bits slots,
  per-component-per-`class0` `MV_FR_SIZE - 1 = 3` fr slots + a global
  `MV_FR_SIZE - 1 = 3` fr slot, and (when `allow_high_precision_mv ==
  1`) per-component `class0-hp` + hp slots — 66 or 70 cells per frame
  depending on the high-precision-MV flag. Each cell calls
  `update_mv_prob( )` once.

Validation (+8 lib tests, lib total 374 → 382; suite total 394 → 402)
covers: zero-buffer pass-through preserving each base in
`{0, 1, 7, 64, 127, 128, 129, 200, 254, 255}`; a cursor-equivalence
proof that the zero-buffer fast path consumes exactly one `B(252)`
flag against a parallel-coder walker; a brute-forced flag-set buffer
(deterministic 256-candidate search for the smallest first byte that
triggers `read_bool(252) == 1` after the §9.2.1 marker) producing a
deterministic output independent of input base; a cursor-equivalence
proof that the flag-set branch consumes exactly one `B(252)` + one
`L(7)` against a parallel-coder walker; a parity + range invariant
sweep across every L(7) value 0..=127 (`(literal << 1) | 1` always
odd and in `[1, 255]`); a baseline cross-check confirming the
flag-set output is input-prob-independent; a direct distinction test
against §6.3.3 `read_diff_update_prob` proving the two primitives are
not aliases (different outputs on the same flag-set buffer with the
same base); and an explicit step-walk equivalence against a hand-coded
§6.3.17 listing walker (zero buffer + flag-set buffer × 6 base values).

Out of scope for round 27: §6.3.16 `mv_probs( )` itself — that needs
the §3 MV constants (`MV_JOINTS = 4`, `MV_CLASSES = 11`,
`MV_OFFSET_BITS = 10`, `CLASS0_SIZE = 2`, `MV_FR_SIZE = 4`), the
§10.5 default MV-probability tables (`default_mv_joint_probs`,
`default_mv_sign_prob`, `default_mv_class_probs`,
`default_mv_class0_bit_prob`, `default_mv_bits_prob`,
`default_mv_class0_fr_probs`, `default_mv_fr_probs`,
`default_mv_class0_hp_prob`, `default_mv_hp_prob`), and the
`allow_high_precision_mv` flag from §6.2.5 which the
uncompressed-header walker still rejects with `Error::Unsupported`.
The round-27 surface stays internal-only (`pub(crate)` with
`#[allow(dead_code)]`); the public API still exposes
`parse_uncompressed_header`, `parse_compressed_header` and their
result types exclusively.

## Status — 2026-05-29 (round 26)

**Round 26: §6.3.15 `read_partition_probs( )` compressed-header
sweep.** Round 26 extends the §6.3 inter-arm primitives chain by the
unconditional `PARTITION_CONTEXTS × (PARTITION_TYPES - 1) = 16 × 3 = 48`
cell sweep that populates the running inter-frame `partition_probs[ ][ ]`
table, alongside the round-22..25 §6.3.9 / §6.3.10 / §6.3.11 / §6.3.13 /
§6.3.14 primitives:

* `read_partition_probs( coder, partition_probs )` per §6.3.15
  (`vp9-spec.txt` lines 2227-2232). 48 sequential
  `read_diff_update_prob` calls — one `B(252)` `update_prob` flag per
  cell and, on 1, a `decode_term_subexp( )` + `inv_remap_prob( )`
  cascade — updating `partition_probs[ ][ ]` in place.
* §3 constants `PARTITION_CONTEXTS = 16` (line 463) and
  `PARTITION_TYPES = 4` (line 497) reused from the round-18 `partition`
  module transcription. The §10.5 `default_partition_probs` table
  (lines 7623-7651; 16 rows of 3 columns each) was transcribed verbatim
  into `partition::DEFAULT_PARTITION_PROBS` in round 18.
* `DEFAULT_PARTITION_PROBS_TABLE` re-export in `compressed.rs` keeps
  `partition::DEFAULT_PARTITION_PROBS` as the single source of truth
  (mirroring the round-22..25 staging pattern of one re-export per
  sweep). Same constant feeds the §6.4.3 `decode_partition_type( )`
  per-call partition decoder on inter frames via the §9.3.2
  `partition_plane_context( )` ctx.

Validation (+9 lib tests, lib total 365 → 374; suite total 385 → 394)
covers: §3 constant pinning (`PARTITION_CONTEXTS = 16`,
`PARTITION_TYPES = 4`); verbatim §10.5 transcription of the
`default_partition_probs` table (16 × 3 layout with the
block-size-group / `(above, left)` split annotations preserved);
zero-buffer `update_prob = 0` pass-through preserving the starting
table; all-cells-visited check with a non-uniform custom starting
table; cursor-equivalence proof that the sweep consumes exactly 48
`B(252)` flags against a parallel-coder walker; row-major walk
equivalence against a parallel coder for two distinct starting
tables; a tuple-sweep across `{0, 1, 7, 64, 127, 128, 129, 200, 254,
255}` starting bases surviving zero-buffer pass-through; and a
single-source-of-truth check tying the `compressed.rs` re-export back
to the `partition::DEFAULT_PARTITION_PROBS` constant.

Out of scope for round 26: §6.3.12 `frame_reference_mode( )` /
§6.3.16 `mv_probs( )` / §6.3.17 — the partition-probs sweep lives
between §6.3.14 `read_y_mode_probs( )` and §6.3.16 `mv_probs( )` in
the §6.3 outer dispatch, but wiring any subset into
`parse_compressed_header` before §6.3.12 lands would mis-position the
coder cursor — and §6.3.12 needs `ref_frame_sign_bias[ ]` state the
uncompressed-header walker still rejects with `Error::Unsupported`.
The round-26 surface stays internal-only (`pub(crate)` with
`#[allow(dead_code)]` on the function + re-export const); the public
API still exposes `parse_uncompressed_header`,
`parse_compressed_header` and their result types exclusively.

## Status — 2026-05-27 (round 25)

**Round 25: §6.3.13 `frame_reference_mode_probs( )` compressed-header
sweep.** Round 25 extends the §6.3 inter-arm primitives chain by the
three reference-mode-gated sweeps over `comp_mode_prob`,
`single_ref_prob`, and `comp_ref_prob`, alongside the round-22 §6.3.11
`read_is_inter_probs( )`, round-23 §6.3.9 / §6.3.10, and round-24
§6.3.14 sweeps:

* `read_frame_reference_mode_probs( coder, reference_mode,
  comp_mode_prob, single_ref_prob, comp_ref_prob )` per §6.3.13
  (`vp9-spec.txt` lines 2195-2210). Three conditional sweeps gated by
  the `reference_mode` enum decided in §6.3.12:
  * `REFERENCE_MODE_SELECT` fires the `comp_mode_prob` sweep
    (`COMP_MODE_CONTEXTS = 5` cells); the other two modes skip it.
  * Any mode `!= COMPOUND_REFERENCE` fires the `single_ref_prob`
    sweep (`REF_CONTEXTS × 2 = 10` cells).
  * Any mode `!= SINGLE_REFERENCE` fires the `comp_ref_prob` sweep
    (`REF_CONTEXTS = 5` cells).
  * Cell totals per mode: `SINGLE_REFERENCE` 10, `COMPOUND_REFERENCE`
    5, `REFERENCE_MODE_SELECT` 20. Every cell consumes one `B(252)`
    `update_prob` flag and, on 1, a `decode_term_subexp( )` +
    `inv_remap_prob( )` cascade.
* `ReferenceMode` public enum mirrors the §3 sentinels
  (`SINGLE_REFERENCE = 0`, `COMPOUND_REFERENCE = 1`,
  `REFERENCE_MODE_SELECT = 2`); the §6.3.12 walker still pending lands
  the decode that picks one.
* §3 constants `COMP_MODE_CONTEXTS = 5` (line 472) and `REF_CONTEXTS = 5`
  (line 473) plus §10.5 defaults `default_comp_mode_prob =
  {239, 183, 119, 96, 41}`, `default_comp_ref_prob = {50, 126, 123,
  221, 226}`, and the 5×2 `default_single_ref_prob` table transcribed
  verbatim into `mode_info.rs`. `compressed.rs` re-exports each as
  `DEFAULT_COMP_MODE_PROB_TABLE` / `DEFAULT_COMP_REF_PROB_TABLE` /
  `DEFAULT_SINGLE_REF_PROB_TABLE`, keeping `mode_info` as the single
  source of truth (mirroring round-22..24 staging).

Validation (+12 lib tests, lib total 353 → 365; suite total 373 → 385)
covers: §3 constant pinning (`COMP_MODE_CONTEXTS = 5`,
`REF_CONTEXTS = 5`); verbatim §10.5 transcription of each default
table; the `SINGLE_REFERENCE` branch only touching `single_ref_prob`
(other two snapshots preserved); the `COMPOUND_REFERENCE` branch only
touching `comp_ref_prob`; the `REFERENCE_MODE_SELECT` branch firing
all three sweeps with starting tables preserved on a zero buffer;
cursor-equivalence proofs that each branch consumes exactly its
prescribed cell count (5 / 10 / 20) of `B(252)` flags against a
parallel-coder walker; an explicit row-major walk equivalence against
a parallel coder for the `REFERENCE_MODE_SELECT` branch with two
starting-table triples; and single-source-of-truth checks tying each
`compressed.rs` re-export back to its `mode_info` constant.

Out of scope for round 25: §6.3.12 `frame_reference_mode( )` itself —
that needs the §6.2 `ref_frame_sign_bias[ ]` derivation which the
uncompressed-header walker still rejects with `Error::Unsupported`.
The round-25 surface stays internal-only (`pub(crate)` with
`#[allow(dead_code)]` on the function + re-export consts; only the
`ReferenceMode` enum is `pub` so the §6.3.12 walker can land it).

## Status — 2026-05-27 (round 24)

**Round 24: §6.3.14 `read_y_mode_probs( )` compressed-header sweep.**
Round 24 extends the §6.3 inter-arm primitives chain by one cell
alongside the round-22 §6.3.11 `read_is_inter_probs( )` and round-23
§6.3.9 / §6.3.10 sweeps:

* `read_y_mode_probs( coder, y_mode_probs )` per §6.3.14
  (`vp9-spec.txt` lines 2220-2225). A `BLOCK_SIZE_GROUPS = 4` (§3 line
  460) × `INTRA_MODES - 1 = 9` (§3 line 505) = 36-cell row-major
  sweep of `read_diff_update_prob` calls — one `B(252)` `update_prob`
  flag per cell and, on 1, a `decode_term_subexp( )` +
  `inv_remap_prob( )` cascade — updating `y_mode_probs[ ][ ]` in
  place.
* `DEFAULT_Y_MODE_PROBS` (`mode_info`, transcribed verbatim from §9.3 /
  §10.5) carries the inter-frame `intra_mode` initial / reset
  probabilities (row annotations preserved: 0 = block_size < 8x8,
  1 = < 16x16, 2 = < 32x32, 3 = >= 32x32). The same constant feeds the
  (still-deferred) §7.4.5 intra-mode tree decoder of
  `inter_block_mode_info( )`.
* `DEFAULT_Y_MODE_PROBS_TABLE` re-export in `compressed.rs` keeps
  `mode_info` as the single source of truth (mirroring the round-22
  `DEFAULT_IS_INTER_PROB_TABLE` and round-23 inter-mode / interp-filter
  staging patterns).

Validation (+8 tests, lib total 345 → 353; suite total 365 → 373)
covers the §3 constant pinning for `BLOCK_SIZE_GROUPS = 4` and
`INTRA_MODES = 10`; the verbatim transcription of the §9.3
`default_y_mode_probs` table; the zero-buffer `update_prob = 0`
pass-through preserving the starting table; an all-cells-visited
check with a non-uniform custom starting table; a cursor-equivalence
proof that the sweep consumes exactly 36 `B(252)` flags; explicit
row-major walk equivalence against a parallel-coder reference (two
starting tables); and a single-source-of-truth check that the
`compressed.rs` re-export equals the `mode_info` constant.

Out of scope for round 24: the §6.3 outer-dispatch `FrameIsIntra == 0`
arm itself — the y-mode sweep lives alongside `read_is_inter_probs( )`
+ `read_inter_mode_probs( )` + `read_interp_filter_probs( )` between
`read_skip_prob( )` (§6.3.8) and the (still-deferred) §6.3.15
`read_partition_probs( )` / §6.3.16 `mv_probs( )` cells. Wiring any
subset into `parse_compressed_header` before §6.3.12 / §6.3.13
(`frame_reference_mode( )` + `frame_reference_mode_probs( )`) land
would mis-position the coder cursor — and §6.3.12 needs
`ref_frame_sign_bias[ ]` state the uncompressed-header walker still
rejects with `Error::Unsupported`. The round-24 surface stays
internal-only (`pub(crate)` with `#[allow(dead_code)]` until the
outer dispatch grows the inter arm); the public API still exposes
`parse_uncompressed_header`, `parse_compressed_header` and their
result types exclusively.

## Status — 2026-05-26 (round 23)

**Round 23: §6.3.9 `read_inter_mode_probs( )` + §6.3.10
`read_interp_filter_probs( )` compressed-header sweeps.** Round 23
extends the §6.3 inter-arm primitives chain by two cells alongside
the round-22 §6.3.11 `read_is_inter_probs( )`:

* `read_inter_mode_probs( coder, inter_mode_probs )` per §6.3.9
  (`vp9-spec.txt` lines 2138-2143). An `INTER_MODE_CONTEXTS = 7` (§3
  line 507) × `INTER_MODES - 1 = 3` (§3 line 506) = 21-cell
  row-major sweep of `read_diff_update_prob` calls.
* `read_interp_filter_probs( coder, interp_filter_probs )` per
  §6.3.10 (`vp9-spec.txt` lines 2146-2151). An
  `INTERP_FILTER_CONTEXTS = 4` (§3 line 495) × `SWITCHABLE_FILTERS - 1
  = 2` (§3 line 487) = 8-cell row-major sweep. The spec swaps the
  loop-index names (outer `j`, inner `i`) — the visit order still
  matches the array layout `[INTERP_FILTER_CONTEXTS][SWITCHABLE_FILTERS - 1]`.
* `DEFAULT_INTER_MODE_PROBS` (`mode_info`, transcribed verbatim from
  §10.5 lines 7758-7766) and `DEFAULT_INTERP_FILTER_PROBS`
  (`mode_info`, transcribed verbatim from §10.5 lines 7769-7775)
  carry the spec's annotated initial / reset values. The same
  constants will feed the (still-deferred) §6.4.16
  `inter_block_mode_info( )` per-block reader once that lands.
* `DEFAULT_INTER_MODE_PROBS_TABLE` and `DEFAULT_INTERP_FILTER_PROBS_TABLE`
  re-exports in `compressed.rs` keep `mode_info` as the single source
  of truth (mirroring the round-22 `DEFAULT_IS_INTER_PROB_TABLE`
  staging pattern).

Validation (+16 tests, lib total 329 → 345; suite total 349 → 365)
covers the §3 constant pinning for `INTER_MODE_CONTEXTS = 7`,
`INTER_MODES = 4`, `INTERP_FILTER_CONTEXTS = 4`, `SWITCHABLE_FILTERS = 3`;
the verbatim transcription of each §10.5 default table; the
zero-buffer `update_prob = 0` pass-through preserving each starting
table; an all-cells-visited check with non-uniform starts (custom
tables preserved across the sweep); cursor-equivalence proofs that
each sweep consumes exactly its prescribed cell count of `B(252)`
flags (21 for inter-mode, 8 for interp-filter); explicit row-major
walk equivalence against a parallel-coder reference; and
single-source-of-truth checks that the `compressed.rs` re-exports
equal the `mode_info` constants.

Out of scope for round 23: the §6.3 outer-dispatch `FrameIsIntra == 0`
arm itself — the inter-mode / interp-filter sweeps live alongside
`read_is_inter_probs( )` between `read_skip_prob( )` (§6.3.8) and
`frame_reference_mode( )` (§6.3.12). Wiring any subset into
`parse_compressed_header` before §6.3.12 / §6.3.13 land would
mis-position the coder cursor. The round-23 surface stays
internal-only (`pub(crate)` with `#[allow(dead_code)]` until the
outer dispatch grows the inter arm); the public API still exposes
`parse_uncompressed_header`, `parse_compressed_header` and their
result types exclusively.

## Status — 2026-05-26 (round 22)

**Round 22: §6.3.11 `read_is_inter_probs( )` compressed-header sweep.**
Round 22 lands the unconditional `IS_INTER_CONTEXTS = 4`
`diff_update_prob` walk that populates the running `is_inter_prob[ ]`
table the round-21 §6.4.13 `read_is_inter( )` per-block decoder
consumes via the §9.3.2 ctx:

* `read_is_inter_probs( coder, is_inter_prob )` per §6.3.11
  (`vp9-spec.txt` lines 2154-2167). Four sequential
  `read_diff_update_prob` calls — one `B(252)` `update_prob` flag per
  slot and, on 1, a `decode_term_subexp` + `inv_remap_prob` cascade —
  updating `is_inter_prob[0..IS_INTER_CONTEXTS]` in place.
* `DEFAULT_IS_INTER_PROB_TABLE` re-export of the round-21
  `mode_info::DEFAULT_IS_INTER_PROB = {9, 102, 187, 225}` (§10.5)
  initial / reset table — same constant feeds both the §6.4.13
  per-block decoder and the §6.3.11 compressed-header sweep so there's
  a single source of truth.

Validation covers the §10.5 default re-export matching the
`mode_info` source-of-truth constant, the zero-buffer
`update_prob = 0` path passing every cell through unchanged, the
four-context cell-count visiting (custom probabilities preserved
across the sweep), an explicit equivalence test that the sweep is
identical to four sequential `read_diff_update_prob` calls against a
parallel coder (proves cursor advancement matches and the order of
slots is preserved), the §3 `IS_INTER_CONTEXTS = 4` constant pinning,
a tuple-sweep exhaustive round-trip across distinct starting
probabilities, and an independent cursor-equivalence check confirming
the function consumes exactly four `B(252)` flags on the zero buffer.

Out of scope for round 22: the §6.3 outer-dispatch `FrameIsIntra == 0`
arm itself — `read_is_inter_probs( )` lives *between* `read_skip_prob`
(§6.3.8) and `frame_reference_mode( )` (§6.3.12) inside the gated
inter branch alongside `read_inter_mode_probs( )` (§6.3.9) and
`read_interp_filter_probs( )` (§6.3.10); wiring it into
`parse_compressed_header` ahead of those companions would mis-position
the coder cursor. Each of those primitives lands in its own round; the
outer dispatch grows the inter arm only when §6.3.9 / §6.3.10 are also
in place. The round-22 surface stays internal-only (`pub(crate)`); the
public API still exposes `parse_uncompressed_header`,
`parse_compressed_header` and their result types exclusively.

## Status — 2026-05-26 (round 21)

**Round 21: §6.4.13 `read_is_inter( )` + §9.3.2 `is_inter` context +
§10.5 `default_is_inter_prob`.** Round 21 lands the per-block
inter/intra reader the §6.4.11 `inter_frame_mode_info( )` driver fires
between `read_skip( )` and `read_tx_size( !skip || !is_inter )`:

* §3 constants `SEG_LVL_REF_FRAME = 2` and `IS_INTER_CONTEXTS = 4`
  transcribed verbatim.
* §10.5 `default_is_inter_prob[IS_INTER_CONTEXTS] = {9, 102, 187, 225}`
  transcribed verbatim (the initial / reset table the §6.3.10
  `read_is_inter_probs( )` compressed-header sweep updates with
  `diff_update_prob` deltas — that sweep lands in a separate round
  alongside the rest of §6.3.9..§6.3.16).
* `IsInterNeighbours { above: Option<i32>, left: Option<i32> }` — the
  §6.4.11 above/left `RefFrames[ ][ ][ 0 ]` view: `Some( rf )` carries
  the neighbour's `ref_frame[0]` (`INTRA_FRAME` / `LAST_FRAME` /
  `GOLDEN_FRAME` / `ALTREF_FRAME` / `NONE`), `None` encodes
  `!AvailU` / `!AvailL` (the §6.4.11 listing forces the neighbour to
  `INTRA_FRAME` when unavailable, so the §9.3.2 `*Intra` rule resolves
  the same way).
* `is_inter_context( nb )` (§9.3.2) — the four-branch ctx derivation:
  - both available, both intra → `3`
  - both available, exactly one intra → `1` (`true || false = 1`)
  - both available, neither intra → `0`
  - only one available, that one intra → `2`
  - only one available, that one inter → `0`
  - neither available → `0`
  Returns one of `0..=3` indexing `is_inter_prob[ ctx ]`.
* `read_is_inter( coder, seg_feature_ref_frame_active,
  segment_ref_frame_data, is_inter_prob, nb )` (§6.4.13) — two paths:
  - `seg_feature_active( SEG_LVL_REF_FRAME )` → `is_inter =
    FeatureData[ segment_id ][ SEG_LVL_REF_FRAME ] != INTRA_FRAME`
    without consuming any coder bits.
  - otherwise → one §9.3.3 `BINARY_TREE` bit under
    `is_inter_prob[ is_inter_context( nb ) ]`.

Validation covers the §10.5 / §3 constants, every branch of the
§9.3.2 ctx derivation (including the `NONE = -1` ref-frame sentinel
which satisfies `<= INTRA_FRAME` and is treated as intra-side), the
§6.4.13 seg-feature path for `INTRA_FRAME` (→ false) and each of
`LAST_FRAME` / `GOLDEN_FRAME` / `ALTREF_FRAME` overrides (→ true), the
zero-coder bit=0 path, the bias-coder bit=1 path with both-intra
neighbours, an `is_inter_prob[ ctx ]` indexing sweep across all four
ctxs that confirms no panic / out-of-range, and a `seg_feature_active`
short-circuit test that ignores both neighbours and the coder.

Out of scope for round 21: the §6.4.11 `inter_frame_mode_info( )`
orchestrator itself (still composes this with `inter_segment_id( )` /
`read_skip( )` / `read_tx_size( )` / `inter_block_mode_info( )` or
`intra_block_mode_info( )`); the §6.3.10 `read_is_inter_probs( )`
compressed-header sweep (lands with the rest of §6.3.9..§6.3.16);
`inter_block_mode_info( )` (§6.4.16 — blocked on reference-buffer
state and MV decode); the §8.4 `counts_is_inter` probability-adaption
accumulator (§9.3.4 bookkeeping for end-of-frame adaption). The
round-21 surface stays internal-only (`pub(crate)`); the public API
still exposes `parse_uncompressed_header`, `parse_compressed_header`
and their result types exclusively.

## Status — 2026-05-26 (round 20)

**Round 20: §6.4.12 `inter_segment_id( )` + §6.4.14 `get_segment_id( )`
+ §7.4 `AboveSegPredContext` / `LeftSegPredContext` strips.** Round 20
lands the inter-frame companion to the round-16 `intra_segment_id`
primitive — the per-block segment-id reader the §6.4.11
`inter_frame_mode_info( )` driver fires before `read_skip( )` /
`read_is_inter( )` / `read_tx_size( )`:

* §10.2 `num_8x8_blocks_high_lookup[ BLOCK_SIZES ]` =
  `{1, 1, 1, 1, 2, 1, 2, 4, 2, 4, 8, 4, 8}` transcribed verbatim
  (alongside the existing `num_8x8_blocks_wide_lookup`), keying both
  the §6.4.14 `bh` clamp and the §6.4.12 `LeftSegPredContext[ ]`
  write-back length.
* `PrevSegmentIds<'a>` — a borrowed row-major `MiRows × MiCols` view
  of the previous frame's segment-id plane (the §6.4.4 `SegmentIds[ ][
  ]` write-back).
* `get_segment_id( prev, mi_row, mi_col, mi_size )` (§6.4.14) — the
  `bw` / `bh` clamp via `Min( MiCols - MiCol, bw )` /
  `Min( MiRows - MiRow, bh )` and the `seg = 7; seg = Min( seg,
  PrevSegmentIds[ … ] )` spatial-minimum sweep.
* `SegPredContextState { above[MiCols], left[MiRows] }` — the §7.4.1 /
  §7.4.2 strip storage with `new( )` zero-init,
  `clear_left( )` per-superblock-row reset, and `above( ) ` /
  `left( )` ctx accessors.
* `read_seg_id_predicted( coder, pred_prob, seg_pred_ctx, mi_row,
  mi_col )` — the §9.3.2 `ctx = LeftSegPredContext[ MiRow ] +
  AboveSegPredContext[ MiCol ]` derivation and §9.3.1 `BINARY_TREE`
  one-bit decode under `segmentation_pred_prob[ ctx ]`.
* `inter_segment_id( )` (§6.4.12) — the four-path orchestrator:
  (1) `!segmentation_enabled` → 0; (2) enabled but `!update_map` →
  `predictedSegmentId`; (3) `update_map && !temporal_update` →
  `read_segment_id` (the round-16 §9.3.1 walk); (4) `update_map &&
  temporal_update` → `read_seg_id_predicted` then either the
  predictor or a fresh `read_segment_id`, followed by the spec's
  trailing write-back of `seg_id_predicted` into
  `AboveSegPredContext[ MiCol + i ]` (`i ∈ 0..bw`) and
  `LeftSegPredContext[ MiRow + i ]` (`i ∈ 0..bh`).

Validation covers `get_segment_id` (interior 2x2 min, partial-edge
clamp via `Min( MiCols - MiCol, bw )`, all-7 fallback), the §7.4
zero-init contract (and `clear_left` not touching `Above`), the §9.3.2
`Left + Above` ctx wiring of `read_seg_id_predicted`, each of the
four §6.4.12 paths independently, the `Error::InvalidBitstream`
surface when `tree_probs` (paths 3 / 4-not-predicted) or `pred_prob`
(path 4) are missing, and the §6.4.12 trailing write-back clamping on
a partial-edge `BLOCK_32X32` at `(1, 1)` of a 3-wide frame.

Out of scope for round 20: the §6.4.11 `inter_frame_mode_info( )`
orchestrator itself (composes this primitive with `read_skip( )` /
`read_is_inter( )` / `read_tx_size( )` / `inter_block_mode_info( )`
or `intra_block_mode_info( )`); `read_is_inter( )` (§6.4.13 — needs
the §9.3.2 `is_inter` ctx and the `is_inter_prob[ 4 ]` compressed-
header table); `inter_block_mode_info( )` (§6.4.16 — blocked on
reference-buffer state and MV decode); the `PrevSegmentIds[ ][ ]` /
`SegmentIds[ ][ ]` frame-wide write-back (left to the §6.4.4 driver);
and the §8.4 `counts_*` probability-adaption accumulators (§9.3.4
bookkeeping for end-of-frame adaption). The round-20 surface stays
internal-only (`pub(crate)`); the public API still exposes
`parse_uncompressed_header`, `parse_compressed_header` and their
result types exclusively.

## Status — 2026-05-26

**Round 19: §6.4.3 recursive `decode_partition()` driver (extending the
`partition` module).** Round 19 composes the round-18
`decode_partition_type()` primitive into the full §6.4.3 recursive
partition driver:

* `decode_partition(coder, r, c, bsize, mi_rows, mi_cols, ctx_state,
  probs_kind, leaves)` — walks the §6.4.3 listing line-for-line: the
  `(r >= MiRows || c >= MiCols)` quadrant short-circuit, the `num8x8`
  / `halfBlock8x8` / `hasRows` / `hasCols` derivation, the
  `partition` decode via the round-18 primitive (with the §9.3.2
  `partition_plane_context` ctx + the per-frame probability source),
  the four-way dispatch on the decoded `PARTITION_*` value (with
  HORZ second-leaf gated by `hasRows`, VERT second-leaf gated by
  `hasCols`, and SPLIT recursing in spec order TL → TR → BL → BR),
  and the §6.4.3 tail write-back into the partition-context strips
  (gated by `bsize == BLOCK_8X8 || partition != PARTITION_SPLIT`,
  writing `15 >> b_*_log2_lookup[subsize]` per cell).
* `PartitionContextState` — the `AbovePartitionContext[]` /
  `LeftPartitionContext[]` strips (sized `Sb64Cols * 8` /
  `Sb64Rows * 8` per the §7.4 listing). Exposes `new(mi_cols, mi_rows)`
  with the §7.4 zero-reset and `clear_left()` for the §6.4.2
  per-superblock-row reset.
* `PartitionProbsKind` — the per-frame probability source enum
  (`Keyframe` → indexes the §10.4 `KF_PARTITION_PROBS` directly;
  `Inter(&table)` → indexes the caller's running 16 × 3 table,
  typically initialised from §10.5 `DEFAULT_PARTITION_PROBS` and
  conditionally updated by the §6.3 `read_partition_probs()` sweep —
  still pending in a later round).
* `LeafBlock { r, c, subsize }` log records — emitted in §6.4.3
  traversal order in lieu of the §6.4.4 `decode_block(r, c, subsize)`
  call site (the per-block `mode_info` / `residual` decode is
  downstream and not yet wired into this driver).

Validation includes three hand-built bitstreams produced by a
test-only minimal range encoder that mirrors the §9.2.2 decode steps
inverse-by-inverse:

* a single 64x64 superblock with `PARTITION_NONE` → one leaf
  `{ 0, 0, BLOCK_64X64 }` and the §6.4.3 tail `15 >> 4 = 0`
  write-back;
* a single 64x64 superblock with `PARTITION_SPLIT` then four 32x32
  `PARTITION_NONE` children → four leaves in TL → TR → BL → BR order,
  with the §6.4.3 tail write-back firing on each child (`15 >> 3 = 1`)
  but not on the parent SPLIT;
* a 64x64 superblock split with mixed HORZ / VERT children (TL HORZ →
  2 leaves at BLOCK_32X16, TR VERT → 2 leaves at BLOCK_16X32, BL
  HORZ, BR VERT) exercising the HORZ second-leaf and VERT
  second-leaf §6.4.3 paths plus the §6.4.3 tail write-back across
  mixed partitions and the §9.3.2 ctx derivation across the
  successively-populated strip state.

Out of scope for round 19:

* The §6.3 `read_partition_probs()` compressed-header sweep
  (`PARTITION_CONTEXTS × (PARTITION_TYPES - 1) = 16 × 3 = 48`
  `diff_update_prob` cells against `DEFAULT_PARTITION_PROBS`) — the
  driver consumes the `Inter` running table, but constructing it
  lands in a later round.
* The §6.4.4 `decode_block()` mode-info + residual decode that
  `LeafBlock` stands in for — wiring it into this driver is
  downstream of all the §6.4 mode-info readers landing first.
* The §6.4.2 `decode_tile()` outer loop (the `r += 8, c += 8`
  superblock walk + per-row `clear_left_context()`) — composes this
  driver but is a separate round.

The round-19 surface stays internal-only (`pub(crate)`); the public
API still exposes `parse_uncompressed_header`,
`parse_compressed_header` and their result types exclusively.

## Status — 2026-05-26 (round 18)

**Round 18: §6.4.3 `decode_partition_type()` per-call partition reader (new
`partition` module).** Round 18 lands the single-call partition decoder
the recursive §6.4.3 `decode_partition(r, c, bsize)` driver fires once per
quadrant inside a tile, plus every §3 / §10.2 / §10.4 / §10.5 surface it
consumes:

* §3 enumeration `PARTITION_NONE = 0`, `PARTITION_HORZ = 1`,
  `PARTITION_VERT = 2`, `PARTITION_SPLIT = 3` plus `PARTITION_TYPES = 4`
  and `PARTITION_CONTEXTS = 16`.
* §9.3.1 trees `PARTITION_TREE[6]`, `COLS_PARTITION_TREE[2]`,
  `ROWS_PARTITION_TREE[2]` transcribed verbatim from the spec listing.
* §10.2 lookups `B_WIDTH_LOG2_LOOKUP` / `B_HEIGHT_LOG2_LOOKUP` (the
  §6.4.3 tail `15 >> b_*_log2_lookup[subsize]` write-back inputs),
  `MI_WIDTH_LOG2_LOOKUP` (the §9.3.2 `bsl` input), and
  `NUM_8X8_BLOCKS_WIDE_LOOKUP` (the §6.4.3 `num8x8` input) — all
  transcribed verbatim.
* §10.2 `SUBSIZE_LOOKUP[4][13]` transcribed verbatim (with
  `BLOCK_INVALID = 14` for the HORZ / VERT / SPLIT entries with no
  legal child at non-square parents).
* §10.4 `KF_PARTITION_PROBS[16][3]` (keyframe / intra-only fixed table)
  and §10.5 `DEFAULT_PARTITION_PROBS[16][3]` (inter-frame initial
  table prior to §6.3 `read_partition_probs()`) transcribed verbatim
  with per-table shape + listing-anchor + §9.2 min-prob tests.
* `partition_plane_context(bsize, above_ctx, left_ctx)` — the §9.3.2
  `ctx = bsl * 4 + left * 2 + above` derivation: `bsl =
  mi_width_log2_lookup[bsize]`, `boffset = 3 - bsl`, OR-fold of the
  `AbovePartitionContext[]` / `LeftPartitionContext[]` strips across
  `num8x8` cells, extract the `bsl`-th bit. Covers `bsl ∈ {0, 1, 2, 3}`
  for the four superblock recursion sizes (`BLOCK_8X8` / `BLOCK_16X16`
  / `BLOCK_32X32` / `BLOCK_64X64`) with the full `ctx ∈ 0..=15`
  reached via the included exhaustive sweep.
* `decode_partition_type(coder, has_rows, has_cols, probs)` — the
  §6.4.3 reader proper. Dispatches on `(has_rows, has_cols)` per the
  §9.3.1 tree-selection rule: interior (`(true, true)`) walks the
  6-entry `PARTITION_TREE` with `node2 = node`; right-edge
  (`(false, true)`) walks 2-entry `COLS_PARTITION_TREE` with
  `node2 = 1`; bottom-edge (`(true, false)`) walks 2-entry
  `ROWS_PARTITION_TREE` with `node2 = 2`; corner (`(false, false)`)
  returns `PARTITION_SPLIT` directly without consuming any bool-coder
  bits. Returns one of the four `PARTITION_*` constants.

The recursive §6.4.3 driver itself — which threads
`SUBSIZE_LOOKUP[partition][bsize]` into four recursive calls when
`PARTITION_SPLIT` and writes back the `AbovePartitionContext[]` /
`LeftPartitionContext[]` strips with `15 >> b_*_log2_lookup[subsize]` —
lands in a later round. The §6.3 `read_partition_probs()`
compressed-header sweep (`PARTITION_CONTEXTS × (PARTITION_TYPES - 1) =
16 × 3 = 48` `diff_update_prob` cells against `DEFAULT_PARTITION_PROBS`)
also lands in a later round. The round-18 surface is internal-only;
the public API still exposes `parse_uncompressed_header`,
`parse_compressed_header` and their result types exclusively.

## Status — 2026-05-25

**§6.4.15 `intra_block_mode_info()` inter-frame intra-block reader.**
The companion to the §6.4.6 `intra_frame_mode_info()` keyframe driver,
fired by the §6.4.11 `inter_frame_mode_info()` path when a block in a
non-keyframe frame is coded intra (`is_inter == 0`):

* §9.3.2 `size_group_lookup[BLOCK_SIZES]`
  (`{0,0,0,1,1,1,2,2,2,3,3,3,3}`) and §9.3
  `default_y_mode_probs[BLOCK_SIZE_GROUPS][INTRA_MODES - 1]` (4 × 9) /
  `default_uv_mode_probs[INTRA_MODES][INTRA_MODES - 1]` (10 × 9)
  transcribed verbatim — the compressed-header `y_mode_probs` /
  `uv_mode_probs` defaults, distinct from the §10.5 keyframe
  `kf_*_mode_probs`. Per-table shape + anchor + §9.2 min-prob tests.
* `intra_mode( coder, y_mode_probs, mi_size )` — §9.3.3 walk over
  `intra_mode_tree` with ctx `size_group_lookup[MiSize]`;
  `sub_intra_mode( coder, y_mode_probs )` — ctx fixed at 0;
  `uv_mode( coder, uv_mode_probs, y_mode )` — ctx `y_mode`. Each ctx
  derivation has an instrumented-callback test pinning the row reached,
  plus hand-traced bias-buffer cases (`intra_mode` / `uv_mode` →
  `D207_PRED`).
* `intra_block_mode_info()` (§6.4.15) → `Vp9IntraBlockModeInfo
  { ref_frame_0, ref_frame_1, y_mode, sub_modes[4], uv_mode }`.
  `ref_frame[0] = INTRA_FRAME`, `ref_frame[1] = NONE`; the
  `MiSize >= BLOCK_8X8` arm decodes one `intra_mode` replicated across
  `sub_modes[]`, the sub-8x8 arm walks the `(idy, idx)` grid decoding
  one `sub_intra_mode` per cell (`y_mode` = last). Unlike §6.4.6 it
  reads **only** modes — `segment_id` / `skip` / `tx_size` are decoded
  by the §6.4.11 driver beforehand. A per-block bias-buffer scenario
  pins the contiguous `intra_mode → uv_mode` decode
  (`D207_PRED` then `D153_PRED`).
* §6.4.5 `mode_info()` dispatch shape: a `Vp9ModeInfo` enum with
  `IntraFrame(Vp9IntraMiBlock)` (the `FrameIsIntra` /
  `intra_frame_mode_info()` path) and
  `InterFrameIntraBlock(Vp9IntraBlockModeInfo)` (the `!FrameIsIntra`,
  `is_inter == 0` / `intra_block_mode_info()` sub-path), plus
  `inter_frame_intra_block_mode_info()` wiring the latter. The
  surrounding §6.4.11 prelude (`inter_segment_id` / `read_is_inter` /
  the `inter_block_mode_info()` arm) lands when its
  reference-buffer-dependent primitives do.

**Round 17: §6.4.6 `intra_frame_mode_info()` keyframe driver.** Round
17 wires the rounds 15 / 16 primitives into the §6.4.6 per-block
mode-info reader for keyframe (and intra-only) frames — the spec's
top-level entry point for an intra MI block:

* §9.3.1 `intra_mode_tree[18]` — the 18-entry / 10-leaf tree shared by
  `default_intra_mode` / `default_uv_mode` / `intra_mode` /
  `sub_intra_mode` / `uv_mode`. Transcribed verbatim
  (`{ -DC_PRED, 2, -TM_PRED, 4, -V_PRED, 6, 8, 12, -H_PRED, 10,
  -D135_PRED, -D117_PRED, -D45_PRED, 14, -D63_PRED, 16, -D153_PRED,
  -D207_PRED }`); the all-bit-0 walk lands on the `-DC_PRED` leaf
  (= 0).
* §10.5 `kf_y_mode_probs[INTRA_MODES][INTRA_MODES][INTRA_MODES - 1]`
  (a 10 × 10 × 9 = 900-byte table indexed by `[abovemode][leftmode]
  [node]` per the §9.3.2 `default_intra_mode` listing) transcribed
  verbatim from the §10.5 listing. Anchor checks pin five rows
  including `[dc][dc]`, `[dc][tm]`, `[h][h]`, and the last row
  `[tm][tm]`.
* §10.5 `kf_uv_mode_probs[INTRA_MODES][INTRA_MODES - 1]` (10 × 9 = 90
  bytes, indexed by `[y_mode][node]` per the §9.3.2 `default_uv_mode`
  listing) transcribed verbatim. Anchor checks pin the `[dc]`, `[h]`
  and `[tm]` rows.
* `default_intra_mode( coder, abovemode, leftmode )` — the §9.3.3
  walk over `intra_mode_tree` with `kf_y_mode_probs[above][left][node]`
  per row. Hand-traced bias-buffer test pins
  `default_intra_mode( DC_PRED, DC_PRED ) -> D207_PRED` (right-branch
  on every node, terminating at the §9.3.1 `-D207_PRED` leaf).
* `default_uv_mode( coder, y_mode )` — the §9.3.3 walk with
  `kf_uv_mode_probs[y_mode][node]`. Same hand-traced bias-buffer test
  pins `default_uv_mode( DC_PRED ) -> D207_PRED`.
* `intra_frame_mode_info()` (§6.4.6) — the orchestrator threading
  `intra_segment_id( )` (round 16) + `read_skip( )` (round 15) +
  `read_tx_size( 1 )` (round 15) + `default_intra_mode` +
  `default_uv_mode` into a `Vp9IntraMiBlock { segment_id, skip,
  tx_size, ref_frame_0, ref_frame_1, is_inter, y_mode, sub_modes[4],
  uv_mode }`. `ref_frame[0] = INTRA_FRAME = 0`, `ref_frame[1] = NONE
  = -1`, `is_inter = false` are hardwired per the §6.4.6 listing
  (NONE = -1 derived from `isCompound = ref_frame[1] > NONE` plus
  `ref_frame[1] > INTRA_FRAME = 0` for compound; the unique integer
  strictly below INTRA_FRAME). The `MiSize >= BLOCK_8X8` arm decodes
  one `default_intra_mode` and replicates it into all four
  `sub_modes[ ]` cells; the `MiSize < BLOCK_8X8` arm walks the §6.4.6
  `(idy, idx)` grid stepped by `num_4x4_blocks_high_lookup[MiSize]` /
  `num_4x4_blocks_wide_lookup[MiSize]` — 4 reads for BLOCK_4X4, 2 for
  BLOCK_4X8 / BLOCK_8X4 — with each cell receiving its own decoded
  mode replicated across the (num4x4h × num4x4w) `sub_modes[ ]`
  sub-grid, and `y_mode` set to the *last* decoded
  `default_intra_mode` per the spec listing.
* `IntraFrameNeighbours { avail_u, avail_l, above_sub_modes_23[2],
  left_sub_modes_13[2] }` — the per-MI-block neighbour bundle a tile
  driver builds from its frame-wide `SubModes[ ][ ][ ]` array. The
  §9.3.2 listing reads only positions {2, 3} of the above neighbour's
  `sub_modes[ ]` and positions {1, 3} of the left neighbour's
  `sub_modes[ ]`, so the bundle only carries those two cells per
  side; `DC_PRED` is substituted when `avail_u` / `avail_l` is false
  per the §9.3.2 fallback.

Out of scope for round 17 (the §6.4.15 `intra_block_mode_info` reader
itself landed subsequently — see the section above):
inter-frame mode info (§6.4.11+, blocked on reference-buffer state);
the §8.4 `counts_intra_mode` / `counts_uv_mode` probability-adaption
accumulators (§9.3.4 bookkeeping); and the `SubModes[ ][ ][ ]` /
`YModes[ ][ ]` frame-wide write-back the next MI block consumes from
the just-decoded `Vp9IntraMiBlock` (left to the §6.4.4 driver). The
round-17 surface is internal-only; the public API still exposes
`parse_uncompressed_header`, `parse_compressed_header` and their
result types exclusively.

**Round 16: §6.4.7 `intra_segment_id` + §9.3.1 `segment_tree`.** Round
16 extends the round-15 `mode_info` module (crate-internal,
`pub(crate)`) with the next slice of the §6.4.6
`intra_frame_mode_info()` orchestrator's primitives:

* §9.3.1 `segment_tree[14]` —
  `{ 2, 4, 6, 8, 10, 12, 0, -1, -2, -3, -4, -5, -6, -7 }` transcribed
  verbatim. A 7-leaf binary tree mapping to segment ids `0..=7`; note
  the §9.3.1 packing means the all-bit-0 walk visits node indices
  `{0, 1, 3}` (not the contiguous `0..3` a regular binary tree would).
* `read_segment_id( coder, tree_probs )` — the §9.3.3 `tree_decode`
  walk over `SEGMENT_TREE` with per-node probability
  `segmentation_tree_probs[node]` per the §9.3.2 listing's
  `segment_id` entry. Returns the decoded segment id directly (the
  §9.3.3 post-loop `-n` already produces it).
* `intra_segment_id( coder, segmentation_enabled,
  segmentation_update_map, tree_probs )` (§6.4.7) — the
  `segmentation_enabled && segmentation_update_map` gate around
  `read_segment_id`. The intra-only path has no
  `segmentation_temporal_update` / `seg_id_predicted` machinery; when
  the gate fails `segment_id = 0` per the spec listing.

Out of scope for round 16: the §6.4.12 `inter_segment_id( )` syntax
(with the `predictedSegmentId = get_segment_id( )` spatial-prediction
helper + the `seg_id_predicted` binary decode + the
`AboveSegPredContext` / `LeftSegPredContext` write-back) — that's an
inter-frame primitive blocked on the reference-buffer state the
round-2 header walker still rejects with `Error::Unsupported`. The
§6.4.15 `intra_block_mode_info` (`default_intra_mode` /
`default_uv_mode` decode against the §10.5 `kf_y_mode_probs` /
`kf_uv_mode_probs` 3D / 2D tables) and the §6.4.6
`intra_frame_mode_info()` orchestrator that composes
`intra_segment_id` + `read_skip` + `read_tx_size` +
`intra_block_mode_info` into a single `Vp9IntraMiBlock` are deferred
to the next round. The round-16 surface is internal-only; the public
API still exposes `parse_uncompressed_header`,
`parse_compressed_header` and their result types exclusively.

**Round 15: §6.4.8 `read_skip` + §6.4.10 `read_tx_size` + §9.3.3
`tree_decode`.** Round 15 adds a `mode_info` module (crate-internal,
`pub(crate)`) implementing the first slice of the §6.4 per-block
mode-info decode that the round-14 `residual_intra` driver currently
consumes via a caller-supplied bundle — the building blocks the §6.4.6
`intra_frame_mode_info()` orchestrator will compose:

* §9.3.3 `tree_decode( coder, tree, prob )` — the generic
  `do { n = T[n + read_bool(P(n >> 1))] } while (n > 0)` walker that
  every tree-coded syntax element (skip, tx_size, intra_mode, …)
  routes through; the probability callback is a `FnMut(usize) -> u8`
  so call-sites splice the right §9.3.2 row in without the helper
  needing to know which syntax element it's decoding.
* §9.3.1 trees `tx_size_8_tree[2]`, `tx_size_16_tree[4]`,
  `tx_size_32_tree[6]`, and `binary_tree[2]` transcribed verbatim
  from the spec listing.
* §9.3.2 `skip_context` (the `Skips[MiRow-1][MiCol] +
  Skips[MiRow][MiCol-1]` derivation modulated by `AvailU` / `AvailL`)
  and `tx_size_context` (the `(above + left) > maxTxSize` rule that
  consults neighbour `TxSizes[ ]` only on unskipped MI blocks, and
  mirrors the side when a neighbour is unavailable).
* §6.4.8 `read_skip` — the `seg_feature_active(SEG_LVL_SKIP)`
  early-return rule plus the §9.3.2 binary-tree decode under
  `skip_prob[skip_context(nb)]`.
* §6.4.10 `read_tx_size` — the `allow_select && tx_mode ==
  TX_MODE_SELECT && MiSize >= BLOCK_8X8` path that walks the §9.3.1
  tree picked by `max_txsize_lookup[MiSize]`, indexed by the §9.3.2
  ctx; falls through to `Min(maxTxSize,
  tx_mode_to_biggest_tx_size[tx_mode])` per the spec's `else` branch.
* `NeighbourSkips` / `NeighbourTxSizes` — per-MI-block neighbour-state
  bundles a tile driver builds from its frame-wide `Skips[ ][ ]` /
  `TxSizes[ ][ ]` arrays.

Out of scope for round 15: the §6.4.6 `intra_frame_mode_info()`
orchestrator (which composes `read_skip` + `read_tx_size` + the
deferred §6.4.7 `intra_segment_id` + §6.4.15 `intra_block_mode_info`
into a single MI block); the `Skips[ ][ ]` / `TxSizes[ ][ ]`
frame-wide write-back (left to the §6.4.6 driver); inter-frame mode
info (§6.4.11+, needs reference-buffer state); and the §8.4
`counts_skip` / `counts_tx_size` probability-adaption accumulators
(§9.3.4 bookkeeping for the end-of-frame adaption round). The
round-15 surface is internal-only; the public API still exposes
`parse_uncompressed_header`, `parse_compressed_header` and their
result types exclusively.

**Round 14: §6.4.21 `residual( )` intra driver.** Round 14 adds a
`residual` module (crate-internal, `pub(crate)`) implementing the §6.4.21
`residual( )` outer loop for the **intra** path — the per-plane,
per-4x4-sub-block walk that owns the `AboveNonzeroContext` /
`LeftNonzeroContext` write-back across a whole MI block, drives the
round-13 §6.4.24 `tokens( )` per-block decode, and feeds the round-11
§8.6.2 `reconstruct_block` with real per-block `Tokens` arrays,
availability flags and plane/quantizer state:

* The §10.2 `num_4x4_blocks_wide_lookup` / `num_4x4_blocks_high_lookup`
  / `max_txsize_lookup` tables and the §6.4.23 `ss_size_lookup[ 13 ][ 2
  ][ 2 ]` table transcribed verbatim, alongside the `BLOCK_4X4 ..
  BLOCK_64X64` / `BLOCK_INVALID` `subsize` constants from §3.
* `get_plane_block_size( subsize, plane, subsampling_x, subsampling_y )`
  (§6.4.23) and `get_uv_tx_size( tx_size, mi_size, subsampling_x,
  subsampling_y )` (§6.4.22) — the chroma-plane block-size /
  transform-size derivations that key the per-plane loop.
* `ResidualBlockCtx` — the per-MI-block / per-frame bundle (`MiCol` /
  `MiRow` / `MiCols` / `MiRows`, `MiSize`, `tx_size`, `subsampling_x` /
  `y`, `skip`, `Lossless`, `BitDepth`, the per-block `PredMode` for luma
  + chroma, and the per-plane DC/AC quantizers from round 8); plus
  `AvailFlags` for §7.4.4 `AvailL` / `AvailU` and a `PlaneBuffers`
  wrapper for the three `CurrFrame[ plane ]` planes.
* `residual_intra` — the §6.4.21 driver proper: per plane, computes
  `bsize = MiSize < BLOCK_8X8 ? BLOCK_8X8 : MiSize`, the per-plane
  `planeSz` + `num4x4w` / `num4x4h` dimensions and chroma `txSz`, then
  walks the `(y, x)` 4x4 grid stepping by `step = 1 << txSz`. For each
  in-bounds transform block (`startX < maxx && startY < maxy`) it calls
  the round-10 `predict_intra` with the resolved `have_left` /
  `have_above` / `not_on_right` flags, pulls `Tokens[ ]` from a per-block
  `TokenSource` callback (when `!skip`), derives the §6.4.25 `TxType`
  (chroma / `TX_32X32` / lossless force `DCT_DCT`; luma intra uses
  round-11 `tx_type_for_intra`), runs the round-11 `reconstruct_block`,
  and writes `AboveNonzeroContext[ plane ][ x4 + i ] =
  LeftNonzeroContext[ plane ][ y4 + i ] = nonzero` for `i ∈ 0..step` per
  the §6.4.21 trailing write-back.

The `is_inter` branch of §6.4.21 (which calls `predict_inter( )` before
the per-block loop) is deferred until the §8.5.2 inter prediction
process and reference-buffer state land in a later round; the
per-block mode-info decode (`y_mode` / `sub_modes` / `tx_size` / `skip`
/ `segment_id` from §6.4) that the residual loop reads is also a
later-round increment — for round 14 the per-block mode-info bundle is
passed in by the test caller, and a production caller would thread it
in once the §6.4.6 / §6.4.7 / §6.4.10 mode-info syntax lands. The
round-14 surface is internal-only; the public API still exposes
`parse_uncompressed_header`, `parse_compressed_header` and their result
types exclusively.

**Round 13: §6.4.24 `tokens( )` per-block coefficient driver.** Round 13
adds the §6.4.24 `tokens( )` driver to the `tokens` module — the per-block
loop that walks the round-12 §6.4.25 scan order (`pos = scan[ c ]`) and
feeds each scan position through the round-7 `read_coef_token` pipeline,
recovering one transform block's quantised coefficients into a `Tokens[ ]`
array:

* The §10 band tables — `coefband_4x4[ 16 ]` transcribed verbatim and
  `coefband_8x8plus[ 1024 ]` built from the verbatim 21-entry prefix plus
  the all-`5` tail — picked by `coef_band( c, txSz )` per the §6.4.24
  `(txSz == TX_4X4) ? coefband_4x4 : coefband_8x8plus` rule.
* `token_cache_neighbours( c, pos, txSz, txType )` — the §9.3.2 neighbour
  pair (`nb[ 0 ]` / `nb[ 1 ]`): `(0, 0)` for the DC coefficient, and for
  `c > 0` the above (`(i-1)*n + j`) / left (`i*n + j-1`) raster cells with
  the `DCT_ADST` (double above) / `ADST_DCT` (double left) / first-row /
  first-column variants (`n = 4 << txSz`).
* `build_token_probs( cell )` — the §9.3.2 10-node probability array
  (node 0 → `cell[1]`, node 1 → `cell[2]`, node `2..=9` →
  `pareto( node, cell[2] )`).
* `NonzeroContext` (the per-plane `AboveNonzeroContext` /
  `LeftNonzeroContext` 4-sample strips) and `TokenBlockCtx` (the per-block
  / per-frame state the driver reads).
* `tokens( coder, block, txSz, scan, coef_probs, nz, token_cache, tokens )`
  — the §6.4.24 driver proper: `segEob = 16 << (txSz << 1)`, the `checkEob`
  gating, the §9.3.2 per-coefficient `ctx` (DC from the non-zero strips,
  `c > 0` from `TokenCache`), the
  `coef_probs[txSz][plane>0][is_inter][band][ctx]` cell pick, the
  `more_coefs` / `token` / `read_coef` / `sign_bit` decode, the
  `TokenCache[ pos ] = energy_class[ token ]` write, the
  `ZERO_TOKEN`-clears-`checkEob` rule, the trailing `Tokens[ scan[ i ] ] =
  0` zero-fill, and the `nonzero = c > 0` return.

The §6.4.21 `residual( )` plane / sub-block driver — which owns the
`AboveNonzeroContext` / `LeftNonzeroContext` write-back across a whole
frame, the per-block mode-info decode, and the wiring into the round-11
§8.6.2 `reconstruct_block` — lands in a later round. The round-13 surface
is internal-only; the public API still exposes
`parse_uncompressed_header`, `parse_compressed_header` and their result
types exclusively.

**Round 12: §6.4.25 `get_scan` scan-order selection.** Round 12 adds a
`scan` module (crate-internal, `pub(crate)`) implementing the §6.4.25
`get_scan( )` process — the first step of the §6.4.24 `tokens( )`
per-block driver, which selects the scan order (the sequence of raster
positions `pos = scan[ c ]` the coefficient loop walks) for a transform
block:

* The §10.1 scan tables transcribed verbatim — `default_scan_4x4` /
  `col_scan_4x4` / `row_scan_4x4` (16), the 8x8 trio (64), the 16x16
  trio (256), and `default_scan_32x32` (1024). The element type is
  `u16` so the 32x32 table's `0..=1023` raster positions fit.
* `get_scan( plane, tx_sz, tx_type )` (§6.4.25) — selects between the
  tables by the resolved `TxType`: `ADST_DCT` → `row_scan`, `DCT_ADST`
  → `col_scan`, else (`DCT_DCT` / `ADST_ADST`) → `default`. The §6.4.25
  first half (a chroma plane `plane > 0` or a `TX_32X32` block forces
  `TxType = DCT_DCT`) is applied here, so a caller passing the luma
  `TxType` for every plane still selects the right scan. The
  mode-info-dependent `mode2txfm_map[ y_mode ]` `TxType` derivation
  already lives in `reconstruct::tx_type_for_intra` (round 11); the
  per-block mode-info state (`y_mode`, `sub_modes`, `Lossless`,
  `is_inter`) is owned by the deferred §6.4.21 residual driver.
* `TX_4X4` / `TX_8X8` / `TX_16X16` / `TX_32X32` `txSz` index constants
  (§3).

The §6.4.24 `tokens( )` loop that walks `pos = scan[ c ]`, derives the
per-coefficient `ctx` from `AboveNonzeroContext` / `LeftNonzeroContext`
and `TokenCache`, and feeds the round-7 token decode lands in a later
round. The round-12 surface is internal-only; the public API still
exposes `parse_uncompressed_header`, `parse_compressed_header` and their
result types exclusively.

**Round 11: §8.6.2 reconstruct driver.** Round 11 adds a `reconstruct`
module (crate-internal, `pub(crate)`) implementing the §8.6.2
reconstruct process — the conceptual `reconstruct( plane, startX,
startY, txSz )` call site of the §6.4.21 residual syntax — that finally
ties the rounds 7-10 pieces into `reconstruct = predict + residual`:

* `tx_type_for_intra( mode )` (§6.4.25) — the `mode2txfm_map[ y_mode ]`
  lookup selecting the `TxType` (`DCT_DCT` / `ADST_DCT` / `DCT_ADST` /
  `ADST_ADST`) for a luma intra block from its `PredMode`. The 10-entry
  intra prefix of `mode2txfm_map` (§10.5) is transcribed verbatim (the
  four inter-mode entries, all `DCT_DCT`, are omitted as the helper
  only indexes the intra prefix).
* `reconstruct_block( plane_buf, x, y, tx_sz, tokens, dc_quant,
  ac_quant, tx_type, lossless, bit_depth )` (§8.6.2) — sets `dqDenom =
  2` for `txSz == TX_32X32` else `1`, `n = 2 + txSz`, `n0 = 1 << n`;
  step 1 `Dequant[i][j] = (Tokens[i*n0+j] * get_ac_quant) / dqDenom`,
  step 2 the `Dequant[0][0] = (Tokens[0] * get_dc_quant) / dqDenom` DC
  override, step 3 the round-9 §8.7.2 `inverse_transform_2d`, step 4
  `CurrFrame[ plane ][ y+i ][ x+j ] = Clip1( CurrFrame[..] +
  Dequant[i][j] )`. Integer division truncates toward zero per §4.1.
* `reconstruct_intra_block( .. )` — the end-to-end one-block driver:
  predicts via the round-10 §8.5.1 `predict_intra`, derives the
  `TxType` with the §6.4.25 `TX_32X32` / lossless `DCT_DCT` overrides,
  then runs `reconstruct_block`. This is the single-block shape the
  deferred §6.4.21 residual loop will drive once it threads real
  availability and quantizer state.

The §6.4.21 residual loop — which supplies the real per-block `Tokens`
arrays, availability flags and segment/quantizer state across a whole
frame, and wires this driver into a public decode path — lands in a
later round. The round-11 surface is internal-only; the public API
still exposes `parse_uncompressed_header`, `parse_compressed_header`
and their result types exclusively.

**Round 10: §8.5.1 intra prediction process.** Round 10 adds an
`intra` module (crate-internal, `pub(crate)`) implementing the §8.5.1
intra prediction the §8.6.2 reconstruct process invokes for intra
blocks (the prediction half of `reconstruct = predict + residual`):

* `PredMode` — the 10 §7.4.5 intra prediction modes, with
  discriminants matching the spec numbering exactly (`DC_PRED` = 0,
  `V_PRED` = 1, `H_PRED` = 2, `D45_PRED` = 3, `D135_PRED` = 4,
  `D117_PRED` = 5, `D153_PRED` = 6, `D207_PRED` = 7, `D63_PRED` = 8,
  `TM_PRED` = 9), plus `from_raw` for the (deferred) mode-info decode.
* `Plane` — a minimal row-major `i32` plane buffer standing in for
  `CurrFrame[ plane ]`, read for neighbour samples and written with
  the prediction.
* `predict_intra( .. )` (§8.5.1) — builds the `aboveRow[-1 .. 2*size-1]`
  and `leftCol[0 .. size-1]` neighbour arrays per the `haveAbove` /
  `haveLeft` / `notOnRight` availability rules (including the
  upper-right extension that fires only for `txSz == 0` and the
  `(1<<(BitDepth-1)) ± 1` no-neighbour fills), then forms the `pred`
  block for the selected mode: `V`/`H` copies, the four `DC` neighbour
  cases (`avg` / `leftAvg` / `aboveAvg` / midpoint), the
  `D45`/`D63`/`D117`/`D135`/`D153`/`D207` directional `Round2`
  recurrences, and `TM` with `Clip1`. Neighbour reads clamp with
  `Min(maxX, .)` / `Min(maxY, .)`; the result is stored back into the
  plane.

The §8.6.2 reconstruct driver — which supplies the real `haveAbove` /
`haveLeft` / `notOnRight` flags (from tile / frame edges) and adds the
round-9 inverse-transformed residual to this prediction — lands in a
later round. The round-10 surface is internal-only; the public API
still exposes `parse_uncompressed_header`, `parse_compressed_header`
and their result types exclusively.

**Round 9: §8.7 inverse transform process.** Round 9 adds an `idct`
module (crate-internal, `pub(crate)`) implementing the §8.7 inverse
transform stage the §8.6.2 reconstruct process invokes after the
round-8 dequantization step:

* The §8.7.1.1 butterfly primitives — `B` (butterfly rotation, with
  the `16 + 32*k` two-multiply fast path), `H` (Hadamard rotation),
  `SB` (butterfly into the high-precision `S` array) and `SH`
  (Hadamard rotation + `Round2(·, 14)` out of `S`) — plus `cos64` /
  `sin64` backed by the verbatim 33-entry `cos64_lookup` quarter-wave
  table and the `brev` bit-reversal helper. Fixed-point intermediates
  are evaluated in `i64` (the spec notes `S` needs `24 + BitDepth`
  bits of precision).
* `inverse_dct( t, n )` (§8.7.1.2 + §8.7.1.3) — the inverse-DCT array
  bit-reversal permutation followed by the recursive inverse DCT
  process for `2 <= n <= 5` (4/8/16/32-point).
* `inverse_adst( t, n )` (§8.7.1.4 .. §8.7.1.9) — the ADST
  input/output permutations and the ADST4 / ADST8 / ADST16 processes
  (with the `SINPI_1_9 .. SINPI_4_9` constants transcribed verbatim)
  dispatched by `n` for `2 <= n <= 4`.
* `inverse_wht( t, shift )` (§8.7.1.10) — the in-place inverse
  Walsh-Hadamard transform.
* `inverse_transform_2d( dequant, n, tx_type, lossless )` (§8.7.2) —
  the 2D driver applying the per-`TxType` row transform then column
  transform over a `(1<<n)` by `(1<<n)` block, the lossless WHT path
  (`shift = 2` rows / `0` columns), and the `Round2( T[i], Min(6,
  n+2) )` column rounding. `TxType` constants `DCT_DCT` / `ADST_DCT`
  / `DCT_ADST` / `ADST_ADST` follow §3.

The §8.6.2 reconstruct driver — which builds the `Dequant` input
(round-7 token magnitudes scaled by the round-8 quantizers, with the
`dqDenom = 2` halving for `TX_32X32`), calls this transform layer and
adds the residual to the prediction — lands in a later round. The
round-9 surface is internal-only; the public API still exposes
`parse_uncompressed_header`, `parse_compressed_header` and their
result types exclusively.

**Round 8: §8.6.1 dequantization functions.** Round 8 adds a
`dequant` module (crate-internal, `pub(crate)`) implementing the
quantizer-value derivation the §8.6.2 reconstruct process consumes
between the round-7 coefficient-token decode and the §8.7 inverse
transform:

* `dc_q( bit_depth, b )` / `ac_q( bit_depth, b )` (§8.6.1) — index
  the `dc_qlookup[3][256]` / `ac_qlookup[3][256]` tables by the
  `(BitDepth - 8) >> 1` row (0 / 1 / 2 for 8- / 10- / 12-bit) and the
  `Clip3(0, 255, b)` column. Both 256-entry tables are transcribed
  verbatim from the §8.6.1 listing into `DC_QLOOKUP` / `AC_QLOOKUP`.
* `seg_feature_active( seg, segment_id, feature )` (§6.4.9) —
  `segmentation_enabled && FeatureEnabled[ segment_id ][ feature ]`.
* `get_qindex( seg, quant, segment_id )` (§8.6.1) — the per-block
  quantizer index. When `seg_feature_active( SEG_LVL_ALT_Q )`, the
  segment's `FeatureData` either replaces `base_q_idx` (absolute
  update) or offsets it (delta update), then `Clip3(0, 255, .)`;
  otherwise `base_q_idx` is returned directly.
* `get_dc_quant( plane, .. )` / `get_ac_quant( plane, .. )`
  (§8.6.1) — combine `get_qindex()` with the plane-appropriate
  header delta (`delta_q_y_dc` luma DC, `delta_q_uv_dc` chroma DC,
  `delta_q_uv_ac` chroma AC; luma AC has no delta in VP9) and
  dispatch to `dc_q` / `ac_q`.

The §8.6.2 reconstruct driver — which scales the round-7 `Tokens`
array by these quantizers (with the `dqDenom = 2` halving for
`TX_32X32`), runs the §8.7 inverse transform and adds the residual
to the prediction — lands in a later round. The round-8 surface is
internal-only; the public API still exposes
`parse_uncompressed_header`, `parse_compressed_header` and their
result types exclusively.

**Round 7: §6.4.24 / §6.4.26 coefficient-token decoder.** Round 7
adds a `tokens` module (crate-internal, `pub(crate)`) implementing
the pieces needed to recover one quantised DCT coefficient at a time
from the §9.2 Boolean coder, given a pre-selected
`coef_probs[ txSz ][ plane>0 ][ is_inter ][ band ][ ctx ][..3 ]` cell:

* The §9.3 `token_tree[20]` walker — `read_token( coder, &probs )` —
  returning one of the 11 spec tokens (`ZERO_TOKEN` through
  `DCT_VAL_CATEGORY6`).
* The §9.3.2 `pareto( node, prob )` helper backed by the §10.3
  128×8 `pareto_table` (transcribed verbatim).
* The §6.4.24 `more_coefs` `B(p)` reader.
* The §6.4.26 `read_coef( coder, token, bit_depth )` extra-bits
  decoder consuming the `extra_bits[11][3]` and `cat_probs[7][14]`
  tables (transcribed verbatim) plus the CAT6 `high_bit` `B(255)`
  loop that prepends `BitDepth - 8` MSBs for 10- and 12-bit profiles
  at shift positions `5 + BitDepth - e`.
* A `read_coef_token` driver that wires `more_coefs` + `read_token` +
  `read_coef` + `L(1) sign_bit` together and returns
  `CoefStep::{ Eob, Coef { token, value } }`.
* The §10.2 `energy_class[12]` table (for `TokenCache` once the
  residual driver lands).

The hand-traced golden buffers in the test suite cover every token
slot's "all-zero extra bits" base value, both legs of the
`more_coefs` decision, and two non-trivial tree-walk paths
(`ONE_TOKEN` from buffer `0x40 0x00 …`, `TWO_TOKEN` from buffer
`0x60 0x00 …`).

The §6.4.21 `residual( )` plane/sub-block driver still lands in a
later round — it owns the `AboveNonzeroContext` / `LeftNonzeroContext`
arrays and the per-block `ctx` derivation that picks the right
`coef_probs` cell. The round-7 surface is internal-only; the public
API still exposes `parse_uncompressed_header` and
`parse_compressed_header` exclusively.

**Round 6: §6.3.7 `read_coef_probs` 6D coefficient-probability sweep
wired in.** Round 6 inserts the §6.3.7 walker between the round-5
§6.3.2 `tx_mode_probs` and §6.3.8 `read_skip_prob` calls. The
walker:

* Visits `txSz ∈ [TX_4X4, maxTxSize]` where `maxTxSize =
  tx_mode_to_biggest_tx_size[ tx_mode ]` per §10.5 (1, 2, 3, 4 or 4
  active tx-size slabs for `ONLY_4X4` / `ALLOW_8X8` / `ALLOW_16X16`
  / `ALLOW_32X32` / `TX_MODE_SELECT` respectively).
* For each active slab, reads an outer `L(1) update_probs` flag.
* If 1, walks the nested `(i, j, k, l, m)` sweep — `BLOCK_TYPES=2 ×
  REF_TYPES=2 × COEF_BANDS=6 × maxL(k) × UNCONSTRAINED_NODES=3`
  cells where `maxL = (k == 0) ? 3 : 6` (band 0 has only 3 valid
  previous-coef contexts; the §10 listing trails it with `{0, 0, 0}
  // unused` rows).
* Each cell becomes `read_diff_update_prob( coder, cell )` against
  the running `coef_probs[ txSz ][ i ][ j ][ k ][ l ][ m ]` table.

When `update_probs == 1` the inner sweep traverses `2 × 2 × (3 + 5
× 6) × 3 = 396` cells per active tx-size, totalling up to `4 × 396
= 1584` `diff_update_prob` calls for a fully-active
`TX_MODE_SELECT` frame.

Initial probabilities come from the §10 `default_coef_probs[
TX_SIZES ][ BLOCK_TYPES ][ REF_TYPES ][ COEF_BANDS ][
PREV_COEF_CONTEXTS ][ UNCONSTRAINED_NODES ]` listing (1728 u8
entries) transcribed verbatim into a new `DEFAULT_COEF_PROBS`
constant in `src/coef_probs.rs`. The public type alias `CoefProbs`
names the 6D shape so callers of [`parse_compressed_header`] can
match on the new `Vp9CompressedHeader::coef_probs` field.

The §6.3.7 walker is **structural-parse only**: it advances the
§9.2 Boolean coder by exactly the bits the spec dictates and
updates the running `coef_probs` table, but the actual entropy
decode of residual coefficients (§6.4) still lands in a later
round.

**Round 5: §6.3.2 `tx_mode_probs` + §6.3.8 `read_skip_prob` sweeps
wired into the compressed-header walker.** Round 5 consumes the
round-4 `read_diff_update_prob` primitive in two table sweeps:

* `tx_mode_probs( )` (§6.3.2) — gated on `tx_mode == TX_MODE_SELECT`,
  walks `tx_probs_8x8[2][1]`, `tx_probs_16x16[2][2]`,
  `tx_probs_32x32[2][3]` (12 cells total) via
  `read_diff_update_prob` starting from the §10 `default_tx_probs`
  initials transcribed verbatim into `DEFAULT_TX_PROBS`.
* `read_skip_prob( )` (§6.3.8) — unconditional 3-element
  (`SKIP_CONTEXTS = 3`) sweep over the §10 `default_skip_prob = {
  192, 128, 64 }` initials transcribed verbatim into
  `DEFAULT_SKIP_PROB`.

`Vp9CompressedHeader` now exposes the post-sweep `tx_probs` /
`skip_prob` tables. When `tx_mode != TX_MODE_SELECT` the §6.3
syntax skips `tx_mode_probs( )` entirely, so the field is left
equal to `DEFAULT_TX_PROBS`. `parse_compressed_header` runs both
sweeps in spec order: `read_tx_mode` → optional `read_tx_mode_probs`
→ `read_skip_prob`.

The inter-only §6.3.9+ syntax (`read_inter_mode_probs`,
`read_interp_filter_probs`, `read_is_inter_probs`,
`frame_reference_mode`, `mv_probs`) remains deferred — those fire
only on `FrameIsIntra == 0` and need reference-buffer state which
the uncompressed-header walker still rejects with
`Error::Unsupported`.

**Round 4: §6.3.3 `diff_update_prob` chain (`decode_term_subexp` +
`inv_remap_prob` + `inv_recenter_nonneg` + 255-entry
`inv_map_table`).** Round 4 lands the helper chain every §6.3.7+
probability sweep is built on:

* `read_diff_update_prob( coder, base_prob )` (§6.3.3) — reads the
  `B(252)` `update_prob` flag and, on 1, pulls a
  `decode_term_subexp` value (§6.3.4) then remaps the previous
  probability through `inv_remap_prob` (§6.3.5). On 0, passes the
  base probability straight through.
* `decode_term_subexp` (§6.3.4) — the 5-leg cascade
  (`L(1) → L(4)` / `L(1) → L(4)+16` / `L(1) → L(5)+32` /
  `L(7) → +64` / `L(7), L(1) → (v<<1)-1+bit`) yielding a value in
  `0..=254`.
* `inv_remap_prob` (§6.3.5) — the low-half / high-half piecewise
  remap, with the 255-entry `INV_MAP_TABLE` constant transcribed
  verbatim from the §6.3.5 listing.
* `inv_recenter_nonneg` (§6.3.6) — pure arithmetic helper.

The chain is structural — no caller in §6.3.2 / §6.3.7+ uses it yet.
That's deferred to the next round so this drop lands the primitive
in isolation. All four helpers are `pub(crate)` with explicit
`#[allow(dead_code)]` until the next round wires them into the
table sweeps.

**Round 3: §9.2 Boolean decoder + §6.3.1 `read_tx_mode` walk.** Building
on round 2's full §6.2 uncompressed-header walker, round 3 lands the
arithmetic-decode primitive — `init_bool( sz )` / `read_bool( p )` /
`read_literal( n )` / `exit_bool( )` per spec §9.2.1–§9.2.4 — plus the
first §6.3 compressed-header field (`read_tx_mode( )`, §6.3.1) exposed
via `parse_compressed_header(payload, lossless) -> Vp9CompressedHeader`.

Decoded `tx_mode` covers all five §3 values (`ONLY_4X4`, `ALLOW_8X8`,
`ALLOW_16X16`, `ALLOW_32X32`, `TX_MODE_SELECT`), including the lossless
short-circuit to `ONLY_4X4`. The §9.2.1 marker-bit zero-conformance
check is enforced (`InvalidBitstream` on a nonzero marker), and the
§9.2.2 `BoolMaxBits` underflow path is surfaced rather than silently
papered over.

The remaining §6.3.7+ syntax — `tx_mode_probs`, `read_coef_probs`,
`read_skip_prob`, `read_inter_mode_probs`,
`read_interp_filter_probs`, … — all funnel through
`read_diff_update_prob` (landed this round) and the table sweeps
themselves land in the next round. `read_inter_mode_probs` /
`read_interp_filter_probs` only fire on inter-frame headers, which
already return `Error::Unsupported` until reference-buffer state
lands.

**Round 2: full §6.2 uncompressed-header walk.** Building on round 1,
`parse_uncompressed_header(&[u8]) -> Result<Vp9FrameHeader, Error>` now
walks the entire `uncompressed_header()` syntax tree plus the §6.1.1
`trailing_bits()` zero-fill alignment. Per-field coverage:

* Round 1: `frame_marker`, `Profile` (with the `profile == 3`
  `reserved_zero` bit), `show_existing_frame` early-return path
  (returns `frame_to_show_map_idx`), `frame_type`, `show_frame`,
  `error_resilient_mode`, `frame_sync_code` (`0x49 / 0x83 / 0x42`),
  `color_config()` (bit depth 8/10/12, all 8 `color_space` values,
  `color_range`, `subsampling_x` / `subsampling_y` with the §7.2.2
  reserved-zero and CS_RGB-on-profile-0/2 constraint checks), and
  `frame_size` / `render_size` (with the
  `render_and_frame_size_different == 1` override).
* Round 2: post-`render_size` syntax — `refresh_frame_context`,
  `frame_parallel_decoding_mode`, `frame_context_idx`, the
  `setup_past_independence` `frame_context_idx = 0` reset for intra /
  error-resilient frames, `loop_filter_params()` (§6.2.8) including
  full delta-update walk with `s(6)` ref/mode deltas,
  `quantization_params()` (§6.2.9) with `read_delta_q()` (§6.2.10),
  `segmentation_params()` (§6.2.11) with `read_prob()` (§6.2.12) +
  per-segment / per-feature data using the `segmentation_feature_bits`
  / `segmentation_feature_signed` tables, `tile_info()` (§6.2.13)
  driven by `Sb64Cols` computed from `FrameWidth` per §6.2.6 /
  §6.2.14, the f(16) `header_size_in_bytes`, and the §6.1.1
  `trailing_bits()` zero-pad alignment with §7.1.1 zero-bit
  conformance check.

`decode_vp9()` / `encode_vp9()` still return `Error::NotImplemented`;
the compressed header (§6.3) plus entropy decoder, intra/inter
prediction, transforms and loop filter land in later rounds. Inter
(non-intra-only) headers — which require `frame_size_with_refs` and
reference-buffer state — return `Error::Unsupported` from the header
walker for now.

The `Vp9FrameHeader` struct now also exposes
`uncompressed_header_size_bytes` (the byte-aligned offset at which the
§6.3 compressed header starts), giving callers everything needed to
slice the compressed-header payload of `header_size_in_bytes` from the
remainder of the frame.

## Test surface

* `cargo test`: 285 unit tests + 20 integration tests (8 in
  `tests/compressed_header.rs` plus 12 in
  `tests/uncompressed_header.rs`).
* Round-18 additions: 37 unit tests covering the §3 partition
  enumeration (`PARTITION_NONE` / `_HORZ` / `_VERT` / `_SPLIT` =
  0..=3) and dimensions (`PARTITION_TYPES = 4`,
  `PARTITION_CONTEXTS = 16`); the four §10.2 lookups
  (`B_WIDTH_LOG2_LOOKUP` / `B_HEIGHT_LOG2_LOOKUP` /
  `MI_WIDTH_LOG2_LOOKUP` / `NUM_8X8_BLOCKS_WIDE_LOOKUP`) against
  the spec listings; `SUBSIZE_LOOKUP` `PARTITION_NONE` identity,
  `PARTITION_SPLIT` superblock anchors (`BLOCK_8X8 -> BLOCK_4X4`,
  …, `BLOCK_64X64 -> BLOCK_32X32`), `PARTITION_HORZ` /
  `PARTITION_VERT` superblock anchors, and the
  `BLOCK_INVALID = 14` cell at non-square parents; the three
  §9.3.1 trees (`PARTITION_TREE[6]`, `COLS_PARTITION_TREE[2]`,
  `ROWS_PARTITION_TREE[2]`) verbatim transcription;
  `KF_PARTITION_PROBS` + `DEFAULT_PARTITION_PROBS` shape + four
  per-table listing anchors + §9.2 minimum-probability sanity
  check; `partition_plane_context` zero-strips + above-bit-set +
  left-bit-set + both-set across each of the four superblock
  sizes, the OR-fold across the strip width, unrelated-bit
  masking, the panic-on-`BLOCK_INVALID` / mismatched-strip
  guards, and an exhaustive sweep that proves all 16
  `ctx ∈ 0..=15` are reachable across (`bsize`, `above_bit`,
  `left_bit`); `decode_partition_type` against the zero coder
  (`PARTITION_NONE` / `_HORZ` / `_VERT` for the four arm
  combinations), the all-ones coder (interior right-branch walk
  → `PARTITION_SPLIT`), the one-then-zero coder (interior →
  `PARTITION_HORZ`, right-edge → `PARTITION_SPLIT`, bottom-edge →
  `PARTITION_SPLIT`, with a follow-up confirming the same outcome
  for uniform probs = 255), the corner (`(false, false)`) case
  with a bool-coder reuse probe proving zero bits are consumed,
  and an exhaustive 4 × 3 (arm × buffer) smoke-test confirming
  the output stays in `0..=3`.
* Round-16 additions: 7 unit tests covering the §9.3.1
  `segment_tree[14]` verbatim transcription, `read_segment_id` against
  the zero coder (every bit=0 walks `tree[0]=2 → tree[2]=6 →
  tree[6]=0` and returns segment 0), the bias-buffer + `[255; 7]`
  trace producing segment id 4 (first read flips to bit=1 picking
  inner branch 4, the next two collapse to bit=0 picking leaf -4), the
  §9.3.1 node-index packing fact that a pure-left walk visits node
  indices `{0, 1, 3}` (not contiguous), `intra_segment_id` short-
  circuiting to `segment_id = 0` when either `segmentation_enabled`
  or `segmentation_update_map` is false (without dereferencing
  `tree_probs`), the active path decoding through `read_segment_id`,
  and the `Error::InvalidBitstream` surface when the caller forgets
  to thread `tree_probs` despite the active gate.
* Round-15 additions: 22 unit tests covering the §9.3.1 tree-listing
  anchors (`tx_size_8_tree` / `tx_size_16_tree` / `tx_size_32_tree` /
  `binary_tree` verbatim transcription), the §9.3.3 `tree_decode`
  walker with a zero-coder (every read yields bit=0, every tree picks
  its first leaf), a "bias buffer" prefix `[0x7F, 0x00, ...]` whose
  post-marker coder state lets the first `read_bool(255)` flip to 1
  (one right-branch step), the node-index argument-order invariant of
  the prob callback, exhaustive `skip_context` cases (no neighbours,
  single-side, both-sides mixed and matching), `tx_size_context`
  against the §9.3.2 listing across `max_tx_size = 0..=3` plus the
  skipped-neighbour fallback and the `!AvailL` / `!AvailU` mirroring,
  the §6.4.8 `seg_feature_active(SEG_LVL_SKIP)` early-return path,
  `read_skip` against both the zero coder (always false) and the bias
  coder + `p=255` (true), the §6.4.10 `else`-branch
  `Min(maxTxSize, biggest)` fallback for `allow_select == false` /
  non-SELECT `tx_mode` / sub-8x8 `MiSize`, the §9.3.1 tree-dispatch
  for `BLOCK_8X8` / `BLOCK_16X16` / `BLOCK_32X32`, and the row-by-ctx
  selection correctness via lockstep against `tx_size_context` at
  both ctx=0 and ctx=1.
* Round-14 additions: 12 unit tests covering the §10.2
  `num_4x4_blocks_wide_lookup` / `_high_lookup` and §6.4.10
  `max_txsize_lookup` listings, the §6.4.23 `ss_size_lookup` luma
  identity invariant + the 4:2:0 / asymmetric-subsampling anchors
  (`BLOCK_8X8 -> BLOCK_4X4`, `BLOCK_64X64 -> BLOCK_32X32`, the
  `(1,0)` / `(0,1)` mixed chroma cases), the §6.4.22 `get_uv_tx_size`
  chroma cap (`MiSize=BLOCK_16X16 -> chroma TX_8X8`) and the sub-8x8
  short-circuit, the `skip = true` path leaving every `AboveNonzero`
  / `LeftNonzero` cell at 0 (no token decode), the full `skip = false`
  walk firing `token_source` exactly 16 luma + 4 U + 4 V times for a
  `BLOCK_16X16` MI block at `tx_size = TX_4X4` (each call recorded
  with `(plane, block_idx, tx_sz)`), the `nonzero = true` strip
  write-back over `step = 1 << tx_sz` 4-sample units for `tx_size =
  TX_8X8`, the out-of-bounds (`startX >= maxx`) block skip with
  intact zero context, a DC-only luma block at MI (1,1) lockstep
  against an independent `predict_intra` + `reconstruct_block` probe
  (proving the residual loop ties the rounds 10/11/13 pieces
  identically), and the §6.4.21 `bsize = max(MiSize, BLOCK_8X8)`
  widening for a `BLOCK_4X4` MI block (4 luma + 1 U + 1 V blocks
  decoded).
* Round-13 additions: 15 unit tests covering `coefband_4x4` /
  `coefband_8x8plus` against the §10 listing (21-entry prefix + all-`5`
  tail) and the `coef_band` tx-size dispatch, the §9.3.2
  `token_cache_neighbours` derivation (DC origin, interior DCT_DCT,
  the `DCT_ADST` / `ADST_DCT` double-neighbour variants, first-row /
  first-column fallbacks, and the `n = 4 << txSz` 8x8 width scaling),
  `build_token_probs` node mapping (node 0 → `cell[1]`, node 1 →
  `cell[2]`, node `2..=9` → `pareto`), and the `tokens( )` driver
  itself: a zero-buffer immediate EOB (returns `nonzero = false`, all
  `Tokens` zeroed), the `ZERO_TOKEN`-clears-`checkEob` block fill
  (`nonzero = true`, explicit zeros), a lockstep equality against an
  independent `read_coef_token` walk over the same scan / buffer
  (`Tokens` + `TokenCache` + return value), the DC non-zero-context
  cell routing (a non-zero strip selects the `ctx = 2` cell), the
  trailing `c..segEob` zero-fill on an 8x8 block, and the
  `NonzeroContext::new` all-zero invariant.
* Round-12 additions: 10 unit tests covering every §10.1 scan table's
  spec length, the permutation invariant (each raster position appears
  exactly once — the property the §6.4.24 loop relies on to zero
  untouched coefficients), the DC-first invariant, §10.1 listing
  anchors (first four + last entry of each table), the §6.4.25
  `txType` → table selection for 4x4 / 8x8 / 16x16, the
  `TX_32X32`-always-default and chroma-forces-`DCT_DCT` first-half
  overrides, and the `16 << (txSz << 1)` `segEob` scan-length match.
* Round-11 additions: 10 unit tests covering the `mode2txfm_map` intra
  prefix vs the §10.5 listing, the local `clip1` range clamping, an
  all-zero `Tokens` block leaving the prediction unchanged, a DC-only
  `Tokens` block adding a flat residual to a flat prediction, step-4
  `Clip1` saturation at both the bit-depth max and zero, the
  `TX_32X32` `dqDenom = 2` halving exercised through the real
  `reconstruct_block`, and three `reconstruct_intra_block` end-to-end
  cases (DC_PRED + zero tokens equalling the pure DC prediction,
  DC_PRED + a known DC residual reconstructing the expected pixels,
  and the lossless WHT path).
* Round-10 additions: 16 unit tests covering `PredMode::from_raw`
  round-tripping the §7.4.5 numbering (and rejecting 10+), the local
  `round2` / `clip1` helpers against their §3 definitions, `V_PRED`
  copying `aboveRow` down and `H_PRED` copying `leftCol` across, all
  four `DC_PRED` neighbour cases (both / left-only / above-only /
  none), `TM_PRED`'s `Clip1(aboveRow[j] + leftCol[i] - aboveRow[-1])`
  formula (including out-of-range clipping), a constant-neighbour
  invariant that collapses every directional mode to the neighbour
  constant across all four transform sizes, the `D207_PRED` bottom-row
  / step-2 formula, the `notOnRight`-gated upper-right extension of
  `aboveRow` (enabled only for `txSz == 0`) via `D45_PRED`, and the
  `Min(maxX, .)` plane-edge clamping of neighbour reads.
* Round-8 additions: 13 unit tests covering the `DC_QLOOKUP` /
  `AC_QLOOKUP` shape (256 entries per row) and §8.6.1 listing
  anchors (first/last entry of every row plus two interior anchors),
  `qlookup_row` bit-depth mapping (8→0 / 10→1 / 12→2), `clip3`
  both-end clamping, `dc_q` / `ac_q` index clipping + bit-depth row
  selection, `get_qindex` for the segmentation-off / delta-update /
  absolute-update paths (with the `Clip3` clamps), `seg_feature_active`
  needing both `segmentation_enabled` and the per-segment bit,
  `get_dc_quant` / `get_ac_quant` plane-delta selection (luma AC has
  no delta), a segment-overridden qindex threaded through both, and
  the high-bit-depth divergence of the same qindex across the three
  table rows.
* Round-6 additions: 7 unit tests covering
  `tx_mode_to_biggest_tx_size` against the §10.5 listing,
  `read_coef_probs` zero-buffer passthrough for both `ONLY_4X4`
  (1-slab) and `TX_MODE_SELECT` (4-slab) paths plus an across-mode
  sweep, `DEFAULT_COEF_PROBS` shape (4 × 2 × 2 × 6 × 6 × 3 = 1728
  entries) + four hand-picked anchor values from the §10 listing,
  the band-0 unused-row sentinel invariant (every `(txSz, i, j)`
  triple's band 0 has `{0, 0, 0}` at contexts 3..6), and the inner
  sweep cell-count check `2 × 2 × (3 + 5 × 6) × 3 = 396`. 2 new
  end-to-end integration tests (`tests/compressed_header.rs`)
  splicing a TX_MODE_SELECT frame and an ONLY_4X4 frame through
  the full pipeline and verifying §10 default anchors survive the
  zero-buffer §6.3.7 sweep verbatim.
* Round-5 additions: 10 unit tests covering `DEFAULT_TX_PROBS`
  shape + value spot-check against the §10 listing, `DEFAULT_SKIP_PROB
  = [192, 128, 64]`, `read_tx_mode_probs` on a zero buffer
  (defaults pass through; row 0 (`TX_4X4`) is never touched), the
  12-cell loop-count shape, `read_skip_prob` on a zero buffer plus
  a 3-context-visit sanity check, and three `parse_compressed_header`
  integration tests confirming the §10 defaults survive the
  zero-buffer sweep for the `ONLY_4X4`, lossless, and `TX_MODE_SELECT`
  paths. 2 new end-to-end integration tests
  (`tests/compressed_header.rs`): one driving the
  `TX_MODE_SELECT` → `tx_mode_probs` → `read_skip_prob` chain from
  a 64×64 uncompressed-header splice, one confirming `ALLOW_16X16`
  skips the §6.3.2 sweep.
* Round-4 additions: 16 unit tests covering `inv_recenter_nonneg`
  (all three piecewise branches plus the `v == 2*m` boundary),
  `INV_MAP_TABLE` length + spot-checks against the spec listing
  anchors (`[0]=7`, `[19]=254`, `[20]=1`, `[254]=253` with the
  duplicated trailing 253), `inv_remap_prob` covering both
  low-half and high-half `(m << 1) ≤ 255` branches plus the
  `v > 2m` short-circuit, `decode_term_subexp` against
  hand-derived §9.2 buffers (leg-1 zero plus a sweep that confirms
  the result stays in `0..=254` for every first-byte family), and
  `read_diff_update_prob` confirming the `update_prob == 0`
  passthrough against the full 1..=255 base-probability sweep.
* Round-3 additions: 9 `bool_coder` unit tests covering
  `init_bool` (zero-size / short-slice / nonzero-marker rejection +
  the zero-buffer accept path), `read_bool` against hand-traced
  golden buffers (mixed probabilities, extreme p=255 run), `read_literal`,
  and `exit_bool` accept/reject paths; plus 7 `compressed` unit tests
  for `parse_compressed_header` covering each `TxMode` value
  (lossless short-circuit, `ONLY_4X4`/`ALLOW_8X8`/`ALLOW_16X16`/
  `ALLOW_32X32`/`TX_MODE_SELECT` golden buffers) and the
  marker / empty-buffer rejection paths; and 4 end-to-end
  integration tests (`tests/compressed_header.rs`) that build a
  64x64 key-frame uncompressed header, splice a §9.2 payload past
  the byte-aligned `trailing_bits` pad, and verify the walker
  picks up the right `tx_mode` from the spliced payload.
* Uncompressed-header coverage (rounds 1 + 2, unchanged) spans the
  four profiles, studio/full-swing color ranges, render-size
  overrides, the `show_existing_frame` early return, the intra-only
  inter-frame branch (with the spec's BT.601 / 4:2:0 / 8-bit defaults
  for Profile 0), full `loop_filter_params` delta update with mixed
  `update_ref_delta` / `update_mode_delta` flags and signed `s(6)`
  deltas, `quantization_params` with a nonzero `base_q_idx` and
  signed `delta_q_y_dc`, full segmentation with `update_map` +
  `temporal_update` + `update_data` driving the per-segment /
  per-feature inner loop including the 0-magnitude-bit skip feature,
  `tile_info` increment-walk for a 4K-wide frame, plus three failure
  paths (bad `frame_marker`, bad `frame_sync_code`, truncated buffer)
  and the §7.1.1 nonzero trailing-bit rejection.
* No external fixtures are involved yet; each test constructs its
  input bit-by-bit (and §9.2 golden buffers are hand-derived by
  stepping the decoder, not borrowed from any third-party VP9
  implementation).

## Provenance

Single source of truth: VP9 Bitstream & Decoding Process Specification
v0.7 (`docs/video/vp9/vp9-spec.txt`) plus any clean-room trace material
staged under `docs/video/vp9/`. The workspace clean-room rule applies
in full: only material in this project's `docs/` tree informs the
code. Black-box `ffmpeg` binary invocations remain permissible as
opaque validators but are not yet wired into the test harness.

## Roadmap

Future rounds, roughly in order:

1. Per-block mode-info decode (`y_mode` / `sub_modes` / `tx_size` /
   `skip` / `segment_id`) per §6.4.6 / §6.4.7 / §6.4.10. Round 15
   landed §6.4.8 `read_skip` + §6.4.10 `read_tx_size` + the §9.3.3
   `tree_decode` walker; round 16 added §6.4.7 `intra_segment_id` +
   the §9.3.1 `segment_tree[14]`; round 17 added the §6.4.6
   `intra_frame_mode_info()` keyframe orchestrator on top of the
   §9.3.1 `intra_mode_tree[18]` + §10.5 `kf_y_mode_probs` /
   `kf_uv_mode_probs` decode (`default_intra_mode` plus
   `default_uv_mode`, with the `MiSize < BLOCK_8X8` `(idy, idx)`
   sub-block walk fanning a per-cell `default_intra_mode` into the
   `sub_modes[ ]` array, and the §9.3.2 above/left neighbour
   derivation handled by the `IntraFrameNeighbours` bundle). The
   remaining intra piece is §6.4.15 `intra_block_mode_info` — the
   intra-block branch used inside inter frames, which uses
   `y_mode_probs[size_group_lookup[MiSize]][node]` /
   `y_mode_probs[0][node]` / `uv_mode_probs[y_mode][node]` from the
   compressed header rather than the keyframe `kf_*_mode_probs`
   tables; it lands alongside the §6.4.11 inter-frame driver. Once
   that lands the per-block `BoolCoder` token decode can replace the
   round-14 `TokenSource` callback to expose a public single-MI-block
   intra decode path; the partition-tree walk (§6.4.3) closes the
   per-tile loop.
2. Inter (non-intra-only) header path — `frame_size_with_refs`,
   `allow_high_precision_mv`, `read_interpolation_filter` — plus
   the inter-only §6.3.9–§6.3.16 syntax (`read_inter_mode_probs`,
   `read_interp_filter_probs`, `read_is_inter_probs`,
   `frame_reference_mode`, `mv_probs`) once reference-buffer state
   is in place.
3. Per-tile partition-tree walk (§6.4) including the §6.4.21
   `residual()` driver that finally consumes the round-7 tokens and
   the round-6 `coef_probs` tables, plus the per-block mode-info
   decode that feeds `predict_intra`. Round 18 added the §6.4.3
   `decode_partition_type()` per-call primitive plus the §10.2
   `subsize_lookup` / `b_*_log2_lookup` / `num_8x8_blocks_wide_lookup`
   tables and the §10.4 `kf_partition_probs` / §10.5
   `default_partition_probs` probability tables; the recursive
   `decode_partition(r, c, bsize)` driver that splits on the
   decoded `partition`, threads `subsize_lookup[partition][bsize]`
   into four recursive calls when `PARTITION_SPLIT`, and writes
   back `AbovePartitionContext[]` / `LeftPartitionContext[]`
   with `15 >> b_*_log2_lookup[subsize]` lands in the next round,
   alongside the §6.3 `read_partition_probs()` compressed-header
   sweep that updates `default_partition_probs` on inter frames.
4. Inter prediction (§8.5.2), loop filter (§8.8), multi-tile, then
   encoder paths.

## License

MIT. See `LICENSE`.
