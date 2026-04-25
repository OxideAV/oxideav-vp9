# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.6](https://github.com/OxideAV/oxideav-vp9/compare/v0.0.5...v0.0.6) - 2026-04-25

### Other

- drop oxideav-codec/oxideav-container shims, import from oxideav-core

## [0.0.5](https://github.com/OxideAV/oxideav-vp9/compare/v0.0.4...v0.0.5) - 2026-04-24

### Other

- skip-context infrastructure + §6.4.25 32x32 tx_type fix
- skip EOB read after ZERO_TOKEN in decode_coefs (§6.4.24)
- track AboveNonzeroContext / LeftNonzeroContext per §6.4.22
- MV probs sourced from per-frame context (§6.3.16 updates)
- inter tile partition_probs from per-frame context
- carry probability contexts across frames via §8.10 saved slots
- plumb skip / is_inter / inter_mode / ref / filter / intra-mode probs
- plumb per-frame coef_probs through detokenize
- add §6.3 compressed-header probability-update decode
- inter-mode context derivation (§6.5 counter_to_context)
- bit-accurate MV candidate list construction (§6.5)
- per-block segmentation decode (§6.4.7 / §6.4.12 / §6.4.14)
- vp9 encoder: README + YuvFrame placeholder for pixel sources
- vp9 encoder: ffmpeg-acceptance + spec-correct compressed header
- document partition-context indexing convention
- vp9 encoder: keyframe tile/partition walk + DC_PRED block emit
- neighbour-aware skip + is_inter contexts (§7.4.6)
- neighbour-aware partition + KF intra-mode contexts (§7.4.6)
- vp9 encoder: §6.3 compressed header emitter
- vp9 encoder: forward boolean (range) coder (§9.2 inverse)
- find_mv_refs + MV candidate list (§6.5)
- vp9 encoder: bit writer + §6.2 uncompressed header emitter
- superframe (Annex B) splitter + compound-fixture smoke test
- README + lib.rs docs for compound, scaled refs, multi-tile, seg-deltas
- multi-tile frame support (§6.4)
- apply SEG_LVL_ALT_Q / SEG_LVL_ALT_L segmentation deltas
- add compound prediction (§6.4.17 / §8.5.2) + scaled references (§8.5.2.3)
- document loop filter, add keyframe smoke test, tidy clippy
- wire loop filter into IntraTile + InterTile decode
- add loop filter (§8.8) module — standalone, not yet wired

## [0.0.4](https://github.com/OxideAV/oxideav-vp9/compare/v0.0.3...v0.0.4) - 2026-04-19

### Other

- document inter support + deferrals in README
- add P-frame fixture integration test
- wire inter-frame block decode + DPB refresh
- add 8-tap sub-pel interpolation filter (§8.5.1)
- add MV decode (§6.4.19) + default MV prob tables
- add 8-slot reference-frame DPB scaffold
- wire keyframe pixel reconstruction end-to-end
- add tables, reconintra, expanded transforms, detokenize
- bump oxideav-core / oxideav-codec dep examples to "0.1"
- migrate register() to CodecInfo builder
- bump oxideav-core + oxideav-codec deps to "0.1"
