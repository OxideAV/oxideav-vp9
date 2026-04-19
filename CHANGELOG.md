# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
