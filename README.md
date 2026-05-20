# oxideav-vp9

Pure-Rust VP9 codec.

## Status — 2026-05-20

**Orphan-rebuild scaffold.** The crate's prior implementation was
retired under the workspace clean-room policy: provenance for several
core decoder modules could not be defended against the "no external
library source as reference" rule that governs every crate in this
workspace.

Per workspace policy, the only acceptable response is a full
clean-room re-implementation against the VP9 standards documents and
black-box validator binaries. That work has not yet been scheduled.

Every public entry point currently returns `Error::NotImplemented`.

## Planned clean-room sources

The clean-room rebuild will consult only:

* VP9 Bitstream & Decoding Process Specification (v0.7) as published
  at the WebM project's spec page (snapshotted into
  `docs/video/vp9/`).
* RFC 6386 / RFC 7741 references where they overlap with the VP9
  bitstream as documented in the v0.7 spec.
* Black-box invocations of `ffmpeg` (the binary — not its source) as
  an opaque validator.

No external library source — libvpx, libaom, FFmpeg's `libavcodec/vp9*`,
etc. — is permitted as a reference under the workspace clean-room
policy.

## License

MIT. See `LICENSE`.
