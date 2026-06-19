//! Annex B superframe splitting (`vp9-spec.txt` §B.1-B.4).
//!
//! VP9 lets an encoder consolidate several coded frames into one chunk
//! ("superframe"). The enclosed frames are concatenated, then an index is
//! appended in the last up-to-34 bytes:
//!
//! ```text
//! superframe( sz ) {                                     §B.2
//!     for( i = 0; i < NumFrames; i++ )
//!         frame( frame_sizes[ i ] )
//!     superframe_index( )
//! }
//! superframe_index( ) {                                  §B.2.1
//!     superframe_header( )                               // trailing copy is at the START
//!     for( i = 0; i < NumFrames; i++ )
//!         frame_sizes[ i ]                               f(SzBytes)
//!     superframe_header( )                               // and again at the END
//! }
//! superframe_header( ) {                                 §B.2.2
//!     superframe_marker            f(3)                  // 0b110
//!     bytes_per_framesize_minus_1  f(2)
//!     frames_in_superframe_minus_1 f(3)
//! }
//! ```
//!
//! Detection per §B.4:
//!
//! 1. Parse the final byte of the chunk; the top three bits must equal the
//!    `superframe_marker` `0b110`.
//! 2. `SzIndex = 2 + NumFrames * SzBytes`.
//! 3. The first byte of the index (`chunk[ len - SzIndex ]`) must equal the
//!    final byte.
//!
//! If both checks pass the chunk is a superframe and each enclosed frame is
//! decoded in turn; otherwise the whole chunk is a single frame. The
//! frame-size fields are little-endian (`f(SzBytes)` here reads the size
//! bytes least-significant first, matching the §9.1 `f(n)` *byte* ordering
//! the index uses).
//!
//! This is VP9-intrinsic framing (Annex B of the bitstream spec), not a
//! separate container: every VP9 chunk a decoder receives may or may not
//! carry the index, and the split must happen before the per-frame header
//! walk. It therefore lives in the codec crate alongside the rest of the
//! §6.2 frame parse, mirroring the doctrine's "native framing" placement
//! for codecs whose framing is part of their own bitstream.
//!
//! Single source of truth: `docs/video/vp9/vp9-spec.txt` Annex B.

/// The §B.2.2 `superframe_marker`: the top three bits of the trailing (and
/// leading) index byte (`0b110`).
const SUPERFRAME_MARKER: u8 = 0b110;

/// Split a chunk into its enclosed coded frames per Annex B.
///
/// Returns the byte ranges of each enclosed frame, in decode order. When
/// the chunk is not a superframe (no valid §B.4 index) the whole chunk is
/// returned as a single frame — the §B.4 fallback.
///
/// The returned slices borrow from `chunk`; the superframe index bytes
/// themselves are never returned (they are not part of any coded frame).
///
/// An empty chunk yields an empty list.
pub fn split_superframe(chunk: &[u8]) -> Vec<&[u8]> {
    match superframe_frame_sizes(chunk) {
        Some(sizes) => {
            let mut out = Vec::with_capacity(sizes.len());
            let mut off = 0usize;
            for sz in sizes {
                // `superframe_frame_sizes` already verified the cumulative
                // sizes fit before the index, so the slice is in range.
                out.push(&chunk[off..off + sz]);
                off += sz;
            }
            out
        }
        None => {
            if chunk.is_empty() {
                Vec::new()
            } else {
                vec![chunk]
            }
        }
    }
}

/// Parse the §B.2.1 index and return the enclosed `frame_sizes[ ]` when the
/// chunk is a valid superframe, or `None` for the §B.4 single-frame
/// fallback.
///
/// `None` is returned when any §B.4 check fails: the trailing byte's marker
/// is wrong, the chunk is too short to hold the computed index, the leading
/// index byte does not match the trailing one, or the declared frame sizes
/// plus the index overflow the chunk. A malformed index is treated as
/// "not a superframe" rather than an error, exactly as §B.4 specifies.
fn superframe_frame_sizes(chunk: &[u8]) -> Option<Vec<usize>> {
    let len = chunk.len();
    if len == 0 {
        return None;
    }

    // §B.4 step 1 — the final byte must carry the superframe_marker.
    let marker_byte = chunk[len - 1];
    if (marker_byte >> 5) != SUPERFRAME_MARKER {
        return None;
    }

    // §B.2.2 — decode the trailing superframe_header.
    let bytes_per_framesize = ((marker_byte >> 3) & 0b11) as usize + 1; // SzBytes
    let num_frames = (marker_byte & 0b111) as usize + 1; // NumFrames

    // §B.4 step 2 — SzIndex = 2 + NumFrames * SzBytes.
    let sz_index = 2 + num_frames * bytes_per_framesize;
    if len < sz_index {
        return None;
    }

    // §B.4 step 3 — the leading index byte must equal the trailing one.
    let index_start = len - sz_index;
    if chunk[index_start] != marker_byte {
        return None;
    }

    // §B.2.1 — the NumFrames frame_sizes, little-endian f(SzBytes) each,
    // sit between the two superframe_header bytes.
    let mut sizes = Vec::with_capacity(num_frames);
    let mut p = index_start + 1;
    let mut total: usize = 0;
    for _ in 0..num_frames {
        let mut sz = 0usize;
        for b in 0..bytes_per_framesize {
            sz |= (chunk[p + b] as usize) << (8 * b);
        }
        p += bytes_per_framesize;
        // Guard against a declared size that overruns the frame payload
        // region (everything before the index); a conforming stream never
        // does, but a corrupt one must not panic the slice in `split`.
        total = total.checked_add(sz)?;
        if total > index_start {
            return None;
        }
        sizes.push(sz);
    }

    Some(sizes)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a superframe index byte (the value that appears at both ends
    /// of the §B.2.1 index).
    fn marker(bytes_per_minus_1: u8, frames_minus_1: u8) -> u8 {
        (SUPERFRAME_MARKER << 5) | ((bytes_per_minus_1 & 0b11) << 3) | (frames_minus_1 & 0b111)
    }

    /// Assemble a superframe chunk from explicit frame payloads, with a
    /// `bytes_per_framesize` of `szb` (1..=4).
    fn build(frames: &[&[u8]], szb: usize) -> Vec<u8> {
        let mut out = Vec::new();
        for f in frames {
            out.extend_from_slice(f);
        }
        let m = marker(szb as u8 - 1, frames.len() as u8 - 1);
        out.push(m);
        for f in frames {
            let sz = f.len();
            for b in 0..szb {
                out.push(((sz >> (8 * b)) & 0xff) as u8);
            }
        }
        out.push(m);
        out
    }

    #[test]
    fn non_superframe_returns_whole_chunk() {
        // A normal coded frame whose final byte is not a superframe_marker.
        let chunk = [0x82u8, 0x49, 0x83, 0x42, 0x00];
        let parts = split_superframe(&chunk);
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0], &chunk);
    }

    #[test]
    fn empty_chunk_yields_no_frames() {
        assert!(split_superframe(&[]).is_empty());
    }

    #[test]
    fn two_frame_superframe_one_byte_sizes() {
        let f0: Vec<u8> = (0..10u8).collect();
        let f1: Vec<u8> = (100..130u8).collect();
        let chunk = build(&[&f0, &f1], 1);
        let parts = split_superframe(&chunk);
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0], f0.as_slice());
        assert_eq!(parts[1], f1.as_slice());
    }

    #[test]
    fn single_frame_superframe_is_legal() {
        // §B.3 NOTE — NumFrames == 1 is legal.
        let f0: Vec<u8> = (0..40u8).collect();
        let chunk = build(&[&f0], 1);
        let parts = split_superframe(&chunk);
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0], f0.as_slice());
    }

    #[test]
    fn multibyte_framesize_round_trips() {
        // A frame larger than 255 bytes needs SzBytes >= 2.
        let f0: Vec<u8> = (0..300u32).map(|v| v as u8).collect();
        let f1: Vec<u8> = (0..17u8).collect();
        let chunk = build(&[&f0, &f1], 2);
        let parts = split_superframe(&chunk);
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].len(), 300);
        assert_eq!(parts[1].len(), 17);
        assert_eq!(parts[0], f0.as_slice());
    }

    #[test]
    fn mismatched_leading_index_byte_falls_back() {
        // §B.4 step 3: leading index byte must equal the trailing one.
        let f0: Vec<u8> = (0..10u8).collect();
        let f1: Vec<u8> = (0..20u8).collect();
        let mut chunk = build(&[&f0, &f1], 1);
        // Corrupt the leading index byte. SzIndex = 2 + NumFrames(2) *
        // SzBytes(1) = 4.
        let sz_index = 4;
        let lead = chunk.len() - sz_index;
        chunk[lead] ^= 0x01;
        let parts = split_superframe(&chunk);
        // Falls back to the whole chunk as one frame.
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0], chunk.as_slice());
    }

    #[test]
    fn declared_sizes_overrunning_chunk_falls_back() {
        // A marker that claims NumFrames=2 with sizes that exceed the
        // payload region must not panic; §B.4 fallback to single frame.
        let f0: Vec<u8> = (0..5u8).collect();
        let mut chunk = build(&[&f0], 1);
        // Hand-craft an index claiming two 200-byte frames in a tiny chunk.
        chunk.clear();
        chunk.extend_from_slice(&[1, 2, 3, 4, 5]);
        let m = marker(0, 1); // SzBytes=1, NumFrames=2
        chunk.push(m);
        chunk.push(200);
        chunk.push(200);
        chunk.push(m);
        let parts = split_superframe(&chunk);
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0], chunk.as_slice());
    }

    #[test]
    fn chunk_too_short_for_index_falls_back() {
        // Final byte looks like a marker but the chunk is shorter than the
        // computed SzIndex.
        let chunk = [marker(3, 7)]; // claims SzBytes=4, NumFrames=8 -> SzIndex=34
        let parts = split_superframe(&chunk);
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0], &chunk);
    }

    #[test]
    fn three_frame_superframe_with_hidden_first() {
        // Mirrors the superframe-2 fixture shape: a hidden ARF followed by
        // a visible frame, both inside one chunk.
        let arf: Vec<u8> = (0..50u8).collect();
        let vis: Vec<u8> = (0..120u8).collect();
        let chunk = build(&[&arf, &vis], 1);
        let parts = split_superframe(&chunk);
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0], arf.as_slice());
        assert_eq!(parts[1], vis.as_slice());
    }
}
