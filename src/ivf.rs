//! IVF container demux helpers.
//!
//! IVF is a trivial wrapper the libvpx / ffmpeg toolchains use around raw
//! VP8 / VP9 frames. It predates WebM and is convenient for round-trip
//! tests because an IVF frame payload is byte-for-byte the VP9 bitstream
//! the spec talks about.
//!
//! This module exposes two pure functions: [`parse_header`] and
//! [`iter_frames`]. No allocator is required — frame payloads are returned
//! as slices into the caller's buffer.
//!
//! Layout (little-endian throughout):
//!
//! ```text
//!   offset  size  field
//!   0       4     signature "DKIF"
//!   4       2     version (must be 0)
//!   6       2     header length (must be 32)
//!   8       4     fourcc (e.g. "VP90")
//!   12      2     width in pixels
//!   14      2     height in pixels
//!   16      4     framerate numerator
//!   20      4     framerate denominator
//!   24      4     frame count
//!   28      4     reserved
//! ```
//!
//! Each frame is `{ u32 size_le; u64 pts_le; u8 payload[size] }`.

use oxideav_core::{Error, Result};

/// Parsed IVF file header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IvfHeader {
    /// FourCC identifying the payload codec — `b"VP90"` for VP9.
    pub fourcc: [u8; 4],
    pub width: u16,
    pub height: u16,
    pub frame_rate_num: u32,
    pub frame_rate_den: u32,
    /// Frame count reported by the header. Not always trustworthy —
    /// ffmpeg often writes 0 because it doesn't know the count up front.
    pub frame_count: u32,
}

/// Parse the 32-byte IVF file header from `buf`. Returns the header and
/// the offset at which frame data starts (always 32 for well-formed IVF).
pub fn parse_header(buf: &[u8]) -> Result<(IvfHeader, usize)> {
    if buf.len() < 32 {
        return Err(Error::invalid("ivf: header shorter than 32 bytes"));
    }
    if &buf[0..4] != b"DKIF" {
        return Err(Error::invalid("ivf: bad signature (expect DKIF)"));
    }
    let version = u16::from_le_bytes([buf[4], buf[5]]);
    if version != 0 {
        return Err(Error::invalid(format!(
            "ivf: unsupported version {version} (expect 0)"
        )));
    }
    let header_len = u16::from_le_bytes([buf[6], buf[7]]);
    if header_len != 32 {
        return Err(Error::invalid(format!(
            "ivf: unexpected header length {header_len} (expect 32)"
        )));
    }
    let hdr = IvfHeader {
        fourcc: [buf[8], buf[9], buf[10], buf[11]],
        width: u16::from_le_bytes([buf[12], buf[13]]),
        height: u16::from_le_bytes([buf[14], buf[15]]),
        frame_rate_num: u32::from_le_bytes([buf[16], buf[17], buf[18], buf[19]]),
        frame_rate_den: u32::from_le_bytes([buf[20], buf[21], buf[22], buf[23]]),
        frame_count: u32::from_le_bytes([buf[24], buf[25], buf[26], buf[27]]),
    };
    Ok((hdr, 32))
}

/// One IVF frame's metadata + payload.
#[derive(Clone, Copy, Debug)]
pub struct IvfFrame<'a> {
    /// Presentation timestamp in units of `frame_rate_den / frame_rate_num`.
    pub pts: u64,
    pub payload: &'a [u8],
}

/// Iterator over frames in an IVF file. Yields `Err` on malformed data and
/// stops cleanly at EOF.
pub struct IvfIter<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> IvfIter<'a> {
    pub fn new(buf: &'a [u8]) -> Result<Self> {
        let (_, body_off) = parse_header(buf)?;
        Ok(Self { buf, pos: body_off })
    }
}

impl<'a> Iterator for IvfIter<'a> {
    type Item = Result<IvfFrame<'a>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.buf.len() {
            return None;
        }
        if self.pos + 12 > self.buf.len() {
            return Some(Err(Error::invalid(
                "ivf: truncated frame header (need 12 bytes)",
            )));
        }
        let size = u32::from_le_bytes([
            self.buf[self.pos],
            self.buf[self.pos + 1],
            self.buf[self.pos + 2],
            self.buf[self.pos + 3],
        ]) as usize;
        let pts = u64::from_le_bytes([
            self.buf[self.pos + 4],
            self.buf[self.pos + 5],
            self.buf[self.pos + 6],
            self.buf[self.pos + 7],
            self.buf[self.pos + 8],
            self.buf[self.pos + 9],
            self.buf[self.pos + 10],
            self.buf[self.pos + 11],
        ]);
        let body = self.pos + 12;
        let end = body + size;
        if end > self.buf.len() {
            return Some(Err(Error::invalid(format!(
                "ivf: truncated frame payload (need {size} bytes, have {})",
                self.buf.len() - body
            ))));
        }
        self.pos = end;
        Some(Ok(IvfFrame {
            pts,
            payload: &self.buf[body..end],
        }))
    }
}

/// Returns an iterator yielding each IVF frame.
pub fn iter_frames(buf: &[u8]) -> Result<IvfIter<'_>> {
    IvfIter::new(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal synthetic IVF: header + 1 frame of 4 bytes.
    fn synth_ivf() -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(b"DKIF");
        v.extend_from_slice(&0u16.to_le_bytes()); // version
        v.extend_from_slice(&32u16.to_le_bytes()); // header length
        v.extend_from_slice(b"VP90");
        v.extend_from_slice(&64u16.to_le_bytes()); // width
        v.extend_from_slice(&48u16.to_le_bytes()); // height
        v.extend_from_slice(&24u32.to_le_bytes()); // rate_num
        v.extend_from_slice(&1u32.to_le_bytes()); // rate_den
        v.extend_from_slice(&1u32.to_le_bytes()); // frame_count
        v.extend_from_slice(&0u32.to_le_bytes()); // reserved
                                                  // frame #0
        v.extend_from_slice(&4u32.to_le_bytes()); // size
        v.extend_from_slice(&0u64.to_le_bytes()); // pts
        v.extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]);
        v
    }

    #[test]
    fn parse_header_succeeds() {
        let buf = synth_ivf();
        let (h, off) = parse_header(&buf).unwrap();
        assert_eq!(&h.fourcc, b"VP90");
        assert_eq!(h.width, 64);
        assert_eq!(h.height, 48);
        assert_eq!(h.frame_rate_num, 24);
        assert_eq!(h.frame_rate_den, 1);
        assert_eq!(h.frame_count, 1);
        assert_eq!(off, 32);
    }

    #[test]
    fn iterate_single_frame() {
        let buf = synth_ivf();
        let frames: Vec<_> = iter_frames(&buf).unwrap().collect::<Result<_>>().unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].pts, 0);
        assert_eq!(frames[0].payload, &[0xDE, 0xAD, 0xBE, 0xEF]);
    }

    #[test]
    fn bad_signature_rejected() {
        let mut buf = synth_ivf();
        buf[0] = b'X';
        assert!(parse_header(&buf).is_err());
    }

    #[test]
    fn truncated_frame_header_errors() {
        let mut buf = synth_ivf();
        buf.truncate(32 + 6); // less than 12 bytes of frame header
        let mut it = iter_frames(&buf).unwrap();
        match it.next() {
            Some(Err(Error::InvalidData(_))) => {}
            other => panic!("expected error, got {other:?}"),
        }
    }

    #[test]
    fn truncated_payload_errors() {
        let mut buf = synth_ivf();
        buf.truncate(buf.len() - 2); // cut off last 2 bytes of payload
        let mut it = iter_frames(&buf).unwrap();
        match it.next() {
            Some(Err(Error::InvalidData(_))) => {}
            other => panic!("expected error, got {other:?}"),
        }
    }
}
