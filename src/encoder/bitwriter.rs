//! MSB-first bit writer used by the VP9 uncompressed-header emitter (§6.2).
//!
//! Symmetric inverse of [`crate::bitreader::BitReader`]: appends up to 32
//! bits at a time into a byte-aligned output buffer, with `finish()`
//! flushing any trailing bits with zero-padding.
//!
//! VP9 §4.3: the uncompressed header packs fields MSB-first — the first
//! bit of each `f(n)` lands in the MSB of the next output byte.

/// MSB-first bit writer.
#[derive(Debug, Default)]
pub struct BitWriter {
    out: Vec<u8>,
    cur: u8,
    bits: u32,
}

impl BitWriter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Write the low `n` bits of `value`, MSB-first. Panics if `n > 32`.
    pub fn write(&mut self, value: u32, n: u32) {
        assert!(n <= 32);
        for i in (0..n).rev() {
            let b = ((value >> i) & 1) as u8;
            self.cur = (self.cur << 1) | b;
            self.bits += 1;
            if self.bits == 8 {
                self.out.push(self.cur);
                self.cur = 0;
                self.bits = 0;
            }
        }
    }

    /// Write a single bit (0 or 1).
    pub fn bit(&mut self, v: bool) {
        self.write(v as u32, 1);
    }

    /// Advance to the next byte boundary; no-op if already aligned.
    pub fn byte_align(&mut self) {
        if self.bits > 0 {
            self.cur <<= 8 - self.bits;
            self.out.push(self.cur);
            self.cur = 0;
            self.bits = 0;
        }
    }

    /// Append pre-formed bytes. The writer must be byte-aligned.
    pub fn append_bytes(&mut self, bytes: &[u8]) {
        assert_eq!(self.bits, 0, "bit writer not byte-aligned");
        self.out.extend_from_slice(bytes);
    }

    /// Return the current byte-aligned length of the emitted buffer (after
    /// flushing partial bits). Used by callers that need to back-patch a
    /// length field that includes the partial tail.
    pub fn len_bytes(&self) -> usize {
        self.out.len() + if self.bits > 0 { 1 } else { 0 }
    }

    /// Finalize and return the byte buffer. Trailing bits are zero-padded.
    pub fn finish(mut self) -> Vec<u8> {
        self.byte_align();
        self.out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_byte_msb_first() {
        let mut bw = BitWriter::new();
        bw.write(0b1010_0010, 8);
        let b = bw.finish();
        assert_eq!(b, vec![0xA2]);
    }

    #[test]
    fn trailing_bits_zero_padded() {
        let mut bw = BitWriter::new();
        // write 11 (3 bits)
        bw.write(0b11, 2);
        let b = bw.finish();
        assert_eq!(b, vec![0b1100_0000]);
    }

    #[test]
    fn multi_field_ordering() {
        // Emulate header start: frame_marker=2 (2b), profile_low=0 (1b),
        // profile_high=0 (1b), show_existing_frame=0 (1b), frame_type=0
        // (1b), show_frame=1 (1b), error_resilient_mode=0 (1b)
        // -> 10_0_0_0_0_1_0 = 0b1000_0010 = 0x82.
        let mut bw = BitWriter::new();
        bw.write(2, 2);
        bw.bit(false);
        bw.bit(false);
        bw.bit(false);
        bw.bit(false);
        bw.bit(true);
        bw.bit(false);
        assert_eq!(bw.finish(), vec![0x82]);
    }
}
