//! Bit-level reader for VP9 uncompressed headers.
//!
//! VP9 spec v0.7 §9.1 defines the `f(n)` parsing process: read `n` bits
//! from the stream, MSB-first within each byte, accumulating `x = 2 * x
//! + read_bit()`. This module exposes a thin reader with that contract.
//!
//! The reader is intentionally minimal — it serves the uncompressed
//! header walker only and does not implement the Boolean coder (spec
//! §9.2) used by the compressed header.

use crate::Error;

/// Most-significant-bit-first bit reader over a byte slice.
///
/// Position is tracked in bits; reads up to 32 bits at a time are
/// supported (the VP9 uncompressed header never asks for more).
#[derive(Debug)]
pub(crate) struct BitReader<'a> {
    data: &'a [u8],
    bit_pos: usize,
}

impl<'a> BitReader<'a> {
    /// Wrap `data` for MSB-first bit reading starting at bit 0.
    pub(crate) fn new(data: &'a [u8]) -> Self {
        Self { data, bit_pos: 0 }
    }

    /// Current absolute bit position (number of bits already consumed).
    #[cfg(test)]
    pub(crate) fn position(&self) -> usize {
        self.bit_pos
    }

    /// `f(n)` from spec §9.1: read `n` bits MSB-first, return as `u32`.
    ///
    /// Returns [`Error::UnexpectedEof`] if the stream runs out of bits
    /// before `n` are consumed. `n` must be at most 32.
    pub(crate) fn read_bits(&mut self, n: u32) -> Result<u32, Error> {
        debug_assert!(n <= 32);
        let mut value: u32 = 0;
        for _ in 0..n {
            value = (value << 1) | self.read_bit()?;
        }
        Ok(value)
    }

    /// Convenience for `f(1)` reads that should be interpreted as flags.
    pub(crate) fn read_flag(&mut self) -> Result<bool, Error> {
        Ok(self.read_bit()? != 0)
    }

    fn read_bit(&mut self) -> Result<u32, Error> {
        let byte_index = self.bit_pos >> 3;
        if byte_index >= self.data.len() {
            return Err(Error::UnexpectedEof);
        }
        // Spec §9.1: "the first bit is given by the most significant
        // bit of the first byte".
        let bit_in_byte = 7 - (self.bit_pos & 7);
        let bit = (self.data[byte_index] >> bit_in_byte) & 1;
        self.bit_pos += 1;
        Ok(bit as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn msb_first_byte_aligned() {
        // 0xA5 = 1010_0101. Read 8 bits as MSB-first must give 0xA5.
        let mut r = BitReader::new(&[0xA5]);
        assert_eq!(r.read_bits(8).unwrap(), 0xA5);
        assert_eq!(r.position(), 8);
    }

    #[test]
    fn msb_first_across_byte_boundary() {
        // Bytes 0xA5 0x5A = 1010_0101 0101_1010.
        // Reading 4 + 8 + 4 must reconstruct 0xA, 0x55, 0xA.
        let mut r = BitReader::new(&[0xA5, 0x5A]);
        assert_eq!(r.read_bits(4).unwrap(), 0xA);
        assert_eq!(r.read_bits(8).unwrap(), 0x55);
        assert_eq!(r.read_bits(4).unwrap(), 0xA);
        assert_eq!(r.position(), 16);
    }

    #[test]
    fn eof_returns_error() {
        let mut r = BitReader::new(&[0xFF]);
        assert!(r.read_bits(8).is_ok());
        assert_eq!(r.read_bits(1).unwrap_err(), Error::UnexpectedEof);
    }
}
