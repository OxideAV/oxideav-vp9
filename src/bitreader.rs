//! Bit-level reader for VP9 uncompressed headers.
//!
//! VP9 spec v0.7 §9.1 defines the `f(n)` parsing process: read `n` bits
//! from the stream, MSB-first within each byte, accumulating `x = 2 * x
//! + read_bit()`. §4.9.2 defines `s(n)` (signed): an `f(n)` magnitude
//! followed by an `f(1)` sign bit (1 = negative).
//!
//! This module exposes a thin reader with those contracts plus the
//! `trailing_bits()` zero-fill check from §6.1.1 (zero-pad to the next
//! byte boundary).
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

    /// `s(n)` from spec §4.9.2: read an `n`-bit magnitude followed by a
    /// 1-bit sign (1 = negative). `n` must be at most 31 since the
    /// magnitude fits in `i32` after sign application.
    pub(crate) fn read_signed(&mut self, n: u32) -> Result<i32, Error> {
        debug_assert!(n <= 31);
        let magnitude = self.read_bits(n)? as i32;
        let sign = self.read_bit()? != 0;
        Ok(if sign { -magnitude } else { magnitude })
    }

    /// `trailing_bits()` from spec §6.1.1: read zero bits until the
    /// stream position is byte-aligned. Returns
    /// [`Error::InvalidBitstream`] if any of the padding bits is set
    /// (spec §7.1.1: "zero_bit shall be equal to 0").
    pub(crate) fn trailing_bits(&mut self) -> Result<(), Error> {
        while self.bit_pos & 7 != 0 {
            if self.read_bit()? != 0 {
                return Err(Error::InvalidBitstream);
            }
        }
        Ok(())
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

    #[test]
    fn signed_round_trips_positive_and_negative() {
        // Two consecutive s(6) values laid out MSB-first into a byte
        // buffer. We construct the bit stream explicitly and then
        // pack it into bytes. The first s(6) is +10 (magnitude 6 bits
        // = 001010, sign = 0). The second s(6) is -3 (magnitude =
        // 000011, sign = 1). Total = 14 bits.
        let bits: [u32; 14] = [
            0, 0, 1, 0, 1, 0, // magnitude 10
            0, // sign +
            0, 0, 0, 0, 1, 1, // magnitude 3
            1, // sign -
        ];
        let mut bytes = vec![0u8; 2];
        for (i, b) in bits.iter().enumerate() {
            let bit_in_byte = 7 - (i & 7);
            bytes[i / 8] |= (*b as u8) << bit_in_byte;
        }
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.read_signed(6).unwrap(), 10);
        assert_eq!(r.read_signed(6).unwrap(), -3);
    }

    #[test]
    fn trailing_bits_accepts_zero_padding() {
        // Three bits in, expect 5 zero pad bits.
        let mut r = BitReader::new(&[0b1010_0000]);
        let _ = r.read_bits(3).unwrap();
        r.trailing_bits().unwrap();
        assert_eq!(r.position(), 8);
    }

    #[test]
    fn trailing_bits_rejects_nonzero_padding() {
        // Three bits in, then padding contains a 1 -> reject.
        let mut r = BitReader::new(&[0b1010_0100]);
        let _ = r.read_bits(3).unwrap();
        assert_eq!(r.trailing_bits().unwrap_err(), Error::InvalidBitstream);
    }

    #[test]
    fn trailing_bits_noop_when_aligned() {
        let mut r = BitReader::new(&[0xFF, 0x00]);
        let _ = r.read_bits(8).unwrap();
        r.trailing_bits().unwrap();
        assert_eq!(r.position(), 8);
    }
}
