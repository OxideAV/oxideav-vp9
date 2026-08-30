//! VP9 Boolean (range) decoder per spec v0.7 §9.2.
//!
//! VP9 entropy-codes everything past the uncompressed header. The
//! compressed header (§6.3) and per-tile partitions are arithmetic
//! coded under a small carry-less range coder whose state is three
//! variables `BoolValue`, `BoolRange`, `BoolMaxBits` (§9.2.1) updated
//! by `read_bool( p )` calls (§9.2.2). The decoder is initialised
//! once per partition / compressed-header block via `init_bool( sz )`
//! and shut down via `exit_bool( )` (§9.2.3).
//!
//! `read_literal( n )` (§9.2.4) is implemented as a fold over
//! `read_bool( 128 )`.
//!
//! Provenance: VP9 Bitstream & Decoding Process Specification v0.7,
//! `docs/video/vp9/vp9-spec.txt` §9.2.

use crate::Error;

/// Decoder state for the VP9 Boolean (range) coder (spec §9.2).
///
/// The state holds the three §9.2.1 variables (`BoolValue`,
/// `BoolRange`, `BoolMaxBits`) plus a borrow on the slice of bytes
/// being consumed and a bit-cursor into that slice. The cursor only
/// ever advances forward — there is no rewind.
#[derive(Debug)]
pub(crate) struct BoolCoder<'a> {
    data: &'a [u8],
    /// Absolute bit position within `data` of the next bit to read.
    bit_pos: usize,
    /// Spec §9.2.1 `BoolValue`. 8-bit window per `f(8)` init, then
    /// shifted under `read_bool` renormalisation up to 8 bits at a
    /// time.
    bool_value: u32,
    /// Spec §9.2.1 `BoolRange`. Initialised to 255 and kept in the
    /// half-open range \[128, 256) after each `read_bool`.
    bool_range: u32,
    /// Spec §9.2.1 `BoolMaxBits = 8 * sz - 8`. Decremented for each
    /// renormalisation refill bit. The spec requires the partition
    /// never underflow this counter (§9.2.2: "It is a requirement of
    /// bitstream conformance that this never happens").
    bool_max_bits: i64,
}

impl<'a> BoolCoder<'a> {
    /// `init_bool( sz )` per spec §9.2.1.
    ///
    /// `data` is the bitstream slice the coder will read from
    /// starting at bit 0; `sz` is the number of bytes the coder is
    /// permitted to consume (the compressed header's
    /// `header_size_in_bytes`, or a per-tile `tile_size`). The
    /// constructor reads `BoolValue` via `f(8)`, sets `BoolRange` to
    /// 255, sets `BoolMaxBits = 8 * sz - 8`, then invokes
    /// `read_bool(128)` once to consume the §9.2.1 marker bit (which
    /// shall decode to 0 for a conformant stream).
    ///
    /// Returns [`Error::InvalidBitstream`] if `sz < 1` (the spec
    /// forbids this) or if the marker bit decodes to a nonzero value.
    /// Returns [`Error::UnexpectedEof`] if the slice is shorter than
    /// `sz` bytes or the marker read runs off the slice.
    pub(crate) fn init_bool(data: &'a [u8], sz: usize) -> Result<Self, Error> {
        if sz < 1 {
            // §9.2.1: "The bitstream shall not contain data that
            // results in this process being called with sz < 1."
            return Err(Error::InvalidBitstream);
        }
        if data.len() < sz {
            return Err(Error::UnexpectedEof);
        }
        // Only the first `sz` bytes are visible to the coder; clip
        // the borrow so a renormalisation refill that overshoots
        // `sz` triggers `UnexpectedEof` rather than silently
        // borrowing the next packet's bytes.
        let visible = &data[..sz];
        let mut c = Self {
            data: visible,
            bit_pos: 0,
            bool_value: 0,
            bool_range: 255,
            bool_max_bits: (8 * sz as i64) - 8,
        };
        // `BoolValue` is read using `f(8)` (§9.2.1).
        c.bool_value = c.read_bits_raw(8)?;
        // Marker bit — must decode to 0.
        let marker = c.read_bool(128)?;
        if marker != 0 {
            return Err(Error::InvalidBitstream);
        }
        Ok(c)
    }

    /// `read_bool( p )` per spec §9.2.2.
    ///
    /// `p` is the probability in `0..=255` that the decoded bit is 0.
    /// Returns the decoded bit (0 or 1).
    ///
    /// May return [`Error::UnexpectedEof`] if renormalisation needs
    /// more bits than the slice has and `BoolMaxBits` is positive (the
    /// slice underran), or [`Error::InvalidBitstream`] if
    /// `BoolMaxBits` is exhausted (spec: "It is a requirement of
    /// bitstream conformance that this never happens").
    pub(crate) fn read_bool(&mut self, p: u32) -> Result<u32, Error> {
        debug_assert!(p <= 255);
        let split = 1 + (((self.bool_range - 1) * p) >> 8);
        let bit = if self.bool_value < split {
            self.bool_range = split;
            0
        } else {
            self.bool_range -= split;
            self.bool_value -= split;
            1
        };
        // Renormalise: while BoolRange < 128, shift in fresh bits.
        while self.bool_range < 128 {
            let new_bit = if self.bool_max_bits > 0 {
                let nb = self.read_bits_raw(1)?;
                self.bool_max_bits -= 1;
                nb
            } else {
                // Spec §9.2.2: a conformant stream never reaches
                // this branch. Flag the bitstream as invalid rather
                // than silently inject 0 — the spec's "set equal to
                // 0" is a permissive fallback; we want to surface
                // truncation explicitly.
                return Err(Error::InvalidBitstream);
            };
            self.bool_range <<= 1;
            self.bool_value = (self.bool_value << 1) + new_bit;
        }
        Ok(bit)
    }

    /// `read_literal( n )` per spec §9.2.4.
    ///
    /// Folds `n` calls to `read_bool(128)` into an MSB-first
    /// unsigned integer. `n` must be at most 32 — VP9 never asks for
    /// more.
    pub(crate) fn read_literal(&mut self, n: u32) -> Result<u32, Error> {
        debug_assert!(n <= 32);
        let mut x: u32 = 0;
        for _ in 0..n {
            x = (x << 1) + self.read_bool(128)?;
        }
        Ok(x)
    }

    /// `exit_bool( )` per spec §9.2.3.
    ///
    /// Consumes the remaining `BoolMaxBits` of padding from the
    /// stream. Spec requires every padding bit to be 0, so any 1 in
    /// the tail yields [`Error::InvalidBitstream`]. Returns the bit
    /// position (within the slice handed to `init_bool`) where the
    /// coder finishes.
    ///
    /// Not yet wired into the round-3 compressed-header walker
    /// (which only reads `read_tx_mode` and exits early); kept as
    /// part of the §9.2 surface so the next round's
    /// `diff_update_prob` chain can shut the coder down properly.
    #[allow(dead_code)]
    pub(crate) fn exit_bool(&mut self) -> Result<usize, Error> {
        while self.bool_max_bits > 0 {
            let pad = self.read_bits_raw(1)?;
            self.bool_max_bits -= 1;
            if pad != 0 {
                return Err(Error::InvalidBitstream);
            }
        }
        Ok(self.bit_pos)
    }

    /// MSB-first raw `f(n)` reader used by `init_bool` for `BoolValue`
    /// and by `read_bool` / `exit_bool` for renormalisation /
    /// padding consumption. Not exposed outside the module — the
    /// §9.2 spec only describes raw bit fetches in the context of
    /// the Boolean coder itself.
    fn read_bits_raw(&mut self, n: u32) -> Result<u32, Error> {
        debug_assert!(n <= 32);
        let mut value: u32 = 0;
        for _ in 0..n {
            let byte_index = self.bit_pos >> 3;
            if byte_index >= self.data.len() {
                return Err(Error::UnexpectedEof);
            }
            let bit_in_byte = 7 - (self.bit_pos & 7);
            let bit = (self.data[byte_index] >> bit_in_byte) & 1;
            self.bit_pos += 1;
            value = (value << 1) | (bit as u32);
        }
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test vectors below were derived by directly stepping the §9.2
    // decoder by hand and confirming each buffer produces the
    // expected sequence of bits / literals.

    #[test]
    fn init_bool_rejects_zero_size() {
        let data = [0u8; 4];
        assert_eq!(
            BoolCoder::init_bool(&data, 0).unwrap_err(),
            Error::InvalidBitstream
        );
    }

    #[test]
    fn init_bool_rejects_short_slice() {
        let data = [0u8; 1];
        assert_eq!(
            BoolCoder::init_bool(&data, 4).unwrap_err(),
            Error::UnexpectedEof
        );
    }

    #[test]
    fn init_bool_rejects_nonzero_marker() {
        // BoolValue starts at 0x80 (MSB set). split for p=128 is
        // `1 + ((254 * 128) >> 8) = 128`. Since BoolValue (0x80=128)
        // is NOT less than split (128), the marker decodes to 1 →
        // InvalidBitstream.
        let bytes = [0x80, 0x00, 0x00, 0x00];
        assert_eq!(
            BoolCoder::init_bool(&bytes, 4).unwrap_err(),
            Error::InvalidBitstream
        );
    }

    #[test]
    fn init_bool_accepts_zero_buffer_marker() {
        // First byte 0x00 → BoolValue=0, marker split=128, 0<128 → bit=0
        // (the spec's required zero marker). No renorm fires
        // because BoolRange becomes 128 ≥ 128.
        let bytes = [0x00, 0x00, 0x00, 0x00];
        let _ = BoolCoder::init_bool(&bytes, 4).expect("zero-buffer marker accepted");
    }

    #[test]
    fn read_bool_hand_traced_buffer() {
        // Golden buffer 0x36 0x00 0x00 0x00 derived by brute-force
        // search over the §9.2 decoder: produces read_bool(128)=0,
        // then read_bool(64)=1, then read_bool(200)=1 after the
        // marker. Confirmed by re-stepping the decoder by hand.
        let bytes = [0x36u8, 0x00, 0x00, 0x00];
        let mut dec = BoolCoder::init_bool(&bytes, 4).unwrap();
        assert_eq!(dec.read_bool(128).unwrap(), 0);
        assert_eq!(dec.read_bool(64).unwrap(), 1);
        assert_eq!(dec.read_bool(200).unwrap(), 1);
    }

    #[test]
    fn read_literal_hand_traced_buffer() {
        // Golden buffer 0x58 0x00 0x00 0x00 → read_literal(4) =
        // 0b1011 = 11.
        let bytes = [0x58u8, 0x00, 0x00, 0x00];
        let mut dec = BoolCoder::init_bool(&bytes, 4).unwrap();
        assert_eq!(dec.read_literal(4).unwrap(), 0b1011);
    }

    #[test]
    fn read_bool_extreme_high_probability_runs_zeros() {
        // With p=255, split = 1 + ((range-1)*255 >> 8). For range=128
        // (post-marker), split = 1 + ((127*255)>>8) = 1 + 126 = 127.
        // BoolValue stays at 0 in our all-zero buffer, which is
        // < 127 so bit=0 each time; range becomes 127 < 128 so a
        // single renorm bit (still 0) is pulled in. Repeats stably.
        let bytes = [0x00u8, 0x00, 0x00, 0x00];
        let mut dec = BoolCoder::init_bool(&bytes, 4).unwrap();
        for _ in 0..4 {
            assert_eq!(dec.read_bool(255).unwrap(), 0);
        }
    }

    #[test]
    fn exit_bool_accepts_all_zero_padding() {
        // The decoder consumes BoolMaxBits = 24 (sz=4, after the
        // initial f(8) BoolValue read). For an all-zero buffer the
        // padding is all 0 and exit_bool succeeds.
        let bytes = [0x00u8, 0x00, 0x00, 0x00];
        let mut dec = BoolCoder::init_bool(&bytes, 4).unwrap();
        let _ = dec.read_bool(128).unwrap();
        dec.exit_bool().expect("zero padding accepted");
    }

    #[test]
    fn exit_bool_rejects_nonzero_padding() {
        // Build a valid stream and poke a nonzero bit into the
        // padding tail. After reading just a couple of bools we
        // still have enough BoolMaxBits remaining to encounter the
        // tail byte's stray 1.
        let bytes = [0x00u8, 0x00, 0x00, 0x01];
        let mut dec = BoolCoder::init_bool(&bytes, 4).unwrap();
        // Marker consumed by init; just exit.
        assert_eq!(dec.exit_bool().unwrap_err(), Error::InvalidBitstream);
    }
}
