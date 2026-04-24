//! VP9 forward boolean (range) encoder — mirror of §9.2.
//!
//! Produces the byte stream the decoder's [`BoolDecoder`] consumes.
//! Derived from inverting the §9.2 decode equations; libvpx encoder
//! source intentionally not read.
//!
//! The encoder tracks a 32-bit lower bound `low` (MSB-aligned) and an
//! 8-bit `range` in `[128, 255]`. On each symbol the interval is split
//! proportionally to the probability, and renormalisation shifts bits
//! out of `low`'s high end.
//!
//! Carries are handled with the "pending byte" idiom: a carry from the
//! active window propagates back through a buffered `held` byte plus
//! a run of `ff_run` 0xff bytes (all of which flip to 0x00). The
//! `held` slot is the only position where a carry can still arrive;
//! once we emit a non-0xff byte it supersedes the old `held`.
//!
//! [`BoolDecoder`]: crate::bool_decoder::BoolDecoder

/// Forward binary range coder — inverse of `BoolDecoder`.
#[derive(Debug)]
pub struct BoolEncoder {
    /// Fully committed output bytes.
    out: Vec<u8>,
    /// Buffered byte awaiting possible carry, or `None` if empty.
    held: Option<u8>,
    /// Number of `0xff` bytes queued between `held` and the live
    /// encoding window; flip to `0x00` on carry.
    ff_run: u32,
    /// Lower bound of the interval. The high 8 bits are the next byte
    /// to emit; bits 24..=31 may overflow into a carry in
    /// `low.bit(32)` (we use u64 to observe the carry).
    low: u64,
    /// Interval width in `[128, 255]`.
    range: u32,
    /// Bits buffered in `low` that haven't been emitted yet. Starts at
    /// `-24` (the spec's 8-bit priming read consumes the first byte;
    /// we need to buffer three 8-bit chunks before the decoder's
    /// initial read lines up with our output).
    count: i32,
}

impl Default for BoolEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl BoolEncoder {
    pub fn new() -> Self {
        Self {
            out: Vec::new(),
            held: None,
            ff_run: 0,
            low: 0,
            range: 255,
            count: -24,
        }
    }

    /// Write one symbol `b` (0 or 1) with `prob = P(b==0)`.
    pub fn write(&mut self, b: u32, prob: u8) {
        let split = 1 + (((self.range - 1) * prob as u32) >> 8);
        if b != 0 {
            self.low = self.low.wrapping_add(split as u64);
            self.range -= split;
        } else {
            self.range = split;
        }
        // Renormalise.
        let shift = self.range.leading_zeros().saturating_sub(24); // 1..7 while range<128
                                                                   // Apply `shift` successive doublings.
        for _ in 0..shift {
            self.range <<= 1;
        }
        // Shift `low` up by the same amount, emitting bytes as bits
        // overflow.
        self.low <<= shift;
        self.count += shift as i32;
        // Emit full bytes while count >= 0 — i.e. while at least 1 byte
        // of resolved data sits above bit 24 of `low`.
        while self.count >= 0 {
            // The next byte is the 8 bits at positions (24+count)..(32+count)
            // of `low`. When count=0 it's bits 24..32. Extract and clear.
            let byte_shift = 24 + self.count as u32;
            let byte = ((self.low >> byte_shift) & 0xff) as u32;
            // Detect carry — any bit above bit (31+1)=32 signals that
            // the current emitted byte is >= 0x100, so we need to
            // bump `held`.
            let carry = (self.low >> (byte_shift + 8)) & 1 != 0;
            // Mask out the emitted byte (and any carry above it).
            let mask = !((0x1ffu64) << byte_shift);
            self.low &= mask;
            self.count -= 8;
            self.emit_byte_maybe_carry(byte as u8, carry);
        }
    }

    pub fn write_bool(&mut self, b: bool, prob: u8) {
        self.write(b as u32, prob);
    }

    pub fn write_literal(&mut self, value: u32, n: u32) {
        for i in (0..n).rev() {
            self.write(((value >> i) & 1) as u32, 128);
        }
    }

    /// Queue one output byte. If `carry` is set, first propagate a +1
    /// through queued bytes (held + ff_run).
    fn emit_byte_maybe_carry(&mut self, byte: u8, carry: bool) {
        if carry {
            // Propagate +1 through the pending chain.
            if let Some(h) = self.held.take() {
                self.out.push(h.wrapping_add(1));
            }
            // All pending 0xff bytes flip to 0x00.
            for _ in 0..self.ff_run {
                self.out.push(0x00);
            }
            self.ff_run = 0;
        }
        // Now place `byte` in the pending slot per the buffer rules.
        if byte == 0xff {
            match self.held {
                Some(_) => self.ff_run += 1,
                None => self.held = Some(0xff),
            }
        } else {
            if let Some(h) = self.held.take() {
                self.out.push(h);
                for _ in 0..self.ff_run {
                    self.out.push(0xff);
                }
                self.ff_run = 0;
            }
            self.held = Some(byte);
        }
    }

    /// Finalise — push the remaining 32 bits of `low` and drain the
    /// pending buffer. The decoder zero-fills past EOF so trailing
    /// zeros are harmless.
    pub fn finish(mut self) -> Vec<u8> {
        // Flush 32 more bits to ensure the decoder's 24-bit window has
        // valid data for any reads issued after our last `write`.
        for _ in 0..32 {
            self.low <<= 1;
            self.count += 1;
            if self.count >= 0 {
                let byte_shift = 24 + self.count as u32;
                let byte = ((self.low >> byte_shift) & 0xff) as u32;
                let carry = (self.low >> (byte_shift + 8)) & 1 != 0;
                let mask = !((0x1ffu64) << byte_shift);
                self.low &= mask;
                self.count -= 8;
                self.emit_byte_maybe_carry(byte as u8, carry);
            }
        }
        // Drain held + pending.
        if let Some(h) = self.held.take() {
            self.out.push(h);
            for _ in 0..self.ff_run {
                self.out.push(0xff);
            }
            self.ff_run = 0;
        }
        if self.out.is_empty() {
            self.out.push(0);
        }
        self.out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bool_decoder::BoolDecoder;

    #[test]
    fn roundtrip_bit_stream_equal_probs() {
        // Write a deterministic bit pattern then decode it back.
        let bits = [1u32, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0];
        let mut e = BoolEncoder::new();
        for &b in &bits {
            e.write(b, 128);
        }
        let buf = e.finish();
        let mut d = BoolDecoder::new(&buf).unwrap();
        for &b in &bits {
            assert_eq!(d.read(128).unwrap(), b, "bit mismatch");
        }
    }

    #[test]
    fn roundtrip_skewed_probs() {
        let bits = [0u32, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0];
        let mut e = BoolEncoder::new();
        for &b in &bits {
            e.write(b, 250);
        }
        let buf = e.finish();
        let mut d = BoolDecoder::new(&buf).unwrap();
        for &b in &bits {
            assert_eq!(d.read(250).unwrap(), b);
        }
    }

    #[test]
    fn roundtrip_mixed_probs() {
        // 256 symbols with rolling probability.
        let mut e = BoolEncoder::new();
        let mut expected = Vec::new();
        for i in 0..256u32 {
            let prob = ((i * 17 + 3) & 0xff) as u8;
            let prob = prob.max(1).min(254);
            let bit = (i * 31 + 7) & 1;
            e.write(bit, prob);
            expected.push((bit, prob));
        }
        let buf = e.finish();
        let mut d = BoolDecoder::new(&buf).unwrap();
        for (bit, prob) in expected {
            assert_eq!(d.read(prob).unwrap(), bit);
        }
    }

    #[test]
    fn roundtrip_literals() {
        let mut e = BoolEncoder::new();
        e.write_literal(0b1011_0010, 8);
        e.write_literal(0b0110_1100_1001, 12);
        let buf = e.finish();
        let mut d = BoolDecoder::new(&buf).unwrap();
        assert_eq!(d.read_literal(8).unwrap(), 0b1011_0010);
        assert_eq!(d.read_literal(12).unwrap(), 0b0110_1100_1001);
    }

    #[test]
    fn roundtrip_long_stream_pseudo_random() {
        // Stress test: 2048 symbols with mixed probs — exercises the
        // carry path and byte-flush logic. Use a tiny PRNG.
        let mut state: u64 = 0x1234_5678_9abc_def0;
        let mut bits = Vec::with_capacity(2048);
        let mut probs = Vec::with_capacity(2048);
        for _ in 0..2048 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let prob = ((state >> 32) as u32 & 0xfe) as u8 | 1; // 1..=255
            let bit = (state as u32) & 1;
            bits.push(bit);
            probs.push(prob);
        }
        let mut e = BoolEncoder::new();
        for i in 0..bits.len() {
            e.write(bits[i], probs[i]);
        }
        let buf = e.finish();
        let mut d = BoolDecoder::new(&buf).unwrap();
        for i in 0..bits.len() {
            assert_eq!(d.read(probs[i]).unwrap(), bits[i], "mismatch at {i}");
        }
    }

    #[test]
    fn roundtrip_forces_carry_chain() {
        // Repeatedly write "1" with high prob (prob close to 0 means
        // low-probability-of-0, so bit=1 keeps being encoded via
        // split shrinks). Alternating patterns should force the
        // ff-run + carry propagation path.
        let mut e = BoolEncoder::new();
        let bits: Vec<u32> = (0..500).map(|i| ((i * 7 + 3) & 1) as u32).collect();
        for &b in &bits {
            // Prob=2 keeps the split near 1, most of range flows to
            // the "bit=1" side — exercises arithmetic edge cases.
            e.write(b, 2);
        }
        let buf = e.finish();
        let mut d = BoolDecoder::new(&buf).unwrap();
        for &b in &bits {
            assert_eq!(d.read(2).unwrap(), b);
        }
    }
}
