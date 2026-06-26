//! VP9 Boolean (range) **encoder** — the inverse of the §9.2 decoder.
//!
//! The bitstream specification (v0.7 §9.2) only describes the *decode*
//! direction: `init_bool( sz )` reads an 8-bit `BoolValue` window, sets
//! `BoolRange = 255`, then each `read_bool( p )` narrows `BoolRange`
//! around a `split` point and renormalises by shifting in fresh bits.
//! An encoder that produces a bitstream this decoder accepts is the
//! arithmetic-coder inverse of that process; it is *derived here from
//! the decode equations*, not borrowed from any external implementation.
//!
//! # Derivation
//!
//! The decoder keeps `BoolValue` in `[0, BoolRange)` and, for a bit with
//! `split = 1 + (((BoolRange - 1) * p) >> 8)`, decides
//!
//! * `bool = 0` when `BoolValue <  split` and sets `BoolRange = split`,
//! * `bool = 1` when `BoolValue >= split`, sets `BoolValue -= split` and
//!   `BoolRange -= split`.
//!
//! The encoder maintains the complementary interval bottom `low` and the
//! interval width `range` (with `range == BoolRange` at every step).
//! Encoding bit `b`:
//!
//! * `b == 0` keeps the *low* sub-interval: `range = split`.
//! * `b == 1` keeps the *high* sub-interval: `low += split`,
//!   `range -= split`.
//!
//! After each symbol the encoder renormalises by the same amount the
//! decoder would: `shift` is the number of left-shifts that bring `range`
//! back into `[128, 256)`. Those `shift` bits of `low` settle into the
//! output most-significant-bit first; a `count` accumulator (initialised
//! to `-24`, tracking how many settled bits sit above the live 8-bit
//! window) tells us when a whole byte is ready to emit. Because the
//! `low += split` step can carry past the byte that has already been
//! written, the emitter ripples a carry back through any trailing
//! `0xff` output bytes ([`BoolEncoder::propagate_carry`]).
//!
//! Initialisation mirrors §9.2.1: `range = 255`, `low = 0`, then a single
//! `write_bool( 0, 128 )` emits the marker bit the decoder's `init_bool`
//! consumes. Shutdown ([`BoolEncoder::finish`]) flushes the residual
//! interval by coding 32 trailing zero-bits (so every settled bit of
//! `low` reaches the output) and then guarantees the final byte is not a
//! §9.2.3 superframe marker (`(b & 0xe0) == 0xc0`).
//!
//! Provenance: derived from VP9 Bitstream & Decoding Process
//! Specification v0.7 (`docs/video/vp9/vp9-spec.txt`) §9.2; no external
//! encoder source consulted.

/// Arithmetic (Boolean / range) encoder mirroring the §9.2 decoder.
///
/// Produced bytes accumulate internally; call [`BoolEncoder::finish`] to
/// flush the residual interval and obtain the completed partition.
///
/// Built bottom-up ahead of the higher encoder layers (header / token
/// writers) that consume it; the `dead_code` allowance is removed as
/// those callers land.
#[derive(Debug)]
#[allow(dead_code)]
pub(crate) struct BoolEncoder {
    /// Interval bottom. The live 8-bit window sits at bits `[16, 24)`;
    /// the bits above are settled output awaiting emission and the bit
    /// at position 24 is the carry guard produced by `low += split`.
    low: u32,
    /// Interval width, kept in `[128, 256)` after every `write_bool`
    /// exactly like the decoder's `BoolRange`.
    range: u32,
    /// Settled-bit counter (starts at `-24`). When it reaches `>= 0`
    /// after a renorm a byte is ready to emit from the top of `low`.
    count: i32,
    /// Emitted bytes (carry-resolved).
    out: Vec<u8>,
}

/// Number of left-shifts that bring `range` back into `[128, 256)`.
#[allow(dead_code)]
fn renorm_shift(range: u32) -> u32 {
    debug_assert!(range > 0);
    let mut r = range;
    let mut s = 0;
    while r < 128 {
        r <<= 1;
        s += 1;
    }
    s
}

#[allow(dead_code)]
impl BoolEncoder {
    /// Initialise mirroring §9.2.1: `range = 255`, `low = 0`, then emit
    /// the marker bit (`write_bool( 0, 128 )`) the decoder consumes.
    pub(crate) fn new() -> Self {
        let mut e = Self {
            low: 0,
            range: 255,
            count: -24,
            out: Vec::new(),
        };
        e.write_bool(0, 128);
        e
    }

    /// Encode one bit `b` (0 or 1) with probability `p` (the decoder's
    /// probability that the bit is 0, `0..=255`).
    pub(crate) fn write_bool(&mut self, b: u32, p: u32) {
        debug_assert!(p <= 255);
        debug_assert!(b <= 1);
        let split = 1 + (((self.range - 1) * p) >> 8);
        if b == 0 {
            self.range = split;
        } else {
            self.low += split;
            self.range -= split;
        }

        let mut shift = renorm_shift(self.range) as i32;
        self.range <<= shift;
        self.count += shift;
        if self.count >= 0 {
            // `offset` settled bits belong to the byte being emitted.
            let offset = shift - self.count;
            // Carry: the top settled bit folded a `+1` into already-
            // emitted bytes. `offset == 0` means the carry bit is the
            // current MSB of `low`.
            let carry = if offset > 0 {
                (self.low << (offset - 1)) & 0x8000_0000
            } else {
                self.low & 0x8000_0000
            };
            if carry != 0 {
                self.propagate_carry();
            }
            self.out.push(((self.low >> (24 - offset)) & 0xff) as u8);
            self.low <<= offset;
            self.low &= 0x00ff_ffff;
            shift = self.count;
            self.count -= 8;
        }
        self.low <<= shift;
    }

    /// Encode `n` MSB-first literal bits of `value` as `write_bool( ·, 128 )`
    /// — the inverse of the decoder's §9.2.4 `read_literal`.
    pub(crate) fn write_literal(&mut self, value: u32, n: u32) {
        debug_assert!(n <= 32);
        for i in (0..n).rev() {
            self.write_bool((value >> i) & 1, 128);
        }
    }

    /// Add one to the most recently emitted byte, rippling a carry
    /// through any trailing `0xff` bytes that turn into `0x00`.
    fn propagate_carry(&mut self) {
        let mut i = self.out.len();
        while i > 0 {
            i -= 1;
            if self.out[i] == 0xff {
                self.out[i] = 0;
            } else {
                self.out[i] += 1;
                return;
            }
        }
    }

    /// Flush the residual interval and return the completed partition.
    ///
    /// Codes 32 trailing zero-bits so every settled bit of `low` reaches
    /// the output (the decoder discards the surplus as it exhausts
    /// `BoolMaxBits`), then guarantees the final byte is not a §9.2.3
    /// superframe marker (`(b & 0xe0) == 0xc0`) by appending a `0x00`
    /// byte when needed.
    pub(crate) fn finish(mut self) -> Vec<u8> {
        for _ in 0..32 {
            self.write_bool(0, 128);
        }
        if let Some(&last) = self.out.last() {
            if (last & 0xe0) == 0xc0 {
                self.out.push(0x00);
            }
        }
        if self.out.is_empty() {
            self.out.push(0x00);
        }
        self.out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bool_coder::BoolCoder;

    /// Round-trip a known bit/probability sequence through the encoder
    /// and back through the §9.2 decoder.
    fn roundtrip(bits: &[(u32, u32)]) {
        let mut enc = BoolEncoder::new();
        for &(b, p) in bits {
            enc.write_bool(b, p);
        }
        let buf = enc.finish();
        let mut dec = BoolCoder::init_bool(&buf, buf.len()).expect("init_bool");
        for &(b, p) in bits {
            assert_eq!(dec.read_bool(p).unwrap(), b, "bit/prob {b}/{p}");
        }
    }

    #[test]
    fn roundtrip_empty_partition() {
        let enc = BoolEncoder::new();
        let buf = enc.finish();
        BoolCoder::init_bool(&buf, buf.len()).expect("marker-only partition");
    }

    #[test]
    fn roundtrip_single_bits() {
        roundtrip(&[(0, 128)]);
        roundtrip(&[(1, 128)]);
        roundtrip(&[(0, 1)]);
        roundtrip(&[(1, 255)]);
        roundtrip(&[(1, 1)]);
        roundtrip(&[(0, 255)]);
    }

    #[test]
    fn roundtrip_mixed_sequence() {
        roundtrip(&[
            (1, 200),
            (0, 64),
            (1, 1),
            (0, 255),
            (1, 128),
            (1, 250),
            (0, 5),
            (0, 200),
        ]);
    }

    #[test]
    fn roundtrip_literals() {
        let mut enc = BoolEncoder::new();
        enc.write_literal(0b1011, 4);
        enc.write_literal(0xABCD, 16);
        enc.write_literal(0, 8);
        enc.write_literal(0xFF, 8);
        let buf = enc.finish();
        let mut dec = BoolCoder::init_bool(&buf, buf.len()).unwrap();
        assert_eq!(dec.read_literal(4).unwrap(), 0b1011);
        assert_eq!(dec.read_literal(16).unwrap(), 0xABCD);
        assert_eq!(dec.read_literal(8).unwrap(), 0);
        assert_eq!(dec.read_literal(8).unwrap(), 0xFF);
    }

    #[test]
    fn roundtrip_carry_heavy_runs() {
        let mut bits = Vec::new();
        for i in 0..400u32 {
            bits.push(((i.wrapping_mul(7) & 1), 250 - (i % 200)));
        }
        roundtrip(&bits);
    }

    #[test]
    fn roundtrip_long_low_prob_ones() {
        // p=1 with bit=1 repeatedly keeps the high sub-interval, the
        // worst case for carry into emitted bytes.
        let bits: Vec<(u32, u32)> = (0..500).map(|_| (1, 1)).collect();
        roundtrip(&bits);
    }

    #[test]
    fn roundtrip_pseudorandom_sequences() {
        // Deterministic LCG over a spread of lengths / probabilities.
        for trial in 0..256u32 {
            let n = 1 + trial % 300;
            let mut bits = Vec::with_capacity(n as usize);
            for i in 0..n {
                let h = trial
                    .wrapping_mul(2_654_435_761)
                    .wrapping_add(i.wrapping_mul(40_503));
                let b = (h >> 16) & 1;
                let p = 1 + ((trial * 7 + i * 13) % 254);
                bits.push((b, p));
            }
            roundtrip(&bits);
        }
    }

    #[test]
    fn finish_never_ends_in_superframe_marker() {
        for seed in 0..64u32 {
            let mut enc = BoolEncoder::new();
            for i in 0..50u32 {
                let b = ((seed >> (i % 5)) & 1) ^ (i & 1);
                enc.write_bool(b, 1 + ((seed * 3 + i * 13) % 254));
            }
            let buf = enc.finish();
            let last = *buf.last().unwrap();
            assert_ne!(last & 0xe0, 0xc0, "seed {seed}: final byte is a marker");
        }
    }
}
