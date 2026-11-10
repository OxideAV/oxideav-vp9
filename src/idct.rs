//! VP9 inverse transform process per spec v0.7 §8.7.
//!
//! Round 9 lands the §8.7 inverse transform stage that the §8.6.2
//! reconstruct process invokes after the round-8 dequantization step
//! (`src/dequant.rs`). The pieces this module exposes:
//!
//! * The §8.7.1.1 butterfly primitives — `B` (butterfly rotation, with
//!   the §8.7.1.1 `16 + 32*k` two-multiply fast path), `H` (Hadamard
//!   rotation), `SB` (butterfly into the high-precision `S` array) and
//!   `SH` (Hadamard rotation + rounding out of `S`) — together with the
//!   `cos64` / `sin64` angle functions backed by the verbatim 33-entry
//!   [`COS64_LOOKUP`] table and the `brev` bit-reversal helper.
//! * The §8.7.1.2 inverse-DCT array permutation and the recursive
//!   §8.7.1.3 inverse DCT process ([`inverse_dct`]) for `2 <= n <= 5`.
//! * The §8.7.1.4 / §8.7.1.5 ADST input/output permutations and the
//!   §8.7.1.6 / §8.7.1.7 / §8.7.1.8 ADST4 / ADST8 / ADST16 processes
//!   dispatched by the §8.7.1.9 [`inverse_adst`] for `2 <= n <= 4`.
//! * The §8.7.1.10 inverse Walsh-Hadamard transform ([`inverse_wht`]).
//! * The §8.7.2 2D inverse transform driver ([`inverse_transform_2d`])
//!   that applies the per-`TxType` row transform then column transform
//!   (with the lossless WHT path) to a `1<<n` by `1<<n` `Dequant`
//!   block, including the `Round2( T[ i ], Min( 6, n + 2 ) )` column
//!   rounding.
//!
//! The §8.6.2 reconstruct driver that feeds this module its `Dequant`
//! block (built from the round-7 token magnitudes scaled by the round-8
//! quantizers) and adds the residual to the prediction lands in a later
//! round; this module is the pure transform layer it will call.
//!
//! Provenance: VP9 Bitstream & Decoding Process Specification v0.7
//! (`docs/video/vp9/vp9-spec.txt` §8.7). The `cos64_lookup` table and
//! the `SINPI_*_9` constants are transcribed directly from the spec
//! §8.7.1 listings.

// The §8.6.2 reconstruct process that drives these transforms lands in
// a subsequent round; until then the module is exercised exclusively
// from `#[cfg(test)]`.
#![allow(dead_code)]

/// `TxType` value `DCT_DCT` from §3 (Table of constants): inverse
/// transform rows with DCT and columns with DCT.
pub(crate) const DCT_DCT: u8 = 0;
/// `TxType` value `ADST_DCT` from §3: rows DCT, columns ADST.
pub(crate) const ADST_DCT: u8 = 1;
/// `TxType` value `DCT_ADST` from §3: rows ADST, columns DCT.
pub(crate) const DCT_ADST: u8 = 2;
/// `TxType` value `ADST_ADST` from §3: rows ADST, columns ADST.
pub(crate) const ADST_ADST: u8 = 3;

/// §8.7.1.6 ADST4 constant `SINPI_1_9`.
const SINPI_1_9: i64 = 5283;
/// §8.7.1.6 ADST4 constant `SINPI_2_9`.
const SINPI_2_9: i64 = 9929;
/// §8.7.1.6 ADST4 constant `SINPI_3_9`.
const SINPI_3_9: i64 = 13377;
/// §8.7.1.6 ADST4 constant `SINPI_4_9`.
const SINPI_4_9: i64 = 15212;

/// `cos64_lookup[ 33 ]` from §8.7.1.1, transcribed verbatim. Entry `i`
/// is `round( 16384 * cos( i * pi / 64 ) )` for `i = 0..32`.
pub(crate) const COS64_LOOKUP: [i64; 33] = [
    16384, 16364, 16305, 16207, 16069, 15893, 15679, 15426, 15137, 14811, 14449, 14053, 13623,
    13160, 12665, 12140, 11585, 11003, 10394, 9760, 9102, 8423, 7723, 7005, 6270, 5520, 4756, 3981,
    3196, 2404, 1606, 804, 0,
];

/// `Round2( x, n ) = ( x + (1 << (n − 1)) ) >> n` from §4.6, evaluated
/// in 64-bit precision so the transform's high-precision intermediates
/// do not overflow. `n` is always `>= 1` in the §8.7 transform paths.
#[inline]
fn round2(x: i64, n: u32) -> i64 {
    (x + (1 << (n - 1))) >> n
}

/// `brev( numBits, x )` from §8.7.1.1 — the bit-reversal of the low
/// `numBits` of `x`.
pub(crate) fn brev(num_bits: u32, x: usize) -> usize {
    let mut t = 0usize;
    for i in 0..num_bits {
        let bit = (x >> i) & 1;
        t += bit << (num_bits - 1 - i);
    }
    t
}

/// `cos64( angle )` from §8.7.1.1 — `round( 16384 * cos( angle*pi/64 ) )`
/// realised via the 33-entry quarter-wave [`COS64_LOOKUP`] table.
pub(crate) fn cos64(angle: i32) -> i64 {
    let angle2 = (angle & 127) as usize;
    if angle2 <= 32 {
        COS64_LOOKUP[angle2]
    } else if angle2 <= 64 {
        -COS64_LOOKUP[64 - angle2]
    } else if angle2 <= 96 {
        -COS64_LOOKUP[angle2 - 64]
    } else {
        COS64_LOOKUP[128 - angle2]
    }
}

/// `sin64( angle )` from §8.7.1.1, defined as `cos64( angle - 32 )`.
pub(crate) fn sin64(angle: i32) -> i64 {
    cos64(angle - 32)
}

/// §8.7.1.1 butterfly rotation `B( a, b, angle, flip )` over the array
/// `t`. When `angle == 16 + 32*k` the two-multiply fast path is used.
/// `flip == true` exchanges `t[a]` and `t[b]` afterwards.
fn butterfly(t: &mut [i64], a: usize, b: usize, angle: i32, flip: bool) {
    // §8.7.1.1: the `16 + 32*k` fast path applies when (angle & 31) == 16.
    if (angle & 31) == 16 {
        let (v, w) = if (angle & 32) != 0 {
            (t[a] + t[b], -t[a] + t[b])
        } else {
            (t[a] - t[b], t[a] + t[b])
        };
        let c = cos64(angle);
        let x = v * c;
        let y = w * c;
        t[a] = round2(x, 14);
        t[b] = round2(y, 14);
    } else {
        let x = t[a] * cos64(angle) - t[b] * sin64(angle);
        let y = t[a] * sin64(angle) + t[b] * cos64(angle);
        t[a] = round2(x, 14);
        t[b] = round2(y, 14);
    }
    if flip {
        t.swap(a, b);
    }
}

/// §8.7.1.1 Hadamard rotation `H( a, b, flip )` over the array `t`.
/// `flip == true` is specified as `H( b, a, 0 )`.
fn hadamard(t: &mut [i64], a: usize, b: usize, flip: bool) {
    let (a, b) = if flip { (b, a) } else { (a, b) };
    let x = t[a];
    let y = t[b];
    t[a] = x + y;
    t[b] = x - y;
}

/// §8.7.1.1 `SB( a, b, angle, flip )` — butterfly into the
/// high-precision `s` array. `flip == true` exchanges `s[a]`/`s[b]`.
fn sb(t: &[i64], s: &mut [i64], a: usize, b: usize, angle: i32, flip: bool) {
    s[a] = t[a] * cos64(angle) - t[b] * sin64(angle);
    s[b] = t[a] * sin64(angle) + t[b] * cos64(angle);
    if flip {
        s.swap(a, b);
    }
}

/// §8.7.1.1 `SH( a, b )` — Hadamard rotation + `Round2( ., 14 )` out of
/// the high-precision `s` array back into `t`.
fn sh(t: &mut [i64], s: &[i64], a: usize, b: usize) {
    t[a] = round2(s[a] + s[b], 14);
    t[b] = round2(s[a] - s[b], 14);
}

/// §8.7.1.2 inverse-DCT array permutation: `T[i] = copyT[ brev(n, i) ]`.
fn idct_permute(t: &mut [i64], n: u32) {
    let len = 1usize << n;
    let mut copy_t = [0i64; 32];
    copy_t[..len].copy_from_slice(&t[..len]);
    for i in 0..len {
        t[i] = copy_t[brev(n, i)];
    }
}

/// §8.7.1.3 inverse DCT process over the (already-permuted) array `t`
/// of length `1<<n` for `2 <= n <= 5`. Recursive per the spec.
fn inverse_dct(t: &mut [i64], n: u32) {
    let n0 = 1usize << n;
    let n1 = 1usize << (n - 1);
    let n2 = if n >= 2 { 1usize << (n - 2) } else { 0 };
    let n3 = if n >= 3 { 1usize << (n - 3) } else { 0 };
    let ni = n as i32;

    // 1. Base case n == 2, otherwise recurse on n-1.
    if n == 2 {
        butterfly(t, 0, 1, 16, true);
    } else {
        inverse_dct(t, n - 1);
    }

    // 2. B( n1+i, n0-1-i, 32 - brev(5, n1+i), 0 ) for i = 0..(n2-1).
    for i in 0..n2 {
        let angle = 32 - brev(5, n1 + i) as i32;
        butterfly(t, n1 + i, n0 - 1 - i, angle, false);
    }

    // 3. n >= 3: H( n1+4*i+2*j, n1+1+4*i+2*j, j ) for i=0..(n3-1), j=0..1.
    if n >= 3 {
        for i in 0..n3 {
            for j in 0..2 {
                hadamard(t, n1 + 4 * i + 2 * j, n1 + 1 + 4 * i + 2 * j, j != 0);
            }
        }
    }

    // 4. n == 5.
    if n == 5 {
        // a. B( n0-n+3-n2*j-4*i, n1+n-4+n2*j+4*i, 28-16*i+56*j, 1 ) i,j=0..1.
        for i in 0..2usize {
            for j in 0..2usize {
                let a = n0 as i32 - ni + 3 - n2 as i32 * j as i32 - 4 * i as i32;
                let b = n1 as i32 + ni - 4 + n2 as i32 * j as i32 + 4 * i as i32;
                let angle = 28 - 16 * i as i32 + 56 * j as i32;
                butterfly(t, a as usize, b as usize, angle, true);
            }
        }
        // b. H( n1+n3*j+i, n1+n2-5+n3*j-i, j&1 ) for i=0..1, j=0..3.
        for i in 0..2usize {
            for j in 0..4usize {
                let a = n1 + n3 * j + i;
                let b = n1 + n2 - 5 + n3 * j - i;
                hadamard(t, a, b, (j & 1) != 0);
            }
        }
    }

    // 5. n >= 4.
    if n >= 4 {
        // a. B( n0-n+2-i-n2*j, n1+n-3+i+n2*j, 24+48*j, 1 ) for i=0..(n==5), j=0..1.
        let i_max = if n == 5 { 1usize } else { 0usize };
        for i in 0..=i_max {
            for j in 0..2usize {
                let a = n0 as i32 - ni + 2 - i as i32 - n2 as i32 * j as i32;
                let b = n1 as i32 + ni - 3 + i as i32 + n2 as i32 * j as i32;
                let angle = 24 + 48 * j as i32;
                butterfly(t, a as usize, b as usize, angle, true);
            }
        }
        // b. H( n1+n2*j+i, n1+n2-1+n2*j-i, j&1 ) for i=0..(2n-7), j=0..1.
        let i_max_b = (2 * ni - 7) as usize;
        for i in 0..=i_max_b {
            for j in 0..2usize {
                let a = n1 + n2 * j + i;
                let b = n1 + n2 - 1 + n2 * j - i;
                hadamard(t, a, b, (j & 1) != 0);
            }
        }
    }

    // 6. n >= 3: B( n0-n3-1-i, n1+n3+i, 16, 1 ) for i=0..(n3-1).
    if n >= 3 {
        for i in 0..n3 {
            butterfly(t, n0 - n3 - 1 - i, n1 + n3 + i, 16, true);
        }
    }

    // 7. H( i, n0-1-i, 0 ) for i = 0..(n1-1).
    for i in 0..n1 {
        hadamard(t, i, n0 - 1 - i, false);
    }
}

/// §8.7.1.4 inverse ADST input array permutation over `t` of length
/// `1<<n`.
fn adst_input_permute(t: &mut [i64], n: u32) {
    let n0 = 1usize << n;
    let n1 = 1usize << (n - 1);
    let mut copy_t = [0i64; 32];
    copy_t[..n0].copy_from_slice(&t[..n0]);
    for i in 0..n1 {
        t[2 * i] = copy_t[n0 - 1 - 2 * i];
        t[2 * i + 1] = copy_t[2 * i];
    }
}

/// §8.7.1.5 inverse ADST output array permutation over `t` of length
/// `1<<n` for `n` in `{3, 4}`.
fn adst_output_permute(t: &mut [i64], n: u32) {
    let n0 = 1usize << n;
    let mut copy_t = [0i64; 32];
    copy_t[..n0].copy_from_slice(&t[..n0]);
    if n == 4 {
        for a in 0..2usize {
            for b in 0..2usize {
                for c in 0..2usize {
                    for d in 0..2usize {
                        let dst = 8 * a + 4 * b + 2 * c + d;
                        let src = 8 * (d ^ c) + 4 * (c ^ b) + 2 * (b ^ a) + a;
                        t[dst] = copy_t[src];
                    }
                }
            }
        }
    } else {
        // n == 3
        for a in 0..2usize {
            for b in 0..2usize {
                for c in 0..2usize {
                    let dst = 4 * a + 2 * b + c;
                    let src = 4 * (c ^ b) + 2 * (b ^ a) + a;
                    t[dst] = copy_t[src];
                }
            }
        }
    }
}

/// §8.7.1.6 inverse ADST4 process over `t[0..4]`.
fn inverse_adst4(t: &mut [i64]) {
    let mut s0 = SINPI_1_9 * t[0];
    let mut s1 = SINPI_2_9 * t[0];
    let mut s2 = SINPI_3_9 * t[1];
    let s3 = SINPI_4_9 * t[2];
    let s4 = SINPI_1_9 * t[2];
    let s5 = SINPI_2_9 * t[3];
    let s6 = SINPI_4_9 * t[3];
    let v = t[0] - t[2] + t[3];
    let s7 = SINPI_3_9 * v;

    let x0 = s0 + s3 + s5;
    let x1 = s1 - s4 - s6;
    let x2 = s7;
    let x3 = s2;

    s0 = x0 + x3;
    s1 = x1 + x3;
    s2 = x2;
    let s3b = x0 + x1 - x3;

    t[0] = round2(s0, 14);
    t[1] = round2(s1, 14);
    t[2] = round2(s2, 14);
    t[3] = round2(s3b, 14);
}

/// §8.7.1.7 inverse ADST8 process over `t[0..8]`, using the
/// high-precision `s` scratch array.
fn inverse_adst8(t: &mut [i64]) {
    let mut s = [0i64; 8];
    // 1.
    adst_input_permute(t, 3);
    // 2. SB( 2*i, 1+2*i, 30-8*i, 1 ) for i = 0..3.
    for i in 0..4 {
        sb(t, &mut s, 2 * i, 1 + 2 * i, 30 - 8 * i as i32, true);
    }
    // 3. SH( i, 4+i ) for i = 0..3.
    for i in 0..4 {
        sh(t, &s, i, 4 + i);
    }
    // 4. SB( 4+3*i, 5+i, 24-16*i, 1 ) for i = 0..1.
    for i in 0..2 {
        sb(t, &mut s, 4 + 3 * i, 5 + i, 24 - 16 * i as i32, true);
    }
    // 5. SH( 4+i, 6+i ) for i = 0..1.
    for i in 0..2 {
        sh(t, &s, 4 + i, 6 + i);
    }
    // 6. H( i, 2+i, 0 ) for i = 0..1.
    for i in 0..2 {
        hadamard(t, i, 2 + i, false);
    }
    // 7. B( 2+4*i, 3+4*i, 16, 1 ) for i = 0..1.
    for i in 0..2 {
        butterfly(t, 2 + 4 * i, 3 + 4 * i, 16, true);
    }
    // 8.
    adst_output_permute(t, 3);
    // 9. T[ 1+2*i ] = -T[ 1+2*i ] for i = 0..3.
    for i in 0..4 {
        t[1 + 2 * i] = -t[1 + 2 * i];
    }
}

/// §8.7.1.8 inverse ADST16 process over `t[0..16]`, using the
/// high-precision `s` scratch array.
fn inverse_adst16(t: &mut [i64]) {
    let mut s = [0i64; 16];
    // 1.
    adst_input_permute(t, 4);
    // 2. SB( 2*i, 1+2*i, 31-4*i, 1 ) for i = 0..7.
    for i in 0..8 {
        sb(t, &mut s, 2 * i, 1 + 2 * i, 31 - 4 * i as i32, true);
    }
    // 3. SH( i, 8+i ) for i = 0..7.
    for i in 0..8 {
        sh(t, &s, i, 8 + i);
    }
    // 4. SB( 8+2*i, 9+2*i, 28-16*i, 1 ) for i = 0..3.
    for i in 0..4 {
        sb(t, &mut s, 8 + 2 * i, 9 + 2 * i, 28 - 16 * i as i32, true);
    }
    // 5. SH( 8+i, 12+i ) for i = 0..3.
    for i in 0..4 {
        sh(t, &s, 8 + i, 12 + i);
    }
    // 6. H( i, 4+i, 0 ) for i = 0..3.
    for i in 0..4 {
        hadamard(t, i, 4 + i, false);
    }
    // 7. SB( 4+8*i+3*j, 5+8*i+j, 24-16*j, 1 ) for i = 0..1, j = 0..1.
    for i in 0..2 {
        for j in 0..2 {
            sb(
                t,
                &mut s,
                4 + 8 * i + 3 * j,
                5 + 8 * i + j,
                24 - 16 * j as i32,
                true,
            );
        }
    }
    // 8. SH( 4+8*j+i, 6+8*j+i ) for i = 0..1, j = 0..1.
    for j in 0..2 {
        for i in 0..2 {
            sh(t, &s, 4 + 8 * j + i, 6 + 8 * j + i);
        }
    }
    // 9. H( 8*j+i, 2+8*j+i, 0 ) for i = 0..1, j = 0..1.
    for j in 0..2 {
        for i in 0..2 {
            hadamard(t, 8 * j + i, 2 + 8 * j + i, false);
        }
    }
    // 10. B( 2+4*j+8*i, 3+4*j+8*i, 48+64*(i^j), 0 ) for i = 0..1, j = 0..1.
    for i in 0..2 {
        for j in 0..2 {
            butterfly(
                t,
                2 + 4 * j + 8 * i,
                3 + 4 * j + 8 * i,
                48 + 64 * (i ^ j) as i32,
                false,
            );
        }
    }
    // 11.
    adst_output_permute(t, 4);
    // 12. T[ 1+12*j+2*i ] = -T[ 1+12*j+2*i ] for i = 0..1, j = 0..1.
    for j in 0..2 {
        for i in 0..2 {
            let idx = 1 + 12 * j + 2 * i;
            t[idx] = -t[idx];
        }
    }
}

/// §8.7.1.9 inverse ADST process over `t` of length `1<<n` for
/// `2 <= n <= 4`, dispatching to ADST4 / ADST8 / ADST16.
///
/// `pub(crate)` so the encoder's forward-transform derivation
/// ([`crate::fwd_transform`]) can measure this 1D inverse's response
/// matrix and invert it numerically — the forward ADST8 / ADST16 bases
/// are defined *as* the inverse of this §8.7.1.7 / §8.7.1.8 network.
pub(crate) fn inverse_adst(t: &mut [i64], n: u32) {
    match n {
        2 => inverse_adst4(t),
        3 => inverse_adst8(t),
        _ => inverse_adst16(t),
    }
}

/// §8.7.1.10 inverse Walsh-Hadamard transform over `t[0..4]`. `shift`
/// is the amount of pre-scaling (2 for the row pass, 0 for the column
/// pass, per §8.7.2).
pub(crate) fn inverse_wht(t: &mut [i64], shift: u32) {
    let mut a = t[0] >> shift;
    let mut c = t[1] >> shift;
    let mut d = t[2] >> shift;
    let mut b = t[3] >> shift;
    a += c;
    d -= b;
    let e = (a - d) >> 1;
    b = e - b;
    c = e - c;
    a -= b;
    d += c;
    t[0] = a;
    t[1] = b;
    t[2] = c;
    t[3] = d;
}

/// Applies the §8.7.1 1D transform selected for one pass to `t` (length
/// `1<<n`). `use_dct == true` runs the DCT (permute + process), false
/// runs the ADST.
fn apply_1d(t: &mut [i64], n: u32, use_dct: bool) {
    if use_dct {
        idct_permute(t, n);
        inverse_dct(t, n);
    } else {
        inverse_adst(t, n);
    }
}

/// §8.7.2 2D inverse transform of a `(1<<n)` by `(1<<n)` block stored
/// row-major in `dequant`.
///
/// * `n` is the base-2 log of the transform width (`2..=5`).
/// * `tx_type` is one of [`DCT_DCT`] / [`ADST_DCT`] / [`DCT_ADST`] /
///   [`ADST_ADST`]; ignored when `lossless` is set.
/// * `lossless == true` selects the WHT path on both passes.
///
/// On return `dequant` holds the spatial-domain residual.
pub(crate) fn inverse_transform_2d(dequant: &mut [i64], n: u32, tx_type: u8, lossless: bool) {
    match n {
        2 => inverse_transform_2d_n::<2, 4, 4>(dequant, tx_type, lossless),
        3 => inverse_transform_2d_n::<3, 8, 4>(dequant, tx_type, lossless),
        4 => inverse_transform_2d_n::<4, 16, 4>(dequant, tx_type, lossless),
        _ => inverse_transform_2d_n::<5, 32, 8>(dequant, tx_type, lossless),
    }
}

/// The spec-literal §8.7.2 driver over the recursive §8.7.1 processes
/// — the oracle the fast [`inverse_transform_2d`] is pinned against.
#[cfg(test)]
pub(crate) fn inverse_transform_2d_reference(
    dequant: &mut [i64],
    n: u32,
    tx_type: u8,
    lossless: bool,
) {
    let n0 = 1usize << n;
    let mut t = vec![0i64; n0];

    // Row transforms (i = 0..n0-1).
    for i in 0..n0 {
        let base = i * n0;
        t[..n0].copy_from_slice(&dequant[base..base + n0]);
        if lossless {
            inverse_wht(&mut t, 2);
        } else {
            // Row uses DCT when TxType is DCT_DCT or ADST_DCT.
            let row_dct = tx_type == DCT_DCT || tx_type == ADST_DCT;
            apply_1d(&mut t, n, row_dct);
        }
        dequant[base..base + n0].copy_from_slice(&t[..n0]);
    }

    // Column transforms (j = 0..n0-1).
    for j in 0..n0 {
        for i in 0..n0 {
            t[i] = dequant[i * n0 + j];
        }
        if lossless {
            inverse_wht(&mut t, 0);
        } else {
            // Column uses DCT when TxType is DCT_DCT or DCT_ADST.
            let col_dct = tx_type == DCT_DCT || tx_type == DCT_ADST;
            apply_1d(&mut t, n, col_dct);
        }
        if lossless {
            for i in 0..n0 {
                dequant[i * n0 + j] = t[i];
            }
        } else {
            let shift = core::cmp::min(6, n + 2);
            for i in 0..n0 {
                dequant[i * n0 + j] = round2(t[i], shift);
            }
        }
    }
}

// ───────────── fast §8.7 paths (same arithmetic, unrolled) ─────────────
//
// The processes below perform exactly the §8.7.1 / §8.7.2 operations of
// the spec-literal functions above, in the same order and with the same
// 64-bit intermediates: the recursive §8.7.1.3 DCT is flattened into its
// stage sequence (`inverse_dct( n )` is `B( 0, 1, 16, 1 )` followed by
// the per-level tails for 2..=n), the arrays live on the stack, and the
// transform size is a compile-time constant so every angle / index
// folds. An all-zero input row stays zero through every 1D process (each
// output is a sum of products of the inputs, rounded: Round2( 0, k ) =
// 0), so the row pass skips such rows; a DC-only DCT_DCT block is
// constant (each DCT stage maps a vector whose only non-zero entry is
// T[ 0 ] to a constant vector: the base butterfly writes Round2( T[ 0 ] *
// cos64( 16 ), 14 ) to T[ 0 ] and T[ 1 ], and every later butterfly sees
// zeros while every Hadamard adds a zero), which is evaluated directly.

/// `brev( n, i )` for `n = 2..=5`, `i < 1 << n`.
const BREV: [[u8; 32]; 6] = {
    let mut t = [[0u8; 32]; 6];
    let mut n = 2;
    while n <= 5 {
        let mut i = 0;
        while i < (1 << n) {
            let mut r = 0;
            let mut b = 0;
            while b < n {
                r |= ((i >> b) & 1) << (n - 1 - b);
                b += 1;
            }
            t[n][i] = r as u8;
            i += 1;
        }
        n += 1;
    }
    t
};

#[inline(always)]
fn fb(t: &mut [i64; 32], a: usize, b: usize, angle: i32, flip: bool) {
    let (x, y) = if (angle & 31) == 16 {
        let (v, w) = if (angle & 32) != 0 {
            (t[a] + t[b], -t[a] + t[b])
        } else {
            (t[a] - t[b], t[a] + t[b])
        };
        let c = cos64(angle);
        (v * c, w * c)
    } else {
        (
            t[a] * cos64(angle) - t[b] * sin64(angle),
            t[a] * sin64(angle) + t[b] * cos64(angle),
        )
    };
    if flip {
        t[b] = round2(x, 14);
        t[a] = round2(y, 14);
    } else {
        t[a] = round2(x, 14);
        t[b] = round2(y, 14);
    }
}

#[inline(always)]
fn fh(t: &mut [i64; 32], a: usize, b: usize, flip: bool) {
    let (a, b) = if flip { (b, a) } else { (a, b) };
    let (x, y) = (t[a], t[b]);
    t[a] = x + y;
    t[b] = x - y;
}

/// The §8.7.1.3 level-`n` tail (everything after the recursive call).
#[inline(always)]
fn dct_level(t: &mut [i64; 32], n: u32) {
    let n0 = 1usize << n;
    let n1 = 1usize << (n - 1);
    let n2 = 1usize << (n - 2);
    let n3 = if n >= 3 { 1usize << (n - 3) } else { 0 };
    let ni = n as i32;
    for i in 0..n2 {
        let angle = 32 - BREV[5][n1 + i] as i32;
        fb(t, n1 + i, n0 - 1 - i, angle, false);
    }
    if n >= 3 {
        for i in 0..n3 {
            for j in 0..2 {
                fh(t, n1 + 4 * i + 2 * j, n1 + 1 + 4 * i + 2 * j, j != 0);
            }
        }
    }
    if n == 5 {
        for i in 0..2usize {
            for j in 0..2usize {
                let a = n0 as i32 - ni + 3 - n2 as i32 * j as i32 - 4 * i as i32;
                let b = n1 as i32 + ni - 4 + n2 as i32 * j as i32 + 4 * i as i32;
                fb(
                    t,
                    a as usize,
                    b as usize,
                    28 - 16 * i as i32 + 56 * j as i32,
                    true,
                );
            }
        }
        for i in 0..2usize {
            for j in 0..4usize {
                fh(t, n1 + n3 * j + i, n1 + n2 - 5 + n3 * j - i, (j & 1) != 0);
            }
        }
    }
    if n >= 4 {
        let i_max = if n == 5 { 1usize } else { 0usize };
        for i in 0..=i_max {
            for j in 0..2usize {
                let a = n0 as i32 - ni + 2 - i as i32 - n2 as i32 * j as i32;
                let b = n1 as i32 + ni - 3 + i as i32 + n2 as i32 * j as i32;
                fb(t, a as usize, b as usize, 24 + 48 * j as i32, true);
            }
        }
        for i in 0..=(2 * ni - 7) as usize {
            for j in 0..2usize {
                fh(t, n1 + n2 * j + i, n1 + n2 - 1 + n2 * j - i, (j & 1) != 0);
            }
        }
    }
    if n >= 3 {
        for i in 0..n3 {
            fb(t, n0 - n3 - 1 - i, n1 + n3 + i, 16, true);
        }
    }
    for i in 0..n1 {
        fh(t, i, n0 - 1 - i, false);
    }
}

/// §8.7.1.2 permutation + §8.7.1.3 inverse DCT of `t[ ..1 << N ]`.
#[inline(always)]
fn idct_fast<const N: u32>(t: &mut [i64; 32]) {
    let len = 1usize << N;
    let c = *t;
    for i in 0..len {
        t[i] = c[BREV[N as usize][i] as usize];
    }
    fb(t, 0, 1, 16, true);
    for m in 2..=N {
        dct_level(t, m);
    }
}

#[inline(always)]
fn apply_1d_fast<const N: u32>(t: &mut [i64; 32], use_dct: bool) {
    if use_dct {
        idct_fast::<N>(t);
    } else {
        inverse_adst(&mut t[..1 << N], N);
    }
}

/// [`inverse_transform_2d`] at a constant size `N` (see the section
/// note for the zero-row and DC-only shortcuts).
fn inverse_transform_2d_n<const N: u32, const L: usize, const Q: usize>(
    dequant: &mut [i64],
    tx_type: u8,
    lossless: bool,
) {
    let n0 = 1usize << N;
    let block = &mut dequant[..n0 * n0];
    let shift = core::cmp::min(6, N + 2);

    if !lossless && tx_type == DCT_DCT && block[1..].iter().all(|&v| v == 0) {
        block.fill(dc_only_residual(block[0], N));
        return;
    }

    let row_dct = tx_type == DCT_DCT || tx_type == ADST_DCT;
    let col_dct = tx_type == DCT_DCT || tx_type == DCT_ADST;
    if !lossless {
        // Rows below the last non-zero one stay zero through the row
        // pass: a block whose non-zero rows fit `Q` lanes runs it on `Q`.
        let few_rows = Q < L && block[Q * L..].iter().all(|&v| v == 0);
        let done = if few_rows {
            lanes_2d::<N, L, Q>(block, row_dct, col_dct, shift)
        } else {
            lanes_2d::<N, L, L>(block, row_dct, col_dct, shift)
        };
        if done {
            return;
        }
    }
    inverse_transform_2d_scalar::<N>(block, row_dct, col_dct, lossless, shift);
}

/// The one-vector-at-a-time 64-bit form of [`inverse_transform_2d_n`]
/// (lossless blocks, and blocks the lane form declines).
fn inverse_transform_2d_scalar<const N: u32>(
    block: &mut [i64],
    row_dct: bool,
    col_dct: bool,
    lossless: bool,
    shift: u32,
) {
    let n0 = 1usize << N;
    let mut t = [0i64; 32];
    for row in block.chunks_exact_mut(n0) {
        if row.iter().all(|&v| v == 0) {
            continue;
        }
        t[..n0].copy_from_slice(row);
        if lossless {
            inverse_wht(&mut t, 2);
        } else {
            apply_1d_fast::<N>(&mut t, row_dct);
        }
        row.copy_from_slice(&t[..n0]);
    }

    for j in 0..n0 {
        for i in 0..n0 {
            t[i] = block[i * n0 + j];
        }
        if lossless {
            inverse_wht(&mut t, 0);
            for i in 0..n0 {
                block[i * n0 + j] = t[i];
            }
        } else {
            apply_1d_fast::<N>(&mut t, col_dct);
            for i in 0..n0 {
                block[i * n0 + j] = round2(t[i], shift);
            }
        }
    }
}

// ───────────── lane form: every row / column of a block at once ─────────────
//
// The same §8.7.1 stage sequences as above with each array entry `T[ i ]`
// widened to `L = 1 << n` lanes — one lane per row of the block in the
// row pass, one per column in the column pass — so every butterfly /
// Hadamard is a short loop over lanes of `i32`.
//
// 32-bit lanes are exact while every value stored in `T` fits a signed
// 16-bit integer: a butterfly then forms `a * c + b * s` with `|a|, |b| <
// 1 << 15` and `c * c + s * s <= 1 << 28` (to rounding), so the sum stays
// under `1 << 30`; the ADST's high-precision `S` entries are bounded by
// `1 << 29.5` and an `SH` adds two of them plus the rounding constant,
// under `1 << 31`; the ADST4 sums carry weights of at most 43801 on
// 16-bit inputs. Each stage's outputs are therefore the exact values as
// long as its inputs passed the check, and the check is applied to the
// inputs and to every value a stage stores (`note`): by induction a
// clean flag proves the whole 32-bit evaluation equals the 64-bit one.
// A block that trips the flag is evaluated by the 64-bit form from its
// untouched input instead.

type Lanes<const L: usize> = [i32; L];

/// Fold `v` into the range flag: the flag's bits 16.. stay clear exactly
/// while every noted value lies in `-32768..=32767`.
#[inline(always)]
fn note<const L: usize>(flag: &mut Lanes<L>, v: &Lanes<L>) {
    for l in 0..L {
        flag[l] |= v[l].wrapping_add(0x8000);
    }
}

#[inline(always)]
fn round14(x: i32) -> i32 {
    x.wrapping_add(1 << 13) >> 14
}

/// §8.7.1.1 `B( a, b, angle, flip )` on lanes.
#[inline(always)]
fn lb<const L: usize>(
    t: &mut [Lanes<L>; 32],
    flag: &mut Lanes<L>,
    a: usize,
    b: usize,
    angle: i32,
    flip: bool,
) {
    let (ta, tb) = (t[a], t[b]);
    let (mut x, mut y) = ([0i32; L], [0i32; L]);
    let c = cos64(angle) as i32;
    if (angle & 31) == 16 {
        let neg = (angle & 32) != 0;
        for l in 0..L {
            let (v, w) = if neg {
                (ta[l].wrapping_add(tb[l]), tb[l].wrapping_sub(ta[l]))
            } else {
                (ta[l].wrapping_sub(tb[l]), ta[l].wrapping_add(tb[l]))
            };
            x[l] = round14(v.wrapping_mul(c));
            y[l] = round14(w.wrapping_mul(c));
        }
    } else {
        let s = sin64(angle) as i32;
        for l in 0..L {
            x[l] = round14(ta[l].wrapping_mul(c).wrapping_sub(tb[l].wrapping_mul(s)));
            y[l] = round14(ta[l].wrapping_mul(s).wrapping_add(tb[l].wrapping_mul(c)));
        }
    }
    note(flag, &x);
    note(flag, &y);
    if flip {
        t[b] = x;
        t[a] = y;
    } else {
        t[a] = x;
        t[b] = y;
    }
}

/// §8.7.1.1 `H( a, b, flip )` on lanes.
#[inline(always)]
fn lh<const L: usize>(t: &mut [Lanes<L>; 32], flag: &mut Lanes<L>, a: usize, b: usize, flip: bool) {
    let (a, b) = if flip { (b, a) } else { (a, b) };
    let (ta, tb) = (t[a], t[b]);
    let (mut x, mut y) = ([0i32; L], [0i32; L]);
    for l in 0..L {
        x[l] = ta[l].wrapping_add(tb[l]);
        y[l] = ta[l].wrapping_sub(tb[l]);
    }
    note(flag, &x);
    note(flag, &y);
    t[a] = x;
    t[b] = y;
}

/// §8.7.1.1 `SB( a, b, angle, flip )` on lanes.
#[inline(always)]
fn lsb<const L: usize>(
    t: &[Lanes<L>; 32],
    s: &mut [Lanes<L>; 16],
    a: usize,
    b: usize,
    angle: i32,
    flip: bool,
) {
    let (ta, tb) = (t[a], t[b]);
    let (c, sn) = (cos64(angle) as i32, sin64(angle) as i32);
    let (mut x, mut y) = ([0i32; L], [0i32; L]);
    for l in 0..L {
        x[l] = ta[l].wrapping_mul(c).wrapping_sub(tb[l].wrapping_mul(sn));
        y[l] = ta[l].wrapping_mul(sn).wrapping_add(tb[l].wrapping_mul(c));
    }
    if flip {
        s[b] = x;
        s[a] = y;
    } else {
        s[a] = x;
        s[b] = y;
    }
}

/// §8.7.1.1 `SH( a, b )` on lanes.
#[inline(always)]
fn lsh<const L: usize>(
    t: &mut [Lanes<L>; 32],
    flag: &mut Lanes<L>,
    s: &[Lanes<L>; 16],
    a: usize,
    b: usize,
) {
    let (sa, sb) = (s[a], s[b]);
    let (mut x, mut y) = ([0i32; L], [0i32; L]);
    for l in 0..L {
        x[l] = round14(sa[l].wrapping_add(sb[l]));
        y[l] = round14(sa[l].wrapping_sub(sb[l]));
    }
    note(flag, &x);
    note(flag, &y);
    t[a] = x;
    t[b] = y;
}

/// The §8.7.1.3 level-`n` tail on lanes (the stage list of
/// [`dct_level`]).
#[inline(always)]
fn dct_level_lanes<const L: usize>(t: &mut [Lanes<L>; 32], flag: &mut Lanes<L>, n: u32) {
    let n0 = 1usize << n;
    let n1 = 1usize << (n - 1);
    let n2 = 1usize << (n - 2);
    let n3 = if n >= 3 { 1usize << (n - 3) } else { 0 };
    let ni = n as i32;
    for i in 0..n2 {
        let angle = 32 - BREV[5][n1 + i] as i32;
        lb(t, flag, n1 + i, n0 - 1 - i, angle, false);
    }
    if n >= 3 {
        for i in 0..n3 {
            for j in 0..2 {
                lh(t, flag, n1 + 4 * i + 2 * j, n1 + 1 + 4 * i + 2 * j, j != 0);
            }
        }
    }
    if n == 5 {
        for i in 0..2usize {
            for j in 0..2usize {
                let a = n0 as i32 - ni + 3 - n2 as i32 * j as i32 - 4 * i as i32;
                let b = n1 as i32 + ni - 4 + n2 as i32 * j as i32 + 4 * i as i32;
                lb(
                    t,
                    flag,
                    a as usize,
                    b as usize,
                    28 - 16 * i as i32 + 56 * j as i32,
                    true,
                );
            }
        }
        for i in 0..2usize {
            for j in 0..4usize {
                lh(
                    t,
                    flag,
                    n1 + n3 * j + i,
                    n1 + n2 - 5 + n3 * j - i,
                    (j & 1) != 0,
                );
            }
        }
    }
    if n >= 4 {
        let i_max = if n == 5 { 1usize } else { 0usize };
        for i in 0..=i_max {
            for j in 0..2usize {
                let a = n0 as i32 - ni + 2 - i as i32 - n2 as i32 * j as i32;
                let b = n1 as i32 + ni - 3 + i as i32 + n2 as i32 * j as i32;
                lb(t, flag, a as usize, b as usize, 24 + 48 * j as i32, true);
            }
        }
        for i in 0..=(2 * ni - 7) as usize {
            for j in 0..2usize {
                lh(
                    t,
                    flag,
                    n1 + n2 * j + i,
                    n1 + n2 - 1 + n2 * j - i,
                    (j & 1) != 0,
                );
            }
        }
    }
    if n >= 3 {
        for i in 0..n3 {
            lb(t, flag, n0 - n3 - 1 - i, n1 + n3 + i, 16, true);
        }
    }
    for i in 0..n1 {
        lh(t, flag, i, n0 - 1 - i, false);
    }
}

/// §8.7.1.2 permutation + §8.7.1.3 inverse DCT on lanes.
#[inline(always)]
fn idct_lanes<const N: u32, const L: usize>(t: &mut [Lanes<L>; 32], flag: &mut Lanes<L>) {
    let c = *t;
    for i in 0..1usize << N {
        t[i] = c[BREV[N as usize][i] as usize];
    }
    lb(t, flag, 0, 1, 16, true);
    for m in 2..=N {
        dct_level_lanes(t, flag, m);
    }
}

/// §8.7.1.6 inverse ADST4 on lanes.
#[inline(always)]
fn adst4_lanes<const L: usize>(t: &mut [Lanes<L>; 32], flag: &mut Lanes<L>) {
    const S1: i32 = SINPI_1_9 as i32;
    const S2: i32 = SINPI_2_9 as i32;
    const S3: i32 = SINPI_3_9 as i32;
    const S4: i32 = SINPI_4_9 as i32;
    let (t0, t1, t2, t3) = (t[0], t[1], t[2], t[3]);
    let mut o = [[0i32; L]; 4];
    for l in 0..L {
        let s0 = S1.wrapping_mul(t0[l]);
        let s1 = S2.wrapping_mul(t0[l]);
        let s2 = S3.wrapping_mul(t1[l]);
        let s3 = S4.wrapping_mul(t2[l]);
        let s4 = S1.wrapping_mul(t2[l]);
        let s5 = S2.wrapping_mul(t3[l]);
        let s6 = S4.wrapping_mul(t3[l]);
        let v = t0[l].wrapping_sub(t2[l]).wrapping_add(t3[l]);
        let s7 = S3.wrapping_mul(v);
        let x0 = s0.wrapping_add(s3).wrapping_add(s5);
        let x1 = s1.wrapping_sub(s4).wrapping_sub(s6);
        let x2 = s7;
        let x3 = s2;
        o[0][l] = round14(x0.wrapping_add(x3));
        o[1][l] = round14(x1.wrapping_add(x3));
        o[2][l] = round14(x2);
        o[3][l] = round14(x0.wrapping_add(x1).wrapping_sub(x3));
    }
    for (i, row) in o.iter().enumerate() {
        note(flag, row);
        t[i] = *row;
    }
}

/// `T[ i ] = -T[ i ]` on lanes (the negated value is a stored value).
#[inline(always)]
fn negate_lanes<const L: usize>(t: &mut [Lanes<L>; 32], flag: &mut Lanes<L>, i: usize) {
    let mut v = t[i];
    for x in v.iter_mut() {
        *x = x.wrapping_neg();
    }
    note(flag, &v);
    t[i] = v;
}

/// §8.7.1.4 input permutation on lanes.
#[inline(always)]
fn adst_input_permute_lanes<const L: usize>(t: &mut [Lanes<L>; 32], n: u32) {
    let n0 = 1usize << n;
    let c = *t;
    for i in 0..n0 / 2 {
        t[2 * i] = c[n0 - 1 - 2 * i];
        t[2 * i + 1] = c[2 * i];
    }
}

/// §8.7.1.5 output permutation on lanes.
#[inline(always)]
fn adst_output_permute_lanes<const L: usize>(t: &mut [Lanes<L>; 32], n: u32) {
    let c = *t;
    if n == 4 {
        for (dst, slot) in t.iter_mut().enumerate().take(16) {
            let (a, b, cc, d) = (dst >> 3, (dst >> 2) & 1, (dst >> 1) & 1, dst & 1);
            *slot = c[8 * (d ^ cc) + 4 * (cc ^ b) + 2 * (b ^ a) + a];
        }
    } else {
        for (dst, slot) in t.iter_mut().enumerate().take(8) {
            let (a, b, cc) = (dst >> 2, (dst >> 1) & 1, dst & 1);
            *slot = c[4 * (cc ^ b) + 2 * (b ^ a) + a];
        }
    }
}

/// §8.7.1.7 inverse ADST8 on lanes (the stage list of
/// [`inverse_adst8`]).
#[inline(always)]
fn adst8_lanes<const L: usize>(t: &mut [Lanes<L>; 32], flag: &mut Lanes<L>) {
    let mut s = [[0i32; L]; 16];
    adst_input_permute_lanes(t, 3);
    for i in 0..4 {
        lsb(t, &mut s, 2 * i, 1 + 2 * i, 30 - 8 * i as i32, true);
    }
    for i in 0..4 {
        lsh(t, flag, &s, i, 4 + i);
    }
    for i in 0..2 {
        lsb(t, &mut s, 4 + 3 * i, 5 + i, 24 - 16 * i as i32, true);
    }
    for i in 0..2 {
        lsh(t, flag, &s, 4 + i, 6 + i);
    }
    for i in 0..2 {
        lh(t, flag, i, 2 + i, false);
    }
    for i in 0..2 {
        lb(t, flag, 2 + 4 * i, 3 + 4 * i, 16, true);
    }
    adst_output_permute_lanes(t, 3);
    for i in 0..4 {
        negate_lanes(t, flag, 1 + 2 * i);
    }
}

/// §8.7.1.8 inverse ADST16 on lanes (the stage list of
/// [`inverse_adst16`]).
#[inline(always)]
fn adst16_lanes<const L: usize>(t: &mut [Lanes<L>; 32], flag: &mut Lanes<L>) {
    let mut s = [[0i32; L]; 16];
    adst_input_permute_lanes(t, 4);
    for i in 0..8 {
        lsb(t, &mut s, 2 * i, 1 + 2 * i, 31 - 4 * i as i32, true);
    }
    for i in 0..8 {
        lsh(t, flag, &s, i, 8 + i);
    }
    for i in 0..4 {
        lsb(t, &mut s, 8 + 2 * i, 9 + 2 * i, 28 - 16 * i as i32, true);
    }
    for i in 0..4 {
        lsh(t, flag, &s, 8 + i, 12 + i);
    }
    for i in 0..4 {
        lh(t, flag, i, 4 + i, false);
    }
    for i in 0..2 {
        for j in 0..2 {
            lsb(
                t,
                &mut s,
                4 + 8 * i + 3 * j,
                5 + 8 * i + j,
                24 - 16 * j as i32,
                true,
            );
        }
    }
    for j in 0..2 {
        for i in 0..2 {
            lsh(t, flag, &s, 4 + 8 * j + i, 6 + 8 * j + i);
        }
    }
    for j in 0..2 {
        for i in 0..2 {
            lh(t, flag, 8 * j + i, 2 + 8 * j + i, false);
        }
    }
    for i in 0..2 {
        for j in 0..2 {
            lb(
                t,
                flag,
                2 + 4 * j + 8 * i,
                3 + 4 * j + 8 * i,
                48 + 64 * (i ^ j) as i32,
                false,
            );
        }
    }
    adst_output_permute_lanes(t, 4);
    for j in 0..2 {
        for i in 0..2 {
            negate_lanes(t, flag, 1 + 12 * j + 2 * i);
        }
    }
}

#[inline(always)]
fn apply_1d_lanes<const N: u32, const L: usize>(
    t: &mut [Lanes<L>; 32],
    flag: &mut Lanes<L>,
    use_dct: bool,
) {
    if use_dct {
        idct_lanes::<N, L>(t, flag);
    } else {
        match N {
            2 => adst4_lanes(t, flag),
            3 => adst8_lanes(t, flag),
            _ => adst16_lanes(t, flag),
        }
    }
}

/// The §8.7.2 row and column passes of a non-lossless block on 32-bit
/// lanes. Returns `false` — leaving `block` untouched — when a value
/// left the 16-bit range the lanes are exact for (see the section note).
#[inline(always)]
fn lanes_2d<const N: u32, const L: usize, const R: usize>(
    block: &mut [i64],
    row_dct: bool,
    col_dct: bool,
    shift: u32,
) -> bool {
    debug_assert_eq!(block.len(), L * L);
    debug_assert!(R <= L && block[R * L..].iter().all(|&v| v == 0));
    let mut res = [[0i32; L]; 32];
    let coefs = block[..R * L].iter().copied().enumerate();
    if !lanes_2d_from::<N, L, R>(coefs, row_dct, col_dct, shift, &mut res) {
        return false;
    }
    for (row, src) in block.as_chunks_mut::<L>().0.iter_mut().zip(&res) {
        for (d, &v) in row.iter_mut().zip(src) {
            *d = i64::from(v);
        }
    }
    true
}

/// [`lanes_2d`] from `( position, Dequant value )` pairs — every entry
/// not listed is zero, every listed position lies in the first `R` rows
/// — into `res` (`res[ i ][ j ]`: the residual of row `i`, column `j`).
/// On `false`, `res` holds no result.
#[inline(always)]
fn lanes_2d_from<const N: u32, const L: usize, const R: usize>(
    coefs: impl Iterator<Item = (usize, i64)>,
    row_dct: bool,
    col_dct: bool,
    shift: u32,
    res: &mut [Lanes<L>; 32],
) -> bool {
    debug_assert_eq!(L, 1usize << N);
    // Row pass over the first R rows (the rest are zero and stay zero):
    // entry j of row i lives in lane i of t[ j ].
    let mut t = [[0i32; R]; 32];
    let mut wide = 0u64;
    for (p, v) in coefs {
        wide |= v.wrapping_add(0x8000) as u64;
        t[p & (L - 1)][p >> N] = v as i32;
    }
    if wide >> 16 != 0 {
        return false;
    }
    let mut row_flag = [0i32; R];
    apply_1d_lanes::<N, R>(&mut t, &mut row_flag, row_dct);
    // Column pass: entry i of column j lives in lane j of res[ i ].
    for (i, row) in res.iter_mut().enumerate().take(L) {
        if i < R {
            for j in 0..L {
                row[j] = t[j][i];
            }
        } else {
            *row = [0; L];
        }
    }
    let mut flag = [0i32; L];
    apply_1d_lanes::<N, L>(res, &mut flag, col_dct);
    let seen = flag.iter().chain(&row_flag).fold(0, |a, &f| a | f);
    if seen as u32 >> 16 != 0 {
        return false;
    }
    let half = 1i32 << (shift - 1);
    for row in res.iter_mut().take(L) {
        for v in row.iter_mut() {
            *v = (*v + half) >> shift;
        }
    }
    true
}

/// The §8.7.2 inverse transform of a non-lossless block given as its
/// non-zero `( position, Dequant value )` pairs, on lanes: `res[ i ][ j ]`
/// receives the residual. `rows` bounds the rows holding a listed
/// position (`position >> N < rows`). Returns `false` (no result) for a
/// block outside the lanes' exact range — the caller then runs
/// [`inverse_transform_2d`] on the dense block.
#[inline(always)]
pub(crate) fn inverse_transform_2d_sparse<const N: u32, const L: usize, const Q: usize>(
    coefs: impl Iterator<Item = (usize, i64)>,
    rows: usize,
    tx_type: u8,
    res: &mut [[i32; L]; 32],
) -> bool {
    let shift = core::cmp::min(6, N + 2);
    let row_dct = tx_type == DCT_DCT || tx_type == ADST_DCT;
    let col_dct = tx_type == DCT_DCT || tx_type == DCT_ADST;
    if rows <= Q {
        lanes_2d_from::<N, L, Q>(coefs, row_dct, col_dct, shift, res)
    } else {
        lanes_2d_from::<N, L, L>(coefs, row_dct, col_dct, shift, res)
    }
}

/// The residual of a `DCT_DCT` block whose only non-zero entry is
/// `Dequant[ 0 ][ 0 ] = dc` (every sample equals it — see the fast-path
/// section note).
#[inline]
pub(crate) fn dc_only_residual(dc: i64, n: u32) -> i64 {
    let c = COS64_LOOKUP[16];
    let row = round2(dc * c, 14);
    let col = round2(row * c, 14);
    round2(col, core::cmp::min(6, n + 2))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The lane form takes every conforming-range block (values of the
    /// size a `BitDepth = 8` residual produces) and declines — never
    /// mis-evaluates — blocks that leave the 16-bit range; either way the
    /// driver's output is the spec-literal one (covered together with
    /// `fast_transform_matches_spec_literal`, whose magnitudes straddle
    /// the boundary).
    #[test]
    fn lane_form_accepts_small_blocks_and_declines_wide_ones() {
        let mut seed = 0x0bad_5eed_1234_5678u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let (mut taken, mut declined) = (0u32, 0u32);
        for n in 2..=5u32 {
            let n0 = 1usize << n;
            for tx_type in [DCT_DCT, ADST_DCT, DCT_ADST, ADST_ADST] {
                if n == 5 && tx_type != DCT_DCT {
                    continue;
                }
                let row_dct = tx_type == DCT_DCT || tx_type == ADST_DCT;
                let col_dct = tx_type == DCT_DCT || tx_type == DCT_ADST;
                let shift = core::cmp::min(6, n + 2);
                for trial in 0..600 {
                    let bits = 4 + trial % 14;
                    let mut block = vec![0i64; n0 * n0];
                    // Every other trial keeps only the rows the reduced
                    // row pass covers.
                    let few = trial % 2 == 1;
                    let live = if few {
                        n0 * (n0 / 4).max(4).min(n0)
                    } else {
                        n0 * n0
                    };
                    for v in block[..live].iter_mut() {
                        if next() % 3 != 0 {
                            let mag = (next() % (1 << bits)) as i64;
                            *v = if next() & 1 == 0 { mag } else { -mag };
                        }
                    }
                    let mut want = block.clone();
                    inverse_transform_2d_reference(&mut want, n, tx_type, false);
                    let mut got = block.clone();
                    let ok = match (n, few) {
                        (2, _) => lanes_2d::<2, 4, 4>(&mut got, row_dct, col_dct, shift),
                        (3, false) => lanes_2d::<3, 8, 8>(&mut got, row_dct, col_dct, shift),
                        (3, true) => lanes_2d::<3, 8, 4>(&mut got, row_dct, col_dct, shift),
                        (4, false) => lanes_2d::<4, 16, 16>(&mut got, row_dct, col_dct, shift),
                        (4, true) => lanes_2d::<4, 16, 4>(&mut got, row_dct, col_dct, shift),
                        (_, false) => lanes_2d::<5, 32, 32>(&mut got, row_dct, col_dct, shift),
                        (_, true) => lanes_2d::<5, 32, 8>(&mut got, row_dct, col_dct, shift),
                    };
                    if ok {
                        taken += 1;
                        assert_eq!(got, want, "n={n} tx_type={tx_type} bits={bits}");
                    } else {
                        declined += 1;
                        assert_eq!(got, block, "a declined block is left untouched");
                    }
                    if bits <= 6 {
                        assert!(ok, "small block declined: n={n} tx_type={tx_type}");
                    }
                }
            }
        }
        assert!(taken > 1000 && declined > 1000, "{taken} / {declined}");
    }

    /// The fast driver is the spec-literal one, bit for bit, at every
    /// size / type (including the lossless WHT path) on sparse and dense
    /// random coefficient blocks.
    #[test]
    fn fast_transform_matches_spec_literal() {
        let mut seed = 0x1234_5678_9abc_def0u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for n in 2..=5u32 {
            let n0 = 1usize << n;
            for tx_type in [DCT_DCT, ADST_DCT, DCT_ADST, ADST_ADST] {
                if n == 5 && tx_type != DCT_DCT {
                    continue;
                }
                for lossless in [false, true] {
                    if lossless && n != 2 {
                        continue;
                    }
                    for trial in 0..400 {
                        let mut block = vec![0i64; n0 * n0];
                        let density = trial % 5;
                        for (k, v) in block.iter_mut().enumerate() {
                            let r = next();
                            let keep = match density {
                                0 => k == 0,
                                1 => k < 3 && r % 2 == 0,
                                2 => r % 7 == 0,
                                3 => r % 2 == 0,
                                _ => true,
                            };
                            if keep {
                                let mag = (next() % (1 << (6 + trial % 12))) as i64;
                                *v = if next() & 1 == 0 { mag } else { -mag };
                            }
                        }
                        let mut a = block.clone();
                        let mut b = block;
                        inverse_transform_2d(&mut a, n, tx_type, lossless);
                        inverse_transform_2d_reference(&mut b, n, tx_type, lossless);
                        assert_eq!(a, b, "n={n} tx_type={tx_type} lossless={lossless}");
                    }
                }
            }
        }
    }

    #[test]
    fn cos64_lookup_anchors() {
        // §8.7.1.1: cos64_lookup is 33 entries, peak 16384, zero at 32.
        assert_eq!(COS64_LOOKUP.len(), 33);
        assert_eq!(COS64_LOOKUP[0], 16384);
        assert_eq!(COS64_LOOKUP[32], 0);
    }

    #[test]
    fn cos64_quarter_wave_symmetry() {
        // cos64(0) = 16384, cos64(32) = 0, cos64(64) = -16384.
        assert_eq!(cos64(0), 16384);
        assert_eq!(cos64(32), 0);
        assert_eq!(cos64(64), -16384);
        // §8.7.1.1 step 3: angle2 in (32,64] -> -cos64_lookup[64-angle2].
        assert_eq!(cos64(48), -COS64_LOOKUP[16]);
        // step 4: angle2 in (64,96] -> -cos64_lookup[angle2-64].
        assert_eq!(cos64(80), -COS64_LOOKUP[16]);
        // step 5: angle2 in (96,128) -> cos64_lookup[128-angle2].
        assert_eq!(cos64(112), COS64_LOOKUP[16]);
        // periodicity: angle & 127.
        assert_eq!(cos64(128), cos64(0));
    }

    #[test]
    fn sin64_is_cos64_shifted() {
        // sin64(angle) = cos64(angle - 32); sin64(32) = cos64(0).
        assert_eq!(sin64(32), 16384);
        assert_eq!(sin64(0), cos64(-32));
        assert_eq!(sin64(64), cos64(32));
    }

    #[test]
    fn brev_reverses_low_bits() {
        // §8.7.1.1 brev: reverse low numBits.
        assert_eq!(brev(5, 0b00001), 0b10000);
        assert_eq!(brev(5, 0b00011), 0b11000);
        assert_eq!(brev(3, 0b001), 0b100);
        assert_eq!(brev(2, 0b01), 0b10);
        assert_eq!(brev(4, 0b1010), 0b0101);
    }

    #[test]
    fn hadamard_sum_and_difference() {
        // H(a,b,0): T[a]=x+y, T[b]=x-y.
        let mut t = [3i64, 5];
        hadamard(&mut t, 0, 1, false);
        assert_eq!(t, [8, -2]);
        // flip is specified as H(b,a,0): with t=[3,5], x=t[1]=5, y=t[0]=3,
        // so t[1]=x+y=8 and t[0]=x-y=2.
        let mut t = [3i64, 5];
        hadamard(&mut t, 0, 1, true);
        assert_eq!(t, [2, 8]);
    }

    #[test]
    fn wht_round_trips_known_vector() {
        // The forward WHT (with matching scaling) of [a,b,c,d] is its own
        // structural inverse up to the spec's fixed scaling. Verify the
        // §8.7.1.10 ordered steps directly for a hand vector.
        let mut t = [40i64, 0, 0, 0]; // shift 2 -> a=10, rest 0
        inverse_wht(&mut t, 2);
        // a=10, c=0, d=0, b=0; a+=c=10; d-=b=0; e=(10-0)>>1=5;
        // b=5-0=5; c=5-0=5; a-=b=5; d+=c=5.
        assert_eq!(t, [5, 5, 5, 5]);
    }

    #[test]
    fn wht_dc_only_is_flat() {
        // A pure DC input through both WHT passes (shift 2 then 0) should
        // spread to a flat block. Build a 4x4 DC-only block.
        let mut block = vec![0i64; 16];
        block[0] = 64; // DC coefficient
        inverse_transform_2d(&mut block, 2, DCT_DCT, true);
        // Lossless 4x4: all 16 outputs equal.
        let first = block[0];
        assert!(block.iter().all(|&v| v == first));
    }

    #[test]
    fn idct_permute_is_bit_reversal() {
        // §8.7.1.2: T[i] = copyT[brev(n,i)].
        let mut t: Vec<i64> = (0..4).collect();
        idct_permute(&mut t, 2);
        // brev(2, .): 0->0, 1->2, 2->1, 3->3.
        assert_eq!(t, vec![0, 2, 1, 3]);
    }

    #[test]
    fn dct4_dc_only_produces_flat_row() {
        // For a DCT, a DC-only input yields a constant output row. Run a
        // 1D 4-point inverse DCT on [k,0,0,0] after the permute.
        let mut t = [128i64, 0, 0, 0];
        idct_permute(&mut t, 2);
        inverse_dct(&mut t, 2);
        // All four outputs must be equal (flat) and non-zero.
        assert_eq!(t[0], t[1]);
        assert_eq!(t[1], t[2]);
        assert_eq!(t[2], t[3]);
        assert_ne!(t[0], 0);
    }

    #[test]
    fn dct8_dc_only_flat() {
        let mut t = [256i64, 0, 0, 0, 0, 0, 0, 0];
        idct_permute(&mut t, 3);
        inverse_dct(&mut t, 3);
        for w in t.windows(2) {
            assert_eq!(w[0], w[1]);
        }
        assert_ne!(t[0], 0);
    }

    #[test]
    fn dct16_dc_only_flat() {
        let mut t = [512i64; 16];
        for v in t.iter_mut().skip(1) {
            *v = 0;
        }
        idct_permute(&mut t, 4);
        inverse_dct(&mut t, 4);
        for w in t.windows(2) {
            assert_eq!(w[0], w[1]);
        }
        assert_ne!(t[0], 0);
    }

    #[test]
    fn dct32_dc_only_flat() {
        let mut t = [0i64; 32];
        t[0] = 1024;
        idct_permute(&mut t, 5);
        inverse_dct(&mut t, 5);
        for w in t.windows(2) {
            assert_eq!(w[0], w[1]);
        }
        assert_ne!(t[0], 0);
    }

    #[test]
    fn adst4_zero_in_zero_out() {
        // §8.7.1.6 over an all-zero input must stay zero.
        let mut t = [0i64; 4];
        inverse_adst(&mut t, 2);
        assert_eq!(t, [0, 0, 0, 0]);
    }

    #[test]
    fn adst8_and_adst16_zero_in_zero_out() {
        let mut t8 = [0i64; 8];
        inverse_adst(&mut t8, 3);
        assert!(t8.iter().all(|&v| v == 0));
        let mut t16 = [0i64; 16];
        inverse_adst(&mut t16, 4);
        assert!(t16.iter().all(|&v| v == 0));
    }

    #[test]
    fn adst4_single_impulse_is_nonzero() {
        // A single non-zero ADST4 input must produce a non-trivial output.
        let mut t = [100i64, 0, 0, 0];
        inverse_adst4(&mut t);
        assert!(t.iter().any(|&v| v != 0));
    }

    /// Independent closed-form reference for the §8.7.1.6 ADST4: probing
    /// the integer [`inverse_adst4`] with a single non-zero input
    /// coefficient `T[ k ] = c` reproduces the inverse-DST-VII basis row
    ///
    /// ```text
    ///   x[ n ] = A * c * sin( (n+1)(2k+1) pi / 9 )
    /// ```
    ///
    /// where `9 == 2N+1` for `N == 4` and the amplitude `A` is the
    /// constant the four `SINPI_*_9` scalings collapse to (recovered
    /// below from the largest-magnitude basis entry so the test shares no
    /// constant with the implementation). By linearity an arbitrary input
    /// is the superposition of these rows; the random cross-check uses
    /// that. Derived from the transform definition, not from the
    /// butterfly listing — an independent oracle.
    #[test]
    fn inverse_adst4_matches_dst_vii_closed_form_random() {
        let basis = |n: usize, k: usize| {
            ((n + 1) as f64 * (2 * k + 1) as f64 * std::f64::consts::PI / 9.0).sin()
        };
        // Recover the amplitude from a single impulse at k=0, n=3 (the
        // largest |basis| entry) so no SINPI constant is hard-coded here.
        let amp = {
            let mut t = [4096i64, 0, 0, 0];
            inverse_adst4(&mut t);
            t[3] as f64 / (4096.0 * basis(3, 0))
        };

        let mut state: u64 = 0x4F6C_2A11_9B3D_77E5;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as i64
        };
        for _ in 0..400 {
            let mut t = [0i64; 4];
            let mut r_in = [0.0f64; 4];
            for j in 0..4 {
                let c = (next() % 4097) - 2048;
                t[j] = c;
                r_in[j] = c as f64;
            }
            inverse_adst4(&mut t);
            for (n, &tn) in t.iter().enumerate() {
                let r: f64 = (0..4).map(|k| amp * r_in[k] * basis(n, k)).sum();
                let d = (tn as f64 - r).abs();
                assert!(d <= 3.0, "n={n}: int={tn} ref={r:.2} d={d:.2}");
            }
        }
    }

    /// The inverse ADST is linear: `IADST(a) + IADST(b)` equals
    /// `IADST(a + b)` up to the per-stage `Round2` rounding (bounded by
    /// the stage count). Pinned directly on the integer network for all
    /// three sizes — no floating reference, so it certifies ADST8 / ADST16
    /// (whose size-specific butterfly structure has no single clean
    /// closed form) too.
    #[test]
    fn inverse_adst_is_linear() {
        for n in 2..=4u32 {
            let n0 = 1usize << n;
            let mut a: Vec<i64> = (0..n0)
                .map(|j| ((j * 41 + 13) % 811) as i64 - 405)
                .collect();
            let mut b: Vec<i64> = (0..n0).map(|j| ((j * 59 + 5) % 611) as i64 - 305).collect();
            let mut sum: Vec<i64> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
            for v in [&mut a, &mut b, &mut sum] {
                inverse_adst(v, n);
            }
            // ADST8/16 use a higher-precision S array with two rounding
            // layers; bound the per-element drift by 2*n.
            for i in 0..n0 {
                let lhs = a[i] + b[i];
                let d = (lhs - sum[i]).abs();
                assert!(
                    d <= 2 * n as i64,
                    "N={n0} i={i}: a+b={lhs} sum={} d={d}",
                    sum[i]
                );
            }
        }
    }

    #[test]
    fn adst_output_permute_n3_is_involution_on_identity() {
        // §8.7.1.5 n=3 permutation maps copyT[ 4*(c^b)+2*(b^a)+a ].
        // Applying it to the identity index vector gives the permutation.
        let mut t: Vec<i64> = (0..8).collect();
        adst_output_permute(&mut t, 3);
        // dst index built from a,b,c; verify a couple of entries.
        // a=0,b=0,c=0 -> dst 0, src 0.
        assert_eq!(t[0], 0);
        // a=1,b=0,c=0 -> dst 4, src 4*(0)+2*(0^1)+1=3.
        assert_eq!(t[4], 3);
    }

    #[test]
    fn transform_2d_zero_in_zero_out_all_types() {
        for &tt in &[DCT_DCT, ADST_DCT, DCT_ADST, ADST_ADST] {
            for &n in &[2u32, 3, 4] {
                let n0 = 1usize << n;
                let mut block = vec![0i64; n0 * n0];
                inverse_transform_2d(&mut block, n, tt, false);
                assert!(block.iter().all(|&v| v == 0), "tt={tt} n={n}");
            }
        }
        // 32x32 only valid for DCT_DCT in VP9, but exercise the path.
        let mut big = vec![0i64; 32 * 32];
        inverse_transform_2d(&mut big, 5, DCT_DCT, false);
        assert!(big.iter().all(|&v| v == 0));
    }

    #[test]
    fn transform_2d_dc_only_is_flat_block() {
        // A DC-only DCT_DCT block yields a flat (constant) residual.
        let mut block = vec![0i64; 16];
        block[0] = 800;
        inverse_transform_2d(&mut block, 2, DCT_DCT, false);
        let first = block[0];
        assert!(block.iter().all(|&v| v == first));
        assert_ne!(first, 0);
    }

    #[test]
    fn round2_matches_spec_formula() {
        // Round2( x, n ) = ( x + (1 << (n-1)) ) >> n.
        assert_eq!(round2(0, 1), 0);
        assert_eq!(round2(1, 1), 1); // (1+1)>>1 = 1
        assert_eq!(round2(3, 2), 1); // (3+2)>>2 = 1
        assert_eq!(round2(-4, 1), -2); // (-4+1)>>1 = -2 (arithmetic)
    }

    #[test]
    fn butterfly_fast_path_matches_general_path() {
        // angle == 16 (== 16 + 32*0) must take the fast path and match
        // the equivalent general two-multiply formulation numerically.
        let mut t_fast = [1000i64, 200];
        butterfly(&mut t_fast, 0, 1, 16, false);
        // General-path reference using the cos/sin formula.
        let a = 1000i64;
        let b = 200i64;
        let x = a * cos64(16) - b * sin64(16);
        let y = a * sin64(16) + b * cos64(16);
        let exp = [round2(x, 14), round2(y, 14)];
        assert_eq!(t_fast, exp);
    }

    /// A from-spec closed-form reference for the VP9 inverse DCT, derived
    /// independently of the butterfly network from the transform
    /// definition the §8.7.1.1 `cos64` note encodes
    /// (`cos64( angle ) == round( 16384 * cos( angle * pi / 64 ) )`).
    ///
    /// Probing the integer [`inverse_dct`] with a single non-zero input
    /// coefficient `T[ k ] = c` (after the §8.7.1.2 bit-reversal permute)
    /// reproduces the closed form
    ///
    /// ```text
    ///   x[ n ] = ( k == 0 ? 1/sqrt(2) : 1 ) * c * cos( (2n+1) k pi / (2N) )
    /// ```
    ///
    /// i.e. the orthonormal inverse-DCT-III basis with the DC term scaled
    /// by `1/sqrt(2)` and *no* `2/N` normalization (the butterfly network
    /// folds that scaling into its `Round2( ., 14 )` rotations — recovered
    /// empirically as the 0.70703125 DC ratio). By linearity the response
    /// to an arbitrary input is the superposition of the per-coefficient
    /// basis rows. This shares no code with the network it validates.
    fn idct_reference(input: &[f64]) -> Vec<f64> {
        let n0 = input.len();
        let nf = n0 as f64;
        (0..n0)
            .map(|n| {
                (0..n0)
                    .map(|k| {
                        let scale = if k == 0 {
                            std::f64::consts::FRAC_1_SQRT_2
                        } else {
                            1.0
                        };
                        scale
                            * input[k]
                            * ((2 * n + 1) as f64 * k as f64 * std::f64::consts::PI / (2.0 * nf))
                                .cos()
                    })
                    .sum()
            })
            .collect()
    }

    /// The closed-form reference reproduces the single-impulse integer
    /// response within the accumulated fixed-point error of the butterfly
    /// network for every size and every basis index. This pins the oracle
    /// itself before it validates arbitrary vectors below.
    #[test]
    fn idct_reference_matches_single_impulse_response() {
        for n in 2..=5u32 {
            let n0 = 1usize << n;
            for k in 0..n0 {
                let mut t = vec![0i64; n0];
                t[k] = 4096;
                idct_permute(&mut t, n);
                inverse_dct(&mut t, n);
                let mut r_in = vec![0.0f64; n0];
                r_in[k] = 4096.0;
                let r = idct_reference(&r_in);
                for (i, (&ti, &ri)) in t.iter().zip(&r).enumerate() {
                    let d = (ti as f64 - ri).abs();
                    assert!(d <= 2.0, "N={n0} k={k} i={i}: int={ti} ref={ri} d={d}");
                }
            }
        }
    }

    /// Cross-validate the full integer inverse DCT against the
    /// independent closed form over many deterministic pseudo-random
    /// coefficient vectors, for every transform size. The error bound
    /// scales with the number of butterfly stages (`log2(N)`, each adding
    /// at most ~1 from its `Round2`) plus a small constant for the 14-bit
    /// `cos64` quantization accumulated across the basis sum.
    #[test]
    fn inverse_dct_matches_closed_form_reference_random() {
        let mut state: u64 = 0xD1B5_4A32_D192_ED03;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as i64
        };
        for n in 2..=5u32 {
            let n0 = 1usize << n;
            // |coef| <= 2048 keeps the conformance 8+BitDepth range; a
            // transposed butterfly or dropped Round2 diverges by hundreds,
            // far past this `log2(N)+3` bound.
            let bound = (n + 3) as f64;
            for _ in 0..400 {
                let mut t = vec![0i64; n0];
                let mut r_in = vec![0.0f64; n0];
                for j in 0..n0 {
                    let c = (next() % 4097) - 2048;
                    t[j] = c;
                    r_in[j] = c as f64;
                }
                idct_permute(&mut t, n);
                inverse_dct(&mut t, n);
                let r = idct_reference(&r_in);
                for (i, (&ti, &ri)) in t.iter().zip(&r).enumerate() {
                    let d = (ti as f64 - ri).abs();
                    assert!(
                        d <= bound,
                        "N={n0} i={i}: int={ti} ref={ri:.2} d={d:.2} > bound {bound}"
                    );
                }
            }
        }
    }

    /// The inverse DCT is a linear operator: the (rounded) response to a
    /// summed input equals the summed responses, up to the per-stage
    /// rounding (`log2(N)` stages). Pinned directly on the integer network
    /// with no floating reference.
    #[test]
    fn inverse_dct_is_linear() {
        for n in 2..=5u32 {
            let n0 = 1usize << n;
            let mut a: Vec<i64> = (0..n0)
                .map(|j| ((j * 37 + 11) % 901) as i64 - 450)
                .collect();
            let mut b: Vec<i64> = (0..n0).map(|j| ((j * 53 + 7) % 701) as i64 - 350).collect();
            let mut sum: Vec<i64> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
            for v in [&mut a, &mut b, &mut sum] {
                idct_permute(v, n);
                inverse_dct(v, n);
            }
            for i in 0..n0 {
                let lhs = a[i] + b[i];
                let d = (lhs - sum[i]).abs();
                assert!(
                    d <= n as i64,
                    "N={n0} i={i}: a+b={lhs} sum={} d={d}",
                    sum[i]
                );
            }
        }
    }

    /// The §8.7.1.10 inverse WHT realises the orthogonal matrix
    ///
    /// ```text
    ///   M = 1/2 * [[ 1,  1,  1,  1],
    ///              [ 1,  1, -1, -1],
    ///              [ 1, -1, -1,  1],
    ///              [ 1, -1,  1, -1]]
    /// ```
    ///
    /// recovered here by probing each unit impulse (input pre-scaled so
    /// the `>> shift` is exact). The matrix is its own inverse up to the
    /// `2x` integer scaling the transform carries, so this independently
    /// certifies the butterfly sequence without re-deriving it.
    #[test]
    fn inverse_wht_realises_the_orthogonal_matrix() {
        // Rows of 2*M (so the entries are the integer ±1 pattern).
        let expected_2m: [[i64; 4]; 4] =
            [[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, -1, 1], [1, -1, 1, -1]];
        for shift in [0u32, 2] {
            for (k, row) in expected_2m.iter().enumerate() {
                let mut t = [0i64; 4];
                // input = 2 << shift at position k -> output entries are
                // (2 << shift) * M[n][k] = (row entry) since 2*M is ±1 and
                // the >> shift cancels the << shift exactly.
                t[k] = 2 << shift;
                inverse_wht(&mut t, shift);
                for (n, exp) in row.iter().enumerate() {
                    assert_eq!(t[n], *exp, "shift={shift} e{k} out[{n}]");
                }
            }
        }
    }

    /// The §8.7.1.10 inverse WHT is a perfect involution at `shift == 0`:
    /// the interior `e = (a - d) >> 1` exactly halves the matrix's `2x`
    /// scaling, so applying the transform twice returns the input
    /// bit-exactly (no rounding loss, as befits the lossless path). This
    /// round-trips the whole butterfly sequence with no floating
    /// reference and certifies it is bijective over the integers.
    #[test]
    fn inverse_wht_applied_twice_is_identity() {
        let mut state: u64 = 0x91E1_0DA5_DF55_1015;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as i64
        };
        for _ in 0..500 {
            let input: [i64; 4] = [
                (next() % 2001) - 1000,
                (next() % 2001) - 1000,
                (next() % 2001) - 1000,
                (next() % 2001) - 1000,
            ];
            let mut t = input;
            inverse_wht(&mut t, 0);
            inverse_wht(&mut t, 0);
            assert_eq!(t, input, "WHT is not an involution at shift 0");
        }
    }
}
