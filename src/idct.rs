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
    let copy_t: Vec<i64> = t[..len].to_vec();
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
    let copy_t: Vec<i64> = t[..n0].to_vec();
    for i in 0..n1 {
        t[2 * i] = copy_t[n0 - 1 - 2 * i];
        t[2 * i + 1] = copy_t[2 * i];
    }
}

/// §8.7.1.5 inverse ADST output array permutation over `t` of length
/// `1<<n` for `n` in `{3, 4}`.
fn adst_output_permute(t: &mut [i64], n: u32) {
    let n0 = 1usize << n;
    let copy_t: Vec<i64> = t[..n0].to_vec();
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
fn inverse_adst(t: &mut [i64], n: u32) {
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

#[cfg(test)]
mod tests {
    use super::*;

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
