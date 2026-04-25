//! VP9 inverse transforms — §8.7.1 full port.
//!
//! Covers:
//! * Sizes 4×4 / 8×8 / 16×16 / 32×32.
//! * Types `DCT_DCT`, `ADST_DCT`, `DCT_ADST`, `ADST_ADST`.
//! * A 4×4 Walsh-Hadamard transform for the lossless mode (§8.7.1.1).
//!
//! Arithmetic matches libvpx `vpx_dsp/inv_txfm.c` bit-for-bit. The cosine
//! constants are 14-bit fixed-point (`cospi_k_64` = `round(cos(k·π/64) ·
//! 2^14)`); the post-row/column shift of 4/5/6 matches libvpx
//! `ROUND_POWER_OF_TWO`.

use oxideav_core::{Error, Result};

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TxType {
    DctDct = 0,
    AdstDct = 1,
    DctAdst = 2,
    AdstAdst = 3,
    WhtWht = 4,
}

impl TxType {
    pub fn from_u32(v: u32) -> Result<Self> {
        Ok(match v {
            0 => Self::DctDct,
            1 => Self::AdstDct,
            2 => Self::DctAdst,
            3 => Self::AdstAdst,
            4 => Self::WhtWht,
            _ => return Err(Error::invalid(format!("vp9 tx: invalid tx_type {v}"))),
        })
    }
}

// 14-bit fixed-point cosine constants — libvpx `vpx_dsp/txfm_common.h`.
const DCT_CONST_BITS: i32 = 14;
const DCT_CONST_ROUNDING: i32 = 1 << (DCT_CONST_BITS - 1);

const COSPI_1_64: i32 = 16364;
const COSPI_2_64: i32 = 16305;
const COSPI_3_64: i32 = 16207;
const COSPI_4_64: i32 = 16069;
const COSPI_5_64: i32 = 15893;
const COSPI_6_64: i32 = 15679;
const COSPI_7_64: i32 = 15426;
const COSPI_8_64: i32 = 15137;
const COSPI_9_64: i32 = 14811;
const COSPI_10_64: i32 = 14449;
const COSPI_11_64: i32 = 14053;
const COSPI_12_64: i32 = 13623;
const COSPI_13_64: i32 = 13160;
const COSPI_14_64: i32 = 12665;
const COSPI_15_64: i32 = 12140;
const COSPI_16_64: i32 = 11585;
const COSPI_17_64: i32 = 11003;
const COSPI_18_64: i32 = 10394;
const COSPI_19_64: i32 = 9760;
const COSPI_20_64: i32 = 9102;
const COSPI_21_64: i32 = 8423;
const COSPI_22_64: i32 = 7723;
const COSPI_23_64: i32 = 7005;
const COSPI_24_64: i32 = 6270;
const COSPI_25_64: i32 = 5520;
const COSPI_26_64: i32 = 4756;
const COSPI_27_64: i32 = 3981;
const COSPI_28_64: i32 = 3196;
const COSPI_29_64: i32 = 2404;
const COSPI_30_64: i32 = 1606;
const COSPI_31_64: i32 = 804;

const SINPI_1_9: i32 = 5283;
const SINPI_2_9: i32 = 9929;
const SINPI_3_9: i32 = 13377;
const SINPI_4_9: i32 = 15212;

#[inline]
fn dct_const_round_shift(x: i32) -> i32 {
    (x + DCT_CONST_ROUNDING) >> DCT_CONST_BITS
}

#[inline]
fn round_power_of_two(v: i32, n: i32) -> i32 {
    (v + (1 << (n - 1))) >> n
}

#[inline]
fn clip_pixel_add(dst: u8, delta: i32) -> u8 {
    ((dst as i32) + delta).clamp(0, 255) as u8
}

// WRAPLOW in libvpx is a signed saturation to 16-bit for tran_low_t; with
// 8-bit decode the intermediate values fit in i32 comfortably. libvpx's
// WRAPLOW macro just casts through int16_t. We mimic that exactly to keep
// behaviour bit-identical.
#[inline]
fn wraplow(v: i32) -> i32 {
    v as i16 as i32
}

// ===== iDCT / iADST 1-D kernels =====

fn idct4(x: &[i32; 4]) -> [i32; 4] {
    let (i0, i1, i2, i3) = (x[0], x[1], x[2], x[3]);
    let t1 = (i0 + i2) * COSPI_16_64;
    let t2 = (i0 - i2) * COSPI_16_64;
    let s0 = wraplow(dct_const_round_shift(t1));
    let s1 = wraplow(dct_const_round_shift(t2));
    let t1 = i1 * COSPI_24_64 - i3 * COSPI_8_64;
    let t2 = i1 * COSPI_8_64 + i3 * COSPI_24_64;
    let s2 = wraplow(dct_const_round_shift(t1));
    let s3 = wraplow(dct_const_round_shift(t2));
    [
        wraplow(s0 + s3),
        wraplow(s1 + s2),
        wraplow(s1 - s2),
        wraplow(s0 - s3),
    ]
}

fn iadst4(x: &[i32; 4]) -> [i32; 4] {
    let (x0, x1, x2, x3) = (x[0], x[1], x[2], x[3]);
    if (x0 | x1 | x2 | x3) == 0 {
        return [0; 4];
    }
    let s0 = SINPI_1_9 * x0;
    let s1 = SINPI_2_9 * x0;
    let s2 = SINPI_3_9 * x1;
    let s3 = SINPI_4_9 * x2;
    let s4 = SINPI_1_9 * x2;
    let s5 = SINPI_2_9 * x3;
    let s6 = SINPI_4_9 * x3;
    let s7 = wraplow(x0 - x2 + x3);

    let a0 = s0 + s3 + s5;
    let a1 = s1 - s4 - s6;
    let a3 = s2;
    let a2 = SINPI_3_9 * s7;

    [
        wraplow(dct_const_round_shift(a0 + a3)),
        wraplow(dct_const_round_shift(a1 + a3)),
        wraplow(dct_const_round_shift(a2)),
        wraplow(dct_const_round_shift(a0 + a1 - a3)),
    ]
}

fn idct8(input: &[i32; 8]) -> [i32; 8] {
    // Port of libvpx idct8_c.
    let mut step1 = [0i32; 8];
    let mut step2 = [0i32; 8];
    step1[0] = input[0];
    step1[2] = input[4];
    step1[1] = input[2];
    step1[3] = input[6];
    let t1 = input[1] * COSPI_28_64 - input[7] * COSPI_4_64;
    let t2 = input[1] * COSPI_4_64 + input[7] * COSPI_28_64;
    step1[4] = wraplow(dct_const_round_shift(t1));
    step1[7] = wraplow(dct_const_round_shift(t2));
    let t1 = input[5] * COSPI_12_64 - input[3] * COSPI_20_64;
    let t2 = input[5] * COSPI_20_64 + input[3] * COSPI_12_64;
    step1[5] = wraplow(dct_const_round_shift(t1));
    step1[6] = wraplow(dct_const_round_shift(t2));

    let t1 = (step1[0] + step1[2]) * COSPI_16_64;
    let t2 = (step1[0] - step1[2]) * COSPI_16_64;
    step2[0] = wraplow(dct_const_round_shift(t1));
    step2[1] = wraplow(dct_const_round_shift(t2));
    let t1 = step1[1] * COSPI_24_64 - step1[3] * COSPI_8_64;
    let t2 = step1[1] * COSPI_8_64 + step1[3] * COSPI_24_64;
    step2[2] = wraplow(dct_const_round_shift(t1));
    step2[3] = wraplow(dct_const_round_shift(t2));
    step2[4] = wraplow(step1[4] + step1[5]);
    step2[5] = wraplow(step1[4] - step1[5]);
    step2[6] = wraplow(-step1[6] + step1[7]);
    step2[7] = wraplow(step1[6] + step1[7]);

    step1[0] = wraplow(step2[0] + step2[3]);
    step1[1] = wraplow(step2[1] + step2[2]);
    step1[2] = wraplow(step2[1] - step2[2]);
    step1[3] = wraplow(step2[0] - step2[3]);
    step1[4] = step2[4];
    let t1 = (step2[6] - step2[5]) * COSPI_16_64;
    let t2 = (step2[5] + step2[6]) * COSPI_16_64;
    step1[5] = wraplow(dct_const_round_shift(t1));
    step1[6] = wraplow(dct_const_round_shift(t2));
    step1[7] = step2[7];

    [
        wraplow(step1[0] + step1[7]),
        wraplow(step1[1] + step1[6]),
        wraplow(step1[2] + step1[5]),
        wraplow(step1[3] + step1[4]),
        wraplow(step1[3] - step1[4]),
        wraplow(step1[2] - step1[5]),
        wraplow(step1[1] - step1[6]),
        wraplow(step1[0] - step1[7]),
    ]
}

fn iadst8(input: &[i32; 8]) -> [i32; 8] {
    let mut x0 = input[7];
    let mut x1 = input[0];
    let mut x2 = input[5];
    let mut x3 = input[2];
    let mut x4 = input[3];
    let mut x5 = input[4];
    let mut x6 = input[1];
    let mut x7 = input[6];
    if (x0 | x1 | x2 | x3 | x4 | x5 | x6 | x7) == 0 {
        return [0; 8];
    }
    // stage 1
    let s0 = COSPI_2_64 * x0 + COSPI_30_64 * x1;
    let s1 = COSPI_30_64 * x0 - COSPI_2_64 * x1;
    let s2 = COSPI_10_64 * x2 + COSPI_22_64 * x3;
    let s3 = COSPI_22_64 * x2 - COSPI_10_64 * x3;
    let s4 = COSPI_18_64 * x4 + COSPI_14_64 * x5;
    let s5 = COSPI_14_64 * x4 - COSPI_18_64 * x5;
    let s6 = COSPI_26_64 * x6 + COSPI_6_64 * x7;
    let s7 = COSPI_6_64 * x6 - COSPI_26_64 * x7;

    x0 = wraplow(dct_const_round_shift(s0 + s4));
    x1 = wraplow(dct_const_round_shift(s1 + s5));
    x2 = wraplow(dct_const_round_shift(s2 + s6));
    x3 = wraplow(dct_const_round_shift(s3 + s7));
    x4 = wraplow(dct_const_round_shift(s0 - s4));
    x5 = wraplow(dct_const_round_shift(s1 - s5));
    x6 = wraplow(dct_const_round_shift(s2 - s6));
    x7 = wraplow(dct_const_round_shift(s3 - s7));

    // stage 2
    let s0 = x0;
    let s1 = x1;
    let s2 = x2;
    let s3 = x3;
    let s4 = COSPI_8_64 * x4 + COSPI_24_64 * x5;
    let s5 = COSPI_24_64 * x4 - COSPI_8_64 * x5;
    let s6 = -COSPI_24_64 * x6 + COSPI_8_64 * x7;
    let s7 = COSPI_8_64 * x6 + COSPI_24_64 * x7;

    x0 = wraplow(s0 + s2);
    x1 = wraplow(s1 + s3);
    x2 = wraplow(s0 - s2);
    x3 = wraplow(s1 - s3);
    x4 = wraplow(dct_const_round_shift(s4 + s6));
    x5 = wraplow(dct_const_round_shift(s5 + s7));
    x6 = wraplow(dct_const_round_shift(s4 - s6));
    x7 = wraplow(dct_const_round_shift(s5 - s7));

    // stage 3
    let s2 = COSPI_16_64 * (x2 + x3);
    let s3 = COSPI_16_64 * (x2 - x3);
    let s6 = COSPI_16_64 * (x6 + x7);
    let s7 = COSPI_16_64 * (x6 - x7);

    x2 = wraplow(dct_const_round_shift(s2));
    x3 = wraplow(dct_const_round_shift(s3));
    x6 = wraplow(dct_const_round_shift(s6));
    x7 = wraplow(dct_const_round_shift(s7));

    [
        wraplow(x0),
        wraplow(-x4),
        wraplow(x6),
        wraplow(-x2),
        wraplow(x3),
        wraplow(-x7),
        wraplow(x5),
        wraplow(-x1),
    ]
}

fn idct16(input: &[i32; 16]) -> [i32; 16] {
    let mut step1 = [0i32; 16];
    let mut step2 = [0i32; 16];

    step1[0] = input[0];
    step1[1] = input[8];
    step1[2] = input[4];
    step1[3] = input[12];
    step1[4] = input[2];
    step1[5] = input[10];
    step1[6] = input[6];
    step1[7] = input[14];
    step1[8] = input[1];
    step1[9] = input[9];
    step1[10] = input[5];
    step1[11] = input[13];
    step1[12] = input[3];
    step1[13] = input[11];
    step1[14] = input[7];
    step1[15] = input[15];

    // stage 2
    step2[..8].copy_from_slice(&step1[..8]);
    let t1 = step1[8] * COSPI_30_64 - step1[15] * COSPI_2_64;
    let t2 = step1[8] * COSPI_2_64 + step1[15] * COSPI_30_64;
    step2[8] = wraplow(dct_const_round_shift(t1));
    step2[15] = wraplow(dct_const_round_shift(t2));

    let t1 = step1[9] * COSPI_14_64 - step1[14] * COSPI_18_64;
    let t2 = step1[9] * COSPI_18_64 + step1[14] * COSPI_14_64;
    step2[9] = wraplow(dct_const_round_shift(t1));
    step2[14] = wraplow(dct_const_round_shift(t2));

    let t1 = step1[10] * COSPI_22_64 - step1[13] * COSPI_10_64;
    let t2 = step1[10] * COSPI_10_64 + step1[13] * COSPI_22_64;
    step2[10] = wraplow(dct_const_round_shift(t1));
    step2[13] = wraplow(dct_const_round_shift(t2));

    let t1 = step1[11] * COSPI_6_64 - step1[12] * COSPI_26_64;
    let t2 = step1[11] * COSPI_26_64 + step1[12] * COSPI_6_64;
    step2[11] = wraplow(dct_const_round_shift(t1));
    step2[12] = wraplow(dct_const_round_shift(t2));

    // stage 3
    step1[0] = step2[0];
    step1[1] = step2[1];
    step1[2] = step2[2];
    step1[3] = step2[3];

    let t1 = step2[4] * COSPI_28_64 - step2[7] * COSPI_4_64;
    let t2 = step2[4] * COSPI_4_64 + step2[7] * COSPI_28_64;
    step1[4] = wraplow(dct_const_round_shift(t1));
    step1[7] = wraplow(dct_const_round_shift(t2));
    let t1 = step2[5] * COSPI_12_64 - step2[6] * COSPI_20_64;
    let t2 = step2[5] * COSPI_20_64 + step2[6] * COSPI_12_64;
    step1[5] = wraplow(dct_const_round_shift(t1));
    step1[6] = wraplow(dct_const_round_shift(t2));

    step1[8] = wraplow(step2[8] + step2[9]);
    step1[9] = wraplow(step2[8] - step2[9]);
    step1[10] = wraplow(-step2[10] + step2[11]);
    step1[11] = wraplow(step2[10] + step2[11]);
    step1[12] = wraplow(step2[12] + step2[13]);
    step1[13] = wraplow(step2[12] - step2[13]);
    step1[14] = wraplow(-step2[14] + step2[15]);
    step1[15] = wraplow(step2[14] + step2[15]);

    // stage 4
    let t1 = (step1[0] + step1[1]) * COSPI_16_64;
    let t2 = (step1[0] - step1[1]) * COSPI_16_64;
    step2[0] = wraplow(dct_const_round_shift(t1));
    step2[1] = wraplow(dct_const_round_shift(t2));
    let t1 = step1[2] * COSPI_24_64 - step1[3] * COSPI_8_64;
    let t2 = step1[2] * COSPI_8_64 + step1[3] * COSPI_24_64;
    step2[2] = wraplow(dct_const_round_shift(t1));
    step2[3] = wraplow(dct_const_round_shift(t2));
    step2[4] = wraplow(step1[4] + step1[5]);
    step2[5] = wraplow(step1[4] - step1[5]);
    step2[6] = wraplow(-step1[6] + step1[7]);
    step2[7] = wraplow(step1[6] + step1[7]);

    step2[8] = step1[8];
    step2[15] = step1[15];
    let t1 = -step1[9] * COSPI_8_64 + step1[14] * COSPI_24_64;
    let t2 = step1[9] * COSPI_24_64 + step1[14] * COSPI_8_64;
    step2[9] = wraplow(dct_const_round_shift(t1));
    step2[14] = wraplow(dct_const_round_shift(t2));
    let t1 = -step1[10] * COSPI_24_64 - step1[13] * COSPI_8_64;
    let t2 = -step1[10] * COSPI_8_64 + step1[13] * COSPI_24_64;
    step2[10] = wraplow(dct_const_round_shift(t1));
    step2[13] = wraplow(dct_const_round_shift(t2));
    step2[11] = step1[11];
    step2[12] = step1[12];

    // stage 5
    step1[0] = wraplow(step2[0] + step2[3]);
    step1[1] = wraplow(step2[1] + step2[2]);
    step1[2] = wraplow(step2[1] - step2[2]);
    step1[3] = wraplow(step2[0] - step2[3]);
    step1[4] = step2[4];
    let t1 = (step2[6] - step2[5]) * COSPI_16_64;
    let t2 = (step2[5] + step2[6]) * COSPI_16_64;
    step1[5] = wraplow(dct_const_round_shift(t1));
    step1[6] = wraplow(dct_const_round_shift(t2));
    step1[7] = step2[7];

    step1[8] = wraplow(step2[8] + step2[11]);
    step1[9] = wraplow(step2[9] + step2[10]);
    step1[10] = wraplow(step2[9] - step2[10]);
    step1[11] = wraplow(step2[8] - step2[11]);
    step1[12] = wraplow(-step2[12] + step2[15]);
    step1[13] = wraplow(-step2[13] + step2[14]);
    step1[14] = wraplow(step2[13] + step2[14]);
    step1[15] = wraplow(step2[12] + step2[15]);

    // stage 6
    step2[0] = wraplow(step1[0] + step1[7]);
    step2[1] = wraplow(step1[1] + step1[6]);
    step2[2] = wraplow(step1[2] + step1[5]);
    step2[3] = wraplow(step1[3] + step1[4]);
    step2[4] = wraplow(step1[3] - step1[4]);
    step2[5] = wraplow(step1[2] - step1[5]);
    step2[6] = wraplow(step1[1] - step1[6]);
    step2[7] = wraplow(step1[0] - step1[7]);
    step2[8] = step1[8];
    step2[9] = step1[9];
    let t1 = (-step1[10] + step1[13]) * COSPI_16_64;
    let t2 = (step1[10] + step1[13]) * COSPI_16_64;
    step2[10] = wraplow(dct_const_round_shift(t1));
    step2[13] = wraplow(dct_const_round_shift(t2));
    let t1 = (-step1[11] + step1[12]) * COSPI_16_64;
    let t2 = (step1[11] + step1[12]) * COSPI_16_64;
    step2[11] = wraplow(dct_const_round_shift(t1));
    step2[12] = wraplow(dct_const_round_shift(t2));
    step2[14] = step1[14];
    step2[15] = step1[15];

    // stage 7
    [
        wraplow(step2[0] + step2[15]),
        wraplow(step2[1] + step2[14]),
        wraplow(step2[2] + step2[13]),
        wraplow(step2[3] + step2[12]),
        wraplow(step2[4] + step2[11]),
        wraplow(step2[5] + step2[10]),
        wraplow(step2[6] + step2[9]),
        wraplow(step2[7] + step2[8]),
        wraplow(step2[7] - step2[8]),
        wraplow(step2[6] - step2[9]),
        wraplow(step2[5] - step2[10]),
        wraplow(step2[4] - step2[11]),
        wraplow(step2[3] - step2[12]),
        wraplow(step2[2] - step2[13]),
        wraplow(step2[1] - step2[14]),
        wraplow(step2[0] - step2[15]),
    ]
}

#[allow(clippy::too_many_lines)]
fn iadst16(input: &[i32; 16]) -> [i32; 16] {
    let mut x0 = input[15];
    let mut x1 = input[0];
    let mut x2 = input[13];
    let mut x3 = input[2];
    let mut x4 = input[11];
    let mut x5 = input[4];
    let mut x6 = input[9];
    let mut x7 = input[6];
    let mut x8 = input[7];
    let mut x9 = input[8];
    let mut x10 = input[5];
    let mut x11 = input[10];
    let mut x12 = input[3];
    let mut x13 = input[12];
    let mut x14 = input[1];
    let mut x15 = input[14];

    if (x0 | x1 | x2 | x3 | x4 | x5 | x6 | x7 | x8 | x9 | x10 | x11 | x12 | x13 | x14 | x15) == 0 {
        return [0; 16];
    }

    // stage 1
    let s0 = x0 * COSPI_1_64 + x1 * COSPI_31_64;
    let s1 = x0 * COSPI_31_64 - x1 * COSPI_1_64;
    let s2 = x2 * COSPI_5_64 + x3 * COSPI_27_64;
    let s3 = x2 * COSPI_27_64 - x3 * COSPI_5_64;
    let s4 = x4 * COSPI_9_64 + x5 * COSPI_23_64;
    let s5 = x4 * COSPI_23_64 - x5 * COSPI_9_64;
    let s6 = x6 * COSPI_13_64 + x7 * COSPI_19_64;
    let s7 = x6 * COSPI_19_64 - x7 * COSPI_13_64;
    let s8 = x8 * COSPI_17_64 + x9 * COSPI_15_64;
    let s9 = x8 * COSPI_15_64 - x9 * COSPI_17_64;
    let s10 = x10 * COSPI_21_64 + x11 * COSPI_11_64;
    let s11 = x10 * COSPI_11_64 - x11 * COSPI_21_64;
    let s12 = x12 * COSPI_25_64 + x13 * COSPI_7_64;
    let s13 = x12 * COSPI_7_64 - x13 * COSPI_25_64;
    let s14 = x14 * COSPI_29_64 + x15 * COSPI_3_64;
    let s15 = x14 * COSPI_3_64 - x15 * COSPI_29_64;

    x0 = wraplow(dct_const_round_shift(s0 + s8));
    x1 = wraplow(dct_const_round_shift(s1 + s9));
    x2 = wraplow(dct_const_round_shift(s2 + s10));
    x3 = wraplow(dct_const_round_shift(s3 + s11));
    x4 = wraplow(dct_const_round_shift(s4 + s12));
    x5 = wraplow(dct_const_round_shift(s5 + s13));
    x6 = wraplow(dct_const_round_shift(s6 + s14));
    x7 = wraplow(dct_const_round_shift(s7 + s15));
    x8 = wraplow(dct_const_round_shift(s0 - s8));
    x9 = wraplow(dct_const_round_shift(s1 - s9));
    x10 = wraplow(dct_const_round_shift(s2 - s10));
    x11 = wraplow(dct_const_round_shift(s3 - s11));
    x12 = wraplow(dct_const_round_shift(s4 - s12));
    x13 = wraplow(dct_const_round_shift(s5 - s13));
    x14 = wraplow(dct_const_round_shift(s6 - s14));
    x15 = wraplow(dct_const_round_shift(s7 - s15));

    // stage 2
    let s0 = x0;
    let s1 = x1;
    let s2 = x2;
    let s3 = x3;
    let s4 = x4;
    let s5 = x5;
    let s6 = x6;
    let s7 = x7;
    let s8 = x8 * COSPI_4_64 + x9 * COSPI_28_64;
    let s9 = x8 * COSPI_28_64 - x9 * COSPI_4_64;
    let s10 = x10 * COSPI_20_64 + x11 * COSPI_12_64;
    let s11 = x10 * COSPI_12_64 - x11 * COSPI_20_64;
    let s12 = -x12 * COSPI_28_64 + x13 * COSPI_4_64;
    let s13 = x12 * COSPI_4_64 + x13 * COSPI_28_64;
    let s14 = -x14 * COSPI_12_64 + x15 * COSPI_20_64;
    let s15 = x14 * COSPI_20_64 + x15 * COSPI_12_64;

    x0 = wraplow(s0 + s4);
    x1 = wraplow(s1 + s5);
    x2 = wraplow(s2 + s6);
    x3 = wraplow(s3 + s7);
    x4 = wraplow(s0 - s4);
    x5 = wraplow(s1 - s5);
    x6 = wraplow(s2 - s6);
    x7 = wraplow(s3 - s7);
    x8 = wraplow(dct_const_round_shift(s8 + s12));
    x9 = wraplow(dct_const_round_shift(s9 + s13));
    x10 = wraplow(dct_const_round_shift(s10 + s14));
    x11 = wraplow(dct_const_round_shift(s11 + s15));
    x12 = wraplow(dct_const_round_shift(s8 - s12));
    x13 = wraplow(dct_const_round_shift(s9 - s13));
    x14 = wraplow(dct_const_round_shift(s10 - s14));
    x15 = wraplow(dct_const_round_shift(s11 - s15));

    // stage 3
    let s0 = x0;
    let s1 = x1;
    let s2 = x2;
    let s3 = x3;
    let s4 = x4 * COSPI_8_64 + x5 * COSPI_24_64;
    let s5 = x4 * COSPI_24_64 - x5 * COSPI_8_64;
    let s6 = -x6 * COSPI_24_64 + x7 * COSPI_8_64;
    let s7 = x6 * COSPI_8_64 + x7 * COSPI_24_64;
    let s8 = x8;
    let s9 = x9;
    let s10 = x10;
    let s11 = x11;
    let s12 = x12 * COSPI_8_64 + x13 * COSPI_24_64;
    let s13 = x12 * COSPI_24_64 - x13 * COSPI_8_64;
    let s14 = -x14 * COSPI_24_64 + x15 * COSPI_8_64;
    let s15 = x14 * COSPI_8_64 + x15 * COSPI_24_64;

    x0 = wraplow(s0 + s2);
    x1 = wraplow(s1 + s3);
    x2 = wraplow(s0 - s2);
    x3 = wraplow(s1 - s3);
    x4 = wraplow(dct_const_round_shift(s4 + s6));
    x5 = wraplow(dct_const_round_shift(s5 + s7));
    x6 = wraplow(dct_const_round_shift(s4 - s6));
    x7 = wraplow(dct_const_round_shift(s5 - s7));
    x8 = wraplow(s8 + s10);
    x9 = wraplow(s9 + s11);
    x10 = wraplow(s8 - s10);
    x11 = wraplow(s9 - s11);
    x12 = wraplow(dct_const_round_shift(s12 + s14));
    x13 = wraplow(dct_const_round_shift(s13 + s15));
    x14 = wraplow(dct_const_round_shift(s12 - s14));
    x15 = wraplow(dct_const_round_shift(s13 - s15));

    // stage 4
    let s2 = (-COSPI_16_64) * (x2 + x3);
    let s3 = COSPI_16_64 * (x2 - x3);
    let s6 = COSPI_16_64 * (x6 + x7);
    let s7 = COSPI_16_64 * (-x6 + x7);
    let s10 = COSPI_16_64 * (x10 + x11);
    let s11 = COSPI_16_64 * (-x10 + x11);
    let s14 = (-COSPI_16_64) * (x14 + x15);
    let s15 = COSPI_16_64 * (x14 - x15);

    x2 = wraplow(dct_const_round_shift(s2));
    x3 = wraplow(dct_const_round_shift(s3));
    x6 = wraplow(dct_const_round_shift(s6));
    x7 = wraplow(dct_const_round_shift(s7));
    x10 = wraplow(dct_const_round_shift(s10));
    x11 = wraplow(dct_const_round_shift(s11));
    x14 = wraplow(dct_const_round_shift(s14));
    x15 = wraplow(dct_const_round_shift(s15));

    [
        wraplow(x0),
        wraplow(-x8),
        wraplow(x12),
        wraplow(-x4),
        wraplow(x6),
        wraplow(x14),
        wraplow(x10),
        wraplow(x2),
        wraplow(x3),
        wraplow(x11),
        wraplow(x15),
        wraplow(x7),
        wraplow(x5),
        wraplow(-x13),
        wraplow(x9),
        wraplow(-x1),
    ]
}

#[allow(clippy::too_many_lines)]
fn idct32(input: &[i32; 32]) -> [i32; 32] {
    let mut step1 = [0i32; 32];
    let mut step2 = [0i32; 32];

    // stage 1
    step1[0] = input[0];
    step1[1] = input[16];
    step1[2] = input[8];
    step1[3] = input[24];
    step1[4] = input[4];
    step1[5] = input[20];
    step1[6] = input[12];
    step1[7] = input[28];
    step1[8] = input[2];
    step1[9] = input[18];
    step1[10] = input[10];
    step1[11] = input[26];
    step1[12] = input[6];
    step1[13] = input[22];
    step1[14] = input[14];
    step1[15] = input[30];

    let t1 = input[1] * COSPI_31_64 - input[31] * COSPI_1_64;
    let t2 = input[1] * COSPI_1_64 + input[31] * COSPI_31_64;
    step1[16] = wraplow(dct_const_round_shift(t1));
    step1[31] = wraplow(dct_const_round_shift(t2));

    let t1 = input[17] * COSPI_15_64 - input[15] * COSPI_17_64;
    let t2 = input[17] * COSPI_17_64 + input[15] * COSPI_15_64;
    step1[17] = wraplow(dct_const_round_shift(t1));
    step1[30] = wraplow(dct_const_round_shift(t2));

    let t1 = input[9] * COSPI_23_64 - input[23] * COSPI_9_64;
    let t2 = input[9] * COSPI_9_64 + input[23] * COSPI_23_64;
    step1[18] = wraplow(dct_const_round_shift(t1));
    step1[29] = wraplow(dct_const_round_shift(t2));

    let t1 = input[25] * COSPI_7_64 - input[7] * COSPI_25_64;
    let t2 = input[25] * COSPI_25_64 + input[7] * COSPI_7_64;
    step1[19] = wraplow(dct_const_round_shift(t1));
    step1[28] = wraplow(dct_const_round_shift(t2));

    let t1 = input[5] * COSPI_27_64 - input[27] * COSPI_5_64;
    let t2 = input[5] * COSPI_5_64 + input[27] * COSPI_27_64;
    step1[20] = wraplow(dct_const_round_shift(t1));
    step1[27] = wraplow(dct_const_round_shift(t2));

    let t1 = input[21] * COSPI_11_64 - input[11] * COSPI_21_64;
    let t2 = input[21] * COSPI_21_64 + input[11] * COSPI_11_64;
    step1[21] = wraplow(dct_const_round_shift(t1));
    step1[26] = wraplow(dct_const_round_shift(t2));

    let t1 = input[13] * COSPI_19_64 - input[19] * COSPI_13_64;
    let t2 = input[13] * COSPI_13_64 + input[19] * COSPI_19_64;
    step1[22] = wraplow(dct_const_round_shift(t1));
    step1[25] = wraplow(dct_const_round_shift(t2));

    let t1 = input[29] * COSPI_3_64 - input[3] * COSPI_29_64;
    let t2 = input[29] * COSPI_29_64 + input[3] * COSPI_3_64;
    step1[23] = wraplow(dct_const_round_shift(t1));
    step1[24] = wraplow(dct_const_round_shift(t2));

    // stage 2
    step2[..8].copy_from_slice(&step1[..8]);

    let t1 = step1[8] * COSPI_30_64 - step1[15] * COSPI_2_64;
    let t2 = step1[8] * COSPI_2_64 + step1[15] * COSPI_30_64;
    step2[8] = wraplow(dct_const_round_shift(t1));
    step2[15] = wraplow(dct_const_round_shift(t2));

    let t1 = step1[9] * COSPI_14_64 - step1[14] * COSPI_18_64;
    let t2 = step1[9] * COSPI_18_64 + step1[14] * COSPI_14_64;
    step2[9] = wraplow(dct_const_round_shift(t1));
    step2[14] = wraplow(dct_const_round_shift(t2));

    let t1 = step1[10] * COSPI_22_64 - step1[13] * COSPI_10_64;
    let t2 = step1[10] * COSPI_10_64 + step1[13] * COSPI_22_64;
    step2[10] = wraplow(dct_const_round_shift(t1));
    step2[13] = wraplow(dct_const_round_shift(t2));

    let t1 = step1[11] * COSPI_6_64 - step1[12] * COSPI_26_64;
    let t2 = step1[11] * COSPI_26_64 + step1[12] * COSPI_6_64;
    step2[11] = wraplow(dct_const_round_shift(t1));
    step2[12] = wraplow(dct_const_round_shift(t2));

    step2[16] = wraplow(step1[16] + step1[17]);
    step2[17] = wraplow(step1[16] - step1[17]);
    step2[18] = wraplow(-step1[18] + step1[19]);
    step2[19] = wraplow(step1[18] + step1[19]);
    step2[20] = wraplow(step1[20] + step1[21]);
    step2[21] = wraplow(step1[20] - step1[21]);
    step2[22] = wraplow(-step1[22] + step1[23]);
    step2[23] = wraplow(step1[22] + step1[23]);
    step2[24] = wraplow(step1[24] + step1[25]);
    step2[25] = wraplow(step1[24] - step1[25]);
    step2[26] = wraplow(-step1[26] + step1[27]);
    step2[27] = wraplow(step1[26] + step1[27]);
    step2[28] = wraplow(step1[28] + step1[29]);
    step2[29] = wraplow(step1[28] - step1[29]);
    step2[30] = wraplow(-step1[30] + step1[31]);
    step2[31] = wraplow(step1[30] + step1[31]);

    // stage 3
    step1[..4].copy_from_slice(&step2[..4]);

    let t1 = step2[4] * COSPI_28_64 - step2[7] * COSPI_4_64;
    let t2 = step2[4] * COSPI_4_64 + step2[7] * COSPI_28_64;
    step1[4] = wraplow(dct_const_round_shift(t1));
    step1[7] = wraplow(dct_const_round_shift(t2));
    let t1 = step2[5] * COSPI_12_64 - step2[6] * COSPI_20_64;
    let t2 = step2[5] * COSPI_20_64 + step2[6] * COSPI_12_64;
    step1[5] = wraplow(dct_const_round_shift(t1));
    step1[6] = wraplow(dct_const_round_shift(t2));

    step1[8] = wraplow(step2[8] + step2[9]);
    step1[9] = wraplow(step2[8] - step2[9]);
    step1[10] = wraplow(-step2[10] + step2[11]);
    step1[11] = wraplow(step2[10] + step2[11]);
    step1[12] = wraplow(step2[12] + step2[13]);
    step1[13] = wraplow(step2[12] - step2[13]);
    step1[14] = wraplow(-step2[14] + step2[15]);
    step1[15] = wraplow(step2[14] + step2[15]);

    step1[16] = step2[16];
    step1[31] = step2[31];
    let t1 = -step2[17] * COSPI_4_64 + step2[30] * COSPI_28_64;
    let t2 = step2[17] * COSPI_28_64 + step2[30] * COSPI_4_64;
    step1[17] = wraplow(dct_const_round_shift(t1));
    step1[30] = wraplow(dct_const_round_shift(t2));
    let t1 = -step2[18] * COSPI_28_64 - step2[29] * COSPI_4_64;
    let t2 = -step2[18] * COSPI_4_64 + step2[29] * COSPI_28_64;
    step1[18] = wraplow(dct_const_round_shift(t1));
    step1[29] = wraplow(dct_const_round_shift(t2));
    step1[19] = step2[19];
    step1[20] = step2[20];
    let t1 = -step2[21] * COSPI_20_64 + step2[26] * COSPI_12_64;
    let t2 = step2[21] * COSPI_12_64 + step2[26] * COSPI_20_64;
    step1[21] = wraplow(dct_const_round_shift(t1));
    step1[26] = wraplow(dct_const_round_shift(t2));
    let t1 = -step2[22] * COSPI_12_64 - step2[25] * COSPI_20_64;
    let t2 = -step2[22] * COSPI_20_64 + step2[25] * COSPI_12_64;
    step1[22] = wraplow(dct_const_round_shift(t1));
    step1[25] = wraplow(dct_const_round_shift(t2));
    step1[23] = step2[23];
    step1[24] = step2[24];
    step1[27] = step2[27];
    step1[28] = step2[28];

    // stage 4
    let t1 = (step1[0] + step1[1]) * COSPI_16_64;
    let t2 = (step1[0] - step1[1]) * COSPI_16_64;
    step2[0] = wraplow(dct_const_round_shift(t1));
    step2[1] = wraplow(dct_const_round_shift(t2));
    let t1 = step1[2] * COSPI_24_64 - step1[3] * COSPI_8_64;
    let t2 = step1[2] * COSPI_8_64 + step1[3] * COSPI_24_64;
    step2[2] = wraplow(dct_const_round_shift(t1));
    step2[3] = wraplow(dct_const_round_shift(t2));
    step2[4] = wraplow(step1[4] + step1[5]);
    step2[5] = wraplow(step1[4] - step1[5]);
    step2[6] = wraplow(-step1[6] + step1[7]);
    step2[7] = wraplow(step1[6] + step1[7]);

    step2[8] = step1[8];
    step2[15] = step1[15];
    let t1 = -step1[9] * COSPI_8_64 + step1[14] * COSPI_24_64;
    let t2 = step1[9] * COSPI_24_64 + step1[14] * COSPI_8_64;
    step2[9] = wraplow(dct_const_round_shift(t1));
    step2[14] = wraplow(dct_const_round_shift(t2));
    let t1 = -step1[10] * COSPI_24_64 - step1[13] * COSPI_8_64;
    let t2 = -step1[10] * COSPI_8_64 + step1[13] * COSPI_24_64;
    step2[10] = wraplow(dct_const_round_shift(t1));
    step2[13] = wraplow(dct_const_round_shift(t2));
    step2[11] = step1[11];
    step2[12] = step1[12];

    step2[16] = wraplow(step1[16] + step1[19]);
    step2[17] = wraplow(step1[17] + step1[18]);
    step2[18] = wraplow(step1[17] - step1[18]);
    step2[19] = wraplow(step1[16] - step1[19]);
    step2[20] = wraplow(-step1[20] + step1[23]);
    step2[21] = wraplow(-step1[21] + step1[22]);
    step2[22] = wraplow(step1[21] + step1[22]);
    step2[23] = wraplow(step1[20] + step1[23]);

    step2[24] = wraplow(step1[24] + step1[27]);
    step2[25] = wraplow(step1[25] + step1[26]);
    step2[26] = wraplow(step1[25] - step1[26]);
    step2[27] = wraplow(step1[24] - step1[27]);
    step2[28] = wraplow(-step1[28] + step1[31]);
    step2[29] = wraplow(-step1[29] + step1[30]);
    step2[30] = wraplow(step1[29] + step1[30]);
    step2[31] = wraplow(step1[28] + step1[31]);

    // stage 5
    step1[0] = wraplow(step2[0] + step2[3]);
    step1[1] = wraplow(step2[1] + step2[2]);
    step1[2] = wraplow(step2[1] - step2[2]);
    step1[3] = wraplow(step2[0] - step2[3]);
    step1[4] = step2[4];
    let t1 = (step2[6] - step2[5]) * COSPI_16_64;
    let t2 = (step2[5] + step2[6]) * COSPI_16_64;
    step1[5] = wraplow(dct_const_round_shift(t1));
    step1[6] = wraplow(dct_const_round_shift(t2));
    step1[7] = step2[7];

    step1[8] = wraplow(step2[8] + step2[11]);
    step1[9] = wraplow(step2[9] + step2[10]);
    step1[10] = wraplow(step2[9] - step2[10]);
    step1[11] = wraplow(step2[8] - step2[11]);
    step1[12] = wraplow(-step2[12] + step2[15]);
    step1[13] = wraplow(-step2[13] + step2[14]);
    step1[14] = wraplow(step2[13] + step2[14]);
    step1[15] = wraplow(step2[12] + step2[15]);

    step1[16] = step2[16];
    step1[17] = step2[17];
    let t1 = -step2[18] * COSPI_8_64 + step2[29] * COSPI_24_64;
    let t2 = step2[18] * COSPI_24_64 + step2[29] * COSPI_8_64;
    step1[18] = wraplow(dct_const_round_shift(t1));
    step1[29] = wraplow(dct_const_round_shift(t2));
    let t1 = -step2[19] * COSPI_8_64 + step2[28] * COSPI_24_64;
    let t2 = step2[19] * COSPI_24_64 + step2[28] * COSPI_8_64;
    step1[19] = wraplow(dct_const_round_shift(t1));
    step1[28] = wraplow(dct_const_round_shift(t2));
    let t1 = -step2[20] * COSPI_24_64 - step2[27] * COSPI_8_64;
    let t2 = -step2[20] * COSPI_8_64 + step2[27] * COSPI_24_64;
    step1[20] = wraplow(dct_const_round_shift(t1));
    step1[27] = wraplow(dct_const_round_shift(t2));
    let t1 = -step2[21] * COSPI_24_64 - step2[26] * COSPI_8_64;
    let t2 = -step2[21] * COSPI_8_64 + step2[26] * COSPI_24_64;
    step1[21] = wraplow(dct_const_round_shift(t1));
    step1[26] = wraplow(dct_const_round_shift(t2));
    step1[22] = step2[22];
    step1[23] = step2[23];
    step1[24] = step2[24];
    step1[25] = step2[25];
    step1[30] = step2[30];
    step1[31] = step2[31];

    // stage 6
    step2[0] = wraplow(step1[0] + step1[7]);
    step2[1] = wraplow(step1[1] + step1[6]);
    step2[2] = wraplow(step1[2] + step1[5]);
    step2[3] = wraplow(step1[3] + step1[4]);
    step2[4] = wraplow(step1[3] - step1[4]);
    step2[5] = wraplow(step1[2] - step1[5]);
    step2[6] = wraplow(step1[1] - step1[6]);
    step2[7] = wraplow(step1[0] - step1[7]);
    step2[8] = step1[8];
    step2[9] = step1[9];
    let t1 = (-step1[10] + step1[13]) * COSPI_16_64;
    let t2 = (step1[10] + step1[13]) * COSPI_16_64;
    step2[10] = wraplow(dct_const_round_shift(t1));
    step2[13] = wraplow(dct_const_round_shift(t2));
    let t1 = (-step1[11] + step1[12]) * COSPI_16_64;
    let t2 = (step1[11] + step1[12]) * COSPI_16_64;
    step2[11] = wraplow(dct_const_round_shift(t1));
    step2[12] = wraplow(dct_const_round_shift(t2));
    step2[14] = step1[14];
    step2[15] = step1[15];

    step2[16] = wraplow(step1[16] + step1[23]);
    step2[17] = wraplow(step1[17] + step1[22]);
    step2[18] = wraplow(step1[18] + step1[21]);
    step2[19] = wraplow(step1[19] + step1[20]);
    step2[20] = wraplow(step1[19] - step1[20]);
    step2[21] = wraplow(step1[18] - step1[21]);
    step2[22] = wraplow(step1[17] - step1[22]);
    step2[23] = wraplow(step1[16] - step1[23]);

    step2[24] = wraplow(-step1[24] + step1[31]);
    step2[25] = wraplow(-step1[25] + step1[30]);
    step2[26] = wraplow(-step1[26] + step1[29]);
    step2[27] = wraplow(-step1[27] + step1[28]);
    step2[28] = wraplow(step1[27] + step1[28]);
    step2[29] = wraplow(step1[26] + step1[29]);
    step2[30] = wraplow(step1[25] + step1[30]);
    step2[31] = wraplow(step1[24] + step1[31]);

    // stage 7
    step1[0] = wraplow(step2[0] + step2[15]);
    step1[1] = wraplow(step2[1] + step2[14]);
    step1[2] = wraplow(step2[2] + step2[13]);
    step1[3] = wraplow(step2[3] + step2[12]);
    step1[4] = wraplow(step2[4] + step2[11]);
    step1[5] = wraplow(step2[5] + step2[10]);
    step1[6] = wraplow(step2[6] + step2[9]);
    step1[7] = wraplow(step2[7] + step2[8]);
    step1[8] = wraplow(step2[7] - step2[8]);
    step1[9] = wraplow(step2[6] - step2[9]);
    step1[10] = wraplow(step2[5] - step2[10]);
    step1[11] = wraplow(step2[4] - step2[11]);
    step1[12] = wraplow(step2[3] - step2[12]);
    step1[13] = wraplow(step2[2] - step2[13]);
    step1[14] = wraplow(step2[1] - step2[14]);
    step1[15] = wraplow(step2[0] - step2[15]);

    step1[16] = step2[16];
    step1[17] = step2[17];
    step1[18] = step2[18];
    step1[19] = step2[19];
    let t1 = (-step2[20] + step2[27]) * COSPI_16_64;
    let t2 = (step2[20] + step2[27]) * COSPI_16_64;
    step1[20] = wraplow(dct_const_round_shift(t1));
    step1[27] = wraplow(dct_const_round_shift(t2));
    let t1 = (-step2[21] + step2[26]) * COSPI_16_64;
    let t2 = (step2[21] + step2[26]) * COSPI_16_64;
    step1[21] = wraplow(dct_const_round_shift(t1));
    step1[26] = wraplow(dct_const_round_shift(t2));
    let t1 = (-step2[22] + step2[25]) * COSPI_16_64;
    let t2 = (step2[22] + step2[25]) * COSPI_16_64;
    step1[22] = wraplow(dct_const_round_shift(t1));
    step1[25] = wraplow(dct_const_round_shift(t2));
    let t1 = (-step2[23] + step2[24]) * COSPI_16_64;
    let t2 = (step2[23] + step2[24]) * COSPI_16_64;
    step1[23] = wraplow(dct_const_round_shift(t1));
    step1[24] = wraplow(dct_const_round_shift(t2));
    step1[28] = step2[28];
    step1[29] = step2[29];
    step1[30] = step2[30];
    step1[31] = step2[31];

    let mut out = [0i32; 32];
    out[0] = wraplow(step1[0] + step1[31]);
    out[1] = wraplow(step1[1] + step1[30]);
    out[2] = wraplow(step1[2] + step1[29]);
    out[3] = wraplow(step1[3] + step1[28]);
    out[4] = wraplow(step1[4] + step1[27]);
    out[5] = wraplow(step1[5] + step1[26]);
    out[6] = wraplow(step1[6] + step1[25]);
    out[7] = wraplow(step1[7] + step1[24]);
    out[8] = wraplow(step1[8] + step1[23]);
    out[9] = wraplow(step1[9] + step1[22]);
    out[10] = wraplow(step1[10] + step1[21]);
    out[11] = wraplow(step1[11] + step1[20]);
    out[12] = wraplow(step1[12] + step1[19]);
    out[13] = wraplow(step1[13] + step1[18]);
    out[14] = wraplow(step1[14] + step1[17]);
    out[15] = wraplow(step1[15] + step1[16]);
    out[16] = wraplow(step1[15] - step1[16]);
    out[17] = wraplow(step1[14] - step1[17]);
    out[18] = wraplow(step1[13] - step1[18]);
    out[19] = wraplow(step1[12] - step1[19]);
    out[20] = wraplow(step1[11] - step1[20]);
    out[21] = wraplow(step1[10] - step1[21]);
    out[22] = wraplow(step1[9] - step1[22]);
    out[23] = wraplow(step1[8] - step1[23]);
    out[24] = wraplow(step1[7] - step1[24]);
    out[25] = wraplow(step1[6] - step1[25]);
    out[26] = wraplow(step1[5] - step1[26]);
    out[27] = wraplow(step1[4] - step1[27]);
    out[28] = wraplow(step1[3] - step1[28]);
    out[29] = wraplow(step1[2] - step1[29]);
    out[30] = wraplow(step1[1] - step1[30]);
    out[31] = wraplow(step1[0] - step1[31]);
    out
}

// ===== 4x4 WHT (lossless) =====
//
// Spec ref: §8.7.1.10 (Inverse Walsh-Hadamard transform process) +
// §8.7.2 (2D Inverse Transform). The row pass is applied with shift=2
// (since the encoder pre-multiplies by ac_q[0]=4 in the lossless
// quantizer), the column pass with shift=0. After the column pass the
// result is added directly to the predictor (no Round2), per the
// "If Lossless is equal to 1, set Dequant[i][j] = T[i]" branch in §8.7.2.
fn wht_1d(t: &mut [i32; 4], shift: u32) {
    let a = t[0] >> shift;
    let c = t[1] >> shift;
    let d = t[2] >> shift;
    let b = t[3] >> shift;
    let a = a + c;
    let d = d - b;
    let e = (a - d) >> 1;
    let b = e - b;
    let c = e - c;
    let a = a - b;
    let d = d + c;
    t[0] = a;
    t[1] = b;
    t[2] = c;
    t[3] = d;
}

fn iwht4x4_add(input: &[i32], dst: &mut [u8], stride: usize) {
    let mut tmp = [0i32; 16];
    // Row pass — shift=2 per §8.7.2 first bullet for Lossless.
    for i in 0..4 {
        let mut t = [
            input[i * 4],
            input[i * 4 + 1],
            input[i * 4 + 2],
            input[i * 4 + 3],
        ];
        wht_1d(&mut t, 2);
        tmp[i * 4] = t[0];
        tmp[i * 4 + 1] = t[1];
        tmp[i * 4 + 2] = t[2];
        tmp[i * 4 + 3] = t[3];
    }
    // Column pass — shift=0 per §8.7.2. Result is added without Round2.
    for j in 0..4 {
        let mut t = [tmp[j], tmp[4 + j], tmp[8 + j], tmp[12 + j]];
        wht_1d(&mut t, 0);
        dst[j] = clip_pixel_add(dst[j], t[0]);
        dst[stride + j] = clip_pixel_add(dst[stride + j], t[1]);
        dst[2 * stride + j] = clip_pixel_add(dst[2 * stride + j], t[2]);
        dst[3 * stride + j] = clip_pixel_add(dst[3 * stride + j], t[3]);
    }
}

// ===== 2-D inverse with clip-add =====

/// Apply a 2-D inverse transform to `coeffs` (row-major `w × h` `i32`
/// dequantised coefficients) and add the residual to `dst`.
pub fn inverse_transform_add(
    tx: TxType,
    w: usize,
    h: usize,
    coeffs: &[i32],
    dst: &mut [u8],
    dst_stride: usize,
) -> Result<()> {
    if coeffs.len() != w * h {
        return Err(Error::invalid(format!(
            "vp9 tx: coeffs len {} != {w}*{h}",
            coeffs.len()
        )));
    }
    if !matches!((w, h), (4, 4) | (8, 8) | (16, 16) | (32, 32)) {
        return Err(Error::invalid(format!("vp9 tx: unsupported size {w}×{h}")));
    }
    if tx == TxType::WhtWht {
        if !matches!((w, h), (4, 4)) {
            return Err(Error::invalid("vp9 tx: WHT only valid at 4×4"));
        }
        iwht4x4_add(coeffs, dst, dst_stride);
        return Ok(());
    }
    // 32×32 only supports DCT_DCT.
    if (w, h) == (32, 32) && tx != TxType::DctDct {
        return Err(Error::invalid(
            "vp9 tx: 32×32 only supports DCT_DCT (§8.7.1)",
        ));
    }

    let shift = match (w, h) {
        (4, 4) => 4,
        (8, 8) => 5,
        (16, 16) => 6,
        (32, 32) => 6,
        _ => unreachable!(),
    };

    // Row pass into tmp.
    let mut tmp = vec![0i32; w * h];
    for r in 0..h {
        let row = &coeffs[r * w..r * w + w];
        match (tx, w) {
            (TxType::DctDct | TxType::AdstDct, 4) => {
                let mut a = [0i32; 4];
                a.copy_from_slice(row);
                let out = idct4(&a);
                tmp[r * w..r * w + 4].copy_from_slice(&out);
            }
            (TxType::DctAdst | TxType::AdstAdst, 4) => {
                let mut a = [0i32; 4];
                a.copy_from_slice(row);
                let out = iadst4(&a);
                tmp[r * w..r * w + 4].copy_from_slice(&out);
            }
            (TxType::DctDct | TxType::AdstDct, 8) => {
                let mut a = [0i32; 8];
                a.copy_from_slice(row);
                let out = idct8(&a);
                tmp[r * w..r * w + 8].copy_from_slice(&out);
            }
            (TxType::DctAdst | TxType::AdstAdst, 8) => {
                let mut a = [0i32; 8];
                a.copy_from_slice(row);
                let out = iadst8(&a);
                tmp[r * w..r * w + 8].copy_from_slice(&out);
            }
            (TxType::DctDct | TxType::AdstDct, 16) => {
                let mut a = [0i32; 16];
                a.copy_from_slice(row);
                let out = idct16(&a);
                tmp[r * w..r * w + 16].copy_from_slice(&out);
            }
            (TxType::DctAdst | TxType::AdstAdst, 16) => {
                let mut a = [0i32; 16];
                a.copy_from_slice(row);
                let out = iadst16(&a);
                tmp[r * w..r * w + 16].copy_from_slice(&out);
            }
            (TxType::DctDct, 32) => {
                let mut a = [0i32; 32];
                a.copy_from_slice(row);
                let mut out = idct32(&a);
                // 32x32 applies an extra round_shift(2) after each 1-D pass per libvpx.
                for v in &mut out {
                    *v = round_power_of_two(*v, 2);
                }
                tmp[r * w..r * w + 32].copy_from_slice(&out);
            }
            _ => unreachable!(),
        }
    }
    // Column pass.
    let mut out = vec![0i32; w * h];
    for c in 0..w {
        match (tx, h) {
            (TxType::DctDct | TxType::DctAdst, 4) => {
                let mut a = [0i32; 4];
                for r in 0..4 {
                    a[r] = tmp[r * w + c];
                }
                let v = idct4(&a);
                for r in 0..4 {
                    out[r * w + c] = v[r];
                }
            }
            (TxType::AdstDct | TxType::AdstAdst, 4) => {
                let mut a = [0i32; 4];
                for r in 0..4 {
                    a[r] = tmp[r * w + c];
                }
                let v = iadst4(&a);
                for r in 0..4 {
                    out[r * w + c] = v[r];
                }
            }
            (TxType::DctDct | TxType::DctAdst, 8) => {
                let mut a = [0i32; 8];
                for r in 0..8 {
                    a[r] = tmp[r * w + c];
                }
                let v = idct8(&a);
                for r in 0..8 {
                    out[r * w + c] = v[r];
                }
            }
            (TxType::AdstDct | TxType::AdstAdst, 8) => {
                let mut a = [0i32; 8];
                for r in 0..8 {
                    a[r] = tmp[r * w + c];
                }
                let v = iadst8(&a);
                for r in 0..8 {
                    out[r * w + c] = v[r];
                }
            }
            (TxType::DctDct | TxType::DctAdst, 16) => {
                let mut a = [0i32; 16];
                for r in 0..16 {
                    a[r] = tmp[r * w + c];
                }
                let v = idct16(&a);
                for r in 0..16 {
                    out[r * w + c] = v[r];
                }
            }
            (TxType::AdstDct | TxType::AdstAdst, 16) => {
                let mut a = [0i32; 16];
                for r in 0..16 {
                    a[r] = tmp[r * w + c];
                }
                let v = iadst16(&a);
                for r in 0..16 {
                    out[r * w + c] = v[r];
                }
            }
            (TxType::DctDct, 32) => {
                let mut a = [0i32; 32];
                for r in 0..32 {
                    a[r] = tmp[r * w + c];
                }
                let v = idct32(&a);
                for r in 0..32 {
                    out[r * w + c] = v[r];
                }
            }
            _ => unreachable!(),
        }
    }
    // Apply final round_shift and clip-add.
    for r in 0..h {
        for c in 0..w {
            let residual = round_power_of_two(out[r * w + c], shift);
            dst[r * dst_stride + c] = clip_pixel_add(dst[r * dst_stride + c], residual);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_coeffs_4x4_is_noop() {
        let coeffs = [0i32; 16];
        let mut dst = [42u8; 16];
        inverse_transform_add(TxType::DctDct, 4, 4, &coeffs, &mut dst, 4).unwrap();
        for &v in &dst {
            assert_eq!(v, 42);
        }
    }

    #[test]
    fn zero_coeffs_all_sizes_is_noop() {
        for &n in &[4, 8, 16, 32] {
            let coeffs = vec![0i32; n * n];
            let mut dst = vec![64u8; n * n];
            inverse_transform_add(TxType::DctDct, n, n, &coeffs, &mut dst, n).unwrap();
            for &v in &dst {
                assert_eq!(v, 64);
            }
        }
    }

    #[test]
    fn dc_only_4x4_shifts_uniformly() {
        let mut coeffs = [0i32; 16];
        coeffs[0] = 64;
        let mut dst = [50u8; 16];
        inverse_transform_add(TxType::DctDct, 4, 4, &coeffs, &mut dst, 4).unwrap();
        let first = dst[0];
        for &v in &dst {
            assert!(v.abs_diff(first) <= 1);
        }
    }

    #[test]
    fn dc_only_16x16_shifts_uniformly() {
        let mut coeffs = [0i32; 256];
        coeffs[0] = 128;
        let mut dst = [50u8; 256];
        inverse_transform_add(TxType::DctDct, 16, 16, &coeffs, &mut dst, 16).unwrap();
        let first = dst[0];
        for &v in &dst {
            assert!(v.abs_diff(first) <= 1);
        }
    }

    #[test]
    fn zero_coeffs_adst_is_noop() {
        let coeffs = [0i32; 16];
        let mut dst = [99u8; 16];
        inverse_transform_add(TxType::AdstAdst, 4, 4, &coeffs, &mut dst, 4).unwrap();
        for &v in &dst {
            assert_eq!(v, 99);
        }
    }
}
