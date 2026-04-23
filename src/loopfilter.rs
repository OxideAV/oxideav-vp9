//! VP9 loop filter (§8.8).
//!
//! Implements the spec's 8x8-MI-unit deblocking pass that smooths the
//! vertical and horizontal edges left behind by per-block prediction +
//! transform. Eight-bit 4:2:0 only for now — matches the rest of the
//! decoder. Higher bit depths slot in by parameterising `BitDepth`
//! everywhere the spec references it (the code keeps the shifts explicit
//! so the upgrade is mechanical).
//!
//! The public entry point is [`LoopFilter::apply_frame`], which walks the
//! frame MI grid exactly as §8.8 "Loop filter process" prescribes:
//!
//! ```text
//! for ( row = 0; row < MiRows; row += 8 )
//!   for ( col = 0; col < MiCols; col += 8 )
//!     for ( plane = 0; plane < 3; plane++ )
//!       for ( pass = 0; pass < 2; pass++ )
//!         superblock_loop_filter(plane, pass, row, col)
//! ```
//!
//! Per-pass, per-edge filtering uses §8.8.5 sample filter semantics:
//! narrow (`filter4`), wide-8 (`filter8`), wide-16 (`filter16`). The
//! `filter_mask`, `flat_mask`, `flat_mask2` machinery is §8.8.5.1.
//!
//! Per-8x8-MI metadata (`mi_size`, `tx_size`, `skip`, `ref_frame`,
//! `y_mode`, `segment_id`) is collected during block decode into an
//! [`MiInfoPlane`] grid. The loop filter reads this grid to drive the
//! adaptive filter strength pick (§8.8.4) and the edge-type checks
//! (§8.8.2 items 10-14).

use crate::headers::LoopFilterParams;

/// Max filter level clamp — spec constant `MAX_LOOP_FILTER` (§3 Table 3-2).
pub const MAX_LOOP_FILTER: i32 = 63;

/// `MAX_REF_FRAMES` — intra + LAST + GOLDEN + ALTREF (§3).
pub const MAX_REF_FRAMES: usize = 4;

/// `MAX_MODE_LF_DELTAS` — 0 (intra/zeromv) or 1 (non-zero inter).
pub const MAX_MODE_LF_DELTAS: usize = 2;

/// `MAX_SEGMENTS` — spec constant.
pub const MAX_SEGMENTS: usize = 8;

/// Reference-frame code used in the §8.8.4 `ref` lookup. Intra blocks
/// report `INTRA_FRAME = 0`; inter blocks report `1 + ref_slot` (LAST=1,
/// GOLDEN=2, ALTREF=3).
pub const INTRA_FRAME: u8 = 0;

/// Per-8x8 MI-block metadata the loop filter consumes.
///
/// The decoder populates these during block decode, keyed on the 8x8
/// grid coordinates (`mi_row = row_pixels / 8`, `mi_col = col_pixels / 8`).
#[derive(Clone, Copy, Debug, Default)]
pub struct MiInfo {
    /// Width of the prediction block in 8x8 units (1, 2, 4, 8).
    pub mi_w_8x8: u8,
    /// Height of the prediction block in 8x8 units.
    pub mi_h_8x8: u8,
    /// Transform size log2 (0=4x4, 1=8x8, 2=16x16, 3=32x32). Used
    /// to compute `isTxEdge` in §8.8.2.
    pub tx_size_log2: u8,
    /// 1 if the block has no residual (§6.4.8 `skip`).
    pub skip: bool,
    /// Reference frame code (0=INTRA, 1=LAST, 2=GOLDEN, 3=ALTREF).
    /// Drives the §8.8.4 ref delta lookup.
    pub ref_frame: u8,
    /// `true` when mode ∈ {NEARESTMV, NEARMV, NEWMV} — §8.8.4 modeType=1.
    pub mode_is_non_zero_inter: bool,
    /// Segment ID (0 when segmentation is disabled).
    pub segment_id: u8,
}

/// Per-plane mi-grid storage. Each cell is one 8x8 luma block worth of
/// metadata. When we're filtering chroma we still index by luma mi
/// coordinates — the spec does the same.
#[derive(Clone, Debug, Default)]
pub struct MiInfoPlane {
    pub mi_cols: usize,
    pub mi_rows: usize,
    pub cells: Vec<MiInfo>,
}

impl MiInfoPlane {
    pub fn new(mi_cols: usize, mi_rows: usize) -> Self {
        Self {
            mi_cols,
            mi_rows,
            cells: vec![MiInfo::default(); mi_cols.max(1) * mi_rows.max(1)],
        }
    }

    pub fn get(&self, mi_row: usize, mi_col: usize) -> MiInfo {
        if self.mi_cols == 0 || self.mi_rows == 0 {
            return MiInfo::default();
        }
        let r = mi_row.min(self.mi_rows - 1);
        let c = mi_col.min(self.mi_cols - 1);
        self.cells[r * self.mi_cols + c]
    }

    /// Stamp the same `info` into every 8x8 cell covered by the
    /// prediction block at `(mi_row, mi_col)` with size
    /// `info.mi_w_8x8 × info.mi_h_8x8`. Called once per block decode.
    pub fn fill(&mut self, mi_row: usize, mi_col: usize, info: MiInfo) {
        let w = info.mi_w_8x8 as usize;
        let h = info.mi_h_8x8 as usize;
        for dy in 0..h.max(1) {
            let r = mi_row + dy;
            if r >= self.mi_rows {
                break;
            }
            for dx in 0..w.max(1) {
                let c = mi_col + dx;
                if c >= self.mi_cols {
                    break;
                }
                self.cells[r * self.mi_cols + c] = info;
            }
        }
    }
}

/// Precomputed §8.8.1 `LvlLookup[segment][ref][mode_type]` table.
#[derive(Clone, Debug)]
pub struct LvlLookup {
    /// `[segment][ref][mode_type]`.
    table: [[[u8; MAX_MODE_LF_DELTAS]; MAX_REF_FRAMES]; MAX_SEGMENTS],
}

impl LvlLookup {
    /// Build the lookup table per §8.8.1. Segmentation is not yet applied
    /// by the decoder, so we treat every segment as having zero deltas.
    pub fn build(lf: &LoopFilterParams) -> Self {
        let base = lf.level as i32;
        let n_shift = base >> 5;
        let mut table = [[[0u8; MAX_MODE_LF_DELTAS]; MAX_REF_FRAMES]; MAX_SEGMENTS];
        for seg in 0..MAX_SEGMENTS {
            let lvl_seg = base; // SEG_LVL_ALT_L not applied yet.
            if !lf.mode_ref_delta_enabled {
                let clamped = lvl_seg.clamp(0, MAX_LOOP_FILTER) as u8;
                for rf in 0..MAX_REF_FRAMES {
                    for m in 0..MAX_MODE_LF_DELTAS {
                        table[seg][rf][m] = clamped;
                    }
                }
                continue;
            }
            // Intra: modeType is always 0.
            let intra_lvl = lvl_seg + ((lf.ref_deltas[0] as i32) << n_shift);
            table[seg][INTRA_FRAME as usize][0] = intra_lvl.clamp(0, MAX_LOOP_FILTER) as u8;
            table[seg][INTRA_FRAME as usize][1] = table[seg][INTRA_FRAME as usize][0];
            // Inter.
            for rf in 1..MAX_REF_FRAMES {
                for mode in 0..MAX_MODE_LF_DELTAS {
                    let inter_lvl = lvl_seg
                        + ((lf.ref_deltas[rf] as i32) << n_shift)
                        + ((lf.mode_deltas[mode] as i32) << n_shift);
                    table[seg][rf][mode] = inter_lvl.clamp(0, MAX_LOOP_FILTER) as u8;
                }
            }
        }
        Self { table }
    }

    pub fn get(&self, segment: u8, ref_frame: u8, mode_type: u8) -> u8 {
        let s = (segment as usize).min(MAX_SEGMENTS - 1);
        let r = (ref_frame as usize).min(MAX_REF_FRAMES - 1);
        let m = (mode_type as usize).min(MAX_MODE_LF_DELTAS - 1);
        self.table[s][r][m]
    }
}

/// One 2-D plane view: writable buffer + stride + dimensions.
pub struct PlaneMut<'a> {
    pub buf: &'a mut [u8],
    pub stride: usize,
    pub width: usize,
    pub height: usize,
}

impl PlaneMut<'_> {
    fn read(&self, row: isize, col: isize) -> u8 {
        // Edge clamp so the filter handles image borders without extra
        // guards at every tap. The spec's §8.8 doesn't actually clamp —
        // but it never reads off-image because `onScreen` gates the
        // whole edge. Clamping here is therefore safe belt-and-braces.
        let r = row.clamp(0, self.height.saturating_sub(1) as isize) as usize;
        let c = col.clamp(0, self.width.saturating_sub(1) as isize) as usize;
        self.buf[r * self.stride + c]
    }

    fn write(&mut self, row: isize, col: isize, value: u8) {
        if row < 0 || col < 0 {
            return;
        }
        let r = row as usize;
        let c = col as usize;
        if r >= self.height || c >= self.width {
            return;
        }
        self.buf[r * self.stride + c] = value;
    }
}

/// Loop filter driver — holds frame-level params + the Mi-grid so a
/// caller can invoke it once after decode.
pub struct LoopFilter {
    pub lf: LoopFilterParams,
    pub lvl: LvlLookup,
    pub mi_cols: usize,
    pub mi_rows: usize,
    pub subsampling_x: bool,
    pub subsampling_y: bool,
}

impl LoopFilter {
    pub fn new(
        lf: &LoopFilterParams,
        mi_cols: usize,
        mi_rows: usize,
        subsampling_x: bool,
        subsampling_y: bool,
    ) -> Self {
        Self {
            lf: *lf,
            lvl: LvlLookup::build(lf),
            mi_cols,
            mi_rows,
            subsampling_x,
            subsampling_y,
        }
    }

    /// Apply the loop filter to the three planes. Early-outs on
    /// `loop_filter_level == 0` (§8.8 the spec is a no-op in this case).
    #[allow(clippy::too_many_arguments)]
    pub fn apply_frame(
        &self,
        info: &MiInfoPlane,
        y: &mut [u8],
        y_stride: usize,
        y_w: usize,
        y_h: usize,
        u: &mut [u8],
        v: &mut [u8],
        uv_stride: usize,
        uv_w: usize,
        uv_h: usize,
    ) {
        if self.lf.level == 0 {
            return;
        }
        // Raster walk in 8x8 MI units, 64x64-aligned (superblock).
        let sb_mi_rows = self.mi_rows.div_ceil(8) * 8;
        let sb_mi_cols = self.mi_cols.div_ceil(8) * 8;
        let mut row_mi = 0usize;
        while row_mi < sb_mi_rows {
            let mut col_mi = 0usize;
            while col_mi < sb_mi_cols {
                // Y plane.
                self.superblock_pass(info, y, y_stride, y_w, y_h, 0, row_mi, col_mi, 0);
                self.superblock_pass(info, y, y_stride, y_w, y_h, 0, row_mi, col_mi, 1);
                // U plane.
                self.superblock_pass(info, u, uv_stride, uv_w, uv_h, 1, row_mi, col_mi, 0);
                self.superblock_pass(info, u, uv_stride, uv_w, uv_h, 1, row_mi, col_mi, 1);
                // V plane.
                self.superblock_pass(info, v, uv_stride, uv_w, uv_h, 2, row_mi, col_mi, 0);
                self.superblock_pass(info, v, uv_stride, uv_w, uv_h, 2, row_mi, col_mi, 1);
                col_mi += 8;
            }
            row_mi += 8;
        }
    }

    /// §8.8.2 Superblock loop filter process for one `(plane, pass)`
    /// combination at the given 8x8-mi-units superblock origin.
    #[allow(clippy::too_many_arguments)]
    fn superblock_pass(
        &self,
        info: &MiInfoPlane,
        buf: &mut [u8],
        stride: usize,
        plane_w: usize,
        plane_h: usize,
        plane: u8,
        row_mi: usize,
        col_mi: usize,
        pass: u8,
    ) {
        let (sub_x, sub_y) = if plane == 0 {
            (0u32, 0u32)
        } else {
            (self.subsampling_x as u32, self.subsampling_y as u32)
        };
        let (dx, dy, sub, edge_len) = if pass == 0 {
            (1i32, 0i32, sub_x, 64u32 >> sub_y)
        } else {
            (0i32, 1i32, sub_y, 64u32 >> sub_x)
        };
        let edges = 16u32 >> sub;
        let mut plane_view = PlaneMut {
            buf,
            stride,
            width: plane_w,
            height: plane_h,
        };
        for edge in 0..edges {
            for i in 0..edge_len {
                // Step 1: luma-coordinate (x, y).
                let (x, y) = if pass == 0 {
                    (
                        (col_mi as i32) * 8 + (edge as i32) * (4 << sub_x),
                        (row_mi as i32) * 8 + ((i << sub_y) as i32),
                    )
                } else {
                    (
                        (col_mi as i32) * 8 + ((i << sub_x) as i32),
                        (row_mi as i32) * 8 + (edge as i32) * (4 << sub_y),
                    )
                };
                let loop_col_u = (x >> 3).max(0) as usize;
                let loop_row_u = (y >> 3).max(0) as usize;
                let loop_col = (loop_col_u >> sub_x) << sub_x;
                let loop_row = (loop_row_u >> sub_y) << sub_y;
                if loop_row >= self.mi_rows || loop_col >= self.mi_cols {
                    continue;
                }
                let mi = info.get(loop_row, loop_col);
                let mi_size_8x8 = mi.mi_w_8x8.max(mi.mi_h_8x8).max(1) as u32;
                let tx_size = mi.tx_size_log2 as u32;
                let tx_sz = if plane > 0 {
                    // get_uv_tx_size — libvpx caps chroma tx at the max
                    // tx that fits the chroma block. Approximation: use
                    // the luma tx clamped to TX_16X16.
                    tx_size.min(2)
                } else {
                    tx_size
                };
                let sb_size = if sub == 0 {
                    mi_size_8x8
                } else {
                    // BLOCK_16X16 in mi-units = 2.
                    mi_size_8x8.max(2)
                };
                let skip = mi.skip;
                let is_intra = mi.ref_frame == INTRA_FRAME;
                // §8.8.2 step 10 isBlockEdge: x (or y) is an exact
                // multiple of 8 * num_8x8_blocks_*_lookup[sbSize].
                let is_block_edge = if pass == 0 {
                    let step = 8 * sb_size as i32;
                    step > 0 && (x % step) == 0
                } else {
                    let step = 8 * sb_size as i32;
                    step > 0 && (y % step) == 0
                };
                // §8.8.2 step 11: chroma cross-of-right-edge exception.
                let is_tx_edge = if pass == 1
                    && sub_x == 1
                    && (self.mi_cols & 1) == 1
                    && (edge & 1) == 1
                    && (x + 8) >= (self.mi_cols as i32) * 8
                {
                    false
                } else {
                    let step_tx = 1u32 << tx_sz;
                    (edge % step_tx) == 0
                };
                // §8.8.2 step 12 is32Edge: edge is an exact multiple of 8.
                let is32_edge = (edge % 8) == 0;
                // §8.8.2 step 13 onScreen.
                let on_screen = !(x >= 8 * self.mi_cols as i32
                    || y >= 8 * self.mi_rows as i32
                    || (pass == 0 && x == 0)
                    || (pass == 1 && y == 0));
                // §8.8.2 step 14 applyFilter.
                let apply_filter = if !on_screen {
                    false
                } else if is_block_edge {
                    true
                } else if is_tx_edge && is_intra {
                    true
                } else {
                    is_tx_edge && !skip
                };
                // §8.8.3 filter size: log2 of filter side (0=TX_4X4,
                // 1=TX_8X8, 2=TX_16X16).
                let base_size = if tx_sz == 0 && is32_edge {
                    1u32
                } else {
                    tx_sz.min(2)
                };
                // Right/bottom edge reduction for chroma 16x16 base_size.
                let filter_size = if pass == 0
                    && sub_x == 1
                    && base_size == 2
                    && (x >> 3) == (self.mi_cols as i32) - 1
                {
                    1
                } else if pass == 1
                    && sub_y == 1
                    && base_size == 2
                    && (y >> 3) == (self.mi_rows as i32) - 1
                {
                    1
                } else {
                    base_size
                };
                // §8.8.4 adaptive filter strength.
                let segment = mi.segment_id;
                let ref_frame = mi.ref_frame;
                let mode_type = if is_intra {
                    0
                } else if mi.mode_is_non_zero_inter {
                    1
                } else {
                    0
                };
                let lvl = self.lvl.get(segment, ref_frame, mode_type) as i32;
                let shift = if self.lf.sharpness > 4 {
                    2
                } else if self.lf.sharpness > 0 {
                    1
                } else {
                    0
                };
                let limit = if self.lf.sharpness > 0 {
                    (lvl >> shift).clamp(1, 9 - self.lf.sharpness as i32) as u8
                } else {
                    (lvl >> shift).max(1) as u8
                };
                let blimit = ((2 * (lvl + 2) + limit as i32).clamp(0, 255)) as u8;
                let thresh = (lvl >> 4).clamp(0, 255) as u8;
                if !apply_filter || lvl == 0 {
                    continue;
                }
                // §8.8.5 sample filtering at the plane-scaled coordinate.
                let x_plane = x >> sub_x;
                let y_plane = y >> sub_y;
                apply_sample_filter(
                    &mut plane_view,
                    x_plane,
                    y_plane,
                    dx,
                    dy,
                    filter_size,
                    limit,
                    blimit,
                    thresh,
                );
            }
        }
    }
}

/// §8.8.5 sample filtering process: pick between narrow/wide-8/wide-16
/// based on the filter-size decision and the flat-mask tests.
#[allow(clippy::too_many_arguments)]
fn apply_sample_filter(
    plane: &mut PlaneMut<'_>,
    x: i32,
    y: i32,
    dx: i32,
    dy: i32,
    filter_size: u32,
    limit: u8,
    blimit: u8,
    thresh: u8,
) {
    // Gather 8 samples on each side of the edge:
    // q[i] at +i*(dx,dy), p[i] at -(i+1)*(dx,dy).
    let q: [i32; 8] = std::array::from_fn(|i| {
        let k = i as i32;
        plane.read((y + dy * k) as isize, (x + dx * k) as isize) as i32
    });
    let p: [i32; 8] = std::array::from_fn(|i| {
        let k = (i + 1) as i32;
        plane.read((y - dy * k) as isize, (x - dx * k) as isize) as i32
    });
    // §8.8.5.1 masks.
    let hev_mask =
        (p[1] - p[0]).abs() > thresh as i32 || (q[1] - q[0]).abs() > thresh as i32;
    let mut mask = false;
    mask |= (p[3] - p[2]).abs() > limit as i32;
    mask |= (p[2] - p[1]).abs() > limit as i32;
    mask |= (p[1] - p[0]).abs() > limit as i32;
    mask |= (q[1] - q[0]).abs() > limit as i32;
    mask |= (q[2] - q[1]).abs() > limit as i32;
    mask |= (q[3] - q[2]).abs() > limit as i32;
    mask |= (p[0] - q[0]).abs() * 2 + (p[1] - q[1]).abs() / 2 > blimit as i32;
    let filter_mask = !mask;
    if !filter_mask {
        return;
    }
    let flat_mask = if filter_size >= 1 {
        let mut m = false;
        m |= (p[1] - p[0]).abs() > 1;
        m |= (q[1] - q[0]).abs() > 1;
        m |= (p[2] - p[0]).abs() > 1;
        m |= (q[2] - q[0]).abs() > 1;
        m |= (p[3] - p[0]).abs() > 1;
        m |= (q[3] - q[0]).abs() > 1;
        !m
    } else {
        false
    };
    let flat_mask2 = if filter_size >= 2 {
        let mut m = false;
        m |= (p[7] - p[0]).abs() > 1;
        m |= (q[7] - q[0]).abs() > 1;
        m |= (p[6] - p[0]).abs() > 1;
        m |= (q[6] - q[0]).abs() > 1;
        m |= (p[5] - p[0]).abs() > 1;
        m |= (q[5] - q[0]).abs() > 1;
        m |= (p[4] - p[0]).abs() > 1;
        m |= (q[4] - q[0]).abs() > 1;
        !m
    } else {
        false
    };
    if filter_size == 0 || !flat_mask {
        filter4(plane, x, y, dx, dy, hev_mask);
    } else if filter_size == 1 || !flat_mask2 {
        filter_wide(plane, x, y, dx, dy, 3);
    } else {
        filter_wide(plane, x, y, dx, dy, 4);
    }
}

/// §8.8.5.2 narrow (4-tap) filter. Modifies up to two samples on each
/// side of the boundary.
fn filter4(plane: &mut PlaneMut<'_>, x: i32, y: i32, dx: i32, dy: i32, hev_mask: bool) {
    let q0 = plane.read(y as isize, x as isize) as i32;
    let q1 = plane.read((y + dy) as isize, (x + dx) as isize) as i32;
    let p0 = plane.read((y - dy) as isize, (x - dx) as isize) as i32;
    let p1 = plane.read((y - dy * 2) as isize, (x - dx * 2) as isize) as i32;
    let ps1 = p1 - 128;
    let ps0 = p0 - 128;
    let qs0 = q0 - 128;
    let qs1 = q1 - 128;
    let clamp = |v: i32| v.clamp(-128, 127);
    let filter = if hev_mask { clamp(ps1 - qs1) } else { 0 };
    let filter = clamp(filter + 3 * (qs0 - ps0));
    let filter1 = clamp(filter + 4) >> 3;
    let filter2 = clamp(filter + 3) >> 3;
    let oq0 = (clamp(qs0 - filter1) + 128) as u8;
    let op0 = (clamp(ps0 + filter2) + 128) as u8;
    plane.write(y as isize, x as isize, oq0);
    plane.write((y - dy) as isize, (x - dx) as isize, op0);
    if !hev_mask {
        // Round2(filter1, 1) — round-half-up on the signed value.
        let filter = (filter1 + 1) >> 1;
        let oq1 = (clamp(qs1 - filter) + 128) as u8;
        let op1 = (clamp(ps1 + filter) + 128) as u8;
        plane.write((y + dy) as isize, (x + dx) as isize, oq1);
        plane.write((y - dy * 2) as isize, (x - dx * 2) as isize, op1);
    }
}

/// §8.8.5.3 wide filter. `log2_size` is 3 (8-tap) or 4 (16-tap).
///
/// The spec's averaging is a boxcar over 2n+1 samples with edge
/// reflection via `Clip3(-(n+1), n, i+j)`. We output 2n samples around
/// the edge (indices `-n..n`).
fn filter_wide(plane: &mut PlaneMut<'_>, x: i32, y: i32, dx: i32, dy: i32, log2_size: u32) {
    let n = (1i32 << (log2_size - 1)) - 1;
    // Gather 2n+2 samples indexed -(n+1) .. n inclusive.
    let nn = (n + 1) as usize;
    let len = (2 * n + 2) as usize;
    let mut samples = vec![0i32; len];
    for (idx, i) in (-(n + 1)..=n).enumerate() {
        samples[idx] = plane.read((y + i * dy) as isize, (x + i * dx) as isize) as i32;
    }
    // Compute F[-n..n-1] per the spec's nested-for formula.
    let mut out = vec![0i32; (2 * n) as usize];
    for i in -n..n {
        let mut t = samples[(i + (n + 1)) as usize];
        for j in -n..=n {
            let p_clamped = (i + j).clamp(-(n + 1), n);
            t += samples[(p_clamped + (n + 1)) as usize];
        }
        out[(i + n) as usize] = (t + (1 << (log2_size - 1))) >> log2_size;
    }
    for i in -n..n {
        let val = out[(i + n) as usize].clamp(0, 255) as u8;
        plane.write((y + i * dy) as isize, (x + i * dx) as isize, val);
    }
    let _ = nn;
    let _ = &samples; // keep the mutable-unused lint quiet if any
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lvl_lookup_uniform_when_deltas_disabled() {
        let lf = LoopFilterParams {
            level: 30,
            sharpness: 0,
            mode_ref_delta_enabled: false,
            mode_ref_delta_update: false,
            ref_deltas: [0; 4],
            mode_deltas: [0; 2],
        };
        let l = LvlLookup::build(&lf);
        for s in 0..MAX_SEGMENTS as u8 {
            for r in 0..MAX_REF_FRAMES as u8 {
                for m in 0..MAX_MODE_LF_DELTAS as u8 {
                    assert_eq!(l.get(s, r, m), 30);
                }
            }
        }
    }

    #[test]
    fn lvl_lookup_applies_ref_delta() {
        let lf = LoopFilterParams {
            level: 32,
            sharpness: 0,
            mode_ref_delta_enabled: true,
            mode_ref_delta_update: true,
            ref_deltas: [1, -2, 0, 0],
            mode_deltas: [0, 1],
        };
        let l = LvlLookup::build(&lf);
        // n_shift = 32 >> 5 = 1.
        // intra: 32 + (1 << 1) = 34.
        assert_eq!(l.get(0, 0, 0), 34);
        // last frame, mode 0: 32 + (-2 << 1) + (0 << 1) = 28.
        assert_eq!(l.get(0, 1, 0), 28);
        // last frame, mode 1: 32 + (-2 << 1) + (1 << 1) = 30.
        assert_eq!(l.get(0, 1, 1), 30);
    }

    #[test]
    fn loop_filter_zero_level_is_noop() {
        let lf = LoopFilterParams::default(); // level = 0
        let f = LoopFilter::new(&lf, 2, 2, true, true);
        let info = MiInfoPlane::new(2, 2);
        let mut y = vec![100u8; 16 * 16];
        let before = y.clone();
        let mut u = vec![100u8; 8 * 8];
        let mut v = vec![100u8; 8 * 8];
        f.apply_frame(&info, &mut y, 16, 16, 16, &mut u, &mut v, 8, 8, 8);
        assert_eq!(y, before);
    }

    #[test]
    fn loop_filter_smooths_small_seam() {
        // 16x16 Y plane with a modest seam between the two 8x8 blocks
        // (jump of 10). The §8.8.5.1 filter_mask test requires the jump
        // to stay below blimit, so we pick parameters where the filter
        // fires. After the filter the seam discontinuity must shrink.
        let w = 16;
        let h = 16;
        let mut y = vec![0u8; w * h];
        for r in 0..h {
            for c in 0..w {
                y[r * w + c] = if c < 8 { 120 } else { 130 };
            }
        }
        let lf = LoopFilterParams {
            level: 32,
            sharpness: 0,
            mode_ref_delta_enabled: false,
            mode_ref_delta_update: false,
            ref_deltas: [0; 4],
            mode_deltas: [0; 2],
        };
        let mi_cols = 2;
        let mi_rows = 2;
        let mut info = MiInfoPlane::new(mi_cols, mi_rows);
        for mi_row in 0..mi_rows {
            for mi_col in 0..mi_cols {
                info.fill(
                    mi_row,
                    mi_col,
                    MiInfo {
                        mi_w_8x8: 1,
                        mi_h_8x8: 1,
                        tx_size_log2: 1,
                        skip: false,
                        ref_frame: 1,
                        mode_is_non_zero_inter: true,
                        segment_id: 0,
                    },
                );
            }
        }
        let f = LoopFilter::new(&lf, mi_cols, mi_rows, false, false);
        let mut u = vec![128u8; w * h];
        let mut v = vec![128u8; w * h];
        let jump_before: i32 = (y[8 * w + 8] as i32 - y[8 * w + 7] as i32).abs();
        f.apply_frame(&info, &mut y, w, w, h, &mut u, &mut v, w, w, h);
        let jump_after: i32 = (y[8 * w + 8] as i32 - y[8 * w + 7] as i32).abs();
        assert!(
            jump_after < jump_before,
            "loop filter should reduce the seam (before={jump_before}, after={jump_after})"
        );
    }
}
