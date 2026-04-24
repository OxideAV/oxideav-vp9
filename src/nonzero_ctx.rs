//! Per-frame `AboveNonzeroContext` / `LeftNonzeroContext` tracking
//! (§6.4.22.3 + §6.4.24 tokens). Each entry is a 0/1 flag stored at the
//! 4×4-pixel granularity — 1 when the previously-decoded TX block at
//! that position produced at least one non-zero coefficient.
//!
//! Spec §6.4.24 `more_coefs` context for the first scan position:
//! ```text
//! x4 = startX >> 2
//! y4 = startY >> 2
//! numpts = 1 << txSz
//! above = 0 ; left = 0
//! for i in 0..numpts:
//!     if x4 + i < maxX: above |= AboveNonzeroContext[plane][x4 + i]
//!     if y4 + i < maxY: left  |= LeftNonzeroContext[plane][y4 + i]
//! ctx = above + left
//! ```
//!
//! After the block is detokenised:
//! ```text
//! nonzero = (eob > 0) ? 1 : 0
//! for i in 0..step:
//!     AboveNonzeroContext[plane][(startX >> 2) + i] = nonzero
//!     LeftNonzeroContext [plane][(startY >> 2) + i] = nonzero
//! ```
//!
//! §7.4.1 `clear_above_context` zeroes every Above array at tile start.
//! §7.4.2 `clear_left_context` zeroes every Left array at each
//! superblock row within a tile.

/// Above / Left nonzero-coef tracking arrays. One array per plane
/// (Y, U, V). Arrays are sized in units of 4×4 pixels.
#[derive(Clone, Debug)]
pub struct NonzeroCtx {
    /// `above[plane][x4]` — 0/1 flag per 4×4 pixel column in the plane.
    pub above: [Vec<u8>; 3],
    /// `left[plane][y4]` — 0/1 flag per 4×4 pixel row in the plane.
    pub left: [Vec<u8>; 3],
    /// Per-plane column count bound (maxX) for the `above` scan.
    pub max_x: [usize; 3],
    /// Per-plane row count bound (maxY) for the `left` scan.
    pub max_y: [usize; 3],
}

impl NonzeroCtx {
    /// Allocate for a frame with `mi_cols` × `mi_rows` MI grid and the
    /// supplied chroma subsampling (0 = no subsampling, 1 = half).
    pub fn new(mi_cols: usize, mi_rows: usize, subsampling_x: usize, subsampling_y: usize) -> Self {
        // maxX / maxY per spec = (2 * MiCols) >> sx (units of 4×4 pixels).
        let y_x = 2 * mi_cols;
        let y_y = 2 * mi_rows;
        let uv_x = (2 * mi_cols) >> subsampling_x;
        let uv_y = (2 * mi_rows) >> subsampling_y;
        // Allocate luma-sized for all three; the max bounds drive the
        // scan, so extra trailing bytes are harmless.
        let len_x = y_x.max(1);
        let len_y = y_y.max(1);
        Self {
            above: [vec![0u8; len_x], vec![0u8; len_x], vec![0u8; len_x]],
            left: [vec![0u8; len_y], vec![0u8; len_y], vec![0u8; len_y]],
            max_x: [y_x, uv_x, uv_x],
            max_y: [y_y, uv_y, uv_y],
        }
    }

    /// §7.4.1 `clear_above_context` — zero every plane's Above array.
    /// Called once at the start of each tile.
    pub fn clear_above(&mut self) {
        for plane in 0..3 {
            for b in self.above[plane].iter_mut() {
                *b = 0;
            }
        }
    }

    /// §7.4.2 `clear_left_context` — zero every plane's Left array.
    /// Called at the top of every superblock row within a tile.
    pub fn clear_left(&mut self) {
        for plane in 0..3 {
            for b in self.left[plane].iter_mut() {
                *b = 0;
            }
        }
    }

    /// Compute the `more_coefs` first-position context for a single TX
    /// block. `start_x` / `start_y` are the plane-pixel coordinates of
    /// the block's top-left corner; `tx_size_log2` is the TX size
    /// (0..=3 for 4×4..32×32).
    ///
    /// Returns ctx in {0, 1, 2} — the sum of (bitwise OR over the
    /// Above slice) + (bitwise OR over the Left slice).
    pub fn token_ctx(
        &self,
        plane: usize,
        start_x: usize,
        start_y: usize,
        tx_size_log2: usize,
    ) -> usize {
        let x4 = start_x >> 2;
        let y4 = start_y >> 2;
        let numpts = 1usize << tx_size_log2;
        let max_x = self.max_x[plane];
        let max_y = self.max_y[plane];
        let mut above = 0u8;
        let mut left = 0u8;
        for i in 0..numpts {
            let xi = x4 + i;
            if xi < max_x && xi < self.above[plane].len() {
                above |= self.above[plane][xi];
            }
            let yi = y4 + i;
            if yi < max_y && yi < self.left[plane].len() {
                left |= self.left[plane][yi];
            }
        }
        (above as usize) + (left as usize)
    }

    /// Update both Above and Left arrays after a TX block has been
    /// detokenised. `nonzero` is 1 when at least one coefficient was
    /// decoded (eob > 0), 0 otherwise.
    pub fn update(
        &mut self,
        plane: usize,
        start_x: usize,
        start_y: usize,
        tx_size_log2: usize,
        nonzero: u8,
    ) {
        let x4 = start_x >> 2;
        let y4 = start_y >> 2;
        let step = 1usize << tx_size_log2;
        let above = &mut self.above[plane];
        for i in 0..step {
            let xi = x4 + i;
            if xi < above.len() {
                above[xi] = nonzero;
            }
        }
        let left = &mut self.left[plane];
        for i in 0..step {
            let yi = y4 + i;
            if yi < left.len() {
                left[yi] = nonzero;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clear_above_zeroes() {
        let mut ctx = NonzeroCtx::new(8, 8, 1, 1);
        ctx.update(0, 0, 0, 0, 1);
        assert_eq!(ctx.above[0][0], 1);
        ctx.clear_above();
        assert_eq!(ctx.above[0][0], 0);
    }

    #[test]
    fn token_ctx_sums_above_and_left() {
        let mut ctx = NonzeroCtx::new(8, 8, 0, 0);
        // Above at x4=0..=0 = 1, Left at y4=0 = 1 → ctx = 2
        ctx.update(0, 0, 0, 0, 1);
        let c = ctx.token_ctx(0, 0, 0, 0);
        assert_eq!(c, 2);
    }

    #[test]
    fn token_ctx_returns_zero_when_clean() {
        let ctx = NonzeroCtx::new(8, 8, 0, 0);
        assert_eq!(ctx.token_ctx(0, 0, 0, 0), 0);
        assert_eq!(ctx.token_ctx(1, 4, 4, 1), 0);
    }
}
