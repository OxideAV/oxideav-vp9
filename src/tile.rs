//! VP9 tile / partition / block decode.
//!
//! Reference: VP9 spec §6.4 (`decode_tiles`), §6.4.1 (`decode_tile`),
//! §6.4.2 (`decode_partition`), §6.4.3 (`decode_block`), §7.4 (block-level
//! semantic process), §8.5 (intra prediction), §8.6 (inter prediction),
//! §8.7 (reconstruction), §8.8 (loop filter).
//!
//! Status (this revision): the partition quadtree (§6.4.2) is implemented
//! against the keyframe default partition probabilities (§10.5). Given a
//! bool-engine cursor and a superblock origin, the decoder recurses down
//! to 8×8 and records each leaf's size + position in a flat
//! [`PartitionPlan`]. This is the plumbing the next milestone —
//! per-block mode / coefficient decode (§6.4.3) — hangs off.
//!
//! Stops at the first `decode_block` call: that's where
//! `kf_intra_mode_probs` (§10.5), coefficient tree probs (§10.5 again)
//! and residual dequant (§8.6) would kick in — several thousand lines
//! of tables and arithmetic. The error carries a precise §ref so callers
//! know exactly how far the decoder reached.
//!
//! The intra primitives and inverse transforms (`crate::intra`,
//! `crate::transform`) are available standalone.

use oxideav_core::{Error, Result};

use crate::bool_decoder::BoolDecoder;
use crate::compressed_header::CompressedHeader;
use crate::headers::UncompressedHeader;
use crate::probs::{read_partition_from_tree, PartitionType, KF_PARTITION_PROBS, PARTITION_PROBS};

/// VP9 superblock size is always 64×64 (§3).
pub const SUPERBLOCK_SIZE: u32 = 64;

/// Per-superblock partition decisions — §6.4.2 Table 9-5.
///
/// The `PARTITION_NONE` / `PARTITION_HORZ` / `PARTITION_VERT` /
/// `PARTITION_SPLIT` enumeration is reused at every level of the partition
/// quadtree (§6.4.2). `Partition` is kept as a public alias of the
/// internal `PartitionType` so downstream consumers can reference it.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Partition {
    None = 0,
    Horz = 1,
    Vert = 2,
    Split = 3,
}

impl Partition {
    pub fn from_u32(v: u32) -> Result<Self> {
        Ok(match v {
            0 => Self::None,
            1 => Self::Horz,
            2 => Self::Vert,
            3 => Self::Split,
            _ => return Err(Error::invalid(format!("vp9 partition: invalid {v}"))),
        })
    }

    fn from_ptype(p: PartitionType) -> Self {
        match p {
            PartitionType::None => Self::None,
            PartitionType::Horz => Self::Horz,
            PartitionType::Vert => Self::Vert,
            PartitionType::Split => Self::Split,
        }
    }
}

/// Tile-grid geometry derived from the uncompressed header's tile_info
/// field (§6.2.6).
#[derive(Clone, Copy, Debug)]
pub struct TileGrid {
    pub tile_cols: u32,
    pub tile_rows: u32,
    /// Total frame width in 8-pixel mi units (`MiCols` in spec parlance).
    pub mi_cols: u32,
    /// Total frame height in 8-pixel mi units (`MiRows`).
    pub mi_rows: u32,
    /// Total superblocks in the frame.
    pub sbs_x: u32,
    pub sbs_y: u32,
}

impl TileGrid {
    pub fn from_header(hdr: &UncompressedHeader) -> Self {
        let tile_cols = 1u32 << hdr.tile_info.log2_tile_cols as u32;
        let tile_rows = 1u32 << hdr.tile_info.log2_tile_rows as u32;
        // Spec §7.2 — MiCols = ALIGN(width, 8) / 8, MiRows = ALIGN(height, 8) / 8.
        let mi_cols = hdr.width.div_ceil(8);
        let mi_rows = hdr.height.div_ceil(8);
        let sbs_x = hdr.width.div_ceil(SUPERBLOCK_SIZE);
        let sbs_y = hdr.height.div_ceil(SUPERBLOCK_SIZE);
        Self {
            tile_cols,
            tile_rows,
            mi_cols,
            mi_rows,
            sbs_x,
            sbs_y,
        }
    }
}

/// One leaf of the partition quadtree — a block we would feed to
/// `decode_block` (§6.4.3) if block-level decode were implemented.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PartitionLeaf {
    /// Pixel column of the block's top-left corner within the tile.
    pub col: u32,
    /// Pixel row of the block's top-left corner within the tile.
    pub row: u32,
    /// Block width in pixels (power of two, 8..=64).
    pub width: u32,
    /// Block height in pixels.
    pub height: u32,
    /// Partition type that produced this leaf.
    pub kind: Partition,
}

/// Full partition plan for a superblock — the ordered list of leaves
/// `decode_block` would have been invoked on, in VP9's raster-scan order.
#[derive(Clone, Debug, Default)]
pub struct PartitionPlan {
    pub leaves: Vec<PartitionLeaf>,
}

impl PartitionPlan {
    fn push(&mut self, l: PartitionLeaf) {
        self.leaves.push(l);
    }
}

/// A single-tile decode context. `TileDecoder::new` establishes the bool
/// engine over `tile_data`; `walk_partitions` then runs the full
/// superblock / partition-tree traversal of §6.4.2. Block-level decode
/// (§6.4.3) is not implemented — the walk returns the plan instead.
pub struct TileDecoder<'a> {
    pub hdr: &'a UncompressedHeader,
    pub ch: &'a CompressedHeader,
    pub bool_dec: BoolDecoder<'a>,
    /// Tile's column index within the tile grid.
    pub tile_col: u32,
    /// Tile's row index within the tile grid.
    pub tile_row: u32,
}

impl<'a> TileDecoder<'a> {
    /// Begin decoding a tile whose compressed payload is `tile_data`.
    pub fn new(
        hdr: &'a UncompressedHeader,
        ch: &'a CompressedHeader,
        tile_data: &'a [u8],
        tile_col: u32,
        tile_row: u32,
    ) -> Result<Self> {
        let bool_dec = BoolDecoder::new(tile_data)?;
        Ok(Self {
            hdr,
            ch,
            bool_dec,
            tile_col,
            tile_row,
        })
    }

    /// Walk the partition quadtree for every superblock in this tile,
    /// returning the flat list of leaves the §6.4.3 block decoder would
    /// have run against. The bool-engine cursor ends up pointing at the
    /// first unread bit — the one `decode_block` would consume.
    ///
    /// This intentionally does NOT invoke `decode_block`: we don't yet
    /// have the §10.5 mode / coefficient probabilities wired up, so any
    /// further progress would be guesswork. Callers wanting a terminal
    /// `Error::Unsupported` should use [`Self::decode`].
    pub fn walk_partitions(&mut self, tile_width: u32, tile_height: u32) -> Result<PartitionPlan> {
        let mut plan = PartitionPlan::default();
        let keyframe =
            matches!(self.hdr.frame_type, crate::headers::FrameType::Key,) || self.hdr.intra_only;
        let sbs_x = tile_width.div_ceil(SUPERBLOCK_SIZE);
        let sbs_y = tile_height.div_ceil(SUPERBLOCK_SIZE);
        for sby in 0..sbs_y {
            for sbx in 0..sbs_x {
                let col = sbx * SUPERBLOCK_SIZE;
                let row = sby * SUPERBLOCK_SIZE;
                decode_partition_recursive(
                    &mut self.bool_dec,
                    row,
                    col,
                    SUPERBLOCK_SIZE,
                    tile_width,
                    tile_height,
                    keyframe,
                    &mut plan,
                )?;
            }
        }
        Ok(plan)
    }

    /// Walk the partition tree, then surface `Error::Unsupported` at the
    /// first `decode_block` call site (§6.4.3). This is the recommended
    /// entry point for higher-level code — the walk fully exercises
    /// [`read_partition_from_tree`] against real probabilities, and the
    /// error message carries the exact stopping clause.
    pub fn decode(&mut self) -> Result<()> {
        let w = self.hdr.width;
        let h = self.hdr.height;
        let plan = self.walk_partitions(w, h)?;
        Err(Error::unsupported(format!(
            "vp9 decode_block §6.4.3 not implemented \
             (tile={},{}; partition tree walked OK, {} leaves planned). \
             Next: kf_intra_mode_probs (§10.5), coef tree + probs \
             (§10.5), dequant (§8.6), clip-add reconstruction (§8.7).",
            self.tile_col,
            self.tile_row,
            plan.leaves.len(),
        )))
    }
}

/// Walk the tile / partition / block tree per §6.4. The compressed header
/// must already have been parsed by the caller. For keyframe / intra_only
/// frames this drives the full [`crate::block::IntraTile`] pipeline and
/// returns `Ok(())` on success. Inter frames surface `Unsupported`.
///
/// Multi-tile (`log2_tile_{rows,cols} > 0`) also surfaces `Unsupported`
/// for the moment: the tile-size-prefix syntax (§6.4) is simple to add
/// but brings no new coverage until the fixture exercises it.
pub fn decode_tiles(
    tile_payload: &[u8],
    hdr: &UncompressedHeader,
    ch: &CompressedHeader,
) -> Result<()> {
    let grid = TileGrid::from_header(hdr);
    if grid.sbs_x == 0 || grid.sbs_y == 0 {
        return Err(Error::invalid(
            "vp9 decode_tiles: zero-sized frame — impossible per §6.2.2",
        ));
    }
    if tile_payload.is_empty() {
        return Err(Error::invalid(
            "vp9 decode_tiles: tile payload empty — §6.4",
        ));
    }
    if hdr.tile_info.log2_tile_cols != 0 || hdr.tile_info.log2_tile_rows != 0 {
        return Err(Error::unsupported(
            "vp9 multi-tile frame — single-tile only for now",
        ));
    }
    let is_intra = matches!(hdr.frame_type, crate::headers::FrameType::Key) || hdr.intra_only;
    if !is_intra {
        return Err(Error::unsupported("vp9 inter frame pending"));
    }
    let mut tile = crate::block::IntraTile::new(hdr, ch);
    let mut bd = BoolDecoder::new(tile_payload)?;
    tile.decode(&mut bd)
}

/// Public recursive-descent entry — decode one partition node at (row,
/// col) of the current superblock. The caller owns the bool-engine
/// cursor; this function advances it. On exit either `plan` has gained
/// the leaves the node expanded into, or an error is returned.
///
/// `bsize` is the block's side in pixels (64 → top of a SB, 8 → smallest
/// VP9 block). `frame_w` and `frame_h` clip against the frame edge —
/// when a block spills off the bottom or right edge VP9 implicitly
/// forces SPLIT per §6.4.2.
#[allow(clippy::too_many_arguments)]
pub fn decode_partition_recursive(
    bd: &mut BoolDecoder<'_>,
    row: u32,
    col: u32,
    bsize: u32,
    frame_w: u32,
    frame_h: u32,
    keyframe: bool,
    plan: &mut PartitionPlan,
) -> Result<()> {
    debug_assert!(matches!(bsize, 64 | 32 | 16 | 8));
    if row >= frame_h || col >= frame_w {
        return Ok(()); // fully outside frame — no partition symbol.
    }
    let partition = if bsize == 8 {
        // At the 8×8 level §6.4.2 removes SPLIT from the tree — the only
        // valid outcomes are NONE / HORZ / VERT. We still consume the
        // two bits the tree shape dictates.
        read_partition_8x8(bd, partition_probs(bsize, keyframe))?
    } else {
        let on_right = col + bsize > frame_w;
        let on_bottom = row + bsize > frame_h;
        if on_right && on_bottom {
            // Both edges cross — SPLIT is forced, no probability bit is
            // read (§6.4.2 last paragraph).
            Partition::Split
        } else if on_right {
            // Right edge — either VERT or SPLIT. One bit with probs[2].
            let bit = bd.read(partition_probs(bsize, keyframe)[2])?;
            if bit == 0 {
                Partition::Vert
            } else {
                Partition::Split
            }
        } else if on_bottom {
            // Bottom edge — either HORZ or SPLIT. One bit with probs[1].
            let bit = bd.read(partition_probs(bsize, keyframe)[1])?;
            if bit == 0 {
                Partition::Horz
            } else {
                Partition::Split
            }
        } else {
            let probs = partition_probs(bsize, keyframe);
            Partition::from_ptype(read_partition_from_tree(bd, probs)?)
        }
    };
    let half = bsize / 2;
    match partition {
        Partition::None => {
            plan.push(PartitionLeaf {
                col,
                row,
                width: bsize,
                height: bsize,
                kind: partition,
            });
        }
        Partition::Horz => {
            plan.push(PartitionLeaf {
                col,
                row,
                width: bsize,
                height: half,
                kind: partition,
            });
            if row + half < frame_h {
                plan.push(PartitionLeaf {
                    col,
                    row: row + half,
                    width: bsize,
                    height: half,
                    kind: partition,
                });
            }
        }
        Partition::Vert => {
            plan.push(PartitionLeaf {
                col,
                row,
                width: half,
                height: bsize,
                kind: partition,
            });
            if col + half < frame_w {
                plan.push(PartitionLeaf {
                    col: col + half,
                    row,
                    width: half,
                    height: bsize,
                    kind: partition,
                });
            }
        }
        Partition::Split => {
            if bsize == 8 {
                return Err(Error::invalid("vp9 §6.4.2: PARTITION_SPLIT invalid at 8×8"));
            }
            for (dr, dc) in [(0, 0), (0, half), (half, 0), (half, half)] {
                decode_partition_recursive(
                    bd,
                    row + dr,
                    col + dc,
                    half,
                    frame_w,
                    frame_h,
                    keyframe,
                    plan,
                )?;
            }
        }
    }
    Ok(())
}

/// Back-compat stub; new code should use [`decode_partition_recursive`].
pub fn decode_partition(bd: &mut BoolDecoder<'_>, row: u32, col: u32, sb_size: u32) -> Result<()> {
    let mut plan = PartitionPlan::default();
    decode_partition_recursive(bd, row, col, sb_size, u32::MAX, u32::MAX, true, &mut plan)
}

/// Decode one block per §6.4.3 — not implemented. Surfaces the exact
/// stopping point so callers can report where the decoder gave up.
pub fn decode_block(_bd: &mut BoolDecoder<'_>, _row: u32, _col: u32, _bsize: u32) -> Result<()> {
    Err(Error::unsupported(
        "vp9 §6.4.3 decode_block: block decode (residual + prediction + \
         reconstruction) not implemented — needs §10.5 kf_intra_mode_probs, \
         coef tree probs, dequant tables (§8.6), clip-add (§8.7)",
    ))
}

/// 8×8 partition decode (§6.4.2): SPLIT is forbidden, so the tree is
/// pruned to `{NONE, HORZ, VERT}`. The boolean engine reads `probs[0]`
/// then — on "not NONE" — `probs[1]` to pick HORZ vs VERT.
fn read_partition_8x8(bd: &mut BoolDecoder<'_>, probs: [u8; 3]) -> Result<Partition> {
    let b0 = bd.read(probs[0])?;
    if b0 == 0 {
        return Ok(Partition::None);
    }
    let b1 = bd.read(probs[1])?;
    Ok(if b1 == 0 {
        Partition::Horz
    } else {
        Partition::Vert
    })
}

/// Look up the `[3]` probability row for a given block size. `bsize=64`
/// maps to bsl=0, `bsize=8` maps to bsl=3. The partition context is not
/// yet derived from above / left neighbours — we use context 0 for now,
/// which matches `decode_partition` when no decoded neighbours exist
/// (the top-left superblock). This is one of the known TODOs listed in
/// the crate README.
fn partition_probs(bsize: u32, keyframe: bool) -> [u8; 3] {
    // Match block::read_partition — KF_PARTITION_PROBS / PARTITION_PROBS
    // are laid out 64×64-first, so invert the usual bsl.
    let bsl = match bsize {
        64 => 0usize,
        32 => 1,
        16 => 2,
        8 => 3,
        _ => 0,
    };
    // Context 0 always; correct multi-SB decoding needs §6.4.2's
    // `partition_plane_context`, which requires tracking decoded sizes
    // across the whole frame — deferred until block decode lands.
    let ctx = 0usize;
    let idx = ctx + bsl * 4;
    if keyframe {
        KF_PARTITION_PROBS[idx]
    } else {
        PARTITION_PROBS[idx]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::headers::{
        ColorConfig, ColorSpace, FrameType, LoopFilterParams, QuantizationParams,
        SegmentationParams, TileInfo, UncompressedHeader,
    };

    fn synth_header(width: u32, height: u32) -> UncompressedHeader {
        UncompressedHeader {
            profile: 0,
            show_existing_frame: false,
            existing_frame_to_show: 0,
            frame_type: FrameType::Key,
            show_frame: true,
            error_resilient_mode: false,
            intra_only: false,
            reset_frame_context: 0,
            color_config: ColorConfig {
                bit_depth: 8,
                color_space: ColorSpace::Bt709,
                color_range: false,
                subsampling_x: true,
                subsampling_y: true,
            },
            width,
            height,
            render_width: None,
            render_height: None,
            refresh_frame_flags: 0,
            ref_frame_idx: [0; 3],
            ref_frame_sign_bias: [false; 4],
            allow_high_precision_mv: false,
            interpolation_filter: 0,
            refresh_frame_context: false,
            frame_parallel_decoding_mode: false,
            frame_context_idx: 0,
            loop_filter: LoopFilterParams::default(),
            quantization: QuantizationParams::default(),
            segmentation: SegmentationParams::default(),
            tile_info: TileInfo {
                log2_tile_cols: 0,
                log2_tile_rows: 0,
            },
            header_size: 0,
            uncompressed_header_size: 0,
        }
    }

    #[test]
    fn tile_grid_64x64_is_one_superblock() {
        let h = synth_header(64, 64);
        let g = TileGrid::from_header(&h);
        assert_eq!(g.tile_cols, 1);
        assert_eq!(g.tile_rows, 1);
        assert_eq!(g.sbs_x, 1);
        assert_eq!(g.sbs_y, 1);
        assert_eq!(g.mi_cols, 8);
        assert_eq!(g.mi_rows, 8);
    }

    #[test]
    fn tile_grid_128x96_rounds_up() {
        let h = synth_header(128, 96);
        let g = TileGrid::from_header(&h);
        assert_eq!(g.sbs_x, 2);
        assert_eq!(g.sbs_y, 2);
    }

    #[test]
    fn decode_tiles_runs_to_completion_on_synth_keyframe() {
        // The tile walker + block decoder will happily chew through an
        // arbitrary byte stream as long as it doesn't run out of bits:
        // probabilities never reject, they just bias. So any non-empty
        // payload paired with a valid single-tile keyframe header should
        // return `Ok(())` with a reconstructed (garbage-looking) plane.
        let h = synth_header(64, 64);
        let ch = CompressedHeader::default();
        let payload = [0xAB, 0x00, 0xCD, 0xEF, 0x12, 0x34, 0x56, 0x78];
        decode_tiles(&payload, &h, &ch).expect("decode completes");
    }

    #[test]
    fn walk_partitions_all_zeros_gives_single_64x64_leaf() {
        // With p[0]=158 (high), a run of zero bits from a flat bool-decoder
        // input tends to yield "not SPLIT, not HORZ, not VERT" => NONE
        // at the top level, producing a single 64×64 leaf.
        let h = synth_header(64, 64);
        let ch = CompressedHeader::default();
        // Bool decoder needs a non-zero initial 8 bits so the window is
        // non-trivial; marker bit at position 9 must be zero.
        let payload = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let mut td = TileDecoder::new(&h, &ch, &payload, 0, 0).unwrap();
        let plan = td.walk_partitions(64, 64).unwrap();
        assert_eq!(plan.leaves.len(), 1, "expected PARTITION_NONE leaf");
        let leaf = plan.leaves[0];
        assert_eq!(leaf.col, 0);
        assert_eq!(leaf.row, 0);
        assert_eq!(leaf.width, 64);
        assert_eq!(leaf.height, 64);
        assert_eq!(leaf.kind, Partition::None);
    }

    #[test]
    fn partition_from_u32_accepts_valid() {
        for v in 0u32..=3 {
            Partition::from_u32(v).unwrap();
        }
        assert!(Partition::from_u32(4).is_err());
    }
}
