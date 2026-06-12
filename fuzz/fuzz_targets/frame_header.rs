//! Panic-surface fuzz for the §6.2 uncompressed-header walker and the
//! frame-walk that hangs off it: when the header parses, the harness
//! also slices out the §6.3 compressed-header payload (driving the
//! §9.2 Boolean decoder with the header-derived `Lossless` flag) and
//! walks the §6.4 tile-size prefix chain over the remaining bytes.
//!
//! No oracle — arbitrary bytes in, "no panic / no overflow / no OOM"
//! out. Every return value is intentionally discarded; `Err` is a
//! correct answer for garbage input.

#![no_main]

use libfuzzer_sys::fuzz_target;
use oxideav_vp9::{parse_compressed_header, parse_uncompressed_header, tile_payload_sizes};

fuzz_target!(|data: &[u8]| {
    let Ok(h) = parse_uncompressed_header(data) else {
        return;
    };

    // Frame walk per §6.1: the compressed header occupies the
    // `header_size_in_bytes` bytes that follow the byte-aligned
    // uncompressed header; the tile payloads follow that.
    let ch_start = h.uncompressed_header_size_bytes;
    let ch_end = ch_start.saturating_add(h.header_size_in_bytes as usize);
    if h.header_size_in_bytes > 0 && ch_end <= data.len() {
        let _ = parse_compressed_header(&data[ch_start..ch_end], h.quantization.lossless);
    }

    let tiles = &data[ch_end.min(data.len())..];
    let _ = tile_payload_sizes(
        tiles,
        tiles.len() as u32,
        h.tile_info.tile_rows_log2,
        h.tile_info.tile_cols_log2,
    );
});
