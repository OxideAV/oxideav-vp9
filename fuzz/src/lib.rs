//! Runtime libavcodec interop for the VP9 cross-decode fuzz oracle.
//!
//! libavcodec is loaded via `dlopen` at first call — there is no
//! `ffmpeg-sys` / `rsmpeg`-style build-script dep that would pull
//! libavcodec or libvpx source into the workspace's cargo dep tree.
//! The harness checks `available()` up front and `eprintln!`s a
//! `[oracle skip]` line + early-returns when the shared library isn't
//! installed, so fuzz binaries built on a host without ffmpeg simply
//! do nothing instead of panicking. **No `#[ignore]`.**
//!
//! Workspace policy bars consulting libavcodec / libvpx source; we only
//! inspect the public C headers (`<libavcodec/avcodec.h>`,
//! `<libavcodec/packet.h>`) for function signatures + opaque struct
//! pointer typedefs.
//!
//! Install on Debian/Ubuntu via `apt-get install -y ffmpeg` (pulls
//! libavcodec.so.* as a transitive dep) or, more directly,
//! `libavcodec-dev`. On macOS use `brew install ffmpeg`. The loader
//! probes the conventional shared-object names for both platforms,
//! starting at the newest known SONAME (`libavcodec.so.62` is FFmpeg
//! 8.x; `.so.61` is 7.x; `.so.60` is 6.x; `.so` is the dev symlink).
//!
//! ## AVPacket layout
//!
//! `AVPacket` has a documented public layout (see <libavcodec/packet.h>);
//! the prefix has been stable from libavcodec 58 (FFmpeg 4.x) through
//! 62 (FFmpeg 8.x):
//!
//! ```text
//!   off  0  AVBufferRef* buf
//!   off  8  int64_t      pts
//!   off 16  int64_t      dts
//!   off 24  uint8_t*     data
//!   off 32  int          size
//!   off 36  int          stream_index
//!   off 40  int          flags
//! ```
//!
//! We use the official `av_new_packet(pkt, size)` to allocate the
//! managed buffer, then read `pkt->data` from offset 24 to memcpy our
//! input bytes in. `pkt->size` is already set by `av_new_packet`.
//!
//! ## AVFrame layout
//!
//! Likewise the AVFrame prefix is stable across libavcodec 58-62:
//!
//! ```text
//!   off   0  uint8_t*  data[AV_NUM_DATA_POINTERS=8]   (8 * 8 = 64 B)
//!   off  64  int       linesize[8]                    (8 * 4 = 32 B)
//!   off  96  uint8_t** extended_data
//!   off 104  int       width
//!   off 108  int       height
//!   off 112  int       nb_samples
//!   off 116  int       format
//! ```
//!
//! We only read `data[0..3]`, `linesize[0..3]`, `width`, `height`, and
//! `format`.

#![allow(unsafe_code)]

pub mod libavcodec {
    use libloading::{Library, Symbol};
    use std::ffi::c_void;
    use std::sync::OnceLock;

    /// Conventional libavcodec shared-object names, newest first. We
    /// fall back to the unversioned `.so` / `.dylib` symlink (only
    /// present when `libavcodec-dev` is installed). On Windows the
    /// import library is `avcodec-NN.dll` (NN is the SONAME); we list
    /// the unversioned form as a courtesy.
    const CANDIDATES: &[&str] = &[
        "libavcodec.so.62",
        "libavcodec.so.61",
        "libavcodec.so.60",
        "libavcodec.so.59",
        "libavcodec.so.58",
        "libavcodec.so",
        "libavcodec.dylib",
        "libavcodec.62.dylib",
        "libavcodec.61.dylib",
        "libavcodec.60.dylib",
        "libavcodec.59.dylib",
        "libavcodec.58.dylib",
        "avcodec.dll",
    ];

    fn lib() -> Option<&'static Library> {
        static LIB: OnceLock<Option<Library>> = OnceLock::new();
        LIB.get_or_init(|| {
            for name in CANDIDATES {
                // SAFETY: `Library::new` is documented as unsafe because
                // the loaded library may run code at load time. We
                // accept that risk for fuzz tooling — libavcodec is a
                // well-behaved shared library.
                if let Ok(l) = unsafe { Library::new(name) } {
                    return Some(l);
                }
            }
            None
        })
        .as_ref()
    }

    /// True iff a libavcodec shared library was successfully loaded.
    /// Cross-decode fuzz harnesses early-return when this is false so
    /// the binary still runs without an oracle (the assertions just
    /// don't fire).
    pub fn available() -> bool {
        lib().is_some()
    }

    /// AVCodecID enum value for VP9 — stable across libavcodec
    /// 58-62. The enum is sparse; pinning the literal value avoids
    /// reading the public header at runtime.
    pub const AV_CODEC_ID_VP9: i32 = 167;

    // AVPacket prefix offsets (see module docstring).
    const PKT_OFF_DATA: usize = 24;
    const PKT_OFF_SIZE: usize = 32;

    // AVFrame prefix offsets (see module docstring).
    const FRM_OFF_DATA: usize = 0;
    const FRM_OFF_LINESIZE: usize = 64;
    const FRM_OFF_WIDTH: usize = 104;
    const FRM_OFF_HEIGHT: usize = 108;
    const FRM_OFF_FORMAT: usize = 116;

    /// AV_PIX_FMT_YUV420P enum constant per <libavutil/pixfmt.h>.
    /// Stable since libavutil 1.x.
    pub const AV_PIX_FMT_YUV420P: i32 = 0;
    pub const AV_PIX_FMT_YUV422P: i32 = 4;
    pub const AV_PIX_FMT_YUV444P: i32 = 5;
    pub const AV_PIX_FMT_YUV420P10LE: i32 = 64;
    pub const AV_PIX_FMT_YUV422P10LE: i32 = 66;
    pub const AV_PIX_FMT_YUV444P10LE: i32 = 68;
    pub const AV_PIX_FMT_YUV420P12LE: i32 = 124;
    pub const AV_PIX_FMT_YUV422P12LE: i32 = 122;
    pub const AV_PIX_FMT_YUV444P12LE: i32 = 120;

    /// One decoded YUV frame as exported by libavcodec. Pixel buffers
    /// are tightly-packed copies of the corresponding planes (the
    /// libavcodec linesize stride is dropped during the copy).
    pub struct DecodedFrame {
        pub width: u32,
        pub height: u32,
        /// Pixel format value as read from `AVFrame.format`. See the
        /// `AV_PIX_FMT_*` constants above; we only export YUV planar
        /// 8/10/12-bit variants.
        pub pix_fmt: i32,
        /// Y plane (width * height bytes for 8-bit, 2x for 10/12-bit LE).
        pub y: Vec<u8>,
        /// Cb plane (chroma_w * chroma_h bytes).
        pub u: Vec<u8>,
        /// Cr plane.
        pub v: Vec<u8>,
        /// (chroma_w, chroma_h) in samples — derived from pix_fmt.
        pub chroma_dims: (u32, u32),
        /// 1 for 8-bit pix fmts, 2 for 10/12-bit LE pix fmts. Used to
        /// size each sample.
        pub bytes_per_sample: u32,
    }

    /// Map an AV pix_fmt to (chroma_w_shift, chroma_h_shift, bytes_per_sample).
    /// Returns None for formats we don't compare.
    fn pix_fmt_geom(pix_fmt: i32) -> Option<(u32, u32, u32)> {
        match pix_fmt {
            // 8-bit
            AV_PIX_FMT_YUV420P => Some((1, 1, 1)),
            AV_PIX_FMT_YUV422P => Some((1, 0, 1)),
            AV_PIX_FMT_YUV444P => Some((0, 0, 1)),
            // 10-bit LE
            AV_PIX_FMT_YUV420P10LE => Some((1, 1, 2)),
            AV_PIX_FMT_YUV422P10LE => Some((1, 0, 2)),
            AV_PIX_FMT_YUV444P10LE => Some((0, 0, 2)),
            // 12-bit LE
            AV_PIX_FMT_YUV420P12LE => Some((1, 1, 2)),
            AV_PIX_FMT_YUV422P12LE => Some((1, 0, 2)),
            AV_PIX_FMT_YUV444P12LE => Some((0, 0, 2)),
            _ => None,
        }
    }

    /// AVERROR(EAGAIN) — libavcodec's "no output yet, send more input"
    /// signal. Per AVERROR macro in <libavutil/error.h>, AVERROR(e) is
    /// `-e` on POSIX systems, so AVERROR(EAGAIN) is -11 on Linux/glibc.
    /// We compare absolute value to be portable across the macOS
    /// EAGAIN=35 and Linux EAGAIN=11 split.
    fn is_eagain(rc: i32) -> bool {
        let mag = rc.unsigned_abs();
        // Cover both Linux (11) and macOS (35); also the WSA (10035)
        // variant just in case ffmpeg got cross-compiled weirdly.
        matches!(mag, 11 | 35 | 10035)
    }

    /// Decode a VP9 superframe payload through libavcodec.
    ///
    /// Returns the list of decoded `DecodedFrame`s on success. Returns
    /// `None` when libavcodec rejected the input (this is the path that
    /// tells the caller "the input wasn't valid VP9, so don't expect
    /// our decoder to accept it either"). Returns `Some(empty)` only
    /// when libavcodec accepted the packet but produced no frames
    /// (a degenerate possibility we don't currently special-case).
    pub fn decode_vp9(data: &[u8]) -> Option<Vec<DecodedFrame>> {
        // Don't bother libavcodec with empty / huge inputs.
        if data.is_empty() || data.len() > 1 << 22 {
            return None;
        }

        type FindDecoderFn = unsafe extern "C" fn(i32) -> *const c_void;
        type AllocCtxFn = unsafe extern "C" fn(*const c_void) -> *mut c_void;
        type Open2Fn = unsafe extern "C" fn(*mut c_void, *const c_void, *mut *mut c_void) -> i32;
        type AllocPktFn = unsafe extern "C" fn() -> *mut c_void;
        type AllocFrmFn = unsafe extern "C" fn() -> *mut c_void;
        type SendPktFn = unsafe extern "C" fn(*mut c_void, *const c_void) -> i32;
        type RecvFrmFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i32;
        type FreePktFn = unsafe extern "C" fn(*mut *mut c_void);
        type FreeFrmFn = unsafe extern "C" fn(*mut *mut c_void);
        type FreeCtxFn = unsafe extern "C" fn(*mut *mut c_void);
        type NewPktFn = unsafe extern "C" fn(*mut c_void, i32) -> i32;

        let l = lib()?;
        unsafe {
            let avcodec_find_decoder: Symbol<FindDecoderFn> =
                l.get(b"avcodec_find_decoder").ok()?;
            let avcodec_alloc_context3: Symbol<AllocCtxFn> =
                l.get(b"avcodec_alloc_context3").ok()?;
            let avcodec_open2: Symbol<Open2Fn> = l.get(b"avcodec_open2").ok()?;
            let av_packet_alloc: Symbol<AllocPktFn> = l.get(b"av_packet_alloc").ok()?;
            let av_frame_alloc: Symbol<AllocFrmFn> = l.get(b"av_frame_alloc").ok()?;
            let av_new_packet: Symbol<NewPktFn> = l.get(b"av_new_packet").ok()?;
            let avcodec_send_packet: Symbol<SendPktFn> = l.get(b"avcodec_send_packet").ok()?;
            let avcodec_receive_frame: Symbol<RecvFrmFn> = l.get(b"avcodec_receive_frame").ok()?;
            let av_packet_free: Symbol<FreePktFn> = l.get(b"av_packet_free").ok()?;
            let av_frame_free: Symbol<FreeFrmFn> = l.get(b"av_frame_free").ok()?;
            let avcodec_free_context: Symbol<FreeCtxFn> = l.get(b"avcodec_free_context").ok()?;

            let codec = avcodec_find_decoder(AV_CODEC_ID_VP9);
            if codec.is_null() {
                return None;
            }
            let ctx = avcodec_alloc_context3(codec);
            if ctx.is_null() {
                return None;
            }
            // RAII-style cleanup epilogue via a closure.
            let result = (|| -> Option<Vec<DecodedFrame>> {
                if avcodec_open2(ctx, codec, std::ptr::null_mut()) < 0 {
                    return None;
                }
                let pkt = av_packet_alloc();
                if pkt.is_null() {
                    return None;
                }
                let alloc_rc = av_new_packet(pkt, data.len() as i32);
                if alloc_rc < 0 {
                    let mut p = pkt;
                    av_packet_free(&mut p as *mut *mut c_void);
                    return None;
                }
                // Copy our input into pkt->data (offset 24 in AVPacket).
                let pkt_bytes = pkt as *mut u8;
                let pkt_data_ptr = (pkt_bytes.add(PKT_OFF_DATA) as *const *mut u8).read_unaligned();
                let pkt_size = (pkt_bytes.add(PKT_OFF_SIZE) as *const i32).read_unaligned();
                if pkt_data_ptr.is_null() || pkt_size < data.len() as i32 {
                    let mut p = pkt;
                    av_packet_free(&mut p as *mut *mut c_void);
                    return None;
                }
                std::ptr::copy_nonoverlapping(data.as_ptr(), pkt_data_ptr, data.len());

                let send_rc = avcodec_send_packet(ctx, pkt);
                // Free the packet now — libavcodec internally refs the
                // buffer if it needs to keep it around.
                let mut p = pkt;
                av_packet_free(&mut p as *mut *mut c_void);
                if send_rc < 0 && !is_eagain(send_rc) {
                    return None;
                }
                // Drain frames.
                let mut out: Vec<DecodedFrame> = Vec::new();
                let frame = av_frame_alloc();
                if frame.is_null() {
                    return None;
                }
                for _ in 0..32 {
                    let recv_rc = avcodec_receive_frame(ctx, frame);
                    if recv_rc == 0 {
                        if let Some(df) = read_frame(frame) {
                            out.push(df);
                        }
                        // Continue draining.
                        continue;
                    }
                    // EAGAIN / EOF — stop draining cleanly.
                    break;
                }
                let mut f = frame;
                av_frame_free(&mut f as *mut *mut c_void);
                Some(out)
            })();
            let mut c = ctx;
            avcodec_free_context(&mut c as *mut *mut c_void);
            result
        }
    }

    /// Read the public AVFrame prefix into a `DecodedFrame`. Returns
    /// None for pix formats we don't know how to compare.
    unsafe fn read_frame(frame: *mut c_void) -> Option<DecodedFrame> {
        let bytes = frame as *const u8;
        let width = (bytes.add(FRM_OFF_WIDTH) as *const i32).read_unaligned();
        let height = (bytes.add(FRM_OFF_HEIGHT) as *const i32).read_unaligned();
        let format = (bytes.add(FRM_OFF_FORMAT) as *const i32).read_unaligned();
        if width <= 0 || height <= 0 {
            return None;
        }
        let (cs_x, cs_y, bps) = pix_fmt_geom(format)?;
        let cw = (width as u32 + (1 << cs_x) - 1) >> cs_x;
        let ch = (height as u32 + (1 << cs_y) - 1) >> cs_y;
        let data_arr = bytes.add(FRM_OFF_DATA) as *const *const u8;
        let line_arr = bytes.add(FRM_OFF_LINESIZE) as *const i32;
        let y_ptr = data_arr.read_unaligned();
        let u_ptr = data_arr.add(1).read_unaligned();
        let v_ptr = data_arr.add(2).read_unaligned();
        let y_stride = line_arr.read_unaligned();
        let u_stride = line_arr.add(1).read_unaligned();
        let v_stride = line_arr.add(2).read_unaligned();
        if y_ptr.is_null() || u_ptr.is_null() || v_ptr.is_null() {
            return None;
        }
        if y_stride <= 0 || u_stride <= 0 || v_stride <= 0 {
            return None;
        }
        let row_bytes_y = (width as u32 * bps) as usize;
        let row_bytes_c = (cw * bps) as usize;
        let h = height as usize;
        let chh = ch as usize;
        let mut y = vec![0u8; row_bytes_y * h];
        for row in 0..h {
            let src = y_ptr.add(row * y_stride as usize);
            std::ptr::copy_nonoverlapping(src, y.as_mut_ptr().add(row * row_bytes_y), row_bytes_y);
        }
        let mut u = vec![0u8; row_bytes_c * chh];
        let mut v = vec![0u8; row_bytes_c * chh];
        for row in 0..chh {
            let su = u_ptr.add(row * u_stride as usize);
            let sv = v_ptr.add(row * v_stride as usize);
            std::ptr::copy_nonoverlapping(su, u.as_mut_ptr().add(row * row_bytes_c), row_bytes_c);
            std::ptr::copy_nonoverlapping(sv, v.as_mut_ptr().add(row * row_bytes_c), row_bytes_c);
        }
        Some(DecodedFrame {
            width: width as u32,
            height: height as u32,
            pix_fmt: format,
            y,
            u,
            v,
            chroma_dims: (cw, ch),
            bytes_per_sample: bps,
        })
    }
}
