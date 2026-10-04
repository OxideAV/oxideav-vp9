//! The registry encoder defaults to 8-bit 4:2:0 (profile 0) when the
//! stream parameters carry no pixel format, and the result decodes.

use oxideav_core::{CodecId, CodecParameters, Frame, PixelFormat, VideoFrame, VideoPlane};

#[test]
fn encoder_without_pixel_format_defaults_to_yuv420p_and_decodes() {
    let (w, h) = (64u32, 48u32);
    let mut p = CodecParameters::video(CodecId::new("vp9"));
    p.width = Some(w);
    p.height = Some(h);
    let mut enc = oxideav_vp9::make_encoder(&p).expect("encoder without pixel_format");
    assert_eq!(enc.output_params().pixel_format, Some(PixelFormat::Yuv420P));

    let (cw, ch) = (w as usize / 2, h as usize / 2);
    let luma: Vec<u8> = (0..(w * h) as usize).map(|i| (i % 251) as u8).collect();
    let frame = VideoFrame {
        pts: Some(0),
        planes: vec![
            VideoPlane {
                stride: w as usize,
                data: luma,
            },
            VideoPlane {
                stride: cw,
                data: vec![128; cw * ch],
            },
            VideoPlane {
                stride: cw,
                data: vec![128; cw * ch],
            },
        ],
    };
    enc.send_frame(&Frame::Video(frame)).expect("send");
    enc.flush().expect("flush");
    let pkt = enc.receive_packet().expect("packet");

    let mut dp = CodecParameters::video(CodecId::new("vp9"));
    dp.width = Some(w);
    dp.height = Some(h);
    let mut dec = oxideav_vp9::make_decoder(&dp).expect("decoder");
    dec.send_packet(&pkt).expect("decode");
    let Frame::Video(out) = dec.receive_frame().expect("frame") else {
        panic!("video frame expected");
    };
    assert_eq!(out.planes.len(), 3);
}
