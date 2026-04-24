use oxideav_vp9::encoder::{encode_keyframe, EncoderParams};

fn main() {
    let p = EncoderParams::keyframe(64, 64);
    let a = encode_keyframe(&p);
    let b = encode_keyframe(&p);
    if a != b {
        println!("NOT DETERMINISTIC!");
        println!("a: {:02x?}", a);
        println!("b: {:02x?}", b);
    } else {
        println!("deterministic, {} bytes", a.len());
        println!("{:02x?}", a);
    }
}
