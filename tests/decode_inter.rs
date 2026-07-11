//! End-to-end inter-frame (P-frame) decode pins for
//! [`oxideav_vp9::decode_vp9_sequence`] — VP9 spec v0.7 §6.4.11
//! `inter_frame_mode_info( )` + §6.5 motion-vector reference search +
//! §8.5.2 inter prediction + §8.10 reference-buffer update.
//!
//! The `i-frame-then-p-frame-64x64` clean-room fixture (a keyframe
//! followed by one P-frame, 64x64, 4:2:0, 8-bit, `refresh_mask=0x01`,
//! `highprec_mvs=1`, `filter_mode=1`, single-reference LAST) is embedded
//! verbatim (both frame payloads + the expected planar output) so the pin
//! runs in standalone CI where the docs tree is not checked out. A second
//! docs-gated test re-parses the IVF directly when the corpus is present.
//!
//! `expected.yuv` (12288 B) is byte-for-byte ground truth produced by an
//! independent decoder run over the same bitstream (see the fixture
//! `notes.md`): 2 x (64x64 Y + 32x32 U + 32x32 V).

use oxideav_vp9::{decode_vp9_sequence, split_superframe};

fn from_hex_lines(lines: &[&str]) -> Vec<u8> {
    let mut out = Vec::new();
    for line in lines {
        assert!(line.len() % 2 == 0);
        for i in (0..line.len()).step_by(2) {
            out.push(u8::from_str_radix(&line[i..i + 2], 16).expect("hex"));
        }
    }
    out
}

const KEYFRAME: &[&str] = &[
    "824983420003f003f60438241c18420002606fb0faffc9f2346742bca6b8196f",
    "e06a3e6233c7edf9defcbe970323147ec38bbb63ed0000007e18e564da60f280",
    "13bce860a4c1f073c6482f68adff8123a0b8585beb7ffffd1207ddf5061071e9",
    "0ffa672b982ea090ec7dc5f3dd31587dd8b8491b5b2ed75ffffbdf50a469d059",
    "6f0c8b00caec4d280fca589f28bf3cef442472e3822eef6127324d177b73415a",
    "0e68b9f68d9a022ebad2af1ab0d242e5cfb04cc3260ef1bf01f6fd6d12edaae8",
    "0e960e6cf0a2864e8d72a86f7d16bd53c5ff60f23720f59e66186b1c4254109c",
    "0a36c68def71c13d3dd58fcc0127ba5b5cd331c08d934429aefb7951d57c0123",
    "a888ec77d2236d29deae3119498642e22106161c15698a5e97258e3feff6d05e",
    "bb289cbb60d2217fc04ef00fb363e499ed40bd6e06ff2b0ba43c3a0a2ba92c83",
    "0ff5b1c34ef4676138cfae879efd6ee70e1c877bc2186e140c19fbc22ff135e3",
    "ae813c027ab6c85ea10a6f204e69819a62dd91959e5e7b9ff575fd060b23eba1",
    "d51e979f87cda200209cc3897166248f4f800b5988394dcf47d2dc0d9e2cff5f",
    "0829030b69fbb56655ddfa6b77fe4be3045d385793b136f505313775f8759e75",
    "fd143631a1aba6cdb8f791be4867227cf0abbf45aef1832288c66621443eddae",
    "cbf97d0144bf6582d40da9700a925a723ed589ea04253fc55fe0f718dd32f4bb",
    "4bc29c706059b349dda27f66c1afe59bf13959f565843578bfa6d337d6215ff0",
    "25fe6f5b60ee854dd7d44fe7124cb712ff544646a14ab1f4df77a3c2c908c655",
    "3cc565f58fcd45b5cfe347b42a1aaabeed96011b8336bfdcbb3c5ad873246d16",
    "0b6788cf6967bf484c756167c52efb8ebae043fe20251727308b2f8fc1385195",
    "6be1a7b98bff650bdb4ffe1dd9f140ec2b8d5ca0967a40cb484030465929c744",
    "3ffd062302621cd40b99af130fdce9df6d1dbf79b760f5211df6aa15671a5082",
    "7797bf6af318b0469bae807f24f70e01dc9f1714ee18edb2b6d6dbb73417807a",
    "12592b42b8fe626f73edb8b47ea3763f46fe07362962c8e739f51d5e7fe52440",
    "2a121921b164b6d69f44ff04d9e8e9af89b1fec773c14425d406169056e2f4d4",
    "3d876fd5eff279fcbbc9c0ec17ee57fd412927a5ad93119e7ce7ab462d1716d2",
    "46c7a003fcc1e309d3a8d6e2c7d923d27931764f2fe884d8189fa7a016345fe7",
    "ebfab1cd56c51d277ba211caa02d08bacd94db0058b63c588277bb27a463943a",
    "2b13cd076ebab1f3d8e1b6e82a7aaba292c34d8816182630b465cb71acce62e1",
    "479d45a9a16a5dad148603b7cb5e3bff452e48a735c9b32204131684d86f31ca",
    "195ffbfe15c690bc2b864b6179b645b6ec961622771d23318bc95bf31fefdc89",
    "279863e85b6b80322217b746ed416ac74540678e0fbf0000",
];

const PFRAME: &[&str] = &["860040929c08518000036000007a4906e16fab29fdd570"];

const EXPECTED: &[&str] = &[
    "101010101010101051515151515151519090929090919191d4d3d2d2d2d2d1d1",
    "29292929292929296b6b6b6a6a6a6c6caaaaaaaaaaaaaaaaebebebebebebebeb",
    "1010101010101010515151515151515190918f9191909090d129292a2a292929",
    "d2d2d2d2d2d2d2d26b6b6b6b6a6a6d69aaaaaaaaaaaaaaaaebebebebebebebeb",
    "1010101010101010515151515151515194919194906d6a6a2a2a2a2a29292928",
    "d2d2d2d2d2d2d2d2919191916b696b69aaaaaaaaaaaaaaaaebebebebebebebeb",
    "101010101010101051515151515151519192906d6a6a6a6a27292c2a29292828",
    "d2d2d2d2d2d2d2d2919191919391696aaaaaaaaaaaaaaaaaebebebebebebebeb",
    "101010101010101051515151514f52508c6a696b6a6a6a6a292b2a2929292929",
    "d2d2d2d2d2d2d2d29191919191919191a9abaaaaaaaaaaaaebebebebebebebeb",
    "1010101010101010515151514f5150a86a696a69696a6a6a2b2a292929292929",
    "d2d2d2d2d2d2d2d291919191919191915152aaabaaaaaaaaebebebebebebebeb",
    "1010101010101010515151515250a9ab696a696a6a6a6a6a2a29292929292929",
    "d2d2d2d2d2d2d2d29191919191919191525050a8aaaaaaaaebebebebebebebeb",
    "10101010101010105151515150a8abaa6b696b6b6b6a6a6a2929282929292929",
    "d2d2d2d2d2d2d2d29191919191919191514f5053aaaaaaaaebebebebebebebeb",
    "1010101010101010524f51aaababaaaa6a6a6b6b6a6a6a6a2928282929292929",
    "d2d2d2d2d2d2d2d29191919191919191515051524f52aaa9edecebeaeaebebeb",
    "10101010101010104f51a9aaaaaaaaaa6a6a6b6a6a6a6a6a2928292929292929",
    "d2d2d2d2d2d2d2d2919191919191919151515151535152a7ebebecebebebebeb",
    "101010101010101051a9aba8aaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d291919191919191915151525152535154ececebecebebebeb",
    "1010101011101010aaaaa8a9aaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d29191919191919191515252505051515211ecebedebebebeb",
    "1010101111111112aaa9a9aaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d2919191919191919151525250505151510deae9e9ebebebeb",
    "10101010111111e8aaa9aaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d2919191919191919151525250505151511011eaecebebebeb",
    "101010101111e9e9aaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d291919191919191915152525050515151130f0fe9ebebebeb",
    "1110101012e7e9ecaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d29191919191919191515252505051515110101011ebebebeb",
    "12100f0f11ebececaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d29191919191919191515251515151515110101011e9eceaec",
    "100f1010ebebe8eaaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d2919191919191919151515151515151511010101111eae8ed",
    "0f100f0fece8ecebaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d2919191919191919151515151515151511010101113eaece9",
    "0f100feaebedeceaaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d291919191919191915151515151515151101010111211eaed",
    "0f1010ebebebebeaaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d29191919191919191515151515151515110101011100debeb",
    "0f10edebebebebebaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d29191919191919191515151515151515110101011100e10eb",
    "0f10e9ebebebebebaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d29191919191919191515151515151515110101011110f11e9",
    "0f10e6ebebebebebaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d2919191919191919151515151515151511010101110100feb",
    "1010edebebebebebaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d29191919191919191515151515151515110101011100f13e6",
    "10ebedebebebebebaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d2919191919191919151515151515151511010101212111010",
    "12eaece9ebebebebaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d2919191919191919151515151515151511010101111111110",
    "10ebe9ebebebebebaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d291919191919191915151515151515151101010101111100f",
    "10ebe9ebebebebebaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d2919191919191919151515151515151511010101110111010",
    "10ebe9ebebebebebaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d2919191919191919151515151515151511010101110101010",
    "10ebe9ebebebebebaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d2919191919191919151515151515151511010101110101010",
    "10ebe9ebebebebebaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d2919191919191919151515151515151511010101110101010",
    "10ebe9ebebebebebaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d2919191919191919151515151515151511010101110101010",
    "10ebe9ebebebebebaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d2919191919191919151515151515151511010101110101010",
    "10ebe9ebebebebebaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d2919191919191919151515151515151511010101110101010",
    "10ebe9ebebebebebaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d2919191919191919151515151515151511010101110101010",
    "10ebe9ebebebebebaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d2919191919191919151515151515151511010101110101010",
    "10ebe9ebebebebebaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d2919191919191919151515151515151511010101110101010",
    "10ebeaebebebebebaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d2919191919191919151515151515151511010101110101010",
    "10ebeaebebebebebaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d2919191919191919151515151515151511010101110101010",
    "0f10eaebebebebebaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d2919191919191919151515151515151511010101110100feb",
    "0f10eaebebebebebaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d2919191919191919151515151515151511010101110100feb",
    "0f10ebebebebebebaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d2919191919191919151515151515151511010101110100feb",
    "0f10ebebebebebebaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d2919191919191919151515151515151511010101110100feb",
    "101112eaebebebebaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d291919191919191915151515151515151101010110e10e9ee",
    "0f1113ecebebebebaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d291919191919191915151525151515151101010111312edea",
    "120f110eebebebebaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d291919191919191915151525151515151101010110febedea",
    "12111012ebebebebaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d2919191919191919151515351515151511010101110ececea",
    "525d6a76828d9aa6b3bfc9d0cac4bcb6b1aba5a099929295979a9c9ea0a3a5a7",
    "aa9e9185796e6155493c322b31373d444b50545c62696a6765625f5d5b595654",
    "525d6a76828d9aa6b3bfc9d0cac4bcb6b1aba5a099929295979a9c9ea0a3a5a7",
    "aa9e9185796e6155493c322b31373d444b50555c62696a6765625f5d5b595654",
    "525d6a76828d9aa6b3bfc9d0cac4bcb6b1aba5a099929295979a9c9ea0a3a5a7",
    "aa9e9185796e6155493c322b31373d444b50555c62696a6765625f5d5b595654",
    "525d6a76828d9aa6b3bfc9d0cac4bcb6b1aba5a099929295979a9c9ea0a3a5a7",
    "aa9e9185796e6155493c322b31373d444b50565c62696a6765625f5d5b595654",
    "525d6a76828d9aa6b3bfc9d0cac4bcb6b1aba5a099929295979a9c9ea0a3a5a7",
    "aa9e9185796e6155493c322b31373d444b50565c62696a6765625f5d5b595654",
    "525d6a76828d9aa6b3bfc9d0cac4bcb6b1aba5a099929295979a9c9ea0a3a5a7",
    "aa9e9185796e6155493c322b31373d444b50565c62696a6765625f5d5b595654",
    "525d6a76828d9aa6b3bfc9d0cac4bcb6b1aba5a099929295979a9c9ea0a3a5a7",
    "aa9e9185796e6155493c322b31373d444b50565c62696a6765625f5d5b595654",
    "525d6a76828d9aa6b3bfc9d0cac4bcb6b1aba5a099929295979a9c9ea0a3a5a7",
    "aa9e9185796e6155493c322b31373d444b50565c62696a6765625f5d5b595654",
    "535d6b757f8c9ba5b6bec9d2cbc0bcb8b0aba49f99959494999a9c9da0a3a5a6",
    "aa9e90847b706259463e322a323a3f424c4f585e696969696361605e5b585655",
    "0f0f10100f0f10105152525253abadac6a6a6a6a6a696a6c28292a2a29292928",
    "d2d0d3d1d1d2d2d1919090909091919252545250aaaaaaaaedecebebececeded",
    "100f0f0f1010100f515150504f52aaab6c6c6b6a6a696a6b2928272727282828",
    "d4d1d3d1d1d1d3d49192929291908f8f505251a8aaaaaaaaebebecececebebeb",
    "0d0f1111100f0f1052525151525253a868696b6b6a696a6b2b2a282728292a2a",
    "d2d0d2d2d2d3d2d18f8f9090909191915152aaa8aaaaaaaae8eaebedecebeaea",
    "10101010101010105252515151515151906a6a6a696b696a27282a2a29292929",
    "d1d1d2d2d2d2d2d28f9090909192918faaaaaaaaaaaaaaaaeaebececebebebeb",
    "101010101010101052525151515151519091906b68696a672a29262a29292929",
    "d1d1d2d3d2d2d2d29292929393916969aaaaaaaaaaaaaaaaebececebebebebeb",
    "1010101010101010525251515151515192909292906d6a6b2a29292a29292929",
    "d1d2d2d3d2d2d2d2919192926c696b69aaaaaaaaaaaaaaaaececebeaebebebeb",
    "101010101010101052525151515151519193909192909292d2292b2829292929",
    "d1d2d3d3d2d2d2d26a6b6b6b6b6c696caaaaaaaaaaaaaaaaecebeaeaebebebeb",
    "808080805a5a5b5a353139404f81808080808080c4c0d3cda6a6a6a680808080",
    "808080805a59585b3661a0bdedfefeff040404043c3f7fc0a7a5a7a880808080",
    "808080805b585b71a2c8d2cbf2efeef00f0f0f0f33332e3f80a2a7a480808080",
    "808080805a5b71a5d0ccc6cbf0f0eeef1010101036343334576da0a880808080",
    "808080805a8ea5a5ccc9cacbf1f0f0f010101010363535355758638e80808080",
    "808080808ea9a6a4c9cacbcbf0f0f0f010101010353535355a5b595b80808080",
    "80808080a9a4a5a4cacbcbcbf0f0f0f0101010103535353559585a5780808080",
    "80808080a7a9a5a6cbcbcbcbf0f0f0f01010101035353535595a595a80808080",
    "80808080a7a6a5a6cacacacaf0f0f0f01010101035353535595b5a5a80808080",
    "80808080a6a6a5a6cacacacaf0f0f0f01010101035353535595a5a5a80808080",
    "80808080a6a6a6a6cacacacaf0f0f0f010101010353535355a5a5a5a80808080",
    "80808080a6a6a6a6cacacacaf0f0f0f010101010353535355a5a5a5a80808080",
    "80808080a6a6a6a6cacacacaf0f0f0f010101010353535355a5a5a5a80808080",
    "80808080a6a6a6a6cacacacaf0f0f0f010101010353535355a5a5a5a80808080",
    "80808080a6a6a6a6cacacacaf0f0f0f010101010353535355a5a5a5a80808080",
    "80808080a6a6a6a6cacacacaf0f0f0f010101010353535355a5a5a5a80808080",
    "80808080a6a6a6a6cacacacaf0f0f0f010101010353535355a5a5a5a80808080",
    "80808080a6a6a6a6cacacacaf0f0f0f010101010353535355a5a5a5a80808080",
    "80808080a6a6a6a6cacacacaf0f0f0f010101010353535355a5a5a5a80808080",
    "80808080a6a6a6a6cacacacaf0f0f0f010101010353535355a5a5a5a80808080",
    "80808080a6a6a6a6cacacacaf0f0f0f011111111343434345a5a5a5a80808080",
    "7f7f7f7fa7a7a7a7cbcbcbcbf1f1f1f110101010343434345959595980808080",
    "81818181a8a8a8a8ccccccccf2f2f2f20d0d0d0d333333335858585880808080",
    "7b7b7b7b9e9e9e9ec2c2c2c2e8e8e8e81a1a1a1a414141416262626280807f80",
    "5a4c3f31271a1e252d353c496073899d9facb9c7d5e1ded6d3ccc5b8a6937f6a",
    "56483b2d1d11141c2129303d566b8194acbac7d5e3efece4dcd4ccbfa9957e6a",
    "56483b2e2013171e252c3441596c8297aab8c4d3e1edeae2ded5ccbea9957e6a",
    "524437291a0e11191d242c384f647c91b2c1cfdeecf8f5ede2d9cebfa9957e6a",
    "6b655d573b374a62797a7e88a7afbbc35d636c718d91908c9b96b7b2928a7e76",
    "818284855b5c6198d4d8d2d2f9f5f5f2090806052f2c2a235a89a3a481818080",
    "80808282585b5a5a609ecdd3f8eeefef1211100f2d2e3f81a1aaa7a27f7f7f80",
    "808080815a5b5b5833366184b8edeef1100f0f0f7f82c4d0a6a7a7a880808181",
    "80808080efefeff0231e242e8583807f80818181d2d5e9e31011111180808080",
    "80808080eef0f0f62554a8d3716f6d6c949494942e2d7fd103100d1380808080",
    "80808080f1f2f9b0a9dbe5e06f6e6e6d93939393211f162e7e110d1180808080",
    "80808080f4eeaf12e3e0ddde6e6e6f6f9191919123212220fdb2210980808080",
    "80808080f24e100ee0dedede6d6e6e6e9292929223232221f0f9e45180808080",
    "808080804f0a0c0fdededede6e6e6e6e9292929223222221efeef1ef80808080",
    "80808080090f0e0ddededede6e6e6e6e9292929223222121f1f3eff280808080",
    "8080808010120e11dededede6e6e6e6e9292929222222121f1f2f2ef80808080",
    "8080808011101011dededede6e6e6e6e9292929222222121f2f1f0ef80808080",
    "8080808010101111dededede6e6e6e6e9292929222222121f1f0efef80808080",
    "8080808010111111dededede6e6e6e6e9292929222222121f0efefef80808080",
    "8080808011111111dededede6e6e6e6e9292929222222121efefefef80808080",
    "8080808010101010dededede6e6e6e6e9292929222222121f0f0f0f080808080",
    "8080808010101010dededede6e6e6e6e9292929222222121f0f0f0f080808080",
    "8080808010101010dededede6e6e6e6e9292929222222121f0f0f0f080808080",
    "8080808010101010dededede6e6e6e6e9292929222222121f0f0f0f080808080",
    "8080808010101010dededede6e6e6e6e9292929222222121f0f0f0f080808080",
    "8080808010101010dededede6e6e6e6e9292929222222121f0f0f0f080808080",
    "8080808010101010dededede6e6e6e6e9292929222222121f0f0f0f080808080",
    "8080808010101010dededede6e6e6e6e9292929222222121f0f0f0f080808080",
    "8080808010101010dddddddd6d6d6d6d9191919122222121f0f0f0f07e7e7e7e",
    "818181810f0f0f0fdededede6f6f6f6f9393939323232222f0f0f0f080808080",
    "7f7f7f7f0e0e0e0ee1e1e1e1707070709494949421212020f0f0f0f080808080",
    "8585858518181818d4d4d4d4696969698d8d8d8d28282727f0f0f0f086868686",
    "e8d6c4b29b8a77665c49362c211f1c1a1d2c3d4d586a7d90b0c3d7e1dddfe3e9",
    "ecdac8b6a8947f6b503d281e1b191512142537485b6e8295abc0d6e0e4e8ecef",
    "ebdac7b6a8947f6b56412a201d18131214253648596d8397acc0d6e0e3e6ebec",
    "f0deccbaa3927f6e4c3720141714110f0e1f32435c71889da6bfdcebeaedf1f3",
    "b3ada39cc9c289369a8f84804647443f555c656c3f44535bd0d77575b3b3b3b3",
    "7c7c7c7cf4f3e74ce6ede8e772747472989897971e1f1911e45305057b7b7b7b",
    "7f7f7f7ef1f0f3f057a6e0e76f6d6d6f91919191191a2d7d2008101080808080",
    "8281807fefeff0f51c24578175706d6e909192928281cfe90c0f0f0e7f7f7f7f",
    "101010101010101051515151515151519090929090919191d4d3d2d2d2d2d1d1",
    "29292929292929296b6b6b6a6a6a6c6caaaaaaaaaaaaaaaaebebebebebebebeb",
    "1010101010101010515151515151515190918f9191909090d129292a2a292929",
    "d2d2d2d2d2d2d2d26b6b6b6b6a6a6d69aaaaaaaaaaaaaaaaebebebebebebebeb",
    "1010101010101010515151515151515194919194906d6a6a2a2a2a2a29292928",
    "d2d2d2d2d2d2d2d2919191916b696b69aaaaaaaaaaaaaaaaebebebebebebebeb",
    "101010101010101051515151515151519192906d6a6a6a6a27292c2a29292828",
    "d2d2d2d2d2d2d2d2919191919391696aaaaaaaaaaaaaaaaaebebebebebebebeb",
    "101010101010101051515151514f52508c6a696b6a6a6a6a292b2a2929292929",
    "d2d2d2d2d2d2d2d29191919191919191a9abaaaaaaaaaaaaebebebebebebebeb",
    "1010101010101010515151514f5150a86a696a69696a6a6a2b2a292929292929",
    "d2d2d2d2d2d2d2d291919191919191915152aaabaaaaaaaaebebebebebebebeb",
    "1010101010101010515151515250a9ab696a696a6a6a6a6a2a29292929292929",
    "d2d2d2d2d2d2d2d29191919191919191525050a8aaaaaaaaebebebebebebebeb",
    "10101010101010105151515150a8abaa6b696b6b6b6a6a6a2929282929292929",
    "d2d2d2d2d2d2d2d29191919191919191514f5053aaaaaaaaebebebebebebebeb",
    "1010101010101010524f51aaababaaaa6a6a6b6b6a6a6a6a2928282929292929",
    "d2d2d2d2d2d2d2d29191919191919191515051524f52aaa9edecebeaeaebebeb",
    "10101010101010104f51a9aaaaaaaaaa6a6a6b6a6a6a6a6a2928292929292929",
    "d2d2d2d2d2d2d2d2919191919191919151515151535152a7ebebecebebebebeb",
    "101010101010101051a9aba8aaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d291919191919191915151525152535154ececebecebebebeb",
    "1010101011101010aaaaa8a9aaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d29191919191919191515252505051515211ecebedebebebeb",
    "1010101111111112aaa9a9aaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d2919191919191919151525250505151510deae9e9ebebebeb",
    "10101010111111e8aaa9aaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d2919191919191919151525250505151511011eaecebebebeb",
    "101010101111e9e9aaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d291919191919191915152525050515151130f0fe9ebebebeb",
    "1110101012e7e9ecaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d29191919191919191515252505051515110101011ebebebeb",
    "12100f0f11ebececaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d29191919191919191515251515151515110101011e9eceaec",
    "100f1010ebebe8eaaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d2919191919191919151515151515151511010101111eae8ed",
    "0f100f0fece8ecebaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d2919191919191919151515151515151511010101113eaece9",
    "0f100feaebedeceaaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d291919191919191915151515151515151101010111211eaed",
    "0f1010ebebebebeaaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d29191919191919191515151515151515110101011100debeb",
    "0f10edebebebebebaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d29191919191919191515151515151515110101011100e10eb",
    "0f10e9ebebebebebaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d29191919191919191515151515151515110101011110f11e9",
    "0f10e6ebebebebebaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d2919191919191919151515151515151511010101110100feb",
    "1010edebebebebebaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d29191919191919191515151515151515110101011100f13e6",
    "10ebedebebebebebaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d2919191919191919151515151515151511010101112101010",
    "12eaece9ebebebebaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d2919191919191919151515151515151511010101111101110",
    "10ebe9ebebebebebaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d291919191919191915151515151515151101010111110100f",
    "10ebe9ebebebebebaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d2919191919191919151515151515151511010101110101010",
    "10ebe9ebebebebebaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d2919191919191919151515151515151511010101110101010",
    "10ebe9ebebebebebaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d2919191919191919151515151515151511010101110101010",
    "10ebe9ebebebebebaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d2919191919191919151515151515151511010101110101010",
    "10ebe9ebebebebebaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d2919191919191919151515151515151511010101110101010",
    "10ebe9ebebebebebaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d2919191919191919151515151515151511010101110101010",
    "10ebe9ebebebebebaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d2919191919191919151515151515151511010101110101010",
    "10ebe9ebebebebebaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d2919191919191919151515151515151511010101110101010",
    "10ebe9ebebebebebaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d2919191919191919151515151515151511010101110101010",
    "10ebe9ebebebebebaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d2919191919191919151515151515151511010101110101010",
    "10ebeaebebebebebaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d2919191919191919151515151515151511010101110101010",
    "10ebeaebebebebebaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d2919191919191919151515151515151511010101110101010",
    "0f10eaebebebebebaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d2919191919191919151515151515151511010101110100feb",
    "0f10eaebebebebebaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d2919191919191919151515151515151511010101110100feb",
    "0f10ebebebebebebaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d2919191919191919151515151515151511010101110100feb",
    "0f10ebebebebebebaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d2919191919191919151515151515151511010101110100feb",
    "101112eaebebebebaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d291919191919191915151515151515151101010110e10e9ee",
    "0f1113ecebebebebaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d291919191919191915151525151515151101010111312edea",
    "120f110eebebebebaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d291919191919191915151525151515151101010110febedea",
    "12111012ebebebebaaaaaaaaaaaaaaaa6a6a6a6a6a6a6a6a2929292929292929",
    "d2d2d2d2d2d2d2d2919191919191919151515351515151511010101110ececea",
    "55626f7b86929fabb8c3cdcfc8c1b9b4b1aba5a099929295979a9c9ea0a3a5a7",
    "a195887d72685c51463a312b32383f464b50545c62696a6765625f5d5b595654",
    "55626f7b86929fabb8c3cdcfc8c1b9b4b1aba5a099929295979a9c9ea0a3a5a7",
    "a195897d72685c51463a312b32383f464b50555c62696a6765625f5d5b595654",
    "55626f7b86929fabb8c3cdcfc8c1b9b4b1aba5a099929295979a9c9ea0a3a5a7",
    "a196897e72685c51463a312b32383f464b50555c62696a6765625f5d5b595654",
    "55626f7b86929fabb8c3cdcfc8c1b9b4b1aba5a099929295979a9c9ea0a3a5a7",
    "a296897e73695c51463a312b32383f464b50565c62696a6765625f5d5b595654",
    "55626f7b86929fabb8c3cdcfc8c1b9b4b1aba5a099929295979a9c9ea0a3a5a7",
    "a3978a7f73695d52473b312b32383f464b50565c62696a6765625f5d5b595654",
    "55626f7b86929fabb8c3cdcfc8c1b9b4b1aba5a099929295979a9c9ea0a3a5a7",
    "a4988b80746a5d52473b312b32383e464b50565c62696a6765625f5d5b595654",
    "55626f7b86929fabb8c3cdcfc8c1b9b4b1aba5a099929295979a9c9ea0a3a5a7",
    "a5998c80756a5e53473b322b32383e454b50565c62696a6765625f5d5b595654",
    "55626f7b86929fabb8c3cdcfc8c1b9b4b1aba5a099929295979a9c9ea0a3a5a7",
    "a69a8d81766b5f53483b322b32383e454b50565c62696a6765625f5d5b595654",
    "535d6b757f8c9ba5b6bec9d2cbc0bcb8b0aba49f99959494999a9c9da0a3a5a6",
    "a79b8d81796e6058453e322a333b40434c4f585e696969696361605e5b585655",
    "0f0f10100f0f10105152525253abadac6a6a6a6a6a696a6c28292a2a29292928",
    "d0ced1cfd0d1d1d0919090909092929352545250aaaaaaaaedecebebececeded",
    "100f0f0f1010100f515150504f52aaab6c6c6b6a6a696a6b2928272727282828",
    "d3d0d2d0d0d0d3d49192929291919090505251a8aaaaaaaaebebecececebebeb",
    "0d0f1111100f0f1052525151525253a868696b6b6a696a6b2b2a282728292a2a",
    "d2d0d2d2d2d3d2d18f8f9090909191915152aaa8aaaaaaaae8eaebedecebeaea",
    "10101010101010105252515151515151906a6a6a696b696a27282a2a29292929",
    "d2d2d3d3d3d3d3d3909090909192918faaaaaaaaaaaaaaaaeaebececebebebeb",
    "101010101010101052525151515151519091906b68696a672a29262a29292929",
    "d2d2d3d4d3d3d3d39393939393916969aaaaaaaaaaaaaaaaebececebebebebeb",
    "1010101010101010525251515151515192909292906d6a6b2a29292a29292929",
    "d3d4d4d5d4d3d3d3929293926c696b69aaaaaaaaaaaaaaaaececebeaebebebeb",
    "101010101010101052525151515151519193909192909292d2292b2829292929",
    "d3d4d5d5d4d4d3d36b6c6c6b6b6c696caaaaaaaaaaaaaaaaecebeaeaebebebeb",
    "808080805a5a5b5a353139404f81808080808080c4c0d3cda6a6a6a680808080",
    "808080805a59585b3661a0bdedfefeff040404043c3f7fc0a7a5a7a880808080",
    "808080805b585b71a2c8d2cbf2efeef00f0f0f0f33332e3f80a2a7a480808080",
    "808080805a5b71a5d0ccc6cbf0f0eeef1010101036343334576da0a880808080",
    "808080805a8ea5a5ccc9cacbf1f0f0f010101010363535355758638e80808080",
    "808080808ea9a6a4c9cacbcbf0f0f0f010101010353535355a5b595b80808080",
    "80808080a9a4a5a4cacbcbcbf0f0f0f0101010103535353559585a5780808080",
    "80808080a7a9a5a6cbcbcbcbf0f0f0f01010101035353535595a595a80808080",
    "80808080a7a6a5a6cacacacaf0f0f0f01010101035353535595b5a5a80808080",
    "80808080a6a6a5a6cacacacaf0f0f0f01010101035353535595a5a5a80808080",
    "80808080a6a6a6a6cacacacaf0f0f0f010101010353535355a5a5a5a80808080",
    "80808080a6a6a6a6cacacacaf0f0f0f010101010353535355a5a5a5a80808080",
    "80808080a6a6a6a6cacacacaf0f0f0f010101010353535355a5a5a5a80808080",
    "80808080a6a6a6a6cacacacaf0f0f0f010101010353535355a5a5a5a80808080",
    "80808080a6a6a6a6cacacacaf0f0f0f010101010353535355a5a5a5a80808080",
    "80808080a6a6a6a6cacacacaf0f0f0f010101010353535355a5a5a5a80808080",
    "80808080a6a6a6a6cacacacaf0f0f0f010101010353535355a5a5a5a80808080",
    "80808080a6a6a6a6cacacacaf0f0f0f010101010353535355a5a5a5a80808080",
    "80808080a6a6a6a6cacacacaf0f0f0f010101010353535355a5a5a5a80808080",
    "80808080a6a6a6a6cacacacaf0f0f0f010101010353535355a5a5a5a80808080",
    "80808080a6a6a6a6cacacacaf0f0f0f011111111343434345a5a5a5a80808080",
    "7f7f7f7fa7a7a7a7cbcbcbcbf1f1f1f110101010343434345959595980808080",
    "81818181a8a8a8a8ccccccccf2f2f2f20d0d0d0d333333335858585880808080",
    "7b7b7b7b9e9e9e9ec2c2c2c2e8e8e8e81a1a1a1a414141416262626280807f80",
    "584a3c2f24191f262d353c496073899d9facb9c7d5e1ded6d3ccc5b8a6937f6a",
    "5445392a1a10151d2129303d566b8194acbac7d5e3efece4dcd4ccbfa9957f6a",
    "5445392c1d13181f252c3441596c8297aab8c4d3e1edeae2ded5ccbea9957e6a",
    "50423526170d121a1d242c384f647c91b2c1cfdeecf8f5ede2d9cebfa9957e6a",
    "6b655d573b374a62797a7e88a7afbbc35d636c718d91908c9b96b7b2928a7e76",
    "818284855b5c6198d4d8d2d2f9f5f5f2090806052f2c2a235a89a3a481818080",
    "80808282585b5a5a609ecdd3f8eeefef1211100f2d2e3f81a1aaa7a27f7f7f80",
    "808080815a5b5b5833366184b8edeef1100f0f0f7f82c4d0a6a7a7a880808181",
    "80808080efefeff0231e242e8583807f80818181d2d5e9e31011111180808080",
    "80808080eef0f0f62554a8d3716f6d6c949494942e2d7fd103100d1380808080",
    "80808080f1f2f9b0a9dbe5e06f6e6e6d93939393211f162e7e110d1180808080",
    "80808080f4eeaf12e3e0ddde6e6e6f6f9191919123212220fdb2210980808080",
    "80808080f24e100ee0dedede6d6e6e6e9292929223232221f0f9e45180808080",
    "808080804f0a0c0fdededede6e6e6e6e9292929223222221efeef1ef80808080",
    "80808080090f0e0ddededede6e6e6e6e9292929223222121f1f3eff280808080",
    "8080808010120e11dededede6e6e6e6e9292929222222121f1f2f2ef80808080",
    "8080808011101011dededede6e6e6e6e9292929222222121f2f1f0ef80808080",
    "8080808010101111dededede6e6e6e6e9292929222222121f1f0efef80808080",
    "8080808010111111dededede6e6e6e6e9292929222222121f0efefef80808080",
    "8080808011111111dededede6e6e6e6e9292929222222121efefefef80808080",
    "8080808010101010dededede6e6e6e6e9292929222222121f0f0f0f080808080",
    "8080808010101010dededede6e6e6e6e9292929222222121f0f0f0f080808080",
    "8080808010101010dededede6e6e6e6e9292929222222121f0f0f0f080808080",
    "8080808010101010dededede6e6e6e6e9292929222222121f0f0f0f080808080",
    "8080808010101010dededede6e6e6e6e9292929222222121f0f0f0f080808080",
    "8080808010101010dededede6e6e6e6e9292929222222121f0f0f0f080808080",
    "8080808010101010dededede6e6e6e6e9292929222222121f0f0f0f080808080",
    "8080808010101010dededede6e6e6e6e9292929222222121f0f0f0f080808080",
    "8080808010101010dddddddd6d6d6d6d9191919122222121f0f0f0f07e7e7e7e",
    "818181810f0f0f0fdededede6f6f6f6f9393939323232222f0f0f0f080808080",
    "7f7f7f7f0e0e0e0ee1e1e1e1707070709494949421212020f0f0f0f080808080",
    "8585858518181818d4d4d4d4696969698d8d8d8d28282727f0f0f0f086868686",
    "e6d3c1ae988773645c49362c211f1c1a202f40505b6d8093b0c3d7e1dddfe3e9",
    "ead7c5b4a5907c66503d281e1b19151217283a4b5e718598abc0d6e0e4e8ecef",
    "e9d7c4b4a5907c6756412a201d1813121728394b5c70869aacc0d6e0e3e6ebec",
    "eedbc9b6a08f7d694c3720141714110f112235465f748ba0a6bfdcebeaedf1f3",
    "b3ada39cc9c289369a8f84804647443f585f686f4247565ed0d77575b3b3b3b3",
    "7c7c7c7cf4f3e74ce6ede8e7727474729b9b9a9a21221c14e45305057b7b7b7b",
    "7f7f7f7ef1f0f3f057a6e0e76f6d6d6f949494941c1d30802008101080808080",
    "8281807fefeff0f51c24578175706d6e939495958584d2ec0c0f0f0e7f7f7f7f",
];

/// §6.4.11 / §6.5 / §8.5.2 / §8.10: the embedded keyframe + P-frame
/// sequence decodes byte-exact against the staged `expected.yuv`. This is
/// the headline inter-frame pin: the P-frame reconstructs to real pixels
/// through the inter mode-info decode, the motion-compensated prediction,
/// the residual add, the loop filter, and the reference-buffer update.
#[test]
fn iframe_then_pframe_sequence_byte_exact() {
    let keyframe = from_hex_lines(KEYFRAME);
    let pframe = from_hex_lines(PFRAME);
    let expected = from_hex_lines(EXPECTED);

    let frames = decode_vp9_sequence(&[&keyframe, &pframe]).expect("decode sequence");
    assert_eq!(frames.len(), 2, "two shown frames (keyframe + P-frame)");
    for f in &frames {
        assert_eq!(f.width, 64);
        assert_eq!(f.height, 64);
        assert_eq!(f.bit_depth, 8);
    }

    let mut got = Vec::new();
    for f in &frames {
        got.extend(f.to_planar_bytes());
    }
    assert_eq!(got.len(), expected.len(), "concatenated planar length");
    assert_eq!(got, expected, "sequence byte-exact vs expected.yuv");
}

/// Decoding the keyframe alone (single-element sequence) reproduces the
/// keyframe-sized prefix of the expected output — the §8.10 reference
/// update on the keyframe does not perturb the visible reconstruction.
#[test]
fn keyframe_only_sequence_matches_prefix() {
    let keyframe = from_hex_lines(KEYFRAME);
    let expected = from_hex_lines(EXPECTED);
    let frames = decode_vp9_sequence(&[&keyframe]).expect("decode keyframe");
    assert_eq!(frames.len(), 1);
    let got = frames[0].to_planar_bytes();
    assert_eq!(got, expected[..got.len()], "keyframe prefix byte-exact");
}

/// A `show_existing_frame` short packet (§6.2 / §8.9). The byte encodes,
/// MSB-first: `frame_marker` f(2) = 2, `profile_low_bit` f(1) = 0,
/// `profile_high_bit` f(1) = 0, `show_existing_frame` f(1) = 1, then
/// `frame_to_show_map_idx` f(3). For profile 0 this is
/// `0b10_0_0_1_iii = 0x88 | idx`.
fn show_existing_packet(idx: u8) -> Vec<u8> {
    assert!(idx < 8, "frame_to_show_map_idx is f(3)");
    vec![0x88 | idx]
}

/// §8.9 output process: a `show_existing_frame` packet re-displays
/// `FrameStore[ frame_to_show_map_idx ]` — the §8.10 reference slot — with
/// that stored frame's own dimensions and bit depth. The keyframe refreshes
/// all eight slots (`refresh_frame_flags = 0xFF`), so a `show_existing` of
/// any slot must re-emit the keyframe verbatim.
#[test]
fn show_existing_frame_redisplays_keyframe_slot() {
    let keyframe = from_hex_lines(KEYFRAME);
    let expected = from_hex_lines(EXPECTED);

    let kf_only = decode_vp9_sequence(&[&keyframe]).expect("decode keyframe");
    let kf_bytes = kf_only[0].to_planar_bytes();

    // The keyframe is slot-refreshed into all eight slots; re-display a few
    // distinct indices and confirm each re-emits the keyframe.
    for idx in [0u8, 3, 7] {
        let se = show_existing_packet(idx);
        let frames = decode_vp9_sequence(&[&keyframe, &se]).expect("kf + show_existing");
        assert_eq!(frames.len(), 2, "keyframe shown + the re-displayed frame");
        assert_eq!(
            frames[1].to_planar_bytes(),
            kf_bytes,
            "show_existing(idx={idx}) re-emits the keyframe"
        );
        assert_eq!(frames[1].width, 64);
        assert_eq!(frames[1].height, 64);
        assert_eq!(frames[1].bit_depth, 8);
        // The keyframe-sized prefix of the corpus output is reproduced.
        assert_eq!(frames[1].to_planar_bytes(), expected[..kf_bytes.len()]);
    }
}

/// §8.9 / §8.10 step 1: `show_existing_frame` resolves the *per-slot*
/// `FrameStore[ ]`, so re-displaying a slot the P-frame refreshed surfaces
/// the P-frame, while re-displaying a slot only the keyframe refreshed
/// surfaces the keyframe. The corpus P-frame has `refresh_frame_flags =
/// 0x01` (slot 0 only), so after `[keyframe, P-frame]` slot 0 holds the
/// P-frame and slots 1..7 still hold the keyframe. This distinguishes the
/// slot indexing from a single "last shown frame" fallback.
#[test]
fn show_existing_frame_resolves_per_slot_after_pframe() {
    let keyframe = from_hex_lines(KEYFRAME);
    let pframe = from_hex_lines(PFRAME);

    let shown = decode_vp9_sequence(&[&keyframe, &pframe]).expect("decode shown sequence");
    assert_eq!(shown.len(), 2, "keyframe + P-frame both shown");
    let keyframe_bytes = shown[0].to_planar_bytes();
    let pframe_bytes = shown[1].to_planar_bytes();
    // The fixture is a genuine inter frame: the P-frame differs from the
    // keyframe, so the per-slot distinction below is observable.
    assert_ne!(
        keyframe_bytes, pframe_bytes,
        "P-frame reconstruction differs"
    );

    // show_existing(0): slot 0 was refreshed by the P-frame (refresh_mask
    // 0x01), so it must re-emit the P-frame.
    let se0 = show_existing_packet(0);
    let r0 = decode_vp9_sequence(&[&keyframe, &pframe, &se0]).expect("show_existing(0)");
    assert_eq!(r0.len(), 3, "keyframe + P-frame + re-displayed P-frame");
    assert_eq!(
        r0[2].to_planar_bytes(),
        pframe_bytes,
        "slot 0 (P-frame refresh) re-emits the P-frame"
    );

    // show_existing(1): slot 1 was last written by the keyframe (the
    // P-frame did not refresh it), so it must re-emit the keyframe.
    let se1 = show_existing_packet(1);
    let r1 = decode_vp9_sequence(&[&keyframe, &pframe, &se1]).expect("show_existing(1)");
    assert_eq!(r1.len(), 3);
    assert_eq!(
        r1[2].to_planar_bytes(),
        keyframe_bytes,
        "slot 1 (keyframe refresh, untouched by P-frame) re-emits the keyframe"
    );
}

/// A `show_existing_frame` packet pointing at a never-written slot is an
/// invalid bitstream (`FrameStore[ idx ]` is undefined). Decoding a lone
/// `show_existing_frame` before any frame populates the store must reject
/// rather than panic or silently emit nothing.
#[test]
fn show_existing_frame_unwritten_slot_is_invalid() {
    let se = show_existing_packet(5);
    let r = decode_vp9_sequence(&[&se]);
    assert!(r.is_err(), "show_existing of an unwritten slot must error");
}

/// Workspace-checkout cross-check: when the docs corpus is reachable,
/// re-parse the `i-frame-then-p-frame-64x64` IVF directly and confirm the
/// full two-frame sequence decodes byte-exact. Standalone CI (no docs
/// tree) is covered by the embedded pin above.
#[test]
fn docs_corpus_inter_fixture_decodes_byte_exact() {
    let base = std::path::Path::new("../../docs/video/vp9/fixtures/i-frame-then-p-frame-64x64");
    if !base.is_dir() {
        eprintln!("docs corpus not present; embedded pin covers this configuration");
        return;
    }
    let ivf = std::fs::read(base.join("input.ivf")).expect("input.ivf");
    let expected = std::fs::read(base.join("expected.yuv")).expect("expected.yuv");

    // IVF: 32-byte file header, then per-frame 12-byte headers (first 4
    // bytes = little-endian payload size).
    let mut payloads: Vec<Vec<u8>> = Vec::new();
    let hdr_len = u16::from_le_bytes([ivf[6], ivf[7]]) as usize;
    let mut off = hdr_len;
    while off + 12 <= ivf.len() {
        let size =
            u32::from_le_bytes([ivf[off], ivf[off + 1], ivf[off + 2], ivf[off + 3]]) as usize;
        let start = off + 12;
        payloads.push(ivf[start..start + size].to_vec());
        off = start + size;
    }
    let refs: Vec<&[u8]> = payloads.iter().map(|p| p.as_slice()).collect();
    let frames = decode_vp9_sequence(&refs).expect("decode sequence");
    let mut got = Vec::new();
    for f in &frames {
        got.extend(f.to_planar_bytes());
    }
    assert_eq!(got.len(), expected.len(), "planar length");
    let diffs = got
        .iter()
        .zip(expected.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(diffs, 0, "{diffs} differing bytes vs expected.yuv");
}

/// Split an IVF stream into its per-frame VP9 chunks (32-byte file header,
/// then per-frame 12-byte headers whose first 4 bytes are the little-endian
/// payload size). IVF carriage is a test-fixture convenience here, not a
/// codec responsibility — the codec consumes the raw VP9 chunks.
fn ivf_chunks(ivf: &[u8]) -> Vec<Vec<u8>> {
    let mut payloads: Vec<Vec<u8>> = Vec::new();
    let hdr_len = u16::from_le_bytes([ivf[6], ivf[7]]) as usize;
    let mut off = hdr_len;
    while off + 12 <= ivf.len() {
        let size =
            u32::from_le_bytes([ivf[off], ivf[off + 1], ivf[off + 2], ivf[off + 3]]) as usize;
        let start = off + 12;
        if start + size > ivf.len() {
            break;
        }
        payloads.push(ivf[start..start + size].to_vec());
        off = start + size;
    }
    payloads
}

/// §6.4.11 / §6.5 / §8.5.2 / §8.10: the four-frame `frame-parallel-mode`
/// fixture (`error_resilient=1`, `parallel_mode=1`, `refresh_ctx=0` on every
/// frame — i.e. no inter-frame entropy adaptation, which is exactly the
/// crate's per-frame-reset context model) decodes byte-exact. This is a
/// keyframe followed by *three* consecutive P-frames at 64x64, exercising
/// the §8.10 reference threading across more than the single P-frame the
/// embedded pin covers. The §7.2.5 error-resilient + frame-parallel header
/// path is validated end-to-end against real pixels here.
#[test]
fn frame_parallel_mode_four_frame_sequence_byte_exact() {
    let base = std::path::Path::new("../../docs/video/vp9/fixtures/frame-parallel-mode");
    if !base.is_dir() {
        eprintln!("docs corpus not present; frame-parallel-mode is docs-gated");
        return;
    }
    let ivf = std::fs::read(base.join("input.ivf")).expect("input.ivf");
    let expected = std::fs::read(base.join("expected.yuv")).expect("expected.yuv");

    let payloads = ivf_chunks(&ivf);
    assert_eq!(payloads.len(), 4, "four IVF frames");
    let refs: Vec<&[u8]> = payloads.iter().map(|p| p.as_slice()).collect();
    let frames = decode_vp9_sequence(&refs).expect("decode frame-parallel sequence");
    assert_eq!(frames.len(), 4, "four shown frames");

    let mut got = Vec::new();
    for f in &frames {
        got.extend(f.to_planar_bytes());
    }
    assert_eq!(got.len(), expected.len(), "planar length");
    let diffs = got
        .iter()
        .zip(expected.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        diffs, 0,
        "{diffs} differing bytes vs frame-parallel-mode expected.yuv"
    );
}

/// §6.1.2 entropy-context threading end-to-end: the `profile-0-yuv420-8bit`
/// "common path" fixture is a keyframe + three P-frames at 128x128 (a 2x2
/// superblock grid) with `tx_mode=TX_MODE_SELECT`, deep partitions down to
/// 4x4, sub-8x8 inter blocks, and — critically — `error_resilient_mode=0` +
/// `refresh_frame_context=1`. Unlike the error-resilient `frame-parallel-mode`
/// fixture, each P-frame here `load_probs( )`-es the bank the *previous* frame
/// `save_probs( )`-ed, so the compressed-header forward updates compound onto
/// the prior frame's tables rather than the §10.5 defaults. Without that
/// threading the arithmetic decoder desynchronises partway through the second
/// superblock; with it the whole four-frame sequence reconstructs byte-exact.
#[test]
fn profile0_common_path_inter_sequence_byte_exact() {
    let base = std::path::Path::new("../../docs/video/vp9/fixtures/profile-0-yuv420-8bit");
    if !base.is_dir() {
        eprintln!("docs corpus not present; profile-0 is docs-gated");
        return;
    }
    let ivf = std::fs::read(base.join("input.ivf")).expect("input.ivf");
    let expected = std::fs::read(base.join("expected.yuv")).expect("expected.yuv");

    let payloads = ivf_chunks(&ivf);
    assert_eq!(payloads.len(), 4, "four IVF frames");
    let refs: Vec<&[u8]> = payloads.iter().map(|p| p.as_slice()).collect();
    let frames = decode_vp9_sequence(&refs).expect("decode profile-0 sequence");
    assert_eq!(frames.len(), 4, "four shown frames");

    let mut got = Vec::new();
    for f in &frames {
        got.extend(f.to_planar_bytes());
    }
    assert_eq!(got.len(), expected.len(), "planar length");
    let diffs = got
        .iter()
        .zip(expected.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        diffs, 0,
        "{diffs} differing bytes vs profile-0 expected.yuv"
    );
}

/// Annex B: the `frame-parallel-mode` IVF chunks carry no superframe index,
/// so [`split_superframe`] passes each chunk through unchanged — the §B.4
/// single-frame fallback. Confirms the split is transparent on
/// non-superframe content and the decode is identical when routed through
/// it (the canonical demux order: IVF chunk -> superframe split -> decode).
#[test]
fn superframe_split_is_transparent_on_plain_chunks() {
    let base = std::path::Path::new("../../docs/video/vp9/fixtures/frame-parallel-mode");
    if !base.is_dir() {
        eprintln!("docs corpus not present; docs-gated");
        return;
    }
    let ivf = std::fs::read(base.join("input.ivf")).expect("input.ivf");
    let expected = std::fs::read(base.join("expected.yuv")).expect("expected.yuv");

    let payloads = ivf_chunks(&ivf);
    // Route each IVF chunk through the Annex B split before decode.
    let mut split: Vec<Vec<u8>> = Vec::new();
    for p in &payloads {
        for f in split_superframe(p) {
            split.push(f.to_vec());
        }
    }
    assert_eq!(split.len(), 4, "no superframes -> one frame per chunk");
    let refs: Vec<&[u8]> = split.iter().map(|p| p.as_slice()).collect();
    let frames = decode_vp9_sequence(&refs).expect("decode");
    let mut got = Vec::new();
    for f in &frames {
        got.extend(f.to_planar_bytes());
    }
    assert_eq!(got, expected, "split-then-decode byte-exact");
}

/// §8.9 `show_existing_frame` corpus: the `show-existing-frame` fixture is a
/// 24-frame `auto-alt-ref=2` stream whose visible frames at ARF-release time
/// are `show_existing_frame=1` packets re-displaying a reference slot. The
/// stream uses `error_resilient_mode=0` + `refresh_frame_context=1`, so it
/// requires §6.1.2 `load_probs( ) / save_probs( )` entropy-context threading:
/// each inter frame's compressed-header forward updates fold onto the prior
/// frame's saved tables. With that threading the full 24-frame sequence —
/// keyframe, the hidden alt-ref frames, the `show_existing_frame` re-displays,
/// and every visible P-frame — decodes byte-exact against `expected.yuv`.
#[test]
fn show_existing_corpus_decodes_byte_exact() {
    let base = std::path::Path::new("../../docs/video/vp9/fixtures/show-existing-frame");
    if !base.is_dir() {
        eprintln!("docs corpus not present; docs-gated");
        return;
    }
    let ivf = std::fs::read(base.join("input.ivf")).expect("input.ivf");
    let expected = std::fs::read(base.join("expected.yuv")).expect("expected.yuv");

    // Demux: IVF chunk -> Annex B superframe split -> sub-frames.
    let mut sub: Vec<Vec<u8>> = Vec::new();
    for p in &ivf_chunks(&ivf) {
        for f in split_superframe(p) {
            sub.push(f.to_vec());
        }
    }
    assert!(sub.len() > 1, "fixture has a keyframe plus inter frames");

    let refs: Vec<&[u8]> = sub.iter().map(|p| p.as_slice()).collect();
    let frames = decode_vp9_sequence(&refs).expect("decode full sequence");
    let mut got = Vec::new();
    for f in &frames {
        got.extend(f.to_planar_bytes());
    }
    assert_eq!(got.len(), expected.len(), "24 visible frames planar length");
    let diffs = got
        .iter()
        .zip(expected.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(diffs, 0, "{diffs} differing bytes vs expected.yuv");
}

/// Hidden-ARF superframe corpus + the §8.1 step-2 loop-filter gate: the
/// `superframe-2` fixture (16 shown frames at 64x64, `yac_qi≈4-15`,
/// superframes carrying hidden `show_frame=0` alt-ref frames) decodes
/// byte-exact end-to-end. Every frame codes `loop_filter_level=0` with
/// `lf_delta_enabled=1` and the +1 INTRA ref-delta, so this pins §8.1
/// step 2 ("if loop_filter_level is not equal to 0, the loop filter
/// process ... is invoked") — without the frame-level gate the §8.8.1
/// `LvlLookup` lifts intra edges to `lvl=1` and filters them, the
/// historic ±1-per-frame divergence. Also exercises §8.10 hidden-frame
/// reference updates and per-frame `frame_context_idx` switching.
#[test]
fn superframe2_sequence_byte_exact() {
    let base = std::path::Path::new("../../docs/video/vp9/fixtures/superframe-2");
    if !base.is_dir() {
        eprintln!("docs corpus not present; docs-gated");
        return;
    }
    let ivf = std::fs::read(base.join("input.ivf")).expect("input.ivf");
    let expected = std::fs::read(base.join("expected.yuv")).expect("expected.yuv");

    let mut sub: Vec<Vec<u8>> = Vec::new();
    for p in &ivf_chunks(&ivf) {
        for f in split_superframe(p) {
            sub.push(f.to_vec());
        }
    }
    assert!(
        sub.len() > 16,
        "superframes carry hidden alt-ref sub-frames"
    );

    let refs: Vec<&[u8]> = sub.iter().map(|p| p.as_slice()).collect();
    let frames = decode_vp9_sequence(&refs).expect("decode superframe-2 sequence");
    assert_eq!(frames.len(), 16, "sixteen shown frames");

    let mut got = Vec::new();
    for f in &frames {
        got.extend(f.to_planar_bytes());
    }
    assert_eq!(got.len(), expected.len(), "16-frame planar length");
    // 64x64 4:2:0 => 6144 bytes per frame; report the first bad frame.
    let frame_bytes = 64 * 64 * 3 / 2;
    for k in 0..16usize {
        let s = k * frame_bytes;
        let e = s + frame_bytes;
        let diffs = got[s..e]
            .iter()
            .zip(&expected[s..e])
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(diffs, 0, "frame {k}: {diffs} differing bytes");
    }
}

/// Whole-corpus sweep: every staged fixture — single-frame and
/// multi-frame alike — decodes **fully byte-exact** through
/// [`decode_vp9_sequence`] (IVF demux → Annex B superframe split →
/// sequence decode → planar packing). Beyond the per-fixture tests, this
/// pins the corpus inter paths that only appear here: the profile-1/2/3
/// P-frames run §8.5.2 motion compensation at 4:4:4 chroma (no MV
/// averaging) and 10/12-bit sample depth (bit-depth-scaled convolution
/// clamps), and the two `auto-alt-ref` streams carry real compound
/// (`ref_frame[1] > NONE`) blocks — 64 in `superframe-2`, 117 in
/// `show-existing-frame` — so §8.5.2's `Round2( p0 + p1, 1 )` compound
/// average is corpus-validated, not just unit-tested.
#[test]
fn full_corpus_sequences_byte_exact() {
    let root = std::path::Path::new("../../docs/video/vp9/fixtures");
    if !root.is_dir() {
        eprintln!("docs corpus not present; docs-gated");
        return;
    }
    for name in [
        "tiny-i-only-16x16",
        "i-frame-then-p-frame-64x64",
        "profile-0-yuv420-8bit",
        "profile-1-yuv444-8bit",
        "profile-2-yuv420-10bit",
        "profile-3-yuv444-10bit",
        "profile-3-yuv444-12bit",
        "lossless-i-only",
        "tile-cols-2",
        "frame-parallel-mode",
        "superframe-2",
        "show-existing-frame",
        "segments-aq-mode",
        "q-low",
        "q-high",
        "bit-depth-10-rgb",
        // Round-406 corpus extensions (black-box validator generation;
        // see each fixture's notes.md): 4:2:2 inter (profile 1,
        // subsampling_x=1 / subsampling_y=0 chroma geometry through the
        // whole §8.5.2 + §8.8 chain) and a mid-GOP scene cut that codes
        // 62 intra blocks inside P-frames (the §6.4.13 is_inter=0 arm +
        // §8.5.1 intra prediction inside an inter frame).
        "profile-1-yuv422-8bit-inter",
        "intra-blocks-in-inter",
        // 176x144 12-frame GOP: the corpus's first non-multiple-of-64
        // dimensions (partial superblocks on the right column + bottom
        // row) — §6.4.3-admitted inter leaves overhang the frame edge,
        // pinning the clipped §8.5.2 prediction store — plus long
        // multi-superblock-row motion under continuous zoom.
        "qcif-inter-gop",
        // Round-409 corpus extension: the corpus's first
        // `frame_parallel_decoding_mode=0` stream — every frame runs the
        // full §6.1.2 refresh_probs( ) backward adaptation (§8.4.3
        // adapt_coef_probs from the §9.3.4 counts incl. the errata-#249
        // more_coefs special case, §8.4.4 adapt_noncoef_probs on the
        // P-frames), so byte-exactness proves the whole counting +
        // adaptation chain: any miscount desynchronises the next frame's
        // entropy decode.
        "backward-adaptation",
        // Round-409 corpus extension: the corpus's first SCALED-reference
        // stream (mid-GOP coded-size changes: 128 → 64 → 128 → 96 → 64,
        // every inter frame §8.5.2.3-sampling a differently-sized live
        // reference — the 2x conformance extreme, a 1/2x upscale, the
        // fractional 4/3 ratio, and a scaled NEWMV). Self-encoded by the
        // in-crate writers (deterministic; see notes.md), expected.yuv is
        // a black-box reference decode.
        "scaled-reference",
        // Round-409 corpus extensions (black-box generation, notes.md in
        // each dir): lossless INTER (the §8.7.2 WHT on inter residuals —
        // the corpus's lossless coverage was keyframe-only), a two-tile-
        // column inter GOP (the existing tile fixture is intra-only;
        // this one runs per-tile §9.2 coder brackets + tile-boundary
        // context resets on real P-frames), and a 10-bit profile-2 GOP
        // with frame_parallel_decoding_mode=0 (§8.4 backward adaptation
        // over high-bit-depth CAT6 token counts).
        "lossless-inter",
        "tiles-2col-inter",
        "hbd-backward-adaptation",
        // Round-412 corpus extensions (black-box generation; notes.md in
        // each dir). The two partial-MI 58x36 streams are the corpus's
        // first frames whose luma dimensions are not multiples of 8 on
        // BOTH axes (MiCols*8 = 64 > 58, MiRows*8 = 40 > 36): §6.4 codes
        // blocks overhanging the visible frame, the 4:2:0 chroma planes
        // are odd-width (29), and — critically — the §8.8.2 step-13
        // onScreen predicate keeps filtering edges *between* the visible
        // boundary and the MI-grid boundary. The 444 stream is the
        // regression stream for the round-412 filter-extent fix: its
        // loop filter fires across the y == FrameHeight edge, reading
        // real reconstructed overhang samples (visible-extent clamping
        // diverges on the bottom visible row).
        "partial-mi-58x36-yuv420",
        "partial-mi-58x36-yuv444",
    ] {
        let base = root.join(name);
        let ivf = std::fs::read(base.join("input.ivf")).expect("input.ivf");
        let expected = std::fs::read(base.join("expected.yuv")).expect("expected.yuv");

        let mut sub: Vec<Vec<u8>> = Vec::new();
        for p in &ivf_chunks(&ivf) {
            for f in split_superframe(p) {
                sub.push(f.to_vec());
            }
        }
        let refs: Vec<&[u8]> = sub.iter().map(|p| p.as_slice()).collect();
        let frames =
            decode_vp9_sequence(&refs).unwrap_or_else(|e| panic!("{name}: decode error {e:?}"));

        let mut got = Vec::new();
        for f in &frames {
            got.extend(f.to_planar_bytes());
        }
        assert_eq!(got.len(), expected.len(), "{name}: planar length");
        let diffs = got
            .iter()
            .zip(expected.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(diffs, 0, "{name}: {diffs} differing bytes vs expected.yuv");
    }
}

/// §6.1.2 refresh_probs( ) backward adaptation, corpus-pinned: the
/// `backward-adaptation` fixture is the corpus's only
/// `frame_parallel_decoding_mode=0` stream. This test asserts the header
/// flags that make it exercise the §8.4 path — every frame carries
/// `error_resilient_mode=0`, `frame_parallel_decoding_mode=0` and
/// `refresh_frame_context=1`, and the GOP has real P-frames — so the
/// byte-exact sweep in `full_corpus_sequences_byte_exact` genuinely
/// proves the §9.3.4 counting (incl. the errata-#249 `more_coefs`
/// special case) and the §8.4.3 / §8.4.4 adaptation: each frame's
/// entropy decode runs on probabilities adapted from the previous
/// frame's counts, so any miscount desynchronises the §9.2 boolean
/// decoder almost immediately.
#[test]
fn backward_adaptation_fixture_flags_pin_the_refresh_probs_path() {
    let base = std::path::Path::new("../../docs/video/vp9/fixtures/backward-adaptation");
    if !base.is_dir() {
        eprintln!("docs corpus not present; docs-gated");
        return;
    }
    let ivf = std::fs::read(base.join("input.ivf")).expect("input.ivf");
    let mut n_frames = 0usize;
    let mut n_inter = 0usize;
    let mut last_color: Option<oxideav_vp9::ColorConfig> = None;
    for p in &ivf_chunks(&ivf) {
        for f in split_superframe(p) {
            let ref_dims = vec![(176u32, 144u32); 8];
            let ref_state = last_color.map(|cc| oxideav_vp9::RefFrameState {
                ref_dims: &ref_dims,
                color_config: cc,
            });
            let hdr = oxideav_vp9::parse_uncompressed_header_with_refs(f, ref_state)
                .expect("uncompressed header");
            if hdr.show_existing_frame {
                continue;
            }
            assert!(!hdr.error_resilient_mode, "error_resilient must be 0");
            assert!(
                !hdr.frame_parallel_decoding_mode,
                "frame {n_frames}: frame_parallel_decoding_mode must be 0"
            );
            assert!(
                hdr.refresh_frame_context,
                "frame {n_frames}: refresh_frame_context must be 1"
            );
            if !matches!(hdr.frame_type, oxideav_vp9::FrameType::KeyFrame) && !hdr.intra_only {
                n_inter += 1;
            }
            last_color = Some(hdr.color_config);
            n_frames += 1;
        }
    }
    assert!(n_frames >= 8, "GOP long enough to compound adaptation");
    assert!(n_inter >= 7, "real P-frames present ({n_inter})");
}

/// Round-409 corpus extensions: assert each new fixture's headers carry
/// the flags that make it exercise its target feature class, so the
/// byte-exact sweep can't silently degrade into re-testing an existing
/// path (e.g. if a regenerated stream lost its tiling or lossless
/// flag).
#[test]
fn round_409_fixture_flags_pin_their_feature_classes() {
    let root = std::path::Path::new("../../docs/video/vp9/fixtures");
    if !root.is_dir() {
        eprintln!("docs corpus not present; docs-gated");
        return;
    }
    // Per-fixture predicate over every non-show-existing frame header:
    // (name, check(frame_index, hdr)).
    type Check = fn(usize, &oxideav_vp9::Vp9FrameHeader);
    let cases: [(&str, Check); 3] = [
        ("lossless-inter", |i, hdr| {
            assert!(hdr.quantization.lossless, "frame {i}: lossless=1");
        }),
        ("tiles-2col-inter", |i, hdr| {
            // NOTE: tile ROWS on inter frames remain an encoder-tooling
            // gap — the black-box wrapper exposes a tile-rows knob but
            // emits tile_rows_log2 = 0 regardless (verified with row-mt
            // and realtime deadlines); a tile-rows fixture needs custom
            // encoder tooling, like the scaled-reference one did.
            assert_eq!(hdr.tile_info.tile_cols_log2, 1, "frame {i}: tile cols");
        }),
        ("hbd-backward-adaptation", |i, hdr| {
            assert_eq!(hdr.profile, 2, "frame {i}: profile 2");
            assert_eq!(hdr.color_config.bit_depth, 10, "frame {i}: 10-bit");
            assert!(!hdr.error_resilient_mode, "frame {i}");
            assert!(
                !hdr.frame_parallel_decoding_mode,
                "frame {i}: frame_parallel_decoding_mode must be 0"
            );
            assert!(hdr.refresh_frame_context, "frame {i}");
        }),
    ];
    for (name, check) in cases {
        let ivf = std::fs::read(root.join(name).join("input.ivf")).expect("input.ivf");
        let mut i = 0usize;
        let mut n_inter = 0usize;
        let mut last_color: Option<oxideav_vp9::ColorConfig> = None;
        let mut dims = (0u32, 0u32);
        for p in &ivf_chunks(&ivf) {
            for f in split_superframe(p) {
                let ref_dims = vec![dims; 8];
                let ref_state = last_color.map(|cc| oxideav_vp9::RefFrameState {
                    ref_dims: &ref_dims,
                    color_config: cc,
                });
                let hdr = oxideav_vp9::parse_uncompressed_header_with_refs(f, ref_state)
                    .unwrap_or_else(|e| panic!("{name} frame {i}: header parse {e:?}"));
                if hdr.show_existing_frame {
                    continue;
                }
                check(i, &hdr);
                if !matches!(hdr.frame_type, oxideav_vp9::FrameType::KeyFrame) && !hdr.intra_only {
                    n_inter += 1;
                }
                dims = (hdr.frame_width, hdr.frame_height);
                last_color = Some(hdr.color_config);
                i += 1;
            }
        }
        assert!(n_inter >= 3, "{name}: real P-frames present ({n_inter})");
    }
}

/// §6.4.14 / §8.1 segment-map threading + §7.2.10 segmentation feature
/// persistence: the `segments-aq-mode` fixture (per-segment AQ, 4 frames
/// at 128x128, `frame_parallel_decoding_mode=1` throughout) reconstructs
/// **byte-exact** against `expected.yuv` across all four frames. This pins:
///
/// * the §8.1 step-3 `PrevSegmentIds` refresh-only-on-`update_map` rule
///   (frames 1-3 predict their segment IDs from the keyframe's map);
/// * the §7.2.10 "`segmentation_update_data == 0` keeps the existing
///   values" rule — frames 1-3 carry `update_data=0`, so their per-segment
///   `SEG_LVL_ALT_Q` quantizer deltas persist from the keyframe's feature
///   table instead of resetting to zero (the historic frame-2/3
///   divergence: skip=0 blocks dequantized at `base_q_idx` instead of the
///   segment-adjusted qindex);
/// * per-segment quantizer selection on real inter frames across
///   `tx_mode=ALLOW_8X8` / `ALLOW_32X32` / `TX_MODE_SELECT` and
///   `PARTITION_NONE` / `PARTITION_SPLIT` superblocks, including sub-8x8
///   inter blocks carrying per-sub-block motion vectors.
#[test]
fn segments_aq_sequence_byte_exact() {
    let base = std::path::Path::new("../../docs/video/vp9/fixtures/segments-aq-mode");
    if !base.is_dir() {
        eprintln!("docs corpus not present; docs-gated");
        return;
    }
    let ivf = std::fs::read(base.join("input.ivf")).expect("input.ivf");
    let expected = std::fs::read(base.join("expected.yuv")).expect("expected.yuv");

    let mut sub: Vec<Vec<u8>> = Vec::new();
    for p in &ivf_chunks(&ivf) {
        for f in split_superframe(p) {
            sub.push(f.to_vec());
        }
    }
    let refs: Vec<&[u8]> = sub.iter().map(|p| p.as_slice()).collect();
    let frames = decode_vp9_sequence(&refs).expect("decode segments-aq sequence");
    assert_eq!(frames.len(), 4, "four shown frames");

    // 128x128 4:2:0 => Y 16384 + U 4096 + V 4096 = 24576 bytes/frame.
    let frame_bytes = 128 * 128 * 3 / 2;
    let mut got = Vec::new();
    for f in &frames {
        got.extend(f.to_planar_bytes());
    }
    assert_eq!(got.len(), expected.len(), "four-frame planar length");

    for k in 0..4usize {
        let s = k * frame_bytes;
        let e = s + frame_bytes;
        let diffs = got[s..e]
            .iter()
            .zip(&expected[s..e])
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(
            diffs, 0,
            "frame {k}: {diffs} differing bytes vs expected.yuv"
        );
    }
}
