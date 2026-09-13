//! Byte-level regressions over the whole pipeline: the same inputs must produce the same bytes.
//!
//! These are the checks that have caught every reordering mistake in this repository — a graph
//! recorded with a missing edge, a stage moved onto another stream, a path that resolved to the
//! wrong tree after a directory moved. None of those changes the types, and none of them fails to
//! compile. They lived for a long time as fixture files in a scratch directory driven by a shell
//! loop, which meant the project's real safety net was not in the project.
//!
//! Inputs are generated here rather than stored: the outputs would be gigabytes, and a digest
//! compares them exactly as well as the bytes did. What each constant below records is the state of
//! this pipeline at the moment it was written, which was verified byte-identical to the fixtures it
//! replaced, which were themselves checked against diffusers and ComfyUI. A change to one of these
//! digests is a change to the model's arithmetic, and needs that same chain re-run — not a new
//! constant pasted in.
//!
//! Checkpoints come from the Hugging Face cache; `H3_MODELS` is an optional override.
use h3_hrx::avae::{AudioVae, AUDIO_CH, HOP};
use h3_hrx::compile::Compiler;
use h3_hrx::dispatch::Profile;
use h3_hrx::vvae::VideoVae;
use h3_hrx::{shape_for, DenoiseParams, Sampler, Session, Tokenizer};

/// Deterministic standard normals, so an input is a seed rather than a file.
///
/// SplitMix64 and Box-Muller, both written out: a generator this test owns cannot drift with a
/// dependency, and every digest below is a claim about arithmetic that must not move.
struct Normals(u64);

impl Normals {
    fn bits(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Half-open (0, 1]: the zero is excluded because the logarithm below would take it to infinity.
    fn uniform(&mut self) -> f64 {
        ((self.bits() >> 11) as f64 + 1.0) * (1.0 / (1u64 << 53) as f64)
    }

    fn next(&mut self) -> f32 {
        let (u, v) = (self.uniform(), self.uniform());
        ((-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()) as f32
    }

    fn take(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next()).collect()
    }
}

fn compiler() -> Compiler {
    Compiler::new(
        std::env::var_os("HRX_LOOM_LIBRARY").map(std::path::PathBuf::from),
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("kernels"),
    )
}

fn digest_f32(values: &[f32]) -> String {
    hrx::bundle::digest(bytemuck::cast_slice(values))
}

/// Compare a digest against the one recorded for this case.
#[track_caller]
fn check(got: String, want: &str, case: &str) {
    assert_eq!(got, want, "{case}");
}

/// Every canvas the decoder tiles differently: one tile, several across, several down, and the
/// temporal chunking at both a short and a long frame count.
#[test]
#[ignore = "requires gfx1151, provisioned HRX and the video VAE checkpoint"]
fn video_decode_is_byte_stable_across_the_tilings() {
    const CASES: &[(i32, i32, i32, &str)] = &[
        (
            256,
            256,
            5,
            "de1b5ccb42ed5d2567e305f72ae504a1bb5bd168d9b469802cdc3ad74cb6c1f4",
        ),
        (
            256,
            512,
            39,
            "ef0e9201daedc4bdacda587ea8943359e44bdb34c5bfd59ee18a1dc3aceb5269",
        ),
        (
            320,
            320,
            56,
            "a5265ce69ba09c451317eab859534640d4a1b9f8a707de2a7a4fdd60cf99e116",
        ),
        (
            480,
            864,
            22,
            "85658b4ec1bddfc66c2dba92c81b8e16158f91c93aff5f4878224ee2dcdd2b04",
        ),
        (
            480,
            864,
            5,
            "fa4bddd83729027fab8ae70e0fbe3406f10db6e8a0bf3edda8de3d8fb912795d",
        ),
        (
            512,
            256,
            22,
            "4ae394a7621e067776c1a25b04ab82f2d78805b33005f2f08078aa31622c254c",
        ),
        (
            768,
            1344,
            5,
            "be2c587698034753f2e2c1591e6bef2f5b583b8ab75da9dac418c436f1c2a675",
        ),
    ];
    let mut stream = hrx::Stream::open().expect("stream");
    let c = compiler();
    let path = h3_hrx::models::Resolver::new()
        .offline(true)
        .find(h3_hrx::models::VIDEO_VAE)
        .expect("cached video VAE");
    // Safety: this test does not write to the checkpoint while it is mapped.
    let mut vae = unsafe { VideoVae::open(&mut stream, &path) }.expect("video VAE checkpoint");
    let mut prof = Profile::default();

    for &(height, width, frames, want) in CASES {
        let shape = shape_for(height, width, frames).expect("a shape this model serves");
        let latents = Normals(0x5eed)
            .take(24 * shape.latent_t as usize * shape.lat_h as usize * shape.lat_w as usize);
        let mut out = vec![0u8; shape.frames as usize * height as usize * width as usize * 3];
        vae.decode_video(&mut stream, &c, &mut prof, &shape, &latents, &mut out)
            .expect("decode");
        check(
            hrx::bundle::digest(&out),
            want,
            &format!("{height}x{width}x{frames}"),
        );
    }
}

/// Both directions of the audio VAE, at the lengths that exercise its padding: one latent, one
/// sample, a length on and a length just past the 800-sample hop, and a long clip.
#[test]
#[ignore = "requires gfx1151, provisioned HRX and the audio VAE checkpoint"]
fn audio_conversions_are_byte_stable_at_the_boundaries() {
    const DECODES: &[(usize, &str)] = &[
        (
            1,
            "841e6463f0859897dba1bf1b5a90cd8bb3d38242d5b9c27950516966ce6bfe82",
        ),
        (
            2,
            "5f9fb073b2319ce7297548d018349c70b9cc332a5f7f14d3db67b779004f3f8d",
        ),
        (
            5,
            "da3fceb30b7e891444e8d4fd571bc2477cdabb610f8dae80b272c665756af982",
        ),
        (
            40,
            "4dda0605634486530a43bf552fccd19b3ea0d7d92f773ba368a9b583e486da4e",
        ),
        (
            137,
            "203ea8e877f431e005922a9198d22d875f491394e37d8e6584d179a1100e931c",
        ),
        (
            500,
            "e49f5636df538fe84a28958062c167f170157bc5fdaa95b10c3639c43436b12f",
        ),
    ];
    const ENCODES: &[(usize, &str)] = &[
        (
            1,
            "9edfd51249d0a0b8ac6c5c4fc5d8a83f5be137d94abcd6e05908ba3ec63c699f",
        ),
        (
            800,
            "1dfaf2716822bc1a1f02f53fd23b2d7ec2a991f08a721af00926566d2e1eebf9",
        ),
        (
            801,
            "46e80478b35ef469ff692f8afb6f652a087131984dcb0d5d9f78512dc6b08748",
        ),
        (
            4000,
            "4baa7a6a6ecd842eaba8a0290f555f7198954287ed22268a6c5ff9220f493ac4",
        ),
        (
            12345,
            "fb523adeb96e0c1327bddda5a537925ce74b18771c777a4428d6d692e25e610a",
        ),
        (
            32000,
            "2260ceb4f6b7520e56f7aa5d9d4eefa587e26af63ddd6297096dd51ceb05f3e6",
        ),
        (
            109_600,
            "3a0db373563f03a9f4f291d7834e2740c15f5f3ff21fcf5ed7268cfa7d88097c",
        ),
        (
            400_000,
            "6911f3a6383e5fdb8892fa54fb6dbbd530011ca418ce827ab3247bb94c5430d2",
        ),
    ];
    let mut stream = hrx::Stream::open().expect("stream");
    let c = compiler();
    let path = h3_hrx::models::Resolver::new()
        .offline(true)
        .find(h3_hrx::models::AUDIO_VAE)
        .expect("cached audio VAE");
    // Safety: this test does not write to the checkpoint while it is mapped.
    let mut vae = unsafe { AudioVae::open(&mut stream, &path) }.expect("audio VAE checkpoint");
    let mut prof = Profile::default();

    for &(t, want) in DECODES {
        let latents = Normals(0xa0d10).take(2 * AUDIO_CH * t);
        let mut samples = vec![0.0f32; 2 * t * HOP];
        vae.decode(&mut stream, &c, &mut prof, &latents, t, &mut samples)
            .expect("decode");
        check(digest_f32(&samples), want, &format!("decode {t}"));
    }
    for &(n, want) in ENCODES {
        // scaled into the range a waveform occupies, since the encoder is not scale-free
        let samples: Vec<f32> = Normals(0x5a3f)
            .take(2 * n)
            .iter()
            .map(|v| v * 0.2)
            .collect();
        let (z, _) = vae
            .encode(&mut stream, &c, &mut prof, &samples, n)
            .expect("encode");
        check(digest_f32(&z), want, &format!("encode {n}"));
    }
}

/// Both samplers, a cached trajectory, and the two frame counts that pack differently.
///
/// The middle pair differs only in the sampler, and four steps is the shortest trajectory where
/// that shows: at two steps `ResMultistep` has no history to use and produces exactly what Euler
/// does, so a two-step pair would assert the same arithmetic twice.
///
/// One session for all of them: opening it maps and uploads the DiT and the text encoder, which is
/// most of the cost, and every case after the first reuses them.
#[test]
#[ignore = "requires gfx1151, provisioned HRX and the DiT and text encoder checkpoints"]
fn denoising_is_byte_stable_across_samplers_and_the_step_cache() {
    /// name, canvas, frames, steps, sampler, cache threshold, and the digests of the two latents.
    type Case = (
        &'static str,
        i32,
        i32,
        i32,
        usize,
        Sampler,
        f32,
        &'static str,
        &'static str,
    );
    const CASES: &[Case] = &[
        (
            "shortest",
            256,
            256,
            5,
            2,
            Sampler::Euler,
            0.0,
            "c64edbf4aa8c59a40a75378a6e49cce2af15dc353ab71aa9fbe5ed53dcd2c06c",
            "82bd2428e5d0b1107fc78577eefe7b95c40ad19e0f6b03b2f30f39838875599b",
        ),
        (
            "euler",
            256,
            256,
            5,
            4,
            Sampler::Euler,
            0.0,
            "630e4e2effeb6d39ddb3242488bd20c42f88d880e18e7432510b087c1dc348e0",
            "2740c9ccdd0a706e7bf252ffa3c1991fab09c6f61714ba3af06c1a0ad1f893b0",
        ),
        (
            "res",
            256,
            256,
            5,
            4,
            Sampler::ResMultistep,
            0.0,
            "f22cf0bd1930122dbfce3e6c9db5c26e244decf2db1e9712bb729705aadbb709",
            "4d9ad06a1825a36a68ccde0f76d912cec728c1c1124e60fd2c219da8a95e58f2",
        ),
        (
            "long-clip",
            256,
            256,
            22,
            4,
            Sampler::Euler,
            0.0,
            "c7732f32ab26d080e9ddf07f5c87565a8362cdbdbfd5ac092a710a1aab24538d",
            "9f4a72b17467dae97c14e3e00827cb7cd6d92e5ca3aaf72bcffc717042cf10f8",
        ),
        (
            "step-cache",
            256,
            256,
            22,
            6,
            Sampler::ResMultistep,
            0.15,
            "b91dc015f89185f972c0e4152a2558ca6e350ce1c79487680351de45c6a7c0c1",
            "333e54b3354e7643637e0f0275864334f1d72c7bcb52bb68d4e0ec36db9db39b",
        ),
    ];
    let ids = Tokenizer::new()
        .expect("tokenizer")
        .encode("a red fox trotting through a snowy forest at dawn, cinematic")
        .expect("encode the prompt");
    // Safety: this test does not write to the checkpoints while they are mapped.
    let mut session = unsafe { Session::new(h3_hrx::Config::default()) }.expect("session");

    for &(name, height, width, frames, steps, sampler, cache_threshold, video, audio) in CASES {
        let params = DenoiseParams {
            height,
            width,
            frames,
            steps,
            seed: 7,
            sampler,
            cache_threshold,
            ..DenoiseParams::default()
        };
        let latents = session
            .denoise(&ids, &params, h3_hrx::Noise::default(), &[], &[], None)
            .expect("denoise");
        check(digest_f32(&latents.video), video, &format!("{name} video"));
        check(digest_f32(&latents.audio), audio, &format!("{name} audio"));
    }
}
