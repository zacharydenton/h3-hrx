//! Runs the sampler over synthetic inputs and dumps every intermediate state.
//!
//!   sampler_dump <input.bin> <output.bin>
//!
//! The companion harness compiles the same loop out of the C implementation (deleted; see git history) by slicing the lines with
//! `sed`, so the two outputs are the shipped C text and this port on identical numbers. The comparison
//! is bitwise: nothing here is allowed to be a near miss.
use h3::layout::Schedule;
use h3::model::*;
use h3::sampler::*;
use std::io::{Read, Write};

fn f32s(r: &mut impl Read, n: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; n * 4];
    r.read_exact(&mut bytes).expect("short read");
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn put(w: &mut impl Write, v: &[f32]) {
    for x in v {
        w.write_all(&x.to_le_bytes()).unwrap();
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let input = args.next().expect("usage: sampler_dump <in> <out>");
    let output = args.next().expect("out");
    let mut r = std::io::BufReader::new(std::fs::File::open(input).expect("input"));

    let mut hdr = [0u8; 8 * 4 + 2 * 8];
    r.read_exact(&mut hdr).expect("header");
    let i = |k: usize| i32::from_le_bytes(hdr[k * 4..k * 4 + 4].try_into().unwrap()) as usize;
    let d = |k: usize| f64::from_le_bytes(hdr[32 + k * 8..40 + k * 8].try_into().unwrap());
    let (nv, na, steps) = (i(0), i(1), i(2));
    let res = i(3) != 0;
    let (t_len, h, w, a) = (i(4), i(5), i(6), i(7));
    let (shift_v, shift_a) = (d(0), d(1));
    let ascale = shift_v / shift_a;

    let noise_v = f32s(&mut r, LATENT_CH * t_len * h * w);
    let noise_a = f32s(&mut r, 2 * AUDIO_CH * a);

    let mut vrows = vec![0.0f32; nv * VIDEO_PATCH];
    let mut arows = vec![0.0f32; na * AUDIO_CH];
    tensor_to_rows(&noise_v, &mut vrows, t_len, h, w);
    audio_tensor_to_rows(&noise_a, &mut arows, a);
    // carry = sigma_a / sigma_v is one at sigma_v = 1, so the carried variable starts as the noise
    let mut yrows = if res { arows.clone() } else { Vec::new() };
    let (mut old_v, mut old_a) = (Vec::new(), Vec::new());

    let sv = Schedule::new(steps, shift_v);
    let sa = Schedule::new(steps, shift_a);
    let mut out = std::io::BufWriter::new(std::fs::File::create(output).expect("output"));

    for step in 0..sv.timesteps.len() {
        if res {
            let carry = sa.sigmas[step] / sv.sigmas[step];
            for (x, y) in arows.iter_mut().zip(&yrows) {
                *x = y * carry;
            }
        }
        let out32 = f32s(&mut r, (na + nv) * FINAL_N);
        // the pack's rows are audio first, then video; the final head emits VIDEO_PATCH video channels
        // and AUDIO_CH audio channels side by side in each row
        let vout: Vec<f32> = (0..nv * VIDEO_PATCH)
            .map(|i| out32[(na + i / VIDEO_PATCH) * FINAL_N + i % VIDEO_PATCH])
            .collect();
        let aout: Vec<f32> = (0..na * AUDIO_CH)
            .map(|i| out32[(i / AUDIO_CH) * FINAL_N + VIDEO_PATCH + i % AUDIO_CH])
            .collect();

        let (sg_v, sg_a) = (sv.sigmas[step], sa.sigmas[step]);
        if !res {
            euler_update(&mut vrows, &vout, sg_v, sv.sigmas[step + 1] / sg_v);
            euler_update(&mut arows, &aout, sg_a, sa.sigmas[step + 1] / sg_a);
        } else {
            let den_v = denoised_video(&vrows, &vout, sg_v);
            let den_a = denoised_audio(&yrows, &arows, &aout, sg_v, sg_a, ascale);
            advance(&mut vrows, &den_v, Some(&old_v), &sv.sigmas, step);
            advance(&mut yrows, &den_a, Some(&old_a), &sv.sigmas, step);
            old_v = den_v;
            old_a = den_a;
        }
        put(&mut out, &vrows);
        put(&mut out, &arows);
        put(&mut out, if res { &yrows } else { &arows });
    }

    let mut video = vec![0.0f32; LATENT_CH * t_len * h * w];
    let mut audio = vec![0.0f32; 2 * AUDIO_CH * a];
    rows_to_tensor(&vrows, &mut video, t_len, h, w);
    audio_rows_to_tensor(
        if res { &yrows } else { &arows },
        &mut audio,
        a,
        res.then_some(ascale),
    );
    put(&mut out, &video);
    put(&mut out, &audio);
    out.flush().unwrap();
}
