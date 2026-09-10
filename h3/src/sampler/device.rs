//! Resident Euler latents. Multistep keeps its f64 CPU implementation until the
//! packaged Loom compiler supports the required f64 operations on gfx1151.

use crate::compile::Compiler;
use crate::error::{invalid, Result};
use crate::model::{AUDIO_CH, FINAL_N, VIDEO_PATCH};

pub(crate) struct Euler {
    pub audio: hrx::Buffer,
    pub video: hrx::Buffer,
    audio_kernel: crate::compile::Kernel,
    video_kernel: crate::compile::Kernel,
    audio_rows: usize,
    video_rows: usize,
}

impl Euler {
    pub fn matches(&self, audio_rows: usize, video_rows: usize) -> bool {
        self.audio_rows == audio_rows && self.video_rows == video_rows
    }

    pub fn new(
        stream: &mut hrx::Stream,
        c: &Compiler,
        audio_rows: usize,
        video_rows: usize,
    ) -> Result<Self> {
        if audio_rows == 0 || video_rows == 0 {
            return invalid("Euler requires audio and video rows");
        }
        let mut kernel = |rows: usize, channels: usize, row_offset: usize, col_offset: usize| {
            let cfg = [
                ("count", rows * channels),
                ("channels", channels),
                ("row_offset", row_offset),
                ("col_offset", col_offset),
                ("head_rows", audio_rows + video_rows),
            ]
            .map(|(key, value)| (format!("h3.sampler_update_f32.{key}"), value.to_string()))
            .to_vec();
            c.get(stream, "sampler_update_f32", "h3_sampler_update_f32", &cfg)
        };
        Ok(Self {
            audio_kernel: kernel(audio_rows, AUDIO_CH, 0, VIDEO_PATCH)?,
            video_kernel: kernel(video_rows, VIDEO_PATCH, audio_rows, 0)?,
            audio: stream.allocate(audio_rows * AUDIO_CH * 4)?,
            video: stream.allocate(video_rows * VIDEO_PATCH * 4)?,
            audio_rows,
            video_rows,
        })
    }

    pub fn upload(&self, stream: &mut hrx::Stream, audio: &[f32], video: &[f32]) -> Result<()> {
        if audio.len() != self.audio_rows * AUDIO_CH || video.len() != self.video_rows * VIDEO_PATCH
        {
            return invalid("Euler latent shape mismatch");
        }
        stream.upload(self.audio.binding(), crate::vvae::as_bytes(audio))?;
        stream.upload(self.video.binding(), crate::vvae::as_bytes(video))?;
        Ok(())
    }

    pub fn step(
        &self,
        stream: &mut hrx::Stream,
        head: hrx::View<'_>,
        audio: (f32, f32),
        video: (f32, f32),
    ) -> Result<()> {
        if head.len() < (self.audio_rows + self.video_rows) * FINAL_N * 4 {
            return invalid("Euler head output is too short");
        }
        for (handle, buffer, (sigma, ratio)) in [
            (&self.audio_kernel, &self.audio, audio),
            (&self.video_kernel, &self.video, video),
        ] {
            let kernel = handle.resolve(stream)?;
            let mut constants = hrx::Constants::new();
            constants.push(sigma)?;
            constants.push(ratio)?;
            // Safety: these kernels specialize on the owned latent extents and
            // the checked head extent. Each lane updates one unique latent;
            // both scalar arguments are f32, in declaration order.
            unsafe {
                stream.dispatch(
                    kernel,
                    [(buffer.bytes() / 4).div_ceil(256) as u32, 1, 1],
                    [256, 1, 1],
                    &constants,
                    &[buffer.binding(), head],
                )?;
            }
        }
        Ok(())
    }

    pub fn download(
        &self,
        stream: &mut hrx::Stream,
        audio: &mut [f32],
        video: &mut [f32],
    ) -> Result<()> {
        if audio.len() != self.audio_rows * AUDIO_CH || video.len() != self.video_rows * VIDEO_PATCH
        {
            return invalid("Euler latent shape mismatch");
        }
        stream.read_blocking(self.audio.binding(), crate::vvae::as_bytes_mut(audio))?;
        stream.read_blocking(self.video.binding(), crate::vvae::as_bytes_mut(video))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires gfx1151 and the packaged Loom compiler"]
    fn resident_euler_matches_cpu_across_steps_and_strided_heads() -> Result<()> {
        let mut stream = hrx::Stream::open()?;
        let compiler = Compiler::new(None, "");
        // Both launches include partial workgroups. Head channels belonging to
        // the other modality contain sentinels to detect incorrect gathering.
        let (na, nv) = (9, 7);
        let sampler = Euler::new(&mut stream, &compiler, na, nv)?;
        let mut audio: Vec<f32> = (0..na * AUDIO_CH).map(|i| i as f32 / 37.0 - 3.0).collect();
        let mut video: Vec<f32> = (0..nv * VIDEO_PATCH)
            .map(|i| i as f32 / 91.0 - 4.0)
            .collect();
        sampler.upload(&mut stream, &audio, &video)?;
        let mut head = vec![f32::NAN; (na + nv) * FINAL_N];
        let av: Vec<f32> = (0..audio.len()).map(|i| (i as f32 * 0.17).sin()).collect();
        let vv: Vec<f32> = (0..video.len()).map(|i| (i as f32 * 0.23).cos()).collect();
        for (i, value) in av.iter().enumerate() {
            head[(i / AUDIO_CH) * FINAL_N + VIDEO_PATCH + i % AUDIO_CH] = *value;
        }
        for (i, value) in vv.iter().enumerate() {
            head[(na + i / VIDEO_PATCH) * FINAL_N + i % VIDEO_PATCH] = *value;
        }
        let output = stream.allocate(head.len() * 4)?;
        stream.upload(output.binding(), crate::vvae::as_bytes(&head))?;
        for (sigma, ratio) in [(2.0, 0.1), (1.0, 1.0), (0.75, 0.3), (0.2, 0.0)] {
            sampler.step(
                &mut stream,
                output.binding(),
                (sigma, ratio),
                (sigma * 0.7, ratio),
            )?;
            crate::sampler::euler_update(&mut audio, &av, sigma, ratio);
            crate::sampler::euler_update(&mut video, &vv, sigma * 0.7, ratio);
        }
        let mut got_a = vec![0.0; audio.len()];
        let mut got_v = vec![0.0; video.len()];
        sampler.download(&mut stream, &mut got_a, &mut got_v)?;
        for (got, want) in got_a.iter().zip(&audio).chain(got_v.iter().zip(&video)) {
            assert_eq!(got.to_bits(), want.to_bits(), "GPU {got}, CPU {want}");
        }
        assert!(sampler
            .step(&mut stream, output.slice(0, 4), (1.0, 0.0), (1.0, 0.0))
            .is_err());
        Ok(())
    }
}
