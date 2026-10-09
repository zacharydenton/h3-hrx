//! ResMultistep state retained across model evaluations.
use crate::compile::{Compiler, Kernel};
use crate::error::{invalid, Result};
use crate::model::{AUDIO_CH, FINAL_N, VIDEO_PATCH};

pub(crate) struct Multistep {
    pub audio: hrx::Buffer,
    pub video: hrx::Buffer,
    carried_audio: hrx::Buffer,
    old_audio: hrx::Buffer,
    old_video: hrx::Buffer,
    audio_kernel: [Kernel; 2],
    video_kernel: [Kernel; 2],
    audio_rows: usize,
    video_rows: usize,
}

impl Multistep {
    pub fn matches(&self, audio_rows: usize, video_rows: usize) -> bool {
        self.audio_rows == audio_rows && self.video_rows == video_rows
    }

    pub fn new(stream: &mut hrx::Stream, c: &Compiler, na: usize, nv: usize) -> Result<Self> {
        if na == 0 || nv == 0 {
            return invalid("ResMultistep requires audio and video rows");
        }
        let mut kernel = |rows, channels, row_offset, col_offset, second_order| {
            let cfg = [
                ("count", rows * channels),
                ("channels", channels),
                ("row_offset", row_offset),
                ("col_offset", col_offset),
                ("head_rows", na + nv),
                ("second_order", second_order),
            ]
            .map(|(key, value)| (format!("h3.res_multistep.{key}"), value.to_string()))
            .to_vec();
            c.get(stream, "res_multistep", "h3_res_multistep", &cfg)
        };
        Ok(Self {
            audio_kernel: [
                kernel(na, AUDIO_CH, 0, VIDEO_PATCH, 0)?,
                kernel(na, AUDIO_CH, 0, VIDEO_PATCH, 1)?,
            ],
            video_kernel: [
                kernel(nv, VIDEO_PATCH, na, 0, 0)?,
                kernel(nv, VIDEO_PATCH, na, 0, 1)?,
            ],
            audio: stream.allocate(na * AUDIO_CH * 4)?,
            carried_audio: stream.allocate(na * AUDIO_CH * 4)?,
            old_audio: stream.allocate(na * AUDIO_CH * 4)?,
            video: stream.allocate(nv * VIDEO_PATCH * 4)?,
            old_video: stream.allocate(nv * VIDEO_PATCH * 4)?,
            audio_rows: na,
            video_rows: nv,
        })
    }

    pub fn upload(
        &self,
        stream: &mut hrx::Stream,
        audio: &[f32],
        video: &[f32],
        carry: f32,
    ) -> Result<()> {
        if audio.len() != self.audio_rows * AUDIO_CH || video.len() != self.video_rows * VIDEO_PATCH
        {
            return invalid("ResMultistep latent shape mismatch");
        }
        stream.upload(self.carried_audio.binding(), bytemuck::cast_slice(audio))?;
        let input: Vec<f32> = audio.iter().map(|y| y * carry).collect();
        stream.upload(self.audio.binding(), bytemuck::cast_slice(&input))?;
        stream.upload(self.video.binding(), bytemuck::cast_slice(video))?;
        // The first step overwrites every history element without reading it.
        Ok(())
    }

    pub fn step(
        &self,
        stream: &mut hrx::Stream,
        head: hrx::View<'_>,
        video_sigmas: &[f32],
        audio_sigmas: &[f32],
        step: usize,
        ascale: f64,
    ) -> Result<()> {
        if step + 1 >= video_sigmas.len() || audio_sigmas.len() != video_sigmas.len() {
            return invalid("ResMultistep schedule mismatch");
        }
        if head.len() < (self.audio_rows + self.video_rows) * FINAL_N * 4 {
            return invalid("ResMultistep head output is too short");
        }
        let (sigma, sigma_next) = (video_sigmas[step], video_sigmas[step + 1]);
        let second = step > 0 && sigma_next != 0.0;
        let (decay, h, b1, b2) = if second {
            let t = -f64::from(sigma).ln();
            let t_next = -f64::from(sigma_next).ln();
            let t_prev = -f64::from(video_sigmas[step - 1]).ln();
            let h = t_next - t;
            let c2 = (t_prev - t) / h;
            let phi1 = (-h).exp_m1() / -h;
            let phi2 = (phi1 - 1.0) / -h;
            ((-h).exp(), h, phi1 - phi2 / c2, phi2 / c2)
        } else {
            (0.0, 0.0, 0.0, 0.0)
        };
        let mut constants = hrx::Constants::new();
        for value in [
            1.0 - ascale,
            1.0 + (ascale - 1.0) * f64::from(audio_sigmas[step]),
            f64::from(sigma),
            decay,
            h,
            b1,
            b2,
        ] {
            constants.push(value)?;
        }
        constants.push(sigma)?;
        constants.push(sigma_next / sigma)?;
        // No following input exists after the final step; avoid a 0/0 carry.
        constants.push(if sigma_next == 0.0 {
            1.0
        } else {
            audio_sigmas[step + 1] / sigma_next
        })?;
        for (handle, x, old, input) in [
            (
                &self.audio_kernel[usize::from(second)],
                &self.carried_audio,
                &self.old_audio,
                &self.audio,
            ),
            (
                &self.video_kernel[usize::from(second)],
                &self.video,
                &self.old_video,
                &self.video,
            ),
        ] {
            let kernel = handle.resolve(stream)?;
            let launch = handle.launch_config(kernel, &[])?;
            // Safety: dimensions specialize the checked head and owned buffers.
            // Constants follow the source's seven f64 then three f32 parameters.
            // Video's unused audio-input binding aliases x but is never accessed.
            unsafe {
                stream.dispatch(
                    kernel,
                    launch.workgroup_count,
                    launch.workgroup_size,
                    &constants,
                    &[x.binding(), head, old.binding(), input.binding()],
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
            return invalid("ResMultistep latent shape mismatch");
        }
        stream.read_blocking(
            self.carried_audio.binding(),
            bytemuck::cast_slice_mut(audio),
        )?;
        stream.read_blocking(self.video.binding(), bytemuck::cast_slice_mut(video))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg_attr(
        not(feature = "gpu-tests"),
        ignore = "requires gfx1151 and packaged Loom"
    )]
    fn resident_multistep_matches_cpu_trajectories_and_restarts() -> Result<()> {
        let mut stream = hrx::Stream::open()?;
        let compiler = Compiler::new(None, "");
        let (na, nv) = (9, 7);
        let sampler = Multistep::new(&mut stream, &compiler, na, nv)?;
        let output = stream.allocate((na + nv) * FINAL_N * 4)?;
        for (steps, shift_v, shift_a, finish) in [
            (2, 12.0, 3.0, None),
            (21, 12.0, 3.0, None),
            (21, 3.0, 12.0, Some(4)),
            (9, 3.7, 3.7, None),
        ] {
            let mut sv = crate::layout::Schedule::new(steps, shift_v);
            let mut sa = crate::layout::Schedule::new(steps, shift_a);
            let scale = shift_v / shift_a;
            let values = |count| {
                (0..count)
                    .map(|i| match i % 11 {
                        0 => 0.0,
                        1 => -0.0,
                        2 => f32::from_bits(1),
                        3 => -f32::MIN_POSITIVE,
                        4 => 1.0e20,
                        5 => -1.0e20,
                        _ => ((i * 37 % 101) as f32 - 50.0) / 7.0,
                    })
                    .collect::<Vec<f32>>()
            };
            let mut y = values(na * AUDIO_CH);
            let mut video = values(nv * VIDEO_PATCH);
            let mut old_a = vec![0.0; y.len()];
            let mut old_v = vec![0.0; video.len()];
            let mut da = vec![0.0; y.len()];
            let mut dv = vec![0.0; video.len()];
            sampler.upload(&mut stream, &y, &video, sa.sigmas[0] / sv.sigmas[0])?;
            let mut step = 0;
            while step < sv.timesteps.len() {
                let audio: Vec<f32> = y
                    .iter()
                    .map(|v| v * (sa.sigmas[step] / sv.sigmas[step]))
                    .collect();
                let mut device_input = vec![0.0; audio.len()];
                stream.read_blocking(
                    sampler.audio.binding(),
                    bytemuck::cast_slice_mut(&mut device_input),
                )?;
                assert_bits(&device_input, &audio, "audio input", step);
                let av: Vec<f32> = (0..y.len())
                    .map(|i| ((i as f32 + step as f32) * 0.17).sin())
                    .collect();
                let vv: Vec<f32> = (0..video.len())
                    .map(|i| ((i as f32 + step as f32) * 0.23).cos())
                    .collect();
                let mut head = vec![f32::NAN; (na + nv) * FINAL_N];
                for (i, v) in av.iter().enumerate() {
                    head[i / AUDIO_CH * FINAL_N + VIDEO_PATCH + i % AUDIO_CH] = *v;
                }
                for (i, v) in vv.iter().enumerate() {
                    head[(na + i / VIDEO_PATCH) * FINAL_N + i % VIDEO_PATCH] = *v;
                }
                stream.upload(output.binding(), bytemuck::cast_slice(&head))?;
                sampler.step(
                    &mut stream,
                    output.binding(),
                    &sv.sigmas,
                    &sa.sigmas,
                    step,
                    scale,
                )?;
                crate::sampler::denoised_audio_into(
                    &y,
                    &audio,
                    &av,
                    sv.sigmas[step],
                    sa.sigmas[step],
                    scale,
                    &mut da,
                );
                crate::sampler::denoised_video_into(&video, &vv, sv.sigmas[step], &mut dv);
                crate::sampler::advance(
                    &mut y,
                    &da,
                    (step > 0).then_some(old_a.as_slice()),
                    &sv.sigmas,
                    step,
                );
                crate::sampler::advance(
                    &mut video,
                    &dv,
                    (step > 0).then_some(old_v.as_slice()),
                    &sv.sigmas,
                    step,
                );
                std::mem::swap(&mut old_a, &mut da);
                std::mem::swap(&mut old_v, &mut dv);
                let mut got_a = vec![0.0; y.len()];
                let mut got_v = vec![0.0; video.len()];
                sampler.download(&mut stream, &mut got_a, &mut got_v)?;
                assert_bits(&got_a, &y, "audio", step);
                assert_bits(&got_v, &video, "video", step);
                if finish == Some(step + 1) {
                    sv.finish_after(step + 1);
                    sa.finish_after(step + 1);
                }
                step += 1;
            }
        }
        assert!(sampler
            .step(
                &mut stream,
                output.slice(0, 4),
                &[1.0, 0.0],
                &[1.0, 0.0],
                0,
                4.0
            )
            .is_err());
        Ok(())
    }

    fn assert_bits(actual: &[f32], expected: &[f32], name: &str, step: usize) {
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(
                a.to_bits(),
                e.to_bits(),
                "{name} step {step} element {i}: GPU {a:e}, CPU {e:e}"
            );
        }
    }
}
