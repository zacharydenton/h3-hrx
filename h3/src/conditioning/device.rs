//! Prepared AdaLN projections. Only the eight timestep coefficients cross the
//! host boundary per launch; weights and the resulting tables stay on device.
use std::sync::Arc;

use crate::compile::Compiler;
use crate::error::{invalid, Result};
use crate::model::*;

pub(crate) struct Projection {
    kernel: Arc<hrx::Kernel>,
    weights: hrx::Buffer,
    bias: hrx::Buffer,
    maps: Vec<hrx::Buffer>,
    count: usize,
    output_bytes: usize,
}

impl Projection {
    pub fn blocks(
        stream: &mut hrx::Stream,
        c: &Compiler,
        w: &[Vec<f32>],
        b: &[Vec<f32>],
    ) -> Result<Self> {
        if w.len() != BLOCKS || b.len() != BLOCKS {
            return invalid("AdaLN layer count mismatch");
        }
        Self::new(stream, c, w, b, MODALITIES, 6, 4)
    }

    pub fn final_layer(
        stream: &mut hrx::Stream,
        c: &Compiler,
        w: Vec<f32>,
        b: Vec<f32>,
    ) -> Result<Self> {
        Self::new(stream, c, &[w], &[b], 1, 2, 2)
    }

    #[allow(clippy::too_many_arguments)]
    fn new(
        stream: &mut hrx::Stream,
        c: &Compiler,
        w: &[Vec<f32>],
        b: &[Vec<f32>],
        modalities: usize,
        chunks: usize,
        timesteps: usize,
    ) -> Result<Self> {
        let layer_count = modalities * chunks * HID;
        let count = w.len() * layer_count;
        if w.iter().any(|v| v.len() != layer_count * 8) || b.iter().any(|v| v.len() != layer_count)
        {
            return invalid("AdaLN projection shape mismatch");
        }
        let weights = stream.allocate(count * 8 * 4)?;
        let bias = stream.allocate(count * 4)?;
        for (layer, (w, b)) in w.iter().zip(b).enumerate() {
            crate::dispatch::upload_at(
                stream,
                &weights,
                layer * layer_count * 8 * 4,
                crate::vvae::as_bytes(w),
            )?;
            crate::dispatch::upload_at(
                stream,
                &bias,
                layer * layer_count * 4,
                crate::vvae::as_bytes(b),
            )?;
        }
        let classes = timesteps * modalities;
        let rows = count / HID;
        let mut maps = Vec::with_capacity(timesteps);
        for t in 0..timesteps {
            let mut map = Vec::<i32>::with_capacity(rows);
            for layer in 0..w.len() {
                for modality in 0..modalities {
                    let cls = t * modalities + modality;
                    let destinations = [
                        2 * cls + 1,
                        2 * cls,
                        2 * classes + cls,
                        3 * classes + 2 * cls + 1,
                        3 * classes + 2 * cls,
                        5 * classes + cls,
                    ];
                    for &row in &destinations[..chunks] {
                        map.push((layer * classes * chunks + row) as i32);
                    }
                }
            }
            let buffer = stream.allocate(map.len() * 4)?;
            let bytes: Vec<u8> = map.iter().flat_map(|row| row.to_ne_bytes()).collect();
            stream.upload(buffer.binding(), &bytes)?;
            maps.push(buffer);
        }
        let cfg = [
            ("count", count),
            ("width", HID),
            ("rows", rows),
            ("out_count", count * timesteps),
        ]
        .map(|(k, v)| (format!("h3.modulation_f32.{k}"), v.to_string()))
        .to_vec();
        Ok(Self {
            kernel: c.get(stream, "modulation_f32", "h3_modulation_f32", &cfg)?,
            weights,
            bias,
            maps,
            count,
            output_bytes: count * timesteps * 4,
        })
    }

    pub fn run(
        &self,
        stream: &mut hrx::Stream,
        timesteps: &[[f32; 8]],
        output: &hrx::Buffer,
    ) -> Result<()> {
        if timesteps.len() != self.maps.len() || output.bytes() < self.output_bytes {
            return invalid("AdaLN output or timestep shape mismatch");
        }
        for (te, map) in timesteps.iter().zip(&self.maps) {
            let mut constants = hrx::Constants::new();
            for &value in te {
                constants.push(value)?;
            }
            // Safety: eight f32 constants match the declaration; owned inputs
            // match specialization extents. Construction maps every lane to a
            // unique output within the checked allocation, across all launches.
            unsafe {
                stream.dispatch(
                    &self.kernel,
                    [self.count.div_ceil(256) as u32, 1, 1],
                    [256, 1, 1],
                    &constants,
                    &[
                        self.weights.binding(),
                        self.bias.binding(),
                        map.binding(),
                        output.binding(),
                    ],
                )?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires gfx1151 and the packaged Loom compiler"]
    fn projections_match_cpu_tables_bit_for_bit() -> Result<()> {
        let mut stream = hrx::Stream::open()?;
        let cache = tempfile::tempdir().unwrap();
        let compiler = Compiler::new(None, "", cache.path());
        let ramp = |n: usize, layer: usize| {
            (0..n)
                .map(|i| ((i + layer * 7) % 97) as f32 * 0.01 - 0.5)
                .collect::<Vec<_>>()
        };
        let w: Vec<_> = (0..BLOCKS)
            .map(|i| ramp(MODALITIES * 6 * HID * 8, i))
            .collect();
        let b: Vec<_> = (0..BLOCKS).map(|i| ramp(MODALITIES * 6 * HID, i)).collect();
        let te = [
            [1e8, 1.0, -1e8, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8],
            [0.0; 8],
            [1.0; 8],
        ];
        let blocks = Projection::blocks(&mut stream, &compiler, &w, &b)?;
        let expected = crate::conditioning::mods_table(&w, &b, &te[0], &te[1], &te[2], &te[3]);
        let output = stream.allocate(expected.len() * 4)?;
        blocks.run(&mut stream, &te, &output)?;
        let mut got = vec![0.0f32; expected.len()];
        stream.read(output.binding(), crate::vvae::as_bytes_mut(&mut got))?;
        for (i, (a, b)) in got.iter().zip(&expected).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "block element {i}: GPU {a}, CPU {b}"
            );
        }
        let (w, b) = (ramp(2 * HID * 8, 3), ramp(2 * HID, 3));
        let expected = crate::conditioning::final_table(&w, &b, &te[0], &te[1]);
        let final_layer = Projection::final_layer(&mut stream, &compiler, w, b)?;
        let output = stream.allocate(expected.len() * 4)?;
        final_layer.run(&mut stream, &te[..2], &output)?;
        let mut got = vec![0.0f32; expected.len()];
        stream.read(output.binding(), crate::vvae::as_bytes_mut(&mut got))?;
        for (i, (a, b)) in got.iter().zip(&expected).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "final element {i}: GPU {a}, CPU {b}"
            );
        }
        assert!(final_layer.run(&mut stream, &te, &output).is_err());
        // allocate before the call: the run borrows the stream mutably for its duration
        let short = stream.allocate(4)?;
        assert!(final_layer.run(&mut stream, &te[..2], &short).is_err());
        Ok(())
    }
}
