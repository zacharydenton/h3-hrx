//! Bounded file reads and byte-preserving device row layout.
use super::{layout, Checkpoint, Recipe, Result};
use hrx::storage::{
    StorageConfig, StorageMode, StorageProgress, StorageSession, StorageStatistics,
};
use std::{
    collections::VecDeque,
    num::NonZeroUsize,
    time::{Duration, Instant},
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WeightIoMode {
    #[default]
    Mapped,
    NativeBuffered,
    NativeDirect,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct WeightIo {
    pub mode: WeightIoMode,
    pub progress: StorageProgress,
    /// Retained native I/O slots (1..=64). None uses HRX's default.
    pub slots: Option<NonZeroUsize>,
    /// Page-aligned bytes per slot, at most 64 MiB. None uses HRX's default.
    /// The shared memory budget covers slots * slot_bytes plus native commands.
    pub slot_bytes: Option<NonZeroUsize>,
    /// Collect host-observed loading phase durations.
    pub statistics: bool,
}
impl WeightIo {
    pub fn native_lifetime(self) -> hrx::fabric::NativeLifetime {
        match self.mode {
            WeightIoMode::Mapped => hrx::fabric::NativeLifetime::Instance,
            _ => hrx::fabric::NativeLifetime::Process,
        }
    }
}

/// Host-observed native loading phases; durations are collected only when requested.
#[derive(Clone, Debug, Default)]
pub struct WeightStatistics {
    pub tensors: u64,
    pub logical_bytes: u64,
    pub device_bytes: u64,
    /// Logical copy/gather commands submitted to consume storage reads.
    pub consumer_commands: u64,
    pub allocation: Option<Duration>,
    pub read_submit: Option<Duration>,
    pub read_wait: Option<Duration>,
    pub consumer_submit: Option<Duration>,
    pub consumer_wait: Option<Duration>,
    pub consumer_retire: Option<Duration>,
    /// Tensor loading time, excluding initial storage-session construction.
    pub total: Option<Duration>,
    pub storage: StorageStatistics,
}

pub(super) struct Loader {
    session: StorageSession,
    gather: std::collections::BTreeMap<(usize, usize), crate::compile::Kernel>,
    gather_batches: std::collections::BTreeMap<(usize, usize), crate::compile::Kernel>,
    compiler: crate::compile::Compiler,
    statistics: WeightStatistics,
    timed: bool,
}
struct Chunk {
    file: usize,
    offset: u64,
    bytes: usize,
    destination: usize,
    first: usize,
    span: usize,
}
struct Batch {
    file: usize,
    offset: u64,
    bytes: usize,
    parts: Vec<Chunk>,
}
struct Gather {
    destination: usize,
    span: usize,
    tiles: u32,
    descriptors: Vec<[i32; 4]>,
}
impl Batch {
    fn gather(&self, row: usize, pitch: usize) -> Option<Gather> {
        if !(2..=65535).contains(&self.parts.len())
            || row > i32::MAX as usize
            || pitch > i32::MAX as usize
        {
            return None;
        }
        let destination = self.parts.iter().map(|p| p.destination).min()?;
        let end = self.parts.iter().try_fold(destination, |end, p| {
            p.destination.checked_add(p.span).map(|n| end.max(n))
        })?;
        let span = end.checked_sub(destination)?;
        i32::try_from(span).ok()?;
        i32::try_from(self.bytes).ok()?;
        let mut descriptors = Vec::with_capacity(self.parts.len());
        let mut tiles = 0;
        for part in &self.parts {
            let source = part.offset.checked_sub(self.offset)?;
            let first = if row == pitch { 0 } else { part.first };
            i32::try_from(first.checked_add(part.bytes)?).ok()?;
            descriptors.push([
                i32::try_from(source).ok()?,
                i32::try_from(part.destination - destination).ok()?,
                i32::try_from(part.bytes).ok()?,
                i32::try_from(first).ok()?,
            ]);
            tiles = tiles.max(part.bytes.div_ceil(256) as u32);
        }
        Some(Gather {
            destination,
            span,
            tiles,
            descriptors,
        })
    }
}
fn coalesce(mut chunks: Vec<Chunk>, capacity: usize) -> Vec<Batch> {
    chunks.sort_by_key(|chunk| (chunk.file, chunk.offset));
    let mut batches: Vec<Batch> = Vec::new();
    for chunk in chunks {
        if let Some(batch) = batches.last_mut() {
            if batch.file == chunk.file
                && batch.offset + batch.bytes as u64 == chunk.offset
                && batch.bytes + chunk.bytes <= capacity
            {
                batch.bytes += chunk.bytes;
                batch.parts.push(chunk);
                continue;
            }
        }
        batches.push(Batch {
            file: chunk.file,
            offset: chunk.offset,
            bytes: chunk.bytes,
            parts: vec![chunk],
        });
    }
    batches
}
impl Loader {
    pub fn new(
        stream: &hrx::Stream,
        checkpoint: &Checkpoint,
        io: WeightIo,
        compiler: crate::compile::Compiler,
    ) -> Result<Self> {
        let mode = match io.mode {
            WeightIoMode::NativeBuffered => StorageMode::Buffered,
            WeightIoMode::NativeDirect => StorageMode::Direct,
            WeightIoMode::Mapped => return layout("mapped loading has no native storage session"),
        };
        let session = StorageSession::new(
            stream,
            &checkpoint.storage_files()?,
            StorageConfig {
                mode,
                progress: io.progress,
                statistics: io.statistics,
                slots: io
                    .slots
                    .map_or_else(|| StorageConfig::default().slots, NonZeroUsize::get),
                slot_bytes: io
                    .slot_bytes
                    .map_or_else(|| StorageConfig::default().slot_bytes, NonZeroUsize::get),
            },
        )?;
        Ok(Self {
            session,
            compiler,
            gather: Default::default(),
            gather_batches: Default::default(),
            statistics: WeightStatistics::default(),
            timed: io.statistics,
        })
    }
    pub fn statistics(&self) -> WeightStatistics {
        let mut stats = self.statistics.clone();
        stats.storage = self.session.statistics();
        stats
    }
    pub fn load(
        &mut self,
        stream: &mut hrx::Stream,
        checkpoint: &Checkpoint,
        recipe: &Recipe,
    ) -> Result<hrx::Buffer> {
        let start = Instant::now();
        let Recipe::Rows {
            rows,
            row_bytes,
            pitch_bytes,
            segments,
        } = recipe
        else {
            unreachable!()
        };
        if *row_bytes == 0
            || pitch_bytes < row_bytes
            || segments
                .iter()
                .try_fold(0usize, |n, s| n.checked_add(s.rows))
                != Some(*rows)
        {
            return layout("invalid native row recipe");
        }
        let bytes = rows
            .checked_mul(*pitch_bytes)
            .ok_or_else(|| super::Error::Layout("weight size overflow".into()))?;
        let mut chunks = Vec::new();
        let mut destination_row = 0usize;
        for segment in segments {
            let entry = checkpoint.at(&segment.tensor)?;
            let begin = segment
                .row0
                .checked_mul(*row_bytes)
                .ok_or_else(|| super::Error::Layout("row offset overflow".into()))?;
            let length = segment
                .rows
                .checked_mul(*row_bytes)
                .ok_or_else(|| super::Error::Layout("row length overflow".into()))?;
            // Resolve before allocating or submitting; a synthetic shard offset is never a disk offset.
            let (file, offset) = checkpoint.file_range(entry, begin, length)?;
            let mut consumed = 0;
            while consumed < length {
                let take = (length - consumed).min(self.session.read_chunk_bytes());
                let first = consumed % row_bytes;
                let last = first + take - 1;
                let span = (last / row_bytes) * pitch_bytes + last % row_bytes + 1;
                if pitch_bytes != row_bytes
                    && [first + take, *row_bytes, *pitch_bytes, span]
                        .iter()
                        .any(|&n| n > i32::MAX as usize)
                {
                    return layout("native pitched row chunks require positive 32-bit extents");
                }
                chunks.push(Chunk {
                    file,
                    offset: offset + consumed as u64,
                    bytes: take,
                    destination: (destination_row + consumed / row_bytes) * pitch_bytes
                        + if pitch_bytes == row_bytes { first } else { 0 },
                    first,
                    span: if pitch_bytes == row_bytes { take } else { span },
                });
                consumed += take;
            }
            destination_row += segment.rows;
        }
        if pitch_bytes != row_bytes && !self.gather.contains_key(&(*row_bytes, *pitch_bytes)) {
            let cfg = vec![
                ("h3.weight_rows.row_bytes".into(), row_bytes.to_string()),
                ("h3.weight_rows.pitch_bytes".into(), pitch_bytes.to_string()),
            ];
            let kernel = self
                .compiler
                .get(stream, "weight_rows", "h3_weight_rows", &cfg)?;
            kernel.resolve(stream)?;
            self.gather.insert((*row_bytes, *pitch_bytes), kernel);
        }
        let batches: Vec<_> = coalesce(chunks, self.session.read_chunk_bytes())
            .into_iter()
            .map(|batch| {
                let gather = batch.gather(*row_bytes, *pitch_bytes);
                (batch, gather)
            })
            .collect();
        let batch_kernel = if batches.iter().any(|(_, gather)| gather.is_some()) {
            let key = (*row_bytes, *pitch_bytes);
            if !self.gather_batches.contains_key(&key) {
                let cfg = vec![
                    (
                        "h3.weight_rows_batch.row_bytes".into(),
                        row_bytes.to_string(),
                    ),
                    (
                        "h3.weight_rows_batch.pitch_bytes".into(),
                        pitch_bytes.to_string(),
                    ),
                ];
                let kernel =
                    self.compiler
                        .get(stream, "weight_rows_batch", "h3_weight_rows_batch", &cfg)?;
                kernel.resolve(stream)?;
                self.gather_batches.insert(key, kernel);
            }
            Some(self.gather_batches[&key].clone())
        } else {
            None
        };
        let mut output = None;
        let mut pending = VecDeque::new();
        let mut copies = VecDeque::new();
        let mut next = batches.into_iter().peekable();
        loop {
            while let Some((chunk, _)) = next.peek() {
                let submitted = self.timed.then(Instant::now);
                let result = self.session.read(chunk.file, chunk.offset, chunk.bytes);
                if let Some(start) = submitted {
                    add(&mut self.statistics.read_submit, start.elapsed());
                }
                match result {
                    Ok(ticket) => pending.push_back((next.next().unwrap(), ticket)),
                    Err(hrx::Error::Busy(_)) => break,
                    Err(error) => return Err(error.into()),
                }
            }
            // Service the first bounded set of reads while allocating the destination.
            if output.is_none() {
                let allocation = self.timed.then(Instant::now);
                let buffer = stream.allocate(bytes.max(1))?;
                if pitch_bytes != row_bytes {
                    crate::transfer::fill(stream, buffer.binding(), 0)?;
                }
                output = Some(buffer);
                if let Some(start) = allocation {
                    add(&mut self.statistics.allocation, start.elapsed());
                }
            }
            let output = output.as_ref().unwrap();
            if let Some(((chunk, gather), ticket)) = pending.pop_front() {
                let wait = Instant::now();
                let lease = ticket.wait()?;
                if self.timed {
                    add(&mut self.statistics.read_wait, wait.elapsed());
                }
                drop(ticket);
                let descriptors = gather
                    .as_ref()
                    .map(|plan| stream.allocate_from(bytemuck::cast_slice(&plan.descriptors)))
                    .transpose()?;
                // SAFETY: each part reads a checked subrange of this lease and writes
                // its disjoint destination rows. All consumers are enqueued immediately
                // on this stream; the lease remains held through their common fence.
                let consume =
                    |stream: &mut hrx::Stream, source: hrx::View<'_>| -> hrx::Result<()> {
                        if let Some(plan) = &gather {
                            let kernel = batch_kernel
                                .as_ref()
                                .unwrap()
                                .resolve(stream)
                                .map_err(|error| hrx::Error::Message(error.to_string()))?;
                            let mut constants = hrx::Constants::new();
                            for value in [chunk.bytes, plan.span, plan.descriptors.len()] {
                                constants.push(value as u32)?;
                            }
                            // The plan bounds every descriptor within these views. HRX
                            // retains descriptor backing and the lease through completion.
                            return unsafe {
                                stream.dispatch(
                                    kernel,
                                    [plan.tiles, plan.descriptors.len() as u32, 1],
                                    [256, 1, 1],
                                    &constants,
                                    &[
                                        source,
                                        output.try_slice(plan.destination, plan.span)?,
                                        descriptors.as_ref().unwrap().binding(),
                                    ],
                                )
                            };
                        }
                        for part in &chunk.parts {
                            let source =
                                source.slice((part.offset - chunk.offset) as usize, part.bytes)?;
                            let destination = output.try_slice(part.destination, part.span)?;
                            if pitch_bytes == row_bytes {
                                stream.copy(destination, source)?;
                            } else {
                                let kernel = self
                                    .gather
                                    .get(&(*row_bytes, *pitch_bytes))
                                    .unwrap()
                                    .resolve(stream)
                                    .map_err(|error| hrx::Error::Message(error.to_string()))?;
                                let mut constants = hrx::Constants::new();
                                for value in [part.bytes, part.first, part.span] {
                                    constants.push(value as u32)?;
                                }
                                unsafe {
                                    stream.dispatch(
                                        kernel,
                                        [part.bytes.div_ceil(256) as u32, 1, 1],
                                        [256, 1, 1],
                                        &constants,
                                        &[source, destination],
                                    )
                                }?;
                            }
                        }
                        Ok(())
                    };
                // SAFETY: only the immediate, bounded consumers above use this source.
                let submitted = self.timed.then(Instant::now);
                let copy = unsafe { lease.enqueue(stream, consume) }?;
                self.statistics.consumer_commands += if gather.is_some() {
                    1
                } else {
                    chunk.parts.len() as u64
                };
                if let Some(start) = submitted {
                    add(&mut self.statistics.consumer_submit, start.elapsed());
                }
                copies.push_back(copy);
            } else if let Some(mut copy) = copies.pop_front() {
                let wait = Instant::now();
                copy.wait()?;
                if self.timed {
                    add(&mut self.statistics.consumer_wait, wait.elapsed());
                }
            } else if next.peek().is_none() {
                break;
            }
            // Reclaim completed copies without waiting, allowing the next read to overlap.
            let retired = self.timed.then(Instant::now);
            while copies
                .front_mut()
                .map(|copy| copy.is_complete())
                .transpose()?
                .unwrap_or(false)
            {
                copies.pop_front();
            }
            if let Some(start) = retired {
                add(&mut self.statistics.consumer_retire, start.elapsed());
            }
        }
        self.compiler.check_sanitizers(stream)?;
        self.statistics.tensors += 1;
        self.statistics.logical_bytes += (rows * row_bytes) as u64;
        self.statistics.device_bytes += bytes as u64;
        if self.timed {
            add(&mut self.statistics.total, start.elapsed());
        }
        Ok(output.unwrap())
    }
}
fn add(total: &mut Option<Duration>, elapsed: Duration) {
    *total = Some(total.unwrap_or_default() + elapsed);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::{Recipe, Weights};
    use hrx::execution::{ComputeEngine, CopyEngine, RuntimeOptions};
    use std::{collections::BTreeMap, io::Write};

    #[test]
    fn coalescing_preserves_destination_order_and_file_boundaries() {
        let chunk = |file, offset, destination| Chunk {
            file,
            offset,
            destination,
            bytes: 4,
            first: 0,
            span: 4,
        };
        let batches = coalesce(
            vec![
                chunk(0, 4, 0),
                chunk(1, 0, 12),
                chunk(0, 0, 4),
                chunk(0, 12, 8),
            ],
            8,
        );
        assert_eq!(batches.len(), 3);
        assert_eq!(
            (batches[0].file, batches[0].offset, batches[0].bytes),
            (0, 0, 8)
        );
        assert_eq!(
            batches[0]
                .parts
                .iter()
                .map(|p| p.destination)
                .collect::<Vec<_>>(),
            [4, 0]
        );
        assert_eq!(
            (batches[1].file, batches[1].offset, batches[1].bytes),
            (0, 12, 4)
        );
        assert_eq!(
            (batches[2].file, batches[2].offset, batches[2].bytes),
            (1, 0, 4)
        );
    }

    #[test]
    fn gather_rebases_large_destinations_and_falls_back_for_wide_spans() {
        let base = 1usize << 32;
        let mut batch = Batch {
            file: 0,
            offset: 100,
            bytes: 8,
            parts: vec![
                Chunk {
                    file: 0,
                    offset: 100,
                    bytes: 4,
                    destination: base + 16,
                    first: 3,
                    span: 4,
                },
                Chunk {
                    file: 0,
                    offset: 104,
                    bytes: 4,
                    destination: base,
                    first: 0,
                    span: 4,
                },
            ],
        };
        let plan = batch.gather(7, 7).unwrap();
        assert_eq!((plan.destination, plan.span), (base, 20));
        // Unpitched destinations already include the partial-row offset.
        assert_eq!(plan.descriptors, [[0, 16, 4, 0], [4, 0, 4, 0]]);
        batch.parts[0].span = 7;
        let pitched = batch.gather(7, 12).unwrap();
        assert_eq!(pitched.descriptors[0][3], 3);
        assert_eq!(pitched.span, 23);
        batch.parts[0].destination = base + i32::MAX as usize;
        assert!(batch.gather(7, 7).is_none());
    }

    fn shard(path: &std::path::Path, name: &str, rows: usize, width: usize, seed: usize) {
        let count = rows * width;
        let header=format!("{{\"{name}\":{{\"dtype\":\"U8\",\"shape\":[{rows},{width}],\"data_offsets\":[0,{count}]}}}}");
        let mut file = std::fs::File::create(path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(header.as_bytes()).unwrap();
        let data: Vec<_> = (0..count)
            .map(|i| ((i * 73) ^ (i >> 8) ^ seed) as u8)
            .collect();
        file.write_all(&data).unwrap();
    }
    fn plan(ck: &Checkpoint, out: &mut BTreeMap<String, Recipe>) -> Result<()> {
        out.insert("straight".into(), super::super::rows_of(ck, &["a"], 0)?);
        out.insert("sharded".into(), super::super::rows_of(ck, &["b", "a"], 0)?);
        out.insert(
            "padded".into(),
            super::super::rows_of(ck, &["b", "a"], ck.at("a")?.row_bytes()? + 5)?,
        );
        out.insert(
            "permuted".into(),
            super::super::rows_permuted(ck, "a", &[(8, 9), (0, 8)], 0)?,
        );
        let order: Vec<_> = (0..17)
            .step_by(2)
            .chain((1..17).step_by(2))
            .map(|row| (row, 1))
            .collect();
        for (name, pitch) in [
            ("fragmented", 0),
            ("fragmented_padded", ck.at("a")?.row_bytes()? + 5),
        ] {
            out.insert(
                name.into(),
                super::super::rows_permuted(ck, "a", &order, pitch)?,
            );
        }
        out.insert(
            "built".into(),
            Recipe::Built {
                bytes: 257,
                build: Box::new(|_| Ok((0..257).map(|i| i as u8).collect())),
            },
        );
        Ok(())
    }
    #[test]
    #[cfg_attr(
        not(feature = "gpu-tests"),
        ignore = "requires native GPU and io_uring"
    )]
    fn native_rows_enforce_selected_compiler_sanitizer_bounds() {
        let dir = tempfile::tempdir_in(env!("CARGO_MANIFEST_DIR")).unwrap();
        let path = dir.path().join("a.safetensors");
        shard(&path, "a", 3, 7, 19);
        let options = crate::compile::Options {
            sanitizer: hrx::loom::SanitizerChecks {
                access: true,
                ..Default::default()
            },
            sanitizer_runtime: hrx::fabric::SanitizerRuntimeOptions {
                maximum_shadow_bytes: 1,
                ..Default::default()
            },
            ..Default::default()
        };
        let compiler = crate::compile::Compiler::with_options(
            None,
            std::path::PathBuf::new(),
            options.clone(),
        );
        let runtime = hrx::execution::Runtime::with_options(RuntimeOptions {
            native_lifetime: hrx::fabric::NativeLifetime::Process,
            compute_engine: ComputeEngine::Aql {
                maximum_private_bytes: 512,
            },
            ..Default::default()
        })
        .unwrap();
        let mut stream = runtime.stream(Some(options.sanitizer_runtime)).unwrap();
        // SAFETY: this private fixture remains immutable until the mapping drops.
        let mut weights = unsafe {
            Weights::open(path, |ck, out| {
                out.insert("padded".into(), super::super::rows_of(ck, &["a"], 11)?);
                Ok(())
            })
        }
        .unwrap();
        weights
            .set_io(
                WeightIo {
                    mode: WeightIoMode::NativeBuffered,
                    ..Default::default()
                },
                &compiler,
            )
            .unwrap();
        let expected = weights.assemble("padded").unwrap();
        let error = match weights.at(&mut stream, "padded", expected.len()) {
            Ok(_) => panic!("native rows bypassed the configured sanitizer bound"),
            Err(error) => error,
        };
        assert!(error
            .to_string()
            .contains("address shadow exceeds configured byte limit"));
    }

    #[test]
    #[cfg_attr(
        not(feature = "gpu-tests"),
        ignore = "requires native AQL and io_uring"
    )]
    fn fragmented_rows_pass_native_address_and_race_checks() {
        let dir = tempfile::tempdir_in(env!("CARGO_MANIFEST_DIR")).unwrap();
        let path = dir.path().join("rows.safetensors");
        shard(&path, "a", 17, 259, 43);
        let options = crate::compile::Options {
            sanitizer: hrx::loom::SanitizerChecks {
                access: true,
                race: true,
                ..Default::default()
            },
            ..Default::default()
        };
        let manager = hrx::residency::ResidencyManager::new(256 << 20).unwrap();
        {
            let runtime = hrx::execution::Runtime::with_options(RuntimeOptions {
                native_lifetime: hrx::fabric::NativeLifetime::Process,
                compute_engine: ComputeEngine::Aql {
                    maximum_private_bytes: 512,
                },
                memory_budget: Some(manager.budget()),
                ..Default::default()
            })
            .unwrap();
            let mut stream = runtime
                .stream(Some(options.sanitizer_runtime.clone()))
                .unwrap();
            let compiler =
                crate::compile::Compiler::with_options(None, std::path::PathBuf::new(), options);
            // SAFETY: the private checkpoint remains immutable throughout the test.
            let mut weights = unsafe {
                Weights::open(path, |ck, out| {
                    let order: Vec<_> = (0..17).rev().map(|r| (r, 1)).collect();
                    out.insert(
                        "rows".into(),
                        super::super::rows_permuted(ck, "a", &order, 267)?,
                    );
                    Ok(())
                })
            }
            .unwrap();
            weights
                .set_io(
                    WeightIo {
                        mode: WeightIoMode::NativeBuffered,
                        slots: NonZeroUsize::new(1),
                        slot_bytes: NonZeroUsize::new(4096),
                        ..Default::default()
                    },
                    &compiler,
                )
                .unwrap();
            let expected = weights.assemble("rows").unwrap();
            let output = weights.at(&mut stream, "rows", expected.len()).unwrap();
            let mut actual = vec![0; expected.len()];
            stream.read_blocking(output.binding(), &mut actual).unwrap();
            assert_eq!(actual, expected);
            // Fifteen fragments in the first read and two in the second.
            assert_eq!(weights.io_statistics().unwrap().consumer_commands, 2);
        }
        assert_eq!(manager.budget().reserved_bytes(), 0);
    }

    #[test]
    #[cfg_attr(
        not(feature = "gpu-tests"),
        ignore = "requires native GPU and io_uring"
    )]
    fn prefetched_reads_survive_destination_budget_failure() {
        let dir = tempfile::tempdir_in(env!("CARGO_MANIFEST_DIR")).unwrap();
        let path = dir.path().join("rows.safetensors");
        let width = 1 << 20;
        shard(&path, "a", 17, width, 43);
        let manager = hrx::residency::ResidencyManager::new(16 << 20).unwrap();
        {
            let runtime = hrx::execution::Runtime::with_options(RuntimeOptions {
                native_lifetime: hrx::fabric::NativeLifetime::Process,
                memory_budget: Some(manager.budget()),
                ..Default::default()
            })
            .unwrap();
            let mut stream = runtime.stream(None).unwrap();
            // SAFETY: this private checkpoint remains immutable while mapped.
            let mut weights = unsafe {
                Weights::open(path, |ck, out| {
                    out.insert("large".into(), super::super::rows_of(ck, &["a"], 0)?);
                    out.insert(
                        "small".into(),
                        super::super::rows_permuted(ck, "a", &[(16, 1)], 0)?,
                    );
                    Ok(())
                })
            }
            .unwrap();
            weights
                .set_io(
                    WeightIo {
                        mode: WeightIoMode::NativeBuffered,
                        progress: StorageProgress::Wait,
                        slots: NonZeroUsize::new(1),
                        slot_bytes: NonZeroUsize::new(width),
                        ..Default::default()
                    },
                    &crate::compile::Compiler::new(None, std::path::PathBuf::new()),
                )
                .unwrap();
            assert!(weights.at(&mut stream, "large", 17 * width).is_err());
            let stats = weights.io_statistics().unwrap();
            assert_eq!(stats.storage.logical_requests, 1);
            assert_eq!(stats.consumer_commands, 0);
            // An accepted read is not cancelled by the failed destination allocation.
            // Its slot must retire so the same loader can load a different extent.
            let output = weights.at(&mut stream, "small", width).unwrap();
            let mut actual = vec![0; width];
            stream.read_blocking(output.binding(), &mut actual).unwrap();
            assert_eq!(actual, weights.assemble("small").unwrap());
            assert_eq!(weights.io_statistics().unwrap().storage.peak_slots, 1);
        }
        assert_eq!(manager.budget().reserved_bytes(), 0);
    }

    #[test]
    #[cfg_attr(
        not(feature = "gpu-tests"),
        ignore = "requires native GPU and NVMe direct I/O"
    )]
    fn native_rows_preserve_shards_padding_and_engine_ordering() {
        let dir = tempfile::tempdir_in(env!("CARGO_MANIFEST_DIR")).unwrap();
        let paths = [
            dir.path().join("a.safetensors"),
            dir.path().join("b.safetensors"),
        ];
        let width = (1 << 20) + 7;
        shard(&paths[0], "a", 17, width, 0);
        shard(&paths[1], "b", 2, width, 197);
        for compute in [
            ComputeEngine::Pm4,
            ComputeEngine::Aql {
                maximum_private_bytes: 512,
            },
        ] {
            for copy in [CopyEngine::Compute, CopyEngine::Sdma] {
                for mode in [WeightIoMode::NativeBuffered, WeightIoMode::NativeDirect] {
                    for progress in [StorageProgress::Sqpoll, StorageProgress::Wait] {
                        let manager = hrx::residency::ResidencyManager::new(256 << 20).unwrap();
                        {
                            let runtime = hrx::execution::Runtime::with_options(RuntimeOptions {
                                compute_engine: compute,
                                copy_engine: copy,
                                native_lifetime: hrx::fabric::NativeLifetime::Process,
                                memory_budget: Some(manager.budget()),
                                ..Default::default()
                            })
                            .unwrap();
                            let mut stream = runtime.stream(None).unwrap();
                            // SAFETY: these private fixtures remain immutable while mapped.
                            let mut weights =
                                unsafe { Weights::open_shards(&paths, plan) }.unwrap();
                            weights
                                .set_io(
                                    WeightIo {
                                        mode,
                                        progress,
                                        statistics: true,
                                        slots: NonZeroUsize::new(2),
                                        slot_bytes: NonZeroUsize::new(8 << 20),
                                    },
                                    &crate::compile::Compiler::new(None, std::path::PathBuf::new()),
                                )
                                .unwrap();
                            for name in [
                                "straight",
                                "sharded",
                                "padded",
                                "permuted",
                                "fragmented",
                                "fragmented_padded",
                                "built",
                            ] {
                                let expected = weights.assemble(name).unwrap();
                                let buffer = weights.at(&mut stream, name, expected.len()).unwrap();
                                let mut actual = vec![0; expected.len()];
                                stream.read_blocking(buffer.binding(), &mut actual).unwrap();
                                assert_eq!(
                                    actual, expected,
                                    "{compute:?}/{copy:?}/{mode:?}/{progress:?}/{name}"
                                );
                            }
                            assert!(weights
                                .set_io(
                                    WeightIo::default(),
                                    &crate::compile::Compiler::new(None, std::path::PathBuf::new())
                                )
                                .is_err());
                            let stats = weights.io_statistics().unwrap();
                            assert_eq!(stats.tensors, 6);
                            assert!(stats.storage.peak_slots <= 2);
                            assert!(stats.total.is_some());
                            assert!(stats.allocation.is_some());
                            assert!(stats.consumer_submit.is_some());
                            assert!(stats.consumer_retire.is_some());
                        }
                        assert_eq!(manager.budget().reserved_bytes(), 0);
                    }
                }
            }
        }
    }
}
