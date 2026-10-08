//! Native engine and compiler controls shared by every inference command.
use clap::ValueEnum;
use std::{num::NonZeroUsize, path::PathBuf};

#[derive(Clone, Copy, Debug, Default, ValueEnum)]
enum Compute {
    #[default]
    Pm4,
    Aql,
}
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
enum Copy {
    #[default]
    Compute,
    Sdma,
}
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
enum Processor {
    #[default]
    Default,
    Cu,
    Wgp,
}
#[derive(Clone, Copy, Debug, ValueEnum)]
enum Check {
    Access,
    Value,
    Operation,
    Race,
}

#[derive(Clone, Copy, Debug, Default, ValueEnum)]
enum WeightIo {
    #[default]
    Mapped,
    NativeBuffered,
    NativeDirect,
}
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
enum StorageProgress {
    #[default]
    Sqpoll,
    Wait,
}

#[derive(clap::Args)]
pub(super) struct RuntimeArgs {
    /// Checkpoint loading route; native-direct requires filesystem direct-I/O support
    #[arg(long, value_enum, default_value = "mapped")]
    weight_io: WeightIo,
    /// Kernel progress policy for native weight I/O
    #[arg(long, value_enum, default_value = "sqpoll")]
    storage_progress: StorageProgress,
    /// Retained native I/O slots, charged to the memory budget (HRX default: 4)
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..=64))]
    storage_slots: Option<u32>,
    /// MiB per native I/O slot (HRX default: 16)
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..=64))]
    storage_slot_mib: Option<u32>,
    /// Collect host-observed native loading phase timings
    #[arg(long)]
    weight_io_statistics: bool,
    /// GPU ordinal in HRX's native device enumeration
    #[arg(long, default_value_t = 0)]
    gpu: i32,
    /// Native compute queue protocol
    #[arg(long, value_enum, default_value = "pm4")]
    compute_engine: Compute,
    /// Transfer engine; SDMA uses coherent backing and fences engine transitions
    #[arg(long, value_enum, default_value = "compute")]
    copy_engine: Copy,
    /// AQL scratch ceiling per workitem, fixed before loading models
    #[arg(long, default_value_t = 4096)]
    aql_private_bytes: u32,
    /// Compiler workgroup scheduling policy
    #[arg(long, value_enum, default_value = "default")]
    processor_mode: Processor,
    #[arg(long)]
    compile_workers: Option<NonZeroUsize>,
    /// Report-only kernel checks (comma-separated); requires --compute-engine aql
    #[arg(long, value_enum, value_delimiter = ',')]
    sanitize: Vec<Check>,
    /// Feedback bytes per instrumented kernel (power of two, 128 through 16 MiB)
    #[arg(long, default_value_t = 65536)]
    sanitizer_report_bytes: usize,
    /// Maximum combined address/race shadow bytes per prepared dispatch
    #[arg(long, default_value_t = 67108864)]
    sanitizer_shadow_bytes: usize,
    /// Directory for detailed compiler reports
    #[arg(long)]
    compile_reports: Option<PathBuf>,
    /// Directory for bounded compiler pass traces in text format
    #[arg(long)]
    compile_traces: Option<PathBuf>,
}
impl RuntimeArgs {
    pub(super) fn weight_io(&self) -> h3_hrx::weights::WeightIo {
        use h3_hrx::weights::WeightIoMode;
        h3_hrx::weights::WeightIo {
            mode: match self.weight_io {
                WeightIo::Mapped => WeightIoMode::Mapped,
                WeightIo::NativeBuffered => WeightIoMode::NativeBuffered,
                WeightIo::NativeDirect => WeightIoMode::NativeDirect,
            },
            progress: match self.storage_progress {
                StorageProgress::Sqpoll => hrx::storage::StorageProgress::Sqpoll,
                StorageProgress::Wait => hrx::storage::StorageProgress::Wait,
            },
            statistics: self.weight_io_statistics,
            slots: self
                .storage_slots
                .and_then(|n| NonZeroUsize::new(n as usize)),
            slot_bytes: self
                .storage_slot_mib
                .and_then(|n| NonZeroUsize::new((n as usize) << 20)),
        }
    }

    pub(super) fn report_storage(&self, session: &h3_hrx::Session) -> anyhow::Result<()> {
        if self.weight_io_statistics {
            for (name, stats) in session.weight_io_statistics()? {
                eprintln!("weight I/O {name}: {stats:?}");
            }
        }
        Ok(())
    }
    pub(super) fn compiler(&self) -> h3_hrx::compile::Options {
        let mut sanitizer = hrx::loom::SanitizerChecks::default();
        for check in &self.sanitize {
            match check {
                Check::Access => sanitizer.access = true,
                Check::Value => sanitizer.value = true,
                Check::Operation => sanitizer.operation = true,
                Check::Race => sanitizer.race = true,
            }
        }
        h3_hrx::compile::Options {
            workers: self.compile_workers,
            processor_mode: match self.processor_mode {
                Processor::Default => hrx::loom::ProcessorMode::Default,
                Processor::Cu => hrx::loom::ProcessorMode::ComputeUnit,
                Processor::Wgp => hrx::loom::ProcessorMode::WorkgroupProcessor,
            },
            sanitizer,
            sanitizer_runtime: hrx::fabric::SanitizerRuntimeOptions {
                capacity_bytes: self.sanitizer_report_bytes,
                maximum_shadow_bytes: self.sanitizer_shadow_bytes,
                ..Default::default()
            },
            reports: self.compile_reports.clone(),
            traces: self.compile_traces.clone(),
        }
    }
    pub(super) fn options(
        &self,
        memory_budget: Option<hrx::residency::MemoryBudget>,
    ) -> anyhow::Result<hrx::execution::RuntimeOptions> {
        anyhow::ensure!(self.gpu >= 0, "GPU ordinal must be nonnegative");
        anyhow::ensure!(
            self.sanitize.is_empty() || matches!(self.compute_engine, Compute::Aql),
            "--sanitize requires --compute-engine aql"
        );
        anyhow::ensure!(
            self.sanitizer_report_bytes.is_power_of_two()
                && (128..=16 * 1024 * 1024).contains(&self.sanitizer_report_bytes),
            "sanitizer report bytes must be a power of two from 128 through 16 MiB"
        );
        anyhow::ensure!(
            self.sanitizer_shadow_bytes > 0,
            "sanitizer shadow bytes must be positive"
        );
        Ok(hrx::execution::RuntimeOptions {
            gpu_index: self.gpu,
            native_lifetime: self.weight_io().native_lifetime(),
            compute_engine: match self.compute_engine {
                Compute::Pm4 => hrx::execution::ComputeEngine::Pm4,
                Compute::Aql => hrx::execution::ComputeEngine::Aql {
                    maximum_private_bytes: self.aql_private_bytes,
                },
            },
            copy_engine: match self.copy_engine {
                Copy::Compute => hrx::execution::CopyEngine::Compute,
                Copy::Sdma => hrx::execution::CopyEngine::Sdma,
            },
            memory_budget,
            ..Default::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    #[test]
    fn options_reach_both_runtime_and_compiler() {
        let cli = super::super::Cli::try_parse_from([
            "h3",
            "--compute-engine",
            "aql",
            "--copy-engine",
            "sdma",
            "--aql-private-bytes",
            "512",
            "--processor-mode",
            "cu",
            "--compile-workers",
            "2",
            "--sanitize",
            "access,value,operation,race",
        ])
        .unwrap();
        let runtime = cli.runtime.options(None).unwrap();
        assert_eq!(
            runtime.compute_engine,
            hrx::execution::ComputeEngine::Aql {
                maximum_private_bytes: 512
            }
        );
        assert_eq!(runtime.copy_engine, hrx::execution::CopyEngine::Sdma);
        let compiler = cli.runtime.compiler();
        assert_eq!(
            compiler.processor_mode,
            hrx::loom::ProcessorMode::ComputeUnit
        );
        assert_eq!(compiler.workers.unwrap().get(), 2);
        assert_eq!(
            compiler.sanitizer,
            hrx::loom::SanitizerChecks {
                access: true,
                value: true,
                operation: true,
                race: true
            }
        );
        assert!(compiler.reports.is_none() && compiler.traces.is_none());
    }

    #[test]
    fn storage_selection_reaches_runtime_and_loader() {
        use h3_hrx::weights::WeightIoMode;
        for (flag, mode, lifetime) in [
            (
                "mapped",
                WeightIoMode::Mapped,
                hrx::fabric::NativeLifetime::Instance,
            ),
            (
                "native-buffered",
                WeightIoMode::NativeBuffered,
                hrx::fabric::NativeLifetime::Process,
            ),
            (
                "native-direct",
                WeightIoMode::NativeDirect,
                hrx::fabric::NativeLifetime::Process,
            ),
        ] {
            let cli = super::super::Cli::try_parse_from([
                "h3",
                "--weight-io",
                flag,
                "--storage-progress",
                "wait",
                "--storage-slots",
                "2",
                "--storage-slot-mib",
                "64",
            ])
            .unwrap();
            assert_eq!(cli.runtime.weight_io().mode, mode);
            assert_eq!(
                cli.runtime.weight_io().progress,
                hrx::storage::StorageProgress::Wait
            );
            assert_eq!(cli.runtime.options(None).unwrap().native_lifetime, lifetime);
            assert_eq!(cli.runtime.weight_io().slots.unwrap().get(), 2);
            assert_eq!(cli.runtime.weight_io().slot_bytes.unwrap().get(), 64 << 20);
        }
    }

    #[test]
    fn invalid_options_fail_without_opening_a_gpu() {
        let cli = super::super::Cli::try_parse_from(["h3", "--sanitize", "access"]).unwrap();
        assert!(cli
            .runtime
            .options(None)
            .unwrap_err()
            .to_string()
            .contains("requires"));
        assert!(super::super::Cli::try_parse_from(["h3", "--compile-workers", "0"]).is_err());
        for flag in ["--storage-slots", "--storage-slot-mib"] {
            for value in ["0", "65"] {
                assert!(super::super::Cli::try_parse_from(["h3", flag, value]).is_err());
            }
        }
        let cli = super::super::Cli::try_parse_from(["h3", "--gpu=-1"]).unwrap();
        assert!(cli.runtime.options(None).is_err());
    }
}
