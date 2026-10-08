//! Bounded native transfers for large attention and residual buffers.
use hrx::{Buffer, Result, Stream, View};

// HRX 0.9 encodes one SDMA copy packet (7 dwords) per MiB. The minimum
// supported ring is 4 KiB. Stay below half of it so command plus wrap padding
// fits even at an unfavorable cursor; also allow cache packets and a fence.
// This also stays below the compute transfer's u32 byte-count limit and
// preserves eight-byte alignment for its wide kernels.
const CHUNK: usize = 64 << 20;

fn ranges(bytes: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..bytes)
        .step_by(CHUNK)
        .map(move |offset| (offset, (bytes - offset).min(CHUNK)))
}

pub(crate) fn upload(stream: &mut Stream, dst: View<'_>, data: &[u8]) -> Result<()> {
    // Validate the complete destination before submitting the first chunk.
    let dst = dst.slice(0, data.len())?;
    for (offset, bytes) in ranges(data.len()) {
        stream.upload(dst.slice(offset, bytes)?, &data[offset..offset + bytes])?;
    }
    Ok(())
}

pub(crate) fn upload_at(
    stream: &mut Stream,
    dst: &Buffer,
    offset: usize,
    data: &[u8],
) -> Result<()> {
    upload(stream, dst.try_slice(offset, data.len())?, data)
}

pub(crate) fn fill(stream: &Stream, dst: View<'_>, value: u8) -> Result<()> {
    for (offset, bytes) in ranges(dst.len()) {
        stream.fill(dst.slice(offset, bytes)?, value)?;
    }
    Ok(())
}

pub(crate) fn copy(stream: &Stream, dst: View<'_>, src: View<'_>) -> Result<()> {
    if dst.len() != src.len() {
        return Err(hrx::Error::Message("copy requires equal spans".into()));
    }
    // Reject overlap before enqueuing anything, including overlap across chunks.
    if std::ptr::eq(dst.owner(), src.owner())
        && dst.offset() < src.offset() + src.len()
        && src.offset() < dst.offset() + dst.len()
    {
        return Err(hrx::Error::Message("native copy ranges overlap".into()));
    }
    for (offset, bytes) in ranges(src.len()) {
        stream.copy(dst.slice(offset, bytes)?, src.slice(offset, bytes)?)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_cover_empty_exact_and_multiple_chunks() {
        for bytes in [
            0,
            1,
            CHUNK - 1,
            CHUNK,
            CHUNK + 1,
            1usize << 32,
            3 * CHUNK + 19,
        ] {
            let mut end = 0;
            for (offset, length) in ranges(bytes) {
                assert_eq!(offset, end);
                assert_eq!(offset % 8, 0);
                assert!((1..=u32::MAX as usize).contains(&length));
                end += length;
            }
            assert_eq!(end, bytes);
        }
    }

    #[test]
    #[cfg_attr(not(feature = "gpu-tests"), ignore = "requires provisioned HRX")]
    fn sdma_transfers_cross_queue_capacity_without_touching_guards() {
        for compute_engine in [
            hrx::execution::ComputeEngine::Pm4,
            hrx::execution::ComputeEngine::Aql {
                maximum_private_bytes: 4096,
            },
        ] {
            let manager = hrx::residency::ResidencyManager::new(2 << 30).unwrap();
            {
                let mut stream = hrx::Device::open(0)
                    .unwrap()
                    .stream_with_options(hrx::StreamOptions {
                        compute_engine,
                        copy_engine: hrx::execution::CopyEngine::Sdma,
                        memory_budget: Some(manager.budget()),
                        ..Default::default()
                    })
                    .unwrap();
                // Both fill and copy packet streams exceed a 4 KiB ring if
                // submitted as one command. Unaligned ends exercise byte tails.
                let bytes = (512 << 20) + 19;
                let src = stream.allocate(bytes + 16).unwrap();
                let dst = stream.allocate(bytes + 16).unwrap();
                fill(&stream, src.binding(), 0x35).unwrap();
                fill(&stream, dst.binding(), 0x79).unwrap();
                let probes = [0, CHUNK - 4, CHUNK, 2 * CHUNK, bytes - 4];
                for (i, offset) in probes.iter().enumerate() {
                    stream
                        .upload_blocking_at(&src, 1 + offset, &[i as u8; 4])
                        .unwrap();
                }
                copy(&stream, dst.slice(3, bytes), src.slice(1, bytes)).unwrap();
                for (i, offset) in probes.iter().enumerate() {
                    let mut actual = [0; 4];
                    stream
                        .read_blocking_at(&dst, 3 + offset, &mut actual)
                        .unwrap();
                    assert_eq!(actual, [i as u8; 4], "at {offset}");
                }
                for (offset, expected) in [
                    (2, 0x79),
                    (3 + bytes, 0x79),
                    (11, 0x35),
                    (3 + CHUNK + 4, 0x35),
                ] {
                    let mut actual = [0];
                    stream.read_blocking_at(&dst, offset, &mut actual).unwrap();
                    assert_eq!(actual, [expected], "at {offset}");
                }
                let data: Vec<u8> = (0..bytes)
                    .map(|i| (i.wrapping_mul(37) ^ (i >> 20)) as u8)
                    .collect();
                upload_at(&mut stream, &dst, 3, &data).unwrap();
                let mut actual = vec![0; bytes];
                stream.read_blocking_at(&dst, 3, &mut actual).unwrap();
                assert_eq!(actual, data);
                assert!(upload_at(&mut stream, &dst, 17, &data).is_err());
                for offset in [2, 3 + bytes] {
                    let mut guard = [0];
                    stream.read_blocking_at(&dst, offset, &mut guard).unwrap();
                    assert_eq!(guard, [0x79]);
                }
            }
            assert_eq!(manager.statistics().reserved_bytes, 0);
        }
    }

    #[test]
    #[cfg_attr(not(feature = "gpu-tests"), ignore = "requires provisioned HRX")]
    fn fills_and_copies_across_four_gib_without_touching_guards() {
        let device = hrx::Device::open(0).unwrap();
        let mut stream = device.stream().unwrap();
        let bytes = (1usize << 32) + 257;
        let src = stream.allocate(bytes + 16).unwrap();
        let dst = stream.allocate(bytes + 16).unwrap();
        // This is the failure long-clip attention initialization used to hit.
        assert!(stream
            .fill(src.binding(), 0)
            .unwrap_err()
            .to_string()
            .contains("native transfer range"));
        fill(&stream, src.binding(), 0x35).unwrap();
        fill(&stream, dst.binding(), 0x79).unwrap();
        let probes = [0, CHUNK - 16, CHUNK, (1usize << 32) + 8, bytes - 16];
        for (i, offset) in probes.iter().enumerate() {
            stream
                .upload_at(&src, 8 + offset, &[i as u8 + 1; 8])
                .unwrap();
        }
        copy(&stream, dst.slice(8, bytes), src.slice(8, bytes)).unwrap();
        for (i, offset) in probes.iter().enumerate() {
            let mut got = [0; 8];
            stream.read_blocking_at(&dst, 8 + offset, &mut got).unwrap();
            assert_eq!(got, [i as u8 + 1; 8], "at {offset}");
        }
        for offset in [0, bytes + 8] {
            let mut got = [0; 8];
            stream.read_blocking_at(&dst, offset, &mut got).unwrap();
            assert_eq!(got, [0x79; 8]);
        }
        for offset in [24, CHUNK - 8, CHUNK + 8, (1usize << 32) + 16] {
            let mut got = [0; 8];
            stream.read_blocking_at(&dst, 8 + offset, &mut got).unwrap();
            assert_eq!(got, [0x35; 8]);
        }
        fill(&stream, dst.slice(1, 7), 0x12).unwrap();
        copy(&stream, dst.slice(9, 7), dst.slice(1, 7)).unwrap();
        let mut got = [0; 7];
        stream.read_blocking_at(&dst, 9, &mut got).unwrap();
        assert_eq!(got, [0x12; 7]);
        fill(&stream, dst.slice(0, 0), 0).unwrap();
        copy(&stream, dst.slice(0, 0), src.slice(0, 0)).unwrap();
        assert!(copy(&stream, dst.slice(0, 1), src.slice(0, 2)).is_err());
        assert!(copy(&stream, dst.slice(8, bytes), dst.slice(0, bytes)).is_err());
    }
}
