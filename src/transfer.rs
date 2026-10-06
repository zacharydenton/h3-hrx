//! HRX's native transfer kernels accept at most `u32::MAX` bytes per launch.
//! Long clips can exceed that in a single attention or residual buffer.
use hrx::{Result, Stream, View};

// Preserve eight-byte alignment so aligned transfers use HRX's wide kernels.
const CHUNK: usize = (u32::MAX as usize) & !7;

fn ranges(bytes: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..bytes)
        .step_by(CHUNK)
        .map(move |offset| (offset, (bytes - offset).min(CHUNK)))
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
