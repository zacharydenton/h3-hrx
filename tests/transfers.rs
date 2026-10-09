//! Regression coverage for large activation transfers through HRX.
const PROBE_STRIDE: usize = 64 << 20;

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
            stream.fill(src.binding(), 0x35).unwrap();
            stream.fill(dst.binding(), 0x79).unwrap();
            let probes = [
                0,
                PROBE_STRIDE - 4,
                PROBE_STRIDE,
                2 * PROBE_STRIDE,
                bytes - 4,
            ];
            for (i, offset) in probes.iter().enumerate() {
                stream
                    .upload_blocking_at(&src, 1 + offset, &[i as u8; 4])
                    .unwrap();
            }
            stream
                .copy(dst.slice(3, bytes), src.slice(1, bytes))
                .unwrap();
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
                (3 + PROBE_STRIDE + 4, 0x35),
            ] {
                let mut actual = [0];
                stream.read_blocking_at(&dst, offset, &mut actual).unwrap();
                assert_eq!(actual, [expected], "at {offset}");
            }
            let data: Vec<u8> = (0..bytes)
                .map(|i| (i.wrapping_mul(37) ^ (i >> 20)) as u8)
                .collect();
            stream.upload_at(&dst, 3, &data).unwrap();
            let mut actual = vec![0; bytes];
            stream.read_blocking_at(&dst, 3, &mut actual).unwrap();
            assert_eq!(actual, data);
            assert!(stream.upload_at(&dst, 17, &data).is_err());
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
    stream.fill(src.binding(), 0x35).unwrap();
    stream.fill(dst.binding(), 0x79).unwrap();
    let probes = [
        0,
        PROBE_STRIDE - 16,
        PROBE_STRIDE,
        (1usize << 32) + 8,
        bytes - 16,
    ];
    for (i, offset) in probes.iter().enumerate() {
        stream
            .upload_at(&src, 8 + offset, &[i as u8 + 1; 8])
            .unwrap();
    }
    stream
        .copy(dst.slice(8, bytes), src.slice(8, bytes))
        .unwrap();
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
    for offset in [24, PROBE_STRIDE - 8, PROBE_STRIDE + 8, (1usize << 32) + 16] {
        let mut got = [0; 8];
        stream.read_blocking_at(&dst, 8 + offset, &mut got).unwrap();
        assert_eq!(got, [0x35; 8]);
    }
    let mut graph = stream.owned_graph().unwrap();
    let fill = graph.fill(&[], dst.binding(), 0x79).unwrap();
    graph
        .copy(&[fill], dst.slice(8, bytes), src.slice(8, bytes))
        .unwrap();
    let mut graph = graph.finish().unwrap();
    for _ in 0..2 {
        stream.launch(&mut graph).unwrap();
        for (i, offset) in probes.iter().enumerate() {
            let mut got = [0; 8];
            stream.read_blocking_at(&dst, 8 + offset, &mut got).unwrap();
            assert_eq!(got, [i as u8 + 1; 8], "graph replay at {offset}");
        }
        for offset in [0, bytes + 8] {
            let mut got = [0; 8];
            stream.read_blocking_at(&dst, offset, &mut got).unwrap();
            assert_eq!(got, [0x79; 8]);
        }
    }
    stream.fill(dst.slice(1, 7), 0x12).unwrap();
    stream.copy(dst.slice(9, 7), dst.slice(1, 7)).unwrap();
    let mut got = [0; 7];
    stream.read_blocking_at(&dst, 9, &mut got).unwrap();
    assert_eq!(got, [0x12; 7]);
    assert!(stream.fill(dst.slice(0, 0), 0).is_err());
    assert!(stream.copy(dst.slice(0, 0), src.slice(0, 0)).is_err());
    assert!(stream.copy(dst.slice(0, 1), src.slice(0, 2)).is_err());
    assert!(stream
        .copy(dst.slice(8, bytes), dst.slice(0, bytes))
        .is_err());
}
