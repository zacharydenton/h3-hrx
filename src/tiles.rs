//! Tiling, and the three blends that look alike and are not.
//!
//! The video VAE runs in 256-pixel tiles both ways. The encoder blends in *latent* space and returns a
//! new tile; the decoder blends in pixel space in place; the decoder's temporal pass cross-fades whole
//! frames between chunks. Each clamps its overlap differently, and one of them does not clamp at all.
//! They are kept apart on purpose.
use crate::model::*;

/// ComfyUI's `split_tiles`: 256-pixel tiles with overlaps of at least 64, grown in 16-pixel units.
///
/// The loop widens the tile count until the tiles plus their minimum overlaps actually cover the
/// length, then spreads the slack round-robin across the overlaps — so with three tiles and 48 pixels
/// of slack the first overlap takes 32 and the second 16, not 24 each.
pub fn split_tiles(len: usize) -> (Vec<usize>, Vec<usize>) {
    let (tile, omin, ratio) = (256usize, 64usize, 16usize);
    if tile >= len {
        return (vec![0], Vec::new());
    }
    let mut n = len.div_ceil(tile);
    let overlaps = loop {
        let mut overlaps = vec![omin; n - 1];
        let span = tile * n;
        let need = omin * (n - 1) + len;
        if span < need {
            n += 1;
            continue;
        }
        let remaining = span - need;
        for i in 0..remaining / ratio {
            overlaps[i % (n - 1)] += ratio;
        }
        break overlaps;
    };
    let mut starts = vec![0usize];
    for o in &overlaps {
        starts.push(starts[starts.len() - 1] + tile - o);
    }
    (starts, overlaps)
}

/// The encoder's latent blend: a fresh tile whose leading `ext` rows or columns fade in from `a`.
///
/// The extent is clamped to both tiles' size along the blended axis, which is what keeps a short edge
/// tile from reading outside itself. Latents are `[24][T][h][w]`.
#[allow(clippy::too_many_arguments)]
pub fn blend_latent(
    a: &[f32],
    ah: usize,
    aw: usize,
    b: &[f32],
    bh: usize,
    bw: usize,
    ext: usize,
    ydim: bool,
    t_len: usize,
) -> Vec<f32> {
    // The two tiles always share the axis that is not being blended — a vertical blend joins tiles in
    // the same column, a horizontal one tiles in the same row. The C relies on that without saying so,
    // and reads out of bounds if it is ever false.
    debug_assert_eq!(if ydim { aw } else { ah }, if ydim { bw } else { bh });
    let mut r = b.to_vec();
    let e = ext.min(if ydim { ah.min(bh) } else { aw.min(bw) });
    if e == 0 {
        return r;
    }
    for c in 0..LATENT_CH {
        for t in 0..t_len {
            for y in 0..bh {
                for x in 0..bw {
                    let k = if ydim { y } else { x };
                    if k >= e {
                        continue;
                    }
                    let wb = k as f32 / e as f32;
                    let wa = 1.0 - wb;
                    let (ay, ax) = if ydim {
                        (ah - e + y, x)
                    } else {
                        (y, aw - e + x)
                    };
                    r[((c * t_len + t) * bh + y) * bw + x] = wa
                        * a[((c * t_len + t) * ah + ay) * aw + ax]
                        + wb * b[((c * t_len + t) * bh + y) * bw + x];
                }
            }
        }
    }
    r
}

/// The decoder's spatial blend, in place over `[3][F][th][tw]`.
///
/// Unlike the encoder's, the extent is used as given — both tiles are a full 256 pixels here, so there
/// is nothing to clamp against, and adding a clamp would only hide a caller's mistake.
pub fn blend_pixels(
    tile: &mut [f32],
    a: &[f32],
    extent: usize,
    vertical: bool,
    frames: usize,
    th: usize,
    tw: usize,
) {
    for c in 0..3 {
        for t in 0..frames {
            for y in 0..if vertical { extent } else { th } {
                for x in 0..if vertical { tw } else { extent } {
                    let wb = if vertical { y } else { x } as f32 / extent as f32;
                    let wa = 1.0 - wb;
                    let dst = ((c * frames + t) * th + y) * tw + x;
                    let src = ((c * frames + t) * th + if vertical { th - extent + y } else { y })
                        * tw
                        + if vertical { x } else { tw - extent + x };
                    tile[dst] = wa * a[src] + wb * tile[dst];
                }
            }
        }
    }
}

/// The decoder's temporal cross-fade: the previous chunk's tail faded into this chunk's head.
///
/// The fade length is the smallest of what the overlap holds, what the chunk holds, and what was asked
/// for — the last chunk of a clip is often shorter than the overlap it has to accept.
pub fn crossfade(
    chunk: &mut [f32],
    nf: usize,
    overlap: &[f32],
    plane: usize,
    overlap_frames: usize,
) {
    let ov = overlap.len() / (3 * plane);
    let be = ov.min(nf).min(overlap_frames);
    if be == 0 {
        return;
    }
    for c in 0..3 {
        for k in 0..be {
            let wb = k as f32 / be as f32;
            let wa = 1.0 - wb;
            let d0 = (c * nf + k) * plane;
            let s0 = (c * ov + ov - be + k) * plane;
            for q in 0..plane {
                chunk[d0 + q] = wa * overlap[s0 + q] + wb * chunk[d0 + q];
            }
        }
    }
}

/// How the decoder cuts a clip up in time.
///
/// diffusers' `_decode`: five-token chunks with a two-token overlap, seventeen frames kept per chunk
/// with three dropped off the front, and five-frame cross-fades between them. The token count is
/// padded up so the chunks divide evenly, and the frames the padding produced are dropped at the end —
/// which is not simply `pad_tokens * 4`, because a chunk boundary lands mid-clip and contributes only
/// the clip's intra-chunk tail.
#[derive(Debug, PartialEq, Eq)]
pub struct ChunkPlan {
    /// tokens appended by repeating the last latent frame
    pub pad_tokens: usize,
    /// the token count after padding
    pub padded_tokens: usize,
    pub chunks: usize,
    /// frames one chunk contributes
    pub chunk_frames: usize,
    /// frames dropped off the front of every clip
    pub pre: usize,
    /// how many frames the cross-fade spans
    pub overlap_frames: usize,
    /// frames to drop at the end, from the padding
    pub pad_frames: usize,
}

/// diffusers' chunk count: at least one, and one fewer than the tokens divide into because the last
/// chunk is carried as the overlap rather than emitted.
pub fn decoder_chunks(tokens: usize, padding: usize) -> usize {
    ((tokens + VAE_TOKEN_DROP + padding) / VAE_CHUNK)
        .saturating_sub(1)
        .max(1)
}

/// The plan for `t` latent frames.
pub fn chunk_plan(t: usize) -> ChunkPlan {
    let num_tokens = t + VAE_TOKEN_DROP;
    let pad_tokens = (VAE_CHUNK - num_tokens % VAE_CHUNK) % VAE_CHUNK;
    let chunks = decoder_chunks(t, pad_tokens);
    let pre = (VAE_TRATIO - VAE_CLIP % VAE_TRATIO) % VAE_TRATIO;
    let intra_tail = VAE_CLIP % VAE_TRATIO;
    // each padded token is a whole temporal step, except one that lands on a chunk boundary, which
    // contributes only the clip's tail
    let pad_frames = (0..pad_tokens)
        .map(|k| {
            if intra_tail != 0 && (t + k).is_multiple_of(VAE_CHUNK) {
                intra_tail
            } else {
                VAE_TRATIO
            }
        })
        .sum();
    ChunkPlan {
        pad_tokens,
        padded_tokens: t + pad_tokens,
        chunks,
        chunk_frames: VAE_CHUNK * VAE_TRATIO,
        pre,
        overlap_frames: (VAE_OVERLAP * VAE_TRATIO).saturating_sub(pre),
        pad_frames,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_length_inside_one_tile_needs_no_overlaps() {
        for len in [16, 128, 255, 256] {
            let (starts, overlaps) = split_tiles(len);
            assert_eq!(starts, vec![0], "len {len}");
            assert!(overlaps.is_empty());
        }
    }

    #[test]
    fn the_tiles_cover_the_length_and_stay_inside_it() {
        for len in (272..=2048).step_by(16) {
            let (starts, overlaps) = split_tiles(len);
            let n = starts.len();
            assert_eq!(overlaps.len(), n - 1, "len {len}");
            assert_eq!(starts[0], 0);
            assert_eq!(
                starts[n - 1] + 256,
                len,
                "the last tile must end exactly at the length, len {len}"
            );
            for (i, o) in overlaps.iter().enumerate() {
                assert!(*o >= 64, "overlap {i} of len {len} is {o}");
                assert!(o % 16 == 0, "overlaps move in 16-pixel units");
                assert!(*o <= 256, "an overlap cannot exceed the tile");
            }
        }
    }

    #[test]
    fn the_slack_is_spread_round_robin_not_evenly() {
        // 720 needs four tiles, not three: three would leave the overlaps below the minimum. The
        // 112 pixels of slack then go round-robin in sixteens, so the first overlap takes one more
        // than the other two rather than all three sharing evenly.
        let (starts, overlaps) = split_tiles(720);
        assert_eq!(overlaps, vec![112, 96, 96]);
        assert_eq!(starts, vec![0, 144, 304, 464]);
        assert_eq!(overlaps.iter().sum::<usize>(), 4 * 256 - 720);
    }

    #[test]
    fn a_real_frame_splits_the_way_the_decoder_expects() {
        assert_eq!(
            split_tiles(864),
            (vec![0, 144, 288, 448, 608], vec![112, 112, 96, 96])
        );
        assert_eq!(split_tiles(480), (vec![0, 112, 224], vec![144, 144]));
        assert_eq!(
            split_tiles(1344),
            (
                vec![0, 176, 352, 528, 704, 896, 1088],
                vec![80, 80, 80, 80, 64, 64]
            )
        );
        assert_eq!(split_tiles(768), (vec![0, 160, 336, 512], vec![96, 80, 80]));
        // a length one pixel over a tile still needs two, overlapping almost completely
        assert_eq!(split_tiles(257), (vec![0, 16], vec![240]));
    }

    #[test]
    fn the_chunk_plan_decodes_exactly_the_frames_the_shape_promises() {
        // The decoder emits chunk_frames per chunk, plus whatever the trailing overlap carries, and
        // then drops pad_frames. For every shape the pipeline can be asked for, that has to come to
        // the frame count exactly — the C throws if it does not.
        for frames in [5, 22, 39, 56, 73, 90, 175, 209, 481] {
            let sh = crate::layout::shape_for(480, 864, frames).expect("a valid shape");
            let t = sh.latent_t as usize;
            let p = chunk_plan(t);
            // walk the loop the decoder runs: each chunk covers VAE_CHUNK tokens with VAE_OVERLAP more
            // for context, emits the first chunk_frames past `pre`, and carries the rest forward
            let mut emitted = 0usize;
            let mut carried = 0usize;
            for i in 0..p.chunks {
                let start = i * VAE_CHUNK;
                let ft = (VAE_CHUNK + VAE_OVERLAP).min(p.padded_tokens - start);
                let clip_frames = ft * VAE_TRATIO;
                carried = 0;
                for j in 0..2 {
                    let f0 = j * p.chunk_frames + p.pre;
                    let f1 = ((j + 1) * p.chunk_frames).min(clip_frames);
                    if f0 >= f1 {
                        continue;
                    }
                    if j == 0 {
                        emitted += f1 - f0;
                    } else {
                        carried = f1 - f0;
                    }
                }
            }
            emitted += carried;
            assert_eq!(
                emitted - p.pad_frames,
                sh.frames as usize,
                "{frames} frames, latent_t {t}"
            );
        }
    }

    #[test]
    fn a_short_clip_still_decodes_in_one_chunk() {
        let p = chunk_plan(2);
        assert_eq!(p.chunks, 1);
        assert_eq!(p.pre, 3, "three frames come off the front of every clip");
        assert_eq!(p.overlap_frames, 5, "and the cross-fade spans five");
    }

    #[test]
    fn the_padding_frames_are_not_simply_four_per_token() {
        // a token landing on a chunk boundary contributes only the clip's intra-chunk tail
        let mismatched: Vec<usize> = (1..200)
            .filter(|t| {
                let p = chunk_plan(*t);
                p.pad_tokens > 0 && p.pad_frames != p.pad_tokens * VAE_TRATIO
            })
            .collect();
        assert!(
            !mismatched.is_empty(),
            "the boundary case has to actually occur, or the loop is dead code"
        );
    }

    #[test]
    fn the_latent_blend_is_a_ramp_and_leaves_the_rest_of_the_tile_alone() {
        let (h, w, t) = (4usize, 4usize, 1usize);
        let a = vec![0.0f32; LATENT_CH * t * h * w];
        let b = vec![1.0f32; LATENT_CH * t * h * w];
        let r = blend_latent(&a, h, w, &b, h, w, 2, true, t);
        assert_eq!(r[0], 0.0);
        assert_eq!(r[w], 0.5);
        assert_eq!(r[2 * w], 1.0);
        assert_eq!(r[3 * w], 1.0);
    }

    #[test]
    fn the_latent_blend_clamps_its_extent_to_the_shorter_tile() {
        let (t, w) = (1usize, 2usize);
        let a = vec![0.0f32; LATENT_CH * t * w];
        let b = vec![1.0f32; LATENT_CH * t * 4 * w];
        // asking for four rows of blend against a one-row neighbour must not read past it
        let r = blend_latent(&a, 1, w, &b, 4, w, 4, true, t);
        assert_eq!(r[0], 0.0, "the single overlapping row is all a");
        assert_eq!(r[w], 1.0, "and nothing below it is touched");
    }

    #[test]
    fn the_pixel_blend_reads_the_neighbour_from_its_far_edge() {
        let (f, th, tw) = (1usize, 4usize, 4usize);
        let mut a = vec![0.0f32; 3 * f * th * tw];
        for x in 0..tw {
            a[2 * tw + x] = 10.0;
            a[3 * tw + x] = 20.0;
        }
        let mut tile = vec![100.0f32; 3 * f * th * tw];
        blend_pixels(&mut tile, &a, 2, true, f, th, tw);
        assert_eq!(tile[0], 10.0, "row 0 takes the neighbour's row th - extent");
        assert_eq!(tile[tw], 60.0, "row 1 is half of 20 and half of 100");
        assert_eq!(tile[2 * tw], 100.0, "past the extent the tile is unchanged");
    }

    #[test]
    fn the_crossfade_takes_the_tail_of_the_overlap() {
        let plane = 2usize;
        let (ov, nf) = (4usize, 4usize);
        let overlap: Vec<f32> = (0..3 * ov * plane)
            .map(|i| (i % (ov * plane)) as f32)
            .collect();
        let mut chunk = vec![100.0f32; 3 * nf * plane];
        crossfade(&mut chunk, nf, &overlap, plane, 2);
        // be = 2: frame 0 is all of the overlap's frame 2, frame 1 is halfway to the chunk
        assert_eq!(chunk[0], 4.0);
        assert_eq!(chunk[plane], (6.0 + 100.0) / 2.0);
        assert_eq!(chunk[2 * plane], 100.0);
    }

    #[test]
    fn a_short_chunk_fades_over_what_it_has() {
        let plane = 1usize;
        let overlap = vec![0.0f32; 3 * 8 * plane];
        let mut chunk = vec![100.0f32; 3 * plane];
        crossfade(&mut chunk, 1, &overlap, plane, 8);
        assert_eq!(
            chunk[0], 0.0,
            "one frame means the whole fade is the overlap"
        );
    }
}
