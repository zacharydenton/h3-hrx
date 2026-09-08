//! The packed sequence the blocks see, and the sigma schedule that walks it.
//!
//! Rows are laid out as `[text | keyframes | reference blocks | audio | video]`, each carrying a
//! position `(t, h, w)` for the rotary tables, an AdaLN row, and a timestep class. This is ComfyUI's
//! PackedLayout; the arithmetic is in `f64` throughout because the positions feed `cos`/`sin` tables
//! whose last bits show up in the attention.
use crate::model::*;

/// A reference block's rows, in the order they were packed.
#[derive(Clone, Debug, PartialEq)]
pub struct RefSeg {
    pub kind: i32,
    pub row0: usize,
    pub rows: usize,
    pub latent_t: i32,
    pub lat_h: i32,
    pub lat_w: i32,
    pub audio_t: i32,
    pub audio: bool,
    /// Index into the caller's reference or keyframe list.
    pub index: usize,
}

/// What a caller conditions on: a reference image, a reference sound, or a reference clip.
#[derive(Clone, Debug)]
pub struct Ref {
    /// 0 image, 1 audio, 2 video.
    pub kind: i32,
    pub latent_t: i32,
    pub lat_h: i32,
    pub lat_w: i32,
    pub audio_t: i32,
    pub has_audio: bool,
}

/// A first or last frame the clip must pass through.
#[derive(Clone, Debug)]
pub struct Keyframe {
    pub frame_index: i32,
    pub audio_t: i32,
    pub has_audio: bool,
}

#[derive(Debug)]
pub struct Layout {
    pub text_len: usize,
    pub latent_t: usize,
    pub lat_h: usize,
    pub lat_w: usize,
    pub audio_t: usize,
    pub ref_rows: usize,
    pub audio_rows: usize,
    pub video_rows: usize,
    pub seq_len: usize,
    /// `[seq][3]`: the (t, h, w) each row sits at.
    pub pos: Vec<f64>,
    /// Timestep class * 3 + modality tag. Tags: 0 video, 1 text, 2 audio.
    pub adaln_rows: Vec<i32>,
    /// 0 video, 1 audio, 2 conditioning video, 3 conditioning audio.
    pub tclass: Vec<i32>,
    pub ref_segs: Vec<RefSeg>,
}

impl Layout {
    /// The spatial coordinates of one axis, centred on the canvas.
    fn axis(dim: usize, sqrt_area: f64) -> Vec<f64> {
        let ratio = dim as f64 / sqrt_area;
        let n = dim / 2;
        (0..n)
            .map(|i| (i as f64 * (ratio / n as f64) + (1.0 - ratio) / 2.0) * SPATIAL_SCALE)
            .collect()
    }

    /// How far `n` latent frames advance the time axis. Accumulated term by term, because the running
    /// sum's rounding is part of the position.
    fn video_span(n: usize) -> f64 {
        let mut s = 0.0;
        for k in 0..n {
            s += FRAME_RESCALE * FRAME_PER_TOKEN[k % 5] as f64;
        }
        s
    }

    /// Zeroes the AdaLN row of an embedding span and the tokens flanking it, so a vision span is not
    /// modulated as text.
    pub fn mark_vision(&mut self, start: usize, count: usize) -> Result<(), String> {
        if start > self.text_len || count > self.text_len - start {
            return Err("vision span outside presentation".into());
        }
        let begin = start.saturating_sub(1);
        let end = self.text_len.min(start + count + 1);
        self.adaln_rows[begin..end].fill(0);
        Ok(())
    }

    pub fn new(
        text_len: usize,
        latent_t: usize,
        lat_h: usize,
        lat_w: usize,
        audio_t: usize,
        refs: &[Ref],
        kfs: &[Keyframe],
    ) -> Result<Self, String> {
        let area = ((lat_h * lat_w) as f64).sqrt();
        let ah = Self::axis(lat_h, area);
        let aw = Self::axis(lat_w, area);
        let audio_rows = audio_t * 2;
        let video_rows = latent_t * ah.len() * aw.len();

        let mut ref_rows = 0usize;
        for rf in refs {
            if rf.kind == 1 {
                ref_rows += rf.audio_t as usize * 2;
            } else {
                let (hh, ww) = ((rf.lat_h / 2) as usize, (rf.lat_w / 2) as usize);
                let vt = if rf.kind == 0 {
                    1
                } else {
                    rf.latent_t as usize
                };
                ref_rows += vt * hh * ww;
                if rf.kind == 2 && rf.has_audio && rf.audio_t > 0 {
                    ref_rows += rf.audio_t as usize * 2;
                }
            }
        }
        for kf in kfs {
            ref_rows += ah.len() * aw.len();
            if kf.has_audio && kf.audio_t > 0 {
                ref_rows += kf.audio_t as usize * 2;
            }
        }

        let seq_len = text_len + ref_rows + audio_rows + video_rows;
        let mut me = Self {
            text_len,
            latent_t,
            lat_h,
            lat_w,
            audio_t,
            ref_rows,
            audio_rows,
            video_rows,
            seq_len,
            pos: vec![0.0; seq_len * 3],
            adaln_rows: vec![0; seq_len],
            tclass: vec![0; seq_len],
            ref_segs: Vec::new(),
        };

        let mut r = 0usize;
        for i in 0..text_len {
            me.pos[3 * r] = i as f64;
            me.adaln_rows[r] = 1;
            me.tclass[r] = 0;
            r += 1;
        }
        let mut cursor = text_len as f64;

        // Keyframes sit on the target grid at cursor + FRAME_RESCALE * frame_index, where the cursor
        // has already skipped whatever the references occupy.
        let mut after_refs = cursor;
        for rf in refs {
            after_refs += match rf.kind {
                0 => 1.0,
                1 => rf.audio_t as f64,
                _ => {
                    let sound = if rf.has_audio && rf.audio_t > 0 {
                        rf.audio_t as f64
                    } else {
                        0.0
                    };
                    sound.max(Self::video_span(rf.latent_t as usize))
                }
            };
        }
        for (ki, kf) in kfs.iter().enumerate() {
            let cond_t = after_refs + FRAME_RESCALE * f64::from(kf.frame_index);
            me.ref_segs.push(RefSeg {
                kind: 3,
                row0: r,
                rows: ah.len() * aw.len(),
                latent_t: 1,
                lat_h: lat_h as i32,
                lat_w: lat_w as i32,
                audio_t: 0,
                audio: false,
                index: ki,
            });
            for &hv in &ah {
                for &wv in &aw {
                    me.pos[3 * r] = cond_t;
                    me.pos[3 * r + 1] = hv;
                    me.pos[3 * r + 2] = wv;
                    me.adaln_rows[r] = 2 * MODALITIES as i32;
                    me.tclass[r] = 2;
                    r += 1;
                }
            }
            if kf.has_audio && kf.audio_t > 0 {
                me.ref_segs.push(RefSeg {
                    kind: 3,
                    row0: r,
                    rows: kf.audio_t as usize * 2,
                    latent_t: 0,
                    lat_h: 0,
                    lat_w: 0,
                    audio_t: kf.audio_t,
                    audio: true,
                    index: ki,
                });
                r = audio_grid(
                    &mut me,
                    r,
                    cond_t,
                    kf.audio_t as usize,
                    aw[0],
                    aw[aw.len() - 1],
                    3,
                );
            }
        }

        for (ri, rf) in refs.iter().enumerate() {
            if rf.kind == 0 || rf.kind == 2 {
                let rarea = ((rf.lat_h as f64) * (rf.lat_w as f64)).sqrt();
                let rh = Self::axis(rf.lat_h as usize, rarea);
                let rw = Self::axis(rf.lat_w as usize, rarea);
                let vt = if rf.kind == 0 {
                    1
                } else {
                    rf.latent_t as usize
                };
                let sound = rf.kind == 2 && rf.has_audio && rf.audio_t > 0;
                if sound {
                    me.ref_segs.push(RefSeg {
                        kind: rf.kind,
                        row0: r,
                        rows: rf.audio_t as usize * 2,
                        latent_t: 0,
                        lat_h: 0,
                        lat_w: 0,
                        audio_t: rf.audio_t,
                        audio: true,
                        index: ri,
                    });
                    r = audio_grid(
                        &mut me,
                        r,
                        cursor,
                        rf.audio_t as usize,
                        rw[0],
                        rw[rw.len() - 1],
                        3,
                    );
                }
                me.ref_segs.push(RefSeg {
                    kind: rf.kind,
                    row0: r,
                    rows: vt * rh.len() * rw.len(),
                    latent_t: vt as i32,
                    lat_h: rf.lat_h,
                    lat_w: rf.lat_w,
                    audio_t: 0,
                    audio: false,
                    index: ri,
                });
                r = video_grid(&mut me, r, cursor, vt, &rh, &rw, 2);
                cursor += if rf.kind == 0 {
                    1.0
                } else {
                    let s = if sound { rf.audio_t as f64 } else { 0.0 };
                    s.max(Self::video_span(vt))
                };
            } else {
                if rf.audio_t > 0 {
                    me.ref_segs.push(RefSeg {
                        kind: 1,
                        row0: r,
                        rows: rf.audio_t as usize * 2,
                        latent_t: 0,
                        lat_h: 0,
                        lat_w: 0,
                        audio_t: rf.audio_t,
                        audio: true,
                        index: ri,
                    });
                    r = audio_grid(
                        &mut me,
                        r,
                        cursor,
                        rf.audio_t as usize,
                        aw[0],
                        aw[aw.len() - 1],
                        3,
                    );
                }
                cursor += rf.audio_t as f64;
            }
        }

        r = audio_grid(&mut me, r, cursor, audio_t, aw[0], aw[aw.len() - 1], 1);
        r = video_grid(&mut me, r, cursor, latent_t, &ah, &aw, 0);
        if r != seq_len {
            return Err(format!(
                "layout row count mismatch: {r} rows for a sequence of {seq_len}"
            ));
        }
        Ok(me)
    }

    /// Where the target video's rows begin.
    pub fn video_span_start(&self) -> usize {
        self.seq_len - self.video_rows
    }
}

fn audio_grid(
    l: &mut Layout,
    mut r: usize,
    origin: f64,
    t: usize,
    w_low: f64,
    w_high: f64,
    cls: i32,
) -> usize {
    for c in 0..2 {
        for k in 0..t {
            l.pos[3 * r] = origin + k as f64;
            l.pos[3 * r + 2] = if c == 0 { w_low } else { w_high };
            l.adaln_rows[r] = cls * MODALITIES as i32 + 2;
            l.tclass[r] = cls;
            r += 1;
        }
    }
    r
}

fn video_grid(
    l: &mut Layout,
    mut r: usize,
    origin: f64,
    vt: usize,
    gh: &[f64],
    gw: &[f64],
    cls: i32,
) -> usize {
    let mut acc = origin;
    for k in 0..vt {
        for &hv in gh {
            for &wv in gw {
                l.pos[3 * r] = acc;
                l.pos[3 * r + 1] = hv;
                l.pos[3 * r + 2] = wv;
                l.adaln_rows[r] = cls * MODALITIES as i32;
                l.tclass[r] = cls;
                r += 1;
            }
        }
        acc += FRAME_RESCALE * FRAME_PER_TOKEN[k % 5] as f64;
    }
    r
}

/// diffusers' MiniMaxH3Scheduler.
///
/// The grid is computed in `f64` and narrowed to `f32`, then deduplicated on exact `f32` equality — so
/// the number of evaluations can depend on that rounding, which is why the narrowing happens here and
/// not at the end.
pub struct Schedule {
    pub sigmas: Vec<f32>,
    pub timesteps: Vec<f32>,
}

impl Schedule {
    pub fn new(steps: usize, shift: f64) -> Self {
        let mut sigmas: Vec<f32> = Vec::new();
        for i in 0..steps {
            let base = if steps == 1 {
                1.0
            } else {
                1.0 - i as f64 / (steps - 1) as f64
            };
            let v = (shift * base / (1.0 + (shift - 1.0) * base)) as f32;
            if sigmas.last() != Some(&v) {
                sigmas.push(v);
            }
        }
        let timesteps = sigmas
            .iter()
            .take(sigmas.len().saturating_sub(1))
            .map(|s| 1.0 - s)
            .collect();
        Self { sigmas, timesteps }
    }
}

/// The frame count the model actually produces: the next `17n + 5`.
pub fn align_frames(mut n: i32) -> i32 {
    while n % 17 != 5 {
        n += 1;
    }
    n
}

/// Canvas sides and frame counts the pipeline accepts; every derived count fits an `i32`.
pub const MAX_SIDE: i32 = 8192;
pub const MAX_FRAMES: i32 = 1 << 20;

pub fn valid_canvas(height: i32, width: i32) -> bool {
    height >= 32
        && width >= 32
        && height <= MAX_SIDE
        && width <= MAX_SIDE
        && height % 32 == 0
        && width % 32 == 0
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Shape {
    pub frames: i32,
    pub latent_t: i32,
    pub lat_h: i32,
    pub lat_w: i32,
    pub audio_t: i32,
    pub text_rows_max: i32,
}

/// The shapes a request produces, or `None` when the request is not one this model can serve.
pub fn shape_for(height: i32, width: i32, frames: i32) -> Option<Shape> {
    if !valid_canvas(height, width) || !(1..=MAX_FRAMES).contains(&frames) {
        return None;
    }
    let frames = align_frames(frames.max(5));
    Some(Shape {
        frames,
        latent_t: (frames - 5) / 17 * 5 + 2,
        lat_h: height / 16,
        lat_w: width / 16,
        audio_t: (f64::from(frames) / FPS as f64 * AUDIO_LATENTS_PER_S as f64).round() as i32,
        text_rows_max: 4096,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_snap_to_the_next_seventeen_n_plus_five() {
        assert_eq!(align_frames(1), 5);
        assert_eq!(align_frames(5), 5);
        assert_eq!(align_frames(6), 22);
        assert_eq!(align_frames(22), 22);
        assert_eq!(align_frames(23), 39);
        assert_eq!(align_frames(124), 124);
    }

    #[test]
    fn the_shape_matches_what_the_abi_documents() {
        let s = shape_for(480, 864, 124).unwrap();
        assert_eq!((s.frames, s.latent_t, s.lat_h, s.lat_w), (124, 37, 30, 54));
        assert_eq!(s.audio_t, 207);
        let s = shape_for(480, 864, 22).unwrap();
        assert_eq!((s.frames, s.latent_t, s.audio_t), (22, 7, 37));
        // a one-frame request still produces the model's minimum clip
        let s = shape_for(32, 32, 1).unwrap();
        assert_eq!((s.frames, s.latent_t, s.lat_h), (5, 2, 2));
    }

    #[test]
    fn invalid_canvases_are_refused() {
        assert!(shape_for(480, 863, 124).is_none()); // not a multiple of 32
        assert!(shape_for(16, 864, 124).is_none()); // too small
        assert!(shape_for(480, 864, 0).is_none()); // no frames
        assert!(shape_for(MAX_SIDE + 32, 864, 124).is_none());
        assert!(shape_for(480, 864, MAX_FRAMES + 1).is_none());
    }

    #[test]
    fn the_schedule_deduplicates_on_exact_equality() {
        let s = Schedule::new(31, 12.0);
        assert_eq!(s.sigmas.len(), 31);
        assert_eq!(s.timesteps.len(), 30);
        assert_eq!(s.sigmas[0], 1.0);
        assert_eq!(*s.sigmas.last().unwrap(), 0.0);
        // timesteps are 1 - sigma of every point but the last
        for (i, t) in s.timesteps.iter().enumerate() {
            assert_eq!(*t, 1.0 - s.sigmas[i]);
        }
        // a single point is the whole grid
        assert_eq!(Schedule::new(1, 12.0).sigmas, vec![1.0]);
        assert!(Schedule::new(1, 12.0).timesteps.is_empty());
    }

    #[test]
    fn a_plain_layout_packs_text_then_audio_then_video() {
        let l = Layout::new(13, 7, 30, 54, 37, &[], &[]).unwrap();
        assert_eq!(l.video_rows, 7 * 15 * 27);
        assert_eq!(l.audio_rows, 74);
        assert_eq!(l.ref_rows, 0);
        assert_eq!(l.seq_len, 13 + 74 + 7 * 15 * 27);
        // text rows carry the text tag and climb by one
        assert_eq!(l.adaln_rows[0], 1);
        assert_eq!(l.pos[3 * 5], 5.0);
        // audio follows, then video, and the classes say so
        assert_eq!(l.tclass[13], 1);
        assert_eq!(l.tclass[l.seq_len - 1], 0);
        assert_eq!(l.adaln_rows[l.seq_len - 1], 0);
        assert_eq!(l.video_span_start(), 13 + 74);
    }

    #[test]
    fn marking_a_vision_span_covers_its_flanking_tokens() {
        let mut l = Layout::new(12, 7, 4, 6, 5, &[], &[]).unwrap();
        l.mark_vision(2, 4).unwrap();
        // the span itself and one token either side
        for r in 1..7 {
            assert_eq!(l.adaln_rows[r], 0, "row {r}");
        }
        assert_eq!(l.adaln_rows[0], 1);
        assert_eq!(l.adaln_rows[7], 1);
        assert!(l.mark_vision(11, 2).is_err());
    }

    #[test]
    fn reference_blocks_take_rows_before_the_target() {
        let image = Ref {
            kind: 0,
            latent_t: 1,
            lat_h: 4,
            lat_w: 6,
            audio_t: 0,
            has_audio: false,
        };
        let sound = Ref {
            kind: 1,
            latent_t: 0,
            lat_h: 0,
            lat_w: 0,
            audio_t: 5,
            has_audio: true,
        };
        let l = Layout::new(12, 7, 4, 6, 5, &[image, sound], &[]).unwrap();
        assert_eq!(l.ref_rows, 2 * 3 + 5 * 2);
        assert_eq!(l.seq_len, 12 + l.ref_rows + 10 + 7 * 2 * 3);
        assert_eq!(l.ref_segs.len(), 2);
        // the conditioning classes, not the target's
        assert!(l.ref_segs.iter().all(|s| s.kind == 0 || s.kind == 1));
        assert_eq!(l.tclass[l.ref_segs[0].row0], 2); // the image, conditioning video
        assert_eq!(l.tclass[l.ref_segs[1].row0], 3); // the sound, conditioning audio
        // The final norm has two classes, and the reference rows do not fit them — which is why
        // only the generated span is uploaded to it. That span does fit.
        assert!(
            crate::dispatch::classes_fit(&l.tclass, 2).is_err(),
            "reference rows carry conditioning classes"
        );
        let generated = 12 + l.ref_rows;
        assert!(
            crate::dispatch::classes_fit(&l.tclass[generated..], 2).is_ok(),
            "the generated rows are the final norm's own two classes"
        );
    }

    /// The image reference the differential harness drives: 256x256x5 with one 64x64 reference.
    /// Its conditioning rows carry class 2, which is why the final norm is given the generated span
    /// alone — uploading the whole array against its two classes refuses every reference run.
    #[test]
    fn the_final_norms_span_is_the_generated_rows() {
        let image = Ref {
            kind: 0,
            latent_t: 1,
            lat_h: 4,
            lat_w: 4,
            audio_t: 0,
            has_audio: false,
        };
        let l = Layout::new(18, 2, 16, 16, 9, &[image], &[]).unwrap();
        let lr = 18 + l.ref_rows;
        assert!(l.ref_rows > 0, "the reference takes rows");
        assert!(
            crate::dispatch::classes_fit(&l.tclass, 2).is_err(),
            "the whole array does not fit the final norm's table"
        );
        assert!(
            crate::dispatch::classes_fit(&l.tclass[lr..lr + l.audio_rows + l.video_rows], 2).is_ok(),
            "the generated span does"
        );
    }

    #[test]
    fn a_keyframe_sits_on_the_target_grid() {
        let kf = Keyframe {
            frame_index: 0,
            audio_t: 0,
            has_audio: false,
        };
        let l = Layout::new(4, 7, 4, 6, 5, &[], &[kf]).unwrap();
        assert_eq!(l.ref_rows, 2 * 3);
        assert_eq!(l.ref_segs.len(), 1);
        assert_eq!(l.ref_segs[0].kind, 3);
        assert_eq!(l.tclass[l.ref_segs[0].row0], 2);
    }
}
