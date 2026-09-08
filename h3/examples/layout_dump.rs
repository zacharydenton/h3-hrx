//! The packed layout and the sigma schedule, in the same order as the C implementation's dump.
use h3::layout::{Keyframe, Layout, Ref, Schedule};

/// C's %.17g, which is how the C dump prints its positions.
fn g17(v: f64) -> String {
    h3::compile::num(v)
}

fn emit(name: &str, l: &Layout) {
    println!(
        "layout {name} seq={} ref={} audio={} video={}",
        l.seq_len, l.ref_rows, l.audio_rows, l.video_rows
    );
    for r in 0..l.seq_len {
        println!(
            "r {r} {} {} {} {} {}",
            g17(l.pos[3 * r]),
            g17(l.pos[3 * r + 1]),
            g17(l.pos[3 * r + 2]),
            l.adaln_rows[r],
            l.tclass[r]
        );
    }
    for (i, s) in l.ref_segs.iter().enumerate() {
        println!(
            "s {i} {} {} {} {} {} {} {} {} {}",
            s.kind,
            s.row0,
            s.rows,
            s.latent_t,
            s.lat_h,
            s.lat_w,
            s.audio_t,
            s.audio as i32,
            s.index
        );
    }
}

fn main() {
    for (t, lt, h, w, at) in [
        (13, 7, 30, 54, 37),
        (1, 2, 2, 2, 1),
        (93, 37, 48, 84, 207),
        (40, 7, 4, 6, 5),
        (4, 2, 8, 8, 3),
    ] {
        emit(
            &format!("plain_{t}_{lt}_{h}_{w}_{at}"),
            &Layout::new(t, lt, h, w, at, &[], &[]).unwrap(),
        );
    }

    let img = Ref {
        kind: 0,
        latent_t: 1,
        lat_h: 4,
        lat_w: 6,
        audio_t: 0,
        has_audio: false,
    };
    let snd = Ref {
        kind: 1,
        latent_t: 0,
        lat_h: 0,
        lat_w: 0,
        audio_t: 5,
        has_audio: true,
    };
    let vid = Ref {
        kind: 2,
        latent_t: 3,
        lat_h: 8,
        lat_w: 8,
        audio_t: 0,
        has_audio: false,
    };
    let vids = Ref {
        kind: 2,
        latent_t: 3,
        lat_h: 8,
        lat_w: 8,
        audio_t: 4,
        has_audio: true,
    };
    emit(
        "ref_img",
        &Layout::new(12, 7, 4, 6, 5, std::slice::from_ref(&img), &[]).unwrap(),
    );
    emit(
        "ref_snd",
        &Layout::new(12, 7, 4, 6, 5, std::slice::from_ref(&snd), &[]).unwrap(),
    );
    emit(
        "ref_vid",
        &Layout::new(12, 7, 4, 6, 5, std::slice::from_ref(&vid), &[]).unwrap(),
    );
    emit(
        "ref_vids",
        &Layout::new(12, 7, 4, 6, 5, std::slice::from_ref(&vids), &[]).unwrap(),
    );
    emit(
        "ref_all",
        &Layout::new(20, 7, 4, 6, 5, &[img.clone(), snd.clone(), vid, vids], &[]).unwrap(),
    );

    let k0 = Keyframe {
        frame_index: 0,
        audio_t: 0,
        has_audio: false,
    };
    let k1 = Keyframe {
        frame_index: 123,
        audio_t: 0,
        has_audio: false,
    };
    let ka = Keyframe {
        frame_index: 0,
        audio_t: 6,
        has_audio: true,
    };
    emit(
        "kf_first",
        &Layout::new(4, 7, 4, 6, 5, &[], std::slice::from_ref(&k0)).unwrap(),
    );
    emit(
        "kf_last",
        &Layout::new(4, 7, 4, 6, 5, &[], std::slice::from_ref(&k1)).unwrap(),
    );
    emit(
        "kf_audio",
        &Layout::new(4, 7, 4, 6, 5, &[], std::slice::from_ref(&ka)).unwrap(),
    );
    emit(
        "kf_both",
        &Layout::new(4, 7, 4, 6, 5, &[], &[k0.clone(), k1]).unwrap(),
    );
    emit(
        "kf_and_refs",
        &Layout::new(9, 7, 4, 6, 5, &[img, snd], &[k0, ka]).unwrap(),
    );

    for steps in [1usize, 2, 3, 5, 21, 31, 51] {
        for shift in [12.0f64, 3.0, 1.0, 7.5] {
            let s = Schedule::new(steps, shift);
            println!(
                "sched {steps} {} n={} m={}",
                g17(shift),
                s.sigmas.len(),
                s.timesteps.len()
            );
            for v in &s.sigmas {
                println!("  sg {}", format_g9(*v));
            }
            for v in &s.timesteps {
                println!("  ts {}", format_g9(*v));
            }
        }
    }
}

/// C's %.9g on a float promoted to double.
fn format_g9(v: f32) -> String {
    let d = f64::from(v);
    if d == 0.0 {
        return "0".into();
    }
    let exp = format!("{:.8e}", d);
    let e: i32 = exp[exp.find('e').unwrap() + 1..].parse().unwrap_or(0);
    if !(-4..9).contains(&e) {
        let s = format!("{:.8e}", d);
        let (m, x) = s.split_at(s.find('e').unwrap());
        let value: i32 = x[1..].parse().unwrap_or(0);
        let m = m.trim_end_matches('0').trim_end_matches('.');
        format!(
            "{m}e{}{:02}",
            if value < 0 { '-' } else { '+' },
            value.abs()
        )
    } else {
        let s = format!("{:.*}", (8 - e).max(0) as usize, d);
        let s = if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            s
        };
        s
    }
}
