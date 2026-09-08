//! The conditioning tables, in the same form as the C implementation's dump.
use h3::conditioning::{final_table, mods_table, temb};
use h3::model::*;
use h3::weights::Weights;

fn g9(v: f64) -> String {
    // C's %.9g, which is how the C dump prints these
    if v == 0.0 {
        return "0".into();
    }
    let e: i32 = {
        let s = format!("{:.8e}", v);
        s[s.find('e').unwrap() + 1..].parse().unwrap_or(0)
    };
    if !(-4..9).contains(&e) {
        let s = format!("{:.8e}", v);
        let (m, x) = s.split_at(s.find('e').unwrap());
        let value: i32 = x[1..].parse().unwrap_or(0);
        let m = m.trim_end_matches('0').trim_end_matches('.');
        format!(
            "{m}e{}{:02}",
            if value < 0 { '-' } else { '+' },
            value.abs()
        )
    } else {
        let s = format!("{:.*}", (8 - e).max(0) as usize, v);
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').into()
        } else {
            s
        }
    }
}

fn sums(what: &str, v: &[f32]) {
    let (mut s, mut sa, mut w) = (0.0f64, 0.0f64, 0.0f64);
    for (i, x) in v.iter().enumerate() {
        s += f64::from(*x);
        sa += f64::from(*x).abs();
        w += (i % 1024) as f64 * f64::from(*x);
    }
    println!(
        "{what} n={} sum={} abs={} weighted={} first={} last={}",
        v.len(),
        g9(s),
        g9(sa),
        g9(w),
        g9(f64::from(v[0])),
        g9(f64::from(v[v.len() - 1]))
    );
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: cond_dump <dit checkpoint>");
    let w = Weights::open(&path, h3::plan::dit::plan).expect("plan");
    let curve = w.host_f32("h3.adaln_t_table", 1025 * 8).expect("curve");
    let adaln_w: Vec<Vec<f32>> = (0..BLOCKS)
        .map(|i| {
            w.host_f32(&format!("h3.blocks.{i}.adaln.w"), MODALITIES * 6 * HID * 8)
                .expect("w")
        })
        .collect();
    let adaln_b: Vec<Vec<f32>> = (0..BLOCKS)
        .map(|i| {
            w.host_f32(&format!("h3.blocks.{i}.adaln.b"), MODALITIES * 6 * HID)
                .expect("b")
        })
        .collect();
    let final_w = w
        .host_f32("h3.final.adaln.w", 2 * HID * 8)
        .expect("final w");
    let final_b = w.host_f32("h3.final.adaln.b", 2 * HID).expect("final b");

    for t in [0.0f32, 0.25, 0.5, 0.999, 1.0, 0.123456] {
        let e = temb(&curve, t);
        let parts: Vec<String> = e.iter().map(|x| g9(f64::from(*x))).collect();
        println!("temb {} {}", g9(f64::from(t)), parts.join(" "));
    }
    let (tv, ta) = (temb(&curve, 0.7), temb(&curve, 0.3));
    let (tcv, tca) = (temb(&curve, 0.999), temb(&curve, 1.0));
    sums(
        "mods",
        &mods_table(&adaln_w, &adaln_b, &tv, &ta, &tcv, &tca),
    );
    sums("final", &final_table(&final_w, &final_b, &tv, &ta));
}
