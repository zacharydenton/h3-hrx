//! The kernel-selection choices, in the same order as the C implementation's dump, for comparison.
use h3::model::*;

fn main() {
    let ks = [
        2048usize, 4096, 5376, 7168, 8192, 14336, 16384, 21504, 25600,
    ];
    let ns = [2048usize, 5376, 6144, 7168, 14336, 16384, 21504];
    let toks = [
        1usize, 13, 63, 64, 255, 256, 257, 512, 517, 1024, 1797, 4096, 16000, 32767, 32768, 37723,
        149184,
    ];
    for t in toks {
        println!("mg {t} {} {}", m_group_for(t, 256), m_group_for(t, 128));
        print!("gy {t}");
        for g in [1u32, 2, 3, 4, 15] {
            print!(" {}", gemm_grid_y(t, g, 256));
        }
        println!();
        for k in ks {
            for n in ns {
                for b in [4usize, 8, 16] {
                    println!("gm {t} {k} {n} {b} {}", gemm_m_group_for(t, k, n, b));
                }
            }
        }
        for k in ks {
            for n in ns {
                println!("vf {t} {k} {n} {}", vae_fast_m_group_for(t, k, n));
            }
        }
    }
    let mut w = 32;
    while w <= 32768 {
        match lanes_for(w) {
            Some(l) => println!("ln {w} {l}"),
            None => println!("ln {w} none"),
        }
        w += 32;
    }
    let mut k = 64;
    while k <= 32768 {
        for b in [4usize, 8, 16] {
            println!("gp {k} {b} {}", gemm_pitch(k, b));
        }
        k += 64;
    }
}
