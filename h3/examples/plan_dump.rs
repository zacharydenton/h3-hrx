//! Dumps every recipe a plan builds, with a checksum of its assembled bytes, for comparing this
//! implementation against the C one:  plan_dump dit <checkpoint>
use h3::checkpoint::Checkpoint;
use h3::weights::Recipe;
use std::collections::BTreeMap;

fn digest(bytes: &[u8]) -> (u64, u64) {
    let mut sum: u64 = 0;
    let mut weighted: u64 = 0;
    for (i, b) in bytes.iter().enumerate() {
        sum = sum.wrapping_add(u64::from(*b));
        weighted = weighted.wrapping_add((i as u64).wrapping_mul(u64::from(*b)));
    }
    (sum, weighted)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let which = args.next().expect("usage: plan_dump <plan> <checkpoint>");
    let path = args.next().expect("usage: plan_dump <plan> <checkpoint>");
    let ck = Checkpoint::open(&path).expect("open");
    let mut table: BTreeMap<String, Recipe> = BTreeMap::new();
    match which.as_str() {
        "dit" => h3::plan::dit::plan(&ck, &mut table).expect("plan"),
        "te" => h3::plan::te::plan(&ck, &mut table).expect("plan"),
        "vvae" => h3::plan::vvae::plan(&ck, &mut table).expect("plan"),
        "avae" => h3::plan::avae::plan(&ck, &mut table).expect("plan"),
        other => panic!("unknown plan {other}"),
    }
    for (name, recipe) in &table {
        let bytes = recipe.assemble(&ck).expect("assemble");
        let (sum, weighted) = digest(&bytes);
        println!("{name}\t{}\t{sum}\t{weighted}", bytes.len());
    }
}
