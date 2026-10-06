//! Propose cache thresholds from a complete `H3_STAGE_TRACE=1 --cache-observe` run.
use serde_json::{json, Value};

fn parse_events(log: &str) -> Result<Vec<Value>, serde_json::Error> {
    log.lines()
        .filter_map(|line| line.split_once("H3_STAGE {").map(|(_, tail)| tail))
        .map(|tail| serde_json::from_str(&format!("{{{tail}")))
        .collect()
}

fn quantile(mut values: Vec<f64>, p: f64) -> f64 {
    values.sort_by(f64::total_cmp);
    let at = (values.len() - 1) as f64 * p;
    let low = at.floor() as usize;
    let high = at.ceil() as usize;
    values[low] + (values[high] - values[low]) * (at - low as f64)
}

fn proposals(events: &[Value]) -> Result<Vec<Value>, &'static str> {
    let schedules: Vec<_> = events.iter().filter(|e| e["stage"] == "schedule").collect();
    if schedules.len() != 1 || schedules[0]["cache_mode"] != "observe" {
        return Err("expected one complete observation-only trajectory");
    }
    let count = schedules[0]["evaluations"]
        .as_u64()
        .ok_or("missing evaluation count")?;
    let decisions: Vec<_> = events
        .iter()
        .filter(|e| e["stage"] == "cache_decision")
        .collect();
    if count < 6
        || decisions.len() as u64 != count
        || decisions
            .iter()
            .enumerate()
            .any(|(i, e)| e["evaluation"].as_u64() != Some(i as u64 + 1))
    {
        return Err("missing evaluations or trajectory too short to calibrate");
    }
    let mut metrics = Vec::new();
    for event in decisions {
        if event["skipped"].as_bool() != Some(false) {
            return Err("calibration requires all evaluations to run in full");
        }
        let values = event["relative_conditioning_audio_video"]
            .as_array()
            .filter(|v| v.len() == 3)
            .ok_or("malformed modality metrics")?;
        let mut row = [0.; 3];
        for (out, value) in row.iter_mut().zip(values) {
            *out = value
                .as_f64()
                .filter(|v| v.is_finite() && *v >= 0.)
                .ok_or("nonfinite or malformed modality metrics")?;
        }
        metrics.push(row);
    }
    Ok([0.25, 0.5]
        .into_iter()
        .map(|percentile| {
            let thresholds: [f64; 3] = std::array::from_fn(|k| {
                quantile(
                    metrics[2..metrics.len() - 2].iter().map(|v| v[k]).collect(),
                    percentile,
                )
                .max(1e-12)
            });
            let mut accumulated = [0.; 3];
            let mut last_skipped = false;
            let mut skips = Vec::new();
            for (step, values) in metrics.iter().enumerate() {
                for k in 0..3 {
                    accumulated[k] += values[k];
                }
                let skip = (2..metrics.len() - 2).contains(&step)
                    && !last_skipped
                    && (0..3).all(|k| accumulated[k] < thresholds[k]);
                if skip {
                    skips.push(step + 1);
                } else {
                    accumulated = [0.; 3];
                }
                last_skipped = skip;
            }
            let args: Vec<_> = std::iter::once("--cache-thresholds".to_string())
                .chain(thresholds.iter().map(f64::to_string))
                .collect();
            json!({"quantile": percentile, "conditioning_audio_video_thresholds": thresholds,
               "predicted_skipped_evaluations": skips, "cli_arguments": args})
        })
        .collect())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let path = args
        .next()
        .ok_or("usage: cargo run --example cache_calibrate -- TRACE")?;
    if args.next().is_some() {
        return Err("expected exactly one trace file".into());
    }
    let source = std::fs::read(path)?;
    let trials = proposals(&parse_events(std::str::from_utf8(&source)?)?)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "observation_sha256": hrx::bundle::digest(&source), "trials": trials,
            "release_gate": "Require at least 10% measured whole-render improvement and acceptable identity, motion, temporal coherence, dialogue and audio synchronization against identical initial noise."
        }))?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observation() -> Vec<Value> {
        let mut events =
            vec![json!({"stage": "schedule", "cache_mode": "observe", "evaluations": 20})];
        events.extend((0..20).map(|i| json!({"stage": "cache_decision", "evaluation": i + 1,
            "skipped": false, "relative_conditioning_audio_video": [0.01 * i as f64, 0.02 * i as f64, 0.03 * i as f64]})));
        events
    }

    #[test]
    fn progress_prefixes_do_not_hide_events() {
        let events = parse_events("test fixture ... H3_STAGE {\"stage\":\"conditioning_start\"}\n\r  step 1/2  100 s\r  step 2/2  200 sH3_STAGE {\"stage\":\"denoise_finished\"}\nordinary output\nH3_STAGE {\"stage\":\"video_decode_start\"}\n").unwrap();
        assert_eq!(
            events
                .iter()
                .map(|e| e["stage"].as_str().unwrap())
                .collect::<Vec<_>>(),
            [
                "conditioning_start",
                "denoise_finished",
                "video_decode_start"
            ]
        );
    }

    #[test]
    fn calibration_preserves_warmup_cooldown_and_forced_refresh() {
        let trials = proposals(&observation()).unwrap();
        assert_eq!(trials.len(), 2);
        for (trial, expected) in trials.iter().zip([vec![3, 5], vec![3, 5, 7, 9]]) {
            let skips: Vec<_> = trial["predicted_skipped_evaluations"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap())
                .collect();
            assert_eq!(skips, expected);
            assert!(skips.iter().all(|s| (3..=18).contains(s)));
            assert!(skips.windows(2).all(|s| s[1] - s[0] > 1));
        }
        for (trial, threshold) in trials.iter().zip([0.0575, 0.095]) {
            for k in 0..3 {
                let actual = trial["conditioning_audio_video_thresholds"][k]
                    .as_f64()
                    .unwrap();
                assert!((actual - threshold * (k + 1) as f64).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn calibration_rejects_partial_cached_and_invalid_traces() {
        let events = observation();
        assert!(proposals(&events[..1]).is_err());
        for (key, value) in [
            ("skipped", json!(true)),
            ("relative_conditioning_audio_video", json!([null, 0., 0.])),
            ("relative_conditioning_audio_video", json!([-1., 0., 0.])),
            ("evaluation", json!(99)),
        ] {
            let mut bad = events.clone();
            bad[3][key] = value;
            assert!(proposals(&bad).is_err(), "accepted {key}");
        }
    }
}
