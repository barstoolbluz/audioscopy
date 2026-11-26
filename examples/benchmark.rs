//! Reproducible selective-measurement throughput smoke benchmark.
//! Run: cargo run --release --example benchmark -- 60 96000 [1|2|6]
//! Shared public APIs only, so the same harness runs against approved R5.
use audioscopy::{Analyzer, ChannelRole, Encoding, Format, Measurement as M, Measurements, Settings};
use std::hint::black_box;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() > 4 {
        return Err("usage: benchmark [seconds] [rate] [channels]".into());
    }
    let seconds: usize = args.get(1).map_or("60", String::as_str).parse()?;
    let rate: u32 = args.get(2).map_or("48000", String::as_str).parse()?;
    let channels: usize = args.get(3).map_or("2", String::as_str).parse()?;
    if seconds == 0 || !(8000..=768000).contains(&rate) {
        return Err("invalid duration or rate".into());
    }
    use ChannelRole::*;
    let roles = match channels {
        1 | 2 => None,
        6 => Some(vec![Left, Right, Center, Lfe, LeftSurround, RightSurround]),
        _ => return Err("use 1, 2 or 6 channels".into()),
    };
    let format = Format { sample_rate: rate, channels, encoding: Encoding::Float, loudness_roles: roles };
    let mut pcm = vec![0.0; rate as usize * channels];
    let mut seed = 0x13579bdf_u64;
    let gains = [1.0, 0.7, 0.5, 0.2, -0.4, 0.6];
    for frame in pcm.chunks_exact_mut(channels) {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        let x = ((seed >> 11) as f64 / (1_u64 << 53) as f64 - 0.5) * 0.1;
        for (sample, gain) in frame.iter_mut().zip(gains) {
            *sample = x * gain;
        }
    }
    let quiet: Vec<_> = pcm.iter().map(|x| x * 1e-4).collect();
    for (label, selection) in [
        ("none", Measurements::default()),
        ("bits", Measurements::new(&[M::BitsInUse])),
        ("stereo", Measurements::new(&[M::Correlation, M::Balance])),
        ("noise", Measurements::new(&[M::NoiseFloor, M::EffectiveBits])),
        ("power-spectrum", Measurements::new(&[M::Above20Khz, M::AverageSpectrum])),
        ("bandwidth", Measurements::new(&[M::Bandwidth])),
        ("spectral", Measurements::new(&[M::Bandwidth, M::AverageSpectrum])),
        ("loudness", Measurements::new(&[M::LoudnessRange, M::ShortTermMaximum, M::LoudnessOverTime])),
        ("all", Measurements::all()),
    ] {
        if label == "stereo" && channels != 2 {
            continue;
        }
        let start = Instant::now();
        let mut a = Analyzer::new(format.clone(), selection, Settings::default())?;
        for i in 0..seconds {
            // Alternate active programme and nonzero qualifying quiet PCM.
            // Silence-only noise would skip the actual quiet FFT workload.
            let source = if i % 4 >= 2 { &quiet } else { &pcm };
            for c in source.chunks(4096 * channels) {
                a.push_interleaved(c)?;
            }
        }
        let report = a.finish()?;
        black_box(&report);
        let elapsed = start.elapsed().as_secs_f64();
        println!("{label:>14} {seconds}s @ {rate}Hz {channels}ch: {elapsed:.9}s ({:.2}x real time), {} frames", seconds as f64 / elapsed, report.frames());
    }
    Ok(())
}
