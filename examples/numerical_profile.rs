//! Reproducible per-selection native qualification; shared public APIs only.
use audioscopy::{
    AlbumAnalyzer, Analyzer, ChannelRole, Encoding, Format, Measurement as M, Measurements,
    NoiseView, Settings,
};
use std::hint::black_box;
use std::time::{Duration, Instant};

fn group(name: &str) -> Result<Measurements, String> {
    let s: &[M] = match name {
        "none" => &[],
        "range" => &[M::LoudnessRange],
        "short" => &[M::ShortTermMaximum, M::LoudnessOverTime],
        "loudness" => &[M::LoudnessRange, M::ShortTermMaximum, M::LoudnessOverTime],
        "spectrum" => &[M::AverageSpectrum, M::Above20Khz],
        "bandwidth" => &[M::Bandwidth, M::LikelySourceRate, M::LossyCodingPattern],
        "noise" => &[
            M::NoiseFloor,
            M::EffectiveBits,
            M::NoiseSpectrum,
            M::HumAndIdleTones,
            M::NoiseByBand,
            M::DynamicRange,
        ],
        "stereo" => &[M::Correlation, M::Balance, M::Width, M::MonoCompatibility],
        "correlation" => &[M::Correlation],
        "stereo_spectral" => &[M::BassWidth, M::InterChannelDelay],
        "bits" => &[M::BitsInUse],
        "temporal" => &[M::Limiting, M::FlatToppedPeaks, M::ClicksPerMinute],
        "all" | "album" => return Ok(Measurements::all()),
        "album_loudness" => &[M::LoudnessRange, M::ShortTermMaximum, M::LoudnessOverTime],
        _ => return Err(format!("unknown group: {name}")),
    };
    Ok(Measurements::new(s))
}

// Generation excluded from measured durations. Reuse one second of deterministic
// mixed tones and noise, in exact 24-bit integer PCM, plus low-level quantized
// passages. A quiet/noisy alternation exercises both quiet and programme DSP.
fn fixture(rate: u32, channels: usize) -> (Vec<f64>, Vec<f64>) {
    let mut active = Vec::with_capacity(rate as usize * channels);
    let mut quiet = Vec::with_capacity(rate as usize * channels);
    let mut seed = 0x13579bdf12345678_u64;
    let scale = 8388608.0;
    for i in 0..rate as usize {
        let t = i as f64 / f64::from(rate);
        for c in 0..channels {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let noise = (seed >> 11) as f64 / (1_u64 << 53) as f64 - 0.5;
            let signal = 0.17 * (std::f64::consts::TAU * (997.0 + 31.0 * c as f64) * t).sin()
                + 0.05 * (std::f64::consts::TAU * (211.0 + 17.0 * c as f64) * t).sin()
                + 0.025 * noise;
            active.push((signal * scale).round() / scale);
            quiet.push((noise * 8.0).round() / scale);
        }
    }
    (active, quiet)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 5 {
        return Err("usage: numerical_profile GROUP RATE CHANNELS SECONDS".into());
    }
    let name = a[1].as_str();
    let selection = group(name)?;
    let rate: u32 = a[2].parse()?;
    let channels: usize = a[3].parse()?;
    let seconds: usize = a[4].parse()?;
    if !(8000..=768000).contains(&rate) || seconds == 0 {
        return Err("invalid rate or duration".into());
    }
    use ChannelRole::*;
    let roles = match channels {
        1 | 2 => None,
        3 => Some(vec![Left, Right, Center]),
        6 => Some(vec![Left, Right, Center, Lfe, LeftSurround, RightSurround]),
        _ => return Err("supported benchmark geometry: 1,2,3,6 channels".into()),
    };
    let format = Format {
        sample_rate: rate,
        channels,
        encoding: Encoding::Integer(24),
        loudness_roles: roles,
    };
    let (active, quiet) = fixture(rate, channels);
    let (mut init, mut ingest, mut finish, mut read) = (
        Duration::ZERO,
        Duration::ZERO,
        Duration::ZERO,
        Duration::ZERO,
    );
    let frames = (seconds as u64)
        .checked_mul(u64::from(rate))
        .ok_or("frame overflow")?;
    let wall = Instant::now();
    if name.starts_with("album") {
        let start = Instant::now();
        let mut analyzer = AlbumAnalyzer::new(selection, Settings::default());
        init += start.elapsed();
        // Track boundaries every six seconds, including a short last track.
        for i in 0..seconds {
            if i % 6 == 0 {
                let start = Instant::now();
                analyzer.begin_track(format.clone())?;
                init += start.elapsed();
            }
            let block = if i % 4 >= 2 { &quiet } else { &active };
            let start = Instant::now();
            for chunk in block.chunks(4093 * channels) {
                analyzer.push_interleaved(chunk)?;
            }
            ingest += start.elapsed();
            if i % 6 == 5 || i + 1 == seconds {
                let start = Instant::now();
                analyzer.end_track()?;
                finish += start.elapsed();
            }
        }
        let start = Instant::now();
        let album = analyzer.finish()?;
        finish += start.elapsed();
        assert_eq!(
            album.tracks().iter().map(|t| t.frames()).sum::<u64>(),
            frames
        );
        let start = Instant::now();
        for m in M::ALL.into_iter().filter(|m| selection.contains(*m)) {
            black_box(album.measure(m, NoiseView::default()));
            for track in album.tracks() {
                black_box(track.measure(m, NoiseView::default()));
            }
        }
        read += start.elapsed();
    } else {
        let start = Instant::now();
        let mut analyzer = Analyzer::new(format, selection, Settings::default())?;
        init += start.elapsed();
        let start = Instant::now();
        for i in 0..seconds {
            let block = if i % 4 >= 2 { &quiet } else { &active };
            for chunk in block.chunks(4093 * channels) {
                analyzer.push_interleaved(chunk)?;
            }
        }
        ingest += start.elapsed();
        let start = Instant::now();
        let track = analyzer.finish()?;
        finish += start.elapsed();
        assert_eq!(track.frames(), frames);
        let start = Instant::now();
        for m in M::ALL.into_iter().filter(|m| selection.contains(*m)) {
            black_box(track.measure(m, NoiseView::default()));
        }
        read += start.elapsed();
    }
    let elapsed = wall.elapsed().as_secs_f64();
    println!("PROFILE\t{name}\t{rate}\t{channels}\t{seconds}\t{frames}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{elapsed:.9}",init.as_secs_f64(),ingest.as_secs_f64(),finish.as_secs_f64(),read.as_secs_f64());
    Ok(())
}
