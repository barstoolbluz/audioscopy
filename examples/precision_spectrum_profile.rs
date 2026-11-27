//! Paired public average-spectrum throughput, identical PCM for every mode.
#[cfg(feature = "precision-spectrum")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use audioscopy::precision::spectral::{SpectrumAnalyzer, SpectrumBackend as B};
    use audioscopy::precision::Precision;
    use audioscopy::{Analyzer, Encoding, Format, Measurement, Measurements, NoiseView, Settings};
    use std::hint::black_box;
    use std::time::{Duration, Instant};

    let a: Vec<_> = std::env::args().collect();
    if a.len() != 5 {
        return Err("usage: precision_spectrum_profile MODE RATE CHANNELS SECONDS".into());
    }
    let mode: i32 = a[1].parse()?;
    let rate: u32 = a[2].parse()?;
    let channels: usize = a[3].parse()?;
    let seconds: usize = a[4].parse()?;
    if !(8000..=768000).contains(&rate) || !(1..=64).contains(&channels) || seconds == 0 {
        return Err("invalid geometry".into());
    }
    let f = Format {
        sample_rate: rate,
        channels,
        encoding: Encoding::Integer(24),
        loudness_roles: None,
    };
    let mut seed = 0xc783_9145_u32;
    let pcm: Vec<_> = (0..4093 * channels)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (f64::from(seed >> 8) - 8388608.0) / 8388608.0
        })
        .collect();
    let frames = u64::from(rate)
        .checked_mul(seconds as u64)
        .ok_or("frame overflow")?;
    let mut read = Duration::ZERO;
    let (init, ingest, finish, actual) = if mode == 0 {
        let start = Instant::now();
        let mut analyzer = Analyzer::new(
            f,
            Measurements::new(&[Measurement::AverageSpectrum]),
            Settings::default(),
        )?;
        let init = start.elapsed();
        let start = Instant::now();
        let mut left = frames;
        while left > 0 {
            let take = left.min(4093) as usize;
            analyzer.push_interleaved(black_box(&pcm[..take * channels]))?;
            left -= take as u64;
        }
        let ingest = start.elapsed();
        let start = Instant::now();
        let result = analyzer.finish()?;
        let finish = start.elapsed();
        assert_eq!(result.frames(), frames);
        let start = Instant::now();
        black_box(result.measure(Measurement::AverageSpectrum, NoiseView::default()));
        read = start.elapsed();
        (init, ingest, finish, 0)
    } else {
        let b = match mode {
            1 => B::DoubleDouble,
            2 => B::X87,
            3 => B::DoubleDoublePortable,
            4 => B::DoubleDoubleFma,
            6 => B::DoubleDoubleAvx2,
            7 => B::DoubleDoubleAvx512,
            8 => B::DoubleDoubleNeon,
            _ => return Err("unknown backend".into()),
        };
        let precision = if b == B::X87 {
            Precision::Extended80
        } else {
            Precision::DoubleDouble
        };
        let start = Instant::now();
        let mut analyzer = SpectrumAnalyzer::new(f, precision, b)?;
        let init = start.elapsed();
        let start = Instant::now();
        let mut left = frames;
        while left > 0 {
            let take = left.min(4093) as usize;
            analyzer.push_interleaved(black_box(&pcm[..take * channels]))?;
            left -= take as u64;
        }
        let ingest = start.elapsed();
        let start = Instant::now();
        let result = black_box(analyzer.finish()?);
        let finish = start.elapsed();
        assert_eq!(result.frames, frames);
        let actual = match result.backend {
            B::X87 => 2,
            B::DoubleDoublePortable => 3,
            B::DoubleDoubleFma => 4,
            B::DoubleDoubleAvx2 => 6,
            B::DoubleDoubleAvx512 => 7,
            B::DoubleDoubleNeon => 8,
            _ => return Err("unresolved backend".into()),
        };
        (init, ingest, finish, actual)
    };
    let wall = init + ingest + finish;
    println!(
        "WIDE_PROFILE mode={mode} actual={actual} rate={rate} channels={channels} frames={frames} init_s={:.9} ingest_s={:.9} finish_s={:.9} presentation_s={:.9} wall_s={:.9}",
        init.as_secs_f64(),
        ingest.as_secs_f64(),
        finish.as_secs_f64(),
        read.as_secs_f64(),
        wall.as_secs_f64(),
    );
    Ok(())
}

#[cfg(not(feature = "precision-spectrum"))]
fn main() {
    eprintln!("requires precision-spectrum");
    std::process::exit(2);
}
