//! Native end-to-end throughput for new optional measurement interfaces.
#[cfg(all(feature = "precision-spectrum", feature = "precision-loudness"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use audioscopy::precision::loudness::{LoudnessBackend as L, WideLoudnessMeter};
    use audioscopy::precision::spectral::{
        CrossSpectrumAnalyzer, QuietNoiseAnalyzer, SpectrumBackend as B,
    };
    use audioscopy::precision::Precision;
    use audioscopy::{Band, ChannelRole as R, Encoding, Format, NoiseView, Weighting};
    use std::hint::black_box;
    use std::time::Instant;

    let a: Vec<_> = std::env::args().collect();
    if a.len() != 6 {
        return Err("usage: precision_batch_profile MODE KIND RATE CHANNELS SECONDS".into());
    }
    let mode: i32 = a[1].parse()?;
    let kind = a[2].as_str();
    let rate: u32 = a[3].parse()?;
    let channels: usize = a[4].parse()?;
    let seconds: u64 = a[5].parse()?;
    if !(8000..=768000).contains(&rate) || !(1..=6).contains(&channels) || seconds == 0 {
        return Err("invalid geometry".into());
    }
    let frames = u64::from(rate)
        .checked_mul(seconds)
        .ok_or("frame overflow")?;
    let precision = if mode == 2 {
        Precision::Extended80
    } else {
        Precision::DoubleDouble
    };
    let mut seed = 0xc783_9145_u32;
    let pcm: Vec<_> = (0..4093 * channels)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (f64::from(seed >> 8) - 8388608.0) / 16777216.0
        })
        .collect();
    let f = Format {
        sample_rate: rate,
        channels,
        encoding: Encoding::Float,
        loudness_roles: None,
    };
    let start = Instant::now();
    if kind == "loudness" {
        let b = match mode {
            2 => L::X87,
            3 => L::DoubleDoublePortable,
            4 => L::DoubleDoubleFma,
            _ => return Err("unsupported loudness backend".into()),
        };
        let roles = [
            R::Left,
            R::Right,
            R::Center,
            R::Lfe,
            R::LeftSurround,
            R::RightSurround,
        ];
        let mut meter = WideLoudnessMeter::new(rate, &roles[..channels], precision, b)?;
        let mut left = frames;
        while left > 0 {
            let take = left.min(4093) as usize;
            meter.push_interleaved(black_box(&pcm[..take * channels]))?;
            left -= take as u64;
        }
        let r = black_box(meter.finish()?);
        assert_eq!(r.frames, frames);
        assert_eq!(r.backend, b);
    } else {
        let b = match mode {
            2 => B::X87,
            3 => B::DoubleDoublePortable,
            4 => B::DoubleDoubleFma,
            6 => B::DoubleDoubleAvx2,
            7 => B::DoubleDoubleAvx512,
            8 => B::DoubleDoubleNeon,
            _ => return Err("unsupported spectrum backend".into()),
        };
        if kind == "cross" {
            let mut meter = CrossSpectrumAnalyzer::new(f, precision, b)?;
            let mut left = frames;
            while left > 0 {
                let take = left.min(4093) as usize;
                meter.push_interleaved(black_box(&pcm[..take * channels]))?;
                left -= take as u64;
            }
            let r = black_box(meter.finish()?);
            assert_eq!(r.frames, frames);
            assert_eq!(r.backend, b);
        } else if kind == "noise" {
            let view = NoiseView {
                band: Band::Full,
                weighting: Weighting::A,
            };
            let mut meter = QuietNoiseAnalyzer::new(f, 0.05, view, precision, b)?;
            let mut left = frames;
            while left > 0 {
                let take = left.min(4093) as usize;
                meter.push_interleaved(black_box(&pcm[..take * channels]))?;
                left -= take as u64;
            }
            let r = black_box(meter.finish()?.ok_or("no eligible frames")?);
            assert_eq!(r.quiet_frames, frames);
            assert_eq!(r.backend, b);
        } else {
            return Err("unknown measurement".into());
        }
    }
    println!("WIDE_PROFILE mode={mode} kind={kind} rate={rate} channels={channels} frames={frames} wall_s={:.9}", start.elapsed().as_secs_f64());
    Ok(())
}

#[cfg(not(all(feature = "precision-spectrum", feature = "precision-loudness")))]
fn main() {
    eprintln!("requires precision-spectrum and precision-loudness");
    std::process::exit(2);
}
