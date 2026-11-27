use audioscopy::{
    Album, AlbumAnalyzer, AlbumLoudnessMethod, Analyzer, Band, BassWidth, BitDepth, Dither,
    Encoding, Error, Format, Measurement as M, Measurements, NoiseSnapshot, NoiseView, Outcome,
    Point, Settings, Track, Value, Weighting,
};
use rustfft::{num_complex::Complex, FftPlanner};
use std::f64::consts::TAU;

fn format(rate: u32, channels: usize, encoding: Encoding) -> Format {
    Format {
        sample_rate: rate,
        channels,
        encoding,
        loudness_roles: None,
    }
}
fn scan(f: Format, selected: &[M], pcm: &[f64]) -> Track {
    let mut a = Analyzer::new(f.clone(), Measurements::new(selected), Settings::default()).unwrap();
    // Deliberately not aligned with FFT, block, or short-term clocks.
    for chunk in pcm.chunks(997 * f.channels) {
        a.push_interleaved(chunk).unwrap();
    }
    a.finish().unwrap()
}
fn float(rate: u32, channels: usize, selected: &[M], pcm: &[f64]) -> Track {
    scan(format(rate, channels, Encoding::Float), selected, pcm)
}
fn full() -> NoiseView {
    NoiseView {
        weighting: Weighting::Unweighted,
        band: Band::Full,
    }
}
fn scalar(x: Outcome<Value>) -> f64 {
    match x {
        Outcome::Value(Value::Scalar(v)) => v,
        other => panic!("expected scalar, got {other:?}"),
    }
}
fn count(x: Outcome<Value>) -> u64 {
    match x {
        Outcome::Value(Value::Count(v)) => v,
        other => panic!("expected count, got {other:?}"),
    }
}
fn curve(x: Outcome<Value>) -> Vec<Point> {
    match x {
        Outcome::Value(Value::Curve(v)) => v,
        other => panic!("expected curve, got {other:?}"),
    }
}
fn approximate(x: Outcome<Value>) -> f64 {
    match x {
        Outcome::Value(Value::ApproximateBits(v)) => v,
        other => panic!("expected approximate bits, got {other:?}"),
    }
}
fn near(got: f64, expected: f64, tol: f64) {
    assert!(
        (got - expected).abs() <= tol,
        "got {got}, expected {expected} within {tol}"
    );
}
fn tone(rate: u32, frames: usize, hz: f64, amp: f64) -> Vec<f64> {
    (0..frames)
        .map(|i| amp * (TAU * hz * i as f64 / f64::from(rate)).sin())
        .collect()
}
fn rng(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}
fn white(frames: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    (0..frames)
        .map(|_| ((rng(&mut s) >> 11) as f64 / (1_u64 << 53) as f64 - 0.5) * 0.2)
        .collect()
}
fn stereo_delay(rate: u32, frames: usize, lag: usize) -> Vec<f64> {
    let l = white(frames + lag, 0x7654321);
    let _ = rate;
    (0..frames).flat_map(|i| [l[i + lag], l[i]]).collect()
}
fn stereo_scaled(rate: u32, frames: usize, l: f64, r: f64) -> Vec<f64> {
    tone(rate, frames, 1000.0, 0.25)
        .into_iter()
        .flat_map(|x| [l * x, r * x])
        .collect()
}
fn spectral_fixture(rate: u32, cutoff: f64, repetitions: usize) -> Vec<f64> {
    // Periodic, equal-power occupied bins, independent of the analysis FFT.
    // Zero bins above the cutoff provide a mathematically exact band edge.
    let n = ((rate as usize).div_ceil(6)).next_power_of_two();
    let mut bins = vec![Complex::new(0.0, 0.0); n];
    let mut seed = 0x123456789abcdef;
    for k in 1..n / 2 {
        if k as f64 * f64::from(rate) / n as f64 > cutoff {
            break;
        }
        let phase = TAU * (rng(&mut seed) >> 11) as f64 / (1_u64 << 53) as f64;
        let x = Complex::new(phase.cos(), phase.sin());
        bins[k] = x;
        bins[n - k] = x.conj();
    }
    FftPlanner::<f64>::new()
        .plan_fft_inverse(n)
        .process(&mut bins);
    let peak = bins.iter().map(|x| x.re.abs()).fold(0.0, f64::max);
    let cycle: Vec<_> = bins.into_iter().map(|x| x.re * 0.25 / peak).collect();
    cycle.repeat(repetitions)
}
fn dither_pcm(rate: u32, seconds: usize, shaped: bool) -> Vec<f64> {
    let mut state = 0x2468ace123456789;
    let mut old = 0_i32;
    (0..rate as usize * seconds)
        .map(|_| {
            let x = (rng(&mut state) % 3) as i32 - 1;
            let y = if shaped { x - old } else { x };
            old = x;
            y as f64 / 32768.0
        })
        .collect()
}

#[test]
fn loudness_range_and_ungated_short_term() {
    let rate = 48_000;
    let mut pcm = tone(rate, 8 * rate as usize, 1000.0, 0.05);
    pcm.extend(tone(rate, 8 * rate as usize, 1000.0, 0.5));
    let tr = float(
        rate,
        1,
        &[M::LoudnessRange, M::ShortTermMaximum, M::LoudnessOverTime],
        &pcm,
    );
    near(scalar(tr.measure(M::LoudnessRange, full())), 20.0, 0.15);
    let points = curve(tr.measure(M::LoudnessOverTime, full()));
    assert_eq!(points.len(), 131);
    near(points[0].x, 3.0, 1e-12);
    near(points.last().unwrap().x, 16.0, 1e-12);
    // Independent BS.1770 transfer response for a 1 kHz full-window tone.
    let b = [
        1.5351248595869702,
        -5.761945908580321,
        8.116910049252581,
        -5.08848181111208,
        1.19839281085285,
    ];
    let a = [
        1.0,
        -3.68070674801639,
        5.087045247971131,
        -3.1315463514467305,
        0.7252088884778705,
    ];
    let response = |coeff: &[f64]| {
        coeff
            .iter()
            .enumerate()
            .fold(Complex::new(0.0, 0.0), |s, (i, c)| {
                let p = -TAU * 1000.0 * i as f64 / f64::from(rate);
                s + Complex::new(p.cos(), p.sin()) * *c
            })
    };
    let expected = -0.691 + 10.0 * (0.125_f64 * (response(&b) / response(&a)).norm_sqr()).log10();
    near(
        scalar(tr.measure(M::ShortTermMaximum, full())),
        expected,
        0.015,
    );
    near(points.last().unwrap().y, expected, 0.015);
    let silent = float(
        rate,
        1,
        &[M::LoudnessRange, M::ShortTermMaximum, M::LoudnessOverTime],
        &vec![0.0; 4 * rate as usize],
    );
    assert!(matches!(
        silent.measure(M::LoudnessRange, full()),
        Outcome::CantTell(_)
    ));
    assert_eq!(
        scalar(silent.measure(M::ShortTermMaximum, full())),
        f64::NEG_INFINITY
    );
    let short = float(
        rate,
        1,
        &[M::ShortTermMaximum, M::LoudnessOverTime],
        &[0.1; 16],
    );
    assert!(matches!(
        short.measure(M::ShortTermMaximum, full()),
        Outcome::CantTell(_)
    ));
    assert!(curve(short.measure(M::LoudnessOverTime, full())).is_empty());
}

#[test]
fn limiting_and_flat_extrema() {
    let rate = 48_000;
    let pcm: Vec<_> = [1.0, 0.5, 0.95, 0.1]
        .into_iter()
        .flat_map(|x| vec![x; 2400])
        .collect();
    let tr = float(rate, 1, &[M::Limiting], &pcm);
    near(scalar(tr.measure(M::Limiting, full())), 50.0, 1e-12);
    let tr = float(
        rate,
        1,
        &[M::FlatToppedPeaks],
        &[
            0.0, 0.8, 0.8, 0.8, 0.0, -0.8, -0.8, -0.8, 0.0, 0.3, 0.3, 0.3, 0.0,
        ],
    );
    assert_eq!(count(tr.measure(M::FlatToppedPeaks, full())), 2);
    let tr = float(
        rate,
        1,
        &[M::FlatToppedPeaks],
        &[0.0, 0.3, 0.3, 0.3, 0.0, 0.9, 0.0],
    );
    assert_eq!(count(tr.measure(M::FlatToppedPeaks, full())), 0);
    let tr = float(rate, 1, &[M::Limiting, M::FlatToppedPeaks], &[0.0; 10]);
    assert_eq!(scalar(tr.measure(M::Limiting, full())), 0.0);
    assert_eq!(count(tr.measure(M::FlatToppedPeaks, full())), 0);
}

#[test]
fn exact_bits_padding_affine_grid_float_and_silence() {
    let f = format(48_000, 1, Encoding::Integer(24));
    let make = |step: i32| {
        (-32768..32768)
            .map(|i| (i * step) as f64 / 8388608.0)
            .collect::<Vec<_>>()
    };
    for (step, low) in [(256, 16), (192, 18), (128, 17)] {
        let tr = scan(f.clone(), &[M::BitsInUse], &make(step));
        assert_eq!(
            tr.measure(M::BitsInUse, full()),
            Outcome::Value(Value::Bits(BitDepth::Integer {
                used: 16,
                declared: Some(24),
                low_bit_depth: low,
                observed_lattice_bits: Some(16),
                grid_step: step as u64,
                grid_scale_bits: 23
            }))
        );
    }
    let tr = float(48_000, 1, &[M::BitsInUse], &[-1.0, 1.0 / 32768.0, 0.0]);
    assert!(matches!(
        tr.measure(M::BitsInUse, full()),
        Outcome::Value(Value::Bits(BitDepth::Integer {
            used: 16,
            declared: None,
            ..
        }))
    ));
    let tr = float(48_000, 1, &[M::BitsInUse], &[0.1, 0.1234567]);
    assert_eq!(
        tr.measure(M::BitsInUse, full()),
        Outcome::Value(Value::Bits(BitDepth::NonIntegerGrid))
    );
    let tr = scan(f, &[M::BitsInUse], &[0.0; 100]);
    assert!(matches!(
        tr.measure(M::BitsInUse, full()),
        Outcome::Value(Value::Bits(BitDepth::Integer {
            used: 0,
            grid_step: 0,
            ..
        }))
    ));
}

#[test]
fn spectral_cutoff_and_codec_pattern() {
    let rate = 96_000;
    let tr = float(
        rate,
        1,
        &[M::LikelySourceRate, M::Bandwidth],
        &spectral_fixture(rate, 22_050.0, 8),
    );
    assert_eq!(
        tr.measure(M::LikelySourceRate, full()),
        Outcome::Value(Value::SourceRate(Some(44_100)))
    );
    near(scalar(tr.measure(M::Bandwidth, full())), 22.05, 0.05);
    let tr = float(
        rate,
        1,
        &[M::LossyCodingPattern],
        &spectral_fixture(rate, 18_000.0, 8),
    );
    assert_eq!(
        tr.measure(M::LossyCodingPattern, full()),
        Outcome::Value(Value::Pattern { found: 1, total: 1 })
    );
    let tr = float(
        rate,
        1,
        &[M::LossyCodingPattern, M::LikelySourceRate],
        &spectral_fixture(rate, 47_900.0, 8),
    );
    assert_eq!(
        tr.measure(M::LossyCodingPattern, full()),
        Outcome::Value(Value::Pattern { found: 0, total: 1 })
    );
    assert!(matches!(
        tr.measure(M::LikelySourceRate, full()),
        Outcome::CantTell(_)
    ));
    let tr = float(
        rate,
        1,
        &[M::LikelySourceRate],
        &tone(rate, rate as usize, 1000.0, 0.2),
    );
    assert!(matches!(
        tr.measure(M::LikelySourceRate, full()),
        Outcome::CantTell(_)
    ));
}

#[test]
fn spectrum_power_bandwidth_and_above_twenty() {
    let rate = 96_000;
    let n = 16_384;
    let df = f64::from(rate) / n as f64;
    let lo = tone(rate, 4 * n, 171.0 * df, 0.1);
    let hi = tone(rate, 4 * n, 5120.0 * df, 0.1);
    let pcm: Vec<_> = lo.iter().zip(&hi).map(|(a, b)| a + b).collect();
    let tr = float(
        rate,
        1,
        &[M::Bandwidth, M::Above20Khz, M::AverageSpectrum],
        &pcm,
    );
    near(
        scalar(tr.measure(M::Above20Khz, full())),
        -10.0 * 2.0_f64.log10(),
        0.002,
    );
    near(scalar(tr.measure(M::Bandwidth, full())), 30.0, 0.04);
    let points = curve(tr.measure(M::AverageSpectrum, full()));
    near(
        points.iter().map(|p| 10.0_f64.powf(p.y / 10.0) * df).sum(),
        0.01,
        1e-12,
    );
    assert_eq!(points[0].x, 0.0);
    assert_eq!(points.last().unwrap().x, 48_000.0);
    let low = float(44_100, 1, &[M::Above20Khz], &[0.0; 100]);
    assert_eq!(low.measure(M::Above20Khz, full()), Outcome::NotApplicable);
}

#[test]
fn quiet_floor_range_bits_spectrum_tone_and_bands() {
    let rate = 48_000;
    let loud: Vec<_> = (0..rate as usize)
        .map(|i| 0.1 * (TAU * 1000.0 * i as f64 / f64::from(rate)).cos())
        .collect();
    let mut pcm = loud.clone();
    pcm.extend(tone(rate, 2 * rate as usize, 50.0, 1e-5));
    pcm.extend(loud);
    let sel = [
        M::NoiseFloor,
        M::DynamicRange,
        M::EffectiveBits,
        M::NoiseSpectrum,
        M::HumAndIdleTones,
        M::NoiseByBand,
    ];
    let tr = float(rate, 1, &sel, &pcm);
    near(tr.quiet_seconds().unwrap(), 2.0, 1e-12);
    let expected = 10.0 * 5e-11_f64.log10();
    near(scalar(tr.measure(M::NoiseFloor, full())), expected, 1e-10);
    near(
        scalar(tr.measure(M::DynamicRange, full())),
        -20.0 - expected,
        1e-10,
    );
    near(
        approximate(tr.measure(M::EffectiveBits, full())),
        -expected / (20.0 * 2.0_f64.log10()),
        1e-10,
    );
    match tr.measure(M::HumAndIdleTones, full()) {
        Outcome::Value(Value::Tone(Some(t))) => {
            near(t.frequency_hz, 50.0, 2.0);
            near(t.level_dbfs, expected, 0.4);
        }
        x => panic!("{x:?}"),
    }
    let points = curve(tr.measure(M::NoiseSpectrum, full()));
    let df = points[1].x - points[0].x;
    near(
        points.iter().map(|p| 10.0_f64.powf(p.y / 10.0) * df).sum(),
        5e-11,
        1e-19,
    );
    match tr.measure(M::NoiseByBand, full()) {
        Outcome::Value(Value::Bands(b)) => {
            if let Outcome::Value(v) = &b[0] {
                near(*v, expected, 0.03);
            } else {
                panic!("{:?}", b[0]);
            }
        }
        x => panic!("{x:?}"),
    }
    let loud = float(rate, 1, &sel, &tone(rate, rate as usize, 1000.0, 0.1));
    for m in sel {
        assert!(
            matches!(loud.measure(m, full()), Outcome::CantTell(_)),
            "{m:?}"
        );
    }
    let too_short = float(rate, 1, &sel, &vec![0.0; rate as usize - 1]);
    for m in sel {
        assert!(
            matches!(too_short.measure(m, full()), Outcome::CantTell(_)),
            "{m:?}"
        );
    }
    let silent = float(rate, 1, &sel, &vec![0.0; rate as usize]);
    assert_eq!(
        scalar(silent.measure(M::NoiseFloor, full())),
        f64::NEG_INFINITY
    );
    assert!(matches!(
        silent.measure(M::EffectiveBits, full()),
        Outcome::CantTell(_)
    ));
    assert!(matches!(
        silent.measure(M::DynamicRange, full()),
        Outcome::CantTell(_)
    ));
    assert_eq!(
        silent.measure(M::HumAndIdleTones, full()),
        Outcome::Value(Value::Tone(None))
    );
}

#[test]
fn cached_noise_reweight_reband_and_five_bands() {
    let rate = 96_000;
    let n = 16_384;
    let df = f64::from(rate) / n as f64;
    let frequencies = [17.0 * df, 171.0 * df, 853.0 * df, 2560.0 * df, 5120.0 * df];
    let pcm: Vec<_> = (0..n * 12)
        .map(|i| {
            frequencies
                .iter()
                .map(|f| 1e-5 * (TAU * f * i as f64 / f64::from(rate)).sin())
                .sum::<f64>()
        })
        .collect();
    let sel = [
        M::NoiseFloor,
        M::DynamicRange,
        M::EffectiveBits,
        M::NoiseSpectrum,
        M::HumAndIdleTones,
        M::NoiseByBand,
        M::Dither,
        M::ClicksPerMinute,
    ];
    let tr = float(rate, 1, &sel, &pcm);
    let record = tr.noise_record();
    let snapshot = NoiseSnapshot::from_record(record.clone()).unwrap();
    drop(pcm);
    for weighting in [Weighting::Unweighted, Weighting::A, Weighting::ItuR468] {
        for band in [Band::Full, Band::Audible, Band::ToNyquist] {
            let view = NoiseView { weighting, band };
            for m in sel {
                assert_eq!(
                    snapshot.measure(m, view),
                    tr.measure(m, view),
                    "{m:?} {view:?}"
                );
            }
        }
    }
    let unweighted = scalar(snapshot.measure(M::NoiseFloor, full()));
    near(unweighted, 10.0 * 2.5e-10_f64.log10(), 1e-9);
    let audible = scalar(snapshot.measure(
        M::NoiseFloor,
        NoiseView {
            weighting: Weighting::Unweighted,
            band: Band::Audible,
        },
    ));
    near(audible, 10.0 * 2e-10_f64.log10(), 0.002);
    assert!(
        (scalar(snapshot.measure(
            M::NoiseFloor,
            NoiseView {
                weighting: Weighting::A,
                band: Band::Full
            }
        )) - unweighted)
            .abs()
            > 0.5
    );
    assert!(
        (scalar(snapshot.measure(
            M::NoiseFloor,
            NoiseView {
                weighting: Weighting::ItuR468,
                band: Band::Full
            }
        )) - unweighted)
            .abs()
            > 0.5
    );
    match snapshot.measure(M::NoiseByBand, full()) {
        Outcome::Value(Value::Bands(b)) => {
            for x in b {
                match x {
                    Outcome::Value(v) => near(v, 10.0 * 5e-11_f64.log10(), 0.005),
                    x => panic!("{x:?}"),
                }
            }
        }
        x => panic!("{x:?}"),
    }
    match snapshot.measure(M::NoiseByBand, NoiseView::default()) {
        Outcome::Value(Value::Bands(b)) => assert_eq!(b[4], Outcome::NotApplicable),
        x => panic!("{x:?}"),
    }
    let mut invalid = record.clone();
    invalid.version = 1;
    assert!(NoiseSnapshot::from_record(invalid).is_err());
    let mut invalid = record.clone();
    invalid.quiet.as_mut().unwrap().bin_power_sums[0] = f64::NAN;
    assert!(NoiseSnapshot::from_record(invalid).is_err());
    let mut invalid = record;
    invalid.quiet.as_mut().unwrap().bin_power_sums.pop();
    assert!(NoiseSnapshot::from_record(invalid).is_err());
}

#[test]
fn clicks_and_dither_evidence() {
    let rate = 48_000;
    let mut pcm = vec![0.0; rate as usize];
    pcm[12_000] = 0.5;
    pcm[24_000] = -0.5;
    let tr = float(rate, 1, &[M::ClicksPerMinute], &pcm);
    near(scalar(tr.measure(M::ClicksPerMinute, full())), 120.0, 1e-12);
    let tr = float(
        rate,
        1,
        &[M::ClicksPerMinute],
        &tone(rate, rate as usize, 1000.0, 0.5),
    );
    near(scalar(tr.measure(M::ClicksPerMinute, full())), 0.0, 1e-12);
    let f = format(rate, 1, Encoding::Integer(16));
    let tr = scan(f.clone(), &[M::Dither], &vec![0.0; rate as usize]);
    assert_eq!(
        tr.measure(M::Dither, full()),
        Outcome::Value(Value::Dither(Dither::None))
    );
    for (shaped, expected) in [(false, Dither::Flat), (true, Dither::Shaped)] {
        let tr = scan(f.clone(), &[M::Dither], &dither_pcm(rate, 4, shaped));
        assert_eq!(
            tr.measure(M::Dither, full()),
            Outcome::Value(Value::Dither(expected))
        );
    }
    let hiss: Vec<_> = dither_pcm(rate, 2, false)
        .into_iter()
        .map(|x| x / 256.0 * 100.0)
        .collect();
    let tr = scan(format(rate, 1, Encoding::Integer(24)), &[M::Dither], &hiss);
    assert!(matches!(
        tr.measure(M::Dither, full()),
        Outcome::CantTell(_)
    ));
    let tr = scan(f, &[M::Dither], &vec![0.5; rate as usize]);
    assert!(matches!(
        tr.measure(M::Dither, full()),
        Outcome::CantTell(_)
    ));
}

#[test]
fn stereo_relationships_delay_and_bass_width() {
    let rate = 48_000;
    let sel = [M::Correlation, M::Balance, M::Width, M::MonoCompatibility];
    let tr = float(rate, 2, &sel, &stereo_scaled(rate, rate as usize, 2.0, 1.0));
    near(scalar(tr.measure(M::Correlation, full())), 1.0, 1e-12);
    near(
        scalar(tr.measure(M::Balance, full())),
        10.0 * 4.0_f64.log10(),
        1e-10,
    );
    near(
        scalar(tr.measure(M::Width, full())),
        10.0 * (1.0 / 9.0_f64).log10(),
        1e-10,
    );
    near(
        scalar(tr.measure(M::MonoCompatibility, full())),
        10.0 * (10.0 / 9.0_f64).log10(),
        1e-10,
    );
    let tr = float(
        rate,
        2,
        &sel,
        &stereo_scaled(rate, rate as usize, 1.0, -1.0),
    );
    near(scalar(tr.measure(M::Correlation, full())), -1.0, 1e-12);
    assert_eq!(
        scalar(tr.measure(M::MonoCompatibility, full())),
        f64::INFINITY
    );
    let tr = float(
        rate,
        2,
        &[M::InterChannelDelay],
        &stereo_delay(rate, rate as usize, 3),
    );
    near(scalar(tr.measure(M::InterChannelDelay, full())), 62.5, 2.0);
    let l = white(rate as usize, 0x123456789);
    let r = white(rate as usize, 0x987654321);
    let unrelated: Vec<_> = l.iter().zip(r).flat_map(|(&a, b)| [a, b]).collect();
    let tr = float(rate, 2, &[M::InterChannelDelay], &unrelated);
    assert!(matches!(
        tr.measure(M::InterChannelDelay, full()),
        Outcome::CantTell(_)
    ));
    let tr = float(
        rate,
        2,
        &[M::InterChannelDelay],
        &stereo_scaled(rate, rate as usize, 1.0, 1.0),
    );
    assert!(matches!(
        tr.measure(M::InterChannelDelay, full()),
        Outcome::CantTell(_)
    ));
    let df = f64::from(rate) / 8192.0;
    let low = tone(rate, 8192 * 6, 17.0 * df, 0.1);
    let high = tone(rate, 8192 * 6, 45.0 * df, 0.1);
    let stereo: Vec<_> = low
        .iter()
        .zip(high)
        .flat_map(|(a, b)| [*a + b, *a - b])
        .collect();
    let tr = float(rate, 2, &[M::BassWidth], &stereo);
    match tr.measure(M::BassWidth, full()) {
        Outcome::Value(Value::BassWidth(BassWidth::MonoBelow(f))) => near(f, 250.0, 25.0),
        x => panic!("{x:?}"),
    }
    let stereo: Vec<_> = low.iter().flat_map(|x| [*x, -*x]).collect();
    let tr = float(rate, 2, &[M::BassWidth], &stereo);
    assert_eq!(
        tr.measure(M::BassWidth, full()),
        Outcome::Value(Value::BassWidth(BassWidth::Full))
    );
    let mono = float(
        rate,
        1,
        &[
            M::Correlation,
            M::Balance,
            M::Width,
            M::MonoCompatibility,
            M::InterChannelDelay,
            M::BassWidth,
        ],
        &[0.0; 100],
    );
    for m in [
        M::Correlation,
        M::Balance,
        M::Width,
        M::MonoCompatibility,
        M::InterChannelDelay,
        M::BassWidth,
    ] {
        assert_eq!(mono.measure(m, full()), Outcome::NotApplicable);
    }
}

#[test]
fn continuous_album_loudness_and_ordered_curves() {
    let rate = 48_000;
    let sel = [M::LoudnessRange, M::ShortTermMaximum, M::LoudnessOverTime];
    let a = tone(rate, 4 * rate as usize, 1000.0, 0.05);
    let b = tone(rate, 5 * rate as usize, 1000.0, 0.5);
    let f = format(rate, 1, Encoding::Float);
    let mut ab = AlbumAnalyzer::new(Measurements::new(&sel), Settings::default());
    for pcm in [&a, &b] {
        ab.begin_track(f.clone()).unwrap();
        for c in pcm.chunks(701) {
            ab.push_interleaved(c).unwrap();
        }
        ab.end_track().unwrap();
    }
    let album = ab.finish().unwrap();
    assert_eq!(album.loudness_method(), AlbumLoudnessMethod::Continuous);
    let mut both = a.clone();
    both.extend(&b);
    let direct = float(rate, 1, &[M::LoudnessRange], &both);
    assert_eq!(
        album.measure(M::LoudnessRange, full()),
        direct.measure(M::LoudnessRange, full())
    );
    let max = album
        .tracks()
        .iter()
        .map(|t| scalar(t.measure(M::ShortTermMaximum, full())))
        .fold(f64::NEG_INFINITY, f64::max);
    near(
        scalar(album.measure(M::ShortTermMaximum, full())),
        max,
        1e-12,
    );
    match album.measure(M::LoudnessOverTime, full()) {
        Outcome::Value(Value::Curves(v)) => {
            assert_eq!(v.len(), 2);
            assert_eq!(v[0].len(), 11);
            assert_eq!(v[1].len(), 21);
        }
        x => panic!("{x:?}"),
    }
    let cached = Album::from_tracks(vec![scan(f.clone(), &sel, &a), scan(f, &sel, &b)]).unwrap();
    assert_eq!(cached.loudness_method(), AlbumLoudnessMethod::Unavailable);
    assert!(matches!(
        cached.measure(M::LoudnessRange, full()),
        Outcome::CantTell(_)
    ));
    near(
        scalar(cached.measure(M::ShortTermMaximum, full())),
        max,
        1e-12,
    );
    let mut mixed = AlbumAnalyzer::new(Measurements::new(&[M::LoudnessRange]), Settings::default());
    for r in [48_000, 44_100] {
        mixed.begin_track(format(r, 1, Encoding::Float)).unwrap();
        mixed
            .push_interleaved(&tone(r, r as usize * 4, 1000.0, 0.1))
            .unwrap();
        mixed.end_track().unwrap();
    }
    let mixed = mixed.finish().unwrap();
    assert_eq!(mixed.loudness_method(), AlbumLoudnessMethod::Unavailable);
    assert!(matches!(
        mixed.measure(M::LoudnessRange, full()),
        Outcome::CantTell(_)
    ));
}

#[test]
fn album_dynamics_bits_source_pattern_spectrum() {
    let rate = 48_000;
    let sel = [M::Limiting, M::FlatToppedPeaks];
    let a: Vec<_> = [1.0, 0.5, 0.95, 0.1]
        .into_iter()
        .flat_map(|x| vec![x; 2400])
        .collect();
    let b: Vec<_> = [0.85, 0.9]
        .into_iter()
        .flat_map(|x| vec![x; 2400])
        .collect();
    let album =
        Album::from_tracks(vec![float(rate, 1, &sel, &a), float(rate, 1, &sel, &b)]).unwrap();
    near(
        scalar(album.measure(M::Limiting, full())),
        (0.2 * 50.0 + 0.1 * 100.0) / 0.3,
        1e-9,
    );
    let flats_a = [0.0, 0.8, 0.8, 0.8, 0.0];
    let flats_b = [0.0, -0.8, -0.8, -0.8, 0.0, 0.8, 0.8, 0.8, 0.0];
    let album = Album::from_tracks(vec![
        float(rate, 1, &[M::FlatToppedPeaks], &flats_a),
        float(rate, 1, &[M::FlatToppedPeaks], &flats_b),
    ])
    .unwrap();
    assert_eq!(count(album.measure(M::FlatToppedPeaks, full())), 3);
    let album = Album::from_tracks(vec![
        scan(
            format(rate, 1, Encoding::Integer(16)),
            &[M::BitsInUse],
            &[-1.0, 1.0 / 32768.0],
        ),
        scan(
            format(rate, 1, Encoding::Integer(24)),
            &[M::BitsInUse],
            &[-1.0, 1.0 / 8388608.0],
        ),
    ])
    .unwrap();
    assert!(matches!(
        album.measure(M::BitsInUse, full()),
        Outcome::Value(Value::Bits(BitDepth::Integer { used: 24, .. }))
    ));
    let rate = 96_000;
    let sel = [M::LikelySourceRate, M::LossyCodingPattern, M::Bandwidth];
    let a = spectral_fixture(rate, 22_050.0, 4);
    let b = spectral_fixture(rate, 24_000.0, 4);
    let c = spectral_fixture(rate, 18_000.0, 4);
    let album = Album::from_tracks(vec![
        float(rate, 1, &sel, &a),
        float(rate, 1, &sel, &b),
        float(rate, 1, &sel, &a),
        float(rate, 1, &sel, &c),
    ])
    .unwrap();
    assert_eq!(
        album.measure(M::LikelySourceRate, full()),
        Outcome::Value(Value::SourceRate(Some(44_100)))
    );
    assert_eq!(
        album.measure(M::LossyCodingPattern, full()),
        Outcome::Value(Value::Pattern { found: 1, total: 4 })
    );
    near(scalar(album.measure(M::Bandwidth, full())), 18.0, 0.05);
    let tie = Album::from_tracks(vec![
        float(rate, 1, &[M::LikelySourceRate], &a),
        float(rate, 1, &[M::LikelySourceRate], &b),
    ])
    .unwrap();
    assert_eq!(
        tie.measure(M::LikelySourceRate, full()),
        Outcome::Value(Value::SourceRate(None))
    );
    let n = 16_384;
    let df = f64::from(rate) / n as f64;
    let lo = tone(rate, 4 * n, 171.0 * df, 0.1);
    let hi = tone(rate, 4 * n, 5120.0 * df, 0.1);
    let a: Vec<_> = lo.iter().zip(&hi).map(|(a, b)| a + b).collect();
    let b: Vec<_> = hi.iter().map(|x| 2.0 * x).collect();
    let sel = [M::Above20Khz, M::AverageSpectrum];
    let album =
        Album::from_tracks(vec![float(rate, 1, &sel, &a), float(rate, 1, &sel, &b)]).unwrap();
    near(
        scalar(album.measure(M::Above20Khz, full())),
        10.0 * (0.025 / 0.03_f64).log10(),
        0.002,
    );
    let p = curve(album.measure(M::AverageSpectrum, full()));
    near(
        p.iter().map(|p| 10.0_f64.powf(p.y / 10.0) * df).sum(),
        0.015,
        1e-12,
    );
}

#[test]
fn album_pooled_noise_dynamic_range_hum_bands_dither_and_clicks() {
    let rate = 48_000;
    let sel = [
        M::NoiseFloor,
        M::DynamicRange,
        M::EffectiveBits,
        M::NoiseSpectrum,
        M::HumAndIdleTones,
        M::NoiseByBand,
    ];
    let mut a = tone(rate, rate as usize, 50.0, 1e-5);
    a.extend((0..rate as usize).map(|i| 0.1 * (TAU * 1000.0 * i as f64 / f64::from(rate)).cos()));
    let mut b = tone(rate, 2 * rate as usize, 100.0, 2e-5);
    b.extend((0..rate as usize).map(|i| 0.2 * (TAU * 1000.0 * i as f64 / f64::from(rate)).cos()));
    let a = float(rate, 1, &sel, &a);
    let b = float(rate, 1, &sel, &b);
    let saved = [
        NoiseSnapshot::from_record(a.noise_record()).unwrap(),
        NoiseSnapshot::from_record(b.noise_record()).unwrap(),
    ];
    let album = Album::from_tracks(vec![a, b]).unwrap();
    let expected = 10.0 * 1.5e-10_f64.log10();
    near(
        scalar(album.measure(M::NoiseFloor, full())),
        expected,
        1e-10,
    );
    near(
        scalar(album.measure(M::DynamicRange, full())),
        20.0 * 0.2_f64.log10() - expected,
        1e-10,
    );
    near(
        approximate(album.measure(M::EffectiveBits, full())),
        -expected / (20.0 * 2.0_f64.log10()),
        1e-10,
    );
    match album.measure(M::HumAndIdleTones, full()) {
        Outcome::Value(Value::Tone(Some(t))) => {
            near(t.frequency_hz, 100.0, 2.0);
            near(t.level_dbfs, 10.0 * 2e-10_f64.log10(), 0.4);
        }
        x => panic!("{x:?}"),
    }
    match album.measure(M::NoiseByBand, full()) {
        Outcome::Value(Value::Bands(b)) => match &b[0] {
            Outcome::Value(v) => near(*v, expected, 0.05),
            x => panic!("{x:?}"),
        },
        x => panic!("{x:?}"),
    }
    for m in sel {
        assert_eq!(
            album.measure(m, full()),
            NoiseSnapshot::measure_album(&saved, m, full()).unwrap()
        );
    }
    let p = curve(album.measure(M::NoiseSpectrum, full()));
    let df = p[1].x - p[0].x;
    near(
        p.iter().map(|p| 10.0_f64.powf(p.y / 10.0) * df).sum(),
        1.5e-10,
        1e-18,
    );
    let f = format(rate, 1, Encoding::Integer(16));
    let zero = vec![0.0; rate as usize];
    let flat = dither_pcm(rate, 4, false);
    let tied = Album::from_tracks(vec![
        scan(f.clone(), &[M::Dither], &zero),
        scan(f.clone(), &[M::Dither], &flat),
    ])
    .unwrap();
    assert_eq!(
        tied.measure(M::Dither, full()),
        Outcome::Value(Value::Dither(Dither::Mixed))
    );
    let mode = Album::from_tracks(vec![
        scan(f.clone(), &[M::Dither], &zero),
        scan(f.clone(), &[M::Dither], &flat),
        scan(f, &[M::Dither], &flat),
    ])
    .unwrap();
    assert_eq!(
        mode.measure(M::Dither, full()),
        Outcome::Value(Value::Dither(Dither::Flat))
    );
    let mut a = vec![0.0; rate as usize];
    a[1000] = 0.5;
    let mut b = vec![0.0; 2 * rate as usize];
    b[2000] = 0.5;
    b[4000] = 0.5;
    let album = Album::from_tracks(vec![
        float(rate, 1, &[M::ClicksPerMinute], &a),
        float(rate, 1, &[M::ClicksPerMinute], &b),
    ])
    .unwrap();
    near(
        scalar(album.measure(M::ClicksPerMinute, full())),
        60.0,
        1e-12,
    );
}

#[test]
fn album_stereo_rules() {
    let rate = 48_000;
    let sel = [M::Correlation, M::Balance, M::Width, M::MonoCompatibility];
    let a = float(rate, 2, &sel, &stereo_scaled(rate, rate as usize, 2.0, 1.0));
    let b = float(
        rate,
        2,
        &sel,
        &stereo_scaled(rate, 2 * rate as usize, 1.0, -0.5),
    );
    let aw = scalar(a.measure(M::Width, full()));
    let bw = scalar(b.measure(M::Width, full()));
    let worst = scalar(a.measure(M::MonoCompatibility, full()))
        .max(scalar(b.measure(M::MonoCompatibility, full())));
    let album = Album::from_tracks(vec![a, b]).unwrap();
    near(
        scalar(album.measure(M::Correlation, full())),
        -1.0 / 3.0,
        1e-10,
    );
    near(
        scalar(album.measure(M::Balance, full())),
        10.0 * (6.0 / 1.5_f64).log10(),
        1e-10,
    );
    near(
        scalar(album.measure(M::Width, full())),
        (aw + 2.0 * bw) / 3.0,
        1e-10,
    );
    near(
        scalar(album.measure(M::MonoCompatibility, full())),
        worst,
        1e-10,
    );
    let tracks = [1, 3, 7].map(|lag| {
        float(
            rate,
            2,
            &[M::InterChannelDelay],
            &stereo_delay(rate, rate as usize, lag),
        )
    });
    let album = Album::from_tracks(Vec::from(tracks)).unwrap();
    near(
        scalar(album.measure(M::InterChannelDelay, full())),
        62.5,
        2.0,
    );
    let n = 8192;
    let df = f64::from(rate) / n as f64;
    let low = tone(rate, 6 * n, 17.0 * df, 0.1);
    let tracks = [45.0, 62.0, 79.0].map(|k| {
        let high = tone(rate, 6 * n, k * df, 0.1);
        let pcm: Vec<_> = low
            .iter()
            .zip(high)
            .flat_map(|(a, b)| [*a + b, *a - b])
            .collect();
        float(rate, 2, &[M::BassWidth], &pcm)
    });
    let middle = tracks[1].measure(M::BassWidth, full());
    let album = Album::from_tracks(Vec::from(tracks)).unwrap();
    assert_eq!(album.measure(M::BassWidth, full()), middle);
}

#[test]
fn validation_selection_and_chunk_invariance() {
    let f = format(48_000, 2, Encoding::Integer(16));
    let mut a = Analyzer::new(
        f.clone(),
        Measurements::new(&[M::BitsInUse]),
        Settings::default(),
    )
    .unwrap();
    assert!(matches!(
        a.push_interleaved(&[0.0]),
        Err(Error::IncompleteFrame)
    ));
    assert!(matches!(
        a.push_interleaved(&[0.0, f64::NAN]),
        Err(Error::InvalidSample(1))
    ));
    assert!(matches!(
        a.push_interleaved(&[0.0, 0.1]),
        Err(Error::InvalidSample(1))
    ));
    a.push_interleaved(&[-1.0, 1.0 / 32768.0]).unwrap();
    let tr = a.finish().unwrap();
    assert_eq!(tr.frames(), 1);
    assert_eq!(tr.measure(M::NoiseFloor, full()), Outcome::NotRequested);
    let f = format(48_000, 2, Encoding::Float);
    let mut pcm = stereo_delay(48_000, 3 * 48_000, 3);
    pcm.extend(vec![0.0; 2 * 48_000 * 2]);
    let mut a = Analyzer::new(f.clone(), Measurements::all(), Settings::default()).unwrap();
    a.push_interleaved(&pcm).unwrap();
    let whole = a.finish().unwrap();
    let mut b = Analyzer::new(f, Measurements::all(), Settings::default()).unwrap();
    for c in pcm.chunks(113 * 2) {
        b.push_interleaved(c).unwrap();
        b.push_interleaved(&[]).unwrap();
    }
    let split = b.finish().unwrap();
    for m in M::ALL {
        assert_eq!(whole.measure(m, full()), split.measure(m, full()), "{m:?}");
    }
    assert!(matches!(Album::from_tracks(Vec::new()), Err(Error::Empty)));
    let a = float(48_000, 1, &[M::BitsInUse], &[0.0; 5]);
    let b = float(48_000, 1, &[M::Correlation], &[0.0; 5]);
    assert!(matches!(
        Album::from_tracks(vec![a, b]),
        Err(Error::Album(_))
    ));
}

// Corrective-round regressions. These exercise the public API, not a second
// implementation of the production overlap accumulator.
fn sum_curve_power(points: &[Point]) -> f64 {
    let df = points[1].x - points[0].x;
    points.iter().map(|p| 10.0_f64.powf(p.y / 10.0) * df).sum()
}
fn fixture_weight(f: f64, weighting: Weighting) -> f64 {
    // Independent analog-response evaluation at the fixture's tone frequencies.
    match weighting {
        Weighting::Unweighted => 1.0,
        Weighting::A => {
            let f2 = f * f;
            let a = 12194.0_f64.powi(2) * f2 * f2
                / ((f2 + 20.6_f64.powi(2))
                    * (f2 + 107.7_f64.powi(2)).sqrt()
                    * (f2 + 737.9_f64.powi(2)).sqrt()
                    * (f2 + 12194.0_f64.powi(2)));
            a * a * 10.0_f64.powf(0.2)
        }
        Weighting::ItuR468 => {
            let h1 = 1.0 - 1.363894795463638e-7 * f.powi(2) + 2.043828333606125e-15 * f.powi(4)
                - 4.737338981378384e-24 * f.powi(6);
            let h2 = 5.559488023498642e-4 * f - 2.118150887518656e-11 * f.powi(3)
                + 1.306612257412824e-19 * f.powi(5);
            10.0_f64.powf(1.822) * (1.246332637532143e-4 * f).powi(2) / (h1 * h1 + h2 * h2)
        }
    }
}
#[test]
fn audit_spectral_burst_is_position_independent_including_stream_edges() {
    let rate = 96_000;
    let n = 16_384;
    let frames = 8 * n;
    let burst_len = 2048;
    let low_hz = 171.0 * f64::from(rate) / n as f64;
    let low = tone(rate, frames, low_hz, 0.1);
    let burst: Vec<_> = tone(rate, burst_len, 30_000.0, 0.079)
        .into_iter()
        .enumerate()
        .map(|(i, x)| {
            x * (TAU * 0.5 * (i as f64 + 0.5) / burst_len as f64)
                .sin()
                .powi(2)
        })
        .collect();
    let low_power = low.iter().map(|x| x * x).sum::<f64>() / frames as f64;
    let high_power = burst.iter().map(|x| x * x).sum::<f64>() / frames as f64;
    let mut measured = Vec::new();
    for start in [
        0,
        1,
        n / 2 - burst_len / 2,
        n - burst_len / 2,
        frames - burst_len,
    ] {
        let mut pcm = low.clone();
        for (x, y) in pcm[start..start + burst_len].iter_mut().zip(&burst) {
            *x += y;
        }
        let raw = pcm.iter().map(|x| x * x).sum::<f64>() / frames as f64;
        let t = float(rate, 1, &[M::Above20Khz, M::AverageSpectrum], &pcm);
        let ratio = scalar(t.measure(M::Above20Khz, full()));
        near(ratio, 10.0 * (high_power / raw).log10(), 0.5);
        measured.push(ratio);
        near(
            sum_curve_power(&curve(t.measure(M::AverageSpectrum, full()))),
            raw,
            1e-12,
        );
        let quiet: Vec<_> = pcm.iter().map(|x| x * 1e-4).collect();
        let selected = [
            M::NoiseFloor,
            M::DynamicRange,
            M::EffectiveBits,
            M::NoiseSpectrum,
            M::NoiseByBand,
            M::HumAndIdleTones,
        ];
        let t = float(rate, 1, &selected, &quiet);
        assert_eq!(t.quiet_seconds(), Some(frames as f64 / f64::from(rate)));
        let saved = NoiseSnapshot::from_record(t.noise_record()).unwrap();
        for weighting in [Weighting::Unweighted, Weighting::A, Weighting::ItuR468] {
            for band in [Band::Full, Band::Audible, Band::ToNyquist] {
                let view = NoiseView { weighting, band };
                let expected = (low_power * fixture_weight(low_hz, weighting)
                    + if band == Band::Audible {
                        0.0
                    } else {
                        high_power * fixture_weight(30_000.0, weighting)
                    })
                    * 1e-8;
                near(
                    scalar(saved.measure(M::NoiseFloor, view)),
                    10.0 * expected.log10(),
                    0.5,
                );
                for m in selected {
                    assert_eq!(saved.measure(m, view), t.measure(m, view));
                }
                if band != Band::Audible {
                    let expected =
                        10.0 * (high_power * fixture_weight(30_000.0, weighting) * 1e-8).log10();
                    match saved.measure(M::NoiseByBand, view) {
                        Outcome::Value(Value::Bands(b)) => match &b[4] {
                            Outcome::Value(v) => near(*v, expected, 0.5),
                            x => panic!("{x:?}"),
                        },
                        x => panic!("{x:?}"),
                    }
                }
            }
        }
    }
    assert!(
        measured.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            - measured.iter().copied().fold(f64::INFINITY, f64::min)
            < 0.5
    );
}

#[test]
fn audit_overlap_partial_hops_and_multichannel_parseval() {
    let rate = 48_000;
    let n = 8192;
    let sel = [M::AverageSpectrum, M::Above20Khz];
    for frames in [
        1,
        2,
        n / 2 - 1,
        n / 2,
        n / 2 + 1,
        n - 1,
        n,
        n + 1,
        3 * n + 177,
    ] {
        let pcm = white(frames * 3, 0x237462761);
        let expected = pcm.iter().map(|x| x * x).sum::<f64>() / pcm.len() as f64;
        let f = format(rate, 3, Encoding::Float);
        let split = scan(f.clone(), &sel, &pcm);
        let mut a = Analyzer::new(f, Measurements::new(&sel), Settings::default()).unwrap();
        a.push_interleaved(&pcm).unwrap();
        let whole = a.finish().unwrap();
        assert_eq!(whole.frames(), frames as u64);
        for m in sel {
            assert_eq!(whole.measure(m, full()), split.measure(m, full()));
        }
        near(
            sum_curve_power(&curve(whole.measure(M::AverageSpectrum, full()))),
            expected,
            1e-12,
        );
    }
}

#[test]
fn audit_quiet_transition_duration_and_noise_records() {
    let rate = 48_000;
    let block = 2400;
    let sel = [
        M::NoiseFloor,
        M::DynamicRange,
        M::EffectiveBits,
        M::NoiseSpectrum,
        M::HumAndIdleTones,
        M::NoiseByBand,
    ];
    // Both loudly classified mixed blocks and RMS-quiet blocks with a short
    // loud edge must be refined. A constant loud neighbor makes the exact
    // transition observable; there are no ambiguous sine zero crossings.
    for loud in [0.1, 0.02] {
        for offset in [0, 1, 97, 1200, 2399] {
            for frames in [48_000, 48_960] {
                let quiet = tone(rate, frames, 50.0, 1e-5);
                let mut pcm = vec![loud; block + offset];
                pcm.extend(&quiet);
                pcm.extend(vec![loud; 2 * block]);
                let t = float(rate, 1, &sel, &pcm);
                near(
                    t.quiet_seconds().unwrap(),
                    frames as f64 / f64::from(rate),
                    1e-12,
                );
                let expected = 10.0 * 5e-11_f64.log10();
                near(scalar(t.measure(M::NoiseFloor, full())), expected, 1e-9);
                near(
                    scalar(t.measure(M::DynamicRange, full())),
                    20.0 * loud.log10() - expected,
                    1e-9,
                );
                near(
                    approximate(t.measure(M::EffectiveBits, full())),
                    -expected / (20.0 * 2.0_f64.log10()),
                    1e-9,
                );
                let saved = NoiseSnapshot::from_record(t.noise_record()).unwrap();
                for view in [
                    full(),
                    NoiseView {
                        weighting: Weighting::A,
                        band: Band::Audible,
                    },
                    NoiseView {
                        weighting: Weighting::ItuR468,
                        band: Band::ToNyquist,
                    },
                ] {
                    for m in sel {
                        assert_eq!(saved.measure(m, view), t.measure(m, view));
                    }
                }
                match t.measure(M::HumAndIdleTones, full()) {
                    Outcome::Value(Value::Tone(Some(v))) => near(v.frequency_hz, 50.0, 2.0),
                    x => panic!("{x:?}"),
                }
                near(
                    sum_curve_power(&curve(t.measure(M::NoiseSpectrum, full()))),
                    5e-11,
                    1e-19,
                );
            }
        }
    }
    for offset in [0, 1, 1200, 2399] {
        let mut pcm = vec![0.1; block + offset];
        pcm.extend(vec![0.0; 47_999]);
        pcm.extend(vec![0.1; 2 * block]);
        let t = float(rate, 1, &sel, &pcm);
        assert_eq!(t.quiet_seconds(), Some(0.0));
        for m in sel {
            assert!(matches!(t.measure(m, full()), Outcome::CantTell(_)));
        }
    }
    // A lead-in and a tail each qualify, but two separated subminimum runs
    // must never be concatenated into a qualifying passage.
    for (a, b, expected) in [
        (48_000, 48_960, 96_960),
        (30_000, 30_000, 0),
        (48_000, 30_000, 48_000),
    ] {
        let mut pcm = vec![1e-5; a];
        pcm.extend(vec![0.1; block + 333]);
        pcm.extend(vec![1e-5; b]);
        let t = float(rate, 1, &sel, &pcm);
        near(
            t.quiet_seconds().unwrap(),
            expected as f64 / f64::from(rate),
            1e-12,
        );
        if expected > 0 {
            near(scalar(t.measure(M::NoiseFloor, full())), -100.0, 1e-9);
        }
    }
}

#[test]
fn audit_short_indexed_album_lra_is_continuous_or_unavailable() {
    let rate = 48_000;
    let sel = [M::LoudnessRange, M::FlatToppedPeaks];
    for extra in [0, 37, 2399] {
        let a = tone(rate, 2 * rate as usize + extra, 1000.0, 0.05);
        let b = tone(rate, 2 * rate as usize + 101, 1000.0, 0.5);
        let f = format(rate, 1, Encoding::Float);
        let mut builder = AlbumAnalyzer::new(Measurements::new(&sel), Settings::default());
        let mut direct = true_peak::loudness::LoudnessMeter::with_roles_high_fidelity(
            rate,
            &[audioscopy::ChannelRole::Mono],
        )
        .unwrap();
        for pcm in [&a, &b] {
            builder.begin_track(f.clone()).unwrap();
            for c in pcm.chunks(701) {
                builder.push_interleaved(c).unwrap();
            }
            builder.end_track().unwrap();
            direct.push_interleaved(pcm).unwrap();
        }
        let album = builder.finish().unwrap();
        let expected = direct
            .finalize()
            .unwrap()
            .summary
            .range
            .finite_lu()
            .unwrap();
        near(
            scalar(album.measure(M::LoudnessRange, full())),
            expected,
            1e-10,
        );
        let cached =
            Album::from_tracks(vec![scan(f.clone(), &sel, &a), scan(f, &sel, &b)]).unwrap();
        assert!(matches!(
            cached.measure(M::LoudnessRange, full()),
            Outcome::CantTell(_)
        ));
        assert_eq!(
            cached.measure(M::FlatToppedPeaks, full()),
            album.measure(M::FlatToppedPeaks, full())
        );
    }
}

#[test]
fn audit_integer_flat_top_one_code_shoulders() {
    let plateau = [0, 32765, 32766, 32767, 32767, 32767, 32766, 32765, 0];
    let samples = |codes: &[i32], sign: f64| {
        codes
            .iter()
            .map(|&x| sign * f64::from(x) / 32768.0)
            .collect::<Vec<_>>()
    };
    for sign in [-1.0, 1.0] {
        let pcm = samples(&plateau, sign);
        for chunk in [1, 4, 5, 9] {
            let mut a = Analyzer::new(
                format(48_000, 1, Encoding::Integer(16)),
                Measurements::new(&[M::FlatToppedPeaks]),
                Settings::default(),
            )
            .unwrap();
            for c in pcm.chunks(chunk) {
                a.push_interleaved(c).unwrap();
            }
            assert_eq!(
                count(a.finish().unwrap().measure(M::FlatToppedPeaks, full())),
                1
            );
        }
    }
    let stereo: Vec<_> = samples(&plateau, 1.0)
        .into_iter()
        .zip(samples(&plateau, -1.0))
        .flat_map(|(a, b)| [a, b])
        .collect();
    assert_eq!(
        count(
            scan(
                format(48_000, 2, Encoding::Integer(16)),
                &[M::FlatToppedPeaks],
                &stereo
            )
            .measure(M::FlatToppedPeaks, full())
        ),
        2
    );
    for codes in [
        &[32761, 32762, 32763, 32764, 32765, 32766, 32767][..],
        &[32765, 32766, 32766, 32766, 32767, 32766, 0],
        &[0, 100, 100, 100, 0, 32767, 0],
    ] {
        let t = scan(
            format(48_000, 1, Encoding::Integer(16)),
            &[M::FlatToppedPeaks],
            &samples(codes, 1.0),
        );
        assert_eq!(count(t.measure(M::FlatToppedPeaks, full())), 0, "{codes:?}");
    }
}

#[test]
fn audit_dither_quantizer_alignment_and_restoration() {
    let rate = 48_000;
    for (shaped, expected) in [(false, Dither::Flat), (true, Dither::Shaped)] {
        let pcm = dither_pcm(rate, 4, shaped);
        for (encoding, scale) in [
            (Encoding::Integer(16), 1.0),
            (Encoding::Integer(24), 1.0),
            (Encoding::Float, 1.0),
            (Encoding::Integer(24), 1.0 / 256.0),
        ] {
            let x: Vec<_> = pcm.iter().map(|x| x * scale).collect();
            let t = scan(format(rate, 1, encoding), &[M::Dither], &x);
            assert_eq!(
                t.measure(M::Dither, full()),
                Outcome::Value(Value::Dither(expected)),
                "{encoding:?}, scale {scale}"
            );
            let saved = NoiseSnapshot::from_record(t.noise_record()).unwrap();
            assert_eq!(
                saved.measure(M::Dither, full()),
                t.measure(M::Dither, full())
            );
        }
        let short = scan(
            format(rate, 1, Encoding::Integer(24)),
            &[M::Dither],
            &pcm[..rate as usize - 1],
        );
        assert!(matches!(
            short.measure(M::Dither, full()),
            Outcome::CantTell(_)
        ));
    }
    let hiss: Vec<_> = dither_pcm(rate, 4, false)
        .into_iter()
        .map(|x| x * 100.0 / 256.0)
        .collect();
    for encoding in [Encoding::Integer(24), Encoding::Float] {
        let t = scan(format(rate, 1, encoding), &[M::Dither], &hiss);
        assert!(matches!(t.measure(M::Dither, full()), Outcome::CantTell(_)));
    }
    let arbitrary: Vec<_> = dither_pcm(rate, 4, false)
        .into_iter()
        .map(|x| x * 0.91728374613)
        .collect();
    assert!(matches!(
        float(rate, 1, &[M::Dither], &arbitrary).measure(M::Dither, full()),
        Outcome::CantTell(_)
    ));
}

#[test]
fn audit_resolution_grid_and_headroom_are_distinct() {
    let f = format(48_000, 1, Encoding::Integer(24));
    for (step, offset, low) in [(256, 0, 16), (192, 0, 18), (128, 0, 17), (192, 7, 24)] {
        let pcm: Vec<_> = (-32768..32768)
            .map(|i| f64::from(i * step + offset) / 8388608.0)
            .collect();
        let t = scan(f.clone(), &[M::BitsInUse], &pcm);
        assert_eq!(
            t.measure(M::BitsInUse, full()),
            Outcome::Value(Value::Bits(BitDepth::Integer {
                used: 16,
                declared: Some(24),
                low_bit_depth: low,
                observed_lattice_bits: Some(16),
                grid_step: step as u64,
                grid_scale_bits: 23
            }))
        );
    }
    for (step, low) in [(256, 16), (1, 24)] {
        let pcm: Vec<_> = (-100..=100)
            .map(|i| f64::from(i * step) / 8388608.0)
            .collect();
        assert_eq!(
            scan(f.clone(), &[M::BitsInUse], &pcm).measure(M::BitsInUse, full()),
            Outcome::Value(Value::Bits(BitDepth::Integer {
                used: low,
                declared: Some(24),
                low_bit_depth: low,
                observed_lattice_bits: Some(8),
                grid_step: step as u64,
                grid_scale_bits: 23
            }))
        );
    }
    let pcm: Vec<_> = (-32768..32768).map(|i| f64::from(i) / 65536.0).collect();
    assert!(matches!(
        float(48_000, 1, &[M::BitsInUse], &pcm).measure(M::BitsInUse, full()),
        Outcome::Value(Value::Bits(BitDepth::Integer {
            used: 16,
            low_bit_depth: 17,
            observed_lattice_bits: Some(16),
            ..
        }))
    ));
}

// Round-1 corrective audit R2. These tests exercise the public API, including
// arbitrary decoder chunks; expected PCM intervals and energy come from the
// known source, not from the analyzer's block classifications.
#[test]
fn audit_r2_rms_quiet_edges_and_record_round_trip() {
    let sel = [
        M::NoiseFloor,
        M::DynamicRange,
        M::EffectiveBits,
        M::NoiseSpectrum,
        M::HumAndIdleTones,
        M::NoiseByBand,
    ];
    for rate in [48_000_u32, 96_000] {
        let r = rate as usize;
        let block = r / 20;
        for (channels, phase, crest) in [(1_usize, 0.0, false), (2, 0.7, false), (1, 1.5, true)] {
            for offset in [0, 1, 97, block / 2, block - 1] {
                // Include exact minimum, nonqualifying 1-frame-short, and
                // 1.02 s passages. Some edge fragments have above-threshold
                // instantaneous samples despite their quiet RMS.
                for frames in [r - 1, r, r + r / 50] {
                    let quiet: Vec<_> = (0..frames)
                        .flat_map(|i| {
                            (0..channels).map(move |ch| {
                                let t = TAU * 50.0 * i as f64 / f64::from(rate)
                                    + phase
                                    + ch as f64 * TAU / 4.0;
                                if crest && i % 97 == 0 {
                                    0.003
                                } else {
                                    0.0012 * t.sin()
                                }
                            })
                        })
                        .collect();
                    let loud = 0.02_f64;
                    // Alternate lead-in/tail-only and surrounded cases. All
                    // use an offset relative to the rational 50 ms clock.
                    let (lead, tail) = match offset % 3 {
                        0 => (block + offset, 2 * block),
                        1 => (0, block + offset),
                        _ => (block + offset, 0),
                    };
                    let mut pcm = vec![loud; lead * channels];
                    pcm.extend_from_slice(&quiet);
                    pcm.extend(vec![loud; tail * channels]);
                    let mut a = Analyzer::new(
                        format(rate, channels, Encoding::Float),
                        Measurements::new(&sel),
                        Settings::default(),
                    )
                    .unwrap();
                    let chunk = [113, 997, block + 3][offset % 3];
                    for c in pcm.chunks(chunk * channels) {
                        a.push_interleaved(c).unwrap();
                    }
                    let t = a.finish().unwrap();
                    if frames < r {
                        assert_eq!(t.quiet_seconds(), Some(0.0));
                        assert!(matches!(
                            t.measure(M::NoiseFloor, full()),
                            Outcome::CantTell(_)
                        ));
                        continue;
                    }
                    near(
                        t.quiet_seconds().unwrap(),
                        frames as f64 / f64::from(rate),
                        1e-12,
                    );
                    let power = quiet.iter().map(|x| x * x).sum::<f64>() / quiet.len() as f64;
                    let expected = 10.0 * power.log10();
                    near(scalar(t.measure(M::NoiseFloor, full())), expected, 1e-9);
                    if !crest {
                        near(expected, 10.0 * 7.2e-7_f64.log10(), 1e-9);
                    }
                    near(
                        scalar(t.measure(M::DynamicRange, full())),
                        20.0 * loud.log10() - expected,
                        1e-9,
                    );
                    near(
                        approximate(t.measure(M::EffectiveBits, full())),
                        -expected / (20.0 * 2.0_f64.log10()),
                        1e-9,
                    );
                    let restored = NoiseSnapshot::from_record(t.noise_record()).unwrap();
                    for view in [
                        full(),
                        NoiseView {
                            weighting: Weighting::A,
                            band: Band::Audible,
                        },
                        NoiseView {
                            weighting: Weighting::ItuR468,
                            band: Band::ToNyquist,
                        },
                    ] {
                        for m in sel {
                            assert_eq!(restored.measure(m, view), t.measure(m, view));
                        }
                    }
                    near(
                        sum_curve_power(&curve(t.measure(M::NoiseSpectrum, full()))),
                        power,
                        1e-15,
                    );
                }
            }
        }
    }
    // The exact reported failure: 1.02 s at -61.426675 dBFS, 25 ms
    // offset, louder material on both sides (independent of loop coverage).
    let r = 48_000;
    let quiet = tone(r, 48_960, 50.0, 0.0012);
    let mut pcm = vec![0.1; 1200];
    pcm.extend(quiet);
    pcm.extend(vec![0.1; 4800]);
    let t = float(r, 1, &sel, &pcm);
    assert_eq!(t.quiet_seconds(), Some(1.02));
    near(
        scalar(t.measure(M::NoiseFloor, full())),
        10.0 * 7.2e-7_f64.log10(),
        1e-9,
    );
}

#[test]
fn audit_r2_bandwidth_endpoint_evidence_is_separate_from_power() {
    let r = 96_000;
    let n = 16_384;
    let f = 171.0 * f64::from(r) / n as f64;
    let burst: Vec<_> = tone(r, 960, 30_000.0, 0.01)
        .into_iter()
        .enumerate()
        .map(|(i, x)| x * (TAU * 0.5 * (i as f64 + 0.5) / 960.0).sin().powi(2))
        .collect();
    let both = [M::Bandwidth, M::Above20Khz, M::AverageSpectrum];
    let power_only = [M::Above20Khz, M::AverageSpectrum];
    for frames in [8 * n, 8 * n + 333, n - 1, 12_345, 4_800, 1_000] {
        // Nonzero endpoint samples are deliberately present, even in the
        // short files. A zero-extended unconditioned spectrum is not a valid
        // bandwidth oracle for this case.
        let low: Vec<_> = (0..frames)
            .map(|i| 0.1 * (TAU * f * i as f64 / f64::from(r) + 0.37).sin())
            .collect();
        let t = float(r, 1, &[M::Bandwidth], &low);
        near(scalar(t.measure(M::Bandwidth, full())), f / 1000.0, 0.04);
        let stationary: Vec<_> = low
            .iter()
            .zip(tone(r, frames, 30_000.0, 0.1))
            .map(|(a, b)| a + b)
            .collect();
        near(
            scalar(float(r, 1, &[M::Bandwidth], &stationary).measure(M::Bandwidth, full())),
            30.0,
            0.04,
        );
        for start in [0, (frames - burst.len()) / 2, frames - burst.len()] {
            let mut pcm = low.clone();
            for (x, y) in pcm[start..start + burst.len()].iter_mut().zip(&burst) {
                *x += y;
            }
            let with = float(r, 1, &both, &pcm);
            let without = float(r, 1, &power_only, &pcm);
            near(scalar(with.measure(M::Bandwidth, full())), 30.0, 0.4);
            // No change in the corrected complementary-window bins, even
            // at partial hops. This equality compares public curve outputs.
            for m in power_only {
                assert_eq!(with.measure(m, full()), without.measure(m, full()));
            }
            let mut split = Analyzer::new(
                format(r, 1, Encoding::Float),
                Measurements::new(&both),
                Settings::default(),
            )
            .unwrap();
            for c in pcm.chunks(127) {
                split.push_interleaved(c).unwrap();
            }
            let split = split.finish().unwrap();
            for m in both {
                assert_eq!(with.measure(m, full()), split.measure(m, full()));
            }
        }
    }
    // Multichannel evidence: no cancellation or channel-zero assumption.
    let frames = 4 * n + 71;
    let low = tone(r, frames, f, 0.1);
    let mut pcm: Vec<_> = low.iter().flat_map(|&x| [x, -x]).collect();
    for (frame, b) in pcm
        .as_chunks_mut::<2>()
        .0
        .iter_mut()
        .rev()
        .zip(burst.iter().rev())
    {
        frame[1] += b;
    }
    near(
        scalar(float(r, 2, &both, &pcm).measure(M::Bandwidth, full())),
        30.0,
        0.4,
    );
    // Silent complete windows must not prevent endpoint-only evidence.
    let mut pcm = vec![0.0; 4 * n + 1333];
    let end = pcm.len();
    pcm[end - 960..].copy_from_slice(&burst);
    near(
        scalar(float(r, 1, &both, &pcm).measure(M::Bandwidth, full())),
        30.0,
        0.8,
    );
}

#[test]
fn audit_r2_affine_headroom_width_is_not_complete_codebook_dependent() {
    let f = format(48_000, 1, Encoding::Integer(24));
    let sel = [M::BitsInUse];
    for (step, offset, low, used) in [
        (192, 0, 18, 16),
        (192, 7, 24, 16),
        (256, 7, 24, 16),
        (256, 0, 16, 16),
        (128, 0, 17, 17),
        (1, 0, 24, 24),
    ] {
        let pcm: Vec<_> = (-30_000..=30_000)
            .map(|i| f64::from(i * step + offset) / 8388608.0)
            .collect();
        let expected = Outcome::Value(Value::Bits(BitDepth::Integer {
            used,
            declared: Some(24),
            low_bit_depth: low,
            observed_lattice_bits: Some(16),
            grid_step: step as u64,
            grid_scale_bits: 23,
        }));
        let t = scan(f.clone(), &sel, &pcm);
        assert_eq!(t.measure(M::BitsInUse, full()), expected);
        let mut a = Analyzer::new(f.clone(), Measurements::new(&sel), Settings::default()).unwrap();
        for c in pcm.chunks(19) {
            a.push_interleaved(c).unwrap();
        }
        assert_eq!(a.finish().unwrap().measure(M::BitsInUse, full()), expected);
    }
    let pcm: Vec<_> = (-30_000..=30_000)
        .map(|i| f64::from(i * 192) / 8388608.0)
        .collect();
    assert!(matches!(
        float(48_000, 1, &sel, &pcm).measure(M::BitsInUse, full()),
        Outcome::Value(Value::Bits(BitDepth::Integer {
            used: 16,
            low_bit_depth: 18,
            observed_lattice_bits: Some(16),
            ..
        }))
    ));
    // Three distinct affine positions establish spacing; two endpoints
    // alone do not establish a nontrivial equally spaced subset.
    let three = [-192.0 / 8388608.0, 0.0, 192.0 / 8388608.0];
    assert!(matches!(
        scan(f.clone(), &sel, &three).measure(M::BitsInUse, full()),
        Outcome::Value(Value::Bits(BitDepth::Integer {
            used: 2,
            observed_lattice_bits: Some(2),
            ..
        }))
    ));
    assert!(matches!(
        scan(f.clone(), &sel, &three[..2]).measure(M::BitsInUse, full()),
        Outcome::Value(Value::Bits(BitDepth::Integer {
            used: 18,
            observed_lattice_bits: Some(1),
            ..
        }))
    ));
    let ordinary = scan(f.clone(), &sel, &[-1.0, 0.0, 1.0 / 8388608.0]);
    let album = Album::from_tracks(vec![scan(f, &sel, &pcm), ordinary]).unwrap();
    assert!(matches!(
        album.measure(M::BitsInUse, full()),
        Outcome::Value(Value::Bits(BitDepth::Integer { used: 24, .. }))
    ));
}

#[test]
fn audit_r2_rms_quiet_broadband_with_real_crests() {
    let r = 48_000_u32;
    let sel = [M::NoiseFloor, M::NoiseSpectrum];
    let raw: Vec<_> = white(4 * r as usize, 0x1234abcd)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|x| x.iter().sum::<f64>())
        .collect();
    let rms = (raw.iter().map(|x| x * x).sum::<f64>() / raw.len() as f64).sqrt();
    let quiet: Vec<_> = raw.iter().map(|x| x * 0.00085 / rms).collect();
    assert!(quiet.iter().any(|x| x.abs() > 0.002));
    for offset in [1, 97, 1200, 2301, 2399] {
        let mut pcm = vec![0.02; 2400 + offset];
        pcm.extend_from_slice(&quiet);
        pcm.extend(vec![0.02; 4800]);
        let t = float(r, 1, &sel, &pcm);
        assert_eq!(t.quiet_seconds(), Some(1.0));
        near(
            scalar(t.measure(M::NoiseFloor, full())),
            20.0 * 0.00085_f64.log10(),
            1e-9,
        );
        let saved = NoiseSnapshot::from_record(t.noise_record()).unwrap();
        assert_eq!(
            saved.measure(M::NoiseFloor, full()),
            t.measure(M::NoiseFloor, full())
        );
    }
}

// The R2 detector's last extreme precedes the actual transition by 14
// samples. Its failed aggregate test falls back to just 17 quiet samples
// of the suffix, leaving a 47,777-frame run that it incorrectly rejects.
#[test]
fn audit_r3_oscillatory_quiet_edges() {
    let sel = [
        M::NoiseFloor,
        M::EffectiveBits,
        M::NoiseSpectrum,
        M::NoiseByBand,
        M::Dither,
    ];
    let expected = 10.0 * 7.2e-7_f64.log10();
    for rate in [48_000_u32, 96_000] {
        let r = rate as usize;
        let block = r / 20;
        // First case is the exact failing-first fixture. Others vary the
        // separator phase, quiet crest, block clock, and channel count.
        for (channels, offset, loud_phase, quiet_phase) in [
            (1_usize, block / 2, 0.35_f64, 0.7_f64),
            (1, 1, 1.5, 0.0),
            (1, block - 1, 3.1, 2.8),
            (1, 97, 4.7, 1.4),
            (2, block / 2, 0.35, 0.7),
            (2, 97, 5.9, 1.4),
        ] {
            let lead = block + offset;
            for length in [r + r / 50, r - r / 50] {
                let total = lead + length + 2 * block;
                let pcm: Vec<_> = (0..total)
                    .flat_map(|i| {
                        (0..channels).map(move |ch| {
                            if (lead..lead + length).contains(&i) {
                                0.0012
                                    * (TAU * 50.0 * (i - lead) as f64 / f64::from(rate)
                                        + quiet_phase
                                        + ch as f64 * 1.57)
                                        .sin()
                            } else {
                                0.02 * (TAU * 440.0 * i as f64 / f64::from(rate)
                                    + loud_phase
                                    + ch as f64 * 0.43)
                                    .sin()
                            }
                        })
                    })
                    .collect();
                let mirrored: Vec<_> = pcm
                    .chunks_exact(channels)
                    .rev()
                    .flatten()
                    .copied()
                    .collect();
                for (direction, samples) in [&pcm[..], &mirrored[..]].into_iter().enumerate() {
                    let mut baseline = None;
                    for chunk in [113, 997, block + 3, total] {
                        let mut a = Analyzer::new(
                            format(rate, channels, Encoding::Float),
                            Measurements::new(&sel),
                            Settings::default(),
                        )
                        .unwrap();
                        for c in samples.chunks(chunk * channels) {
                            a.push_interleaved(c).unwrap();
                        }
                        let t = a.finish().unwrap();
                        if length < r {
                            assert_eq!(t.quiet_seconds(), Some(0.0));
                            assert!(matches!(
                                t.measure(M::NoiseFloor, full()),
                                Outcome::CantTell(_)
                            ));
                            continue;
                        }
                        let seconds = t.quiet_seconds().unwrap();
                        // Exact endpoints of a tonal transition can be
                        // ambiguous at a zero crossing, but millisecond-
                        // scale estimation must not discard this passage.
                        near(seconds, 1.02, 0.002);
                        let floor = scalar(t.measure(M::NoiseFloor, full()));
                        near(floor, expected, 0.5);
                        if rate == 48_000 && channels == 1 && offset == block / 2 && direction == 0
                        {
                            near(seconds, 1.02, 1e-12);
                            near(floor, expected, 1e-9);
                        }
                        let restored = NoiseSnapshot::from_record(t.noise_record()).unwrap();
                        for view in [
                            full(),
                            NoiseView {
                                weighting: Weighting::A,
                                band: Band::Audible,
                            },
                            NoiseView {
                                weighting: Weighting::ItuR468,
                                band: Band::ToNyquist,
                            },
                        ] {
                            for m in [
                                M::NoiseFloor,
                                M::EffectiveBits,
                                M::NoiseSpectrum,
                                M::NoiseByBand,
                            ] {
                                assert_eq!(t.measure(m, view), restored.measure(m, view));
                            }
                        }
                        // Every decoder chunking must produce exactly the
                        // same admitted real frames and full-band result.
                        let observation = (seconds, floor);
                        if let Some(prior) = baseline {
                            assert_eq!(observation, prior);
                        } else {
                            baseline = Some(observation);
                        }
                    }
                }
            }
        }
    }
}

// R4: endpoint peaks compete with the same accumulated recording-wide
// power reference. The exact integer PCM is deliberately dither-scale,
// not an artificial arbitrarily low floating-point noise fixture.
fn audit_r4_faded_pcm() -> Vec<f64> {
    let r = 96_000_u32;
    let n = 16_384_usize;
    let mut pcm = tone(r, 4 * n, 171.0 * f64::from(r) / n as f64, 0.1);
    for (i, x) in pcm[3 * n..].iter_mut().enumerate() {
        *x *= 0.5 + 0.5 * (std::f64::consts::PI * i as f64 / (n - 1) as f64).cos();
    }
    for x in &mut pcm {
        *x = (*x * 8388608.0).round() / 8388608.0;
    }
    pcm.resize(8 * n, 0.0);
    pcm
}

#[test]
fn audit_r4_dither_at_either_endpoint_has_programme_relative_materiality() {
    let r = 96_000_u32;
    let n = 16_384_usize;
    let pcm = audit_r4_faded_pcm();
    let format24 = format(r, 1, Encoding::Integer(24));
    let both = [M::Bandwidth, M::Above20Khz, M::AverageSpectrum];
    let power_only = [M::Above20Khz, M::AverageSpectrum];
    let baseline = scalar(scan(format24.clone(), &both, &pcm).measure(M::Bandwidth, full()));
    near(baseline, 1.0, 0.1);
    let mut state = 0x14a6_u64;
    let noise: Vec<_> = (0..n / 2)
        .map(|_| {
            let a = (rng(&mut state) >> 63) as i32;
            let b = (rng(&mut state) >> 63) as i32;
            f64::from(a - b) / 8388608.0
        })
        .collect();
    let rms_db = 10.0 * (noise.iter().map(|x| x * x).sum::<f64>() / noise.len() as f64).log10();
    near(rms_db, -141.48, 0.5);
    for start in [5 * n, pcm.len() - noise.len()] {
        let mut x = pcm.clone();
        x[start..start + noise.len()].copy_from_slice(&noise);
        for reverse in [false, true] {
            if reverse {
                x.reverse();
            }
            let with = scan(format24.clone(), &both, &x);
            // R3 returns almost Nyquist for the endpoint, but not for
            // exactly the same noise in the interior silence.
            near(scalar(with.measure(M::Bandwidth, full())), baseline, 0.02);
            let without = scan(format24.clone(), &power_only, &x);
            for m in power_only {
                assert_eq!(with.measure(m, full()), without.measure(m, full()));
            }
            for chunk in [127, 4096] {
                let mut a = Analyzer::new(
                    format24.clone(),
                    Measurements::new(&both),
                    Settings::default(),
                )
                .unwrap();
                for c in x.chunks(chunk) {
                    a.push_interleaved(c).unwrap();
                }
                let split = a.finish().unwrap();
                for m in both {
                    assert_eq!(with.measure(m, full()), split.measure(m, full()));
                }
            }
            for gain in [2.0_f64.powi(-15), -0.37, 2.0, 16.0] {
                let y: Vec<_> = x.iter().map(|v| v * gain).collect();
                near(
                    scalar(float(r, 1, &[M::Bandwidth], &y).measure(M::Bandwidth, full())),
                    baseline,
                    0.02,
                );
            }
        }
    }
}

#[test]
fn audit_r4_negligible_endpoint_tones_are_rejected() {
    let r = 96_000_u32;
    let n = 16_384_usize;
    let pcm = audit_r4_faded_pcm();
    let selection = [M::Bandwidth];
    let baseline = scalar(float(r, 1, &selection, &pcm).measure(M::Bandwidth, full()));
    let taper = |i: usize, len: usize| {
        (std::f64::consts::PI * (i as f64 + 0.5) / len as f64)
            .sin()
            .powi(2)
    };
    let end = pcm.len();
    for amp in [0.001, 1e-6, 1e-7, 1e-8, 1e-12] {
        let mut x = pcm.clone();
        let len = n / 2;
        for (i, (s, b)) in x[end - len..]
            .iter_mut()
            .zip(tone(r, len, 30_000.0, amp))
            .enumerate()
        {
            *s = b * taper(i, len);
        }
        for reverse in [false, true] {
            if reverse {
                x.reverse();
            }
            let got = scalar(float(r, 1, &selection, &x).measure(M::Bandwidth, full()));
            if amp == 0.001 {
                near(got, 30.0, 0.4);
            } else {
                near(got, baseline, 0.02);
            }
        }
    }
}

#[test]
fn audit_r4_material_lower_endpoint_peak_survives() {
    let r = 96_000_u32;
    let pcm = audit_r4_faded_pcm();
    let selection = [M::Bandwidth];
    let taper = |i: usize, len: usize| {
        (std::f64::consts::PI * (i as f64 + 0.5) / len as f64)
            .sin()
            .powi(2)
    };
    let end = pcm.len();
    // The highest locally eligible peak is immaterial globally. Retaining
    // only that candidate would conceal the significant lower peak as well.
    let len = 960;
    let mut x = pcm.clone();
    for (i, s) in x[end - len..].iter_mut().enumerate() {
        let t = TAU * i as f64 / f64::from(r);
        *s = taper(i, len) * (0.005 * (6000.0 * t).sin() + 3e-6 * (30_000.0 * t).sin());
    }
    for reverse in [false, true] {
        if reverse {
            x.reverse();
        }
        for gain in [1.0, 0.0001, -0.5, 8.0] {
            let y: Vec<_> = x.iter().map(|v| v * gain).collect();
            near(
                scalar(float(r, 1, &selection, &y).measure(M::Bandwidth, full())),
                6.0,
                0.4,
            );
        }
    }
}

#[test]
fn audit_r4_low_level_only_and_multichannel_bandwidth() {
    let r = 96_000_u32;
    let n = 16_384_usize;
    let pcm = audit_r4_faded_pcm();
    let selection = [M::Bandwidth];
    let baseline = scalar(float(r, 1, &selection, &pcm).measure(M::Bandwidth, full()));
    let taper = |i: usize, len: usize| {
        (std::f64::consts::PI * (i as f64 + 0.5) / len as f64)
            .sin()
            .powi(2)
    };
    let end = pcm.len();
    // An actual quiet ultrasonic recording (as opposed to an ultrasonic
    // fragment negligible relative to a louder programme) is still eligible.
    for len in [1_000, n / 2 - 1, n / 2, n / 2 + 1, n - 1] {
        let mut x = vec![0.0; len];
        for (i, (s, b)) in x[len - 960..]
            .iter_mut()
            .zip(tone(r, 960, 30_000.0, 1e-9))
            .enumerate()
        {
            *s = b * taper(i, 960);
        }
        for reverse in [false, true] {
            if reverse {
                x.reverse();
            }
            near(
                scalar(float(r, 1, &selection, &x).measure(M::Bandwidth, full())),
                30.0,
                0.8,
            );
        }
    }
    // Channel-mean normalization, including a high-frequency signal
    // present on only one channel; no dependence on the number of channels.
    for channels in [2, 3] {
        for amp in [0.01, 1e-8] {
            let mut y: Vec<_> = pcm
                .iter()
                .flat_map(|&x| std::iter::repeat_n(x, channels))
                .collect();
            for (i, frame) in y[(end - 960) * channels..]
                .chunks_exact_mut(channels)
                .enumerate()
            {
                frame[channels - 1] =
                    taper(i, 960) * amp * (TAU * 30_000.0 * i as f64 / f64::from(r)).sin();
            }
            let got = scalar(float(r, channels, &selection, &y).measure(M::Bandwidth, full()));
            if amp == 0.01 {
                near(got, 30.0, 0.4);
            } else {
                near(got, baseline, 0.02);
            }
        }
    }
}

// R5: complete windows must use the same programme floor as endpoint
// candidates, even when almost all material energy lies at a boundary.
fn audit_r5_isolated_pcm(frames: usize, hz: f64) -> Vec<f64> {
    let r = 96_000_u32;
    let mut pcm = vec![0.0; 8 * 16_384];
    for (i, (out, sample)) in pcm[..frames]
        .iter_mut()
        .zip(tone(r, frames, hz, 0.1))
        .enumerate()
    {
        let taper = (std::f64::consts::PI * (i as f64 + 0.5) / frames as f64)
            .sin()
            .powi(2);
        *out = (sample * taper * 8388608.0).round() / 8388608.0;
    }
    pcm
}

fn audit_r5_triangular_pcm(frames: usize) -> Vec<f64> {
    let mut state = 0x14a6_u64;
    (0..frames)
        .map(|_| {
            let a = (rng(&mut state) >> 63) as i32;
            let b = (rng(&mut state) >> 63) as i32;
            f64::from(a - b) / 8388608.0
        })
        .collect()
}

#[test]
fn audit_r5_isolated_note_dither_does_not_set_interior_bandwidth() {
    let r = 96_000_u32;
    let pcm = audit_r5_isolated_pcm(960, 1000.0);
    let noise = audit_r5_triangular_pcm(pcm.len());
    let rms_db = 10.0 * (noise.iter().map(|x| x * x).sum::<f64>() / noise.len() as f64).log10();
    near(rms_db, -141.48, 0.05);
    let f = format(r, 1, Encoding::Integer(24));
    let selected = [M::Bandwidth, M::Above20Khz, M::AverageSpectrum];
    let power_only = [M::Above20Khz, M::AverageSpectrum];
    for start in [0, 960] {
        let mut dithered = pcm.clone();
        for (x, d) in dithered[start..].iter_mut().zip(&noise[start..]) {
            *x += *d;
        }
        let mut clean = pcm.clone();
        for reverse in [false, true] {
            if reverse {
                clean.reverse();
                dithered.reverse();
            }
            let baseline = scalar(scan(f.clone(), &selected, &clean).measure(M::Bandwidth, full()));
            let result = scan(f.clone(), &selected, &dithered);
            let bandwidth = scalar(result.measure(M::Bandwidth, full()));
            // R4 reports almost Nyquist although the only change is
            // ordinary one-LSB triangular dither.
            near(bandwidth, baseline, 0.012);
            near(baseline, 1.44140625, 0.025);
            let retained = scan(f.clone(), &power_only, &dithered);
            for m in power_only {
                assert_eq!(result.measure(m, full()), retained.measure(m, full()));
            }
            for chunk in [113, 4096, dithered.len()] {
                let mut a =
                    Analyzer::new(f.clone(), Measurements::new(&selected), Settings::default())
                        .unwrap();
                for c in dithered.chunks(chunk) {
                    a.push_interleaved(c).unwrap();
                }
                let split = a.finish().unwrap();
                for m in selected {
                    assert_eq!(result.measure(m, full()), split.measure(m, full()));
                }
            }
            for gain in [2.0_f64.powi(-15), -0.37, 2.0, 8.0] {
                let y: Vec<_> = dithered.iter().map(|x| x * gain).collect();
                near(
                    scalar(float(r, 1, &[M::Bandwidth], &y).measure(M::Bandwidth, full())),
                    bandwidth,
                    0.012,
                );
            }
        }
    }
}

#[test]
fn audit_r5_nonzero_but_immaterial_interior_uses_endpoint_fallback() {
    let r = 96_000_u32;
    // 32 real frames with four cycles of a 12 kHz carrier. The complete
    // windows contain nonzero dither, but no bin meets the programme floor.
    // The shorter actual-support endpoint view contains an eligible peak.
    let mut pcm = audit_r5_isolated_pcm(32, 12_000.0);
    for (x, d) in pcm.iter_mut().zip(audit_r5_triangular_pcm(8 * 16_384)) {
        *x += d;
    }
    for reverse in [false, true] {
        if reverse {
            pcm.reverse();
        }
        let t = scan(format(r, 1, Encoding::Integer(24)), &[M::Bandwidth], &pcm);
        near(scalar(t.measure(M::Bandwidth, full())), 12.0, 1.0);
    }
}

#[test]
fn high_fidelity_loudness_is_used_by_audioscopy() {
    let rate = 768_000;
    let selected = [M::ShortTermMaximum, M::LoudnessOverTime];
    let pcm: Vec<_> = (0..4 * rate as usize)
        .map(|i| {
            0.01 + 1e-6
                * (TAU * ((i as u64 * 1000) % u64::from(rate)) as f64 / f64::from(rate)).sin()
        })
        .collect();
    let track = float(rate, 1, &selected, &pcm);
    let points = curve(track.measure(M::LoudnessOverTime, full()));
    assert_eq!(points.len(), 11);
    near(points.last().unwrap().y, -123.03842522169084, 1e-7);
    near(points.last().unwrap().x, 4.0, 0.0);
    // Maximum for the unoffset signal must also use the new filter, rather
    // than allowing the initial DC transient to dominate the test fixture.
    let pure: Vec<_> = tone(rate, 4 * rate as usize, 1000.0, 1e-12);
    let t = float(rate, 1, &selected, &pure);
    near(
        scalar(t.measure(M::ShortTermMaximum, full())),
        -243.03842522169084,
        0.002,
    );
    near(
        curve(t.measure(M::LoudnessOverTime, full()))
            .last()
            .unwrap()
            .y,
        -243.03842522169084,
        1e-7,
    );
}

#[test]
fn high_fidelity_album_lra_matches_independent_high_rate_programme() {
    let rate = 768_000;
    let selected = [M::LoudnessRange];
    let mut album = AlbumAnalyzer::new(Measurements::new(&selected), Settings::default());
    let f = format(rate, 1, Encoding::Float);
    for (length, amp, dc) in [
        (2 * rate as usize + 37, 0.05, 0.01),
        (3 * rate as usize + 101, 0.5, -0.02),
    ] {
        album.begin_track(f.clone()).unwrap();
        let pcm: Vec<_> = (0..length)
            .map(|i| {
                dc + amp
                    * (TAU * ((i as u64 * 1000) % u64::from(rate)) as f64 / f64::from(rate)).sin()
            })
            .collect();
        for chunk in pcm.chunks(4093) {
            album.push_interleaved(chunk).unwrap();
        }
        album.end_track().unwrap();
    }
    let album = album.finish().unwrap();
    // Independent closed-form MP transfer, windows, and native EOF tail.
    near(
        scalar(album.measure(M::LoudnessRange, full())),
        3.113696800047,
        1e-7,
    );
}
