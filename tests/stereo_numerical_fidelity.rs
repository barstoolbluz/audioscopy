//! Public-path regressions for DC-conditioned Pearson correlation.
use audioscopy::{
    Album, Analyzer, Encoding, Format, Measurement, Measurements, NoiseView, Outcome, Settings,
    Track, Value,
};

const PATTERN: [(f64, f64); 7] = [
    (0.0, 0.0),
    (1.0, 1.0),
    (0.0, 1.0),
    (1.0, 0.0),
    (1.0, 1.0),
    (1.0, 1.0),
    (0.0, 0.0),
];
const EXPECTED: f64 = 5.0 / 12.0;

fn analyze(
    rate: u32,
    encoding: Encoding,
    pcm: &[f64],
    repeats: usize,
    chunk_frames: usize,
) -> Track {
    let mut a = Analyzer::new(
        Format {
            sample_rate: rate,
            channels: 2,
            encoding,
            loudness_roles: None,
        },
        Measurements::new(&[
            Measurement::Correlation,
            Measurement::Width,
            Measurement::Balance,
            Measurement::MonoCompatibility,
        ]),
        Settings::default(),
    )
    .unwrap();
    for _ in 0..repeats {
        for chunk in pcm.chunks(chunk_frames * 2) {
            a.push_interleaved(chunk).unwrap();
        }
    }
    a.finish().unwrap()
}

fn scalar(result: Outcome<Value>) -> f64 {
    match result {
        Outcome::Value(Value::Scalar(x)) => x,
        other => panic!("expected scalar, got {other:?}"),
    }
}

fn correlation(track: &Track) -> f64 {
    scalar(track.measure(Measurement::Correlation, NoiseView::default()))
}

fn assert_near(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "correlation {actual:.17e}; reference {expected:.17e}; tolerance {tolerance:.2e}"
    );
}

#[test]
fn correlation_retains_one_ulp_variations_on_dc() {
    // Four ones in each channel, three simultaneous ones; exact Pearson = 5/12.
    let lsb = 2.0_f64.powi(-53);
    let cycle: Vec<_> = PATTERN
        .iter()
        .flat_map(|&(l, r)| [0.75 + l * lsb, -0.5 + r * lsb])
        .collect();
    let pcm = cycle.repeat(10_000);
    let track = analyze(48_000, Encoding::Float, &pcm, 1, 997);
    assert_near(correlation(&track), EXPECTED, 2e-11);
    let rechunked = analyze(48_000, Encoding::Float, &pcm, 1, 4096);
    assert_eq!(
        correlation(&rechunked).to_bits(),
        correlation(&track).to_bits()
    );
    for rate in [8_000, 44_100, 192_000, 768_000] {
        let other = analyze(rate, Encoding::Float, &pcm, 1, 4096);
        // The public track reducer also multiplies/divides by duration.
        assert_near(correlation(&other), EXPECTED, 2e-11);
    }
}

#[test]
fn correlation_preserves_32bit_pcm_variations_over_long_stream() {
    let lsb = 2.0_f64.powi(-31);
    let cycle: Vec<_> = PATTERN
        .iter()
        .flat_map(|&(l, r)| [0.75 + l * lsb, -0.5 + r * lsb])
        .collect();
    let pcm = cycle.repeat(1000);
    // 7 million actual input frames; bounded fixture memory.
    let track = analyze(48_000, Encoding::Integer(32), &pcm, 1000, 4093);
    assert_near(correlation(&track), EXPECTED, 2e-10);
}

#[test]
fn correlation_polarity_translation_and_album_semantics() {
    let lsb = 2.0_f64.powi(-40);
    let pcm: Vec<_> = PATTERN
        .iter()
        .flat_map(|&(l, r)| [l * lsb, r * lsb])
        .collect();
    let shifted: Vec<_> = pcm
        .as_chunks::<2>()
        .0
        .iter()
        .flat_map(|f| [0.75 + f[0], -0.5 + f[1]])
        .collect();
    let inverted: Vec<_> = shifted
        .as_chunks::<2>()
        .0
        .iter()
        .flat_map(|f| [f[0], -f[1]])
        .collect();
    let a = analyze(48_000, Encoding::Float, &pcm, 100, 1);
    let b = analyze(48_000, Encoding::Float, &shifted, 100, 3);
    let c = analyze(48_000, Encoding::Float, &inverted, 100, 7);
    let swapped: Vec<_> = shifted
        .as_chunks::<2>()
        .0
        .iter()
        .flat_map(|f| [f[1], f[0]])
        .collect();
    let d = analyze(48_000, Encoding::Float, &swapped, 100, 5);
    assert_eq!(correlation(&b).to_bits(), correlation(&d).to_bits());
    assert_near(correlation(&a), EXPECTED, 2e-12);
    assert_eq!(correlation(&a).to_bits(), correlation(&b).to_bits());
    assert_near(correlation(&c), -EXPECTED, 2e-12);
    let album = Album::from_tracks(vec![b, c]).unwrap();
    assert_near(
        scalar(album.measure(Measurement::Correlation, NoiseView::default())),
        0.0,
        2e-12,
    );
    let dc = analyze(48_000, Encoding::Float, &[0.75, -0.5], 1000, 1);
    assert_eq!(correlation(&dc), 0.0);
    let mono = analyze(
        48_000,
        Encoding::Float,
        &[0.75, 0.75, 0.75 + lsb, 0.75 + lsb],
        100,
        1,
    );
    assert_near(correlation(&mono), 1.0, 2e-12);
    assert_eq!(
        scalar(mono.measure(Measurement::Width, NoiseView::default())),
        f64::NEG_INFINITY
    );
}
