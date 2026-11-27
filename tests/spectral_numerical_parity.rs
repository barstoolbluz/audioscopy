//! Public selection invariance for the power-only spectral candidate.
use audioscopy::{
    Analyzer, Encoding, Format, Measurement as M, Measurements, NoiseView, Settings, Track,
};

fn analyze(rate: u32, pcm: &[f64], selection: &[M], chunk: usize) -> Track {
    let mut a = Analyzer::new(
        Format {
            sample_rate: rate,
            channels: 2,
            encoding: Encoding::Float,
            loudness_roles: None,
        },
        Measurements::new(selection),
        Settings {
            shortest_gap_seconds: 0.1,
            ..Settings::default()
        },
    )
    .unwrap();
    for samples in pcm.chunks(chunk * 2) {
        a.push_interleaved(samples).unwrap();
    }
    a.finish().unwrap()
}

#[test]
fn power_only_and_combined_stereo_produce_identical_spectral_results() {
    let power = [
        M::Bandwidth,
        M::Above20Khz,
        M::AverageSpectrum,
        M::LikelySourceRate,
        M::LossyCodingPattern,
        M::NoiseFloor,
        M::NoiseSpectrum,
        M::EffectiveBits,
    ];
    let mut combined = power.to_vec();
    combined.extend([M::BassWidth, M::InterChannelDelay]);
    for rate in [8_000, 44_101, 48_000, 192_000, 768_000] {
        let mut seed = 0x7a12b4912f3a_u64;
        let mut pcm = Vec::new();
        for i in 0..rate as usize / 2 + 13 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let x = (seed >> 11) as f64 / (1_u64 << 53) as f64 - 0.5;
            let gain = if i < rate as usize / 8 { 0.2 } else { 1e-5 };
            pcm.extend([gain * x, gain * x * 0.5]);
        }
        let a = analyze(rate, &pcm, &power, 997);
        let b = analyze(rate, &pcm, &combined, 4093);
        for m in power {
            assert_eq!(
                a.measure(m, NoiseView::default()),
                b.measure(m, NoiseView::default()),
                "rate={rate}, measurement={m:?}"
            );
        }
        assert!(
            a.noise_record().quiet.is_some(),
            "rate={rate}: quiet fixture did not qualify"
        );
        assert_eq!(a.noise_record(), b.noise_record());
    }
}
