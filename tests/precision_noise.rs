#![cfg(feature = "precision-spectrum")]
use audioscopy::precision::spectral::{QuietNoiseAnalyzer, SpectrumBackend as B};
use audioscopy::precision::Precision;
use audioscopy::{Band, Encoding, Format, NoiseView, Weighting};

fn make(b: B, weighting: Weighting) -> QuietNoiseAnalyzer {
    let precision = if b == B::X87 {
        Precision::Extended80
    } else {
        Precision::DoubleDouble
    };
    QuietNoiseAnalyzer::new(
        Format {
            sample_rate: 8000,
            channels: 1,
            encoding: Encoding::Float,
            loudness_roles: None,
        },
        0.05,
        NoiseView {
            band: Band::Full,
            weighting,
        },
        precision,
        b,
    )
    .unwrap()
}

#[test]
fn passage_minimum_duration_linear_pool_and_silence() {
    for b in [B::X87, B::DoubleDoublePortable, B::DoubleDoubleFma]
        .into_iter()
        .filter(|b| b.available())
    {
        let mut a = make(b, Weighting::Unweighted);
        a.push_interleaved(&[1.0; 399]).unwrap();
        assert!(!a.end_passage().unwrap());
        a.push_interleaved(&[0.25; 400]).unwrap();
        assert!(a.end_passage().unwrap());
        a.push_interleaved(&[0.125; 800]).unwrap();
        let r = a.finish().unwrap().unwrap();
        assert_eq!(r.quiet_frames, 1200);
        assert!((r.floor.as_f64() - 0.03125).abs() < 1e-15);
        assert!(r.bands[3].is_none() && r.bands[4].is_none());
        let mut z = make(b, Weighting::A);
        z.push_interleaved(&[0.0; 400]).unwrap();
        let z = z.finish().unwrap().unwrap();
        assert_eq!(z.floor.hi, 0.0);
        assert!(z.effective_bits().is_none());
        assert!(make(b, Weighting::Unweighted).finish().unwrap().is_none());
    }
}

#[test]
fn subnormal_floor_does_not_round_to_silence() {
    for b in [B::X87, B::DoubleDoublePortable, B::DoubleDoubleFma]
        .into_iter()
        .filter(|b| b.available())
    {
        let mut a = make(b, Weighting::Unweighted);
        a.push_interleaved(&[f64::from_bits(1); 401]).unwrap();
        let r = a.finish().unwrap().unwrap();
        assert!(r.floor.exponent < -2000);
        assert_eq!(r.floor.as_f64(), 0.0);
        assert!(r.noise_floor_dbfs().is_finite());
        assert!((r.effective_bits().unwrap() - 1074.0).abs() < 1e-10);
    }
}
