#![cfg(feature = "precision-spectrum")]
use audioscopy::precision::spectral::{CrossSpectrum, CrossSpectrumAnalyzer, SpectrumBackend as B};
use audioscopy::precision::Precision;
use audioscopy::{Encoding, Error, Format};

fn format(channels: usize) -> Format {
    Format {
        sample_rate: 8000,
        channels,
        encoding: Encoding::Float,
        loudness_roles: None,
    }
}

fn backends() -> impl Iterator<Item = B> {
    [
        B::X87,
        B::DoubleDoublePortable,
        B::DoubleDoubleFma,
        B::DoubleDoubleAvx2,
        B::DoubleDoubleAvx512,
        B::DoubleDoubleNeon,
    ]
    .into_iter()
    .filter(|b| b.available())
}

fn analyzer(b: B) -> CrossSpectrumAnalyzer {
    let precision = if b == B::X87 {
        Precision::Extended80
    } else {
        Precision::DoubleDouble
    };
    CrossSpectrumAnalyzer::new(format(2), precision, b).unwrap()
}

fn run(b: B, pcm: &[f64], chunk: usize) -> CrossSpectrum {
    let mut a = analyzer(b);
    for x in pcm.chunks(chunk * 2) {
        a.push_interleaved(x).unwrap();
    }
    a.finish().unwrap()
}

#[test]
fn admission_transactionality_and_correlation_sign() {
    assert!(CrossSpectrumAnalyzer::new(format(1), Precision::DoubleDouble, B::Auto).is_err());
    assert!(CrossSpectrumAnalyzer::new(format(2), Precision::Binary64, B::Auto).is_err());
    for b in backends() {
        assert!(matches!(analyzer(b).finish(), Err(Error::Empty)));
        let mut a = analyzer(b);
        a.push_interleaved(&[0.25, -0.5]).unwrap();
        assert!(matches!(
            a.push_interleaved(&[0.0]),
            Err(Error::IncompleteFrame)
        ));
        assert!(matches!(
            a.push_interleaved(&[0.0, f64::NAN]),
            Err(Error::InvalidSample(1))
        ));
        assert_eq!(a.frames(), 1);
        let r = a.finish().unwrap();
        assert_eq!(r.frames, 1);
        assert_eq!(r.bins.len(), 1025);
        let l: f64 = r.bins.iter().map(|x| x.left.as_f64()).sum();
        let right: f64 = r.bins.iter().map(|x| x.right.as_f64()).sum();
        let cross: f64 = r.bins.iter().map(|x| x.real.as_f64()).sum();
        assert!((l - 0.0625).abs() < 1e-14);
        assert!((right - 0.25).abs() < 1e-14);
        assert!((cross + 0.125).abs() < 1e-14);
    }
}

#[test]
fn independent_scale_and_vector_parity() {
    let tiny = f64::from_bits(1);
    let pcm = [0.5, tiny].repeat(1117);
    let mut scalar = None;
    for b in backends() {
        let r = run(b, &pcm, 17);
        assert_eq!(r, run(b, &pcm, 4093));
        assert!(r.bins[0].left.hi > 0.0);
        assert!(r.bins[0].right.hi > 0.0 && r.bins[0].right.exponent < -2000);
        assert!(r.bins[0].real.hi > 0.0 && r.bins[0].real.exponent < -1000);
        assert_eq!(r.bins[0].imag.as_f64(), 0.0);
        if b == B::DoubleDoubleFma || b == B::DoubleDoublePortable {
            scalar = Some(r.bins.clone());
        }
        if matches!(
            b,
            B::DoubleDoubleAvx2 | B::DoubleDoubleAvx512 | B::DoubleDoubleNeon
        ) {
            assert_eq!(Some(&r.bins), scalar.as_ref());
        }
    }
}
