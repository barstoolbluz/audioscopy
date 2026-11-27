//! Regression tests for the optional wide-spectrum public interface.
#![cfg(feature = "precision-spectrum")]
use audioscopy::precision::spectral::{PowerSpectrum, SpectrumAnalyzer, SpectrumBackend as B};
use audioscopy::precision::Precision;
use audioscopy::{Encoding, Error, Format};

fn format(rate: u32, channels: usize) -> Format {
    Format {
        sample_rate: rate,
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

fn analyzer(f: Format, b: B) -> SpectrumAnalyzer {
    let precision = if b == B::X87 {
        Precision::Extended80
    } else {
        Precision::DoubleDouble
    };
    SpectrumAnalyzer::new(f, precision, b).unwrap()
}

fn calculate(f: Format, b: B, pcm: &[f64], chunk: usize) -> PowerSpectrum {
    let channels = f.channels;
    let mut a = analyzer(f, b);
    for block in pcm.chunks(chunk * channels) {
        a.push_interleaved(block).unwrap();
    }
    a.finish().unwrap()
}

#[test]
fn admission_and_transactional_push() {
    let f = format(8000, 2);
    assert!(SpectrumAnalyzer::new(f.clone(), Precision::Binary64, B::Auto).is_err());
    assert!(SpectrumAnalyzer::new(f.clone(), Precision::DoubleDouble, B::X87).is_err());
    if !B::DoubleDouble.available() {
        assert!(SpectrumAnalyzer::new(f, Precision::DoubleDouble, B::Auto).is_err());
        return;
    }
    let a = SpectrumAnalyzer::new(f.clone(), Precision::DoubleDouble, B::Auto).unwrap();
    assert!(matches!(a.finish(), Err(Error::Empty)));
    for b in backends() {
        let mut a = analyzer(f.clone(), b);
        assert_eq!(a.backend(), b);
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
        a.push_interleaved(&[]).unwrap();
        let p = a.finish().unwrap();
        assert_eq!(p.frames, 1);
        assert_eq!(p.fft_len, 2048);
        assert_eq!(p.bins.len(), 1025);
        let energy: f64 = p.bins.iter().map(|x| x.as_f64()).sum();
        assert!((energy - 0.15625).abs() < 1.0e-14);
    }
}

#[test]
fn chunk_boundaries_parseval_and_vector_parity() {
    let f = format(12001, 3);
    let pcm: Vec<_> = (0..5079)
        .map(|i| f64::from((i * 79 % 2048) - 1024) / 2048.0)
        .collect();
    let expected = pcm.iter().map(|x| x * x).sum::<f64>() / pcm.len() as f64;
    let mut fma = None;
    for b in backends() {
        let p = calculate(f.clone(), b, &pcm, 127);
        assert_eq!(p, calculate(f.clone(), b, &pcm, 4093));
        assert_eq!(p.frames, 1693);
        assert_eq!(p.fft_len, 2048);
        let energy: f64 = p.bins.iter().map(|x| x.as_f64()).sum();
        assert!((energy - expected).abs() < 1.0e-14);
        if b == B::DoubleDoubleFma || b == B::DoubleDoublePortable {
            fma = Some(p.bins.clone());
        }
        if matches!(
            b,
            B::DoubleDoubleAvx2 | B::DoubleDoubleAvx512 | B::DoubleDoubleNeon
        ) {
            assert_eq!(Some(&p.bins), fma.as_ref());
        }
    }
}

#[test]
fn scaled_subnormal_power_and_silence() {
    for b in backends() {
        let f = format(8000, 1);
        let p = calculate(f.clone(), b, &[f64::from_bits(1); 19], 7);
        assert!(p.bins[0].hi > 0.0);
        assert!(p.bins[0].exponent < -2000);
        assert!(p.bins[0].dbfs().is_finite());
        assert_eq!(p.bins[0].as_f64(), 0.0);
        let z = calculate(f, b, &[0.0; 1027], 51);
        assert!(z
            .bins
            .iter()
            .all(|x| x.hi == 0.0 && x.lo == 0.0 && x.exponent == 0));
    }
}

#[test]
fn independent_analyzers_can_move_between_threads() {
    for b in backends() {
        let f = format(8000, 2);
        let pcm = [0.25, -0.5].repeat(1111);
        let expected = calculate(f.clone(), b, &pcm, 1111);
        let mut a = analyzer(f, b);
        a.push_interleaved(&pcm[..22]).unwrap();
        let actual = std::thread::spawn(move || {
            a.push_interleaved(&pcm[22..]).unwrap();
            a.finish().unwrap()
        })
        .join()
        .unwrap();
        assert_eq!(actual, expected);
    }
}
