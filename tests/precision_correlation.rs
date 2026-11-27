//! Public high-precision correlation qualification; exact-input references.
#![cfg(feature = "precision")]
use audioscopy::precision::{Backend, Correlation, CorrelationAnalyzer, Precision};
use audioscopy::{Encoding, Error, Format};

fn format(encoding: Encoding) -> Format {
    Format {
        sample_rate: 44101,
        channels: 2,
        encoding,
        loudness_roles: None,
    }
}

fn measure(pcm: &[f64], encoding: Encoding, backend: Backend, chunk: usize) -> Correlation {
    let precision = match backend {
        Backend::Binary64 => Precision::Binary64,
        Backend::DoubleDouble => Precision::DoubleDouble,
        _ => Precision::Extended80,
    };
    let mut meter = CorrelationAnalyzer::new(format(encoding), precision, backend).unwrap();
    for block in pcm.chunks(2 * chunk) {
        meter.push_interleaved(block).unwrap();
    }
    assert_eq!(meter.frames(), (pcm.len() / 2) as u64);
    let out = meter.finish().unwrap();
    assert_eq!(out.backend, backend);
    out
}

fn high_backends() -> impl Iterator<Item = Backend> {
    [Backend::X87, Backend::DoubleDouble]
        .into_iter()
        .filter(|b| b.available())
}

fn check_ratio(r: Correlation) {
    // Exact 5/12 = high + low + a residual below 2^-109.
    let hi = f64::from_bits(0x3fda_aaaa_aaaa_aaab);
    let lo = f64::from_bits(0xbc75_5555_5555_5555);
    let error = ((r.hi - hi) + (r.lo - lo)).abs();
    let tolerance = if r.backend == Backend::X87 {
        2e-17
    } else {
        2e-29
    };
    assert!(error < tolerance, "{r:?}, error {error:e}");
}

#[test]
fn precision_selection_and_transactional_validation() {
    for (precision, backend) in [
        (Precision::Extended80, Backend::Binary64),
        (Precision::DoubleDouble, Backend::X87),
        (Precision::Binary64, Backend::DoubleDouble),
    ] {
        assert!(CorrelationAnalyzer::new(format(Encoding::Float), precision, backend).is_err());
    }
    if !Backend::X87.available() {
        assert!(CorrelationAnalyzer::new(
            format(Encoding::Float),
            Precision::Extended80,
            Backend::X87,
        )
        .is_err());
    }
    let auto = CorrelationAnalyzer::new(
        format(Encoding::Float),
        Precision::Extended80,
        Backend::Auto,
    )
    .unwrap();
    assert_eq!(
        auto.backend(),
        if Backend::X87.available() {
            Backend::X87
        } else {
            Backend::DoubleDouble
        }
    );
    assert!(matches!(auto.finish(), Err(Error::Empty)));
    let mut mono = format(Encoding::Float);
    mono.channels = 1;
    assert!(CorrelationAnalyzer::new(mono, Precision::Extended80, Backend::Auto).is_err());
    for backend in high_backends() {
        let mut a = CorrelationAnalyzer::new(
            format(Encoding::Integer(24)),
            Precision::Extended80,
            backend,
        )
        .unwrap();
        a.push_interleaved(&[0.25, -0.5]).unwrap();
        assert!(matches!(
            a.push_interleaved(&[0.0]),
            Err(Error::IncompleteFrame)
        ));
        assert!(matches!(
            a.push_interleaved(&[0.0, f64::NAN]),
            Err(Error::InvalidSample(1))
        ));
        assert!(matches!(
            a.push_interleaved(&[0.0, 1.0]),
            Err(Error::InvalidSample(1))
        ));
        assert_eq!(a.frames(), 1);
        a.push_interleaved(&[]).unwrap();
        assert_eq!(a.finish().unwrap().hi, 0.0);
    }
}

#[test]
fn exact_pcm_and_one_ulp_chunk_invariance() {
    let pattern = [[0, 0], [1, 0], [1, 1], [0, 1], [1, 1], [1, 1], [0, 0]];
    for bits in [16, 24, 32, 54] {
        let delta = 2.0_f64.powi(1 - bits);
        let encoding = if bits == 54 {
            Encoding::Float
        } else {
            Encoding::Integer(bits as u8)
        };
        let pcm: Vec<_> = (0..70_000)
            .flat_map(|i| pattern[i % 7].map(|v| 0.5 + f64::from(v) * delta))
            .collect();
        for backend in high_backends() {
            let whole = measure(&pcm, encoding, backend, pcm.len() / 2);
            check_ratio(whole);
            for chunk in [1, 7, 1023, 1024, 1025, 4093] {
                assert_eq!(whole, measure(&pcm, encoding, backend, chunk));
            }
            eprintln!("PRECISION_CORRELATION_EXECUTED {backend:?} bits={bits}");
        }
    }
}

#[test]
fn subnormal_channels_and_extreme_scale_ratio() {
    let pattern = [[0, 0], [1, 0], [1, 1], [0, 1], [1, 1], [1, 1], [0, 0]];
    for scale in [f64::from_bits(1), 2.0_f64.powi(-1000), 2.0_f64.powi(-600)] {
        for large_right in [false, true] {
            let pcm: Vec<_> = (0..7000)
                .flat_map(|i| {
                    let [l, r] = pattern[i % 7];
                    [
                        f64::from(1 + l) * scale,
                        if large_right {
                            f64::from(r) - 0.5
                        } else {
                            f64::from(1 + r) * scale
                        },
                    ]
                })
                .collect();
            for backend in high_backends() {
                check_ratio(measure(&pcm, Encoding::Float, backend, 37));
            }
        }
    }
}

#[test]
fn result_limbs_retain_sub_binary64_correlation_gap() {
    for sign in [-1.0, 1.0] {
        let pcm: Vec<_> = (0..65_540)
            .flat_map(|i| {
                let a = if i % 4 < 2 { -0.5 } else { 0.5 };
                let b = if i % 2 == 0 { -0.5 } else { 0.5 };
                [a, sign * a + 2.0_f64.powi(-48) * b]
            })
            .collect();
        let r = measure(&pcm, Encoding::Float, Backend::DoubleDouble, 1025);
        assert_eq!(r.hi, sign);
        // Exact r = sign / sqrt(1 + 2^-96). First omitted Taylor
        // term < 2^-190; observed DD error is much smaller than this gate.
        let gap = -sign * 2.0_f64.powi(-97);
        assert!(((r.lo - gap) / gap).abs() < 1e-3, "{r:?}");
    }
}

#[test]
fn independent_parallel_instances() {
    let pcm: Vec<_> = (0..4097)
        .flat_map(|i| [(i % 31) as f64 / 32.0, (i % 19) as f64 / 32.0])
        .collect();
    for backend in high_backends() {
        let expected = measure(&pcm, Encoding::Float, backend, 4097);
        std::thread::scope(|s| {
            let handles: Vec<_> = [1, 7, 1023, 2049]
                .into_iter()
                .map(|chunk| {
                    let pcm = &pcm;
                    s.spawn(move || measure(pcm, Encoding::Float, backend, chunk))
                })
                .collect();
            for h in handles {
                assert_eq!(h.join().unwrap(), expected);
            }
        });
    }
}
