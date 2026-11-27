#![cfg(feature = "precision-loudness")]
use audioscopy::precision::loudness::{
    LoudnessBackend as B, LoudnessOutcome as O, WideLoudnessMeter,
};
use audioscopy::precision::Precision;
use audioscopy::{ChannelRole as R, Error};

fn precision(b: B) -> Precision {
    if b == B::X87 {
        Precision::Extended80
    } else {
        Precision::DoubleDouble
    }
}
fn backends() -> impl Iterator<Item = B> {
    [B::X87, B::DoubleDoublePortable, B::DoubleDoubleFma]
        .into_iter()
        .filter(|b| b.available())
}

#[test]
fn outcomes_chunk_clock_and_invalid_push() {
    for b in backends() {
        let mut meter =
            WideLoudnessMeter::new(8001, &[R::Left, R::Right], precision(b), b).unwrap();
        assert!(matches!(
            meter.push_interleaved(&[0.0]),
            Err(Error::IncompleteFrame)
        ));
        assert!(matches!(
            meter.push_interleaved(&[0.0, f64::NAN]),
            Err(Error::InvalidSample(1))
        ));
        assert_eq!(meter.frames(), 0);
        assert_eq!(meter.finish().unwrap().outcome, O::NoInput);
        let pcm: Vec<_> = (0..10_003)
            .flat_map(|i| {
                let x = if i % 2 == 0 { 0.25 } else { -0.25 };
                [x, -x]
            })
            .collect();
        let run = |chunk: usize| {
            let mut m =
                WideLoudnessMeter::new(8001, &[R::Left, R::Right], precision(b), b).unwrap();
            for x in pcm.chunks(chunk * 2) {
                m.push_interleaved(x).unwrap();
            }
            m.finish().unwrap()
        };
        let r = run(127);
        assert_eq!(r, run(4093));
        assert_eq!(r.outcome, O::Finite);
        assert_eq!(r.required_frames, 3200);
        assert_eq!(r.absolute_observations, 9);
        assert_eq!(r.relative_observations, 9);
        assert!(r.lufs().unwrap().is_finite());
    }
}

#[test]
fn resource_limit_and_role_admission() {
    assert!(WideLoudnessMeter::new(3363, &[R::Mono], Precision::DoubleDouble, B::Auto).is_err());
    assert!(
        WideLoudnessMeter::new(8000, &[R::Left, R::Left], Precision::DoubleDouble, B::Auto)
            .is_err()
    );
    assert!(WideLoudnessMeter::new(8000, &[R::Lfe], Precision::DoubleDouble, B::Auto).is_err());
    for b in backends() {
        for rate in [3364, 2822400] {
            let m = WideLoudnessMeter::new(rate, &[R::Mono], precision(b), b).unwrap();
            assert_eq!(m.finish().unwrap().outcome, O::NoInput);
        }
        let mut m = WideLoudnessMeter::with_observation_limit(8000, &[R::Mono], precision(b), b, 1)
            .unwrap();
        assert!(matches!(
            m.push_interleaved(&[0.5, -0.5].repeat(2500)),
            Err(Error::ResourceLimit)
        ));
        assert!(matches!(m.finish(), Err(Error::Failed)));
    }
}
