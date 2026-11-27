//! Independent-reference driver for optional wide integrated loudness.
#[cfg(feature = "precision-loudness")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use audioscopy::precision::loudness::{
        LoudnessBackend as B, LoudnessOutcome, WideLoudnessMeter,
    };
    use audioscopy::precision::Precision;
    use audioscopy::ChannelRole as R;
    use std::{fs::File, io::Read};

    let args: Vec<_> = std::env::args().skip(1).collect();
    let modes = [
        (1, B::DoubleDouble),
        (2, B::X87),
        (3, B::DoubleDoublePortable),
        (4, B::DoubleDoubleFma),
    ];
    if args.len() == 1 && args[0] == "--available" {
        for (number, backend) in modes {
            if backend.available() {
                print!("{number} ");
            }
        }
        println!();
        return Ok(());
    }
    if args.len() != 5 {
        return Err("usage: precision_loudness_f64le MODE RATE WEIGHTS CHUNK FILE".into());
    }
    let mode: i32 = args[0].parse()?;
    let backend = modes
        .into_iter()
        .find(|&(m, _)| m == mode)
        .ok_or("invalid mode")?
        .1;
    let precision = if backend == B::X87 {
        Precision::Extended80
    } else {
        Precision::DoubleDouble
    };
    let rate = args[1].parse()?;
    let mut units = [R::Left, R::Right, R::Center].into_iter();
    let mut surrounds = [R::LeftSurround, R::RightSurround, R::LeftBack, R::RightBack].into_iter();
    let roles: Vec<_> = args[2]
        .split(',')
        .map(|w| match w {
            "0" => Ok(R::Lfe),
            "1" => units.next().ok_or("too many unit roles"),
            "2" => surrounds.next().ok_or("too many surround roles"),
            _ => Err("invalid role weight"),
        })
        .collect::<Result<_, _>>()?;
    let channels = roles.len();
    let chunk: usize = args[3].parse()?;
    if !(1..=1_048_576).contains(&chunk) {
        return Err("invalid chunk size".into());
    }
    let mut meter = WideLoudnessMeter::new(rate, &roles, precision, backend)?;
    let mut file = File::open(&args[4])?;
    let mut bytes = vec![0_u8; chunk * channels * 8];
    let mut pcm = Vec::with_capacity(chunk * channels);
    loop {
        let mut used = 0;
        while used < bytes.len() {
            let got = file.read(&mut bytes[used..])?;
            if got == 0 {
                break;
            }
            used += got;
        }
        if used == 0 {
            break;
        }
        if !used.is_multiple_of(8 * channels) {
            return Err("incomplete frame in input".into());
        }
        pcm.clear();
        for value in bytes[..used].as_chunks::<8>().0 {
            pcm.push(f64::from_le_bytes(*value));
        }
        meter.push_interleaved(&pcm)?;
    }
    let r = meter.finish()?;
    let actual = match r.backend {
        B::X87 => 2,
        B::DoubleDoublePortable => 3,
        B::DoubleDoubleFma => 4,
        _ => return Err("unresolved backend".into()),
    };
    let outcome = match r.outcome {
        LoudnessOutcome::NoInput => 0,
        LoudnessOutcome::InsufficientFrames => 1,
        LoudnessOutcome::BelowAbsoluteGate => 2,
        LoudnessOutcome::BelowRelativeGate => 3,
        LoudnessOutcome::Finite => 4,
    };
    let b = r.energy.unwrap_or_default();
    println!(
        "LOUDNESS {actual} {} {} {outcome} {} {} {:016x} {:016x} {}",
        r.frames,
        r.required_frames,
        r.absolute_observations,
        r.relative_observations,
        b.hi.to_bits(),
        b.lo.to_bits(),
        b.exponent
    );
    Ok(())
}

#[cfg(not(feature = "precision-loudness"))]
fn main() {
    eprintln!("requires precision-loudness");
    std::process::exit(2);
}
