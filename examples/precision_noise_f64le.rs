//! External independent-reference driver for explicit quiet passages.
#[cfg(feature = "precision-spectrum")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use audioscopy::precision::spectral::{QuietNoiseAnalyzer, SpectrumBackend as B};
    use audioscopy::precision::Precision;
    use audioscopy::{Band, Encoding, Format, NoiseView, Weighting};
    use std::{fs::File, io::Read};

    let args: Vec<_> = std::env::args().skip(1).collect();
    let modes = [
        (1, B::DoubleDouble),
        (2, B::X87),
        (3, B::DoubleDoublePortable),
        (4, B::DoubleDoubleFma),
        (6, B::DoubleDoubleAvx2),
        (7, B::DoubleDoubleAvx512),
        (8, B::DoubleDoubleNeon),
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
    if args.len() < 7 {
        return Err(
            "usage: precision_noise_f64le MODE RATE CHANNELS VIEW WEIGHT CHUNK FILE...".into(),
        );
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
    let channels: usize = args[2].parse()?;
    let view = NoiseView {
        band: match args[3].as_str() {
            "0" => Band::Audible,
            "1" => Band::ToNyquist,
            "2" => Band::Full,
            _ => return Err("invalid band".into()),
        },
        weighting: match args[4].as_str() {
            "0" => Weighting::Unweighted,
            "1" => Weighting::A,
            "2" => Weighting::ItuR468,
            _ => return Err("invalid weighting".into()),
        },
    };
    let chunk: usize = args[5].parse()?;
    if !(1..=1_048_576).contains(&chunk) {
        return Err("invalid chunk size".into());
    }
    let mut analyzer = QuietNoiseAnalyzer::new(
        Format {
            sample_rate: rate,
            channels,
            encoding: Encoding::Float,
            loudness_roles: None,
        },
        0.05,
        view,
        precision,
        backend,
    )?;
    let mut bytes = vec![0_u8; chunk * channels * 8];
    let mut pcm = Vec::with_capacity(chunk * channels);
    for path in &args[6..] {
        let mut file = File::open(path)?;
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
            analyzer.push_interleaved(&pcm)?;
        }
        analyzer.end_passage()?;
    }
    let result = analyzer.finish()?.ok_or("no eligible passage")?;
    let actual = match result.backend {
        B::X87 => 2,
        B::DoubleDoublePortable => 3,
        B::DoubleDoubleFma => 4,
        B::DoubleDoubleAvx2 => 6,
        B::DoubleDoubleAvx512 => 7,
        B::DoubleDoubleNeon => 8,
        _ => return Err("unresolved backend".into()),
    };
    println!("NOISE {actual} {}", result.quiet_frames);
    for b in std::iter::once(Some(result.floor)).chain(result.bands) {
        if let Some(b) = b {
            println!(
                "{:016x} {:016x} {}",
                b.hi.to_bits(),
                b.lo.to_bits(),
                b.exponent
            );
        } else {
            println!("na");
        }
    }
    Ok(())
}

#[cfg(not(feature = "precision-spectrum"))]
fn main() {
    eprintln!("requires precision-spectrum");
    std::process::exit(2);
}
