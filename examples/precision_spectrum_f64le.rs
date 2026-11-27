//! External exact-input fixture driver for the wide-spectrum public API.
#[cfg(feature = "precision-spectrum")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use audioscopy::precision::spectral::{SpectrumAnalyzer, SpectrumBackend as B};
    use audioscopy::precision::Precision;
    use audioscopy::{Encoding, Format};
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
    if args.len() != 5 {
        return Err("usage: precision_spectrum_f64le MODE RATE CHANNELS CHUNK FILE".into());
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
    let channels = args[2].parse::<usize>()?;
    let chunk = args[3].parse::<usize>()?;
    if !(1..=1_048_576).contains(&chunk) {
        return Err("invalid chunk size".into());
    }
    let mut analyzer = SpectrumAnalyzer::new(
        Format {
            sample_rate: rate,
            channels,
            encoding: Encoding::Float,
            loudness_roles: None,
        },
        precision,
        backend,
    )?;
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
        analyzer.push_interleaved(&pcm)?;
    }
    let result = analyzer.finish()?;
    let actual = match result.backend {
        B::X87 => 2,
        B::DoubleDoublePortable => 3,
        B::DoubleDoubleFma => 4,
        B::DoubleDoubleAvx2 => 6,
        B::DoubleDoubleAvx512 => 7,
        B::DoubleDoubleNeon => 8,
        _ => return Err("unresolved backend".into()),
    };
    println!("SPECTRUM {actual} {} {}", result.fft_len, result.frames);
    for b in result.bins {
        println!(
            "{:016x} {:016x} {}",
            b.hi.to_bits(),
            b.lo.to_bits(),
            b.exponent
        );
    }
    Ok(())
}

#[cfg(not(feature = "precision-spectrum"))]
fn main() {
    eprintln!("requires precision-spectrum");
    std::process::exit(2);
}
