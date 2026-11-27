//! Paired end-to-end public correlation timings, fixture generation excluded.
#[cfg(feature = "precision")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use audioscopy::precision::{Backend, CorrelationAnalyzer, Precision};
    use audioscopy::{Encoding, Format};
    use std::hint::black_box;
    use std::time::Instant;
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: precision_profile BACKEND RATE SECONDS".into());
    }
    let (precision, backend) = match args[1].as_str() {
        "binary64" => (Precision::Binary64, Backend::Binary64),
        "x87" => (Precision::Extended80, Backend::X87),
        "dd" => (Precision::DoubleDouble, Backend::DoubleDouble),
        _ => return Err("unknown backend".into()),
    };
    let rate: u32 = args[2].parse()?;
    let seconds: usize = args[3].parse()?;
    if !(8000..=768000).contains(&rate) || seconds == 0 {
        return Err("invalid geometry".into());
    }
    let mut seed = 0x91ae_d40d_8b23_u64;
    let pcm: Vec<_> = (0..2 * rate as usize)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            ((seed & 0xff_ffff) as f64 - 8388608.0) / 8388608.0
        })
        .collect();
    let start = Instant::now();
    let mut a = CorrelationAnalyzer::new(
        Format {
            sample_rate: rate,
            channels: 2,
            encoding: Encoding::Integer(24),
            loudness_roles: None,
        },
        precision,
        backend,
    )?;
    let init = start.elapsed();
    let start = Instant::now();
    for _ in 0..seconds {
        for chunk in pcm.chunks(4093 * 2) {
            a.push_interleaved(black_box(chunk))?;
        }
    }
    let ingest = start.elapsed();
    let start = Instant::now();
    let r = black_box(a.finish()?);
    let finish = start.elapsed();
    let wall = init + ingest + finish;
    println!(
        "PRECISION_PROFILE backend={:?} rate={} seconds={} init_s={:.9} ingest_s={:.9} finish_s={:.9} wall_s={:.9} hi={:016x} lo={:016x}",
        r.backend,
        rate,
        seconds,
        init.as_secs_f64(),
        ingest.as_secs_f64(),
        finish.as_secs_f64(),
        wall.as_secs_f64(),
        r.hi.to_bits(),
        r.lo.to_bits()
    );
    Ok(())
}

#[cfg(not(feature = "precision"))]
fn main() {
    eprintln!("precision_profile requires --features precision");
    std::process::exit(2);
}
