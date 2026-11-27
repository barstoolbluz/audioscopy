//! Native high-precision reference fixture driver (no extra dependencies).
#[cfg(feature = "precision")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use audioscopy::precision::{Backend, CorrelationAnalyzer, Precision};
    use audioscopy::{Encoding, Format};
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: precision_f64le BACKEND FILE CHUNK_FRAMES".into());
    }
    let (precision, backend) = match args[1].as_str() {
        "binary64" => (Precision::Binary64, Backend::Binary64),
        "x87" => (Precision::Extended80, Backend::X87),
        "dd" => (Precision::DoubleDouble, Backend::DoubleDouble),
        _ => return Err("unknown backend".into()),
    };
    if !backend.available() {
        return Err("requested backend unavailable".into());
    }
    let bytes = std::fs::read(&args[2])?;
    if !bytes.len().is_multiple_of(16) {
        return Err("incomplete f64le stereo frame".into());
    }
    let pcm: Vec<_> = bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|chunk| f64::from_le_bytes(*chunk))
        .collect();
    let chunk: usize = args[3].parse()?;
    if chunk == 0 || chunk > usize::MAX / 2 {
        return Err("invalid chunk size".into());
    }
    let mut analyzer = CorrelationAnalyzer::new(
        Format {
            sample_rate: 44101,
            channels: 2,
            encoding: Encoding::Float,
            loudness_roles: None,
        },
        precision,
        backend,
    )?;
    for block in pcm.chunks(chunk * 2) {
        analyzer.push_interleaved(block)?;
    }
    let r = analyzer.finish()?;
    println!(
        "RESULT {:?} {:016x} {:016x}",
        r.backend,
        r.hi.to_bits(),
        r.lo.to_bits()
    );
    Ok(())
}

#[cfg(not(feature = "precision"))]
fn main() {
    eprintln!("precision_f64le requires --features precision");
    std::process::exit(2);
}
