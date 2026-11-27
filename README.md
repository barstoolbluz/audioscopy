# Audioscopy

**A streaming audio measurement library for Rust.**

Audioscopy measures the properties of decoded digital audio, from loudness and dynamics to spectral content, noise, effective resolution and stereo relationships. It is designed for applications that need detailed, repeatable measurements of individual recordings or entire albums, including music library managers, mastering and restoration software, audio conversion utilities, and automated quality-control systems.

Audioscopy can help answer questions that file metadata alone cannot. Does a 24-bit recording use all 24 bits? Is there appreciable signal above 20 kHz? Does a high-resolution file contain evidence of a lower source sample rate? How much noise is present during quiet passages? How does the stereo image behave when summed to mono?

The library operates on decoded PCM. Applications supply samples and receive structured measurements, including numerical data suitable for plotting. Audioscopy is independent of audio codecs, file formats, playback engines and user interfaces. It can analyze the same PCM regardless of whether it came from a FLAC file, a WAV file, a decoder, or a generated signal.

## What Audioscopy measures

Audioscopy currently provides 25 selectable measurements for tracks and albums.

| Category | Measurements |
| --- | --- |
| Loudness | Loudness range (LRA), maximum short-term loudness, loudness over time |
| Dynamics | Limiting, flat-topped peaks |
| Resolution | Bits in use, estimated effective bits |
| Spectral content | Likely source sample rate, lossy coding patterns, occupied bandwidth, proportion of power above 20 kHz, average power spectrum |
| Noise and artifacts | Noise floor, dynamic range, noise spectrum, hum and idle tones, noise by frequency band, clicks per minute, dither classification |
| Stereo | Correlation, channel balance, stereo width, mono compatibility, inter-channel delay, bass stereo width |

Loudness measurements use ITU-R BS.1770 K-weighting and EBU Tech 3342 loudness range. The short-term measurements use complete three-second windows at 100 ms intervals. The companion `true-peak` crate supplies the native loudness implementation; it also provides standalone peak and loudness measurement APIs.

Several measurements deserve particular explanation:

- **Bits in use** examines exact PCM sample codes and their observed integer lattice. It can identify unused low bits and other evidence of reduced resolution in a higher-bit-depth representation.
- **Effective bits** estimates resolution from measured quiet-passage noise. This is distinct from the exact integer-grid measurement and from the nominal bit depth of the file.
- **Noise floor and dynamic range** use qualifying quiet passages in the recording. Dynamic range is the difference between its sample peak and measured quiet-passage floor.
- **Bandwidth and source-rate analysis** examine occupied spectral content and characteristic cutoffs. These can help identify potentially upsampled audio.
- **Stereo analysis** measures relationships between channels, including level differences, cancellation when summed to mono, relative delay and the frequency below which the bass is effectively mono.

Lossy coding, likely source rate, dither, limiting and click detection are evidence-based estimates or heuristics. They cannot establish a recording's mastering history, prove that it was losslessly encoded, or determine its subjective sound quality. Audioscopy reports insufficient evidence when a measurement cannot support a result.

[docs/measurements.md](docs/measurements.md) gives the exact definitions, thresholds and limitations of each measurement.

## Design and performance

Audioscopy is intended for high-resolution audio and large collections. It accepts interleaved binary64 (`f64`) PCM at sample rates from 8 kHz to 768 kHz, with 1–64 channels. Integer PCM from 1–32 bits and floating-point PCM are supported.

- **Streaming analysis.** Supply samples in chunks of any whole-frame length. Measurement state persists across chunk boundaries. A complete recording need never reside in memory.
- **Selective computation.** Request only the measurements you need. Measurements share common computations where possible; unrelated FFTs, filters and observation histories are omitted.
- **Numerical fidelity.** Conditioned K-weighting, numerically stable stereo statistics, compensated loudness energy accumulation and power-preserving spectral normalization reduce arithmetic errors without changing the intended measurements.
- **CPU-specific acceleration.** Native scalar and SIMD paths support x86-64 and AArch64, including SSE2/AVX loudness filtering, wider AVX2/AVX-512 spectral kernels where enabled, and binary64 NEON loudness filtering. RustFFT supplies architecture-specific FFT acceleration. Optional ISA paths use runtime feature detection and retain portable fallbacks.
- **Bounded PCM storage.** FFT buffers, quiet-passage detection and loudness energy cells have bounded working storage. Retained time series, spectra and album reports consume memory according to the selected measurements and input geometry. Resource limits constrain observation histories.

### How fast is it?

Representative native qualification measurements for the optimized binary64 implementation on a single logical CPU of an AMD Ryzen Threadripper 7970X:

| 60 seconds of stereo, 24-bit PCM | 48 kHz | 768 kHz |
| --- | ---: | ---: |
| All 25 measurements | 243x realtime | 13x realtime |
| Loudness range, short-term maximum and time series | 1,117x realtime | 70x realtime |

These are median results from five runs using Rust 1.99.1 with portable code generation. The deterministic fixture alternates active signals with quiet passages. Timings include analyzer initialization, ingestion, finalization and reading the selected results; they exclude decoding, disk I/O and fixture generation. Actual throughput depends on the hardware, selected measurements and audio content.

[docs/performance.md](docs/performance.md) describes the method, gives more results and documents memory use.

## Getting started

Audioscopy is a library. Obtain decoded PCM from your preferred audio decoder and pass the samples to an `Analyzer`.

```rust
use audioscopy::{
    Analyzer, Encoding, Error, Format, Measurement as M, Measurements,
    NoiseView, Outcome, Settings, Track, Value,
};

fn analyze_stereo(pcm: &[f64]) -> Result<Track, Error> {
    let format = Format {
        sample_rate: 96_000,
        channels: 2,
        encoding: Encoding::Float,
        loudness_roles: None,
    };

    let selection = Measurements::new(&[
        M::LoudnessRange,
        M::NoiseFloor,
        M::AverageSpectrum,
        M::Correlation,
    ]);

    let mut analyzer = Analyzer::new(format, selection, Settings::default())?;

    // Process up to 4,096 stereo frames at a time.
    // pcm contains decoded, interleaved left/right samples.
    for chunk in pcm.chunks(4096 * 2) {
        analyzer.push_interleaved(chunk)?;
    }

    let report = analyzer.finish()?;

    if let Outcome::Value(Value::Scalar(lu)) =
        report.measure(M::LoudnessRange, NoiseView::default())
    {
        println!("Loudness range: {lu:.1} LU");
    }

    Ok(report)
}
```

Use `Measurements::all()` to request every measurement. `Measurements::default()` is empty, so callers choose their own defaults.

### PCM requirements

For signed integer PCM with `b` bits, normalize each original signed code by `2^(b-1)` and declare `Encoding::Integer(b)`. For example, 24-bit PCM divides by 8,388,608. This preserves the original integer grid, which is essential for exact resolution analysis. Avoid peak normalization, dithering or requantization before measurement unless those transformations are themselves part of what you intend to analyze.

Floating-point PCM uses `Encoding::Float`; finite overs above nominal full scale are accepted up to an absolute magnitude of 16. NaN and infinite input samples are rejected. Multichannel loudness measurements require explicit channel roles for layouts with more than two channels. Stereo measurements apply to exactly two channels.

### Albums

Use `AlbumAnalyzer` to analyze an ordered sequence of tracks. Call `begin_track(format)`, supply PCM with `push_interleaved`, then call `end_track` for each track. `finish` returns an `Album` with individual track reports and album-level results.

Album measurements follow their own aggregation rules. For example, spectral and noise powers are pooled in linear units using real duration. Loudness range is measured over the continuous programme, including observation windows that cross track boundaries when the sample rate and channel roles remain compatible.

`Album::from_tracks` can combine existing track reports with matching measurement selections and settings. Independent reports cannot reconstruct continuous loudness windows across tracks. In that case, album LRA is unavailable, while other applicable album results remain accessible.

### Noise analysis and reusable results

The quiet-passage detector uses configurable level and duration thresholds. The defaults are -60 dBFS RMS and one second of contiguous quiet audio.

Noise measurements support unweighted, A-weighted and ITU-R 468 frequency weighting (RMS, rather than quasi-peak), with selectable audible-band, Nyquist-limited and full-band views. Measurements can be recalculated from the retained quiet spectrum without decoding or performing another FFT.

A track's `noise_record()` exports versioned sufficient statistics for the selected noise measurements. Applications can persist these in their preferred cache format and restore them with `NoiseSnapshot::from_record`. Changing the quiet-passage detector settings requires a new analysis.

### Interpreting results

Every measurement returns a typed `Outcome`: `Value`, `NotRequested`, `NotApplicable` or `CantTell(reason)`. A short recording may lack a complete loudness window; a recording may contain no qualifying quiet passages; a stereo delay may be ambiguous. These cases are distinct from measured zero.

Spectrum curves report one-sided power density in dBFS/Hz, and loudness curves report LUFS against time. The library returns numerical curve data for applications to display or process. It does not render graphs.

## Where Audioscopy is going

The goal is a general-purpose library for detailed audio measurement, including demanding analytical and metrological applications.

Planned areas of development include:

- **Comparative analysis.** Align recordings in time and gain, estimate sample-rate and speed differences, detect polarity inversion and channel swaps, measure null depth and residual energy, compare tonal balance, and quantify timing drift. This will support comparisons of different masters, encodes, transfers and processing chains.
- **Optional high-precision measurements.** Provide x87 extended-precision paths on x86-64 and portable higher-precision alternatives, such as double-double arithmetic, where independent numerical tests establish a benefit. Callers will be able to choose higher precision for cancellation-sensitive and extreme-dynamic-range measurements while retaining the fast binary64 implementation as the default.
- **Known-signal measurements.** Analyze distortion (THD, THD+N, IMD and individual harmonics), frequency response, crosstalk, linearity, phase, group delay, impulse response, and aliasing and imaging rejection when suitable reference or test signals are available.
- **Further architecture optimization.** Continue investigating SIMD and FMA on x86-64 and AArch64 where measured application performance or numerical fidelity justifies the additional implementation.

The current release analyzes PCM from a single recording or an ordered album. Direct DSD-domain analysis, reference-recording comparison and known-test-signal analysis are not yet implemented. Audio decoding and presentation remain the responsibility of the caller.

## Building from source

Audioscopy requires Rust 1.95 or later. The optional Nix development environment pins the qualification toolchain to Rust 1.99.1.

During development, place the `audioscopy` and `true-peak` source trees beside one another. From the `audioscopy` directory:

```sh
cargo test --release --no-fail-fast
cargo doc --no-deps
cargo run --release --example benchmark -- 60 96000 2
```

To reproduce a workload from the performance table:

```sh
cargo run --release --example numerical_profile -- all 48000 2 60
```

Use `nix develop` before the Cargo commands if you prefer the pinned development environment.

Audioscopy depends directly on `true-peak` for loudness measurements and [RustFFT](https://github.com/ejmahler/RustFFT) for FFT computation. Once both crates are published, applications can depend on Audioscopy directly from the Cargo registry.

## License

audioscopy is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free
Software Foundation, either version 2 of the License, or (at your option)
any later version. The full text is in [COPYING](COPYING).

It is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE.
