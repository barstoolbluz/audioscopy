# audioscopy

Streaming measurements of decoded PCM, independent of any player, editor,
container format or decoder. Implements the 25 file/album measurements in
TonePoet Advanced Analysis measurement inventory revision 2, sections 3.1–3.6.
The crate does not perform file I/O, decode audio, render graphs, or compare
versions of recordings. DSD and test-signal analysis are outside this release.

**Round-1 status:** implementation and qualification fixtures supplied. The
original delivery environment had no Rust compiler or Cargo and could not
resolve the Rust distribution host. Rust compilation, execution and performance
qualification are outstanding; see `../QUALIFICATION.md` and
`qualification/REFERENCE_RESULTS.txt`. Do not treat the numerical Python
reference checks as execution of this crate.

## Use

The input is interleaved `f64`, normalized so signed integer full scale is
`-1.0` through `1.0 - 2^(1-bits)`. An integer format requires exact normalized
codes: do not scale by `i16::MAX`, round float samples, or dither in the decoder
before measurement. Float PCM is accepted as delivered, including finite
overs up to 16.0 in magnitude. Supported PCM rates: 8–768 kHz; channels: 1–64;
integer depths: 1–32. The caller supplies explicit loudness roles for >2 channels
when selecting loudness. Stereo measures apply only to exactly two channels.

```rust
use audioscopy::{Analyzer, Band, Encoding, Format, Measurement,
                 Measurements, NoiseSnapshot, NoiseView, Outcome, Settings,
                 Value, Weighting};

fn main() -> Result<(), audioscopy::Error> {
    let format = Format {
        sample_rate: 48_000,
        channels: 1,
        encoding: Encoding::Integer(16),
        loudness_roles: None,
    };
    let selection = Measurements::new(&[
        Measurement::BitsInUse, Measurement::NoiseFloor,
    ]);
    let mut analyzer = Analyzer::new(format, selection, Settings::default())?;
    // One second of quantized low-level PCM, supplied in decoder-sized chunks.
    let pcm: Vec<f64> = (0..48_000)
        .map(|i| if i % 48 < 24 { 1.0 / 32768.0 } else { -1.0 / 32768.0 })
        .collect();
    for chunk in pcm.chunks(997) {
        analyzer.push_interleaved(chunk)?;
    }
    let report = analyzer.finish()?;
    let full = NoiseView { weighting: Weighting::Unweighted, band: Band::Full };
    if let Outcome::Value(Value::Scalar(dbfs)) =
        report.measure(Measurement::NoiseFloor, full)
    {
        assert!((dbfs - 20.0 * (1.0_f64 / 32768.0).log10()).abs() < 1e-9);
    } else {
        panic!("expected a qualifying quiet passage");
    }
    // Persist this plain versioned record using the caller's cache format.
    let saved = report.noise_record();
    drop(report);
    drop(pcm);
    let restored = NoiseSnapshot::from_record(saved)?;
    let weighted = restored.measure(Measurement::NoiseFloor, NoiseView {
        weighting: Weighting::A,
        band: Band::Audible,
    });
    assert!(matches!(weighted, Outcome::Value(Value::Scalar(_))));
    Ok(())
}
```

`Measurements::new` selects individual rows; `Measurements::all` selects the
full inventory subset. `Measurements::default` is empty (UI defaults belong to
the consumer). Requesting a subset avoids the other branches' DSP and storage;
shared prerequisites, such as spectra for several noise rows, are calculated
once per required stream. The upstream existing LRA meter also computes
integrated loudness internally. If both LRA and short-term graph/max are
selected, two native K-weighting paths run; no integrated/LRA work is started
for short-term-only requests. Chunk boundaries do not reset analysis. Malformed
pushes are validated before state changes; a terminal processing error requires
starting a new analyzer. `finish` consumes the analyzer. It rejects zero frames.

### Album

`AlbumAnalyzer::new(selection, settings)` → `begin_track(format)` →
`push_interleaved` (zero or more valid pushes) → `end_track` for each track in
order → `finish`. Each track is decoded once. `Album::tracks()` exposes ordered
track results. `Album::measure` applies the individual album reduction rules.

`Album::from_tracks(Vec<Track>)` also accepts individually computed results
with identical selections and detector settings. When using `AlbumAnalyzer`,
LRA follows a continuous programme clock/filter across same-rate/role tracks.
When rate/roles change, or when starting from multiple independent results,
LRA is `Outcome::CantTell` with a reason. A single cached track already is
one complete programme. `Album::loudness_method()` identifies continuous,
unavailable or unrequested LRA. Other album measurements remain available.
Independent statistics cannot reconstruct boundary windows; no PCM resampler
or artificial per-track tail is used in the album measurement.

Unavailable rows contribute no evidence to eligible-track modes and medians;
if none apply, the album returns the corresponding typed unavailable outcome.
For power-based pooling, qualifying quiet time or programme time is the weight,
not track count, sample count across mismatched sample rates, or a mean in dB.

### Outcomes and curves

`Outcome::NotRequested`, `Outcome::NotApplicable`, `Outcome::CantTell(reason)`
and `Outcome::Value(value)` are distinct. The first three are never numeric.
Exact zero power has a mathematically defined level of negative infinity;
positive infinity may describe an exact zero denominator. The API does not
use NaN as a display sentinel. Zero/zero stereo energy is defined as 0 dB;
zero-variance correlation is 0. A maximum over no full three-second window,
LRA with no eligible observations, and resolution/dynamic range unbounded by
pure silence carry a reason rather than a fabricated finite result.

`Value::Curve` is ordered numerical data; `Value::Curves` holds per-track
loudness curves for an album. Spectrum curves are one-sided power density in
dBFS/Hz with frequency in Hz. Loudness points use exclusive end time in seconds
and LUFS. Album spectrum bins map to the finest native grid by linear energy
splatting. No graph plotting is part of the crate.

### Re-reading noise

`Track::noise_record()` exports only selected noise rows' sufficient statistics
as a `NoiseRecord` (schema version 2). The consumer may persist its plain fields
in any cache format. Version 1 records must be regenerated from PCM: their
frequency distribution used a different, incorrect normalization. `NoiseSnapshot::from_record` validates masks, geometry,
frames, power, thresholds and numeric values before accepting them. Its
`measure` and `measure_album` reuse the report arithmetic without PCM or FFT.
Records do not supply rows not measured originally. Weighting/band switches
apply at read time; changing the quiet threshold or shortest gap requires a
new measurement. Dither and click classifications do not change with frequency
weighting or display band. A track with no qualifying quiet data is represented
by absence of a spectrum, never by an invented zero noise measurement.

## Definitions and decisions

The inventory does not specify numeric thresholds for several heuristic rows.
The following are explicit first-round definitions, configurable where indicated.
See `../MEASUREMENT-CONTRACT.md` for the complete row-by-row map and limitations.

- Quiet passages: rational 50 ms blocks use maximum per-channel RMS at
  or below `quiet_threshold_dbfs` (default -60 dBFS). Transition-block
  refinement separates clear loud material using extreme excursions or
  short 1 ms local RMS evidence. Aggregate per-channel RMS qualifies the
  actual prefix/suffix, including quiet waveforms with above-threshold
  crests. A consistent adjacent quiet block supplies bounded energy
  context for partial cycles (half its minimum channel headroom per edge,
  peak within 3 dB of its above-threshold crest, no extreme excursion).
  If a mixed-block edge fails its aggregate test, up to two 1 ms local
  slices may be trimmed at the separator-facing end. The retained edge
  must satisfy the same adjacent quiet context. No trimming is used to
  admit an entire final partial block. Clear abrupt transitions retain
  sample-level boundaries; tonal transitions remain estimates. Ambiguous
  low-contrast changes use the coarse detector/instantaneous edges. Only
  contiguous real frames count toward `shortest_gap_seconds` (default
  1 s); no padding or reduction of the minimum is used.
- Spectrum: 50%-overlap sine-of-Hann (Vorbis) windows of size
  `next_power_of_two(ceil(sample_rate/6))`. For H=N/2, w[i]^2+w[i+H]^2=1.
  The first window starts H frames before the stream; outside the actual
  stream/passage all samples are zero. Each real frame is counted once in
  the denominator. Fixed one-sided FFT normalization preserves mean square
  without rescaling frequency bins by block energy. Spectral leakage and
  bin resolution still limit band/line estimates. Bandwidth augments the
  complete-window extent with separately tapered real-support first/last
  half-hop and shorter endpoint views. Complete-window bins must meet the
  −80 dB floor relative to the full recording's maximum accumulated power
  bin, even if the interior contains only dither. Local peaks must meet the −80 dB
  threshold relative to both their own view's maximum and the full recording's
  maximum accumulated power bin. Endpoint power uses the same fixed one-sided
  FFT normalization as the retained spectrum; neither level nor duration is
  independently normalized. The highest eligible peak beyond the interior
  extent's physical-resolution guard can extend bandwidth. No endpoint
  inference bins enter the retained power spectrum. Files without any
  material interior bin use actual-support tapered evidence subject to the
  same materiality test. Source-rate, codec and
  stereo inference still prefer fully observed windows and can underrepresent
  endpoint-only content. Every frame contributes to retained power. Bins use
  half-open frequency bands, with Nyquist included in the final band.
  A and ITU-R 468 apply their frequency response as a power multiplier
  (468 RMS weighting, not quasi-peak).
- Limiting uses sample peaks in rational 50 ms blocks, including a final
  partial block, within 1 dB of the track maximum. The inventory permits this
  instead of block true peaks. All-silent track: 0%.
- Flat extrema: runs of at least 3 samples (relative tolerance 1e-5 or one
  integer LSB, whichever is larger), with both neighboring samples proving a
  local extremum and absolute value within 1 dB of the track peak. Count runs
  per channel. Resampling and lossy coding may erase flat tops; slow/saturated
  waveforms may also trigger them.
- Integer resolution: OR of exact signed codes provides the normalized
  low-bit depth. GCD of differences from the first code provides the exact
  affine lattice step; `observed_lattice_bits` is the ceil-log2 width of
  the observed span in lattice positions (None for constant samples).
  `used` is the smaller width for a demonstrable nonbinary or offset
  affine lattice (at least three positions); ordinary headroom does not
  prevent this reduction. An aligned binary grid retains the normalized
  count unless its span has a power-of-two count of at least four positions
  and at least that many samples. Two points alone do not establish an
  affine subset. This measures observed grid evidence, not source depth
  or proof of occupation of every position.
  Floating PCM is tested exactly on the normalized 2^-52 grid; no snapping.
  Silence uses zero bits. Arbitrary float PCM may be `NonIntegerGrid`.
- Dither: use the nominal integer spacing when eligible; otherwise require
  exact zero-aligned binary spacing and peak no greater than eight such
  LSBs. Both signs and the existing spectral shape test are required.
  Exact integer-grid floats and padded integer PCM can qualify. Affine
  spacing alone does not establish a binary quantizer; multiples of 100
  nominal codes have only four-code alignment and fail the magnitude gate.
  This remains a heuristic, not proof of provenance.
- Effective bits: approximate `-noise_floor_dbfs / (20 log10(2))`; no ideal
  quantizer or full-scale sine correction. The album uses its pooled floor,
  following the inventory's provisional rule. Zero noise does not establish a
  finite bound on effective bits. No UI default is chosen in this library.
- Correlation uses mean-removed Pearson covariance. Balance is positive for
  a louder left channel. Width is side/mid power in dB; mono compatibility
  is the nonnegative loss from averaging channels, relative to their average
  pre-sum power. Delay is positive when right lags left; bass cutoff is in Hz.

## Memory and performance

PCM working storage is bounded by the FFT window(s), two quiet blocks, small local
history, and upstream loudness window. A candidate quiet run accumulates its
own power sums and exact grid/sign evidence and commits only on duration qualification; even
hours of silence do not retain hours of samples. Click detection uses five
frames of context. Delay accumulates upper-band cross power and performs one
inverse FFT at finish. FFT plans and scratch are reused per analysis stream.

One retained spectrum is `(N/2+1)*8` bytes plus metadata: 32,776 bytes at
48 kHz, 65,544 at 96 kHz, 131,080 at 192 kHz. A track retaining both average
and quiet spectra needs roughly twice that. A 100-track 96 kHz album holding
both needs ~12.5 MiB for bin data. The quiet detector may temporarily hold a
candidate and pooled spectrum, not one spectrum per window. Dither adds a
separate candidate spectrum only when selected. Cutoff/codec inference
uses an additional transient spectrum only when selected. The maximum per-engine PCM
window capacity is `N * channels * 8` bytes (64 MiB at the maximum format).

The time curve retains 10 points/s (16 bytes/point); limiting retains 20 block
peaks/s during a track and drops them on finish. Flat-top candidates use an
ordered sparse count map pruned by increasing track peak. The graph, limiting
and candidate-map paths have a 4-million observation/entry admission cap.
The upstream loudness meter enforces its own 256 MiB admission limit. Per-track reports retain only the scalar LRA; continuous album LRA
uses the live native meter. Allocation failures from Rust's global allocator are not guaranteed
recoverable; typical formats are substantially below the maximum geometry.

A reproducible throughput fixture is supplied (`cargo run --release --example
benchmark -- 60 96000`); no throughput claim is made without actual Rust
execution and target-host measurements.

## Build and publish

With the two trees beside one another:

```sh
cd audioscopy
nix develop
cargo test --release --no-fail-fast
cargo doc --no-deps
cargo run --release --example benchmark -- 60 96000
```

Or use the manifest's minimum Rust 1.95 or later; the supplied flakes pin
1.99.1. The existing true-peak README documents the release-vectorizer issue
and workaround for old compilers. The bundle's `qualify.sh` runs both trees.

Direct dependencies:

| Crate | Version | Licence | Reason |
|---|---|---|---|
| `true-peak` | `=0.1.0`, sibling during development | MIT OR Apache-2.0 | Existing qualified native loudness core; additive ungated short-term API. |
| `rustfft` | `=6.4.1` | MIT OR Apache-2.0 | Mature, portable pure-Rust FFT planner, reusable scratch and CPU-specific SIMD. Upstream release 2025-09-18; repo maintained as of 2026-11. |
| `num-complex` (via true-peak and RustFFT) | `=0.4.6` in true-peak | MIT OR Apache-2.0 | Existing complex arithmetic dependency; RustFFT re-exports its compatible complex type. |

RustFFT upstream: <https://docs.rs/crate/rustfft/6.4.1> and
<https://github.com/ejmahler/RustFFT/releases/tag/6.4.1>. Cargo must resolve and
verify the transitive graph; it was not available in the delivery environment.
The direct dependency versions are pinned; the resolved transitive versions,
licenses and lockfile have not been independently checked in this delivery.

`true-peak` 0.1.0 must be published first, **including the additive short-term
API**, then `audioscopy` 0.1.0. A Cargo dependency with both `path` and `version`
uses the sibling locally and the version from the registry in the published
manifest. Packaging audioscopy cannot resolve an unpublished dependency.
Do not publish either crate until the native qualification gates pass.

## Licence

audioscopy is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free
Software Foundation, either version 2 of the License, or (at your option)
any later version. The full text is in [COPYING](COPYING).

It is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE.

### K-weighting numerical fidelity

Short-term maximum, loudness over time and track/continuous-album loudness range explicitly select the opt-in conditioned native K-weighting realization from the sibling `true-peak` crate. Measurement selection, gating, window clocks and album continuity remain unchanged. The delivery's `K-WEIGHTING-REPORT.md` records independent arbitrary-precision and compiled-C evidence. Native qualification on the pinned toolchain is an outstanding release gate.
