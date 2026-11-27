# Measurement definitions

This document records how Audioscopy defines and computes each measurement:
the thresholds, estimator choices and known limitations. The README gives an
overview; this document is the reference for anyone interpreting or comparing
results.

Several heuristic measurements in the TonePoet Advanced Analysis measurement
inventory (revision 2, sections 3.1–3.6) do not specify numeric thresholds. The
definitions below are Audioscopy's explicit choices, configurable where stated.

## Input

The input is interleaved binary64 (`f64`) PCM.

- **Integer PCM.** Normalize each signed `b`-bit code by `2^(b-1)`, so full
  scale is `-1.0` through `1.0 - 2^(1-b)`, and declare `Encoding::Integer(b)`.
  The analyzer requires exact normalized codes. Do not scale by `i16::MAX`,
  round float samples or dither in the decoder before measurement.
- **Float PCM** is accepted as delivered, including finite overs up to an
  absolute magnitude of 16. NaN and infinite samples are rejected.
- Supported rates are 8–768 kHz, channels 1–64, integer depths 1–32.
- Loudness measurements with more than two channels require explicit
  loudness roles. Stereo measurements apply only to exactly two channels.

## Selection and shared computation

`Measurements::new` selects individual measurements and `Measurements::all`
selects all 25. `Measurements::default` is empty; default selections belong to
the application.

Requesting a subset avoids the other measurements' DSP and storage. Shared
prerequisites, such as the spectra used by several noise measurements, are
computed once per required stream.

Loudness uses one of three native meters from `true-peak`, chosen by the
selection:

| Selected | Meter |
|---|---|
| Loudness range only | Range-only meter. No integrated loudness observations are accumulated. |
| Short-term maximum and/or loudness over time only | Short-term meter. No range or integrated work is started. |
| Loudness range and a short-term measurement | One shared meter. A single K-weighting filter path feeds both the range and the short-term observations. |

All three use the conditioned binary64 K-weighting realization (see
[K-weighting](#k-weighting)).

## Streaming and errors

Chunk boundaries do not reset analysis; any whole-frame chunk length gives the
same result. Malformed pushes are validated before any state changes. A
terminal processing error requires a new analyzer. `finish` consumes the
analyzer and rejects a stream of zero frames.

## Quiet passages

Quiet-passage detection drives the noise floor, dynamic range, effective bits,
noise spectrum, hum and idle tones, and noise by band.

- Rational 50 ms blocks qualify when the maximum per-channel RMS is at or below
  `quiet_threshold_dbfs` (default -60 dBFS).
- Transition-block refinement separates clear loud material using extreme
  excursions or short 1 ms local RMS evidence. Aggregate per-channel RMS
  qualifies the actual prefix or suffix, including quiet waveforms with
  above-threshold crests.
- A consistent adjacent quiet block supplies bounded energy context for partial
  cycles: half its minimum channel headroom per edge, peak within 3 dB of its
  above-threshold crest, and no extreme excursion.
- If a mixed-block edge fails its aggregate test, up to two 1 ms local slices
  may be trimmed at the separator-facing end. The retained edge must satisfy
  the same adjacent quiet context. Trimming is never used to admit an entire
  final partial block.
- Clear abrupt transitions retain sample-level boundaries. Tonal transitions
  remain estimates. Ambiguous low-contrast changes use the coarse detector and
  instantaneous edges.
- Only contiguous real frames count toward `shortest_gap_seconds` (default
  1 s). No padding or reduction of the minimum is used.

## Spectrum

- Windows are 50%-overlap sine-of-Hann (Vorbis) windows of size
  `next_power_of_two(ceil(sample_rate / 6))`. For hop `H = N/2`,
  `w[i]^2 + w[i+H]^2 = 1`.
- The first window starts `H` frames before the stream. Outside the actual
  stream or passage, all samples are zero. Each real frame is counted once in
  the denominator.
- A fixed one-sided FFT normalization preserves mean square without rescaling
  frequency bins by block energy. Spectral leakage and bin resolution still
  limit band and line estimates.
- Bins use half-open frequency bands, with Nyquist included in the final band.
  Every frame contributes to the retained power spectrum.
- A-weighting and ITU-R 468 apply their frequency response as a power
  multiplier. ITU-R 468 is RMS-weighted, not quasi-peak.

### Bandwidth

- Bandwidth augments the complete-window extent with separately tapered
  real-support first and last half-hop views and shorter endpoint views.
- Complete-window bins must meet a -80 dB floor relative to the full
  recording's maximum accumulated power bin, even if the interior contains only
  dither.
- Local peaks must meet the -80 dB threshold relative to both their own view's
  maximum and the full recording's maximum accumulated power bin.
- Endpoint power uses the same fixed one-sided FFT normalization as the
  retained spectrum. Neither level nor duration is independently normalized.
- The highest eligible peak beyond the interior extent's physical-resolution
  guard can extend bandwidth. Endpoint inference bins never enter the retained
  power spectrum.
- Files without any material interior bin use actual-support tapered evidence,
  subject to the same materiality test.

Source-rate, codec and stereo inference still prefer fully observed windows,
and can underrepresent content that appears only near the endpoints.

## Dynamics

- **Limiting** uses sample peaks in rational 50 ms blocks, including a final
  partial block, within 1 dB of the track maximum. The inventory permits sample
  peaks instead of block true peaks. An all-silent track reports 0%.
- **Flat-topped peaks** are runs of at least 3 samples (relative tolerance 1e-5
  or one integer LSB, whichever is larger). Both neighbouring samples must prove
  a local extremum, and the absolute value must be within 1 dB of the track
  peak. Runs are counted per channel. Resampling and lossy coding may erase flat
  tops; slow or saturated waveforms may also produce them.

## Resolution

### Bits in use

- The OR of exact signed codes gives the normalized low-bit depth.
- The GCD of differences from the first code gives the exact affine lattice
  step. `observed_lattice_bits` is the ceil-log2 width of the observed span in
  lattice positions (`None` for constant samples).
- `used` is the smaller width for a demonstrable non-binary or offset affine
  lattice of at least three positions. Ordinary headroom does not prevent this
  reduction.
- An aligned binary grid retains the normalized count unless its span has a
  power-of-two count of at least four positions and at least that many samples.
  Two points alone do not establish an affine subset.
- Float PCM is tested exactly on the normalized 2^-52 grid, with no snapping.
  Arbitrary float PCM may report `NonIntegerGrid`. Silence uses zero bits.

This measures observed grid evidence. It does not establish the source bit
depth, and it does not prove that every lattice position was used.

### Effective bits

Effective bits are approximately `-noise_floor_dbfs / (20 log10 2)`. No ideal
quantizer or full-scale sine correction is applied. The album value uses the
pooled album noise floor, following the inventory's provisional rule. Zero
noise does not establish a finite bound on effective bits.

### Dither

- Uses the nominal integer spacing when eligible. Otherwise it requires exact
  zero-aligned binary spacing and a peak of no more than eight such LSBs.
- Both signs and the spectral-shape test are required.
- Exact integer-grid floats and padded integer PCM can qualify. Affine spacing
  alone does not establish a binary quantizer; for example, multiples of 100
  nominal codes have only four-code alignment and fail the magnitude gate.

Dither classification is a heuristic. It is not proof of provenance.

## Stereo

- **Correlation** is mean-removed Pearson covariance. Zero-variance correlation
  is 0.
- **Balance** is positive when the left channel is louder.
- **Width** is side/mid power in dB. Zero/zero stereo energy is defined as 0 dB.
- **Mono compatibility** is the non-negative loss from averaging the channels,
  relative to their average pre-sum power.
- **Inter-channel delay** is positive when the right channel lags the left.
  Delay accumulates upper-band cross power and performs one inverse FFT at
  finish.
- **Bass width** reports the cutoff in Hz below which the bass is effectively
  mono.

## Loudness

Loudness follows ITU-R BS.1770 K-weighting and EBU Tech 3342 loudness range.
Short-term measurements use complete three-second windows at 100 ms intervals.
A maximum over no complete window reports a reason, not a number.

### K-weighting

Short-term maximum, loudness over time, and track and continuous-album loudness
range use the conditioned native K-weighting realization in `true-peak`.
Measurement selection, gating, window clocks and album continuity are unchanged
by the conditioning; it reduces arithmetic error only.

## Albums

`AlbumAnalyzer::new(selection, settings)`, then for each track in order
`begin_track(format)`, zero or more `push_interleaved` calls and `end_track`,
then `finish`. Each track is decoded once. `Album::tracks()` exposes the ordered
track results. `Album::measure` applies each measurement's album reduction rule.

- **Continuous loudness range.** With `AlbumAnalyzer`, loudness range follows a
  continuous programme clock and filter across tracks with the same sample rate
  and channel roles, including windows that cross track boundaries.
- **Independent reports.** `Album::from_tracks(Vec<Track>)` accepts
  individually computed tracks with identical selections and detector settings.
  Independent statistics cannot reconstruct boundary windows, so album loudness
  range is `Outcome::CantTell` with a reason. The same applies when the rate or
  roles change between tracks. A single cached track is already one complete
  programme. No resampler or artificial per-track tail is used.
- `Album::loudness_method()` reports whether loudness range is continuous,
  unavailable or not requested. Other album measurements remain available.
- **Pooling.** Unavailable track results contribute no evidence to modes and
  medians over eligible tracks. If no track applies, the album returns the
  corresponding typed unavailable outcome. Power-based pooling weights by
  qualifying quiet time or programme time, never by track count, by sample
  count across mismatched sample rates, or by a mean in dB.

## Outcomes and curves

`Outcome::NotRequested`, `Outcome::NotApplicable`, `Outcome::CantTell(reason)`
and `Outcome::Value(value)` are distinct. The first three are never numeric.

- Exact zero power has a defined level of negative infinity. Positive infinity
  may describe an exact zero denominator. NaN is never used as a display
  sentinel.
- Loudness range with no eligible observations, and resolution or dynamic range
  unbounded because of pure silence, report a reason rather than a fabricated
  finite result.
- `Value::Curve` is ordered numerical data. `Value::Curves` holds per-track
  loudness curves for an album.
- Spectrum curves are one-sided power density in dBFS/Hz, with frequency in Hz.
  Album spectrum bins map to the finest native grid by linear energy splatting.
- Loudness points use an exclusive end time in seconds and LUFS.

## Noise records

`Track::noise_record()` exports the sufficient statistics of the selected noise
measurements only, as a `NoiseRecord` (schema version 2). Applications may
persist its plain fields in any cache format.

- Version 1 records must be regenerated from PCM. Their frequency distribution
  used a different, incorrect normalization.
- `NoiseSnapshot::from_record` validates masks, geometry, frames, power,
  thresholds and numeric values before accepting a record. Its `measure` and
  `measure_album` reuse the report arithmetic, without PCM or FFT.
- A record cannot supply measurements that were not originally selected.
- Weighting and band changes apply at read time. Changing the quiet threshold or
  the shortest gap requires a new analysis.
- Dither and click classifications do not change with frequency weighting or
  display band.
- A track with no qualifying quiet data has no spectrum. It never reports an
  invented zero noise measurement.
