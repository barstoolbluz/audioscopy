# Performance and memory

This document describes how Audioscopy's performance figures are measured, the
results on the reference host, and how much memory analysis uses.

## Results

Median wall time and peak process memory for 60 seconds of stereo 24-bit PCM on
one logical CPU of an AMD Ryzen Threadripper 7970X. Realtime factor is audio
duration divided by wall time.

| Selection (`numerical_profile` group) | Rate | Wall time | Realtime | Peak RSS |
|---|---:|---:|---:|---:|
| All 25 measurements (`all`) | 48 kHz | 0.247 s | 243x | 7.2 MiB |
| All 25 measurements (`all`) | 768 kHz | 4.63 s | 13x | 68.5 MiB |
| All 25, as an album of 6 s tracks (`album`) | 48 kHz | 0.316 s | 190x | 8.3 MiB |
| All 25, as an album of 6 s tracks (`album`) | 768 kHz | 5.60 s | 11x | 79.4 MiB |
| Loudness range, short-term maximum, loudness over time (`loudness`) | 48 kHz | 0.0537 s | 1,117x | 4.2 MiB |
| Loudness range, short-term maximum, loudness over time (`loudness`) | 768 kHz | 0.862 s | 70x | 26.0 MiB |
| Short-term maximum and loudness over time (`short`) | 48 kHz | 0.0362 s | 1,656x | 4.2 MiB |
| Correlation, balance, width, mono compatibility (`stereo`) | 192 kHz | 0.0801 s | 749x | 8.7 MiB |
| Correlation, balance, width, mono compatibility (`stereo`) | 768 kHz | 0.323 s | 185x | 26.0 MiB |

Source: native qualification round R5 (checkpoint 5, 2026-11-26). Checkpoint 6
changed only formatting and lint hygiene, not DSP code.

These figures describe this fixture on this host. Actual throughput depends on
the hardware, the selected measurements and the audio content.

## Method

### Build

- Rust 1.99.1 from the pinned Nix development environment, with the frozen
  `Cargo.lock` files for both crates.
- Release profile with portable code generation. No `target-cpu`,
  `target-feature` or other global Rust flags are set. ISA-specific paths are
  selected at run time by feature detection.

### Workload

The `numerical_profile` example runs one workload:

```sh
cargo run --release --example numerical_profile -- <group> <rate> <channels> <seconds>
```

- The fixture is one second of deterministic mixed tones and noise in exact
  24-bit integer PCM, plus a low-level quantized quiet second. In each four-second
  cycle, two seconds are active and two are quiet, so both the programme and the
  quiet-passage DSP run. Fixture generation is excluded from the timing.
- PCM is pushed in chunks of 4,093 frames, which do not align with the 50 ms
  block or FFT hop sizes.
- The `album` and `album_loudness` groups use `AlbumAnalyzer` with a track
  boundary every six seconds, including a short last track.
- The timed region covers analyzer construction, ingestion, finalization and
  reading the selected results. It excludes decoding and disk I/O.

### Timing protocol

- Each process is pinned to one logical CPU with `taskset`. On the reference
  host this is CPU 1; its SMT sibling (CPU 33) is kept idle as far as possible.
  Pinning does not guarantee physical isolation or a fixed effective frequency.
- Each workload runs one warm-up and five measured repetitions. When two builds
  are compared, each repetition runs both builds as a pair, alternating which
  runs first.
- Peak resident memory comes from GNU time (`%M`) for the whole process.
- The reported figure is the median. Build comparisons use the median of the
  paired ratios, with the minimum and maximum pair. Ratios from different
  cohorts are not compared as if they were paired.

## Memory

### Bounded working storage

PCM working storage is bounded by the FFT windows, two quiet blocks, a small
local history and the loudness window.

- A candidate quiet run accumulates its own power sums and exact grid and sign
  evidence, and commits only when its duration qualifies. Hours of silence do
  not retain hours of samples.
- Click detection uses five frames of context.
- Inter-channel delay accumulates upper-band cross power and performs one
  inverse FFT at finish.
- FFT plans and scratch buffers are reused for each analysis stream.
- The maximum PCM window capacity per spectral engine is `N * channels * 8`
  bytes, which is 64 MiB at the maximum format.

### Retained results

One retained spectrum needs `(N/2 + 1) * 8` bytes plus metadata, where `N` is
the FFT size:

| Rate | FFT size `N` | One spectrum |
|---:|---:|---:|
| 48 kHz | 8,192 | 32,776 bytes |
| 96 kHz | 16,384 | 65,544 bytes |
| 192 kHz | 32,768 | 131,080 bytes |

- A track that retains both the average and the quiet spectrum needs about twice
  that. A 100-track 96 kHz album that retains both needs about 12.5 MiB of bin
  data.
- The quiet detector may temporarily hold one candidate and one pooled spectrum,
  not one spectrum per window.
- Dither adds a separate candidate spectrum only when selected. Cutoff and codec
  inference use an additional transient spectrum only when selected.
- The loudness curve retains 10 points per second (16 bytes per point).
  Limiting retains 20 block peaks per second during a track and drops them at
  finish.
- Flat-top candidates use an ordered sparse count map, pruned as the track peak
  increases.

### Limits

- The loudness curve, limiting and flat-top candidate map have an admission cap
  of 4 million observations or entries.
- The `true-peak` loudness meter enforces its own 256 MiB storage limit.
- Per-track reports retain only the scalar loudness range. Continuous album
  loudness range uses the live native meter.
- Allocation failures from Rust's global allocator are not guaranteed to be
  recoverable. Typical formats use far less than the maximum geometry.
