use crate::report::Track;
use crate::spectrum::{Quiet, SpectralEngine};
use crate::temporal::{limiting, Clicks, Flats, Grid, Stereo};
use crate::Measurement as M;
use crate::*;
use true_peak::loudness::{
    LoudnessRangeAndShortTermMeter, LoudnessRangeMeter, ShortTermMeter, ShortTermObservation,
};

pub(crate) const MAX_FRAMES: u64 = 1 << 48;
const MAX_OBSERVATIONS: usize = 4_000_000;
const MAX_SAMPLE_ABS: f64 = 16.0;

pub(crate) fn validate_format(f: &Format, settings: Settings) -> Result<(), Error> {
    if !(8_000..=768_000).contains(&f.sample_rate) {
        return Err(Error::Configuration(
            "sample rate must be 8,000 through 768,000 Hz",
        ));
    }
    if !(1..=64).contains(&f.channels) {
        return Err(Error::Configuration("channel count must be 1 through 64"));
    }
    if matches!(f.encoding, Encoding::Integer(n) if !(1..=32).contains(&n)) {
        return Err(Error::Configuration("integer depth must be 1 through 32"));
    }
    if f.loudness_roles
        .as_ref()
        .is_some_and(|r| r.len() != f.channels)
    {
        return Err(Error::Configuration(
            "loudness role count differs from channel count",
        ));
    }
    if !(-200.0..=0.0).contains(&settings.quiet_threshold_dbfs)
        || !(0.05..=60.0).contains(&settings.shortest_gap_seconds)
    {
        return Err(Error::Configuration(
            "invalid quiet threshold or gap duration",
        ));
    }
    Ok(())
}
fn roles(f: &Format) -> Result<Vec<ChannelRole>, Error> {
    if let Some(r) = &f.loudness_roles {
        return Ok(r.clone());
    }
    match f.channels {
        1 => Ok(vec![ChannelRole::Mono]),
        2 => Ok(vec![ChannelRole::Left, ChannelRole::Right]),
        _ => Err(Error::Configuration(
            "multichannel loudness requires explicit channel roles",
        )),
    }
}
fn range_meter(f: &Format) -> Result<LoudnessRangeMeter, Error> {
    Ok(LoudnessRangeMeter::new(f.sample_rate, &roles(f)?)?)
}
pub(crate) fn validate_samples(format: &Format, samples: &[f64]) -> Result<(), Error> {
    if !samples.len().is_multiple_of(format.channels) {
        return Err(Error::IncompleteFrame);
    }
    let integer_scale = match format.encoding {
        Encoding::Integer(n) => Some(2.0_f64.powi(i32::from(n) - 1)),
        Encoding::Float => None,
    };
    for (i, &x) in samples.iter().enumerate() {
        if !x.is_finite() || x.abs() > MAX_SAMPLE_ABS {
            return Err(Error::InvalidSample(i));
        }
        if let Some(scale) = integer_scale {
            let code = x * scale;
            if code.fract() != 0.0 || code < -scale || code >= scale {
                return Err(Error::InvalidSample(i));
            }
        }
    }
    Ok(())
}

// Locate clear loud material inside a transition block. A large isolated
// excursion (>8 times the RMS threshold) is a separator. Lesser peaks
// need sustained local RMS evidence (1 ms, >2 times the RMS threshold).
// Using the same slice partition in both directions prevents overlapping
// prefix/suffix claims, even in short mixed blocks. Ordinary noisy blocks
// generally need just the first and last local slice.
fn loud_bounds(pcm: &[f64], channels: usize, rate: u32, threshold: f64) -> Option<(usize, usize)> {
    let chunk = (rate as usize).div_ceil(1000) * channels;
    let bounds = |slice: &[f64]| {
        let mut strong: Option<(usize, usize)> = None;
        let mut extreme: Option<(usize, usize)> = None;
        let mut energy = [0.0_f64; 64];
        for (i, frame) in slice.chunks_exact(channels).enumerate() {
            let mut peak = 0.0_f64;
            for (sum, &x) in energy.iter_mut().zip(frame) {
                let p = x * x;
                *sum += p;
                peak = peak.max(p);
            }
            if peak > 4.0 * threshold {
                strong = Some((strong.map_or(i, |b| b.0), i));
            }
            if peak > 64.0 * threshold {
                extreme = Some((extreme.map_or(i, |b| b.0), i));
            }
        }
        // Prefer the unambiguous extreme if one exists in this slice;
        // do not absorb an adjacent quiet waveform's ordinary crests.
        extreme.or_else(|| {
            let limit = 4.0 * threshold * (slice.len() / channels) as f64;
            if energy[..channels].iter().any(|&e| e > limit) {
                strong
            } else {
                None
            }
        })
    };
    let first = pcm
        .chunks(chunk)
        .enumerate()
        .find_map(|(i, s)| bounds(s).map(|b| i * chunk / channels + b.0))?;
    let last = pcm
        .chunks(chunk)
        .enumerate()
        .rev()
        .find_map(|(i, s)| bounds(s).map(|b| i * chunk / channels + b.1))?;
    Some((first, last + 1))
}

// Aggregate RMS on an actual contiguous edge fragment. Neighbor energy
// headroom is used only when the neighbor itself contains above-threshold
// crests and the fragment peak is within 3 dB of its peak. This tolerates
// finite-window peak variation without extending digital silence across a
// clearly louder step. Context samples themselves are never re-counted.
fn quiet_edge<'a>(
    frames: impl Iterator<Item = &'a [f64]> + Clone,
    threshold: f64,
    context: (f64, f64),
    max_trim: usize,
) -> usize {
    let original = frames.clone();
    let mut energy = [0.0_f64; 64]; // validated maximum channel count
    let mut count = 0;
    let mut exact = 0;
    let mut channels = 0;
    let mut peak = 0.0_f64;
    for frame in frames {
        channels = frame.len();
        let mut frame_peak = 0.0_f64;
        for (sum, &x) in energy.iter_mut().zip(frame) {
            let p = x * x;
            *sum += p;
            frame_peak = frame_peak.max(p);
        }
        if exact == count && frame_peak <= threshold {
            exact += 1;
        }
        peak = peak.max(frame_peak);
        count += 1;
    }
    let limit = threshold * count as f64;
    let margin = if context.1 > threshold && peak <= 2.0 * context.1 {
        context.0
    } else {
        0.0
    };
    if energy[..channels].iter().all(|&e| e <= limit + margin) {
        return count;
    }

    // The separator is located by extreme/local-RMS evidence, not by the
    // last programme sample. In oscillatory material a few residual loud
    // samples can therefore contaminate an otherwise quiet edge. Before
    // reverting to instantaneous testing, consider a shorter contiguous
    // prefix (suffix for a reversed iterator), trimming only the side
    // facing the separator. Require consistent adjacent quiet context and
    // the same per-channel RMS/peak criteria as the whole-fragment test.
    // Reaccumulate the retained prefix rather than subtracting loud energy
    // from a total (which can erase low-level power by cancellation).
    if max_trim == 0 || context.1 <= threshold {
        return exact;
    }
    let min_keep = count.saturating_sub(max_trim).max(exact + 1);
    if min_keep >= count {
        return exact;
    }
    energy.fill(0.0);
    let mut best = exact;
    for (i, frame) in original.take(count - 1).enumerate() {
        // A violating peak cannot be removed by extending this prefix.
        if frame.iter().any(|&x| x * x > 2.0 * context.1) {
            break;
        }
        for (sum, &x) in energy.iter_mut().zip(frame) {
            *sum += x * x;
        }
        let kept = i + 1;
        if kept >= min_keep
            && energy[..channels]
                .iter()
                .all(|&e| e <= threshold * kept as f64 + context.0)
        {
            best = kept;
        }
    }
    best
}

// One per-track loudness path. Keep the state inline to avoid a new heap
// allocation; the largest variant replaces the two former inline meters.
#[allow(
    clippy::large_enum_variant,
    reason = "inline per-track DSP state avoids a separate allocation"
)]
enum TrackLoudness {
    None,
    Range(LoudnessRangeMeter),
    Short(ShortTermMeter),
    Both(LoudnessRangeAndShortTermMeter),
}

/// Streaming per-file analyzer. Push any number of complete interleaved frame
/// chunks, then consume the analyzer to produce an owned report. PCM is never
/// retained for the entire track. Failed numerical operations are terminal.
pub struct Analyzer {
    format: Format,
    selection: Measurements,
    settings: Settings,
    frames: u64,
    failed: bool,
    peak: f64,
    need_peak: bool,
    grid: Option<Grid>,
    stereo: Option<Stereo>,
    flats: Option<Flats>,
    clicks: Option<Clicks>,
    spectral: Option<SpectralEngine>,
    quiet: Option<Quiet>,
    dither_quiet: Option<Quiet>,
    loudness: TrackLoudness,
    short_max: Option<f64>,
    curve: Vec<Point>,
    block_pcm: Vec<f64>,
    block_energy: Vec<f64>,
    pending_pcm: Vec<f64>,
    pending_quiet: Option<bool>,
    pending_prev_noisy: bool,
    pending_peak: f64,
    pending_context: (f64, f64),
    pending_prev_context: (f64, f64),
    block_frames: u64,
    block_index: u64,
    block_end: u64,
    block_peak: f64,
    block_peaks: Vec<f64>,
    threshold_power: f64,
}
impl Analyzer {
    /// Construct a selective analyzer. All optional DSP state is created only
    /// for selected measurements or their necessary shared dependencies.
    pub fn new(format: Format, selection: Measurements, settings: Settings) -> Result<Self, Error> {
        validate_format(&format, settings)?;
        let rate = format.sample_rate;
        let channels = format.channels;
        let need_power = selection.any(&[
            M::LikelySourceRate,
            M::LossyCodingPattern,
            M::Bandwidth,
            M::AverageSpectrum,
        ]) || (rate > 44_100 && selection.contains(M::Above20Khz));
        let bass = channels == 2 && selection.contains(M::BassWidth);
        let delay = channels == 2 && selection.contains(M::InterChannelDelay);
        let need_quiet = selection.any(&[
            M::EffectiveBits,
            M::NoiseFloor,
            M::DynamicRange,
            M::NoiseSpectrum,
            M::HumAndIdleTones,
            M::NoiseByBand,
        ]);
        let dither = selection.contains(M::Dither);
        let loudness = match (
            selection.contains(M::LoudnessRange),
            selection.any(&[M::ShortTermMaximum, M::LoudnessOverTime]),
        ) {
            (true, true) => {
                TrackLoudness::Both(LoudnessRangeAndShortTermMeter::new(rate, &roles(&format)?)?)
            }
            (true, false) => TrackLoudness::Range(range_meter(&format)?),
            (false, true) => {
                TrackLoudness::Short(ShortTermMeter::new_high_fidelity(rate, &roles(&format)?)?)
            }
            (false, false) => TrackLoudness::None,
        };
        Ok(Self {
            format: format.clone(),
            selection,
            settings,
            frames: 0,
            failed: false,
            peak: 0.0,
            need_peak: selection.contains(M::DynamicRange),
            grid: selection
                .contains(M::BitsInUse)
                .then(|| Grid::new(format.encoding)),
            stereo: (channels == 2
                && selection.any(&[M::Correlation, M::Balance, M::Width, M::MonoCompatibility]))
            .then(|| {
                Stereo::new(
                    selection.contains(M::Correlation),
                    selection.contains(M::Balance),
                    selection.contains(M::Width),
                    selection.contains(M::MonoCompatibility),
                )
            }),
            flats: selection
                .contains(M::FlatToppedPeaks)
                .then(|| Flats::new(channels, format.encoding)),
            clicks: selection
                .contains(M::ClicksPerMinute)
                .then(|| Clicks::new(channels, rate)),
            spectral: (need_power || bass || delay).then(|| {
                SpectralEngine::new(
                    rate,
                    channels,
                    need_power,
                    bass,
                    delay,
                    selection.contains(M::LossyCodingPattern),
                    selection.any(&[M::LikelySourceRate, M::Bandwidth, M::LossyCodingPattern]),
                    selection.contains(M::Bandwidth),
                )
            }),
            quiet: need_quiet
                .then(|| Quiet::new(rate, channels, settings.shortest_gap_seconds, None)),
            dither_quiet: dither.then(|| {
                Quiet::new(
                    rate,
                    channels,
                    settings.shortest_gap_seconds,
                    Some(format.encoding),
                )
            }),
            loudness,
            short_max: None,
            curve: Vec::new(),
            block_pcm: if need_quiet || dither {
                Vec::with_capacity((rate as usize).div_ceil(20) * channels)
            } else {
                Vec::new()
            },
            block_energy: if need_quiet || dither {
                vec![0.0; channels]
            } else {
                Vec::new()
            },
            pending_pcm: if need_quiet || dither {
                Vec::with_capacity((rate as usize).div_ceil(20) * channels)
            } else {
                Vec::new()
            },
            pending_quiet: None,
            pending_prev_noisy: false,
            pending_peak: 0.0,
            pending_context: (0.0, 0.0),
            pending_prev_context: (0.0, 0.0),
            block_frames: 0,
            block_index: 0,
            block_end: u64::from(rate) / 20,
            block_peak: 0.0,
            block_peaks: Vec::new(),
            threshold_power: 10.0_f64.powf(settings.quiet_threshold_dbfs / 10.0),
        })
    }

    /// Validate and ingest complete frames. Empty pushes are no-ops; a
    /// malformed push is rejected without consuming any samples.
    pub fn push_interleaved(&mut self, samples: &[f64]) -> Result<(), Error> {
        if self.failed {
            return Err(Error::Failed);
        }
        validate_samples(&self.format, samples)?;
        let target = self
            .frames
            .checked_add((samples.len() / self.format.channels) as u64)
            .ok_or(Error::ResourceLimit)?;
        if target > MAX_FRAMES {
            return Err(Error::ResourceLimit);
        }
        let rate = u64::from(self.format.sample_rate);
        if (self.selection.contains(M::Limiting)
            && target.saturating_mul(20) / rate + 1 > MAX_OBSERVATIONS as u64)
            || (self.selection.contains(M::LoudnessOverTime)
                && target.saturating_mul(10) / rate + 1 > MAX_OBSERVATIONS as u64)
        {
            return Err(Error::ResourceLimit);
        }
        let result = self.process(samples);
        if result.is_err() {
            self.failed = true;
        }
        result
    }

    fn process(&mut self, samples: &[f64]) -> Result<(), Error> {
        match &mut self.loudness {
            TrackLoudness::None => (),
            TrackLoudness::Range(meter) => meter.push_interleaved(samples)?,
            other => {
                let collect = self.selection.contains(M::LoudnessOverTime);
                let collect_max = self.selection.contains(M::ShortTermMaximum);
                let max = &mut self.short_max;
                let curve = &mut self.curve;
                let rate = f64::from(self.format.sample_rate);
                let emit = |point: ShortTermObservation| {
                    if collect_max && max.is_none_or(|v| point.lufs > v) {
                        *max = Some(point.lufs);
                    }
                    if collect {
                        curve.push(Point {
                            x: point.end_frame as f64 / rate,
                            y: point.lufs,
                        });
                    }
                };
                match other {
                    TrackLoudness::Short(meter) => meter.push_interleaved(samples, emit)?,
                    TrackLoudness::Both(meter) => meter.push_interleaved(samples, emit)?,
                    _ => unreachable!("range and empty demand handled above"),
                }
            }
        }
        if let Some(s) = &mut self.spectral {
            s.push(samples);
        }
        let need_block = self.selection.contains(M::Limiting)
            || self.quiet.is_some()
            || self.dither_quiet.is_some();
        let need_frame = need_block
            || self.need_peak
            || self.grid.is_some()
            || self.stereo.is_some()
            || self.flats.is_some()
            || self.clicks.is_some();
        if !need_frame {
            self.frames += (samples.len() / self.format.channels) as u64;
            return Ok(());
        }
        for frame in samples.chunks_exact(self.format.channels) {
            if let Some(grid) = &mut self.grid {
                grid.push(frame);
            }
            if let Some(stereo) = &mut self.stereo {
                stereo.push(frame[0], frame[1]);
            }
            if let Some(flats) = &mut self.flats {
                flats.push(frame);
                if flats.distinct_candidates() > MAX_OBSERVATIONS {
                    return Err(Error::ResourceLimit);
                }
            }
            if let Some(clicks) = &mut self.clicks {
                clicks.push(frame);
            }
            if self.need_peak || need_block {
                let peak = frame.iter().map(|x| x.abs()).fold(0.0, f64::max);
                if self.need_peak {
                    self.peak = self.peak.max(peak);
                }
                if need_block {
                    self.block_peak = self.block_peak.max(peak);
                }
            }
            if !self.block_energy.is_empty() {
                for (energy, x) in self.block_energy.iter_mut().zip(frame) {
                    *energy += x * x;
                }
                self.block_pcm.extend_from_slice(frame);
            }
            self.frames += 1;
            if need_block {
                self.block_frames += 1;
                if self.frames == self.block_end {
                    self.finish_block();
                }
            }
        }
        Ok(())
    }
    fn finish_block(&mut self) {
        if self.block_frames == 0 {
            return;
        }
        if self.selection.contains(M::Limiting) {
            self.block_peaks.push(self.block_peak);
        }
        if !self.block_energy.is_empty() {
            let limit = self.threshold_power * self.block_frames as f64;
            let quiet = self.block_energy.iter().all(|&e| e <= limit);
            // The minimum per-channel energy headroom is conservative for
            // every channel. Reserve half for each adjacent edge so the
            // same energy allowance cannot be spent twice. The peak bounds
            // the use of context; extreme excursions disqualify it.
            let context =
                if quiet && self.block_peak * self.block_peak <= 64.0 * self.threshold_power {
                    (
                        0.5 * self
                            .block_energy
                            .iter()
                            .map(|&e| limit - e)
                            .fold(f64::INFINITY, f64::min),
                        self.block_peak * self.block_peak,
                    )
                } else {
                    (0.0, 0.0)
                };
            let prev_noisy = self.pending_quiet != Some(true);
            let prev_context = self.pending_context;
            self.consume_pending(!quiet, context, false);
            std::mem::swap(&mut self.block_pcm, &mut self.pending_pcm);
            self.pending_quiet = Some(quiet);
            self.pending_prev_noisy = prev_noisy;
            self.pending_peak = self.block_peak;
            self.pending_prev_context = prev_context;
            self.pending_context = context;
        }
        self.block_frames = 0;
        self.block_pcm.clear();
        self.block_energy.fill(0.0);
        self.block_peak = 0.0;
        self.block_index += 1;
        self.block_end =
            ((u128::from(self.block_index) + 1) * u128::from(self.format.sample_rate) / 20) as u64;
    }

    fn consume_pending(&mut self, next_noisy: bool, next_context: (f64, f64), at_end: bool) {
        let Some(quiet) = self.pending_quiet.take() else {
            return;
        };
        let channels = self.format.channels;
        let frames = self.pending_pcm.len() / channels;
        let transition = self.pending_peak * self.pending_peak > 4.0 * self.threshold_power
            && (!quiet || self.pending_prev_noisy || next_noisy);
        let bounds = if transition {
            loud_bounds(
                &self.pending_pcm,
                channels,
                self.format.sample_rate,
                self.threshold_power,
            )
        } else {
            None
        };
        let split = if let Some((first, last)) = bounds {
            // Test actual fragments by aggregate RMS. A short portion of a
            // quiet waveform can exceed the RMS threshold on its own, so
            // also allow the immediate quiet neighbor's consistent context.
            // Limit additional separator-facing trimming to two 1 ms
            // local slices. This covers residual oscillatory shoulders
            // without searching the whole block for a different gap.
            let max_trim = 2 * (self.format.sample_rate as usize).div_ceil(1000);
            let prefix = quiet_edge(
                self.pending_pcm[..first * channels].chunks_exact(channels),
                self.threshold_power,
                self.pending_prev_context,
                max_trim,
            );
            let suffix = quiet_edge(
                self.pending_pcm[last * channels..]
                    .chunks_exact(channels)
                    .rev(),
                self.threshold_power,
                next_context,
                max_trim,
            );
            debug_assert!(prefix + suffix < frames);
            let half = frames.div_ceil(2);
            if !quiet
                || (self.pending_prev_noisy && suffix >= half)
                || (next_noisy && prefix >= half)
            {
                Some((prefix * channels, suffix * channels))
            } else {
                None
            }
        } else if quiet {
            None
        } else if at_end
            && frames < (self.format.sample_rate as usize).div_ceil(20)
            && quiet_edge(
                self.pending_pcm.chunks_exact(channels),
                self.threshold_power,
                self.pending_prev_context,
                0,
            ) == frames
        {
            // A last partial block may end at a waveform crest. Use the
            // contiguous preceding block's RMS evidence, without padding.
            None
        } else {
            // No clear contrast: retain conservative instantaneous edges,
            // rather than merge an above-threshold programme with quiet time.
            let below = |frame: &[f64]| frame.iter().all(|&x| x * x <= self.threshold_power);
            let prefix = self
                .pending_pcm
                .chunks_exact(channels)
                .take_while(|f| below(f))
                .count();
            let suffix = self
                .pending_pcm
                .chunks_exact(channels)
                .rev()
                .take_while(|f| below(f))
                .count();
            debug_assert!(prefix + suffix < frames);
            Some((prefix * channels, suffix * channels))
        };
        if let Some(q) = &mut self.quiet {
            q.block(&self.pending_pcm, split);
        }
        if let Some(q) = &mut self.dither_quiet {
            q.block(&self.pending_pcm, split);
        }
        self.pending_pcm.clear();
    }

    /// Finish the last partial block, close quiet passages, and return results.
    /// Empty input is rejected instead of fabricating a track.
    pub fn finish(mut self) -> Result<Track, Error> {
        if self.failed {
            return Err(Error::Failed);
        }
        if self.frames == 0 {
            return Err(Error::Empty);
        }
        self.finish_block();
        self.consume_pending(true, (0.0, 0.0), true);
        if let Some(q) = &mut self.quiet {
            q.end_run();
        }
        if let Some(q) = &mut self.dither_quiet {
            q.end_run();
        }
        if let Some(s) = &mut self.spectral {
            s.flush();
        }
        let lra = match self.loudness {
            TrackLoudness::Range(meter) => Some(meter.finalize()?),
            TrackLoudness::Both(meter) => Some(meter.finalize()?),
            TrackLoudness::None | TrackLoudness::Short(_) => None,
        };
        let range = lra.and_then(|x| x.finite_lu());
        let bass = self
            .spectral
            .as_ref()
            .filter(|_| self.format.channels == 2 && self.selection.contains(M::BassWidth))
            .map(|s| s.bass_width());
        let delay = self.spectral.as_ref().and_then(|s| s.delay_us());
        let pattern = self.spectral.as_ref().is_some_and(|s| s.lossy_found());
        let dither = self.dither_quiet.as_ref().and_then(Quiet::dither);
        let source_rate = self
            .spectral
            .as_ref()
            .filter(|_| self.selection.contains(M::LikelySourceRate))
            .and_then(|s| s.inference_spectrum().source_rate());
        let bandwidth = self
            .spectral
            .as_ref()
            .filter(|_| self.selection.contains(M::Bandwidth))
            .map(|s| s.bandwidth() / 1000.0);
        let keep_spectrum = self.selection.any(&[M::Above20Khz, M::AverageSpectrum]);
        let spectrum = self
            .spectral
            .take()
            .map(|s| s.spectrum)
            .filter(|s| keep_spectrum && s.frames > 0);
        let quiet = self.quiet.take().map(|q| q.pooled).filter(|s| s.frames > 0);
        Ok(Track {
            format: self.format,
            selection: self.selection,
            settings: self.settings,
            frames: self.frames,
            peak: self.peak,
            bits: self.grid.map(|g| g.result()),
            range,
            source_rate,
            bandwidth,
            short_max: self.short_max,
            curve: self.curve,
            limiting: if self.block_peaks.is_empty() {
                None
            } else {
                Some(limiting(&self.block_peaks))
            },
            flats: self.flats.map(|f| f.count()),
            clicks: self.clicks.map(|c| c.count),
            stereo: self.stereo,
            spectrum,
            quiet,
            pattern,
            bass,
            delay,
            dither,
        })
    }
}

/// Ordered album measurement with a single decode per track. This additionally
/// maintains one continuous loudness-range meter across same-clock/role track
/// boundaries. If rate/layout changes, continuous LRA is unavailable; other
/// album measurements remain available.
pub struct AlbumAnalyzer {
    selection: Measurements,
    settings: Settings,
    tracks: Vec<Track>,
    current: Option<Analyzer>,
    continuous: Option<LoudnessRangeMeter>,
    clock: Option<(u32, Vec<ChannelRole>)>,
    mixed: bool,
    failed: bool,
}
impl AlbumAnalyzer {
    /// Construct an album analyzer; call begin/push/end for each track.
    pub fn new(selection: Measurements, settings: Settings) -> Self {
        Self {
            selection,
            settings,
            tracks: Vec::new(),
            current: None,
            continuous: None,
            clock: None,
            mixed: false,
            failed: false,
        }
    }
    /// Start the next track in album order.
    pub fn begin_track(&mut self, format: Format) -> Result<(), Error> {
        if self.failed {
            return Err(Error::Failed);
        }
        if self.current.is_some() {
            return Err(Error::Album("finish the current track first"));
        }
        let analyzer = Analyzer::new(format.clone(), self.selection, self.settings)?;
        if self.selection.contains(M::LoudnessRange) && !self.mixed {
            let clock = (format.sample_rate, roles(&format)?);
            match &self.clock {
                None => {
                    self.continuous = Some(range_meter(&format)?);
                    self.clock = Some(clock);
                }
                Some(old) if *old != clock => {
                    self.continuous = None;
                    self.mixed = true;
                }
                _ => {}
            }
        }
        self.current = Some(analyzer);
        Ok(())
    }
    /// Feed the currently open track. Invalid pushes never reach the album meter.
    pub fn push_interleaved(&mut self, samples: &[f64]) -> Result<(), Error> {
        if self.failed {
            return Err(Error::Failed);
        }
        let current = self.current.as_mut().ok_or(Error::Album("no open track"))?;
        if let Err(e) = current.push_interleaved(samples) {
            if current.failed {
                self.failed = true;
            }
            return Err(e);
        }
        if let Some(meter) = &mut self.continuous {
            if let Err(e) = meter.push_interleaved(samples) {
                self.failed = true;
                return Err(e.into());
            }
        }
        Ok(())
    }
    /// End the current track and keep only its sufficient statistics/results.
    pub fn end_track(&mut self) -> Result<(), Error> {
        if self.failed {
            return Err(Error::Failed);
        }
        let current = self.current.take().ok_or(Error::Album("no open track"))?;
        match current.finish() {
            Ok(track) => {
                self.tracks.push(track);
                Ok(())
            }
            Err(e) => {
                self.failed = true;
                Err(e)
            }
        }
    }
    /// Finish the ordered album. A track must not be left open.
    pub fn finish(self) -> Result<Album, Error> {
        if self.failed {
            return Err(Error::Failed);
        }
        if self.current.is_some() {
            return Err(Error::Album("finish the current track first"));
        }
        let continuous = self
            .continuous
            .map(|m| m.finalize().map(|x| x.finite_lu()))
            .transpose()?;
        Album::build(self.tracks, continuous)
    }
}

#[cfg(test)]
mod selection_tests {
    use super::*;
    fn make(selection: &[M], rate: u32, channels: usize) -> Analyzer {
        Analyzer::new(
            Format {
                sample_rate: rate,
                channels,
                encoding: Encoding::Float,
                loudness_roles: None,
            },
            Measurements::new(selection),
            Settings::default(),
        )
        .unwrap()
    }
    #[test]
    fn no_expensive_state_for_scalar_only_or_not_applicable() {
        let a = make(&[M::BitsInUse], 48_000, 2);
        assert!(a.grid.is_some());
        assert!(a.spectral.is_none() && a.quiet.is_none() && a.dither_quiet.is_none());
        assert!(
            matches!(a.loudness, TrackLoudness::None)
                && a.clicks.is_none()
                && a.flats.is_none()
                && a.stereo.is_none()
        );
        assert_eq!(a.block_pcm.capacity(), 0);
        let a = make(&[M::Above20Khz], 44_100, 2);
        assert!(a.spectral.is_none());
        let a = make(&[M::InterChannelDelay], 48_000, 1);
        assert!(a.spectral.is_none());
        let a = make(&[M::ShortTermMaximum], 48_000, 1);
        assert!(matches!(a.loudness, TrackLoudness::Short(_)));
    }
    const RANGE: &[M] = &[M::LoudnessRange];
    const SHORT: &[M] = &[M::ShortTermMaximum, M::LoudnessOverTime];
    const BOTH: &[M] = &[M::LoudnessRange, M::ShortTermMaximum, M::LoudnessOverTime];

    fn loudness_format(rate: u32, channels: usize) -> Format {
        let roles = [
            ChannelRole::Left,
            ChannelRole::Right,
            ChannelRole::Center,
            ChannelRole::Lfe,
            ChannelRole::LeftSurround,
            ChannelRole::RightSurround,
        ];
        Format {
            sample_rate: rate,
            channels,
            encoding: Encoding::Float,
            loudness_roles: (channels > 2).then(|| roles[..channels].to_vec()),
        }
    }

    fn loudness_pcm(rate: u32, channels: usize, frames: usize) -> Vec<f64> {
        (0..frames * channels)
            .map(|i| {
                let n = i / channels;
                let amplitude = if n < rate as usize {
                    0.01
                } else if n < 3 * rate as usize {
                    0.2
                } else {
                    0.03
                };
                amplitude * (((n % 103) as f64 / 51.0 - 1.0) + (i % channels) as f64 * 0.1)
            })
            .collect()
    }

    fn loudness_track(format: &Format, pcm: &[f64], selection: &[M], chunk: usize) -> Track {
        let mut analyzer = Analyzer::new(
            format.clone(),
            Measurements::new(selection),
            Settings::default(),
        )
        .unwrap();
        for input in pcm.chunks(chunk * format.channels) {
            analyzer.push_interleaved(input).unwrap();
        }
        analyzer.finish().unwrap()
    }

    fn check_track_loudness(both: &Track, range: &Track, short: &Track) {
        assert_eq!(both.range.map(f64::to_bits), range.range.map(f64::to_bits));
        assert_eq!(
            both.short_max.map(f64::to_bits),
            short.short_max.map(f64::to_bits)
        );
        let bits = |t: &Track| {
            t.curve
                .iter()
                .map(|p| (p.x.to_bits(), p.y.to_bits()))
                .collect::<Vec<_>>()
        };
        assert_eq!(bits(both), bits(short));
        assert!(both.curve.iter().all(|p| p.x <= both.duration_seconds()));
        assert_eq!(
            range.measure(M::ShortTermMaximum, NoiseView::default()),
            Outcome::NotRequested
        );
        assert_eq!(
            short.measure(M::LoudnessRange, NoiseView::default()),
            Outcome::NotRequested
        );
    }

    #[test]
    fn audioscopy_shared_loudness_selection_and_track_parity() {
        assert!(matches!(
            make(RANGE, 48_000, 2).loudness,
            TrackLoudness::Range(_)
        ));
        assert!(matches!(
            make(SHORT, 48_000, 2).loudness,
            TrackLoudness::Short(_)
        ));
        assert!(matches!(
            make(BOTH, 48_000, 2).loudness,
            TrackLoudness::Both(_)
        ));
        for (rate, channels) in [(8_001, 1), (8_009, 3), (44_101, 2), (48_000, 6)] {
            let format = loudness_format(rate, channels);
            for frames in [
                1,
                3 * rate as usize - 1,
                3 * rate as usize,
                5 * rate as usize + 17,
            ] {
                let pcm = loudness_pcm(rate, channels, frames);
                let both = loudness_track(&format, &pcm, BOTH, 997);
                let range = loudness_track(&format, &pcm, RANGE, 713);
                let short = loudness_track(&format, &pcm, SHORT, 509);
                check_track_loudness(&both, &range, &short);
                assert_eq!(both.curve.is_empty(), frames < 3 * rate as usize);
                let max_only = loudness_track(
                    &format,
                    &pcm,
                    &[M::LoudnessRange, M::ShortTermMaximum],
                    1021,
                );
                assert_eq!(
                    max_only.short_max.map(f64::to_bits),
                    both.short_max.map(f64::to_bits)
                );
                assert!(max_only.curve.is_empty());
                let curve_only =
                    loudness_track(&format, &pcm, &[M::LoudnessRange, M::LoudnessOverTime], 701);
                assert_eq!(curve_only.curve, both.curve);
                assert!(curve_only.short_max.is_none());
            }
        }
    }

    fn loudness_album(
        formats: &[Format],
        tracks: &[Vec<f64>],
        selection: &[M],
        chunk: usize,
    ) -> Album {
        let mut analyzer = AlbumAnalyzer::new(Measurements::new(selection), Settings::default());
        for (format, pcm) in formats.iter().zip(tracks) {
            analyzer.begin_track(format.clone()).unwrap();
            for input in pcm.chunks(chunk * format.channels) {
                analyzer.push_interleaved(input).unwrap();
            }
            analyzer.end_track().unwrap();
        }
        analyzer.finish().unwrap()
    }

    #[test]
    fn audioscopy_shared_loudness_continuous_album_parity() {
        for mixed in [false, true] {
            let first = loudness_format(8_001, 2);
            let second = if mixed {
                loudness_format(8_009, 3)
            } else {
                first.clone()
            };
            let tracks = [
                loudness_pcm(
                    first.sample_rate,
                    first.channels,
                    2 * first.sample_rate as usize,
                ),
                loudness_pcm(
                    second.sample_rate,
                    second.channels,
                    4 * second.sample_rate as usize + 17,
                ),
            ];
            let formats = [first, second];
            let both = loudness_album(&formats, &tracks, BOTH, 997);
            let range = loudness_album(&formats, &tracks, RANGE, 713);
            let short = loudness_album(&formats, &tracks, SHORT, 509);
            assert_eq!(both.loudness_method(), range.loudness_method());
            assert_eq!(
                both.loudness_method(),
                if mixed {
                    AlbumLoudnessMethod::Unavailable
                } else {
                    AlbumLoudnessMethod::Continuous
                }
            );
            assert_eq!(
                both.measure(M::LoudnessRange, NoiseView::default()),
                range.measure(M::LoudnessRange, NoiseView::default())
            );
            for metric in SHORT {
                assert_eq!(
                    both.measure(*metric, NoiseView::default()),
                    short.measure(*metric, NoiseView::default())
                );
            }
            for ((b, r), s) in both.tracks().iter().zip(range.tracks()).zip(short.tracks()) {
                check_track_loudness(b, r, s);
            }
            // The first track is shorter than a full observation. Its native
            // LRA EOF continuation must not leak into either short-term curve.
            assert!(both.tracks()[0].curve.is_empty());
            assert!(!both.tracks()[1].curve.is_empty());
        }
    }

    #[test]
    fn empty_selection_has_no_measurement_state() {
        let mut a = make(&[], 48_000, 2);
        a.push_interleaved(&[0.2, -0.2]).unwrap();
        let t = a.finish().unwrap();
        for m in M::ALL {
            assert_eq!(t.measure(m, NoiseView::default()), Outcome::NotRequested);
        }
    }
}
