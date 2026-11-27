use crate::temporal::{classify_dither, Grid};
use crate::{db, Band, BassWidth, Encoding, NoiseView, Point, Tone, Weighting};
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::{f64::consts::PI, sync::Arc};

type C = Complex<f64>;
const BANDWIDTH_RELATIVE_POWER: f64 = 1e-8;

#[cfg(all(
    target_arch = "x86_64",
    any(
        test,
        feature = "experimental-spectral-avx2",
        feature = "experimental-spectral-avx512"
    )
))]
#[path = "spectral_simd.rs"]
mod spectral_simd;

#[cfg(feature = "experimental-stereo-spectrum-split")]
#[path = "spectral_stereo.rs"]
mod spectral_stereo;

// DC and Nyquist have one-sided weight one; every interior bin has
// weight two. Isolating the two endpoints leaves a contiguous, independent
// binary64 loop that the compiler can vectorize without a per-bin branch.
// Keep the multiplication/addition graph and channel accumulation order.
#[inline]
fn accumulate_power(bins: &[C], power: &mut [f64]) {
    debug_assert_eq!(bins.len(), power.len());
    debug_assert!(power.len() >= 2);
    let last = power.len() - 1;
    power[0] += bins[0].norm_sqr();
    for (sum, x) in power[1..last].iter_mut().zip(&bins[1..last]) {
        *sum += 2.0 * x.norm_sqr();
    }
    power[last] += bins[last].norm_sqr();
}

#[derive(Clone, Debug)]
pub(crate) struct Spectrum {
    pub rate: u32,
    pub n: usize,
    // Sum of complementary-window energy per channel. Divide by real frames.
    pub sums: Vec<f64>,
    pub frames: u64,
}
impl Spectrum {
    fn empty(rate: u32, n: usize) -> Self {
        Self {
            rate,
            n,
            sums: vec![0.0; n / 2 + 1],
            frames: 0,
        }
    }
    pub fn df(&self) -> f64 {
        f64::from(self.rate) / self.n as f64
    }
    pub fn duration(&self) -> f64 {
        self.frames as f64 / f64::from(self.rate)
    }
    pub fn total(&self) -> f64 {
        self.sums.iter().sum::<f64>() / self.frames.max(1) as f64
    }
    pub fn power(&self, lo: f64, hi: f64, w: Weighting) -> f64 {
        if hi <= lo {
            return 0.0;
        }
        let df = self.df();
        let start = ((lo / df).ceil() as usize).min(self.sums.len());
        let mut end = ((hi / df).ceil() as usize).min(self.sums.len());
        if hi >= f64::from(self.rate) * 0.5 {
            end = self.sums.len();
        }
        if end <= start {
            return 0.0;
        }
        if w == Weighting::Unweighted {
            self.sums[start..end].iter().sum::<f64>() / self.frames.max(1) as f64
        } else {
            self.sums[start..end]
                .iter()
                .enumerate()
                .map(|(i, p)| p * weight_power((i + start) as f64 * df, w))
                .sum::<f64>()
                / self.frames.max(1) as f64
        }
    }
    fn add(&mut self, other: &Self) {
        debug_assert_eq!((self.rate, self.n), (other.rate, other.n));
        for (a, b) in self.sums.iter_mut().zip(&other.sums) {
            *a += b;
        }
        self.frames += other.frames;
    }
    fn clear(&mut self) {
        self.sums.fill(0.0);
        self.frames = 0;
    }
    // The caller supplies the recording-wide accumulated bin-power floor.
    // None means no material bin; Some(0.0) is a real occupied DC bin.
    fn bandwidth_at_floor(&self, floor: f64) -> Option<f64> {
        self.sums
            .iter()
            .rposition(|&v| v > 0.0 && v >= floor)
            .map(|bin| bin as f64 * self.df())
    }
    fn density(&self, lo: f64, hi: f64) -> f64 {
        if hi <= lo {
            return 0.0;
        }
        self.power(lo, hi, Weighting::Unweighted) / (hi - lo)
    }
    pub fn source_rate(&self) -> Option<u32> {
        // Require occupied content near the cutoff and a sharp, sustained drop.
        let ny = f64::from(self.rate) / 2.0;
        let total = self.total();
        if total == 0.0 {
            return None;
        }
        [
            8_000, 11_025, 12_000, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000, 88_200, 96_000,
            176_400, 192_000, 352_800, 384_000,
        ]
        .into_iter()
        .rev()
        .find(|&rate| {
            let edge = f64::from(rate) / 2.0;
            if edge * 1.06 >= ny {
                return false;
            }
            let below = self.density(edge * 0.94, edge * 0.985);
            let prior = self.density(edge * 0.88, edge * 0.94);
            let above = self.density(edge * 1.015, (edge * 1.06).min(ny));
            let occupied = (0..4).all(|i| {
                let start = edge * (0.94 + i as f64 * 0.01);
                self.density(start, start + edge * 0.01) > below * 0.03
            });
            below > total / ny * 1e-4
                && prior > below * 0.01
                && occupied
                && above < below * 0.001
                && self.power(edge * 1.015, ny, Weighting::Unweighted) < total * 1e-4
        })
    }
    pub fn lossy_pattern(&self) -> bool {
        let ny = f64::from(self.rate) / 2.0;
        let total = self.total();
        if total == 0.0 || ny < 12_000.0 {
            return false;
        }
        // Codec-like shelf: abrupt non-standard cutoff at common encoder
        // lowpass settings, with appreciable energy immediately below it.
        let shelf = [
            12_000.0, 13_000.0, 14_000.0, 15_000.0, 16_000.0, 17_000.0, 18_000.0, 19_000.0,
            20_000.0,
        ]
        .into_iter()
        .any(|edge| {
            if edge + 500.0 >= ny {
                return false;
            }
            let below = self.density(edge - 400.0, edge - 100.0);
            below > total / ny * 1e-4
                && self.density(edge + 100.0, edge + 400.0) < below * 1e-3
                && self.power(edge + 100.0, ny, Weighting::Unweighted) < total * 1e-4
        });
        // Spectral holes between occupied adjacent bands. Natural notches can
        // also meet this heuristic: this is a pattern, not codec attribution.
        let gap = (8..(ny.min(20_000.0) / 500.0) as usize - 1).any(|i| {
            let f = i as f64 * 500.0;
            let l = self.density(f - 500.0, f);
            let r = self.density(f + 500.0, f + 1000.0);
            let neighbor = l.min(r);
            neighbor > total / ny * 1e-3 && self.density(f, f + 500.0) < neighbor * 1e-3
        });
        shelf || gap
    }
    pub fn tone(&self, view: NoiseView) -> Option<Tone> {
        let (lo, hi) = limits(view.band, self.rate);
        let df = self.df();
        let mut best: Option<Tone> = None;
        for i in 1..self.sums.len() - 1 {
            let f = i as f64 * df;
            if f < lo || f > hi {
                continue;
            }
            let p = self.sums[i];
            if p <= 0.0 || p < self.sums[i - 1] || p <= self.sums[i + 1] {
                continue;
            }
            let start = i.saturating_sub(8);
            let end = (i + 9).min(self.sums.len());
            let mut background = 0.0;
            let mut count = 0;
            for j in start..end {
                if j.abs_diff(i) >= 3 {
                    background += self.sums[j];
                    count += 1;
                }
            }
            if count == 0 || p < 16.0 * background / count as f64 {
                continue;
            }
            // Quadratic interpolation of log power for non-bin-centered lines.
            let a = self.sums[i - 1].max(f64::MIN_POSITIVE).ln();
            let b = p.ln();
            let c = self.sums[i + 1].max(f64::MIN_POSITIVE).ln();
            let den = a - 2.0 * b + c;
            let offset = if den < 0.0 {
                (0.5 * (a - c) / den).clamp(-0.5, 0.5)
            } else {
                0.0
            };
            let frequency_hz = (i as f64 + offset) * df;
            if frequency_hz < lo || frequency_hz > hi {
                continue;
            }
            let level_dbfs = db(
                self.sums[i - 1..=i + 1].iter().sum::<f64>() / self.frames as f64
                    * weight_power(frequency_hz, view.weighting),
            );
            if best.is_none_or(|t| level_dbfs > t.level_dbfs) {
                best = Some(Tone {
                    frequency_hz,
                    level_dbfs,
                });
            }
        }
        let last = self.sums.len() - 1;
        let f = last as f64 * df;
        if f >= lo && f <= hi && self.sums[last] > 0.0 && self.sums[last] >= self.sums[last - 1] {
            let bg = &self.sums[last.saturating_sub(8)..last.saturating_sub(2)];
            if !bg.is_empty() && self.sums[last] > 16.0 * bg.iter().sum::<f64>() / bg.len() as f64 {
                let level_dbfs = db((self.sums[last] + self.sums[last - 1]) / self.frames as f64
                    * weight_power(f, view.weighting));
                if best.is_none_or(|t| level_dbfs > t.level_dbfs) {
                    best = Some(Tone {
                        frequency_hz: f,
                        level_dbfs,
                    });
                }
            }
        }
        best
    }
}

pub(crate) fn limits(band: Band, rate: u32) -> (f64, f64) {
    let ny = f64::from(rate) / 2.0;
    match band {
        Band::Audible => (20.0, ny.min(20_000.0)),
        Band::ToNyquist => (20.0, ny),
        Band::Full => (0.0, ny),
    }
}

pub(crate) fn weight_power(f: f64, w: Weighting) -> f64 {
    if w == Weighting::Unweighted {
        return 1.0;
    }
    if f == 0.0 {
        return 0.0;
    }
    match w {
        Weighting::Unweighted => 1.0,
        Weighting::A => {
            let s = f * f;
            let a = 12194.0_f64.powi(2) * s * s
                / ((s + 20.6_f64.powi(2))
                    * ((s + 107.7_f64.powi(2)) * (s + 737.9_f64.powi(2))).sqrt()
                    * (s + 12194.0_f64.powi(2)));
            a * a * 10.0_f64.powf(0.2)
        }
        Weighting::ItuR468 => {
            // BS.468-4 analog frequency response (normalized near 1 kHz).
            let s = f * f;
            let h1 = ((-4.737338981378384e-24 * s + 2.043828333606125e-15) * s
                - 1.363894795463638e-7)
                * s
                + 1.0;
            let h2 = ((1.306612257412824e-19 * s - 2.118150887518656e-11) * s
                + 5.559488023498642e-4)
                * f;
            let r = 1.246332637532143e-4 * f / h1.hypot(h2);
            r * r * 10.0_f64.powf(1.822)
        }
    }
}

// Average linear power across real time, not a mean of dB values. For mixed
// clocks, each source contributes its native bins; no PCM resampling required.
pub(crate) fn pooled_power(
    spectra: &[&Spectrum],
    view: NoiseView,
    band: (f64, f64),
) -> Option<f64> {
    let mut sum = 0.0;
    let mut duration = 0.0;
    let mut supported = false;
    for s in spectra {
        let (lo, hi) = limits(view.band, s.rate);
        let l = lo.max(band.0);
        let h = hi.min(band.1);
        let d = s.duration();
        duration += d;
        if h <= l {
            continue;
        }
        supported = true;
        sum += d * s.power(l, h, view.weighting);
    }
    (duration > 0.0 && supported).then(|| sum / duration)
}

pub(crate) fn curve(spectra: &[&Spectrum], view: NoiseView) -> Vec<Point> {
    if spectra.is_empty() {
        return Vec::new();
    }
    // Map to the finest native spacing. Linear splatting preserves total bin
    // energy. Graph ordinate is density, not the power of a variable-width bin.
    let df = spectra.iter().map(|s| s.df()).fold(f64::INFINITY, f64::min);
    let max = spectra
        .iter()
        .map(|s| limits(view.band, s.rate).1)
        .fold(0.0, f64::max);
    let duration: f64 = spectra.iter().map(|s| s.duration()).sum();
    let min = if view.band == Band::Full { 0.0 } else { 20.0 };
    let first = (min / df).ceil() as usize;
    let last = (max / df).floor() as usize;
    if first > last {
        return Vec::new();
    }
    let mut bins = vec![0.0; last - first + 1];
    for s in spectra {
        let (lo, hi) = limits(view.band, s.rate);
        for (i, &p) in s.sums.iter().enumerate() {
            let f = i as f64 * s.df();
            if f < lo || (f >= hi && !(f == hi && i == s.n / 2)) {
                continue;
            }
            let x = (f / df).clamp(first as f64, last as f64) - first as f64;
            let j = (x.floor() as usize).min(bins.len() - 1);
            let frac = if j + 1 < bins.len() { x.fract() } else { 0.0 };
            let power = p / f64::from(s.rate) * weight_power(f, view.weighting);
            bins[j] += power * (1.0 - frac);
            if frac > 0.0 {
                bins[j + 1] += power * frac;
            }
        }
    }
    bins.into_iter()
        .enumerate()
        .map(|(i, p)| Point {
            x: (i + first) as f64 * df,
            y: db(p / (duration * df)),
        })
        .collect()
}

// One endpoint view, in descending bin order. Retain the power of each
// locally significant peak until the programme-wide reference is known.
// Power is one-sided channel-mean energy (the same units as Spectrum::sums).
struct BandwidthEdge {
    resolution_hz: f64,
    peaks: Vec<(usize, f64)>,
}
impl BandwidthEdge {
    fn peak_hz(&self, floor: f64, bin_hz: f64) -> f64 {
        self.peaks
            .iter()
            .find(|&&(_, power)| power >= floor)
            .map_or(0.0, |&(bin, _)| bin as f64 * bin_hz)
    }
}

pub(crate) struct SpectralEngine {
    rate: u32,
    channels: usize,
    n: usize,
    fft: Arc<dyn Fft<f64>>,
    scratch: Vec<C>,
    work: Vec<C>,
    left: Vec<C>,
    window: Vec<f64>,
    block: Vec<f64>,
    // Window starts half a window before frame zero. Padding is never counted.
    leading: bool,
    pending_frames: usize,
    pub spectrum: Spectrum,
    bass: Option<(Vec<f64>, Vec<f64>)>,
    delay: Option<(Vec<C>, f64, f64)>,
    #[cfg(feature = "experimental-stereo-spectrum-split")]
    delay_band: std::ops::Range<usize>,
    stereo_complete: bool,
    delay_total: [f64; 2],
    collect_power: bool,
    #[cfg(all(
        target_arch = "x86_64",
        any(
            feature = "experimental-spectral-avx2",
            feature = "experimental-spectral-avx512"
        )
    ))]
    power_backend: spectral_simd::Backend,
    // Fully observed windows for cutoff heuristics, which should not mistake
    // the finite-record discontinuity for a source bandwidth.
    interior: Option<Spectrum>,
    // At most five endpoint views, allocated only for Bandwidth. Their
    // candidate powers never enter the retained power bins.
    bandwidth_edges: Option<Vec<BandwidthEdge>>,
    lossy: bool,
    lossy_windows: u64,
    lossy_hits: u64,
    temporary: Vec<f64>,
}
impl SpectralEngine {
    // Private constructor takes the exact selected spectral measurement flags.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        rate: u32,
        channels: usize,
        collect_power: bool,
        bass: bool,
        delay: bool,
        lossy: bool,
        extent: bool,
        bandwidth: bool,
    ) -> Self {
        let n = ((rate as usize).div_ceil(6)).next_power_of_two();
        let fft = FftPlanner::<f64>::new().plan_fft_forward(n);
        let scratch = vec![C::default(); fft.get_inplace_scratch_len()];
        let bins = n / 2 + 1;
        Self {
            rate,
            channels,
            n,
            fft,
            scratch,
            work: vec![C::default(); n],
            left: if bass || delay {
                vec![C::default(); bins]
            } else {
                Vec::new()
            },
            // Vorbis/sine-of-Hann window. w[i]^2 + w[i+h]^2 = 1,
            // h=N/2, since the two Hann coefficients sum to one.
            window: (0..n)
                .map(|i| (PI / 2.0 * (PI * (i as f64 + 0.5) / n as f64).sin().powi(2)).sin())
                .collect(),
            block: Vec::with_capacity(n * channels),
            leading: true,
            pending_frames: 0,
            spectrum: if collect_power {
                Spectrum::empty(rate, n)
            } else {
                Spectrum {
                    rate,
                    n,
                    sums: Vec::new(),
                    frames: 0,
                }
            },
            bass: bass.then(|| (vec![0.0; bins], vec![0.0; bins])),
            delay: delay.then(|| (vec![C::default(); bins], 0.0, 0.0)),
            #[cfg(feature = "experimental-stereo-spectrum-split")]
            delay_band: if delay {
                spectral_stereo::delay_band(rate, n)
            } else {
                0..0
            },
            stereo_complete: false,
            delay_total: [0.0; 2],
            collect_power,
            #[cfg(all(
                target_arch = "x86_64",
                any(
                    feature = "experimental-spectral-avx2",
                    feature = "experimental-spectral-avx512"
                )
            ))]
            power_backend: if collect_power
                || (cfg!(feature = "experimental-stereo-spectrum-split") && (bass || delay))
            {
                spectral_simd::Backend::select()
            } else {
                spectral_simd::Backend::portable()
            },
            interior: extent.then(|| Spectrum::empty(rate, n)),
            bandwidth_edges: bandwidth.then(|| Vec::with_capacity(5)),
            lossy,
            lossy_windows: 0,
            lossy_hits: 0,
            temporary: if collect_power {
                vec![0.0; bins]
            } else {
                Vec::new()
            },
        }
    }
    pub fn push(&mut self, mut pcm: &[f64]) {
        debug_assert_eq!(pcm.len() % self.channels, 0);
        if self.collect_power {
            self.spectrum.frames += (pcm.len() / self.channels) as u64;
        }
        while !pcm.is_empty() {
            let target = if self.leading {
                self.n / 2 * self.channels
            } else {
                self.n * self.channels
            };
            let take = pcm.len().min(target - self.block.len());
            self.block.extend_from_slice(&pcm[..take]);
            self.pending_frames += take / self.channels;
            pcm = &pcm[take..];
            if self.block.len() == target {
                if self.leading {
                    self.capture_bandwidth_edges(true);
                }
                self.process_window(!self.leading);
                self.advance();
            }
        }
    }
    fn advance(&mut self) {
        if !self.leading {
            let hop = (self.n / 2 * self.channels).min(self.block.len());
            let remain = self.block.len() - hop;
            self.block.copy_within(hop.., 0);
            self.block.truncate(remain);
        }
        // For the initial window, all buffered real frames are the overlap.
        self.leading = false;
        self.pending_frames = 0;
    }
    // Finish at most two windows. Every real frame participates in exactly
    // two windows, including a partial first/last hop. No synthetic frames
    // contribute to the denominator. Repeated flush is a no-op.
    pub fn flush(&mut self) {
        if !self.block.is_empty() {
            self.capture_bandwidth_edges(false);
        }
        if self.pending_frames > 0 {
            self.process_window(false);
            self.advance();
        }
        if !self.leading && !self.block.is_empty() {
            self.process_window(false);
        }
        self.block.clear();
        self.leading = true;
        self.pending_frames = 0;
    }
    fn process_window(&mut self, complete: bool) {
        let offset = if self.leading { self.n / 2 } else { 0 };
        let end = offset + self.block.len() / self.channels;
        debug_assert!(end <= self.n);
        // Silence contributes time and zero power; skip FFTs but retain
        // complete-window duration for the inference spectrum.
        if complete {
            if let Some(s) = &mut self.interior {
                s.frames += (self.n / 2) as u64;
            }
        }
        if self.block.iter().all(|&x| x == 0.0) {
            return;
        }
        self.temporary.fill(0.0);
        let need_stereo = self.bass.is_some() || self.delay.is_some();
        // The zero-extension discontinuity is relevant to finite-record
        // power but can create artificial stereo-band/correlation evidence.
        // Use fully observed windows for these heuristic rows when possible;
        // short tracks fall back to their edge windows.
        if need_stereo && complete && !self.stereo_complete {
            if let Some((mid, side)) = &mut self.bass {
                mid.fill(0.0);
                side.fill(0.0);
            }
            if let Some((cross, le, re)) = &mut self.delay {
                cross.fill(C::default());
                *le = 0.0;
                *re = 0.0;
            }
            self.delay_total = [0.0; 2];
            self.stereo_complete = true;
        }
        let stereo = need_stereo && (complete || !self.stereo_complete);
        if !self.collect_power && !stereo {
            return;
        }
        #[cfg(not(feature = "experimental-stereo-spectrum-split"))]
        let df = f64::from(self.rate) / self.n as f64;
        for ch in 0..self.channels {
            self.work[..offset].fill(C::default());
            self.work[end..].fill(C::default());
            for (i, frame) in self.block.chunks_exact(self.channels).enumerate() {
                let j = i + offset;
                self.work[j] = C::new(frame[ch] * self.window[j], 0.0);
            }
            self.fft
                .process_with_scratch(&mut self.work, &mut self.scratch);
            if !stereo {
                // Power-only and quiet-spectrum paths do not need the
                // stereo loop. The no-power/no-stereo case returned above.
                #[cfg(all(
                    target_arch = "x86_64",
                    any(
                        feature = "experimental-spectral-avx2",
                        feature = "experimental-spectral-avx512"
                    )
                ))]
                self.power_backend
                    .accumulate(&self.work[..=self.n / 2], &mut self.temporary);
                #[cfg(not(all(
                    target_arch = "x86_64",
                    any(
                        feature = "experimental-spectral-avx2",
                        feature = "experimental-spectral-avx512"
                    )
                )))]
                accumulate_power(&self.work[..=self.n / 2], &mut self.temporary);
                continue;
            }
            #[cfg(feature = "experimental-stereo-spectrum-split")]
            self.accumulate_stereo_channel(ch);
            #[cfg(not(feature = "experimental-stereo-spectrum-split"))]
            for i in 0..=self.n / 2 {
                let factor = if i == 0 || i == self.n / 2 { 1.0 } else { 2.0 };
                let x = self.work[i];
                if self.collect_power {
                    self.temporary[i] += factor * x.norm_sqr();
                }
                if stereo && ch == 0 {
                    self.left[i] = x;
                }
                if stereo && ch == 1 {
                    let l = self.left[i];
                    if let Some((mid, side)) = &mut self.bass {
                        mid[i] += factor * (l + x).norm_sqr();
                        side[i] += factor * (l - x).norm_sqr();
                    }
                    if let Some((cross, le, re)) = &mut self.delay {
                        self.delay_total[0] += factor * l.norm_sqr();
                        self.delay_total[1] += factor * x.norm_sqr();
                        let f = i as f64 * df;
                        if f >= 2000.0 && f <= 10_000.0_f64.min(f64::from(self.rate) * 0.45) {
                            cross[i] += l.conj() * x;
                            *le += factor * l.norm_sqr();
                            *re += factor * x.norm_sqr();
                        }
                    }
                }
            }
        }
        if self.collect_power {
            // One-sided Parseval: sum |DFT(w*x)|^2 / N is sum |w*x|^2.
            // Complementary squared weights sum to one over real samples.
            // Fixed normalization cannot transfer energy between frequencies.
            let scale = 1.0 / (self.n as f64 * self.channels as f64);
            for (sum, p) in self.spectrum.sums.iter_mut().zip(&mut self.temporary) {
                *p *= scale;
                *sum += *p;
            }
            if complete {
                if let Some(s) = &mut self.interior {
                    for (sum, p) in s.sums.iter_mut().zip(&self.temporary) {
                        *sum += *p;
                    }
                }
                if self.lossy {
                    let local = Spectrum {
                        rate: self.rate,
                        n: self.n,
                        sums: std::mem::take(&mut self.temporary),
                        frames: (self.n / 2) as u64,
                    };
                    self.lossy_windows += 1;
                    if local.lossy_pattern() {
                        self.lossy_hits += 1;
                    }
                    self.temporary = local.sums;
                }
            }
        }
    }
    #[cfg(feature = "experimental-stereo-spectrum-split")]
    fn accumulate_stereo_channel(&mut self, ch: usize) {
        let bins = &self.work[..=self.n / 2];
        if self.collect_power {
            #[cfg(all(
                target_arch = "x86_64",
                any(
                    feature = "experimental-spectral-avx2",
                    feature = "experimental-spectral-avx512"
                )
            ))]
            self.power_backend.accumulate(bins, &mut self.temporary);
            #[cfg(not(all(
                target_arch = "x86_64",
                any(
                    feature = "experimental-spectral-avx2",
                    feature = "experimental-spectral-avx512"
                )
            )))]
            accumulate_power(bins, &mut self.temporary);
        }
        if ch == 0 {
            self.left.copy_from_slice(bins);
        } else if ch == 1 {
            if let Some((mid, side)) = &mut self.bass {
                #[cfg(all(
                    target_arch = "x86_64",
                    any(
                        feature = "experimental-spectral-avx2",
                        feature = "experimental-spectral-avx512"
                    )
                ))]
                self.power_backend.bass(&self.left, bins, mid, side);
                #[cfg(not(all(
                    target_arch = "x86_64",
                    any(
                        feature = "experimental-spectral-avx2",
                        feature = "experimental-spectral-avx512"
                    )
                )))]
                spectral_stereo::bass(&self.left, bins, mid, side);
            }
            if let Some((cross, le, re)) = &mut self.delay {
                let band = self.delay_band.clone();
                let l = &self.left[band.clone()];
                let r = &bins[band.clone()];
                let c = &mut cross[band.clone()];
                #[cfg(all(
                    target_arch = "x86_64",
                    any(
                        feature = "experimental-spectral-avx2",
                        feature = "experimental-spectral-avx512"
                    )
                ))]
                self.power_backend.cross(l, r, c);
                #[cfg(not(all(
                    target_arch = "x86_64",
                    any(
                        feature = "experimental-spectral-avx2",
                        feature = "experimental-spectral-avx512"
                    )
                )))]
                spectral_stereo::cross(l, r, c);
                spectral_stereo::delay_energy(
                    &self.left,
                    bins,
                    band,
                    &mut self.delay_total,
                    le,
                    re,
                );
            }
        }
    }

    // Endpoint inference is a separate, small constant amount of DSP per
    // file. The real first/last half-hop is tapered on its actual support
    // before zero-padding, rather than abruptly truncating the main window.
    // The shorter endpoint view preserves short transients that even the
    // half-hop taper could attenuate. Both reuse the existing FFT and scratch.
    fn capture_bandwidth_edges(&mut self, first: bool) {
        if self.bandwidth_edges.is_none() {
            return;
        }
        let frames = self.block.len() / self.channels;
        if frames == 0 {
            return;
        }
        let half = frames.min(self.n / 2);
        let short = half.min(self.n / 8);
        if first {
            self.bandwidth_edge(0, half);
            if short < half {
                self.bandwidth_edge(0, short);
            }
        } else if self.leading {
            // Entire file is shorter than the first half-hop.
            self.bandwidth_edge(0, frames);
            if short < frames {
                self.bandwidth_edge(0, short);
                self.bandwidth_edge(frames - short, short);
            }
        } else {
            // If no complete window exists, the buffer still holds the
            // entire file. Also examine its full real support so the middle
            // between two tapered half-hops cannot disappear.
            if self.interior.as_ref().is_some_and(|s| s.frames == 0) && frames > half {
                self.bandwidth_edge(0, frames);
            }
            self.bandwidth_edge(frames - half, half);
            if short < half {
                self.bandwidth_edge(frames - short, short);
            }
        }
    }
    fn bandwidth_edge(&mut self, start: usize, frames: usize) {
        let end = start + frames;
        debug_assert!(frames > 0 && frames <= self.n && end * self.channels <= self.block.len());
        self.temporary.fill(0.0);
        let pcm = &self.block[start * self.channels..end * self.channels];
        if pcm.iter().any(|&x| x != 0.0) {
            // Four-term Blackman-Harris: approximately -92 dB peak
            // sidelobes. Recompute a taper only for these at most five
            // windows per file, independent of programme duration.
            for ch in 0..self.channels {
                for (i, frame) in pcm.chunks_exact(self.channels).enumerate() {
                    let phase = 2.0 * PI * (i as f64 + 0.5) / frames as f64;
                    let w = 0.35875 - 0.48829 * phase.cos() + 0.14128 * (2.0 * phase).cos()
                        - 0.01168 * (3.0 * phase).cos();
                    self.work[i] = C::new(frame[ch] * w, 0.0);
                }
                self.work[frames..].fill(C::default());
                self.fft
                    .process_with_scratch(&mut self.work, &mut self.scratch);
                for (i, p) in self.temporary.iter_mut().enumerate() {
                    let factor = if i == 0 || i == self.n / 2 { 1.0 } else { 2.0 };
                    *p += factor * self.work[i].norm_sqr();
                }
            }
        }
        let max = self.temporary.iter().copied().fold(0.0, f64::max);
        // Require an actual occupied spectral peak rather than the outer
        // skirt of the much wider endpoint window's main lobe. Include
        // DC/Nyquist as one-sided extrema. Retain all locally eligible peaks,
        // not just the highest frequency: a weaker upper peak may fail the
        // programme-wide test while a lower one survives.
        let mut peaks = Vec::new();
        if max > 0.0 {
            let last = self.temporary.len() - 1;
            let scale = 1.0 / (self.n as f64 * self.channels as f64);
            let local_floor = max * BANDWIDTH_RELATIVE_POWER;
            for i in (0..=last).rev() {
                let p = self.temporary[i];
                if p >= local_floor
                    && (i == 0 || p >= self.temporary[i - 1])
                    && (i == last || p > self.temporary[i + 1])
                {
                    // Same fixed Parseval normalization as process_window.
                    // Do not renormalize by endpoint level or support length:
                    // the tapered energy must compete with the whole record.
                    peaks.push((i, p * scale));
                }
            }
        }
        if let Some(edges) = &mut self.bandwidth_edges {
            edges.push(BandwidthEdge {
                resolution_hz: f64::from(self.rate) / frames as f64,
                peaks,
            });
        }
    }
    pub fn bandwidth(&self) -> f64 {
        // Use one recording-wide reference for interior and endpoint views. Both
        // the full accumulated spectrum and endpoint candidates contain
        // energy, not per-window mean power. Dividing either by the full
        // frame count would cancel in this relative test. The reference
        // also covers recordings whose only content is at the endpoints.
        let max = self.spectrum.sums.iter().copied().fold(0.0, f64::max);
        if max == 0.0 {
            return 0.0;
        }
        let floor = max * BANDWIDTH_RELATIVE_POWER;
        let Some(edges) = &self.bandwidth_edges else {
            return self
                .inference_spectrum()
                .bandwidth_at_floor(floor)
                .unwrap_or(0.0);
        };
        let bin_hz = self.spectrum.df();
        // A complete window containing only negligible energy is not a
        // material interior observation. Use the same floor for both paths.
        // If no interior bin qualifies, use the longest actual-support taper,
        // not a zero-extension edge.
        let base = self
            .interior
            .as_ref()
            .filter(|s| s.frames > 0)
            .and_then(|s| s.bandwidth_at_floor(floor))
            .unwrap_or_else(|| {
                let finest = edges
                    .iter()
                    .map(|e| e.resolution_hz)
                    .fold(f64::INFINITY, f64::min);
                edges
                    .iter()
                    .filter(|e| e.resolution_hz == finest)
                    .map(|e| e.peak_hz(floor, bin_hz))
                    .fold(0.0, f64::max)
            });
        // An endpoint may extend bandwidth only beyond a full main-lobe
        // half-width of its physical (not zero-padded) frequency resolution.
        // This avoids broadening an already observed stationary cutoff.
        edges
            .iter()
            .filter_map(|e| {
                let peak = e.peak_hz(floor, bin_hz);
                (peak > base + 4.0 * e.resolution_hz).then_some(peak)
            })
            .fold(base, f64::max)
    }
    pub fn inference_spectrum(&self) -> &Spectrum {
        // Complete windows exclude the artificial zero-extension edges. If
        // no nonzero complete-window evidence exists, use the full spectrum.
        self.interior
            .as_ref()
            .filter(|s| s.frames > 0 && s.total() > 0.0)
            .unwrap_or(&self.spectrum)
    }
    pub fn lossy_found(&self) -> bool {
        self.lossy
            && (self.inference_spectrum().lossy_pattern()
                || (self.lossy_hits >= 3 && self.lossy_hits * 4 >= self.lossy_windows))
    }
    // Quiet uses only power; reset all stream-boundary state for each passage.
    pub fn reset(&mut self) {
        self.block.clear();
        self.leading = true;
        self.pending_frames = 0;
        self.spectrum.clear();
        if let Some(s) = &mut self.interior {
            s.clear();
        }
        if let Some(edges) = &mut self.bandwidth_edges {
            edges.clear();
        }
        self.lossy_hits = 0;
        self.lossy_windows = 0;
        self.stereo_complete = false;
        self.delay_total = [0.0; 2];
        if let Some((m, s)) = &mut self.bass {
            m.fill(0.0);
            s.fill(0.0);
        }
        if let Some((c, l, r)) = &mut self.delay {
            c.fill(C::default());
            *l = 0.0;
            *r = 0.0;
        }
    }
    pub fn bass_width(&self) -> BassWidth {
        let Some((mid, side)) = &self.bass else {
            return BassWidth::Full;
        };
        let df = self.spectrum.df();
        let max = mid.iter().zip(side).map(|(a, b)| a + b).fold(0.0, f64::max);
        let mut occupied_mono = false;
        // Fixed 25 Hz bands, ignore unoccupied bands; first material side
        // energy above -30 dB of mid ends the mono region.
        for i in 0..40 {
            let lo = i as f64 * 25.0;
            let hi = lo + 25.0;
            let a = (lo / df).ceil() as usize;
            let b = ((hi / df).ceil() as usize).min(mid.len());
            if a >= b {
                continue;
            }
            let m: f64 = mid[a..b].iter().sum();
            let s: f64 = side[a..b].iter().sum();
            if m + s <= max * 1e-8 {
                continue;
            }
            if s > m * 1e-3 {
                return if occupied_mono {
                    BassWidth::MonoBelow(lo)
                } else {
                    BassWidth::Full
                };
            }
            occupied_mono = true;
        }
        if occupied_mono {
            BassWidth::MonoBelow(1000.0)
        } else {
            BassWidth::Full
        }
    }
    pub fn delay_us(&self) -> Option<f64> {
        let (cross, le, re) = self.delay.as_ref()?;
        let norm = le.sqrt() * re.sqrt();
        // An out-of-band tone can leak a tiny but perfectly correlated
        // amount into the upper band. Require material evidence in both.
        if norm == 0.0 || *le < self.delay_total[0] * 1e-6 || *re < self.delay_total[1] * 1e-6 {
            return None;
        }
        let mut bins = vec![C::default(); self.n];
        bins[..cross.len()].copy_from_slice(cross);
        for i in 1..self.n / 2 {
            bins[self.n - i] = cross[i].conj();
        }
        FftPlanner::<f64>::new()
            .plan_fft_inverse(self.n)
            .process(&mut bins);
        let max_lag = (self.rate as usize / 1000).max(1) as isize;
        let corr = |lag: isize| bins[lag.rem_euclid(self.n as isize) as usize].re / norm;
        let lag = (-max_lag..=max_lag).max_by(|a, b| corr(*a).abs().total_cmp(&corr(*b).abs()))?;
        let peak = corr(lag).abs();
        if peak < 0.5 || lag.abs() == max_lag {
            return None;
        }
        // Require a unique peak outside the upper-band main lobe. Narrowband
        // periodic signals with several equally good delays are ambiguous.
        let exclusion = (f64::from(self.rate) / 8000.0).ceil() as isize;
        let rival = (-max_lag..=max_lag)
            .filter(|j| (j - lag).abs() > exclusion)
            .map(|j| corr(j).abs())
            .fold(0.0, f64::max);
        if rival > peak * 0.95 {
            return None;
        }
        let a = corr(lag - 1).abs();
        let c = corr(lag + 1).abs();
        let d = a - 2.0 * peak + c;
        let offset = if d < 0.0 {
            (0.5 * (a - c) / d).clamp(-0.5, 0.5)
        } else {
            0.0
        };
        Some((lag as f64 + offset) * 1e6 / f64::from(self.rate))
    }
}

// Quiet runs are accumulated into a candidate spectrum, then committed only
// if their real frame count meets the configured duration. A rejected run
// never contributes to the pooled floor. Memory is independent of gap length.
pub(crate) struct Quiet {
    engine: SpectralEngine,
    pub pooled: Spectrum,
    min_frames: u64,
    run_frames: u64,
    encoding: Option<Encoding>,
    run_grid: Option<Grid>,
    run_peak: f64,
    run_signs: [u64; 2],
    pub signs: [u64; 2],
}
impl Quiet {
    pub fn new(rate: u32, channels: usize, seconds: f64, encoding: Option<Encoding>) -> Self {
        let engine = SpectralEngine::new(rate, channels, true, false, false, false, false, false);
        let pooled = Spectrum::empty(rate, engine.n);
        Self {
            engine,
            pooled,
            min_frames: (seconds * f64::from(rate)).ceil() as u64,
            run_frames: 0,
            encoding,
            run_grid: encoding.map(Grid::new),
            run_peak: 0.0,
            run_signs: [0; 2],
            signs: [0; 2],
        }
    }
    fn append(&mut self, pcm: &[f64]) {
        if pcm.is_empty() {
            return;
        }
        if let Some(enc) = self.encoding {
            // Filter candidate fragments before doing an FFT. Recheck the
            // entire candidate grid before committing, not just each block.
            let mut local = Grid::new(enc);
            local.push(pcm);
            let peak = pcm.iter().map(|x| x.abs()).fold(0.0, f64::max);
            let mut merged = self.run_grid.as_ref().expect("dither grid").clone();
            merged.merge(&local);
            let combined_peak = self.run_peak.max(peak);
            if merged.dither_eligible(combined_peak) {
                self.run_grid = Some(merged);
                self.run_peak = combined_peak;
            } else {
                // Do not discard an already qualifying run just because a
                // subsequent block changes quantizer or contains other noise.
                self.end_run();
                if !local.dither_eligible(peak) {
                    return;
                }
                self.run_grid = Some(local);
                self.run_peak = peak;
            }
            for &x in pcm {
                if x < 0.0 {
                    self.run_signs[0] += 1;
                }
                if x > 0.0 {
                    self.run_signs[1] += 1;
                }
            }
        }
        self.engine.push(pcm);
        self.run_frames += (pcm.len() / self.engine.channels) as u64;
    }
    // Both noise and dither consume the same detector decision. Indices
    // are interleaved sample offsets; each real frame is appended at most
    // once. A split always contains a nonempty rejected middle region.
    pub fn block(&mut self, pcm: &[f64], split: Option<(usize, usize)>) {
        if let Some((prefix, suffix)) = split {
            debug_assert!(prefix + suffix < pcm.len());
            self.append(&pcm[..prefix]);
            self.end_run();
            self.append(&pcm[pcm.len() - suffix..]);
        } else {
            self.append(pcm);
        }
    }
    pub fn end_run(&mut self) {
        if self.run_frames == 0 {
            return;
        }
        if self.run_frames >= self.min_frames
            && self
                .run_grid
                .as_ref()
                .is_none_or(|g| g.dither_eligible(self.run_peak))
        {
            self.engine.flush();
            self.pooled.add(&self.engine.spectrum);
            for (a, b) in self.signs.iter_mut().zip(self.run_signs) {
                *a += b;
            }
        }
        self.engine.reset();
        self.run_frames = 0;
        self.run_grid = self.encoding.map(Grid::new);
        self.run_peak = 0.0;
        self.run_signs = [0; 2];
    }
    pub fn dither(&self) -> Option<crate::Dither> {
        (self.pooled.frames > 0)
            .then(|| classify_dither(&self.pooled, &self.signs))
            .flatten()
    }
}

#[cfg(test)]
mod numerical_kernel_tests {
    use super::*;

    #[test]
    fn power_kernel_preserves_endpoint_weights_and_accumulation_bits() {
        // Both endpoint weights, interior tails, and arbitrary alignment
        // (subslice offsets). Include subnormal powers and signed zero.
        let cases = [
            C::new(0.0, -0.0),
            C::new(0.125, -0.25),
            C::new(1e-155, -1e-160),
            C::new(-0.875, 0.5),
            C::new(1e-180, 1e-150),
            C::new(-3.0, 2.0),
        ];
        for n in [2, 3, 4, 5, 7, 8, 9, 17, 33, 4097, 65537] {
            for offset in [0, 1, 3] {
                let input: Vec<_> = (0..n + offset).map(|i| cases[i % cases.len()]).collect();
                for initial in [0.0, -0.0, 0.125] {
                    let mut actual = vec![initial; n + offset + 1];
                    let mut expected = actual.clone();
                    for _ in 0..13 {
                        accumulate_power(&input[offset..], &mut actual[offset..offset + n]);
                        for (i, x) in input[offset..].iter().enumerate() {
                            let factor = if i == 0 || i == n - 1 { 1.0 } else { 2.0 };
                            expected[offset + i] += factor * x.norm_sqr();
                        }
                    }
                    for (a, b) in actual.iter().zip(expected) {
                        assert_eq!(a.to_bits(), b.to_bits(), "n={n}, offset={offset}");
                    }
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "spectral_reference_tests.rs"]
mod spectral_reference_tests;
