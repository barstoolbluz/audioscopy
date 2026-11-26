use crate::{BitDepth, Encoding, Dither};
use std::collections::BTreeMap;

fn gcd(mut a: u64, mut b: u64) -> u64 { while b != 0 { let r = a % b; a = b; b = r; } a }
fn ceil_log2(n: u64) -> u8 { if n <= 1 { 0 } else { (64 - (n-1).leading_zeros()) as u8 } }

// Integer PCM is exact. Float PCM is tested on a 2^-52 normalized grid,
// without rounding to that grid or silently snapping nearly equal values.
#[derive(Clone)]
pub(crate) struct Grid {
    encoding: Encoding,
    scale_bits: u8,
    scale: f64,
    first: Option<i64>,
    min: i64,
    max: i64,
    step: u64,
    used_or: u64,
    valid: bool,
    samples: u64,
}
impl Grid {
    pub fn new(encoding: Encoding) -> Self {
        let scale_bits = match encoding { Encoding::Integer(n) => n-1, Encoding::Float => 52 };
        Self { encoding, scale_bits, scale: 2.0_f64.powi(i32::from(scale_bits)), first: None,
            min: i64::MAX, max: i64::MIN, step: 0, used_or: 0, valid: true, samples: 0 }
    }
    pub fn push(&mut self, frame: &[f64]) {
        if !self.valid { return; }
        for &s in frame {
            if self.encoding == Encoding::Float && !(-1.0..1.0).contains(&s) { self.valid = false; return; }
            let v = s * self.scale;
            if v.fract() != 0.0 || v < i64::MIN as f64 || v >= -(i64::MIN as f64) { self.valid = false; return; }
            let c = v as i64;
            self.samples += 1;
            self.used_or |= c as u64;
            self.min = self.min.min(c); self.max = self.max.max(c);
            let first = *self.first.get_or_insert(c);
            if self.step != 1 { self.step = gcd(self.step, c.abs_diff(first)); }
        }
    }
    // Merge exact sufficient statistics without another traversal of PCM.
    pub fn merge(&mut self, other: &Self) {
        debug_assert_eq!(self.encoding, other.encoding);
        if !other.valid { self.valid=false; return; }
        if !self.valid { return; }
        if let Some(first) = other.first {
            if let Some(old) = self.first {
                self.step = gcd(gcd(self.step,other.step),old.abs_diff(first));
            } else { self.first=Some(first); self.step=other.step; }
            self.min=self.min.min(other.min); self.max=self.max.max(other.max);
            self.used_or |= other.used_or;
            self.samples += other.samples;
        }
    }
    // A nominal integer quantizer is admissible at its original spacing.
    // For a coarser quantizer, OR gives the largest *zero-aligned binary*
    // spacing, unlike an arbitrary affine GCD. Multiples of 100 have only
    // four-code binary alignment: a 100-code peak exceeds eight such LSBs.
    pub fn dither_eligible(&self, peak: f64) -> bool {
        if peak == 0.0 { return true; }
        if !self.valid { return false; }
        if matches!(self.encoding, Encoding::Integer(_)) && peak*self.scale <= 8.0 { return true; }
        self.step != 0 && peak*self.scale <= 8.0 * 2.0_f64.powi(self.used_or.trailing_zeros() as i32)
    }
    pub fn result(&self) -> BitDepth {
        if !self.valid { return BitDepth::NonIntegerGrid; }
        let declared = match self.encoding { Encoding::Integer(n) => Some(n), Encoding::Float => None };
        let nominal = self.scale_bits + 1;
        let low_bit_depth = nominal.saturating_sub(self.used_or.trailing_zeros().min(64) as u8);
        let slots = (self.step != 0).then(|| self.max.abs_diff(self.min)/self.step+1);
        let observed_lattice_bits = slots.map(ceil_log2);
        // An affine lattice (non-binary spacing or nonzero residue) is
        // evidence beyond amplitude/headroom alone. Two points alone do
        // not establish an evenly spaced subset. Preserve the observed
        // width even when the span is not a complete power-of-two codebook.
        // For an aligned binary lattice, retain the normalized convention
        // unless a complete-sized, nontrivial observed span supports the
        // smaller width. Neither check establishes the mastering quantizer.
        let affine = self.step > 1 && (!self.step.is_power_of_two()
            || self.first.unwrap_or(0).unsigned_abs() % self.step != 0);
        let used = match slots {
            Some(n) if self.step > 1
                && ((affine && n >= 3) || (n >= 4 && n.is_power_of_two() && self.samples >= n)) =>
                low_bit_depth.min(ceil_log2(n)),
            _ => low_bit_depth,
        };
        BitDepth::Integer { used, declared, low_bit_depth, observed_lattice_bits, grid_step: self.step, grid_scale_bits: self.scale_bits }
    }
}

#[derive(Default)]
pub(crate) struct Stereo {
    count: u64,
    mean_l: f64,
    mean_r: f64,
    var_l: f64,
    var_r: f64,
    cov: f64,
    pub l: f64,
    pub r: f64,
    pub mid: f64,
    pub side: f64,
    correlation: bool,
    lr: bool,
    mid_needed: bool,
    side_needed: bool,
}
impl Stereo {
    pub fn new(correlation: bool, balance: bool, width: bool, mono: bool) -> Self {
        Self { correlation, lr: balance || mono, mid_needed: width || mono, side_needed: width, ..Self::default() }
    }
    pub fn push(&mut self, l: f64, r: f64) {
        if self.correlation {
            self.count += 1;
            let n = self.count as f64;
            let dl = l - self.mean_l;
            let dr = r - self.mean_r;
            self.mean_l += dl / n;
            self.mean_r += dr / n;
            self.var_l += dl * (l - self.mean_l);
            self.var_r += dr * (r - self.mean_r);
            self.cov += dl * (r - self.mean_r);
        }
        if self.lr { self.l += l*l; self.r += r*r; }
        if self.mid_needed { self.mid += (l+r).powi(2) * 0.25; }
        if self.side_needed { self.side += (l-r).powi(2) * 0.25; }
    }
    pub fn correlation(&self) -> f64 {
        let norm = self.var_l.max(0.0).sqrt() * self.var_r.max(0.0).sqrt();
        if norm > 0.0 { (self.cov / norm).clamp(-1.0, 1.0) } else { 0.0 }
    }
}

// At most five frames of context; prediction uses samples on both sides.
// Events are counted once per frame, even if several channels trigger.
pub(crate) struct Clicks {
    history: Vec<f64>,
    channels: usize,
    frames: u64,
    pub count: u64,
    cooldown: usize,
    refractory: usize,
}
impl Clicks {
    pub fn new(channels: usize, rate: u32) -> Self { Self { history: vec![0.0; 5*channels], channels, frames: 0, count: 0, cooldown: 0, refractory: (rate as usize / 1000).max(1) } }
    pub fn push(&mut self, frame: &[f64]) {
        let c = self.channels;
        let idx = (self.frames % 5) as usize;
        self.history[idx*c..(idx+1)*c].copy_from_slice(frame);
        self.frames += 1;
        if self.frames < 5 { return; }
        if self.cooldown > 0 { self.cooldown -= 1; return; }
        let base = (idx+1)%5;
        for ch in 0..c {
            let at = |j: usize| self.history[((base+j)%5)*c+ch];
            let (a,b,x,d,e) = (at(0),at(1),at(2),at(3),at(4));
            let prediction = (b+d)*0.5;
            let background = (b-a).abs().max((e-d).abs()).max((d-b).abs()*0.5);
            let error = (x-prediction).abs();
            if error > 0.001 && error > 8.0*background {
                self.count += 1; self.cooldown = self.refractory - 1; break;
            }
        }
    }
}

const NEAR_PEAK: f64 = 0.8912509381337456; // -1 dB amplitude
const FLAT_REL: f64 = 1e-5;

#[derive(Clone, Copy, Default)]
struct Run {
    first: f64,
    last: f64,
    min: f64,
    max: f64,
    before: Option<f64>,
    count: u64,
}

// Runs need both neighboring samples to demonstrate a local extreme. Keep
// only eligible candidate amplitudes as the observed track peak increases.
// Keys are ordered positive f64 bit representations, counts are exact.
pub(crate) struct Flats {
    runs: Vec<Run>,
    candidates: BTreeMap<u64, u64>,
    peak: f64,
    absolute_tolerance: f64,
}
impl Flats {
    pub fn new(channels: usize, encoding: Encoding) -> Self {
        let absolute_tolerance = match encoding {
            Encoding::Integer(n) => 2.0_f64.powi(1-i32::from(n)),
            Encoding::Float => 0.0,
        };
        Self { runs: vec![Run::default(); channels], candidates: BTreeMap::new(), peak: 0.0, absolute_tolerance }
    }
    pub fn push(&mut self, frame: &[f64]) {
        let new_peak = frame.iter().map(|x| x.abs()).fold(self.peak, f64::max);
        if new_peak > self.peak {
            self.peak = new_peak;
            let threshold = (new_peak*NEAR_PEAK).to_bits();
            // Avoid repeated allocation when no old candidate falls below it.
            if self.candidates.first_key_value().is_some_and(|(k,_)| *k < threshold) {
                self.candidates = self.candidates.split_off(&threshold);
            }
        }
        for (run,&x) in self.runs.iter_mut().zip(frame) {
            if run.count == 0 { *run = Run { first: x, last: x, min:x, max:x, before: None, count: 1 }; continue; }
            let eps = (FLAT_REL * run.first.abs().max(x.abs()).max(1e-12)).max(self.absolute_tolerance);
            if (x-run.first).abs() <= eps { run.last = x; run.min=run.min.min(x); run.max=run.max.max(x); run.count += 1; continue; }
            if run.count >= 3 {
                // Plateau membership tolerance is not the extremum test.
                // A one-code shoulder is real evidence of a local extremum.
                // Compare with the actual extremal code, not the first code.
                let extreme = if run.first >= 0.0 { run.max } else { run.min };
                let sign = extreme.signum();
                if extreme != 0.0 && extreme.abs() >= self.peak*NEAR_PEAK
                    && run.before.is_some_and(|b| sign*(extreme-b) > 0.0)
                    && sign*(extreme-x) > 0.0 {
                    *self.candidates.entry(extreme.abs().to_bits()).or_default() += 1;
                }
            }
            *run = Run { first: x, last: x, min:x, max:x, before: Some(run.last), count: 1 };
        }
    }
    pub fn count(&self) -> u64 { self.candidates.values().sum() }
    pub fn distinct_candidates(&self) -> usize { self.candidates.len() }
}

pub(crate) fn limiting(peaks: &[f64]) -> f64 {
    let max = peaks.iter().copied().fold(0.0, f64::max);
    if max == 0.0 { return 0.0; }
    100.0 * peaks.iter().filter(|&&p| p >= max*NEAR_PEAK).count() as f64 / peaks.len() as f64
}

// Digital silence is definite. Dither-like noise is only classified when
// it has the magnitude expected at the admitted binary quantizer and a sufficiently
// broadband, approximately flat or high-frequency-shaped spectrum.
pub(crate) fn classify_dither(s: &crate::spectrum::Spectrum, signs: &[u64; 2]) -> Option<Dither> {
    if s.total() == 0.0 { return Some(Dither::None); }
    let [negative, positive] = *signs;
    let nonzero = negative + positive;
    // Quantized dither should fluctuate on both sides of zero; DC and
    // one-sided low-amplitude fades are not sufficient evidence.
    if nonzero == 0 || negative.min(positive) < nonzero/10 { return None; }
    let ny = f64::from(s.rate)/2.0;
    let w = crate::Weighting::Unweighted;
    let powers: Vec<f64> = (0..8).map(|i| s.power(ny*i as f64/8.0, ny*(i+1) as f64/8.0,w)).collect();
    let total: f64 = powers.iter().sum();
    let lo = powers[..4].iter().sum::<f64>();
    let hi = powers[4..].iter().sum::<f64>();
    let min = powers.iter().copied().fold(f64::INFINITY, f64::min);
    let max = powers.iter().copied().fold(0.0, f64::max);
    // Reject narrow tones and very sparse spectra rather than labeling them
    // dither. This cannot prove a particular recording was dithered.
    if total == 0.0 || s.tone(crate::NoiseView { weighting: w, band: crate::Band::Full }).is_some() { return None; }
    if min > max * 0.25 && hi < 2.0*lo && lo < 2.0*hi { Some(Dither::Flat) }
    else if hi > 2.0*lo && powers[5..].iter().filter(|&&x| x > total*0.04).count() >= 2 { Some(Dither::Shaped) }
    else { None }
}
