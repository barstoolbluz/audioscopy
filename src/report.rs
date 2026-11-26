use crate::*;
use crate::spectrum::{self, Spectrum};
use crate::temporal::Stereo;
use std::collections::BTreeMap;

/// Owned per-file report. No PCM is retained; noise power and curve data can
/// be read repeatedly. Clone only the individual values required by a caller.
pub struct Track {
    pub(crate) format: Format,
    pub(crate) selection: Measurements,
    pub(crate) settings: Settings,
    pub(crate) frames: u64,
    pub(crate) peak: f64,
    pub(crate) bits: Option<BitDepth>,
    pub(crate) range: Option<f64>,
    pub(crate) source_rate: Option<u32>,
    pub(crate) bandwidth: Option<f64>,
    pub(crate) short_max: Option<f64>,
    pub(crate) curve: Vec<Point>,
    pub(crate) limiting: Option<f64>,
    pub(crate) flats: Option<u64>,
    pub(crate) clicks: Option<u64>,
    pub(crate) stereo: Option<Stereo>,
    pub(crate) spectrum: Option<Spectrum>,
    pub(crate) quiet: Option<Spectrum>,
    pub(crate) pattern: bool,
    pub(crate) bass: Option<BassWidth>,
    pub(crate) delay: Option<f64>,
    pub(crate) dither: Option<Dither>,
}
impl Track {
    /// Original format and decoded channel order.
    pub fn format(&self) -> &Format { &self.format }
    /// Selection used to produce the report.
    pub fn selection(&self) -> Measurements { self.selection }
    /// Quiet detector settings used to produce the report.
    pub fn settings(&self) -> Settings { self.settings }
    /// Number of real decoded frames.
    pub fn frames(&self) -> u64 { self.frames }
    /// Duration in seconds.
    pub fn duration_seconds(&self) -> f64 { self.frames as f64 / f64::from(self.format.sample_rate) }
    /// Total qualifying quiet duration in seconds, or None if no noise-power
    /// measurement was requested. Zero means no qualifying quiet passage.
    pub fn quiet_seconds(&self) -> Option<f64> {
        self.selection.any(&[Measurement::EffectiveBits, Measurement::NoiseFloor, Measurement::DynamicRange,
            Measurement::NoiseSpectrum, Measurement::HumAndIdleTones, Measurement::NoiseByBand])
            .then(|| self.quiet.as_ref().map_or(0.0, Spectrum::duration))
    }
    /// Read a measurement. Weighting/band affect noise-derived values only;
    /// this method does not decode, FFT, or mutate the report.
    pub fn measure(&self, measurement: Measurement, view: NoiseView) -> Outcome<Value> {
        measure(&[self], self.selection, measurement, view, self.range, false)
    }
}

/// Method used for the album's LRA.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AlbumLoudnessMethod {
    /// Range was not requested.
    NotRequested,
    /// One continuous K-weighted stream, with windows across track boundaries.
    Continuous,
    /// Independent reports cannot reconstruct continuous boundary windows.
    /// Reanalyze the ordered PCM with the streaming album API.
    Unavailable,
}

/// Ordered album of completed track reports and album-wide loudness range.
/// Noise and spectrum readings pool linear power in real time, not in dB.
pub struct Album {
    tracks: Vec<Track>,
    selection: Measurements,
    range: Option<f64>,
    method: AlbumLoudnessMethod,
}
impl Album {
    /// Reduce independently analyzed tracks in supplied order. All tracks must
    /// have the same selection and quiet settings. For literal continuous
    /// LRA across boundaries, use [`crate::AlbumAnalyzer`] while decoding.
    pub fn from_tracks(tracks: Vec<Track>) -> Result<Self, Error> { Self::build(tracks, None) }
    pub(crate) fn build(tracks: Vec<Track>, continuous: Option<Option<f64>>) -> Result<Self, Error> {
        let first = tracks.first().ok_or(Error::Empty)?;
        let selection = first.selection;
        let settings = first.settings;
        if tracks.iter().any(|t| t.selection != selection || t.settings != settings) {
            return Err(Error::Album("album tracks require identical measurement selection and quiet settings"));
        }
        let (range, method) = if !selection.contains(Measurement::LoudnessRange) {
            (None, AlbumLoudnessMethod::NotRequested)
        } else if let Some(range) = continuous {
            (range, AlbumLoudnessMethod::Continuous)
        } else if tracks.len() == 1 {
            // One independently measured track already is the complete programme.
            (tracks[0].range, AlbumLoudnessMethod::Continuous)
        } else {
            (None, AlbumLoudnessMethod::Unavailable)
        };
        Ok(Self { tracks, selection, range, method })
    }
    /// Reports in the exact supplied track order.
    pub fn tracks(&self) -> &[Track] { &self.tracks }
    /// Total decoded programme duration in seconds.
    pub fn duration_seconds(&self) -> f64 { self.tracks.iter().map(Track::duration_seconds).sum() }
    /// Whether the LRA windows cross track boundaries.
    pub fn loudness_method(&self) -> AlbumLoudnessMethod { self.method }
    /// Read an album measurement according to its inventory reduction rule.
    /// Unavailable tracks contribute no evidence to evidence-dependent modes,
    /// medians and means; if none apply, the outcome explains why.
    pub fn measure(&self, measurement: Measurement, view: NoiseView) -> Outcome<Value> {
        if measurement == Measurement::LoudnessRange && self.method == AlbumLoudnessMethod::Unavailable {
            return unavailable("continuous album LRA requires ordered PCM with matching sample rates and channel roles");
        }
        let tracks: Vec<_> = self.tracks.iter().collect();
        measure(&tracks, self.selection, measurement, view, self.range, true)
    }
}

fn timed_mean(tracks: &[&Track], f: impl Fn(&Track) -> Option<f64>) -> Option<f64> {
    let mut sum = 0.0;
    let mut duration = 0.0;
    let mut pos_inf = false;
    let mut neg_inf = false;
    for t in tracks {
        if let Some(x) = f(t) {
            let d = t.duration_seconds();
            duration += d;
            if x == f64::INFINITY { pos_inf = true; }
            else if x == f64::NEG_INFINITY { neg_inf = true; }
            else { sum += d * x; }
        }
    }
    if duration == 0.0 || (pos_inf && neg_inf) { None }
    else if pos_inf { Some(f64::INFINITY) } else if neg_inf { Some(f64::NEG_INFINITY) }
    else { Some(sum/duration) }
}
fn median(mut values: Vec<f64>) -> Option<f64> {
    if values.is_empty() { return None; }
    values.sort_by(f64::total_cmp);
    let mid = values.len()/2;
    if values.len()%2 == 1 { Some(values[mid]) } else { Some((values[mid-1] + values[mid])*0.5) }
}
pub(crate) fn mode<T: Ord + Copy>(items: impl Iterator<Item=T>) -> Option<Option<T>> {
    let mut counts = BTreeMap::new();
    for x in items { *counts.entry(x).or_insert(0_usize) += 1; }
    let max = counts.values().copied().max()?;
    let mut winners = counts.into_iter().filter(|(_,c)| *c == max).map(|(x,_)| x);
    let first = winners.next()?;
    Some(if winners.next().is_some() { None } else { Some(first) })
}
fn bits(tracks: &[&Track]) -> Option<BitDepth> {
    let mut best: Option<BitDepth> = None;
    let mut non_integer = false;
    for t in tracks {
        match t.bits {
            Some(b @ BitDepth::Integer { used, .. }) => {
                if best.is_none_or(|x| match x { BitDepth::Integer { used: old, .. } => used > old, _ => true }) { best = Some(b); }
            }
            Some(BitDepth::NonIntegerGrid | BitDepth::Mixed) => non_integer = true,
            None => {}
        }
    }
    match (best,non_integer) { (Some(_),true) => Some(BitDepth::Mixed), (None,true) => Some(BitDepth::NonIntegerGrid), (b,false) => b }
}
fn stereo_ratio(t: &Track, width: bool) -> Option<f64> {
    let s = t.stereo.as_ref()?;
    if width { Some(ratio(s.side, s.mid)) }
    else { Some(ratio((s.l+s.r)*0.5, s.mid).max(0.0)) }
}
fn measure(tracks: &[&Track], selection: Measurements, m: Measurement, view: NoiseView, range: Option<f64>, album: bool) -> Outcome<Value> {
    if !selection.contains(m) { return Outcome::NotRequested; }
    let value = Outcome::Value;
    let no_stereo = || Outcome::NotApplicable;
    match m {
        Measurement::LoudnessRange => range.map_or_else(|| unavailable("no eligible short-term loudness windows"), scalar),
        Measurement::ShortTermMaximum => tracks.iter().filter_map(|t| t.short_max).max_by(f64::total_cmp)
            .map_or_else(|| unavailable("no complete three-second loudness window"), scalar),
        Measurement::LoudnessOverTime => if album { value(Value::Curves(tracks.iter().map(|t| t.curve.clone()).collect())) } else { value(Value::Curve(tracks[0].curve.clone())) },
        Measurement::Limiting => timed_mean(tracks, |t| t.limiting).map_or_else(|| unavailable("no peak blocks"), scalar),
        Measurement::FlatToppedPeaks => value(Value::Count(tracks.iter().filter_map(|t| t.flats).sum())),
        Measurement::BitsInUse => bits(tracks).map_or_else(|| unavailable("no lattice statistics"), |b| value(Value::Bits(b))),
        Measurement::EffectiveBits | Measurement::NoiseFloor | Measurement::DynamicRange | Measurement::NoiseSpectrum | Measurement::NoiseByBand | Measurement::HumAndIdleTones => {
            let spectra: Vec<_> = tracks.iter().filter_map(|t| t.quiet.as_ref()).collect();
            let peak = tracks.iter().map(|t| t.peak).fold(0.0,f64::max);
            noise_measure(&spectra, peak, tracks[0].settings.shortest_gap_seconds, m, view)
        }
        Measurement::LikelySourceRate => match mode(tracks.iter().filter_map(|t| t.source_rate)) {
            Some(rate) => value(Value::SourceRate(rate)), None => unavailable("no steep occupied cutoff below Nyquist")
        },
        Measurement::LossyCodingPattern => value(Value::Pattern { found: tracks.iter().filter(|t| t.pattern).count(), total: tracks.len() }),
        Measurement::Bandwidth => tracks.iter().filter_map(|t| t.bandwidth).min_by(f64::total_cmp)
            .map_or_else(|| unavailable("no spectrum"), scalar),
        Measurement::Above20Khz => {
            let mut above = 0.0;
            let mut total = 0.0;
            let mut applicable = false;
            for t in tracks {
                if t.format.sample_rate <= 44_100 { continue; }
                if let Some(s) = &t.spectrum {
                    applicable = true;
                    let d = s.duration();
                    above += d*s.power(20_000.0_f64.next_up(),f64::INFINITY,Weighting::Unweighted);
                    total += d*s.total();
                }
            }
            if !applicable { Outcome::NotApplicable } else if total == 0.0 { scalar(f64::NEG_INFINITY) } else { scalar(ratio(above,total)) }
        }
        Measurement::AverageSpectrum => {
            let spectra: Vec<_> = tracks.iter().filter_map(|t| t.spectrum.as_ref()).collect();
            value(Value::Curve(spectrum::curve(&spectra, NoiseView { weighting: Weighting::Unweighted, band: Band::Full })))
        }
        Measurement::ClicksPerMinute => {
            let clicks: u64 = tracks.iter().filter_map(|t| t.clicks).sum();
            let seconds: f64 = tracks.iter().map(|t| t.duration_seconds()).sum();
            scalar(60.0*clicks as f64/seconds)
        }
        Measurement::Dither => {
            let classification = mode(tracks.iter().filter_map(|t| t.dither).map(|d| match d { Dither::None=>0, Dither::Flat=>1, Dither::Shaped=>2, Dither::Mixed=>3 }));
            match classification {
                Some(Some(0)) => value(Value::Dither(Dither::None)),
                Some(Some(1)) => value(Value::Dither(Dither::Flat)),
                Some(Some(2)) => value(Value::Dither(Dither::Shaped)),
                Some(_) => value(Value::Dither(Dither::Mixed)),
                None => unavailable("no sufficiently quiet passage with identifiable dither evidence"),
            }
        }
        Measurement::Correlation => timed_mean(tracks, |t| t.stereo.as_ref().map(Stereo::correlation)).map_or_else(no_stereo,scalar),
        Measurement::Balance => {
            let mut l = 0.0;
            let mut r = 0.0;
            let mut eligible = false;
            for t in tracks {
                if let Some(s) = &t.stereo { eligible = true; l += s.l/f64::from(t.format.sample_rate); r += s.r/f64::from(t.format.sample_rate); }
            }
            if eligible { scalar(ratio(l,r)) } else { no_stereo() }
        }
        Measurement::Width => {
            if tracks.iter().all(|t| t.stereo.is_none()) { return no_stereo(); }
            timed_mean(tracks, |t| stereo_ratio(t,true)).map_or_else(|| unavailable("opposite infinite stereo-width endpoints cannot be averaged"),scalar)
        }
        Measurement::MonoCompatibility => tracks.iter().filter_map(|t| stereo_ratio(t,false)).max_by(f64::total_cmp).map_or_else(no_stereo,scalar),
        Measurement::InterChannelDelay => {
            if tracks.iter().all(|t| t.format.channels != 2) { return no_stereo(); }
            median(tracks.iter().filter_map(|t| t.delay).collect()).map_or_else(|| unavailable("upper-band channels do not have a unique correlated delay"),scalar)
        }
        Measurement::BassWidth => median(tracks.iter().filter_map(|t| t.bass).map(|b| match b { BassWidth::Full=>0.0, BassWidth::MonoBelow(f)=>f }).collect())
            .map_or_else(no_stereo,|x| value(Value::BassWidth(if x == 0.0 { BassWidth::Full } else { BassWidth::MonoBelow(x) }))),
    }
}

// The same arithmetic serves live reports and validated persisted records.
pub(crate) fn noise_measure(spectra: &[&Spectrum], peak: f64, gap: f64, m: Measurement, view: NoiseView) -> Outcome<Value> {
    if spectra.is_empty() { return unavailable(format!("no quiet passage of {gap} s or longer")); }
    let value = Outcome::Value;
            match m {
                Measurement::NoiseSpectrum => value(Value::Curve(spectrum::curve(spectra, view))),
                Measurement::HumAndIdleTones => value(Value::Tone(spectra.iter().filter_map(|s| s.tone(view)).max_by(|a,b| a.level_dbfs.total_cmp(&b.level_dbfs)))),
                Measurement::NoiseByBand => {
                    let bands = [(20.0,200.0),(200.0,2000.0),(2000.0,10_000.0),(10_000.0,20_000.0),(20_000.0,f64::INFINITY)];
                    value(Value::Bands(bands.map(|b| spectrum::pooled_power(spectra,view,b).map_or(Outcome::NotApplicable,|p| Outcome::Value(db(p))))))
                }
                _ => {
                    let Some(p) = spectrum::pooled_power(spectra,view,(0.0,f64::INFINITY)) else { return Outcome::NotApplicable; };
                    let floor = db(p);
                    match m {
                        Measurement::NoiseFloor => scalar(floor),
                        Measurement::EffectiveBits => {
                            // The inventory uses about 6 dB per bit, without a
                            // full-scale-sine or ideal-quantizer correction.
                            if p == 0.0 { unavailable("silent quiet passages do not bound effective resolution") }
                            else { value(Value::ApproximateBits((-floor / (20.0*2.0_f64.log10())).max(0.0))) }
                        }
                        Measurement::DynamicRange => {
                            
                            if peak == 0.0 { unavailable("no nonzero peak from which to measure dynamic range") }
                            else { scalar(20.0*peak.log10() - floor) }
                        }
                        _ => unreachable!(),
                    }
                }
            }
}
