use crate::*;
use crate::spectrum::Spectrum;
use crate::report::{noise_measure, mode};

const NOISE: [Measurement; 8] = [Measurement::NoiseFloor, Measurement::DynamicRange,
    Measurement::NoiseSpectrum, Measurement::HumAndIdleTones, Measurement::NoiseByBand,
    Measurement::EffectiveBits, Measurement::Dither, Measurement::ClicksPerMinute];

/// Transfer record for pooled quiet power. A caller can store these fields
/// using its own cache or serialization format. Version 2 uses energy-complementary one-sided
/// linear power sums, normalized to mean-square PCM when divided by frames.
#[derive(Clone, Debug, PartialEq)]
pub struct QuietSpectrumRecord {
    /// FFT size (the version-2 power-of-two geometry for the sample rate).
    pub fft_size: usize,
    /// Qualifying quiet frames contributing to the spectrum.
    pub frames: u64,
    /// DC through Nyquist inclusive, in increasing frequency order. These
    /// are sums of mean-square power times real frame count, not dB values.
    pub bin_power_sums: Vec<f64>,
}

/// Versioned, application-independent sufficient statistics for re-reading
/// all selected noise measurements from a file. These fields are intended
/// for persistence by the caller; no on-disk format or I/O is imposed.
#[derive(Clone, Debug, PartialEq)]
pub struct NoiseRecord {
    /// Record schema version (currently 2).
    pub version: u32,
    /// Input sample rate in Hz.
    pub sample_rate: u32,
    /// Total real track frames.
    pub frames: u64,
    /// Selected noise measurements, in inventory bit order.
    pub selection_mask: u32,
    /// Settings used at capture time.
    pub settings: Settings,
    /// Linear absolute sample peak; present when DynamicRange was selected.
    pub sample_peak: Option<f64>,
    /// Quiet spectrum, or None when no eligible quiet passage was found
    /// (also None when no quiet-spectrum-based measurement was selected).
    pub quiet: Option<QuietSpectrumRecord>,
    /// Number of detected clicks, if requested.
    pub clicks: Option<u64>,
    /// Dither estimate, or None when the evidence was insufficient or not requested.
    pub dither: Option<Dither>,
}

impl Track {
    /// Export only the selected noise statistics. An unselected measurement
    /// cannot be reconstructed from this record by changing the selection.
    pub fn noise_record(&self) -> NoiseRecord {
        let selection_mask = Measurements::new(&NOISE).bits() & self.selection.bits();
        NoiseRecord { version: 2, sample_rate: self.format.sample_rate, frames: self.frames,
            selection_mask, settings: self.settings,
            sample_peak: self.selection.contains(Measurement::DynamicRange).then_some(self.peak),
            quiet: self.quiet.as_ref().map(|s| QuietSpectrumRecord { fft_size: s.n, frames: s.frames, bin_power_sums: s.sums.clone() }),
            clicks: self.clicks, dither: self.dither }
    }
}

/// Validated cached noise evidence. Re-reading uses exactly the same
/// arithmetic as the original Track or Album, with no decoding or FFT.
#[derive(Clone, Debug)]
pub struct NoiseSnapshot {
    rate: u32,
    frames: u64,
    selection: Measurements,
    settings: Settings,
    peak: Option<f64>,
    quiet: Option<Spectrum>,
    clicks: Option<u64>,
    dither: Option<Dither>,
}
impl NoiseSnapshot {
    /// Validate and consume a caller-restored record. Reject unknown versions,
    /// invalid geometry, invalid numeric values and inconsistent evidence.
    pub fn from_record(record: NoiseRecord) -> Result<Self, Error> {
        let fail = || Error::Configuration("invalid persisted noise record");
        let selection = Measurements::from_bits(record.selection_mask).ok_or_else(fail)?;
        if record.version != 2 || record.selection_mask & !Measurements::new(&NOISE).bits() != 0
            || !(8_000..=768_000).contains(&record.sample_rate) || record.frames == 0 || record.frames > (1_u64<<48)
            || !(-200.0..=0.0).contains(&record.settings.quiet_threshold_dbfs) || !(0.05..=60.0).contains(&record.settings.shortest_gap_seconds)
            || selection.contains(Measurement::DynamicRange) != record.sample_peak.is_some()
            || record.sample_peak.is_some_and(|p| !(0.0..=16.0).contains(&p))
            || selection.contains(Measurement::ClicksPerMinute) != record.clicks.is_some()
            || record.clicks.is_some_and(|c| c > record.frames)
            || (record.dither.is_some() && !selection.contains(Measurement::Dither)) || record.dither == Some(Dither::Mixed) {
            return Err(fail());
        }
        let quiet = if let Some(q) = record.quiet {
            let n = ((record.sample_rate as usize).div_ceil(6)).next_power_of_two();
            let min_frames = (record.settings.shortest_gap_seconds*f64::from(record.sample_rate)).ceil() as u64;
            if !selection.any(&NOISE[..6]) || q.fft_size != n || q.bin_power_sums.len() != n/2+1
                || q.frames < min_frames || q.frames > record.frames
                || q.bin_power_sums.iter().any(|p| !p.is_finite() || *p < 0.0) {
                return Err(fail());
            }
            let power = q.bin_power_sums.iter().sum::<f64>() / q.frames as f64;
            let bound = 10.0_f64.powf(record.settings.quiet_threshold_dbfs/10.0);
            if !power.is_finite() || power > bound * (1.0+1e-10)
                || record.sample_peak.is_some_and(|p| power > p*p*(1.0+1e-10)) {
                return Err(fail());
            }
            Some(Spectrum { rate: record.sample_rate, n, sums: q.bin_power_sums, frames: q.frames })
        } else { None };
        Ok(Self { rate: record.sample_rate, frames: record.frames, selection, settings: record.settings,
            peak: record.sample_peak, quiet, clicks: record.clicks, dither: record.dither })
    }
    /// Re-read the original file's selected noise measurement.
    pub fn measure(&self, m: Measurement, view: NoiseView) -> Outcome<Value> {
        Self::reduce(&[self], m, view)
    }
    /// Read a noise measurement from an ordered collection of restored
    /// records, using the same album pooling rules. Different sample rates
    /// are supported. The selected noise rows and settings must match.
    pub fn measure_album(tracks: &[Self], m: Measurement, view: NoiseView) -> Result<Outcome<Value>, Error> {
        let first = tracks.first().ok_or(Error::Empty)?;
        if tracks.iter().any(|t| t.selection != first.selection || t.settings != first.settings) {
            return Err(Error::Album("cached noise records require identical selections and settings"));
        }
        Ok(Self::reduce(&tracks.iter().collect::<Vec<_>>(),m,view))
    }
    fn reduce(tracks: &[&Self], m: Measurement, view: NoiseView) -> Outcome<Value> {
        if !tracks[0].selection.contains(m) { return Outcome::NotRequested; }
        match m {
            Measurement::Dither => {
                let d = mode(tracks.iter().filter_map(|t| t.dither).map(|d| match d { Dither::None=>0, Dither::Flat=>1, Dither::Shaped=>2, Dither::Mixed=>3 }));
                match d { Some(Some(0))=>Outcome::Value(Value::Dither(Dither::None)), Some(Some(1))=>Outcome::Value(Value::Dither(Dither::Flat)),
                    Some(Some(2))=>Outcome::Value(Value::Dither(Dither::Shaped)), Some(_)=>Outcome::Value(Value::Dither(Dither::Mixed)),
                    None=>unavailable("no sufficiently quiet passage with identifiable dither evidence") }
            }
            Measurement::ClicksPerMinute => {
                let clicks: f64 = tracks.iter().filter_map(|t| t.clicks).map(|n| n as f64).sum();
                let seconds: f64 = tracks.iter().map(|t| t.frames as f64/f64::from(t.rate)).sum();
                scalar(60.0*clicks/seconds)
            }
            _ => {
                let spectra: Vec<_> = tracks.iter().filter_map(|t| t.quiet.as_ref()).collect();
                let peak = tracks.iter().filter_map(|t| t.peak).fold(0.0,f64::max);
                noise_measure(&spectra,peak,tracks[0].settings.shortest_gap_seconds,m,view)
            }
        }
    }
}
