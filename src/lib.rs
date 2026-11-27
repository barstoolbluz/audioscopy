//! Selective, streaming measurements of decoded, interleaved PCM.
//! No decoder, file I/O, UI, or application-specific types are required.
#![warn(missing_docs)]
mod noise_record;
#[cfg(feature = "precision")]
pub mod precision;
mod report;
mod spectrum;
mod stream;
mod temporal;

pub use noise_record::{NoiseRecord, NoiseSnapshot, QuietSpectrumRecord};
pub use report::{Album, AlbumLoudnessMethod, Track};
use std::fmt;
pub use stream::{AlbumAnalyzer, Analyzer};
/// Decoded loudness channel roles from the shared native loudness core.
pub use true_peak::loudness::ChannelRole;

/// One selectable measurement from inventory revision 2.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Measurement {
    /// EBU Tech 3342 loudness range (LU).
    LoudnessRange,
    /// Maximum full three-second loudness (LUFS).
    ShortTermMaximum,
    /// Three-second loudness against time.
    LoudnessOverTime,
    /// Percentage of 50 ms blocks within 1 dB of the maximum.
    Limiting,
    /// Count of near-maximum flat local extrema.
    FlatToppedPeaks,
    /// Exact integer-lattice resolution.
    BitsInUse,
    /// Approximate noise-derived resolution.
    EffectiveBits,
    /// Candidate source sample rate inferred from a steep cutoff.
    LikelySourceRate,
    /// Heuristic lossy spectral signature.
    LossyCodingPattern,
    /// Estimated upper extent of spectral content (kHz).
    Bandwidth,
    /// Ratio of power above 20 kHz to total power (dB).
    Above20Khz,
    /// Long-term spectrum.
    AverageSpectrum,
    /// Pooled quiet-passage level (dBFS).
    NoiseFloor,
    /// Sample peak minus quiet-passage level (dB).
    DynamicRange,
    /// Quiet-passage power spectrum.
    NoiseSpectrum,
    /// Largest identifiable narrow noise tone.
    HumAndIdleTones,
    /// Quiet power in five standard bands.
    NoiseByBand,
    /// Prediction-error events per minute.
    ClicksPerMinute,
    /// Heuristic dither classification.
    Dither,
    /// Stereo Pearson correlation.
    Correlation,
    /// Left/right energy ratio (dB; positive means left louder).
    Balance,
    /// Side/mid energy ratio (dB).
    Width,
    /// Loss from averaging L/R to mono (dB, nonnegative).
    MonoCompatibility,
    /// Upper-band inter-channel lag (microseconds; positive means R delayed).
    InterChannelDelay,
    /// Bass mono cutoff (Hz) or full stereo bass.
    BassWidth,
}

impl Measurement {
    /// Every in-scope measurement, in inventory order.
    pub const ALL: [Self; 25] = [
        Self::LoudnessRange,
        Self::ShortTermMaximum,
        Self::LoudnessOverTime,
        Self::Limiting,
        Self::FlatToppedPeaks,
        Self::BitsInUse,
        Self::EffectiveBits,
        Self::LikelySourceRate,
        Self::LossyCodingPattern,
        Self::Bandwidth,
        Self::Above20Khz,
        Self::AverageSpectrum,
        Self::NoiseFloor,
        Self::DynamicRange,
        Self::NoiseSpectrum,
        Self::HumAndIdleTones,
        Self::NoiseByBand,
        Self::ClicksPerMinute,
        Self::Dither,
        Self::Correlation,
        Self::Balance,
        Self::Width,
        Self::MonoCompatibility,
        Self::InterChannelDelay,
        Self::BassWidth,
    ];
}

/// Compact selection. Empty selections perform only input validation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Measurements(u32);
impl Measurements {
    /// Select all measurements.
    pub const fn all() -> Self {
        Self((1 << 25) - 1)
    }
    /// Select the supplied measurements (duplicates are harmless).
    pub fn new(items: &[Measurement]) -> Self {
        Self(items.iter().fold(0, |a, m| a | (1 << (*m as u8))))
    }
    /// Whether a particular measurement was requested.
    pub const fn contains(self, m: Measurement) -> bool {
        self.0 & (1 << (m as u8)) != 0
    }
    /// Stable bit mask in inventory order, for caller-owned persistence.
    pub const fn bits(self) -> u32 {
        self.0
    }
    /// Validate a persisted bit mask; unknown bits are rejected.
    pub const fn from_bits(bits: u32) -> Option<Self> {
        if bits & !Self::all().0 == 0 {
            Some(Self(bits))
        } else {
            None
        }
    }
    pub(crate) fn any(self, items: &[Measurement]) -> bool {
        items.iter().any(|m| self.contains(*m))
    }
}

/// Original PCM sample representation. The caller supplies normalized f64
/// values; integer codes must be exactly representable after multiplication
/// by 2^(bits-1). Floating PCM can exceed nominal full scale.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Encoding {
    /// Signed integer PCM (1 through 32 bits, including unsigned PCM decoded
    /// to its equivalent centered signed code).
    Integer(u8),
    /// Floating PCM; tested for an exact normalized integer lattice.
    Float,
}

/// Input clock, channel count/order, and original sample representation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Format {
    /// Sample rate in Hz, 8,000 through 768,000.
    pub sample_rate: u32,
    /// Interleaved channel count, 1 through 64.
    pub channels: usize,
    /// Original PCM representation.
    pub encoding: Encoding,
    /// Optional explicit loudness roles. Required for >2 channels when a
    /// loudness measurement is selected. Stereo metrics require two channels.
    pub loudness_roles: Option<Vec<ChannelRole>>,
}

/// Quiet-passage detection settings, fixed during measurement.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Settings {
    /// Maximum per-channel RMS in a 50 ms block for quiet eligibility, dBFS.
    /// Clear transitions are refined using real below-threshold edge frames.
    pub quiet_threshold_dbfs: f64,
    /// Minimum contiguous quiet duration, seconds (0.05 through 60).
    pub shortest_gap_seconds: f64,
}
impl Default for Settings {
    fn default() -> Self {
        Self {
            quiet_threshold_dbfs: -60.0,
            shortest_gap_seconds: 1.0,
        }
    }
}

/// Frequency weighting of stored noise power (not a quasi-peak detector).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Weighting {
    /// No weighting.
    #[default]
    Unweighted,
    /// IEC A-frequency weighting.
    A,
    /// ITU-R BS.468 frequency weighting, RMS rather than quasi-peak detection.
    ItuR468,
}

/// Noise integration band, clipped to the available Nyquist frequency.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Band {
    /// 20 Hz through 20 kHz.
    #[default]
    Audible,
    /// 20 Hz through Nyquist.
    ToNyquist,
    /// DC through Nyquist.
    Full,
}

/// Read-time noise controls; changing either requires no PCM.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NoiseView {
    /// Frequency weighting.
    pub weighting: Weighting,
    /// Integration band.
    pub band: Band,
}

/// A measurement outcome. An unavailable value is never represented by a
/// numeric sentinel. Negative infinity in a *value* is reserved for an exact
/// zero power/ratio (and positive infinity for a nonzero numerator over zero).
#[derive(Clone, Debug, PartialEq)]
pub enum Outcome<T> {
    /// Computed value.
    Value(T),
    /// Insufficient evidence, with a human-readable one-line reason.
    CantTell(String),
    /// Measurement does not apply to this input.
    NotApplicable,
    /// Caller did not select this measurement.
    NotRequested,
}

/// Exact lattice information; no inference about the original mastering depth.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BitDepth {
    /// Exact integer PCM grid result.
    Integer {
        /// Resolution in use (see README for lattice/headroom convention).
        used: u8,
        /// Declared depth of integer PCM, or None for floating PCM.
        declared: Option<u8>,
        /// Full-scale-normalized depth from the zero-low-bit check alone.
        low_bit_depth: u8,
        /// Bits needed to index the observed affine lattice span, or None
        /// for a constant signal. This is not a source-depth estimate.
        observed_lattice_bits: Option<u8>,
        /// Exact spacing in units of 2^-(grid_scale_bits), if nonconstant.
        grid_step: u64,
        /// Power of two used to express the integer lattice.
        grid_scale_bits: u8,
    },
    /// Floating values do not all lie on a supported normalized integer grid.
    NonIntegerGrid,
    /// Album includes both integer-grid and non-integer-grid tracks.
    Mixed,
}

/// Dither classification, based on quiet low-level evidence, not proof of provenance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dither {
    /// Digital silence / no varying dither detected.
    None,
    /// Approximately flat noise consistent with dither.
    Flat,
    /// High-frequency-shaped noise consistent with dither.
    Shaped,
    /// No unique most-common album classification.
    Mixed,
}

/// Bass stereo extent.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BassWidth {
    /// Side energy is present throughout the occupied bass bands.
    Full,
    /// Side energy is below the mono threshold below this frequency (Hz).
    MonoBelow(f64),
}

/// Numeric graph point. Time curves use seconds/LUFS; spectrum curves use
/// Hz/dBFS per Hz (one-sided power density).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    /// Independent coordinate.
    pub x: f64,
    /// Dependent coordinate.
    pub y: f64,
}

/// Detected narrow-band tone.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tone {
    /// Frequency in Hz.
    pub frequency_hz: f64,
    /// Integrated line power level in dBFS.
    pub level_dbfs: f64,
}

/// Typed value for a selected measurement. Scalar units are documented on
/// [`Measurement`]. Approximate values have their own variant.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    /// Scalar value in the measurement's units.
    Scalar(f64),
    /// Exact event count.
    Count(u64),
    /// Exact resolution result.
    Bits(BitDepth),
    /// Approximate effective bits.
    ApproximateBits(f64),
    /// Source rate in Hz, or None for a mixed album.
    SourceRate(Option<u32>),
    /// Number of tracks with a lossy pattern and total tracks considered.
    Pattern {
        /// Tracks with a detected pattern.
        found: usize,
        /// Tracks considered.
        total: usize,
    },
    /// A numeric graph.
    Curve(Vec<Point>),
    /// Ordered per-track graphs for an album.
    Curves(Vec<Vec<Point>>),
    /// A narrow tone, or no distinguishable narrow line.
    Tone(Option<Tone>),
    /// Five noise levels, in inventory order; bands outside the view are n/a.
    Bands([Outcome<f64>; 5]),
    /// Dither estimate.
    Dither(Dither),
    /// Bass extent.
    BassWidth(BassWidth),
}

/// An invalid format, input, or upstream numerical/resource error.
#[derive(Debug)]
pub enum Error {
    /// Invalid configuration.
    Configuration(&'static str),
    /// Push did not contain complete interleaved frames.
    IncompleteFrame,
    /// Invalid sample (nonfinite, beyond the supported numerical range, or not
    /// a valid normalized code for the declared integer representation).
    InvalidSample(usize),
    /// No decoded frames, or no album tracks.
    Empty,
    /// Incompatible album measurement selections/settings or invalid lifecycle.
    Album(&'static str),
    /// Input exceeds the supported frame or retained-observation limit.
    ResourceLimit,
    /// Upstream loudness meter error.
    Loudness(true_peak::loudness::LoudnessError),
    /// A terminal DSP failure already occurred.
    Failed,
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Configuration(s) | Self::Album(s) => f.write_str(s),
            Self::IncompleteFrame => f.write_str("incomplete interleaved PCM frame"),
            Self::InvalidSample(i) => write!(f, "invalid PCM sample at push index {i}"),
            Self::Empty => f.write_str("empty input"),
            Self::ResourceLimit => f.write_str("analysis resource limit exceeded"),
            Self::Loudness(e) => write!(f, "loudness: {e}"),
            Self::Failed => f.write_str("analyzer is terminal after a DSP error"),
        }
    }
}
impl std::error::Error for Error {}
impl From<true_peak::loudness::LoudnessError> for Error {
    fn from(e: true_peak::loudness::LoudnessError) -> Self {
        Self::Loudness(e)
    }
}

pub(crate) fn db(power: f64) -> f64 {
    10.0 * power.log10()
}
pub(crate) fn ratio(a: f64, b: f64) -> f64 {
    if a == 0.0 && b == 0.0 {
        0.0
    } else {
        db(a) - db(b)
    }
}
pub(crate) fn scalar(x: f64) -> Outcome<Value> {
    Outcome::Value(Value::Scalar(x))
}
pub(crate) fn unavailable(reason: impl Into<String>) -> Outcome<Value> {
    Outcome::CantTell(reason.into())
}
