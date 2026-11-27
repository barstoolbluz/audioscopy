// SPDX-License-Identifier: GPL-2.0-or-later
//! Optional end-to-end complex cross-spectrum, not a delay estimator.
use super::{PowerBin, SpectrumAnalyzer, SpectrumBackend};
use crate::precision::{dd, Precision};
use crate::{Error, Format};

/// Signed, exponent-scaled spectral component: `(hi + lo) * 2^exponent`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SignedPowerBin {
    /// Rounded signed high component.
    pub hi: f64,
    /// Signed residual component.
    pub lo: f64,
    /// Independent binary exponent.
    pub exponent: i32,
}

impl SignedPowerBin {
    /// Binary64 compatibility view. Very small components may underflow.
    pub fn as_f64(self) -> f64 {
        let mantissa = self.hi + self.lo;
        if mantissa == 0.0 || self.exponent < -1075 {
            0.0_f64.copysign(mantissa)
        } else if self.exponent > 1024 {
            f64::INFINITY.copysign(mantissa)
        } else {
            dd::scale(mantissa, self.exponent)
        }
    }
}

/// Left/right auto powers and the complex product conj(L) * R.
/// Each component has its own exponent to preserve very small signals.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ComplexPowerBin {
    /// Left-channel mean-square power.
    pub left: PowerBin,
    /// Right-channel mean-square power.
    pub right: PowerBin,
    /// Real component of conj(L) * R.
    pub real: SignedPowerBin,
    /// Imaginary component of conj(L) * R.
    pub imag: SignedPowerBin,
}

/// Stereo one-sided cross-spectrum with identical window and hop to
/// [`super::PowerSpectrum`]. Auto powers are per channel, not channel mean.
/// DC and Nyquist have unit weight, other bins have weight two.
#[derive(Clone, Debug, PartialEq)]
pub struct CrossSpectrum {
    /// Actual computational backend.
    pub backend: SpectrumBackend,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Real FFT length.
    pub fft_len: usize,
    /// Real frames, excluding padding.
    pub frames: u64,
    /// DC through Nyquist inclusive.
    pub bins: Vec<ComplexPowerBin>,
}

impl CrossSpectrum {
    /// Arithmetic precision used to compute the spectrum.
    pub fn precision(&self) -> Precision {
        if self.backend == SpectrumBackend::X87 {
            Precision::Extended80
        } else {
            Precision::DoubleDouble
        }
    }

    /// Bin spacing in Hz.
    pub fn bin_width_hz(&self) -> f64 {
        f64::from(self.sample_rate) / self.fft_len as f64
    }
}

/// Streaming stereo cross-spectrum with independently normalized channels.
/// Does not change the existing delay or bass-width measurement definitions.
/// Same dynamic-range and target restrictions as SpectrumAnalyzer.
pub struct CrossSpectrumAnalyzer(SpectrumAnalyzer);

impl CrossSpectrumAnalyzer {
    /// Create an explicitly selected wide stereo analyzer.
    pub fn new(
        format: Format,
        precision: Precision,
        backend: SpectrumBackend,
    ) -> Result<Self, Error> {
        SpectrumAnalyzer::new_kind(format, precision, backend, true).map(Self)
    }

    /// Actual selected backend.
    pub const fn backend(&self) -> SpectrumBackend {
        self.0.backend()
    }

    /// Number of accepted input frames.
    pub const fn frames(&self) -> u64 {
        self.0.frames()
    }

    /// Ingest complete interleaved stereo frames.
    pub fn push_interleaved(&mut self, pcm: &[f64]) -> Result<(), Error> {
        self.0.push_interleaved(pcm)
    }

    /// Finish edge windows and return the complex cross power.
    pub fn finish(mut self) -> Result<CrossSpectrum, Error> {
        let inner = &mut self.0;
        if inner.failed {
            return Err(Error::Failed);
        }
        if inner.frames == 0 {
            return Err(Error::Empty);
        }
        let count = inner.fft_len / 2 + 1;
        let mut bins = Vec::new();
        bins.try_reserve_exact(count)
            .map_err(|_| Error::ResourceLimit)?;
        bins.resize(count, ComplexPowerBin::default());
        inner.plan.finish_cross(&mut bins)?;
        Ok(CrossSpectrum {
            backend: inner.backend,
            sample_rate: inner.format.sample_rate,
            fft_len: inner.fft_len,
            frames: inner.frames,
            bins,
        })
    }
}
