// SPDX-License-Identifier: GPL-2.0-or-later
//! Optional end-to-end wider average-power spectrum.
//!
//! The FFT length, half-hop sine-of-Hann windows, edge padding, one-sided
//! power weights and real-frame/channel normalization match the mathematical
//! definition of the ordinary average spectrum. Coefficients, transforms and
//! power sums are computed at the selected wider precision. The result exposes
//! native linear-power bins rather than rounding them through a dB plot.
//! No wider quiet-passage detection, delay, bass width or album claim is made.
pub use super::PowerBin;
use super::Precision;
use crate::stream::{validate_format, validate_samples, MAX_FRAMES};
use crate::{Error, Format, Settings};

mod cross;
mod noise;
pub use cross::{ComplexPowerBin, CrossSpectrum, CrossSpectrumAnalyzer, SignedPowerBin};
pub use noise::{QuietNoiseAnalyzer, QuietNoisePool, QuietNoiseReport};

/// Requested or actual wide-spectrum computational backend.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SpectrumBackend {
    /// Choose a backend that satisfies the requested precision. For
    /// double-double this is the first vector backend that this build and
    /// processor support (AVX2, then NEON), else scalar double-double.
    #[default]
    Auto,
    /// Genuine extended80 on native GNU/Linux x86-64 with `precision-x87`.
    X87,
    /// Automatic double-double selection, the same as `Auto` with
    /// `Precision::DoubleDouble`. Never selects x87.
    DoubleDouble,
    /// Double-double with the platform's baseline ISA and C `fma`.
    DoubleDoublePortable,
    /// Double-double using the explicitly targeted x86 FMA implementation.
    DoubleDoubleFma,
    /// Experimental four-lane AVX2/FMA double-double FFT.
    DoubleDoubleAvx2,
    /// Experimental eight-lane AVX-512F/FMA double-double FFT.
    DoubleDoubleAvx512,
    /// Experimental two-lane AArch64 NEON/FMA double-double FFT.
    DoubleDoubleNeon,
}

impl SpectrumBackend {
    const fn request(self) -> i32 {
        match self {
            Self::Auto | Self::DoubleDouble => 1,
            Self::X87 => 2,
            Self::DoubleDoublePortable => 3,
            Self::DoubleDoubleFma => 4,
            Self::DoubleDoubleAvx2 => 6,
            Self::DoubleDoubleAvx512 => 7,
            Self::DoubleDoubleNeon => 8,
        }
    }

    #[cfg(audioscopy_wide_spectrum)]
    fn actual(value: i32) -> Result<Self, Error> {
        match value {
            2 => Ok(Self::X87),
            3 => Ok(Self::DoubleDoublePortable),
            4 => Ok(Self::DoubleDoubleFma),
            6 => Ok(Self::DoubleDoubleAvx2),
            7 => Ok(Self::DoubleDoubleAvx512),
            8 => Ok(Self::DoubleDoubleNeon),
            _ => Err(Error::Failed),
        }
    }

    /// Whether this build and current processor can execute this backend.
    /// For Auto, this means at least one high-precision mode is available.
    pub fn available(self) -> bool {
        Native::available(self.request())
    }
}

/// Native one-sided channel-mean average spectrum; DC and Nyquist are
/// included once, interior powers twice. Bins sum to mean-square PCM power.
#[derive(Clone, Debug, PartialEq)]
pub struct PowerSpectrum {
    /// Actual selected backend, never Auto or DoubleDouble (auto selection).
    pub backend: SpectrumBackend,
    /// Decoded sample rate in Hz.
    pub sample_rate: u32,
    /// Real FFT length, equal to the ordinary average-spectrum grid.
    pub fft_len: usize,
    /// Real frames contributing to the measurement, excluding padding.
    pub frames: u64,
    /// Linear bin powers, indexed from DC to Nyquist inclusive.
    pub bins: Vec<PowerBin>,
}

impl PowerSpectrum {
    /// Arithmetic precision actually used by the transform and accumulation.
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

#[cfg(audioscopy_wide_spectrum)]
mod native {
    use super::{ComplexPowerBin, Error, PowerBin, SpectrumBackend};
    use std::{ffi::c_void, ptr::NonNull};

    unsafe extern "C" {
        fn audioscopy_spectral_available(mode: i32) -> i32;
        fn audioscopy_spectral_new(
            rate: u32,
            channels: usize,
            mode: i32,
            out: *mut *mut c_void,
            fft_len: *mut usize,
            actual: *mut i32,
        ) -> i32;
        fn audioscopy_cross_new(
            rate: u32,
            mode: i32,
            out: *mut *mut c_void,
            fft_len: *mut usize,
            actual: *mut i32,
        ) -> i32;
        fn audioscopy_cross_finish(
            plan: *mut c_void,
            out: *mut ComplexPowerBin,
            bins: usize,
        ) -> i32;
        fn audioscopy_spectral_push(plan: *mut c_void, pcm: *const f64, frames: usize) -> i32;
        fn audioscopy_spectral_finish(plan: *mut c_void, out: *mut PowerBin, bins: usize) -> i32;
        fn audioscopy_spectral_free(plan: *mut c_void);
    }

    pub(super) struct Native(NonNull<c_void>);

    // SAFETY: the allocation is uniquely owned, has no borrowed pointers,
    // callbacks, mutable globals or thread affinity. Each call saves and
    // restores the calling thread's floating-point environment. Not Sync.
    unsafe impl Send for Native {}

    impl Native {
        pub(super) fn available(mode: i32) -> bool {
            // SAFETY: pure runtime feature availability query.
            unsafe { audioscopy_spectral_available(mode) == 1 }
        }

        pub(super) fn new(
            rate: u32,
            channels: usize,
            mode: i32,
            cross: bool,
        ) -> Result<(Self, usize, SpectrumBackend), Error> {
            let mut raw = std::ptr::null_mut();
            let mut fft_len = 0;
            let mut actual = 0;
            // SAFETY: output slots live for the call; C retains none of them.
            let status = unsafe {
                if cross {
                    audioscopy_cross_new(rate, mode, &mut raw, &mut fft_len, &mut actual)
                } else {
                    audioscopy_spectral_new(
                        rate,
                        channels,
                        mode,
                        &mut raw,
                        &mut fft_len,
                        &mut actual,
                    )
                }
            };
            if status != 1 {
                return Err(match status {
                    -1 => Error::ResourceLimit,
                    -2 => Error::Configuration("wide spectrum backend unavailable"),
                    _ => Error::Failed,
                });
            }
            let plan = Self(NonNull::new(raw).ok_or(Error::Failed)?);
            Ok((plan, fft_len, SpectrumBackend::actual(actual)?))
        }

        pub(super) fn push(&mut self, pcm: &[f64], frames: usize) -> Result<(), Error> {
            // SAFETY: uniquely owned plan; caller validated complete finite
            // frames and sample count. C copies input and retains no pointer.
            if unsafe { audioscopy_spectral_push(self.0.as_ptr(), pcm.as_ptr(), frames) } == 1 {
                Ok(())
            } else {
                Err(Error::Failed)
            }
        }

        pub(super) fn finish(&mut self, bins: &mut [PowerBin]) -> Result<(), Error> {
            // SAFETY: repr(C) output slots match the C struct; C verifies
            // count and writes only within this slice. No retained pointer.
            if unsafe { audioscopy_spectral_finish(self.0.as_ptr(), bins.as_mut_ptr(), bins.len()) }
                == 1
            {
                Ok(())
            } else {
                Err(Error::Failed)
            }
        }
        pub(super) fn finish_cross(&mut self, bins: &mut [ComplexPowerBin]) -> Result<(), Error> {
            // SAFETY: unique native cross plan and repr(C) output slice.
            if unsafe { audioscopy_cross_finish(self.0.as_ptr(), bins.as_mut_ptr(), bins.len()) }
                == 1
            {
                Ok(())
            } else {
                Err(Error::Failed)
            }
        }
    }

    impl Drop for Native {
        fn drop(&mut self) {
            // SAFETY: exactly one owner of a successful native allocation.
            unsafe { audioscopy_spectral_free(self.0.as_ptr()) }
        }
    }
}

#[cfg(not(audioscopy_wide_spectrum))]
mod native {
    use super::{ComplexPowerBin, Error, PowerBin, SpectrumBackend};

    pub(super) struct Native;

    impl Native {
        pub(super) fn available(_mode: i32) -> bool {
            false
        }
        pub(super) fn new(
            _rate: u32,
            _channels: usize,
            _mode: i32,
            _cross: bool,
        ) -> Result<(Self, usize, SpectrumBackend), Error> {
            Err(Error::Configuration("wide spectrum backend unavailable"))
        }
        pub(super) fn push(&mut self, _pcm: &[f64], _frames: usize) -> Result<(), Error> {
            Err(Error::Failed)
        }
        pub(super) fn finish(&mut self, _bins: &mut [PowerBin]) -> Result<(), Error> {
            Err(Error::Failed)
        }
        pub(super) fn finish_cross(&mut self, _bins: &mut [ComplexPowerBin]) -> Result<(), Error> {
            Err(Error::Failed)
        }
    }
}

use native::Native;

/// Streaming, track-level, high-precision average-power spectrum.
///
/// Only native GNU/Linux and macOS x86-64/AArch64 builds currently provide
/// the optional C11 spectral kernels. Unsupported requests fail explicitly.
/// Double-double has a binary64 exponent internally; block normalization
/// and scaled results support all-subnormal tracks, but sufficiently small
/// terms relative to a larger signal may underflow. No precision mode can
/// undo input quantization or eliminate mathematical window leakage.
/// Input validation expects the normal Rust floating-point environment;
/// the native kernel isolates and restores its own temporary environment.
/// Allocations are bounded by FFT size and channel count, not duration.
/// The ordinary binary64 analyzer is unchanged and remains the default.
pub struct SpectrumAnalyzer {
    format: Format,
    plan: Native,
    backend: SpectrumBackend,
    fft_len: usize,
    frames: u64,
    failed: bool,
}

/// Automatic double-double selection. Every vector backend reproduces the
/// scalar double-double limbs exactly, so this choice changes speed only.
/// Selection is by ISA, detected at run time by the native kernel: AVX2 and
/// FMA on x86-64, Advanced SIMD on AArch64. AVX-512F stays an explicit
/// request; AVX2 is always present when AVX-512F is. Without a vector ISA,
/// the native kernel picks scalar FMA or baseline double-double.
fn auto_double_double() -> SpectrumBackend {
    [
        SpectrumBackend::DoubleDoubleAvx2,
        SpectrumBackend::DoubleDoubleNeon,
    ]
    .into_iter()
    .find(|backend| backend.available())
    .unwrap_or(SpectrumBackend::DoubleDouble)
}

fn select_backend(
    precision: Precision,
    backend: SpectrumBackend,
) -> Result<SpectrumBackend, Error> {
    let selected = match (precision, backend) {
        (Precision::Extended80, SpectrumBackend::Auto) if SpectrumBackend::X87.available() => {
            SpectrumBackend::X87
        }
        (
            Precision::Extended80 | Precision::DoubleDouble,
            SpectrumBackend::Auto | SpectrumBackend::DoubleDouble,
        ) => auto_double_double(),
        (Precision::Extended80, SpectrumBackend::X87) => SpectrumBackend::X87,
        (
            Precision::Extended80 | Precision::DoubleDouble,
            SpectrumBackend::DoubleDoublePortable
            | SpectrumBackend::DoubleDoubleFma
            | SpectrumBackend::DoubleDoubleAvx2
            | SpectrumBackend::DoubleDoubleAvx512
            | SpectrumBackend::DoubleDoubleNeon,
        ) => backend,
        _ => {
            return Err(Error::Configuration(
                "unsupported spectrum precision/backend combination",
            ))
        }
    };
    Ok(selected)
}

fn compatible_backend(selected: SpectrumBackend, actual: SpectrumBackend) -> bool {
    match selected {
        SpectrumBackend::DoubleDouble => matches!(
            actual,
            SpectrumBackend::DoubleDoublePortable | SpectrumBackend::DoubleDoubleFma
        ),
        _ => actual == selected,
    }
}

impl SpectrumAnalyzer {
    /// Choose a wide precision and backend. Extended80/Auto uses x87 when
    /// available, otherwise automatic double-double, which prefers a vector
    /// backend. DoubleDouble never selects x87.
    /// Use the ordinary [`crate::Analyzer`] for Binary64 measurements.
    pub fn new(
        format: Format,
        precision: Precision,
        backend: SpectrumBackend,
    ) -> Result<Self, Error> {
        Self::new_kind(format, precision, backend, false)
    }

    fn new_kind(
        format: Format,
        precision: Precision,
        backend: SpectrumBackend,
        cross: bool,
    ) -> Result<Self, Error> {
        validate_format(&format, Settings::default())?;
        if cross && format.channels != 2 {
            return Err(Error::Configuration("cross spectrum requires stereo"));
        }
        let selected = select_backend(precision, backend)?;
        let (plan, fft_len, actual) = Native::new(
            format.sample_rate,
            format.channels,
            selected.request(),
            cross,
        )?;
        if fft_len != ((format.sample_rate as usize).div_ceil(6)).next_power_of_two()
            || !compatible_backend(selected, actual)
        {
            return Err(Error::Failed);
        }
        Ok(Self {
            format,
            plan,
            backend: actual,
            fft_len,
            frames: 0,
            failed: false,
        })
    }

    /// Actual selected computational backend.
    pub const fn backend(&self) -> SpectrumBackend {
        self.backend
    }

    /// Number of accepted real input frames.
    pub const fn frames(&self) -> u64 {
        self.frames
    }

    /// Ingest whole interleaved frames. Malformed input consumes no frames;
    /// a terminal native error invalidates the analyzer.
    pub fn push_interleaved(&mut self, pcm: &[f64]) -> Result<(), Error> {
        if self.failed {
            return Err(Error::Failed);
        }
        validate_samples(&self.format, pcm)?;
        let frames = pcm.len() / self.format.channels;
        let total = self
            .frames
            .checked_add(frames as u64)
            .filter(|&n| n <= MAX_FRAMES)
            .ok_or(Error::ResourceLimit)?;
        if let Err(error) = self.plan.push(pcm, frames) {
            self.failed = true;
            return Err(error);
        }
        self.frames = total;
        Ok(())
    }

    /// Finish all edge windows and return the wider linear-power bins.
    pub fn finish(mut self) -> Result<PowerSpectrum, Error> {
        if self.failed {
            return Err(Error::Failed);
        }
        if self.frames == 0 {
            return Err(Error::Empty);
        }
        let count = self.fft_len / 2 + 1;
        let mut bins = Vec::new();
        bins.try_reserve_exact(count)
            .map_err(|_| Error::ResourceLimit)?;
        bins.resize(count, PowerBin::default());
        self.plan.finish(&mut bins)?;
        Ok(PowerSpectrum {
            backend: self.backend,
            sample_rate: self.format.sample_rate,
            fft_len: self.fft_len,
            frames: self.frames,
            bins,
        })
    }
}
