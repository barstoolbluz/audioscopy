// SPDX-License-Identifier: GPL-2.0-or-later
//! Optional streaming precision modes for selected track measurements.
//!
//! This additive interface is deliberately separate from the ordinary
//! analyzer: no extra state, dispatch or allocation in its default hot path.
//! Correlation is covered here. The separate `precision-spectrum` feature
//! adds wide power, cross-spectrum and quiet-passage reduction. The separate
//! `precision-loudness` feature adds wider native integrated loudness.
//! No wider album or LRA claim.
mod dd;
#[cfg(any(feature = "precision-spectrum", feature = "precision-loudness"))]
mod scaled;
#[cfg(any(feature = "precision-spectrum", feature = "precision-loudness"))]
pub use scaled::PowerBin;
#[cfg(feature = "precision-loudness")]
pub mod loudness;
#[cfg(feature = "precision-spectrum")]
pub mod spectral;
use crate::stream::{validate_format, validate_samples, MAX_FRAMES};
use crate::temporal::Stereo;
use crate::{Error, Format, Settings};
use dd::{Dd, Moments};

const BLOCK: usize = 1024;

/// Requested arithmetic precision for optional measurement backends.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Precision {
    /// Existing conditioned binary64 correlation.
    #[default]
    Binary64,
    /// At least the 64-bit significand of x87; Auto may use double-double.
    Extended80,
    /// Portable double-double significand, with binary64 exponent limits.
    DoubleDouble,
}

/// Explicit computational backend selection.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Backend {
    /// Select a backend that satisfies the requested precision.
    #[default]
    Auto,
    /// Existing binary64 implementation.
    Binary64,
    /// Native x87, initially available on native GNU/Linux x86-64 builds.
    X87,
    /// Portable double-double using error-free transformations and FMA.
    DoubleDouble,
}

impl Backend {
    /// Whether this build can select the backend on the current platform.
    pub const fn available(self) -> bool {
        !matches!(self, Self::X87) || cfg!(audioscopy_x87)
    }
}

/// Non-overlapping binary64 limbs representing one correlation result.
/// Read the limbs separately to retain the additional significand bits.
/// An exact-zero variance uses the existing correlation convention of zero.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Correlation {
    /// Rounded high component.
    pub hi: f64,
    /// Signed residual from the wider result, not a measurement error bound.
    pub lo: f64,
    /// Actual backend, never Auto.
    pub backend: Backend,
}

impl Correlation {
    /// Arithmetic precision actually used.
    pub const fn precision(self) -> Precision {
        match self.backend {
            Backend::X87 => Precision::Extended80,
            Backend::DoubleDouble => Precision::DoubleDouble,
            _ => Precision::Binary64,
        }
    }

    /// Compatibility view rounded to binary64 (loses the low component).
    pub fn as_f64(self) -> f64 {
        self.hi + self.lo
    }
}

#[cfg(audioscopy_x87)]
mod x87 {
    use super::{Dd, Error, BLOCK};

    unsafe extern "C" {
        fn audioscopy_x87_push(state: *mut u8, prior: u64, pcm: *const f64, n: usize) -> i32;
        fn audioscopy_x87_finish(state: *const u8, out: *mut f64) -> i32;
    }

    // Seven canonical 80-bit values; no C long double crosses the FFI ABI.
    pub(super) struct State([u8; 70]);

    impl Default for State {
        fn default() -> Self {
            Self([0; 70])
        }
    }

    impl State {
        pub(super) fn push(&mut self, pcm: &[f64], prior: u64) -> Result<(), Error> {
            assert!(!pcm.is_empty() && pcm.len() <= BLOCK * 2 && pcm.len().is_multiple_of(2));
            // SAFETY: state is 70 initialized bytes, samples are finite,
            // contiguous complete stereo frames. C retains no pointers;
            // count and block sizes are bounded by the caller and C.
            let ok = unsafe {
                audioscopy_x87_push(self.0.as_mut_ptr(), prior, pcm.as_ptr(), pcm.len() / 2)
            };
            if ok == 1 {
                Ok(())
            } else {
                Err(Error::Failed)
            }
        }

        pub(super) fn finish(&self) -> Result<Dd, Error> {
            let mut limbs = [0.0; 2];
            // SAFETY: initialized 70-byte state and two writable f64 slots.
            let ok = unsafe { audioscopy_x87_finish(self.0.as_ptr(), limbs.as_mut_ptr()) };
            if ok == 1 {
                Ok(Dd {
                    hi: limbs[0],
                    lo: limbs[1],
                })
            } else {
                Err(Error::Failed)
            }
        }
    }
}

// Portable DD relies on the ordinary Rust binary64 environment (nearest,
// gradual underflow). Detect incompatible externally modified modes rather
// than silently reducing precision. No global or thread-local mode changes.
fn dd_environment_ok() -> bool {
    use std::hint::black_box;
    let one = black_box(1.0_f64);
    let half = black_box(f64::EPSILON * 0.5);
    let above = black_box(f64::EPSILON * 0.75);
    let normal = black_box(f64::MIN_POSITIVE);
    let tiny = black_box(f64::from_bits(1));
    one + half == one
        && -one - half == -one
        && one + above > one
        && -one - above < -one
        && (normal * black_box(0.5)).to_bits() == (1_u64 << 51)
        && (tiny * black_box(2.0)).to_bits() == 2
}

enum State {
    Binary64(Stereo),
    DoubleDouble(Moments),
    #[cfg(audioscopy_x87)]
    X87(x87::State),
}

impl State {
    fn push(&mut self, pcm: &[f64], prior: u64) -> Result<(), Error> {
        match self {
            Self::Binary64(s) => {
                for f in pcm.as_chunks::<2>().0 {
                    s.push(f[0], f[1]);
                }
            }
            Self::DoubleDouble(s) => s.push(pcm, prior),
            #[cfg(audioscopy_x87)]
            Self::X87(s) => s.push(pcm, prior)?,
        }
        Ok(())
    }

    fn finish(&self) -> Result<Dd, Error> {
        match self {
            Self::Binary64(s) => Ok(Dd::from(s.correlation())),
            Self::DoubleDouble(s) => Ok(s.finish()),
            #[cfg(audioscopy_x87)]
            Self::X87(s) => s.finish(),
        }
    }
}

/// Streaming, track-level Pearson correlation with explicit precision.
///
/// Takes the same decoded PCM and format as [`crate::Analyzer`], restricted
/// to stereo. Fixed internal 1024-frame blocks make arithmetic independent
/// of caller chunk boundaries. A malformed push consumes no frames. This
/// analyzer does not perform any other selected measurement or album reducer.
pub struct CorrelationAnalyzer {
    format: Format,
    backend: Backend,
    state: State,
    pending: Vec<f64>,
    processed: u64,
    failed: bool,
}

impl CorrelationAnalyzer {
    /// Choose requested precision and backend; incompatible explicit choices
    /// return a configuration error. Auto never downgrades to binary64.
    pub fn new(format: Format, precision: Precision, backend: Backend) -> Result<Self, Error> {
        validate_format(&format, Settings::default())?;
        if format.channels != 2 {
            return Err(Error::Configuration("correlation requires stereo PCM"));
        }
        let backend = match (precision, backend) {
            (Precision::Binary64, Backend::Auto | Backend::Binary64) => Backend::Binary64,
            (Precision::Extended80, Backend::Auto) if Backend::X87.available() => Backend::X87,
            (Precision::Extended80, Backend::Auto | Backend::DoubleDouble)
            | (Precision::DoubleDouble, Backend::Auto | Backend::DoubleDouble) => {
                Backend::DoubleDouble
            }
            (Precision::Extended80, Backend::X87) if Backend::X87.available() => Backend::X87,
            _ => {
                return Err(Error::Configuration(
                    "unsupported correlation precision/backend combination",
                ))
            }
        };
        let state = match backend {
            Backend::Binary64 => State::Binary64(Stereo::new(true, false, false, false)),
            Backend::DoubleDouble => {
                if !dd_environment_ok() {
                    return Err(Error::Configuration(
                        "double-double requires nearest rounding and gradual underflow",
                    ));
                }
                State::DoubleDouble(Moments::default())
            }
            #[cfg(audioscopy_x87)]
            Backend::X87 => State::X87(x87::State::default()),
            _ => return Err(Error::Configuration("correlation backend unavailable")),
        };
        Ok(Self {
            format,
            backend,
            state,
            pending: Vec::with_capacity(BLOCK * 2),
            processed: 0,
            failed: false,
        })
    }

    /// Actual selected backend.
    pub const fn backend(&self) -> Backend {
        self.backend
    }

    /// Accepted frame count, including the current incomplete internal block.
    pub fn frames(&self) -> u64 {
        self.processed + (self.pending.len() / 2) as u64
    }

    fn check_environment(&self) -> Result<(), Error> {
        if self.backend == Backend::DoubleDouble && !dd_environment_ok() {
            Err(Error::Configuration(
                "double-double requires nearest rounding and gradual underflow",
            ))
        } else {
            Ok(())
        }
    }

    fn flush(&mut self) -> Result<(), Error> {
        if self.pending.is_empty() {
            return Ok(());
        }
        if let Err(e) = self.state.push(&self.pending, self.processed) {
            self.failed = true;
            return Err(e);
        }
        self.processed += (self.pending.len() / 2) as u64;
        self.pending.clear();
        Ok(())
    }

    /// Validate and ingest whole stereo frames; empty pushes are no-ops.
    pub fn push_interleaved(&mut self, mut pcm: &[f64]) -> Result<(), Error> {
        if self.failed {
            return Err(Error::Failed);
        }
        self.check_environment()?;
        validate_samples(&self.format, pcm)?;
        if self
            .frames()
            .checked_add((pcm.len() / 2) as u64)
            .is_none_or(|n| n > MAX_FRAMES)
        {
            return Err(Error::ResourceLimit);
        }
        while !pcm.is_empty() {
            if self.pending.is_empty() && pcm.len() >= BLOCK * 2 {
                // Avoid copying full blocks; maintain exactly the same
                // boundaries as the buffered path.
                let (block, rest) = pcm.split_at(BLOCK * 2);
                if let Err(e) = self.state.push(block, self.processed) {
                    self.failed = true;
                    return Err(e);
                }
                self.processed += BLOCK as u64;
                pcm = rest;
            } else {
                let take = (BLOCK * 2 - self.pending.len()).min(pcm.len());
                self.pending.extend_from_slice(&pcm[..take]);
                pcm = &pcm[take..];
                if self.pending.len() == BLOCK * 2 {
                    self.flush()?;
                }
            }
        }
        Ok(())
    }

    /// Finish the last internal block and return both result components.
    pub fn finish(mut self) -> Result<Correlation, Error> {
        if self.failed {
            return Err(Error::Failed);
        }
        self.check_environment()?;
        if self.frames() == 0 {
            return Err(Error::Empty);
        }
        self.flush()?;
        let r = self.state.finish()?;
        if !r.hi.is_finite() || !r.lo.is_finite() {
            return Err(Error::Failed);
        }
        Ok(Correlation {
            hi: r.hi,
            lo: r.lo,
            backend: self.backend,
        })
    }
}
