// SPDX-License-Identifier: GPL-2.0-or-later
//! Optional wide integrated native-profile loudness. The ordinary
//! true-peak loudness meters and libebur128 compatibility remain unchanged.
use super::{PowerBin, Precision};
use crate::stream::MAX_FRAMES;
use crate::{ChannelRole, Error};

const MAX_OBSERVATIONS: usize = 4_000_000;

/// Computational backend for the optional native integrated meter.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LoudnessBackend {
    /// Select a backend satisfying the requested precision.
    #[default]
    Auto,
    /// Genuine extended80 on supported x86-64 builds.
    X87,
    /// Automatically select a double-double execution path.
    DoubleDouble,
    /// Double-double with baseline ISA.
    DoubleDoublePortable,
    /// Double-double with explicit x86 scalar FMA target.
    DoubleDoubleFma,
}

impl LoudnessBackend {
    const fn request(self) -> i32 {
        match self {
            Self::Auto | Self::DoubleDouble => 1,
            Self::X87 => 2,
            Self::DoubleDoublePortable => 3,
            Self::DoubleDoubleFma => 4,
        }
    }

    #[cfg(audioscopy_wide_loudness)]
    fn actual(value: i32) -> Result<Self, Error> {
        match value {
            2 => Ok(Self::X87),
            3 => Ok(Self::DoubleDoublePortable),
            4 => Ok(Self::DoubleDoubleFma),
            _ => Err(Error::Failed),
        }
    }

    /// Whether the actual build and processor support this request.
    pub fn available(self) -> bool {
        native::Meter::available(self.request())
    }
}

/// Typed measurement outcome, with no numeric sentinel for no evidence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoudnessOutcome {
    /// No decoded input frames.
    NoInput,
    /// No complete 400 ms window.
    InsufficientFrames,
    /// No observations above the absolute gate.
    BelowAbsoluteGate,
    /// No observations above the relative gate.
    BelowRelativeGate,
    /// Finite gated integrated energy.
    Finite,
}

/// Integrated native-profile measurement with wide linear gated energy.
/// LUFS presentation uses binary64 logarithms; read the power limbs for
/// additional numerical resolution. Exact-decimal prototype and gate
/// constants can intentionally change a decision near a rounded boundary.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WideLoudness {
    /// Actual selected backend.
    pub backend: LoudnessBackend,
    /// Typed measurement outcome.
    pub outcome: LoudnessOutcome,
    /// Real input frame count.
    pub frames: u64,
    /// Frame count needed for the first complete window.
    pub required_frames: u64,
    /// Complete windows strictly above the absolute gate.
    pub absolute_observations: u64,
    /// Absolute-eligible windows strictly above the relative gate.
    pub relative_observations: u64,
    /// Wide linear gated programme energy, present only for Finite.
    pub energy: Option<PowerBin>,
}

impl WideLoudness {
    /// Arithmetic precision actually used.
    pub fn precision(self) -> Precision {
        if self.backend == LoudnessBackend::X87 {
            Precision::Extended80
        } else {
            Precision::DoubleDouble
        }
    }

    /// Rounded logarithmic presentation, or None for no finite result.
    pub fn lufs(self) -> Option<f64> {
        self.energy.map(|p| -0.691 + p.dbfs())
    }
}

#[repr(C)]
#[derive(Default)]
struct NativeResult {
    frames: u64,
    required_frames: u64,
    absolute_count: u64,
    relative_count: u64,
    outcome: i32,
    energy: PowerBin,
}

#[cfg(audioscopy_wide_loudness)]
mod native {
    use super::{Error, LoudnessBackend, NativeResult};
    use std::{ffi::c_void, ptr::NonNull};

    unsafe extern "C" {
        fn audioscopy_spectral_available(mode: i32) -> i32;
        fn audioscopy_loudness_new(
            rate: u32,
            channels: usize,
            weights: *const u8,
            limit: usize,
            mode: i32,
            out: *mut *mut c_void,
            actual: *mut i32,
        ) -> i32;
        fn audioscopy_loudness_push(plan: *mut c_void, pcm: *const f64, frames: usize) -> i32;
        fn audioscopy_loudness_finish(plan: *mut c_void, out: *mut NativeResult) -> i32;
        fn audioscopy_loudness_free(plan: *mut c_void);
    }

    pub(super) struct Meter(NonNull<c_void>);

    // SAFETY: one owner, no retained PCM pointers, global mutable state or
    // affinity. Native calls preserve the invoking thread's FP environment.
    unsafe impl Send for Meter {}

    impl Meter {
        pub(super) fn available(mode: i32) -> bool {
            // SAFETY: pure runtime feature query.
            unsafe { audioscopy_spectral_available(mode) == 1 }
        }

        pub(super) fn new(
            rate: u32,
            weights: &[u8],
            limit: usize,
            mode: i32,
        ) -> Result<(Self, LoudnessBackend), Error> {
            let mut raw = std::ptr::null_mut();
            let mut actual = 0;
            // SAFETY: C copies the weights, output slots live for the call.
            let status = unsafe {
                audioscopy_loudness_new(
                    rate,
                    weights.len(),
                    weights.as_ptr(),
                    limit,
                    mode,
                    &mut raw,
                    &mut actual,
                )
            };
            if status != 1 {
                return Err(match status {
                    -1 => Error::ResourceLimit,
                    -2 => Error::Configuration("wide loudness backend unavailable"),
                    _ => Error::Failed,
                });
            }
            let meter = Self(NonNull::new(raw).ok_or(Error::Failed)?);
            Ok((meter, LoudnessBackend::actual(actual)?))
        }

        pub(super) fn push(&mut self, pcm: &[f64], frames: usize) -> Result<(), Error> {
            // SAFETY: unique plan, complete admitted finite frames, no
            // retained PCM pointer. The caller bounds cumulative frame count.
            match unsafe { audioscopy_loudness_push(self.0.as_ptr(), pcm.as_ptr(), frames) } {
                1 => Ok(()),
                -1 => Err(Error::ResourceLimit),
                _ => Err(Error::Failed),
            }
        }

        pub(super) fn finish(&mut self) -> Result<NativeResult, Error> {
            let mut result = NativeResult::default();
            // SAFETY: uniquely owned plan and repr(C) writable output slot.
            if unsafe { audioscopy_loudness_finish(self.0.as_ptr(), &mut result) } == 1 {
                Ok(result)
            } else {
                Err(Error::Failed)
            }
        }
    }

    impl Drop for Meter {
        fn drop(&mut self) {
            // SAFETY: one owner of a successful native allocation.
            unsafe { audioscopy_loudness_free(self.0.as_ptr()) }
        }
    }
}

#[cfg(not(audioscopy_wide_loudness))]
mod native {
    use super::{Error, LoudnessBackend, NativeResult};
    pub(super) struct Meter;
    impl Meter {
        pub(super) fn available(_mode: i32) -> bool {
            false
        }
        pub(super) fn new(
            _rate: u32,
            _weights: &[u8],
            _limit: usize,
            _mode: i32,
        ) -> Result<(Self, LoudnessBackend), Error> {
            Err(Error::Configuration("wide loudness backend unavailable"))
        }
        pub(super) fn push(&mut self, _pcm: &[f64], _frames: usize) -> Result<(), Error> {
            Err(Error::Failed)
        }
        pub(super) fn finish(&mut self) -> Result<NativeResult, Error> {
            Err(Error::Failed)
        }
    }
}

/// Streaming integrated-only measurement on the native rational 100 ms
/// clock with complete 400 ms windows. No zero-padded EOF observations.
/// Accepts decoded finite PCM in [-16, 16] and the full native meter rate
/// envelope 3364..=2822400 Hz. Native role admission matches the ordinary
/// native profile. Persistent memory is limited to accepted absolute-gate
/// energies and fixed channel/window state, not a per-frame ring. An
/// observation-limit or native failure is terminal; invalid PCM is not.
/// No LRA, short-term observations, libebur128 or album pooling is implied.
pub struct WideLoudnessMeter {
    backend: LoudnessBackend,
    meter: native::Meter,
    channels: usize,
    frames: u64,
    required_frames: u64,
    failed: bool,
}

impl WideLoudnessMeter {
    /// Create a native integrated meter with the default observation limit.
    pub fn new(
        rate: u32,
        roles: &[ChannelRole],
        precision: Precision,
        backend: LoudnessBackend,
    ) -> Result<Self, Error> {
        Self::with_observation_limit(rate, roles, precision, backend, MAX_OBSERVATIONS)
    }

    /// Create a meter with an explicit maximum retained observation count.
    pub fn with_observation_limit(
        rate: u32,
        roles: &[ChannelRole],
        precision: Precision,
        backend: LoudnessBackend,
        limit: usize,
    ) -> Result<Self, Error> {
        if !(3364..=2822400).contains(&rate)
            || !(1..=64).contains(&roles.len())
            || !(1..=MAX_OBSERVATIONS).contains(&limit)
        {
            return Err(Error::Configuration(
                "invalid wide loudness rate, channels or limit",
            ));
        }
        if roles.contains(&ChannelRole::Unused)
            || (roles.len() > 1 && roles.contains(&ChannelRole::Mono))
            || roles
                .iter()
                .enumerate()
                .any(|(i, r)| roles[..i].contains(r))
        {
            return Err(Error::Configuration("invalid native loudness roles"));
        }
        let mut weights = [0_u8; 64];
        for (w, role) in weights.iter_mut().zip(roles) {
            *w = match role {
                ChannelRole::Lfe | ChannelRole::Unused => 0,
                ChannelRole::LeftSurround
                | ChannelRole::RightSurround
                | ChannelRole::LeftBack
                | ChannelRole::RightBack => 2,
                _ => 1,
            };
        }
        let weights = &weights[..roles.len()];
        if weights.iter().all(|&w| w == 0) {
            return Err(Error::Configuration("no weighted programme channels"));
        }
        use LoudnessBackend as B;
        let selected = match (precision, backend) {
            (Precision::Extended80, B::Auto) if B::X87.available() => B::X87,
            (Precision::Extended80 | Precision::DoubleDouble, B::Auto) => B::DoubleDouble,
            (Precision::Extended80, B::X87) => B::X87,
            (
                Precision::Extended80 | Precision::DoubleDouble,
                B::DoubleDouble | B::DoubleDoublePortable | B::DoubleDoubleFma,
            ) => backend,
            _ => {
                return Err(Error::Configuration(
                    "unsupported loudness precision/backend combination",
                ))
            }
        };
        let (meter, actual) = native::Meter::new(rate, weights, limit, selected.request())?;
        if !(actual == selected
            || (selected == B::DoubleDouble
                && matches!(actual, B::DoubleDoublePortable | B::DoubleDoubleFma)))
        {
            return Err(Error::Failed);
        }
        Ok(Self {
            backend: actual,
            meter,
            channels: roles.len(),
            frames: 0,
            required_frames: (2 * u64::from(rate) + 2) / 5,
            failed: false,
        })
    }

    /// Actual selected backend.
    pub const fn backend(&self) -> LoudnessBackend {
        self.backend
    }

    /// Number of accepted real frames.
    pub const fn frames(&self) -> u64 {
        self.frames
    }

    /// Ingest whole interleaved frames without padding.
    pub fn push_interleaved(&mut self, pcm: &[f64]) -> Result<(), Error> {
        if self.failed {
            return Err(Error::Failed);
        }
        if !pcm.len().is_multiple_of(self.channels) {
            return Err(Error::IncompleteFrame);
        }
        for (i, x) in pcm.iter().enumerate() {
            if !x.is_finite() || x.abs() > 16.0 {
                return Err(Error::InvalidSample(i));
            }
        }
        let frames = pcm.len() / self.channels;
        let total = self
            .frames
            .checked_add(frames as u64)
            .filter(|n| *n <= MAX_FRAMES)
            .ok_or(Error::ResourceLimit)?;
        if let Err(error) = self.meter.push(pcm, frames) {
            self.failed = true;
            return Err(error);
        }
        self.frames = total;
        Ok(())
    }

    /// Reduce complete gated windows and return the typed result.
    pub fn finish(mut self) -> Result<WideLoudness, Error> {
        if self.failed {
            return Err(Error::Failed);
        }
        let r = self.meter.finish()?;
        if r.frames != self.frames || r.required_frames != self.required_frames {
            return Err(Error::Failed);
        }
        let outcome = match r.outcome {
            0 => LoudnessOutcome::NoInput,
            1 => LoudnessOutcome::InsufficientFrames,
            2 => LoudnessOutcome::BelowAbsoluteGate,
            3 => LoudnessOutcome::BelowRelativeGate,
            4 => LoudnessOutcome::Finite,
            _ => return Err(Error::Failed),
        };
        Ok(WideLoudness {
            backend: self.backend,
            outcome,
            frames: r.frames,
            required_frames: r.required_frames,
            absolute_observations: r.absolute_count,
            relative_observations: r.relative_count,
            energy: (outcome == LoudnessOutcome::Finite).then_some(r.energy),
        })
    }
}
