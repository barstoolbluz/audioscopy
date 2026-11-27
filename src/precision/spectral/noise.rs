// SPDX-License-Identifier: GPL-2.0-or-later
//! Wide noise statistics for explicitly selected quiet passages. The
//! ordinary automatic quiet detector and its definition are unchanged.
use super::{
    compatible_backend, select_backend, PowerBin, PowerSpectrum, SpectrumAnalyzer, SpectrumBackend,
};
use crate::precision::Precision;
use crate::stream::{validate_format, validate_samples, MAX_FRAMES};
use crate::{Band, Error, Format, NoiseView, Settings, Weighting};

/// Wide, pooled noise power. Each linear value preserves its significand
/// and exponent; logarithmic presentation remains binary64.
#[derive(Clone, Debug, PartialEq)]
pub struct QuietNoiseReport {
    /// Actual selected backend.
    pub backend: SpectrumBackend,
    /// Common sample clock in Hz.
    pub sample_rate: u32,
    /// Fixed integration band and frequency weighting.
    pub view: NoiseView,
    /// Total retained quiet frames.
    pub quiet_frames: u64,
    /// Integrated linear quiet power over the selected view.
    pub floor: PowerBin,
    /// Same five bands as the ordinary noise report: 20-200, 200-2000,
    /// 2000-10000, 10000-20000, 20000-Nyquist Hz. None if no band overlap.
    pub bands: [Option<PowerBin>; 5],
}

impl QuietNoiseReport {
    /// Total retained quiet duration in seconds.
    pub fn quiet_seconds(&self) -> f64 {
        self.quiet_frames as f64 / f64::from(self.sample_rate)
    }

    /// Logarithmic presentation of integrated quiet power.
    pub fn noise_floor_dbfs(&self) -> f64 {
        self.floor.dbfs()
    }

    /// Approximate effective resolution, not source depth or sine-wave ENOB.
    /// Exact-zero power does not bound resolution.
    pub fn effective_bits(&self) -> Option<f64> {
        (self.floor.hi != 0.0).then(|| {
            let log2_power =
                (self.floor.hi + self.floor.lo).log2() + f64::from(self.floor.exponent);
            (-0.5 * log2_power).max(0.0)
        })
    }

    /// Same peak-to-quiet-floor definition as the ordinary report. The
    /// caller supplies the track's nonzero decoded sample peak, not a dB value.
    pub fn dynamic_range_db(&self, peak: f64) -> Option<f64> {
        (peak.is_finite() && peak > 0.0).then(|| 20.0 * peak.log10() - self.floor.dbfs())
    }
}

#[cfg(audioscopy_wide_spectrum)]
mod native {
    use super::{Error, PowerBin, SpectrumBackend};
    use std::{ffi::c_void, ptr::NonNull};

    unsafe extern "C" {
        fn audioscopy_noise_new(
            rate: u32,
            mode: i32,
            view: i32,
            weighting: i32,
            out: *mut *mut c_void,
            actual: *mut i32,
        ) -> i32;
        fn audioscopy_noise_add(
            plan: *mut c_void,
            bins: *const PowerBin,
            count: usize,
            frames: u64,
        ) -> i32;
        fn audioscopy_noise_finish(plan: *mut c_void, out: *mut PowerBin, count: usize) -> i32;
        fn audioscopy_noise_free(plan: *mut c_void);
    }

    pub(super) struct Pool(NonNull<c_void>);
    // SAFETY: one owner, no retained slice, global mutable state or affinity.
    // The native kernel saves and restores the caller's floating environment.
    unsafe impl Send for Pool {}

    impl Pool {
        pub(super) fn new(
            rate: u32,
            mode: i32,
            view: i32,
            weighting: i32,
        ) -> Result<(Self, SpectrumBackend), Error> {
            let mut raw = std::ptr::null_mut();
            let mut actual = 0;
            // SAFETY: writable output slots, no retained pointers to them.
            let status =
                unsafe { audioscopy_noise_new(rate, mode, view, weighting, &mut raw, &mut actual) };
            if status != 1 {
                return Err(match status {
                    -1 => Error::ResourceLimit,
                    -2 => Error::Configuration("wide noise backend unavailable"),
                    _ => Error::Failed,
                });
            }
            let pool = Self(NonNull::new(raw).ok_or(Error::Failed)?);
            Ok((pool, SpectrumBackend::actual(actual)?))
        }

        pub(super) fn add(&mut self, bins: &[PowerBin], frames: u64) -> Result<(), Error> {
            // SAFETY: unique plan; C checks count and bins and retains no pointer.
            if unsafe { audioscopy_noise_add(self.0.as_ptr(), bins.as_ptr(), bins.len(), frames) }
                == 1
            {
                Ok(())
            } else {
                Err(Error::Failed)
            }
        }

        pub(super) fn finish(&mut self, out: &mut [PowerBin; 6]) -> Result<(), Error> {
            // SAFETY: repr(C) output slots, exact count of six.
            if unsafe { audioscopy_noise_finish(self.0.as_ptr(), out.as_mut_ptr(), out.len()) } == 1
            {
                Ok(())
            } else {
                Err(Error::Failed)
            }
        }
    }

    impl Drop for Pool {
        fn drop(&mut self) {
            // SAFETY: the uniquely owned successful native allocation.
            unsafe { audioscopy_noise_free(self.0.as_ptr()) }
        }
    }
}

#[cfg(not(audioscopy_wide_spectrum))]
mod native {
    use super::{Error, PowerBin, SpectrumBackend};
    pub(super) struct Pool;
    impl Pool {
        pub(super) fn new(
            _rate: u32,
            _mode: i32,
            _view: i32,
            _weighting: i32,
        ) -> Result<(Self, SpectrumBackend), Error> {
            Err(Error::Configuration("wide noise backend unavailable"))
        }
        pub(super) fn add(&mut self, _bins: &[PowerBin], _frames: u64) -> Result<(), Error> {
            Err(Error::Failed)
        }
        pub(super) fn finish(&mut self, _out: &mut [PowerBin; 6]) -> Result<(), Error> {
            Err(Error::Failed)
        }
    }
}

/// Bounded reduction of already measured, caller-identified quiet passages
/// at a single clock. Passages below the minimum duration are ignored.
/// The caller must identify quiet spans; this is not the automatic detector.
/// The same weighting and bin inclusion convention as the ordinary floor
/// applies. No PCM or spectral arrays are retained in the pool.
pub struct QuietNoisePool {
    backend: SpectrumBackend,
    sample_rate: u32,
    view: NoiseView,
    minimum_frames: u64,
    frames: u64,
    pool: native::Pool,
    failed: bool,
}

impl QuietNoisePool {
    /// Create a pool with a fixed clock, view and minimum quiet duration.
    pub fn new(
        sample_rate: u32,
        shortest_gap_seconds: f64,
        view: NoiseView,
        precision: Precision,
        backend: SpectrumBackend,
    ) -> Result<Self, Error> {
        if !(8000..=768000).contains(&sample_rate) || !(0.05..=60.0).contains(&shortest_gap_seconds)
        {
            return Err(Error::Configuration(
                "invalid wide noise sample rate or gap duration",
            ));
        }
        let selected = select_backend(precision, backend)?;
        let band = match view.band {
            Band::Audible => 0,
            Band::ToNyquist => 1,
            Band::Full => 2,
        };
        let weighting = match view.weighting {
            Weighting::Unweighted => 0,
            Weighting::A => 1,
            Weighting::ItuR468 => 2,
        };
        let (pool, actual) = native::Pool::new(sample_rate, selected.request(), band, weighting)?;
        if !compatible_backend(selected, actual) {
            return Err(Error::Failed);
        }
        Ok(Self {
            backend: actual,
            sample_rate,
            view,
            minimum_frames: (shortest_gap_seconds * f64::from(sample_rate)).ceil() as u64,
            frames: 0,
            pool,
            failed: false,
        })
    }

    /// Actual selected backend.
    pub const fn backend(&self) -> SpectrumBackend {
        self.backend
    }
    /// Number of retained quiet frames.
    pub const fn quiet_frames(&self) -> u64 {
        self.frames
    }

    /// Add one independently windowed contiguous passage. Returns false for
    /// a too-short passage; true for a retained passage. Clock and actual
    /// arithmetic backend must match. Do not pass an arbitrary track spectrum
    /// unless the whole track is intended to be considered quiet.
    pub fn add_passage(&mut self, passage: &PowerSpectrum) -> Result<bool, Error> {
        if self.failed {
            return Err(Error::Failed);
        }
        let n = (self.sample_rate as usize).div_ceil(6).next_power_of_two();
        if passage.backend != self.backend
            || passage.sample_rate != self.sample_rate
            || passage.fft_len != n
            || passage.bins.len() != n / 2 + 1
            || passage.frames == 0
            || passage.frames > MAX_FRAMES
        {
            return Err(Error::Configuration("incompatible quiet passage spectrum"));
        }
        if passage.frames < self.minimum_frames {
            return Ok(false);
        }
        let total = self
            .frames
            .checked_add(passage.frames)
            .filter(|n| *n <= MAX_FRAMES)
            .ok_or(Error::ResourceLimit)?;
        if let Err(error) = self.pool.add(&passage.bins, passage.frames) {
            self.failed = true;
            return Err(error);
        }
        self.frames = total;
        Ok(true)
    }

    /// None means no eligible passage. A silent eligible passage instead
    /// returns an exact-zero floor and no effective-resolution estimate.
    pub fn finish(mut self) -> Result<Option<QuietNoiseReport>, Error> {
        if self.failed {
            return Err(Error::Failed);
        }
        if self.frames == 0 {
            return Ok(None);
        }
        let mut values = [PowerBin::default(); 6];
        self.pool.finish(&mut values)?;
        let (lo, hi) = crate::spectrum::limits(self.view.band, self.sample_rate);
        let edges = [20.0_f64, 200.0, 2000.0, 10_000.0, 20_000.0, f64::INFINITY];
        let bands = std::array::from_fn(|i| {
            (hi.min(edges[i + 1]) > lo.max(edges[i])).then_some(values[i + 1])
        });
        Ok(Some(QuietNoiseReport {
            backend: self.backend,
            sample_rate: self.sample_rate,
            view: self.view,
            quiet_frames: self.frames,
            floor: values[0],
            bands,
        }))
    }
}

/// Streaming convenience wrapper for explicitly delimited quiet runs.
/// Separators never enter the spectrum. End each run separately, so window
/// padding and duration eligibility match independently processed passages.
/// It uses one bounded spectrum plan at a time, not a retained PCM archive.
pub struct QuietNoiseAnalyzer {
    format: Format,
    precision: Precision,
    pool: QuietNoisePool,
    run: Option<SpectrumAnalyzer>,
    failed: bool,
}

impl QuietNoiseAnalyzer {
    /// Create a streaming analyzer for caller-selected quiet passages.
    pub fn new(
        format: Format,
        shortest_gap_seconds: f64,
        view: NoiseView,
        precision: Precision,
        backend: SpectrumBackend,
    ) -> Result<Self, Error> {
        validate_format(&format, Settings::default())?;
        let pool = QuietNoisePool::new(
            format.sample_rate,
            shortest_gap_seconds,
            view,
            precision,
            backend,
        )?;
        Ok(Self {
            format,
            precision,
            pool,
            run: None,
            failed: false,
        })
    }

    /// Ingest complete frames belonging to the current quiet passage.
    pub fn push_interleaved(&mut self, pcm: &[f64]) -> Result<(), Error> {
        if self.failed {
            return Err(Error::Failed);
        }
        if pcm.is_empty() {
            return Ok(());
        }
        if self.run.is_none() {
            validate_samples(&self.format, pcm)?;
            self.run = Some(SpectrumAnalyzer::new(
                self.format.clone(),
                self.precision,
                self.pool.backend(),
            )?);
        }
        // SpectrumAnalyzer validates transactionally and poisons native failures.
        let result = self
            .run
            .as_mut()
            .expect("initialized run")
            .push_interleaved(pcm);
        if matches!(&result, Err(Error::Failed)) {
            self.failed = true;
        }
        result
    }

    /// End the current passage; returns whether its duration qualified.
    pub fn end_passage(&mut self) -> Result<bool, Error> {
        if self.failed {
            return Err(Error::Failed);
        }
        let Some(run) = self.run.take() else {
            return Ok(false);
        };
        if run.frames() < self.pool.minimum_frames {
            return Ok(false);
        }
        let result = run.finish().and_then(|s| self.pool.add_passage(&s));
        if result.is_err() {
            self.failed = true;
        }
        result
    }

    /// Close the current passage and return the pooled report.
    pub fn finish(mut self) -> Result<Option<QuietNoiseReport>, Error> {
        self.end_passage()?;
        self.pool.finish()
    }
}
