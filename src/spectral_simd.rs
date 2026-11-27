//! Opt-in power-only FFT postprocessing candidates. The default build does
//! not dispatch here. No changes to RustFFT, weighting, or accumulation order.
use super::{accumulate_power, C};
use std::arch::x86_64::*;

// num_complex::Complex<f64> is repr(C). Check the layout assumed by the
// interleaved vector loads at compile time, including both member offsets.
const _: () = {
    assert!(std::mem::size_of::<C>() == 2 * std::mem::size_of::<f64>());
    assert!(std::mem::align_of::<C>() == std::mem::align_of::<f64>());
    assert!(std::mem::offset_of!(C, re) == 0);
    assert!(std::mem::offset_of!(C, im) == std::mem::size_of::<f64>());
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Backend(Kind);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Portable,
    #[cfg(any(test, feature = "experimental-spectral-avx2"))]
    Avx2,
    #[cfg(any(test, feature = "experimental-spectral-avx512"))]
    Avx512,
}

impl Backend {
    pub(super) const fn portable() -> Self {
        Self(Kind::Portable)
    }

    // Detect once per power-collecting SpectralEngine; no CPUID in the loop.
    pub(super) fn select() -> Self {
        #[cfg(feature = "experimental-spectral-avx2")]
        if let Some(backend) = Self::avx2() {
            return backend;
        }
        #[cfg(feature = "experimental-spectral-avx512")]
        if let Some(backend) = Self::avx512() {
            return backend;
        }
        Self::portable()
    }

    #[cfg(any(test, feature = "experimental-spectral-avx2"))]
    fn avx2() -> Option<Self> {
        std::is_x86_feature_detected!("avx2").then_some(Self(Kind::Avx2))
    }

    #[cfg(any(test, feature = "experimental-spectral-avx512"))]
    fn avx512() -> Option<Self> {
        std::is_x86_feature_detected!("avx512f").then_some(Self(Kind::Avx512))
    }

    #[cfg(test)]
    const fn name(self) -> &'static str {
        match self.0 {
            Kind::Portable => "portable",
            Kind::Avx2 => "avx2",
            Kind::Avx512 => "avx512f",
        }
    }

    // These target-feature entry points let LLVM vectorize independent
    // stereo bins at the selected width without handwritten duplicate DSP.
    // Actual Rust codegen and performance still require native evidence.
    #[cfg(feature = "experimental-stereo-spectrum-split")]
    pub(super) fn bass(self, l: &[C], r: &[C], mid: &mut [f64], side: &mut [f64]) {
        match self.0 {
            Kind::Portable => super::spectral_stereo::bass(l, r, mid, side),
            #[cfg(any(test, feature = "experimental-spectral-avx2"))]
            Kind::Avx2 => {
                // SAFETY: constructed only after AVX2 runtime detection.
                unsafe { bass_avx2(l, r, mid, side) }
            }
            #[cfg(any(test, feature = "experimental-spectral-avx512"))]
            Kind::Avx512 => {
                // SAFETY: constructed only after AVX512F runtime detection.
                unsafe { bass_avx512(l, r, mid, side) }
            }
        }
    }

    #[cfg(feature = "experimental-stereo-spectrum-split")]
    pub(super) fn cross(self, l: &[C], r: &[C], dst: &mut [C]) {
        match self.0 {
            Kind::Portable => super::spectral_stereo::cross(l, r, dst),
            #[cfg(any(test, feature = "experimental-spectral-avx2"))]
            Kind::Avx2 => {
                // SAFETY: constructed only after AVX2 runtime detection.
                unsafe { cross_avx2(l, r, dst) }
            }
            #[cfg(any(test, feature = "experimental-spectral-avx512"))]
            Kind::Avx512 => {
                // SAFETY: constructed only after AVX512F runtime detection.
                unsafe { cross_avx512(l, r, dst) }
            }
        }
    }

    pub(super) fn accumulate(self, bins: &[C], power: &mut [f64]) {
        assert_eq!(bins.len(), power.len());
        assert!(power.len() >= 2);
        if self == Self::portable() {
            accumulate_power(bins, power);
            return;
        }
        let last = power.len() - 1;
        // DC and Nyquist have the original weight of one, including signed
        // zero behavior. The vector functions receive only interior bins.
        power[0] += bins[0].norm_sqr();
        let input = &bins[1..last];
        let sums = &mut power[1..last];
        match self.0 {
            Kind::Portable => unreachable!(),
            #[cfg(any(test, feature = "experimental-spectral-avx2"))]
            Kind::Avx2 => {
                // SAFETY: the private constructor checked AVX2; the slices
                // have identical lengths and do not overlap.
                unsafe { interior_avx2(input, sums) }
            }
            #[cfg(any(test, feature = "experimental-spectral-avx512"))]
            Kind::Avx512 => {
                // SAFETY: the private constructor checked AVX512F; the slices
                // have identical lengths and do not overlap.
                unsafe { interior_avx512(input, sums) }
            }
        }
        power[last] += bins[last].norm_sqr();
    }
}

#[inline]
fn tail(bins: &[C], sums: &mut [f64]) {
    for (sum, x) in sums.iter_mut().zip(bins) {
        *sum += 2.0 * x.norm_sqr();
    }
}

#[cfg(any(test, feature = "experimental-spectral-avx2"))]
#[target_feature(enable = "avx2")]
#[inline(never)]
unsafe fn interior_avx2(bins: &[C], sums: &mut [f64]) {
    let mut i = 0;
    let two = _mm256_set1_pd(2.0);
    // Square the interleaved real/imaginary components, add within each
    // complex bin, then permute to increasing bin order. The horizontal
    // add never combines different bins; all FP operations stay separate.
    while bins.len() - i >= 4 {
        // SAFETY: i..i+4 is inside both equal-length slices. The two loads
        // each cover two complex bins; unaligned access is intentional.
        unsafe {
            let ptr = bins.as_ptr().add(i).cast::<f64>();
            let a = _mm256_loadu_pd(ptr);
            let b = _mm256_loadu_pd(ptr.add(4));
            let p = _mm256_permute4x64_pd::<0xd8>(_mm256_hadd_pd(
                _mm256_mul_pd(a, a),
                _mm256_mul_pd(b, b),
            ));
            let dst = sums.as_mut_ptr().add(i);
            let next = _mm256_add_pd(_mm256_loadu_pd(dst), _mm256_mul_pd(two, p));
            _mm256_storeu_pd(dst, next);
        }
        i += 4;
    }
    tail(&bins[i..], &mut sums[i..]);
}

#[cfg(any(test, feature = "experimental-spectral-avx512"))]
#[target_feature(enable = "avx512f")]
#[inline(never)]
unsafe fn interior_avx512(bins: &[C], sums: &mut [f64]) {
    let mut i = 0;
    let two = _mm512_set1_pd(2.0);
    let real = _mm512_set_epi64(14, 12, 10, 8, 6, 4, 2, 0);
    let imag = _mm512_set_epi64(15, 13, 11, 9, 7, 5, 3, 1);
    while bins.len() - i >= 8 {
        // SAFETY: i..i+8 is inside both equal-length slices. Each load
        // covers four complex bins. No mask, overread, or padded input.
        unsafe {
            let ptr = bins.as_ptr().add(i).cast::<f64>();
            let a = _mm512_loadu_pd(ptr);
            let b = _mm512_loadu_pd(ptr.add(8));
            let r = _mm512_permutex2var_pd(a, real, b);
            let j = _mm512_permutex2var_pd(a, imag, b);
            let p = _mm512_add_pd(_mm512_mul_pd(r, r), _mm512_mul_pd(j, j));
            let dst = sums.as_mut_ptr().add(i);
            let next = _mm512_add_pd(_mm512_loadu_pd(dst), _mm512_mul_pd(two, p));
            _mm512_storeu_pd(dst, next);
        }
        i += 8;
    }
    tail(&bins[i..], &mut sums[i..]);
}

#[cfg(all(
    feature = "experimental-stereo-spectrum-split",
    any(test, feature = "experimental-spectral-avx2")
))]
#[target_feature(enable = "avx2")]
#[inline(never)]
unsafe fn bass_avx2(l: &[C], r: &[C], mid: &mut [f64], side: &mut [f64]) {
    super::spectral_stereo::bass(l, r, mid, side);
}

#[cfg(all(
    feature = "experimental-stereo-spectrum-split",
    any(test, feature = "experimental-spectral-avx512")
))]
#[target_feature(enable = "avx512f")]
#[inline(never)]
unsafe fn bass_avx512(l: &[C], r: &[C], mid: &mut [f64], side: &mut [f64]) {
    super::spectral_stereo::bass(l, r, mid, side);
}

#[cfg(all(
    feature = "experimental-stereo-spectrum-split",
    any(test, feature = "experimental-spectral-avx2")
))]
#[target_feature(enable = "avx2")]
#[inline(never)]
unsafe fn cross_avx2(l: &[C], r: &[C], dst: &mut [C]) {
    super::spectral_stereo::cross(l, r, dst);
}

#[cfg(all(
    feature = "experimental-stereo-spectrum-split",
    any(test, feature = "experimental-spectral-avx512")
))]
#[target_feature(enable = "avx512f")]
#[inline(never)]
unsafe fn cross_avx512(l: &[C], r: &[C], dst: &mut [C]) {
    super::spectral_stereo::cross(l, r, dst);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "experimental-stereo-spectrum-split")]
    #[test]
    fn wide_simd_stereo_split_bits() {
        for backend in [Backend::avx2(), Backend::avx512()].into_iter().flatten() {
            for n in [2, 3, 5, 9, 33, 4097, 65537] {
                let l: Vec<_> = (0..n)
                    .map(|i| C::new((i % 17) as f64 * 0.125, (i % 31) as f64 * -0.25))
                    .collect();
                let r: Vec<_> = l.iter().rev().copied().collect();
                let (mut m, mut s, mut c) = (vec![0.0; n], vec![0.0; n], vec![C::default(); n]);
                let (mut em, mut es, mut ec) = (m.clone(), s.clone(), c.clone());
                for _ in 0..3 {
                    backend.bass(&l, &r, &mut m, &mut s);
                    backend.cross(&l, &r, &mut c);
                    super::super::spectral_stereo::bass(&l, &r, &mut em, &mut es);
                    super::super::spectral_stereo::cross(&l, &r, &mut ec);
                }
                for (a, b) in m.iter().chain(&s).zip(em.iter().chain(&es)) {
                    assert_eq!(a.to_bits(), b.to_bits());
                }
                for (a, b) in c.iter().zip(ec) {
                    assert_eq!(a.re.to_bits(), b.re.to_bits());
                    assert_eq!(a.im.to_bits(), b.im.to_bits());
                }
            }
            eprintln!("STEREO_SPLIT_EXECUTED {}", backend.name());
        }
    }

    #[test]
    fn wide_simd_forced_power_bits() {
        let candidates = [Backend::avx2(), Backend::avx512()];
        let cases = [
            C::new(0.0, -0.0),
            C::new(0.125, -0.25),
            C::new(1e-155, -1e-160),
            C::new(-0.875, 0.5),
            C::new(1e-180, 1e-150),
            C::new(-3.0, 2.0),
            C::new(f64::MIN_POSITIVE, -f64::from_bits(1)),
            C::new(1e154, -1e154), // finite input, overflowing squared norm
        ];
        for (name, candidate) in ["avx2", "avx512f"].into_iter().zip(candidates) {
            let Some(backend) = candidate else {
                eprintln!("WIDE_SIMD_UNAVAILABLE {name}");
                continue;
            };
            let mut count = 0;
            let mut seed = 0x587a_4b27_91c3_511d_u64;
            for n in (2_usize..=34).chain([65, 257, 4097, 65537]) {
                for offset in [0, 1, 3, 7] {
                    let input: Vec<_> = (0..n + offset + 2)
                        .map(|i| {
                            seed ^= seed << 13;
                            seed ^= seed >> 7;
                            seed ^= seed << 17;
                            if i.is_multiple_of(3) {
                                cases[i % cases.len()]
                            } else {
                                C::new(
                                    (seed >> 11) as f64 / (1_u64 << 53) as f64 - 0.5,
                                    (seed & 0xffff) as f64 / 65536.0 - 0.5,
                                )
                            }
                        })
                        .collect();
                    for initial in [0.0, -0.0, 0.125, 1e308, f64::from_bits(1)] {
                        // The sentinels on both sides detect unintended writes.
                        let mut actual = vec![123.0; n + offset + 2];
                        actual[offset..offset + n].fill(initial);
                        let mut expected = actual.clone();
                        let bins = &input[offset..offset + n];
                        for _ in 0..3 {
                            backend.accumulate(bins, &mut actual[offset..offset + n]);
                            accumulate_power(bins, &mut expected[offset..offset + n]);
                        }
                        for (a, b) in actual.iter().zip(expected) {
                            assert_eq!(a.to_bits(), b.to_bits(), "{name}: n={n} offset={offset}");
                        }
                        count += 1;
                    }
                }
            }
            eprintln!("WIDE_SIMD_EXECUTED {name} cases={count}");
        }
    }

    #[test]
    #[allow(deprecated)] // x86 MXCSR intrinsics are used only in this test.
    fn wide_simd_subnormal_modes() {
        // Keep this state local to the test thread. Restore the exact prior
        // MXCSR even if a comparison panics; production never changes it.
        struct Restore(u32);
        impl Drop for Restore {
            fn drop(&mut self) {
                // SAFETY: this value was read from MXCSR on this thread.
                unsafe { _mm_setcsr(self.0) };
            }
        }
        let original = unsafe { _mm_getcsr() };
        let _restore = Restore(original);
        let input: Vec<_> = (0..33)
            .map(|i| match i % 5 {
                0 => C::new(1e-155, -1e-160),
                1 => C::new(f64::from_bits(1), -0.0),
                2 => C::new(1e-153, -1e-154),
                3 => C::new(0.0, -0.0),
                _ => C::new(-0.125, 0.5),
            })
            .collect();
        const DAZ_FTZ: u32 = (1 << 6) | (1 << 15);
        for mode in [original & !DAZ_FTZ, original | DAZ_FTZ] {
            // SAFETY: only the supported DAZ and FTZ bits on x86-64 are
            // changed. The restoration guard preserves all other settings.
            unsafe { _mm_setcsr(mode) };
            for backend in [Backend::avx2(), Backend::avx512()].into_iter().flatten() {
                let mut a = vec![f64::from_bits(1); input.len()];
                let mut b = a.clone();
                backend.accumulate(std::hint::black_box(&input), &mut a);
                accumulate_power(std::hint::black_box(&input), &mut b);
                for (a, b) in a.iter().zip(b) {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "{} MXCSR {mode:x}",
                        backend.name()
                    );
                }
            }
        }
    }

    #[test]
    #[ignore = "native paired kernel timings; run explicitly in release mode"]
    fn wide_simd_kernel_timing() {
        use std::{hint::black_box, time::Instant};
        let backends: Vec<_> = [
            Some(Backend::portable()),
            Backend::avx2(),
            Backend::avx512(),
        ]
        .into_iter()
        .flatten()
        .collect();
        for n in [4097_usize, 8193, 16385, 32769, 65537] {
            let input: Vec<_> = (0..n)
                .map(|i| {
                    C::new(
                        ((i * 7919 % 65521) as f64 - 32760.0) / 131072.0,
                        ((i * 1049 % 65519) as f64 - 32759.0) / 131072.0,
                    )
                })
                .collect();
            let iterations = 50_000_000 / n;
            let mut expected = vec![0.0; n];
            for _ in 0..iterations {
                accumulate_power(&input, &mut expected);
            }
            let mut power = vec![0.0; n];
            // All measured blocks share the same initialized input and
            // reduction count. Rotate order to reduce drift bias.
            for repeat in 0..5 {
                for k in 0..backends.len() {
                    let backend = backends[(k + repeat) % backends.len()];
                    power.fill(0.0);
                    for _ in 0..2 {
                        backend.accumulate(&input, &mut power);
                    }
                    power.fill(0.0);
                    let start = Instant::now();
                    for _ in 0..iterations {
                        backend.accumulate(black_box(&input), black_box(&mut power));
                    }
                    let ns = start.elapsed().as_nanos();
                    black_box(&power);
                    for (a, b) in power.iter().zip(&expected) {
                        assert_eq!(a.to_bits(), b.to_bits());
                    }
                    eprintln!(
                        "WIDE_KERNEL\t{}\t{n}\t{iterations}\t{repeat}\t{ns}",
                        backend.name()
                    );
                }
            }
        }
    }

    #[test]
    fn wide_simd_runtime_selection() {
        let selected = Backend::select();
        let expected = if cfg!(feature = "experimental-spectral-avx2") {
            Backend::avx2()
        } else {
            None
        }
        .or_else(|| {
            if cfg!(feature = "experimental-spectral-avx512") {
                Backend::avx512()
            } else {
                None
            }
        })
        .unwrap_or_else(Backend::portable);
        assert_eq!(selected, expected);
        eprintln!("WIDE_SIMD_SELECTED {}", selected.name());
    }

    // Exercise actual streaming construction and the power-only path, not
    // just a forced kernel. The engine's ordinary DFT tests also run below.
    #[cfg(any(
        feature = "experimental-spectral-avx2",
        feature = "experimental-spectral-avx512"
    ))]
    #[test]
    fn wide_simd_streaming_power_bits() {
        use super::super::SpectralEngine;
        for (rate, channels) in [(8001, 1), (44101, 2), (48000, 3), (768000, 6)] {
            let make = || SpectralEngine::new(rate, channels, true, false, false, true, true, true);
            let mut selected = make();
            let mut portable = make();
            portable.power_backend = Backend::portable();
            assert_eq!(selected.power_backend, Backend::select());
            // Include leading and trailing padding and more than two complete windows.
            let frames = selected.n * 2 + 17;
            let pcm: Vec<_> = (0..frames * channels)
                .map(|i| ((i.wrapping_mul(7919) % 65521) as f64 - 32760.0) / 131072.0)
                .collect();
            for block in pcm.chunks(4093 * channels) {
                selected.push(block);
                portable.push(block);
            }
            selected.flush();
            portable.flush();
            assert_eq!(selected.spectrum.frames, portable.spectrum.frames);
            for (a, b) in selected.spectrum.sums.iter().zip(&portable.spectrum.sums) {
                assert_eq!(a.to_bits(), b.to_bits(), "{rate}/{channels}");
            }
            let a = selected.interior.as_ref().unwrap();
            let b = portable.interior.as_ref().unwrap();
            assert_eq!(a.frames, b.frames);
            for (a, b) in a.sums.iter().zip(&b.sums) {
                assert_eq!(a.to_bits(), b.to_bits());
            }
            assert_eq!(
                selected.bandwidth().to_bits(),
                portable.bandwidth().to_bits()
            );
            assert_eq!(selected.lossy_found(), portable.lossy_found());
        }
        eprintln!(
            "WIDE_SIMD_STREAM_EXECUTED {} geometries=4",
            Backend::select().name()
        );
    }
}
