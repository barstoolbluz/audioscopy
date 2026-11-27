// SPDX-License-Identifier: GPL-2.0-or-later
//! Optional same-graph combined stereo-spectrum postprocessing candidate.
//! Independent bin operations can vectorize; serial energy reductions keep
//! their exact original order. No new sample, FFT, or estimator definition.
use super::C;
use std::ops::Range;

// Binary search on the original rounded frequency predicate, not division
// and rounding of frequency limits. Preserve inclusion at exact boundaries.
fn first_false(n: usize, before: impl Fn(usize) -> bool) -> usize {
    let (mut lo, mut hi) = (0, n);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if before(mid) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

pub(super) fn delay_band(rate: u32, n: usize) -> Range<usize> {
    let df = f64::from(rate) / n as f64;
    let upper = 10_000.0_f64.min(f64::from(rate) * 0.45);
    let bins = n / 2 + 1;
    let start = first_false(bins, |i| (i as f64) * df < 2000.0);
    let end = first_false(bins, |i| (i as f64) * df <= upper);
    start.min(end)..end
}

#[inline(always)]
pub(super) fn bass(l: &[C], r: &[C], mid: &mut [f64], side: &mut [f64]) {
    debug_assert!(l.len() >= 2);
    debug_assert_eq!(l.len(), r.len());
    debug_assert_eq!(l.len(), mid.len());
    debug_assert_eq!(l.len(), side.len());
    let last = l.len() - 1;
    mid[0] += (l[0] + r[0]).norm_sqr();
    side[0] += (l[0] - r[0]).norm_sqr();
    for (((m, s), a), b) in mid[1..last]
        .iter_mut()
        .zip(&mut side[1..last])
        .zip(&l[1..last])
        .zip(&r[1..last])
    {
        *m += 2.0 * (*a + *b).norm_sqr();
        *s += 2.0 * (*a - *b).norm_sqr();
    }
    mid[last] += (l[last] + r[last]).norm_sqr();
    side[last] += (l[last] - r[last]).norm_sqr();
}

#[inline(always)]
pub(super) fn cross(l: &[C], r: &[C], dst: &mut [C]) {
    debug_assert_eq!(l.len(), r.len());
    debug_assert_eq!(l.len(), dst.len());
    for ((sum, a), b) in dst.iter_mut().zip(l).zip(r) {
        *sum += a.conj() * *b;
    }
}

pub(super) fn delay_energy(
    l: &[C],
    r: &[C],
    band: Range<usize>,
    total: &mut [f64; 2],
    le: &mut f64,
    re: &mut f64,
) {
    debug_assert_eq!(l.len(), r.len());
    let last = l.len() - 1;
    let (mut a, mut b) = (total[0], total[1]);
    for (i, (x, y)) in l.iter().zip(r).enumerate() {
        let factor = if i == 0 || i == last { 1.0 } else { 2.0 };
        a += factor * x.norm_sqr();
        b += factor * y.norm_sqr();
    }
    *total = [a, b];
    // The admitted delay band excludes DC and Nyquist. The factor for all
    // included bins is therefore two, as in the old combined loop.
    debug_assert!(band.is_empty() || (band.start > 0 && band.end <= last));
    for (x, y) in l[band.clone()].iter().zip(&r[band]) {
        *le += 2.0 * x.norm_sqr();
        *re += 2.0 * y.norm_sqr();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_stereo_same_graph_bits() {
        for rate in [8000, 8001, 44100, 44101, 48000, 192000, 768000] {
            for n in [16, 64, 8192, 32768, 131072] {
                let band = delay_band(rate, n);
                let df = f64::from(rate) / n as f64;
                let upper = 10_000.0_f64.min(f64::from(rate) * 0.45);
                for i in 0..=n / 2 {
                    let f = i as f64 * df;
                    assert_eq!(band.contains(&i), f >= 2000.0 && f <= upper);
                }
                let len = n / 2 + 1;
                let l: Vec<_> = (0..len)
                    .map(|i| match i % 7 {
                        0 => C::new(0.0, -0.0),
                        1 => C::new(1e-155, -1e-160),
                        _ => C::new((i % 13) as f64 * 0.125, (i % 17) as f64 * -0.25),
                    })
                    .collect();
                let r: Vec<_> = l.iter().rev().copied().collect();
                let (mut m, mut s, mut c) =
                    (vec![0.0; len], vec![0.0; len], vec![C::default(); len]);
                let (mut em, mut es, mut ec) = (m.clone(), s.clone(), c.clone());
                let (mut t, mut el, mut er) = ([0.0; 2], 0.0, 0.0);
                let (mut et, mut bl, mut br) = (t, el, er);
                for _ in 0..3 {
                    bass(&l, &r, &mut m, &mut s);
                    cross(&l[band.clone()], &r[band.clone()], &mut c[band.clone()]);
                    delay_energy(&l, &r, band.clone(), &mut t, &mut el, &mut er);
                    for i in 0..len {
                        let factor = if i == 0 || i == len - 1 { 1.0 } else { 2.0 };
                        em[i] += factor * (l[i] + r[i]).norm_sqr();
                        es[i] += factor * (l[i] - r[i]).norm_sqr();
                        et[0] += factor * l[i].norm_sqr();
                        et[1] += factor * r[i].norm_sqr();
                        if band.contains(&i) {
                            ec[i] += l[i].conj() * r[i];
                            bl += factor * l[i].norm_sqr();
                            br += factor * r[i].norm_sqr();
                        }
                    }
                }
                for (a, b) in m.iter().chain(&s).zip(em.iter().chain(&es)) {
                    assert_eq!(a.to_bits(), b.to_bits());
                }
                for (a, b) in c.iter().zip(ec) {
                    assert_eq!(a.re.to_bits(), b.re.to_bits());
                    assert_eq!(a.im.to_bits(), b.im.to_bits());
                }
                assert_eq!(t.map(f64::to_bits), et.map(f64::to_bits));
                assert_eq!(el.to_bits(), bl.to_bits());
                assert_eq!(er.to_bits(), br.to_bits());
            }
        }
        eprintln!("STEREO_SPLIT_PORTABLE_EXECUTED 35 geometries");
    }
}
