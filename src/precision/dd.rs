// SPDX-License-Identifier: GPL-2.0-or-later
//! Double-double arithmetic for normalized, finite stereo central moments.
//! This is not a general-purpose extended-exponent floating-point type.

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Dd {
    pub(super) hi: f64,
    pub(super) lo: f64,
}

impl Dd {
    pub(super) const fn from(x: f64) -> Self {
        Self { hi: x, lo: 0.0 }
    }

    // Knuth TwoSum; no magnitude-order precondition.
    fn sum(a: f64, b: f64) -> Self {
        let hi = a + b;
        let v = hi - a;
        let lo = (a - (hi - v)) + (b - v);
        Self { hi, lo }
    }

    pub(super) fn add(self, b: Self) -> Self {
        let s = Self::sum(self.hi, b.hi);
        let t = Self::sum(self.lo, b.lo);
        let u = Self::sum(s.hi, s.lo + t.hi);
        Self::sum(u.hi, u.lo + t.lo)
    }

    pub(super) fn sub(self, b: Self) -> Self {
        self.add(Self {
            hi: -b.hi,
            lo: -b.lo,
        })
    }

    pub(super) fn mul(self, b: Self) -> Self {
        let p = self.hi * b.hi;
        // Error-free product where the residual is representable. Explicit
        // FMA, not contraction of an unrelated multiply-add expression.
        let e = self.hi.mul_add(b.hi, -p);
        let cross = self.hi.mul_add(b.lo, self.lo * b.hi);
        Self::sum(p, e + (cross + self.lo * b.lo))
    }

    pub(super) fn div(self, b: Self) -> Self {
        let q = Self::from(self.hi / b.hi);
        let rem = self.sub(b.mul(q));
        let q = q.add(Self::from(rem.hi / b.hi));
        let rem = self.sub(b.mul(q));
        q.add(Self::from(rem.hi / b.hi))
    }

    pub(super) fn sqrt(self) -> Self {
        let q = Self::from(self.hi.sqrt());
        let correction = self.sub(q.mul(q)).div(q.add(q));
        q.add(correction)
    }

    pub(super) fn scale(self, exponent: i32) -> Self {
        Self::sum(scale(self.hi, exponent), scale(self.lo, exponent))
    }
}

// Chunk the exponent to avoid constructing an overflowing or zero scale
// factor. Upscaling the smallest binary64 subnormal by 1074 is exact.
pub(super) fn scale(mut x: f64, mut exponent: i32) -> f64 {
    while exponent > 512 {
        x *= f64::from_bits(1535_u64 << 52);
        exponent -= 512;
    }
    while exponent < -512 {
        x *= f64::from_bits(511_u64 << 52);
        exponent += 512;
    }
    x * f64::from_bits(((exponent + 1023) as u64) << 52)
}

// True binary exponent, including subnormals. Zero has no exponent.
fn exponent(x: f64) -> Option<i32> {
    let bits = x.to_bits() & 0x7fff_ffff_ffff_ffff;
    let encoded = (bits >> 52) as i32;
    if encoded > 0 {
        Some(encoded - 1023)
    } else if bits == 0 {
        None
    } else {
        Some(63 - bits.leading_zeros() as i32 - 1074)
    }
}

// A fixed block has its own local origin. Across blocks merge centered
// moments using the parallel variance identity, not raw square subtraction.
// Each channel uses its own growing power-of-two normalization. Thus even
// an all-subnormal channel has usable variances, independently of the other
// channel's amplitude. Terms too small relative to the greatest sample in
// their own channel can underflow; DD is not an extended-exponent format.
#[derive(Default)]
pub(super) struct Moments {
    exp: [Option<i32>; 2],
    origin: [f64; 2],
    mean: [Dd; 2],
    variance: [Dd; 2],
    cov: Dd,
}

impl Moments {
    pub(super) fn push(&mut self, pcm: &[f64], prior: u64) {
        let frames = pcm.as_chunks::<2>().0;
        let n = frames.len() as u64;
        let mut maxima = [0.0_f64; 2];
        for f in frames {
            maxima[0] = maxima[0].max(f[0].abs());
            maxima[1] = maxima[1].max(f[1].abs());
        }
        let mut shifts = [0; 2];
        for (ch, shift) in shifts.iter_mut().enumerate() {
            let next = self.exp[ch].max(exponent(maxima[ch]));
            if let (Some(old), Some(new)) = (self.exp[ch], next) {
                *shift = old - new;
            }
            self.exp[ch] = next;
            if *shift != 0 {
                self.mean[ch] = self.mean[ch].scale(*shift);
                self.variance[ch] = self.variance[ch].scale(2 * *shift);
            }
        }
        if shifts[0] != 0 || shifts[1] != 0 {
            self.cov = self.cov.scale(shifts[0] + shifts[1]);
        }
        let ex = [self.exp[0].unwrap_or(0), self.exp[1].unwrap_or(0)];
        let local = [scale(pcm[0], -ex[0]), scale(pcm[1], -ex[1])];
        let offsets = |f: &[f64; 2]| {
            [
                Dd::from(scale(f[0], -ex[0])).sub(Dd::from(local[0])),
                Dd::from(scale(f[1], -ex[1])).sub(Dd::from(local[1])),
            ]
        };
        let mut mean = [Dd::default(); 2];
        for f in frames {
            let d = offsets(f);
            mean[0] = mean[0].add(d[0]);
            mean[1] = mean[1].add(d[1]);
        }
        let dn = Dd::from(n as f64);
        mean[0] = mean[0].div(dn);
        mean[1] = mean[1].div(dn);
        let mut var = [Dd::default(); 2];
        let mut cov = Dd::default();
        for f in frames {
            let d = offsets(f);
            let l = d[0].sub(mean[0]);
            let r = d[1].sub(mean[1]);
            var[0] = var[0].add(l.mul(l));
            var[1] = var[1].add(r.mul(r));
            cov = cov.add(l.mul(r));
        }
        if prior == 0 {
            self.origin = [pcm[0], pcm[1]];
            self.mean = mean;
            self.variance = var;
            self.cov = cov;
            return;
        }
        let delta = [0, 1].map(|ch| {
            Dd::from(local[ch])
                .sub(Dd::from(scale(self.origin[ch], -ex[ch])))
                .add(mean[ch])
                .sub(self.mean[ch])
        });
        let weight = dn.div(Dd::from((prior + n) as f64));
        let factor = Dd::from(prior as f64).mul(weight);
        for (ch, d) in delta.into_iter().enumerate() {
            self.mean[ch] = self.mean[ch].add(d.mul(weight));
            self.variance[ch] = self.variance[ch].add(var[ch].add(d.mul(d).mul(factor)));
        }
        self.cov = self.cov.add(cov.add(delta[0].mul(delta[1]).mul(factor)));
    }

    pub(super) fn finish(&self) -> Dd {
        if self.variance[0].hi <= 0.0 || self.variance[1].hi <= 0.0 {
            return Dd::from(0.0);
        }
        let norm = self.variance[0].sqrt().mul(self.variance[1].sqrt());
        let r = self.cov.div(norm);
        if r.hi > 1.0 || (r.hi == 1.0 && r.lo > 0.0) {
            Dd::from(1.0)
        } else if r.hi < -1.0 || (r.hi == -1.0 && r.lo < 0.0) {
            Dd::from(-1.0)
        } else {
            r
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_power_of_two_scaling() {
        let tiny = f64::from_bits(1);
        assert_eq!(exponent(tiny), Some(-1074));
        assert_eq!(scale(tiny, 1074), 1.0);
        assert_eq!(scale(1.0, -1074), tiny);
        assert_eq!(exponent(f64::MIN_POSITIVE), Some(-1022));
        assert_eq!(exponent(-16.0), Some(4));
        assert_eq!(exponent(-0.0), None);
    }

    #[test]
    fn fma_product_and_residual() {
        let eps = 2.0_f64.powi(-40);
        let p = Dd::from(1.0 + eps).mul(Dd::from(1.0 - eps));
        assert_eq!(p.hi, 1.0);
        assert_eq!(p.lo, -eps * eps);
        let residual = p.sub(Dd::from(1.0));
        assert_eq!(residual.hi, -eps * eps);
        assert_eq!(residual.lo, 0.0);
    }
}
