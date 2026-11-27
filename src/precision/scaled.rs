// SPDX-License-Identifier: GPL-2.0-or-later
//! Exponent-scaled wide linear powers.
use super::dd;

/// A nonnegative linear bin power represented by two binary64 limbs and
/// an independent binary exponent: `(hi + lo) * 2^exponent`.
///
/// The exponent preserves outputs below the ordinary binary64 power range.
/// The low limb is part of the measured number, not an uncertainty bound.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PowerBin {
    /// Rounded normalized high component (zero, or approximately [0.5, 1]).
    pub hi: f64,
    /// Signed residual component at the selected arithmetic precision.
    pub lo: f64,
    /// Shared exponent of the two limbs; zero power has exponent zero.
    pub exponent: i32,
}

impl PowerBin {
    /// Ordinary rounded linear-power view; very small powers may underflow.
    pub fn as_f64(self) -> f64 {
        let mantissa = self.hi + self.lo;
        if mantissa == 0.0 || self.exponent < -1075 {
            0.0
        } else if self.exponent > 1024 {
            f64::INFINITY
        } else {
            dd::scale(mantissa, self.exponent)
        }
    }

    /// Display level in dBFS per bin. A zero-power bin yields negative infinity.
    /// Logarithmic presentation itself uses binary64, not wider arithmetic.
    pub fn dbfs(self) -> f64 {
        10.0 * (self.hi + self.lo).log10()
            + 10.0 * f64::from(self.exponent) * std::f64::consts::LOG10_2
    }
}
