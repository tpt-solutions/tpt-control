//! Scalar abstraction over the floating-point types used by the kernels.

use core::fmt::Debug;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::math;

/// A scalar type usable by the linear-algebra kernels.
///
/// Implemented for `f32` and `f64`. The trait bundles the arithmetic ops the
/// kernels need plus the elementary functions, so every kernel is generic
/// over precision without depending on `std`.
pub trait Field:
    Copy
    + Default
    + PartialOrd
    + Debug
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
{
    /// Additive identity.
    const ZERO: Self;
    /// Multiplicative identity.
    const ONE: Self;

    /// Builds a value from an `f64` constant (lossy for `f32`).
    fn from_f64(v: f64) -> Self;
    /// Widens to `f64` (used for tolerances and diagnostics).
    fn to_f64(self) -> f64;

    /// Additive identity as a function.
    fn zero() -> Self {
        Self::ZERO
    }
    /// Multiplicative identity as a function.
    fn one() -> Self {
        Self::ONE
    }

    /// Absolute value.
    fn abs(self) -> Self {
        if self < Self::ZERO {
            -self
        } else {
            self
        }
    }
    /// Principal square root.
    fn sqrt(self) -> Self;
    /// Natural exponential.
    fn exp(self) -> Self;
    /// Natural logarithm.
    fn ln(self) -> Self;
    /// Integer power by repeated squaring (handles negative exponents).
    fn powi(self, mut n: i32) -> Self {
        if n < 0 {
            let inv = Self::ONE / self;
            n = -n;
            let mut acc = Self::ONE;
            let mut base = inv;
            while n > 0 {
                if n & 1 == 1 {
                    acc *= base;
                }
                base *= base;
                n >>= 1;
            }
            acc
        } else {
            let mut acc = Self::ONE;
            let mut base = self;
            while n > 0 {
                if n & 1 == 1 {
                    acc *= base;
                }
                base *= base;
                n >>= 1;
            }
            acc
        }
    }
    /// `sqrt(a^2 + b^2)` without spurious overflow/underflow.
    fn hypot(self, other: Self) -> Self;
    /// Sine.
    fn sin(self) -> Self;
    /// Cosine.
    fn cos(self) -> Self;
    /// Four-quadrant arctangent of `y, x`.
    fn atan2(self, other: Self) -> Self;
    /// `true` if the value is neither infinite nor NaN.
    fn is_finite(self) -> bool;
}

macro_rules! impl_field {
    ($ty:ty, $zero:expr, $one:expr, $sqrt:path, $exp:path, $ln:path, $hypot:path, $sin:path, $cos:path, $atan2:path) => {
        impl Field for $ty {
            const ZERO: Self = $zero;
            const ONE: Self = $one;

            fn from_f64(v: f64) -> Self {
                v as $ty
            }
            fn to_f64(self) -> f64 {
                self as f64
            }
            fn sqrt(self) -> Self {
                $sqrt(self)
            }
            fn exp(self) -> Self {
                $exp(self)
            }
            fn ln(self) -> Self {
                $ln(self)
            }
            fn hypot(self, other: Self) -> Self {
                $hypot(self, other)
            }
            fn sin(self) -> Self {
                $sin(self)
            }
            fn cos(self) -> Self {
                $cos(self)
            }
            fn atan2(self, other: Self) -> Self {
                $atan2(self, other)
            }
            fn is_finite(self) -> bool {
                <$ty>::is_finite(self)
            }
        }
    };
}

impl_field!(
    f64,
    0.0,
    1.0,
    math::sqrt,
    math::exp,
    math::ln,
    math::hypot,
    math::sin,
    math::cos,
    math::atan2
);
impl_field!(
    f32,
    0.0,
    1.0,
    math::sqrtf,
    math::expf,
    math::lnf,
    math::hypotf,
    math::sinf,
    math::cosf,
    math::atan2f
);

#[cfg(test)]
mod tests {
    #[test]
    fn powi_handles_negative_exponents() {
        assert_eq!(2.0_f64.powi(-3), 0.125);
        assert_eq!(2.0_f64.powi(10), 1024.0);
        assert_eq!(2.0_f64.powi(0), 1.0);
    }

    #[test]
    fn abs_is_branch_only() {
        assert_eq!((-3.5_f64).abs(), 3.5);
        assert_eq!(3.5_f64.abs(), 3.5);
    }
}
