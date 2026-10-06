//! Minimal complex scalar type (`C64`).
//!
//! Eigenvalues of real systems are generally complex, and frequency response
//! lives on the imaginary axis, so the core owns a small complex type instead
//! of pulling in an external crate.

use core::fmt;
use core::ops::{Add, Div, Mul, Neg, Sub};

use crate::math;

/// A complex number stored as `(re, im)` pair of `f64`s.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct C64 {
    /// Real part.
    pub re: f64,
    /// Imaginary part.
    pub im: f64,
}

impl C64 {
    /// The additive identity.
    pub const ZERO: C64 = C64 { re: 0.0, im: 0.0 };

    /// Constructs a complex number from real and imaginary parts.
    pub const fn new(re: f64, im: f64) -> C64 {
        C64 { re, im }
    }

    /// Constructs a purely real complex number.
    pub const fn real(re: f64) -> C64 {
        C64 { re, im: 0.0 }
    }

    /// Constructs a purely imaginary complex number.
    pub const fn imaginary(im: f64) -> C64 {
        C64 { re: 0.0, im }
    }

    /// Complex conjugate.
    pub fn conj(self) -> C64 {
        C64 {
            re: self.re,
            im: -self.im,
        }
    }

    /// Modulus `sqrt(re^2 + im^2)` (overflow-safe).
    pub fn abs(self) -> f64 {
        math::hypot(self.re, self.im)
    }

    /// Squared modulus.
    pub fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    /// Argument (phase angle) in radians, in `(-pi, pi]`.
    pub fn arg(self) -> f64 {
        math::atan2(self.im, self.re)
    }

    /// Scales both parts by a real factor.
    pub fn scale(self, f: f64) -> C64 {
        C64 {
            re: self.re * f,
            im: self.im * f,
        }
    }

    /// Principal square root.
    pub fn sqrt(self) -> C64 {
        if self.re == 0.0 && self.im == 0.0 {
            return C64::ZERO;
        }
        let a = self.abs();
        let re = math::sqrt((a + self.re) * 0.5);
        let sign = if self.im.is_sign_negative() {
            -1.0
        } else {
            1.0
        };
        let im = sign * math::sqrt((a - self.re) * 0.5);
        C64 { re, im }
    }

    /// Complex exponential `e^z`.
    pub fn exp(self) -> C64 {
        let e = math::exp(self.re);
        C64 {
            re: e * math::cos(self.im),
            im: e * math::sin(self.im),
        }
    }
}

impl From<f64> for C64 {
    fn from(re: f64) -> C64 {
        C64 { re, im: 0.0 }
    }
}

impl Add for C64 {
    type Output = C64;
    fn add(self, rhs: C64) -> C64 {
        C64 {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl Sub for C64 {
    type Output = C64;
    fn sub(self, rhs: C64) -> C64 {
        C64 {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl Neg for C64 {
    type Output = C64;
    fn neg(self) -> C64 {
        C64 {
            re: -self.re,
            im: -self.im,
        }
    }
}

impl Mul for C64 {
    type Output = C64;
    fn mul(self, rhs: C64) -> C64 {
        C64 {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl Mul<f64> for C64 {
    type Output = C64;
    fn mul(self, f: f64) -> C64 {
        self.scale(f)
    }
}

impl Div for C64 {
    type Output = C64;
    fn div(self, rhs: C64) -> C64 {
        // Smith's algorithm: scale by the larger part to avoid overflow.
        if rhs.re.abs() >= rhs.im.abs() {
            let r = rhs.im / rhs.re;
            let den = rhs.re + rhs.im * r;
            C64 {
                re: (self.re + self.im * r) / den,
                im: (self.im - self.re * r) / den,
            }
        } else {
            let r = rhs.re / rhs.im;
            let den = rhs.re * r + rhs.im;
            C64 {
                re: (self.re * r + self.im) / den,
                im: (self.im * r - self.re) / den,
            }
        }
    }
}

impl fmt::Display for C64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.im < 0.0 {
            write!(f, "{}-{}i", self.re, -self.im)
        } else {
            write!(f, "{}+{}i", self.re, self.im)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: C64, b: C64) -> bool {
        (a - b).abs() < 1e-12
    }

    #[test]
    fn arithmetic_basics() {
        let a = C64::new(1.0, 2.0);
        let b = C64::new(-3.0, 0.5);
        assert!(close(a * b, C64::new(-4.0, -5.5)));
        assert!(close(a / a, C64::new(1.0, 0.0)));
        assert!(close(a + b, C64::new(-2.0, 2.5)));
        assert!(close(-a, C64::new(-1.0, -2.0)));
    }

    #[test]
    fn sqrt_principal_branch() {
        assert!(close(C64::new(-1.0, 0.0).sqrt(), C64::new(0.0, 1.0)));
        assert!(close(C64::new(-1.0, -0.0).sqrt(), C64::new(0.0, -1.0)));
        assert!(close(C64::new(4.0, 0.0).sqrt(), C64::new(2.0, 0.0)));
        // (3+4i)^2 = -7+24i
        assert!(close(C64::new(-7.0, 24.0).sqrt(), C64::new(3.0, 4.0)));
    }

    #[test]
    fn exp_matches_euler() {
        let z = C64::new(0.0, core::f64::consts::PI);
        assert!(close(z.exp(), C64::new(-1.0, 0.0)));
    }
}
