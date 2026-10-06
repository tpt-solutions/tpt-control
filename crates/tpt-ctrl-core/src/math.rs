//! Floating-point kernels dispatched to `std` or `libm`.
//!
//! In `std` builds the inherent (hardware/LIBC-backed) methods are used; in
//! `no_std` + `alloc` builds the pure-Rust `libm` implementations take over.
//! This keeps the numeric code identical on both sides of the feature split.

#![allow(dead_code)]

pub(crate) fn ceil(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        f64::ceil(x)
    }
    #[cfg(not(feature = "std"))]
    {
        libm::ceil(x)
    }
}

pub(crate) fn sqrt(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        f64::sqrt(x)
    }
    #[cfg(not(feature = "std"))]
    {
        libm::sqrt(x)
    }
}

pub(crate) fn sqrtf(x: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        f32::sqrt(x)
    }
    #[cfg(not(feature = "std"))]
    {
        libm::sqrtf(x)
    }
}

pub(crate) fn exp(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        f64::exp(x)
    }
    #[cfg(not(feature = "std"))]
    {
        libm::exp(x)
    }
}

pub(crate) fn expf(x: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        f32::exp(x)
    }
    #[cfg(not(feature = "std"))]
    {
        libm::expf(x)
    }
}

pub(crate) fn ln(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        f64::ln(x)
    }
    #[cfg(not(feature = "std"))]
    {
        libm::log(x)
    }
}

pub(crate) fn lnf(x: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        f32::ln(x)
    }
    #[cfg(not(feature = "std"))]
    {
        libm::logf(x)
    }
}

pub(crate) fn hypot(x: f64, y: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        f64::hypot(x, y)
    }
    #[cfg(not(feature = "std"))]
    {
        libm::hypot(x, y)
    }
}

pub(crate) fn hypotf(x: f32, y: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        f32::hypot(x, y)
    }
    #[cfg(not(feature = "std"))]
    {
        libm::hypotf(x, y)
    }
}

pub(crate) fn sin(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        f64::sin(x)
    }
    #[cfg(not(feature = "std"))]
    {
        libm::sin(x)
    }
}

pub(crate) fn sinf(x: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        f32::sin(x)
    }
    #[cfg(not(feature = "std"))]
    {
        libm::sinf(x)
    }
}

pub(crate) fn cos(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        f64::cos(x)
    }
    #[cfg(not(feature = "std"))]
    {
        libm::cos(x)
    }
}

pub(crate) fn cosf(x: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        f32::cos(x)
    }
    #[cfg(not(feature = "std"))]
    {
        libm::cosf(x)
    }
}

pub(crate) fn atan2(y: f64, x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        f64::atan2(y, x)
    }
    #[cfg(not(feature = "std"))]
    {
        libm::atan2(y, x)
    }
}

pub(crate) fn atan2f(y: f32, x: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        f32::atan2(y, x)
    }
    #[cfg(not(feature = "std"))]
    {
        libm::atan2f(y, x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shims_match_expected_values() {
        assert!((sqrt(2.0) - core::f64::consts::SQRT_2).abs() < 1e-15);
        assert!((exp(1.0) - core::f64::consts::E).abs() < 1e-15);
        assert!((hypot(3.0, 4.0) - 5.0).abs() < 1e-15);
        assert!((ln(core::f64::consts::E) - 1.0).abs() < 1e-15);
    }
}
