//! Lead / lag compensator constructors (unity DC gain).

use crate::tf::{TfError, TransferFunction};

/// Phase-lead compensator `G(s) = (1 + T s) / (1 + alpha T s)` with
/// `0 < alpha < 1` (unity DC gain, zero at `-1/T`, pole at `-1/(alpha T)`).
///
/// Maximum phase lead `phi_max` occurs at `omega_max = 1 / (T sqrt(alpha))`
/// with `sin(phi_max) = (1 - alpha) / (1 + alpha)`.
pub fn lead(t: f64, alpha: f64) -> Result<TransferFunction, TfError> {
    if t <= 0.0 || !(0.0 < alpha && alpha < 1.0) {
        return Err(TfError::InvalidDenominator);
    }
    TransferFunction::new(&[t, 1.0], &[alpha * t, 1.0])
}

/// Phase-lag compensator `G(s) = (1 + T s) / (1 + beta T s)` with
/// `beta > 1` (unity DC gain, attenuation at high frequency by `1/beta`).
pub fn lag(t: f64, beta: f64) -> Result<TransferFunction, TfError> {
    if t <= 0.0 || beta <= 1.0 {
        return Err(TfError::InvalidDenominator);
    }
    TransferFunction::new(&[t, 1.0], &[beta * t, 1.0])
}

/// Notch (band-reject) section `G(s) = (s^2 + 2 zeta0 wz s + wz^2) /
/// (s^2 + 2 zeta1 wz s + wz^2)` — unity DC and unity high-frequency gain.
pub fn notch(wz: f64, zeta_zero: f64, zeta_pole: f64) -> Result<TransferFunction, TfError> {
    if wz <= 0.0 || zeta_zero <= 0.0 || zeta_pole <= zeta_zero {
        return Err(TfError::InvalidDenominator);
    }
    TransferFunction::new(
        &[1.0, 2.0 * zeta_zero * wz, wz * wz],
        &[1.0, 2.0 * zeta_pole * wz, wz * wz],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_ctrl_verify::testing::assert_close;

    #[test]
    fn lead_dc_gain_and_zero() {
        let g = lead(0.1, 0.1).unwrap();
        assert_close(g.dc_gain(), 1.0, 1e-12);
        let zeros = g.zeros();
        assert_eq!(zeros.len(), 1);
        assert_close(zeros[0].re, -10.0, 1e-9);
        assert!(zeros[0].im.abs() < 1e-9);
    }

    #[test]
    fn lag_attenuates_high_frequency() {
        let g = lag(1.0, 100.0).unwrap();
        let hf = g.eval_jw(1e4);
        assert!(hf.abs() < 1.05 / 100.0 && hf.abs() > 0.95 / 100.0);
        assert_close(g.dc_gain(), 1.0, 1e-12);
    }

    #[test]
    fn notch_rejects_center_frequency() {
        let g = notch(10.0, 0.05, 0.5).unwrap();
        let center = g.eval_jw(10.0);
        assert!(center.abs() < 0.12, "notch depth {}", center.abs());
        assert_close(g.eval_jw(0.0).abs(), 1.0, 1e-12);
        assert_close(g.eval_jw(1e5).abs(), 1.0, 1e-3);
    }

    #[test]
    fn rejects_bad_parameters() {
        assert!(lead(0.1, 1.5).is_err());
        assert!(lag(0.1, 0.5).is_err());
        assert!(notch(-1.0, 0.1, 0.5).is_err());
    }
}
