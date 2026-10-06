//! Frequency response, Bode/Nyquist data, and stability margins.
//!
//! Margin computation scans a dense logarithmic frequency grid, unwraps
//! the phase, and interpolates the gain/phase crossovers — accurate for
//! the open-loop shapes used in classical loop design.

use std::vec::Vec;

use tpt_ctrl_core::c64::C64;

use crate::tf::TransferFunction;

/// Bode data: magnitude in dB and unwrapped phase in degrees per frequency.
#[derive(Debug, Clone, PartialEq)]
pub struct BodeData {
    /// Frequencies `[rad/s]` (the input grid, ascending).
    pub omega: Vec<f64>,
    /// `20 log10 |H(j omega)|`.
    pub magnitude_db: Vec<f64>,
    /// Continuous (unwrapped) phase in degrees.
    pub phase_deg: Vec<f64>,
}

/// Nyquist-plot data: `Re H(j omega)`, `Im H(j omega)`.
#[derive(Debug, Clone, PartialEq)]
pub struct NyquistData {
    /// Frequencies `[rad/s]`.
    pub omega: Vec<f64>,
    /// Real part of `H(j omega)`.
    pub re: Vec<f64>,
    /// Imaginary part of `H(j omega)`.
    pub im: Vec<f64>,
}

/// Gain and phase stability margins.
#[derive(Debug, Clone, PartialEq)]
pub struct Margins {
    /// Gain margin (linear factor; `f64::INFINITY` when the phase never
    /// crosses -180 degrees).
    pub gain_margin: f64,
    /// Phase margin in degrees (`180` when the gain never crosses unity).
    pub phase_margin_deg: f64,
    /// Phase-crossover frequency (where phase = -180 deg), if finite.
    pub gain_crossover_omega: Option<f64>,
    /// Gain-crossover frequency (where |H| = 1), if finite.
    pub phase_crossover_omega: Option<f64>,
}

/// Evaluates the frequency response on the given (ascending) grid.
pub fn frequency_response(tf: &TransferFunction, omega: &[f64]) -> Vec<C64> {
    omega.iter().map(|&w| tf.eval_jw(w)).collect()
}

/// Bode magnitudes (dB) and unwrapped phases (deg) on the given grid.
pub fn bode_data(tf: &TransferFunction, omega: &[f64]) -> BodeData {
    let mut magnitude_db = Vec::with_capacity(omega.len());
    let mut phase_deg = Vec::with_capacity(omega.len());
    let mut prev = None;
    for &w in omega {
        let h = tf.eval_jw(w);
        magnitude_db.push(20.0 * h.abs().log10());
        let mut deg = h.arg() * 180.0 / core::f64::consts::PI;
        if let Some(p) = prev {
            // Unwrap: move this sample within 180 degrees of the previous.
            while deg - p > 180.0 {
                deg -= 360.0;
            }
            while deg - p < -180.0 {
                deg += 360.0;
            }
        }
        prev = Some(deg);
        phase_deg.push(deg);
    }
    BodeData {
        omega: omega.to_vec(),
        magnitude_db,
        phase_deg,
    }
}

/// Nyquist (polar) data on the given grid.
pub fn nyquist_data(tf: &TransferFunction, omega: &[f64]) -> NyquistData {
    let mut re = Vec::with_capacity(omega.len());
    let mut im = Vec::with_capacity(omega.len());
    for &w in omega {
        let h = tf.eval_jw(w);
        re.push(h.re);
        im.push(h.im);
    }
    NyquistData {
        omega: omega.to_vec(),
        re,
        im,
    }
}

/// Log-spaced frequency grid over `[omega_min, omega_max]` with `points`
/// samples.
pub fn log_grid(omega_min: f64, omega_max: f64, points: usize) -> Vec<f64> {
    assert!(omega_min > 0.0 && omega_max > omega_min);
    assert!(points >= 2);
    let lo = omega_min.ln();
    let hi = omega_max.ln();
    (0..points)
        .map(|i| (lo + (hi - lo) * i as f64 / (points - 1) as f64).exp())
        .collect()
}

/// Computes gain and phase margins on a log grid over
/// `[omega_min, omega_max]`.
///
/// The grid must be dense enough to bracket each crossover once; margins
/// are refined by linear interpolation between grid points.
pub fn margins(
    tf: &TransferFunction,
    omega_min: f64,
    omega_max: f64,
    points: usize,
) -> Option<Margins> {
    let grid = log_grid(omega_min, omega_max, points);
    let data = bode_data(tf, &grid);
    let mut gain_margin = f64::INFINITY;
    let mut phase_margin_deg = 180.0;
    let mut phase_crossover_omega = None;
    let mut gain_crossover_omega = None;

    // k indexes four parallel arrays at once; explicit indexing is clearest.
    #[allow(clippy::needless_range_loop)]
    for k in 0..grid.len() - 1 {
        let (m0, m1) = (data.magnitude_db[k], data.magnitude_db[k + 1]);
        let (p0, p1) = (data.phase_deg[k], data.phase_deg[k + 1]);

        // Unity-gain crossing (0 dB): phase margin.
        if gain_crossover_omega.is_none() && (m0 <= 0.0) != (m1 <= 0.0) {
            let frac = -m0 / (m1 - m0);
            let w_gc = grid[k] + frac * (grid[k + 1] - grid[k]);
            // Refine the phase at the interpolated crossover.
            let phase = p0 + frac * (p1 - p0);
            gain_crossover_omega = Some(w_gc);
            phase_margin_deg = phase + 180.0;
        }

        // -180 degree phase crossing: gain margin.
        if phase_crossover_omega.is_none() && (p0 + 180.0) * (p1 + 180.0) <= 0.0 && p0 != p1 {
            let frac = (-180.0 - p0) / (p1 - p0);
            if (0.0..=1.0).contains(&frac) {
                let w_pc = grid[k] + frac * (grid[k + 1] - grid[k]);
                let g = tf.eval_jw(w_pc).abs();
                phase_crossover_omega = Some(w_pc);
                if g > 0.0 {
                    gain_margin = 1.0 / g;
                }
            }
        }
    }

    Some(Margins {
        gain_margin,
        phase_margin_deg,
        gain_crossover_omega,
        phase_crossover_omega,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tf::TransferFunction;
    use tpt_ctrl_verify::testing::assert_close;

    #[test]
    fn integrator_chain_margins_match_textbook() {
        // G(s) = 1/(s(s+1)): PM ~= 51.8 deg at w ~= 0.786, GM infinite.
        let g = TransferFunction::new(&[1.0], &[1.0, 1.0, 0.0]).unwrap();
        let m = margins(&g, 1e-3, 1e3, 4000).unwrap();
        assert!(
            (m.phase_margin_deg - 51.8).abs() < 0.5,
            "{}",
            m.phase_margin_deg
        );
        assert!(m.gain_margin.is_infinite());
        let w = m.gain_crossover_omega.unwrap();
        assert!((w - 0.786).abs() < 0.02, "gc omega {w}");
    }

    #[test]
    fn third_order_lag_has_finite_gm() {
        // G(s) = 4/(s+1)^3: |G| = 1 at the phase crossover w = sqrt(3)
        // scaled: GM = 2 (loop goes unstable beyond K = 8); PM ~= 27 deg.
        let g = TransferFunction::new(&[4.0], &[1.0, 3.0, 3.0, 1.0]).unwrap();
        let m = margins(&g, 1e-2, 1e2, 8000).unwrap();
        assert!((m.gain_margin - 2.0).abs() < 0.1, "gm = {}", m.gain_margin);
        assert!(
            (m.phase_margin_deg - 27.3).abs() < 1.0,
            "pm = {}",
            m.phase_margin_deg
        );
    }

    #[test]
    fn bode_low_pass_slope() {
        // 1/(s+1): magnitude 0 dB at w << 1, -20 dB/dec above; phase -> -90.
        let g = TransferFunction::new(&[1.0], &[1.0, 1.0]).unwrap();
        let data = bode_data(&g, &[1e-2, 1e2, 1e4]);
        assert!(data.magnitude_db[0].abs() < 0.01);
        assert!((data.magnitude_db[1] + 40.0).abs() < 0.1);
        assert!((data.phase_deg[1] + 90.0).abs() < 1.0);
        assert!((data.phase_deg[2] + 90.0).abs() < 0.01);
    }

    #[test]
    fn phase_unwraps_across_branch_cut() {
        // 1/(s+1)^4: phase = -4 atan(w) falls through -180 deg; the unwrapped
        // trace must continue below -180 rather than jump to +180.
        let g = TransferFunction::new(&[1.0], &[1.0, 4.0, 6.0, 4.0, 1.0]).unwrap();
        let data = bode_data(&g, &[0.1, 1.0, 10.0]);
        assert!(data.phase_deg[0] > -30.0, "{}", data.phase_deg[0]);
        assert!(data.phase_deg[1] < -150.0, "{}", data.phase_deg[1]);
        assert!(data.phase_deg[2] < -330.0, "{}", data.phase_deg[2]);
    }

    #[test]
    fn nyquist_real_parts_match_bode() {
        let g = TransferFunction::new(&[1.0], &[1.0, 1.0]).unwrap();
        let grid = log_grid(1e-2, 1e2, 50);
        let ny = nyquist_data(&g, &grid);
        let bd = bode_data(&g, &grid);
        for (k, &w) in grid.iter().enumerate() {
            let h = g.eval_jw(w);
            assert_close(ny.re[k], h.re, 1e-12);
            assert_close(ny.im[k], h.im, 1e-12);
            assert_close(bd.magnitude_db[k], 20.0 * h.abs().log10(), 1e-12);
        }
    }
}
