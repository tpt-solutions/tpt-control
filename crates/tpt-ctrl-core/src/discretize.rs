//! Continuous-to-discrete conversions, implemented from scratch.
//!
//! * [`zoh`] — exact zero-order hold via the Van Loan block-matrix
//!   exponential trick.
//! * [`tustin`] — bilinear (Tustin) transform.

use crate::dmat::DMat;
use crate::error::Result;
use crate::field::Field;

/// Exact zero-order-hold discretization: given continuous `(A, B)` and sample
/// time `dt`, returns `(A_d, B_d)` such that holding `u` constant over each
/// interval integrates the dynamics exactly.
pub fn zoh<T: Field>(a: &DMat<T>, b: &DMat<T>, dt: T) -> Result<(DMat<T>, DMat<T>)> {
    let n = a.nrows();
    let m = b.ncols();
    if a.ncols() != n || b.nrows() != n {
        return Err(crate::error::LinalgError::DimensionMismatch {
            expected: (n, n),
            found: (a.nrows(), b.nrows()),
        });
    }
    if n == 0 {
        return Ok((a.clone(), b.clone()));
    }
    let zeros_mn = DMat::<T>::zeros(m, n);
    let zeros_mm = DMat::<T>::zeros(m, m);
    // exp([[A, B], [0, 0]] * dt) = [[Ad, Bd], [0, I]] — the whole block is
    // scaled by dt, B included.
    let block = DMat::block2(&(a * dt), &(b * dt), &zeros_mn, &zeros_mm);
    let e = block.expm()?;
    let ad = e.submatrix(0, 0, n, n);
    let bd = e.submatrix(0, n, n, m);
    Ok((ad, bd))
}

/// Tustin (bilinear) discretization of `(A, B, C, D)` with sample time `dt`:
/// `s <- (2/dt) (z - 1) / (z + 1)`.
///
/// Returns `(A_d, B_d, C_d, D_d)` with
/// `A_d = (I - dt/2 A)^{-1} (I + dt/2 A)`, etc.
#[allow(clippy::type_complexity)]
pub fn tustin<T: Field>(
    a: &DMat<T>,
    b: &DMat<T>,
    c: &DMat<T>,
    d: &DMat<T>,
    dt: T,
) -> Result<(DMat<T>, DMat<T>, DMat<T>, DMat<T>)> {
    let n = a.nrows();
    if n == 0 {
        return Ok((a.clone(), b.clone(), c.clone(), d.clone()));
    }
    let eye = DMat::<T>::identity(n);
    let half_dt = dt * T::from_f64(0.5);
    // M = I - dt/2 A; its inverse appears in every output matrix.
    let m = &eye - &(a * half_dt);
    let lu = m.lu();
    if lu.is_singular() {
        return Err(crate::error::LinalgError::SingularMatrix);
    }
    let ad = lu.solve(&(&eye + &(a * half_dt)))?;
    let bd = &lu.solve(b)? * dt;
    // C_d = C (I - dt/2 A)^{-1}
    let m_inv = lu.inverse()?;
    let cd = c * &m_inv;
    let cb = c * &bd;
    let cb_half = &cb * T::from_f64(0.5);
    let dd = d + &cb_half;
    Ok((ad, bd, cd, dd))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12
    }

    #[test]
    fn zoh_known_closed_form() {
        // Double integrator has an exact closed form.
        let a = DMat::from_rows(&[&[0.0, 1.0], &[0.0, 0.0]]);
        let b = DMat::from_rows(&[&[0.0], &[1.0]]);
        let dt = 0.25;
        let (ad, bd) = zoh(&a, &b, dt).unwrap();
        assert!(close(ad[(0, 1)], dt));
        assert!(close(bd[(0, 0)], dt * dt / 2.0));
        assert!(close(bd[(1, 0)], dt));
    }

    #[test]
    fn zoh_oscillator_energy_preserving() {
        // Undamped oscillator: |Ad| must be unitary-ish (rotation * scale 1).
        let w = 2.0;
        let a = DMat::from_rows(&[&[0.0, 1.0], &[-w * w, 0.0]]);
        let b = DMat::from_rows(&[&[0.0], &[1.0]]);
        let dt = 0.1;
        let (ad, _) = zoh(&a, &b, dt).unwrap();
        // Closed form: [[cos(wt), sin(wt)/w], [-w sin(wt), cos(wt)]]
        assert!(close(ad[(0, 0)], (w * dt).cos()));
        assert!(close(ad[(0, 1)], (w * dt).sin() / w));
        assert!(close(ad[(1, 0)], -w * (w * dt).sin()));
        assert!(close(ad[(1, 1)], (w * dt).cos()));
    }

    #[test]
    fn zoh_matches_matrix_identity_step_response() {
        // Discrete propagation must equal exact integration for constant u.
        let a = DMat::from_rows(&[&[-0.5, 1.0], &[0.0, -1.5]]);
        let b = DMat::from_rows(&[&[0.2], &[1.0]]);
        let dt = 0.3;
        let (ad, bd) = zoh(&a, &b, dt).unwrap();
        let x0 = DMat::from_rows(&[&[1.0], &[-0.5]]);
        let u = 2.0;
        let x1 = &(&ad * &x0) + &(&bd * u);
        // Exact: x(dt) = e^{A dt} x0 + A^{-1}(e^{A dt} - I) B u
        let e = (&a * dt).expm().unwrap();
        let ainv = a.inverse().unwrap();
        let exact = &(&e * &x0) + &(&(&ainv * &(&e - &DMat::identity(2))) * &b) * u;
        assert!((&x1 - &exact).norm_fro() < 1e-10);
    }

    #[test]
    fn tustin_gain_at_dc_reasonable() {
        // First-order lag G(s) = 1/(s+1): DC gain 1, so Gd(z=1) = 1.
        let a = DMat::from_rows(&[&[-1.0]]);
        let b = DMat::from_rows(&[&[1.0]]);
        let c = DMat::from_rows(&[&[1.0]]);
        let d = DMat::from_rows(&[&[0.0]]);
        let dt = 0.1;
        let (ad, bd, cd, dd) = tustin(&a, &b, &c, &d, dt).unwrap();
        // Gd(1) = Dd + Cd (I - Ad)^{-1} Bd
        let inv_term = 1.0 / (1.0 - ad[(0, 0)]);
        let g1 = &dd + &(&(&bd * inv_term) * cd[(0, 0)]);
        assert!(close(g1[(0, 0)], 1.0));
    }
}
