//! Matrix exponential via scaling-and-squaring with a degree-13 Pade
//! approximant (Higham's algorithm).
//!
//! This is the numerical core of exact zero-order-hold discretization and of
//! the Van Loan tricks used across the workspace.

use crate::dmat::{require_square, DMat};
use crate::error::{LinalgError, Result};
use crate::field::Field;
use crate::math;

/// Pade-13 coefficients (Higham 2005), `B[k]` multiplies `A^k`
/// (i.e. ascending powers; `B[0]` is the constant term).
const B: [f64; 14] = [
    64764752532480000.0,
    32382376266240000.0,
    7771770303897600.0,
    1187353796428800.0,
    129060195264000.0,
    10559470521600.0,
    670442572800.0,
    33522128640.0,
    1323241920.0,
    40840800.0,
    960960.0,
    16380.0,
    182.0,
    1.0,
];

/// Backward-error threshold for the degree-13 Pade approximant.
const THETA_13: f64 = 5.371_920_351_148_152;

/// Computes `e^A` for a square matrix.
pub fn expm<T: Field>(a: &DMat<T>) -> Result<DMat<T>> {
    require_square(a)?;
    let n = a.nrows();
    if n == 0 {
        return Ok(a.clone());
    }

    // Scaling: s such that ||A / 2^s||_1 <= THETA_13.
    let norm = a.norm_1();
    let mut s = 0usize;
    if norm > T::from_f64(THETA_13) {
        let log2n = (norm / T::from_f64(THETA_13)).ln().to_f64() / core::f64::consts::LN_2;
        s = math::ceil(log2n).max(0.0) as usize;
    }
    let scale = T::from_f64(2.0).powi(-(s as i32));
    let a1 = a * scale;

    let a2 = &a1 * &a1;
    let a4 = &a2 * &a2;
    let a6 = &a2 * &a4;
    let eye = DMat::<T>::identity(n);
    let b = |k: usize| T::from_f64(B[k]);

    // U = odd part of p13(A): powers 13, 11, 9, 7, 5, 3, 1.
    // U = A [ A6 (b13 A6 + b11 A4 + b9 A2 + b7 I) + b5 A4 + b3 A2 + b1 I ]
    let z = &a6 * b(13) + &a4 * b(11) + &a2 * b(9) + &eye * b(7);
    let y = &a4 * b(5) + &a2 * b(3) + &eye * b(1);
    let u = &a1 * (&(&a6 * &z) + &y);

    // V = even part: powers 12, 10, 8, 6, 4, 2, 0.
    // V = A6 (b12 A6 + b10 A4 + b8 A2 + b6 I) + A2 (b4 A2 + b2 I) + b0 I
    let w = &a6 * b(12) + &a4 * b(10) + &a2 * b(8) + &eye * b(6);
    let x = &a2 * b(4) + &eye * b(2);
    let v = &(&a6 * &w) + &(&a2 * &x) + &eye * b(0);

    let numer = &v + &u;
    let denom = &v - &u;
    let mut r = denom
        .solve(&numer)
        .map_err(|_| LinalgError::SingularMatrix)?;

    // Squaring: e^A = (e^{A/2^s})^{2^s}.
    for _ in 0..s {
        r = &r * &r;
    }
    Ok(r)
}

impl<T: Field> DMat<T> {
    /// Matrix exponential (see [`expm`]).
    pub fn expm(&self) -> Result<DMat<T>> {
        expm(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    fn mat_close(a: &DMat<f64>, b: &DMat<f64>) -> bool {
        a.nrows() == b.nrows()
            && a.ncols() == b.ncols()
            && a.as_slice()
                .iter()
                .zip(b.as_slice())
                .all(|(x, y)| close(*x, *y))
    }

    #[test]
    fn exp_nilpotent_unit_entries() {
        // exp([[0, 1], [0, 0]]) = I + N = [[1, 1], [0, 1]].
        let a = DMat::from_rows(&[&[0.0, 1.0], &[0.0, 0.0]]);
        let e = expm(&a).unwrap();
        assert!(mat_close(&e, &DMat::from_rows(&[&[1.0, 1.0], &[0.0, 1.0]])));
    }

    #[test]
    fn exp_diagonal() {
        let a = DMat::from_rows(&[&[1.0, 0.0], &[0.0, 2.0]]);
        let e = expm(&a).unwrap();
        assert!(close(e[(0, 0)], 1.0_f64.exp()));
        assert!(close(e[(1, 1)], 2.0_f64.exp()));
        assert!(close(e[(0, 1)], 0.0));
    }

    #[test]
    fn exp_nilpotent() {
        // exp([[0, c], [0, 0]]) = I + N.
        let a = DMat::from_rows(&[&[0.0, 2.5], &[0.0, 0.0]]);
        let e = expm(&a).unwrap();
        assert!(close(e[(0, 1)], 2.5));
        assert!(close(e[(0, 0)], 1.0) && close(e[(1, 1)], 1.0));
    }

    #[test]
    fn exp_rotation_matrix() {
        // exp(theta * J) with J = [[0, 1], [-1, 0]] is a planar rotation.
        let theta = 0.7;
        let a = DMat::from_rows(&[&[0.0, theta], &[-theta, 0.0]]);
        let e = expm(&a).unwrap();
        assert!(close(e[(0, 0)], theta.cos()));
        assert!(close(e[(0, 1)], theta.sin()));
        assert!(close(e[(1, 0)], -theta.sin()));
        assert!(close(e[(1, 1)], theta.cos()));
    }

    #[test]
    fn exp_large_norm_scales_correctly() {
        let a = DMat::from_rows(&[&[-30.0, 1.0], &[0.0, -30.0]]);
        let e = expm(&a).unwrap();
        // Exact: e^{-30} * [[1, 1], [0, 1]]
        let x = (-30.0_f64).exp();
        assert!(close(e[(0, 0)], x) && close(e[(0, 1)], x) && close(e[(1, 1)], x));
    }

    #[test]
    fn exp_semigroup_property() {
        let a = DMat::from_rows(&[&[0.3, 1.2], &[-0.4, 0.1]]);
        let e1 = expm(&a).unwrap();
        let e2 = expm(&(&a * 0.5)).unwrap();
        let e2sq = &e2 * &e2;
        let rel = (&e1 - &e2sq).norm_fro() / e1.norm_fro();
        assert!(rel < 1e-10, "semigroup property violated: {rel}");
    }
}
