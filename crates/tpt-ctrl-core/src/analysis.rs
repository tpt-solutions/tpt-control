//! LTI system analysis: poles, stability, controllability, observability.

use alloc::vec::Vec;

use crate::c64::C64;
use crate::dmat::{require_square, DMat};
use crate::error::{LinalgError, Result};
use crate::field::Field;

/// Eigenvalues of `A` — the poles of the system (continuous: `Re(p)`;
/// discrete: `|p|` decides stability).
pub fn poles<T: Field>(a: &DMat<T>) -> Result<Vec<C64>> {
    require_square(a)?;
    a.eigenvalues()
}

/// Largest eigenvalue magnitude (spectral radius).
pub fn spectral_radius<T: Field>(a: &DMat<T>) -> Result<T> {
    let mut r = T::zero();
    for p in a.eigenvalues()? {
        let m = T::from_f64((p.re * p.re + p.im * p.im).sqrt());
        if m > r {
            r = m;
        }
    }
    Ok(r)
}

/// `true` when all poles have real part below `-tol` (Hurwitz).
pub fn is_stable_continuous<T: Field>(a: &DMat<T>, tol: f64) -> Result<bool> {
    for p in a.eigenvalues()? {
        if p.re >= -tol {
            return Ok(false);
        }
    }
    Ok(true)
}

/// `true` when all poles are inside the unit circle with margin `tol`.
pub fn is_stable_discrete<T: Field>(a: &DMat<T>, tol: f64) -> Result<bool> {
    for p in a.eigenvalues()? {
        let m = (p.re * p.re + p.im * p.im).sqrt();
        if m >= 1.0 - tol {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Controllability matrix `[B, A B, ..., A^{n-1} B]` (`n x (n*m)`).
pub fn controllability_matrix<T: Field>(a: &DMat<T>, b: &DMat<T>) -> Result<DMat<T>> {
    let n = a.nrows();
    if b.nrows() != n {
        return Err(LinalgError::DimensionMismatch {
            expected: (n, b.ncols()),
            found: (b.nrows(), b.ncols()),
        });
    }
    let mut blocks = Vec::with_capacity(n);
    let mut ak_b = b.clone();
    for _ in 0..n {
        blocks.push(ak_b.clone());
        ak_b = a * &ak_b;
    }
    let mut out = blocks[0].clone();
    for blk in &blocks[1..] {
        out = out.augment(blk);
    }
    Ok(out)
}

/// Observability matrix `[C; C A; ...; C A^{n-1}]` (`(n*p) x n`).
pub fn observability_matrix<T: Field>(a: &DMat<T>, c: &DMat<T>) -> Result<DMat<T>> {
    let at = a.transpose();
    let ct = c.transpose();
    Ok(controllability_matrix(&at, &ct)?.transpose())
}

/// Numerical rank via QR with a relative tolerance
/// (`|R_ii| <= tol * max|R_ii|` counts as zero).
pub fn rank<T: Field>(a: &DMat<T>, tol: f64) -> usize {
    if a.is_empty() {
        return 0;
    }
    let (_, r) = a.qr();
    let mut best = T::zero();
    for i in 0..r.nrows().min(r.ncols()) {
        let v = r[(i, i)].abs();
        if v > best {
            best = v;
        }
    }
    if best == T::zero() {
        return 0;
    }
    let mut count = 0usize;
    for i in 0..r.nrows().min(r.ncols()) {
        if r[(i, i)].abs() > T::from_f64(tol) * best {
            count += 1;
        }
    }
    count
}

/// `true` when `(A, B)` is controllable.
pub fn is_controllable<T: Field>(a: &DMat<T>, b: &DMat<T>, tol: f64) -> Result<bool> {
    let c = controllability_matrix(a, b)?;
    Ok(rank(&c, tol) == a.nrows())
}

/// `true` when `(A, C)` is observable.
pub fn is_observable<T: Field>(a: &DMat<T>, c: &DMat<T>, tol: f64) -> Result<bool> {
    let o = observability_matrix(a, c)?;
    Ok(rank(&o, tol) == a.nrows())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn stability_classification() {
        let stable = DMat::from_rows(&[&[-1.0, 0.0], &[0.0, -2.0]]);
        assert!(is_stable_continuous(&stable, 1e-9).unwrap());
        assert!(!is_stable_continuous(&stable, 3.0).unwrap());

        let ad = DMat::from_rows(&[&[0.5, 0.0], &[0.0, 0.9]]);
        assert!(is_stable_discrete(&ad, 1e-9).unwrap());
        let ad_unstable = DMat::from_rows(&[&[1.5, 0.0], &[0.0, 0.9]]);
        assert!(!is_stable_discrete(&ad_unstable, 1e-9).unwrap());
    }

    #[test]
    fn oscillator_is_marginal_not_stable() {
        let a = DMat::from_rows(&[&[0.0, 1.0], &[-1.0, 0.0]]);
        assert!(!is_stable_continuous(&a, 1e-9).unwrap());
        assert!(close(spectral_radius(&a).unwrap().to_f64(), 1.0));
    }

    #[test]
    fn double_integrator_controllable_not_observable_when_c_zero() {
        let a = DMat::from_rows(&[&[0.0, 1.0], &[0.0, 0.0]]);
        let b = DMat::from_rows(&[&[0.0], &[1.0]]);
        assert!(is_controllable(&a, &b, 1e-9).unwrap());

        let c = DMat::from_rows(&[&[1.0, 0.0]]);
        assert!(is_observable(&a, &c, 1e-9).unwrap());
        let c_bad = DMat::from_rows(&[&[0.0, 0.0]]);
        assert!(!is_observable(&a, &c_bad, 1e-9).unwrap());
    }

    #[test]
    fn rank_of_known_matrices() {
        assert_eq!(rank(&DMat::<f64>::identity(3), 1e-9), 3);
        let a = DMat::from_rows(&[&[1.0, 2.0], &[2.0, 4.0]]);
        assert_eq!(rank(&a, 1e-9), 1);
        assert_eq!(rank(&DMat::<f64>::zeros(2, 2), 1e-9), 0);
    }

    #[test]
    fn controllability_matrix_shape() {
        let a = DMat::from_rows(&[&[0.0, 1.0], &[0.0, 0.0]]);
        let b = DMat::from_rows(&[&[0.0], &[1.0]]);
        let cm = controllability_matrix(&a, &b).unwrap();
        assert_eq!(cm.nrows(), 2);
        assert_eq!(cm.ncols(), 2);
        // [B AB] = [[0, 1], [1, 0]]
        assert!(close(cm[(0, 0)], 0.0) && close(cm[(0, 1)], 1.0));
        assert!(close(cm[(1, 0)], 1.0) && close(cm[(1, 1)], 0.0));
    }
}
