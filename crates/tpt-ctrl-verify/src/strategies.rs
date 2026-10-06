//! `proptest` strategies generating valid, well-conditioned control objects.
//!
//! These are the statistical-verification backbone: every solver in the
//! workspace is property-tested against randomly generated systems drawn
//! from these strategies.

use proptest::prelude::*;

use tpt_ctrl_core::dmat::DMat;
use tpt_ctrl_core::state_space::DynSystem;

/// A random finite `f64` in a controller-friendly range.
pub fn coef() -> impl Strategy<Value = f64> {
    (-10.0_f64..10.0).prop_filter("reject near-zero", |v: &f64| v.abs() > 1e-3)
}

/// A random matrix of the given shape with bounded entries.
pub fn matrix(nrows: usize, ncols: usize) -> impl Strategy<Value = DMat<f64>> {
    proptest::collection::vec(-5.0..5.0, nrows * ncols)
        .prop_map(move |data| DMat::from_row_slice(nrows, ncols, &data))
}

/// A random symmetric positive definite matrix (via `M^T M + 0.5 I`).
pub fn positive_definite(n: usize) -> impl Strategy<Value = DMat<f64>> {
    matrix(n, n).prop_map(move |m| {
        let mt = m.transpose();
        let spd = &(&mt * &m) + &DMat::identity(n) * 0.5;
        // Symmetrize to kill rounding drift.
        let mt_spd = spd.transpose();
        &(&spd + &mt_spd) * 0.5
    })
}

/// A random Hurwitz (continuous-time stable) `A`: shifting by
/// `max(Re(lambda)) + eps` places every eigenvalue strictly in the open
/// left half-plane.
pub fn stable_continuous_a(n: usize) -> impl Strategy<Value = DMat<f64>> {
    matrix(n, n).prop_map(move |m| {
        let radius = m
            .eigenvalues()
            .map(|ev| ev.iter().map(|z| z.re.max(0.0)).fold(0.0, f64::max))
            .unwrap_or(0.0);
        let shift = radius + 0.1;
        &m - &(&DMat::identity(n) * shift)
    })
}

/// A random Schur (discrete-time stable) `A` with spectral radius below 0.95.
///
/// Built as `A = alpha * Q D Q^T` with orthogonal `Q` (from QR of a random
/// matrix) and diagonal `D` with entries in `(-0.95, 0.95)`.
pub fn stable_discrete_a(n: usize) -> impl Strategy<Value = DMat<f64>> {
    (matrix(n, n), proptest::collection::vec(-0.9..0.9, n)).prop_map(move |(m, diag)| {
        let (q, _) = m.qr();
        let d = DMat::diagonal(&diag);
        &(&q * &d) * &q.transpose()
    })
}

/// A random stable LTI system with the given dimensions.
pub fn lti_system(n: usize, m: usize, p: usize) -> impl Strategy<Value = DynSystem<f64>> {
    (
        stable_continuous_a(n),
        matrix(n, m),
        matrix(p, n),
        matrix(p, m),
    )
        .prop_map(move |(a, b, c, d)| DynSystem { a, b, c, d })
}

/// A discrete-time stable system (Schur `A`, sample time in `[0.01, 1]`).
pub fn discrete_lti_system(
    n: usize,
    m: usize,
    p: usize,
) -> impl Strategy<Value = (DynSystem<f64>, f64)> {
    (
        stable_discrete_a(n),
        matrix(n, m),
        matrix(p, n),
        matrix(p, m),
        0.01..1.0,
    )
        .prop_map(move |(a, b, c, d, dt)| (DynSystem { a, b, c, d }, dt))
}

/// Bounded zero-mean noise entries (for simulating sensor noise profiles).
pub fn bounded_noise(len: usize, bound: f64) -> impl Strategy<Value = Vec<f64>> {
    proptest::collection::vec(-bound..bound, len)
}

#[cfg(test)]
mod tests {
    use super::*;

    proptest! {
        #[test]
        fn positive_definite_is_cholky(m in positive_definite(3)) {
            prop_assert!(m.is_positive_definite());
        }

        #[test]
        fn continuous_strategy_is_stable(a in stable_continuous_a(3)) {
            prop_assert!(tpt_ctrl_core::analysis::is_stable_continuous(&a, 1e-9).unwrap());
        }

        #[test]
        fn discrete_strategy_is_stable(a in stable_discrete_a(4)) {
            prop_assert!(tpt_ctrl_core::analysis::is_stable_discrete(&a, 1e-9).unwrap());
        }

        #[test]
        fn generated_systems_have_consistent_shapes(s in lti_system(3, 2, 2)) {
            prop_assert_eq!(s.a.nrows(), 3);
            prop_assert_eq!(s.b.ncols(), 2);
            prop_assert_eq!(s.c.nrows(), 2);
        }
    }
}

#[cfg(test)]
mod eigen_invariants {
    use super::*;
    use tpt_ctrl_core::c64::C64;

    // Schur eigenvalues must reproduce the trace and (for real spectra)
    // stay consistent with the determinant — regression guard for the
    // Francis iteration and the 2x2 block splitting.
    proptest! {
        #[test]
        fn eigenvalues_match_invariants_3x3(a in matrix(3, 3)) {
            let ev = a.eigenvalues().unwrap();
            let sum: f64 = ev.iter().map(|z| z.re).sum();
            let trace = a.trace();
            prop_assert!((sum - trace).abs() <= 1e-8 * (1.0 + trace.abs()));
            // det = prod(lambda) holds exactly in exact arithmetic; the
            // modulus product is checked with a loose relative tolerance
            // because eigenvalue condition numbers amplify rounding.
            let prod: f64 = ev.iter().map(|z: &C64| z.norm_sqr().sqrt()).product();
            let det = a.det().unwrap().abs();
            prop_assert!((prod - det).abs() <= 1e-5 * (1.0 + det.abs()));
        }
    }
}
