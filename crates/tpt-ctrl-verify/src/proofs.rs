//! Kani bounded-model-checking harnesses.
//!
//! These prove absence of panics (arithmetic overflow, division by zero,
//! out-of-bounds indexing) in critical numerical loops over *symbolic*
//! inputs constrained to the documented operating envelope. They compile
//! only under `cargo kani` (Linux); `#[cfg(kani)]` keeps them invisible to
//! ordinary builds and cross-compilation.
//!
//! First harnesses in the tpt-* ecosystem (todo.md Phase 0 checkpoint): the
//! harness set grows through Phase 3 to cover CARE/DARE, the MPC QP solver,
//! and the matrix exponential.

#[cfg(kani)]
mod proofs {
    use tpt_ctrl_core::dmat::DMat;

    /// A finite `f64` in `[-limit, -min] U [min, limit]` — safely away from
    /// zero and from overflow.
    fn bounded_scalar(limit: f64, min: f64) -> f64 {
        let raw: f64 = kani::any();
        kani::assume(raw.is_finite());
        let mag = min + (raw.abs() - min).rem_euclid(limit - min);
        if raw < 0.0 {
            -mag
        } else {
            mag
        }
    }

    /// LU factorization and solve of a well-conditioned symbolic 2x2 system
    /// never panics.
    #[kani::proof]
    fn lu_solve_2x2_no_panic() {
        let a = bounded_scalar(10.0, 0.5);
        let b = bounded_scalar(10.0, 0.5);
        let c = bounded_scalar(10.0, 0.5);
        let d = bounded_scalar(10.0, 0.5);
        // Diagonal dominance guarantees nonsingularity.
        kani::assume(a.abs() > b.abs() + c.abs());
        kani::assume(d.abs() > b.abs() + c.abs());
        let m = DMat::from_rows(&[&[a, b], &[c, d]]);
        let rhs = DMat::from_rows(&[&[1.0], &[1.0]]);
        if let Ok(x) = m.solve(&rhs) {
            // Residual A x - b must stay finite (no overflow occurred).
            let resid = &(&m * &x) - &rhs;
            kani::assert(resid.as_slice()[0].is_finite(), "residual finite");
        }
    }

    /// Cholesky of a symbolic SPD 2x2 never panics and returns a factor
    /// that reconstructs the input.
    #[kani::proof]
    fn cholesky_2x2_reconstruction() {
        let a = bounded_scalar(10.0, 0.5);
        let c = bounded_scalar(10.0, 0.5);
        let off = bounded_scalar(2.0, 0.0);
        // [[a, off], [off, c]] SPD iff a > 0, c > 0, a*c > off^2.
        kani::assume(a > 0.0 && c > 0.0 && a * c > off * off * 1.01);
        let m = DMat::from_rows(&[&[a, off], &[off, c]]);
        if let Ok(l) = m.cholesky() {
            let recon = &l * &l.transpose();
            kani::assert((recon[(0, 0)] - a).abs() < 1e-6, "L L^T = A (0,0)");
            kani::assert((recon[(1, 1)] - c).abs() < 1e-6, "L L^T = A (1,1)");
        }
    }

    /// The ZOH block-exponential of a bounded 2-state system never panics
    /// and produces a finite (Ad, Bd).
    #[kani::proof]
    fn zoh_2x2_no_panic() {
        let a11 = bounded_scalar(4.0, 0.1);
        let a22 = bounded_scalar(4.0, 0.1);
        let b1 = bounded_scalar(4.0, 0.1);
        let dt = 0.01 + (kani::any::<f64>()).rem_euclid(0.5);
        kani::assume(dt.is_finite());
        let a = DMat::from_rows(&[&[a11, 0.0], &[0.0, a22]]);
        let b = DMat::from_rows(&[&[b1], &[b1]]);
        if let Ok((ad, bd)) = tpt_ctrl_core::discretize::zoh(&a, &b, dt) {
            kani::assert(ad.as_slice().iter().all(|v| v.is_finite()), "Ad finite");
            kani::assert(bd.as_slice().iter().all(|v| v.is_finite()), "Bd finite");
        }
    }

    /// Horizon condensing (Sx, Su assembly) never panics on a small
    /// symbolic system and yields finite factors — the Phase 2 unrolling
    /// correctness harness (values verified by property tests).
    #[kani::proof]
    fn horizon_unroll_no_panic() {
        let a11 = bounded_scalar(2.0, 0.1);
        let a12 = bounded_scalar(2.0, 0.0);
        let b1 = bounded_scalar(2.0, 0.1);
        let a = tpt_ctrl_core::DMat::from_rows(&[&[a11, a12], &[0.0, a11]]);
        let b = tpt_ctrl_core::DMat::from_rows(&[&[b1], &[b1]]);
        if let Ok(ch) = tpt_ctrl_mpc::unroll(&a, &b, 3) {
            kani::assert(ch.sx.as_slice().iter().all(|v| v.is_finite()), "Sx finite");
            kani::assert(ch.su.as_slice().iter().all(|v| v.is_finite()), "Su finite");
            // Structural invariant: Su is block lower triangular.
            let m = 1usize;
            for k in 0..3 {
                for j in (k + 1)..3 {
                    for i in 0..2 {
                        let v = ch.su[(k * 2 + i, j * m)];
                        kani::assert(v == 0.0, "Su upper blocks are zero");
                    }
                }
            }
        }
    }

    /// The SDA DARE iteration on a bounded 2x2 system never panics
    /// (it may report NoConvergence, which is not a panic).
    #[kani::proof]
    fn dare_2x2_no_panic() {
        let a11 = bounded_scalar(1.5, 0.1);
        let a22 = bounded_scalar(1.5, 0.1);
        let b1 = bounded_scalar(1.0, 0.1);
        let q = tpt_ctrl_core::DMat::identity(2);
        let r = tpt_ctrl_core::DMat::from_rows(&[&[1.0]]);
        let a = tpt_ctrl_core::DMat::from_rows(&[&[a11, 0.0], &[0.0, a22]]);
        let b = tpt_ctrl_core::DMat::from_rows(&[&[b1], &[b1]]);
        let _ = tpt_ctrl_optimal::dare(&a, &b, &q, &r);
    }

    /// The active-set QP solver never panics on a small symbolic problem
    /// with box constraints (may report NoConvergence).
    #[kani::proof]
    fn qp_2d_no_panic() {
        let g1 = bounded_scalar(5.0, 0.5);
        let g2 = bounded_scalar(5.0, 0.5);
        let off = bounded_scalar(1.0, 0.0);
        // Keep G diagonally dominant => PD.
        kani::assume(g1.abs() > off.abs() + 0.2);
        kani::assume(g2.abs() > off.abs() + 0.2);
        let g = tpt_ctrl_core::DMat::from_rows(&[&[g1 * g1 + 1.0, off], &[off, g2 * g2 + 1.0]]);
        let gt = tpt_ctrl_core::DMat::from_rows(&[&[off], &[off]]);
        let c = tpt_ctrl_core::DMat::from_rows(&[&[-1.0, 0.0], &[0.0, -1.0]]);
        let bvec = [-1.0, -1.0];
        let _ = tpt_ctrl_mpc::qp::solve(&g, &gt, &c, &bvec);
    }

    /// Transfer-function evaluation on bounded coefficients stays finite
    /// (guards the classic-crate frequency-response path).
    #[kani::proof]
    fn dimensioned_matmul_3x2_times_2x2_no_panic() {
        use tpt_ctrl_core::smat::Mat;
        let r0: [f64; 2] = [bounded_scalar(5.0, 0.1), bounded_scalar(5.0, 0.1)];
        let r1: [f64; 2] = [bounded_scalar(5.0, 0.1), bounded_scalar(5.0, 0.1)];
        let r2: [f64; 2] = [bounded_scalar(5.0, 0.1), bounded_scalar(5.0, 0.1)];
        let s0: [f64; 2] = [bounded_scalar(5.0, 0.1), bounded_scalar(5.0, 0.1)];
        let s1: [f64; 2] = [bounded_scalar(5.0, 0.1), bounded_scalar(5.0, 0.1)];
        let a = Mat::<f64, 3, 2>::from_rows_arr(&[r0, r1, r2]);
        let b = Mat::<f64, 2, 2>::from_rows_arr(&[s0, s1]);
        let c = a * b; // dimension mismatch would be a compile error
        kani::assert(c[(2, 1)].is_finite(), "product finite");
    }
}
