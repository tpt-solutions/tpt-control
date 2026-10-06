//! Strictly convex quadratic programming via an active-set method.
//!
//! Problem: `minimize ½ xᵀ G x + gᵀ x` subject to `Cᵀ x >= b`
//! (rows of `Cᵀ` are the constraint normals), with `G` symmetric positive
//! definite.
//!
//! Strategy: maintain a working set of active equality constraints; solve
//! the equality-constrained subproblem exactly through its KKT system;
//! add the most-violated inactive constraint or drop the most-negative
//! multiplier until both KKT conditions hold. Finite for non-degenerate
//! problems (standard anti-cycling via the most-negative-multiplier rule);
//! a generous iteration budget guards against degeneracy.

use std::vec::Vec;

use tpt_ctrl_core::dmat::DMat;
use tpt_ctrl_core::error::{LinalgError, Result};

/// A QP solution: primal `x` and multipliers `lambda` (one per constraint
/// row; zero for inactive constraints).
#[derive(Clone, Debug, PartialEq)]
pub struct QpSolution {
    /// Primal minimizer.
    pub x: DMat<f64>,
    /// Lagrange multipliers (all constraints, inactive = 0).
    pub lambda: Vec<f64>,
    /// Working set at termination.
    pub active: Vec<usize>,
}

/// Solves `min ½ xᵀ G x + gᵀ x  s.t.  Cᵀ x >= b`.
///
/// `c` holds the constraint normals as ROWS: constraint `i` is
/// `c.row(i) · x >= b[i]`.
pub fn solve(g: &DMat<f64>, gt: &DMat<f64>, c: &DMat<f64>, b: &[f64]) -> Result<QpSolution> {
    let n = g.nrows();
    if !g.is_square() || gt.nrows() != n {
        return Err(LinalgError::DimensionMismatch {
            expected: (n, n),
            found: (gt.nrows(), gt.ncols()),
        });
    }
    let m = c.nrows();
    if c.ncols() != n || b.len() != m {
        return Err(LinalgError::DimensionMismatch {
            expected: (m, n),
            found: (c.ncols(), b.len()),
        });
    }
    if !g.is_positive_definite() {
        return Err(LinalgError::NotPositiveDefinite);
    }
    if m == 0 {
        // Pure unconstrained solve.
        let x = g.solve(&(-gt))?;
        return Ok(QpSolution {
            x,
            lambda: Vec::new(),
            active: Vec::new(),
        });
    }

    let tol_violation = 1e-9;
    let tol_lambda = -1e-9;
    let mut active: Vec<usize> = Vec::new();
    let mut lambda = vec![0.0f64; m];

    // Guard against cycling in degenerate problems.
    let max_outer = 20 * (m + n) + 100;
    for _ in 0..max_outer {
        // Solve the equality-constrained subproblem:
        // [[G, C_A], [C_Aᵀ, 0]] [x; ν] = [-g; b_A]
        let na = active.len();
        let kkt = DMat::from_fn(n + na, n + na, |i, j| {
            if i < n && j < n {
                g[(i, j)]
            } else if i < n {
                c[(active[j - n], i)]
            } else if j < n {
                c[(active[i - n], j)]
            } else {
                0.0
            }
        });
        let mut rhs = DMat::zeros(n + na, 1);
        for i in 0..n {
            rhs[(i, 0)] = -gt[(i, 0)];
        }
        for (k, &ai) in active.iter().enumerate() {
            rhs[(n + k, 0)] = b[ai];
        }
        let sol = match kkt.solve(&rhs) {
            Ok(s) => s,
            // Singular KKT (linearly dependent working set, e.g. two
            // constraints fighting over the same variable): drop the oldest
            // working-set member so the newly pushed constraint gets a
            // chance; popping the new one instead would cycle forever.
            Err(LinalgError::SingularMatrix) => {
                if active.is_empty() {
                    return Err(LinalgError::SingularMatrix);
                }
                active.remove(0);
                continue;
            }
            Err(e) => return Err(e),
        };
        let x = sol.submatrix(0, 0, n, 1);
        // The KKT layout [[G, C_A], [C_A^T, 0]][x; nu] = [-g; b] yields
        // subproblem multipliers nu = -lambda, where `lambda >= 0` is the
        // standard inequality multiplier for `c^T x >= b`.
        for (k, _) in active.iter().enumerate() {
            lambda[active[k]] = -sol[(n + k, 0)];
        }

        // Primal feasibility first: if any inactive constraint is violated,
        // add the most violated one. (Checking feasibility before dropping
        // multipliers keeps the working set growing toward the solution and
        // avoids drop/add cycling on the same constraint.)
        let mut worst_con = usize::MAX;
        let mut worst_viol = tol_violation;
        for i in 0..m {
            if active.contains(&i) {
                continue;
            }
            let mut val = 0.0;
            for j in 0..n {
                val += c[(i, j)] * x[(j, 0)];
            }
            let viol = b[i] - val;
            if viol > worst_viol {
                worst_viol = viol;
                worst_con = i;
            }
        }
        if worst_con != usize::MAX {
            active.push(worst_con);
            continue;
        }

        // Feasible: now enforce dual feasibility — drop the constraint with
        // the most negative standard multiplier.
        let mut worst_lam = 0usize;
        let mut worst_val = tol_lambda;
        for (k, &ai) in active.iter().enumerate() {
            if lambda[ai] < worst_val {
                worst_val = lambda[ai];
                worst_lam = k;
            }
        }
        if worst_val < tol_lambda {
            let removed = active.swap_remove(worst_lam);
            lambda[removed] = 0.0;
            continue;
        }

        lambda.iter_mut().enumerate().for_each(|(i, v)| {
            if !active.contains(&i) {
                *v = 0.0;
            }
        });
        return Ok(QpSolution { x, lambda, active });
    }
    Err(LinalgError::NoConvergence)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_ctrl_verify::testing::{assert_close, assert_mat_close};

    #[test]
    fn unconstrained_qp_analytic() {
        // min x^2 + y^2 - 2x - 4y  =>  x = (1, 2).
        let g = DMat::identity(2) * 2.0;
        let gt = DMat::from_rows(&[&[-2.0], &[-4.0]]);
        let c = DMat::zeros(0, 2);
        let sol = solve(&g, &gt, &c, &[]).unwrap();
        assert_close(sol.x[(0, 0)], 1.0, 1e-9);
        assert_close(sol.x[(1, 0)], 2.0, 1e-9);
    }

    #[test]
    fn box_constraint_binds() {
        // min (x-2)^2 s.t. x <= 1  (i.e. -x >= -1): solution x = 1.
        let g = DMat::from_rows(&[&[2.0]]);
        let gt = DMat::from_rows(&[&[-4.0]]);
        let c = DMat::from_rows(&[&[-1.0]]);
        let sol = solve(&g, &gt, &c, &[-1.0]).unwrap();
        assert_close(sol.x[(0, 0)], 1.0, 1e-9);
        assert_close(sol.lambda[0], 2.0, 1e-9);
    }

    #[test]
    fn two_constraints_one_active() {
        // min x^2 + y^2 - 10x s.t. x <= 3, y <= 5, x + y >= 2.
        let g = DMat::identity(2) * 2.0;
        let gt = DMat::from_rows(&[&[-10.0], &[0.0]]);
        let c = DMat::from_rows(&[&[-1.0, 0.0], &[0.0, -1.0], &[1.0, 1.0]]);
        let bvec = [-3.0, -5.0, 2.0];
        let sol = solve(&g, &gt, &c, &bvec).unwrap();
        // Unconstrained optimum (5, 0) violates x <= 3; solution (3, 0),
        // multiplier 4 on the x constraint.
        assert_close(sol.x[(0, 0)], 3.0, 1e-9);
        assert_close(sol.x[(1, 0)], 0.0, 1e-9);
        assert_close(sol.lambda[0], 4.0, 1e-9);
        assert_close(sol.lambda[1], 0.0, 1e-9);
    }

    #[test]
    fn inequality_constraint_slack() {
        // Active-set must recognize a non-binding constraint:
        // min x^2 + y^2 s.t. x + y >= -10 (optimum at origin).
        let g = DMat::identity(2) * 2.0;
        let gt = DMat::zeros(2, 1);
        let c = DMat::from_rows(&[&[1.0, 1.0]]);
        let sol = solve(&g, &gt, &c, &[-10.0]).unwrap();
        assert_close(sol.x[(0, 0)], 0.0, 1e-9);
        assert_close(sol.lambda[0], 0.0, 1e-9);
    }

    #[test]
    fn kkt_conditions_hold_on_random_qp() {
        // Small random strictly convex QP with box constraints; verify the
        // KKT system of the returned active set.
        let g = DMat::from_rows(&[&[4.0, 1.0], &[1.0, 3.0]]);
        let gt = DMat::from_rows(&[&[-1.0], &[-2.0]]);
        let c = DMat::from_rows(&[&[-1.0, 0.0], &[0.0, -1.0], &[1.0, 0.0], &[0.0, 1.0]]);
        let bvec = [-1.5, -0.5, 0.5, 0.2];
        let sol = solve(&g, &gt, &c, &bvec).unwrap();
        // Feasibility.
        for i in 0..4 {
            let val = c[(i, 0)] * sol.x[(0, 0)] + c[(i, 1)] * sol.x[(1, 0)];
            assert!(val >= bvec[i] - 1e-8, "constraint {i} violated: {val}");
        }
        // Stationarity: G x + g - sum lambda_i c_i = 0.
        let mut stat = &(&g * &sol.x) + &gt;
        for (i, &lam) in sol.lambda.iter().enumerate() {
            stat = &stat - &(&c.submatrix(i, 0, 1, 2).transpose() * lam);
        }
        assert!(
            stat.norm_fro() < 1e-8,
            "stationarity residual {}",
            stat.norm_fro()
        );
        let _ = assert_mat_close;
    }
}
