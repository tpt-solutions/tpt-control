//! Horizon condensing: the predicted trajectory as an affine function of
//! the input sequence.
//!
//! For `x+ = A x + B u` the response over `N` steps is
//! `X = Sx x0 + Su U` with
//!
//! ```text
//! X   = [x_1; x_2; ...; x_N]         (n*N x 1)
//! U   = [u_0; u_1; ...; u_{N-1}]     (m*N x 1)
//! Sx  = [A; A^2; ...; A^N]
//! Su  = [[B, 0, ..., 0],
//!        [A B, B, ..., 0],
//!        [...],
//!        [A^{N-1} B, ..., B]]
//! ```

use tpt_ctrl_core::dmat::DMat;
use tpt_ctrl_core::error::{LinalgError, Result};

/// The condensed horizon matrices `(Sx, Su)`.
#[derive(Clone, Debug, PartialEq)]
pub struct CondensedHorizon {
    /// Free response blocks `A^k x0`, stacked (`n*N x 1` after applying x0).
    pub sx: DMat<f64>,
    /// Forced-response block-triangular matrix (`n*N x m*N`).
    pub su: DMat<f64>,
}

/// Builds `(Sx, Su)` for the discrete system `(A, B)` over `N` steps.
pub fn unroll(a: &DMat<f64>, b: &DMat<f64>, horizon: usize) -> Result<CondensedHorizon> {
    let n = a.nrows();
    let m = b.ncols();
    if !a.is_square() || b.nrows() != n || horizon == 0 {
        return Err(LinalgError::DimensionMismatch {
            expected: (n, n),
            found: (b.nrows(), m),
        });
    }
    let mut sx = DMat::zeros(n * horizon, n);
    let mut su = DMat::zeros(n * horizon, m * horizon);

    // Iterate powers A^k * B row-block by row-block (O(N) matrix products).
    let mut ab_acc = b.clone(); // A^k B for k = 0..N-1
    let mut a_acc = DMat::identity(n);
    for k in 0..horizon {
        // Sx block k: A^{k+1}
        a_acc = &a_acc * a;
        set_block(&mut sx, k * n, 0, &a_acc);
        // Su block (k, j) = A^{k-j} B for j <= k
        set_block(&mut su, k * n, 0, &ab_acc);
        ab_acc = a * &ab_acc;
    }
    // Fill the triangular Su blocks by shifting: block (k, j) for j in 1..=k
    // equals block (k-1, j-1) — copy row-block k-1 shifted by m columns.
    for k in 1..horizon {
        for j in 1..=k {
            for i in 0..n {
                for c in 0..m {
                    su[(k * n + i, j * m + c)] = su[((k - 1) * n + i, (j - 1) * m + c)];
                }
            }
        }
    }
    Ok(CondensedHorizon { sx, su })
}

/// Copies `block` into `dst` at `(r0, c0)`.
fn set_block(dst: &mut DMat<f64>, r0: usize, c0: usize, block: &DMat<f64>) {
    for i in 0..block.nrows() {
        for j in 0..block.ncols() {
            dst[(r0 + i, c0 + j)] = block[(i, j)];
        }
    }
}

/// Applies the condensation: returns the predicted states
/// `x_1..x_N` from `X = Sx x0 + Su U` (`m` = inputs per step).
pub fn predict_trajectory(
    horizon: &CondensedHorizon,
    x0: &DMat<f64>,
    u: &DMat<f64>,
    n: usize,
    m: usize,
) -> Vec<DMat<f64>> {
    let count = horizon.su.ncols() / m;
    let x = &(&horizon.sx * x0) + &(&horizon.su * u);
    (0..count).map(|k| x.submatrix(k * n, 0, n, 1)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_ctrl_verify::testing::assert_close;

    #[test]
    fn unroll_matches_manual_propagation() {
        let a = DMat::from_rows(&[&[1.0, 0.1], &[0.0, 1.0]]);
        let b = DMat::from_rows(&[&[0.005], &[0.1]]);
        let n = 2;
        let horizon = 4;
        let ch = unroll(&a, &b, horizon).unwrap();

        let x0 = DMat::from_rows(&[&[1.0], &[-0.5]]);
        // Input sequence: u = [0.5, -0.3, 0.2, 0.0]
        let u = DMat::from_rows(&[&[0.5], &[-0.3], &[0.2], &[0.0]]);
        let traj = predict_trajectory(&ch, &x0, &u, n, 1);

        // Manual propagation.
        let mut x = x0.clone();
        for k in 0..horizon {
            let uk = DMat::from_rows(&[&[u[(k, 0)]]]);
            x = &(&a * &x) + &(&b * &uk);
            for i in 0..n {
                assert_close(traj[k][(i, 0)], x[(i, 0)], 1e-12);
            }
        }
    }

    #[test]
    fn zero_input_gives_free_response() {
        let a = DMat::from_rows(&[&[0.9]]);
        let b = DMat::from_rows(&[&[1.0]]);
        let ch = unroll(&a, &b, 3).unwrap();
        let x0 = DMat::from_rows(&[&[2.0]]);
        let u = DMat::zeros(3, 1);
        let traj = predict_trajectory(&ch, &x0, &u, 1, 1);
        assert_close(traj[0][(0, 0)], 1.8, 1e-12);
        assert_close(traj[1][(0, 0)], 1.62, 1e-12);
        assert_close(traj[2][(0, 0)], 1.458, 1e-12);
    }
}
