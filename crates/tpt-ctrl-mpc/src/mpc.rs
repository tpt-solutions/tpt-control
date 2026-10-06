//! Condensed linear MPC with input bounds and inequality constraints.
//!
//! The cost over the horizon is
//! `sum_{k=1}^{N} x_k^T Q x_k + x_N^T Q_f x_N + sum_{k=0}^{N-1} u_k^T R u_k`,
//! condensed into a QP in the stacked inputs `U` via [`crate::unroll::unroll`]:
//! `min ½ Uᵀ H U + gᵀ U` with `H = S_uᵀ Q̄ S_u + R̄` and
//! `g = S_uᵀ Q̄ S_x x0`, solved with the from-scratch active-set QP at
//! every step (receding horizon).

use std::vec::Vec;

use tpt_ctrl_core::dmat::DMat;
use tpt_ctrl_core::error::Result;

use crate::qp;
use crate::unroll::{unroll, CondensedHorizon};

/// Solution of one MPC solve.
#[derive(Clone, Debug, PartialEq)]
pub struct MpcSolution {
    /// First control move to apply (`u_0`).
    pub u0: Vec<f64>,
    /// Full planned input sequence (stacked `m*N x 1`).
    pub u_sequence: DMat<f64>,
    /// Predicted states `x_1..x_N`.
    pub predicted: Vec<DMat<f64>>,
    /// Optimal cost.
    pub cost: f64,
}

/// A state constraint `g^T x_step <= h` anchored at horizon step `step`.
struct StateConstraint {
    g: DMat<f64>,
    h: f64,
    step: usize,
}

/// Builder for an [`Mpc`] controller.
pub struct MpcBuilder {
    a: DMat<f64>,
    b: DMat<f64>,
    horizon: usize,
    q: Option<DMat<f64>>,
    r: Option<DMat<f64>>,
    qf: Option<DMat<f64>>,
    u_min: Option<Vec<f64>>,
    u_max: Option<Vec<f64>>,
    state_constraints: Vec<StateConstraint>,
}

impl MpcBuilder {
    /// Starts a builder for the discrete system `x+ = A x + B u`.
    pub fn new(a: DMat<f64>, b: DMat<f64>, horizon: usize) -> MpcBuilder {
        MpcBuilder {
            a,
            b,
            horizon,
            q: None,
            r: None,
            qf: None,
            u_min: None,
            u_max: None,
            state_constraints: Vec::new(),
        }
    }

    /// Stage costs `Q` (state) and `R` (input).
    pub fn stage_costs(mut self, q: &DMat<f64>, r: &DMat<f64>) -> MpcBuilder {
        self.q = Some(q.clone());
        self.r = Some(r.clone());
        self
    }

    /// Terminal state cost `Q_f`.
    pub fn terminal_cost(mut self, qf: &DMat<f64>) -> MpcBuilder {
        self.qf = Some(qf.clone());
        self
    }

    /// Box input bounds (per-input; vectors of length `m`).
    pub fn input_bounds(mut self, u_min: Vec<f64>, u_max: Vec<f64>) -> MpcBuilder {
        self.u_min = Some(u_min);
        self.u_max = Some(u_max);
        self
    }

    /// Adds a linear state constraint `g^T x_k <= h` enforced at the
    /// `step`-th predicted state (`step` in `0..horizon`).
    pub fn state_constraint(mut self, g: Vec<f64>, h: f64, step: usize) -> MpcBuilder {
        self.state_constraints.push(StateConstraint {
            g: DMat::from_fn(g.len(), 1, |i, _| g[i]),
            h,
            step,
        });
        self
    }

    /// Builds the controller (freezes the condensed QP data).
    pub fn build(self) -> Mpc {
        let n = self.a.nrows();
        let m = self.b.ncols();
        let n_h = self.horizon;
        let horizon = unroll(&self.a, &self.b, n_h).expect("unroll: bad dimensions");
        let q = self.q.unwrap_or_else(|| DMat::identity(n));
        let r = self.r.unwrap_or_else(|| DMat::identity(m));
        let qf = self.qf.unwrap_or_else(|| q.clone());

        // Q̄ = blkdiag(Q, ..., Q, Q_f)
        let mut q_bar = DMat::zeros(n * n_h, n * n_h);
        for k in 0..n_h {
            let weight = if k == n_h - 1 { &qf } else { &q };
            for i in 0..n {
                for j in 0..n {
                    q_bar[(k * n + i, k * n + j)] = weight[(i, j)];
                }
            }
        }

        // H = S_uᵀ Q̄ S_u + blkdiag(R, ..., R)
        let q_su = &q_bar * &horizon.su;
        let mut h = &horizon.su.transpose() * &q_su;
        for k in 0..n_h {
            for i in 0..m {
                h[(k * m + i, k * m + i)] += r[(i, i)];
            }
        }

        // Constraints as rows `c_i · U >= b_i` (note the flip: our QP is
        // in >= form, so upper bounds enter negated).
        let mut rows: Vec<Vec<f64>> = Vec::new();
        let mut rhs: Vec<f64> = Vec::new();
        if let (Some(umin), Some(umax)) = (self.u_min, self.u_max) {
            for k in 0..n_h {
                for i in 0..m {
                    // u_{k,i} <= umax[i]  <=>  -u >= -umax
                    let mut row = vec![0.0; m * n_h];
                    row[k * m + i] = -1.0;
                    rows.push(row);
                    rhs.push(-umax[i]);
                    // u_{k,i} >= umin[i]
                    let mut row = vec![0.0; m * n_h];
                    row[k * m + i] = 1.0;
                    rows.push(row);
                    rhs.push(umin[i]);
                }
            }
        }
        // State constraints: g^T x_k <= h with x_k = [Sx x0 + Su U]_{k}:
        //   (g^T S_u^{(k)}) U <= h - g^T S_x^{(k)} x0
        //   <=> -(g^T S_u^{(k)}) U >= -h + g^T S_x^{(k)} x0
        // The x0 part is folded at solve time.
        let mut couplings = Vec::new();
        for sc in &self.state_constraints {
            // g^T S_u^{(step)}: the step's n x mN block of S_u.
            let su_block = horizon.su.submatrix(sc.step * n, 0, n, m * n_h);
            let gx_su = &(sc.g.transpose() * &su_block) * -1.0;
            let row_idx = rows.len();
            let mut row = vec![0.0; m * n_h];
            for c in 0..m * n_h {
                row[c] = gx_su[(0, c)];
            }
            rows.push(row);
            rhs.push(-sc.h);
            couplings.push((row_idx, sc.g.clone(), sc.step));
        }

        let constraints_c = if rows.is_empty() {
            DMat::zeros(0, m * n_h)
        } else {
            DMat::from_rows(&rows.iter().map(|r| r.as_slice()).collect::<Vec<_>>())
        };
        Mpc {
            n,
            m,
            horizon: n_h,
            horizon_data: horizon,
            q_bar,
            h,
            constraints_c,
            constraints_b: rhs,
            couplings,
        }
    }
}

/// Condensed receding-horizon MPC controller.
pub struct Mpc {
    n: usize,
    m: usize,
    horizon: usize,
    horizon_data: CondensedHorizon,
    q_bar: DMat<f64>,
    h: DMat<f64>,
    constraints_c: DMat<f64>,
    constraints_b: Vec<f64>,
    couplings: Vec<(usize, DMat<f64>, usize)>,
}

impl Mpc {
    /// Plans the input sequence from state `x0` and returns the first move.
    pub fn solve(&self, x0: &DMat<f64>) -> Result<MpcSolution> {
        // Gradient of ½UᵀHU + (Q̄ Sx x0)ᵀ U.
        let sx_x0 = &self.horizon_data.sx * x0;
        let g = &(self.horizon_data.su.transpose() * &(self.q_bar.clone() * &sx_x0));

        // Constraint right-hand side with the x0 couplings:
        // b[row] += g^T S_x^{(step)} x0.
        let mut b = self.constraints_b.clone();
        for &(row, ref g_col, step) in &self.couplings {
            let mut acc = 0.0;
            for i in 0..self.n {
                for j in 0..self.n {
                    acc +=
                        g_col[(i, 0)] * self.horizon_data.sx[(step * self.n + i, j)] * x0[(j, 0)];
                }
            }
            b[row] += acc;
        }

        let sol = qp::solve(&self.h, g, &self.constraints_c, &b)?;
        let u0: Vec<f64> = (0..self.m).map(|i| sol.x[(i, 0)]).collect();
        let predicted =
            crate::unroll::predict_trajectory(&self.horizon_data, x0, &sol.x, self.n, self.m);
        // Cost ½UᵀHU + gᵀU.
        let quad = &(sol.x.transpose() * &self.h) * &sol.x;
        let lin = &(g.transpose() * &sol.x);
        let cost = 0.5 * quad[(0, 0)] + lin[(0, 0)];
        Ok(MpcSolution {
            u0,
            u_sequence: sol.x,
            predicted,
            cost,
        })
    }

    /// Number of states.
    pub fn n_states(&self) -> usize {
        self.n
    }

    /// Number of inputs.
    pub fn n_inputs(&self) -> usize {
        self.m
    }

    /// Horizon length.
    pub fn horizon(&self) -> usize {
        self.horizon
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_ctrl_verify::testing::assert_close;

    #[test]
    fn unconstrained_mpc_matches_dlqr() {
        // With no active constraints and a long horizon (large terminal
        // weight approximating the infinite-horizon cost), the first MPC
        // move approaches the DLQR feedback -K x.
        let a = DMat::from_rows(&[&[1.0, 0.1], &[0.0, 1.0]]);
        let b = DMat::from_rows(&[&[0.005], &[0.1]]);
        let q = DMat::identity(2);
        let r = DMat::from_rows(&[&[1.0]]);
        let sol_dlqr = tpt_ctrl_optimal::dlqr(&a, &b, &q, &r).unwrap();

        let mpc = MpcBuilder::new(a.clone(), b.clone(), 40)
            .stage_costs(&q, &r)
            .terminal_cost(&sol_dlqr.x)
            .build();
        let x0 = DMat::from_rows(&[&[0.7], &[-0.3]]);
        let sol = mpc.solve(&x0).unwrap();
        let u_dlqr = -(sol_dlqr.k * &x0)[(0, 0)];
        assert_close(sol.u0[0], u_dlqr, 1e-4);
    }

    #[test]
    fn input_bounds_are_respected() {
        let a = DMat::from_rows(&[&[1.0, 0.1], &[0.0, 1.0]]);
        let b = DMat::from_rows(&[&[0.005], &[0.1]]);
        let mpc = MpcBuilder::new(a, b, 15)
            .stage_costs(&DMat::identity(2), &DMat::from_rows(&[&[0.1]]))
            .terminal_cost(&(&DMat::identity(2) * 10.0))
            .input_bounds(vec![-0.4], vec![0.4])
            .build();
        // A large initial state demands saturation.
        let sol = mpc.solve(&DMat::from_rows(&[&[3.0], &[0.0]])).unwrap();
        assert!(
            sol.u0[0].abs() <= 0.4 + 1e-7,
            "u0 = {} violates bounds",
            sol.u0[0]
        );
        for k in 0..15 {
            let uk = sol.u_sequence[(k, 0)];
            assert!((-0.4 - 1e-7..=0.4 + 1e-7).contains(&uk));
        }
    }

    #[test]
    fn closed_loop_regulates_with_constraints() {
        let a = DMat::from_rows(&[&[1.0, 0.1], &[0.0, 1.0]]);
        let b = DMat::from_rows(&[&[0.005], &[0.1]]);
        let mpc = MpcBuilder::new(a.clone(), b.clone(), 20)
            .stage_costs(&DMat::identity(2), &DMat::from_rows(&[&[0.5]]))
            .terminal_cost(&(&DMat::identity(2) * 5.0))
            .input_bounds(vec![-0.3], vec![0.3])
            .build();
        let mut x = DMat::from_rows(&[&[1.5], &[0.0]]);
        for _ in 0..120 {
            let sol = mpc.solve(&x).unwrap();
            x = &(&a * &x) + &(&b * sol.u0[0]);
        }
        assert!(
            x.norm_fro() < 1e-2,
            "closed loop did not regulate: {}",
            x.norm_fro()
        );
    }

    #[test]
    fn state_constraint_respected() {
        // Constrain the first predicted state's velocity: x1[1] <= 0.1.
        let a = DMat::from_rows(&[&[1.0, 0.1], &[0.0, 1.0]]);
        let b = DMat::from_rows(&[&[0.005], &[0.1]]);
        let mpc = MpcBuilder::new(a, b, 12)
            .stage_costs(&DMat::identity(2), &DMat::from_rows(&[&[0.1]]))
            .terminal_cost(&(&DMat::identity(2) * 5.0))
            .input_bounds(vec![-1.0], vec![1.0])
            .state_constraint(vec![0.0, -1.0], 0.05, 0)
            .build();
        // From x0 = [0.8, 0] the optimizer wants to brake as hard as the
        // input bound allows (u0 = -1). The constraint -velocity(x_1) <= 0.05
        // means 0.1 u0 >= -0.05, i.e. u0 >= -0.5: it must bind first.
        let sol = mpc.solve(&DMat::from_rows(&[&[0.8], &[0.0]])).unwrap();
        assert!(
            -0.1 * sol.u0[0] <= 0.05 + 1e-7,
            "state constraint violated: {}",
            -0.1 * sol.u0[0]
        );
        assert!(
            (sol.u0[0] - (-0.5)).abs() < 1e-6,
            "expected binding u0 = -0.5, got {}",
            sol.u0[0]
        );
    }
}
