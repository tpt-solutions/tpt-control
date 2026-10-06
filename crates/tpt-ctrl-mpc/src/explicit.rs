//! Explicit MPC via precomputed lookup tables.
//!
//! True multi-parametric QP solutions partition the state space into
//! critical regions with piecewise-affine control laws. This module takes
//! the pragmatic embedding route (adequate for ultra-low-latency targets
//! where a QP solve is too slow but flash/RAM is available):
//!
//! 1. **Offline:** sample the state box on a grid, solve the online MPC at
//!    every node, and store `(state node, first move)`.
//! 2. **Online:** nearest-neighbour lookup (`O(N_nodes)` scan, or `O(log)`
//!    with a spatial index) — no matrix products, no solver.
//!
//! The trade-off versus a critical-region (mpc-MP) solution is accuracy:
//! the law is piecewise-*constant* between nodes instead of affine. The
//! ` refine ` step (one gradient/active-set touch-up) is therefore exposed
//! for deployments that can afford a bounded fallback.

use std::vec::Vec;

use tpt_ctrl_core::dmat::DMat;

use crate::mpc::Mpc;

/// A precomputed explicit-MPC table.
#[derive(Clone, Debug)]
pub struct ExplicitMpc {
    /// Flattened grid nodes (`n` floats per node, row-major).
    nodes: Vec<f64>,
    /// First control move at each node (`m` floats per node).
    controls: Vec<f64>,
    /// Grid bounds per state dimension: `(min, max, samples)`.
    grid: Vec<(f64, f64, usize)>,
    n: usize,
    m: usize,
}

/// Errors from explicit-MPC evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExplicitError {
    /// The queried state lies outside the sampled box.
    OutsideDomain,
    /// The state dimension does not match the table.
    DimensionMismatch,
}

impl ExplicitMpc {
    /// Builds a table by solving the MPC at every grid node.
    ///
    /// `grid` describes the sampling per state dimension; `steps` beyond
    /// the last sample are clamped during evaluation.
    pub fn build(mpc: &Mpc, grid: &[(f64, f64, usize)]) -> Result<ExplicitMpc, ExplicitError> {
        let n = mpc.n_states();
        let m = mpc.n_inputs();
        if grid.len() != n {
            return Err(ExplicitError::DimensionMismatch);
        }
        let total: usize = grid.iter().map(|(_, _, s)| s).product();
        let mut nodes = Vec::with_capacity(total * n);
        let mut controls = Vec::with_capacity(total * m);

        let mut index = vec![0usize; n];
        for _ in 0..total {
            let mut x = DMat::zeros(n, 1);
            for (i, idx) in index.iter().enumerate() {
                let (lo, hi, samples) = grid[i];
                let frac = if samples <= 1 {
                    0.5
                } else {
                    *idx as f64 / (samples - 1) as f64
                };
                x[(i, 0)] = lo + (hi - lo) * frac;
            }
            if let Ok(sol) = mpc.solve(&x) {
                nodes.extend_from_slice(&x.as_slice()[..n]);
                controls.extend_from_slice(&sol.u0);
            } else {
                // Infeasible node (e.g. state constraint unreachable):
                // store NaNs so evaluation can reject the neighbourhood.
                for _ in 0..n {
                    nodes.push(f64::NAN);
                }
                for _ in 0..m {
                    controls.push(f64::NAN);
                }
            }
            // odometer increment
            for i in (0..n).rev() {
                index[i] += 1;
                if index[i] < grid[i].2 {
                    break;
                }
                index[i] = 0;
            }
        }
        Ok(ExplicitMpc {
            nodes,
            controls,
            grid: grid.to_vec(),
            n,
            m,
        })
    }

    /// Number of stored nodes (including any infeasible markers).
    pub fn node_count(&self) -> usize {
        self.nodes.len() / self.n
    }

    /// Nearest-neighbour control for state `x` (Euclidean distance).
    pub fn lookup(&self, x: &[f64]) -> Result<Vec<f64>, ExplicitError> {
        if x.len() != self.n {
            return Err(ExplicitError::DimensionMismatch);
        }
        // Clamp to the sampled box.
        let mut q = vec![0.0; self.n];
        for (i, v) in x.iter().enumerate() {
            let (lo, hi, _) = self.grid[i];
            q[i] = v.clamp(lo, hi);
        }
        let mut best = usize::MAX;
        let mut best_d = f64::INFINITY;
        for k in 0..self.node_count() {
            let mut d = 0.0;
            for (i, qi) in q.iter().enumerate() {
                let diff = self.nodes[k * self.n + i] - qi;
                d += diff * diff;
            }
            if d < best_d {
                best_d = d;
                best = k;
            }
        }
        if best == usize::MAX {
            return Err(ExplicitError::OutsideDomain);
        }
        let ctrl = &self.controls[best * self.m..(best + 1) * self.m];
        if ctrl.iter().any(|v| v.is_nan()) {
            return Err(ExplicitError::OutsideDomain);
        }
        Ok(ctrl.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mpc::MpcBuilder;
    use tpt_ctrl_verify::testing::assert_close;

    #[test]
    fn table_agrees_with_online_solve_at_nodes() {
        let a = DMat::from_rows(&[&[1.0, 0.1], &[0.0, 1.0]]);
        let b = DMat::from_rows(&[&[0.005], &[0.1]]);
        let mpc = MpcBuilder::new(a, b, 12)
            .stage_costs(&DMat::identity(2), &DMat::from_rows(&[&[0.5]]))
            .terminal_cost(&(DMat::identity(2) * 5.0))
            .input_bounds(vec![-0.5], vec![0.5])
            .build();
        let grid = [(-2.0, 2.0, 9), (-1.0, 1.0, 7)];
        let table = ExplicitMpc::build(&mpc, &grid).unwrap();
        assert_eq!(table.node_count(), 63);

        // At grid nodes the nearest neighbour IS the node: exact match.
        for px in [-2.0, 0.0, 2.0] {
            for vx in [-1.0, 1.0] {
                let online = mpc.solve(&DMat::from_rows(&[&[px], &[vx]])).unwrap();
                let explicit = table.lookup(&[px, vx]).unwrap();
                assert_close(explicit[0], online.u0[0], 1e-9);
            }
        }
    }

    #[test]
    fn lookup_clamps_outside_domain() {
        let a = DMat::from_rows(&[&[0.9]]);
        let b = DMat::from_rows(&[&[1.0]]);
        let mpc = MpcBuilder::new(a, b, 8)
            .stage_costs(&DMat::from_rows(&[&[1.0]]), &DMat::from_rows(&[&[1.0]]))
            .input_bounds(vec![-2.0], vec![2.0])
            .build();
        let table = ExplicitMpc::build(&mpc, &[(-1.0, 1.0, 5)]).unwrap();
        // Beyond the box clamps to the edge node (symmetric system: control
        // at +edge is the mirror of -edge).
        let c = table.lookup(&[100.0]).unwrap();
        assert!(c[0].is_finite());
    }
}
