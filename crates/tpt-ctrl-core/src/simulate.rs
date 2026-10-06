//! Simulation helpers for state-space and generic dynamics.
//!
//! Discrete systems propagate exactly; continuous systems use classical
//! fixed-step RK4. The [`Dynamics`] trait accepts any closure with the right
//! signature, so nonlinear plants can be simulated without ceremony.

use alloc::vec;
use alloc::vec::Vec;

use crate::dmat::DMat;
use crate::field::Field;

/// First-order dynamics `x'(t) = f(t, x, u)`.
///
/// Implemented automatically for any
/// `Fn(t, x: &[T], u: &[T], out: &mut [T])` closure.
pub trait Dynamics<T> {
    /// Writes `x'(t, x, u)` into `out`.
    fn rhs(&self, t: T, x: &[T], u: &[T], out: &mut [T]);
}

impl<T, F> Dynamics<T> for F
where
    F: Fn(T, &[T], &[T], &mut [T]),
{
    fn rhs(&self, t: T, x: &[T], u: &[T], out: &mut [T]) {
        self(t, x, u, out);
    }
}

/// One classical RK4 step in place with a constant control over the step.
pub fn rk4_step<D, T>(dynamics: &D, t: T, dt: T, x: &mut [T], u: &[T])
where
    D: Dynamics<T>,
    T: Field,
{
    let n = x.len();
    let mut k1 = vec![T::zero(); n];
    let mut k2 = vec![T::zero(); n];
    let mut k3 = vec![T::zero(); n];
    let mut k4 = vec![T::zero(); n];
    let mut tmp = vec![T::zero(); n];

    dynamics.rhs(t, x, u, &mut k1);
    let half = T::from_f64(0.5);
    for i in 0..n {
        tmp[i] = x[i] + half * dt * k1[i];
    }
    dynamics.rhs(t + half * dt, &tmp, u, &mut k2);
    for i in 0..n {
        tmp[i] = x[i] + half * dt * k2[i];
    }
    dynamics.rhs(t + half * dt, &tmp, u, &mut k3);
    for i in 0..n {
        tmp[i] = x[i] + dt * k3[i];
    }
    dynamics.rhs(t + dt, &tmp, u, &mut k4);
    let sixth = T::from_f64(1.0 / 6.0);
    for i in 0..n {
        x[i] += sixth * dt * (k1[i] + T::from_f64(2.0) * (k2[i] + k3[i]) + k4[i]);
    }
}

/// A simulated trajectory: `states[k]` is the state at time `k * dt`.
pub struct Trajectory<T> {
    /// State history (one entry per step, including the initial state).
    pub states: Vec<Vec<T>>,
}

/// Simulates a continuous-time system with a constant input using RK4.
pub fn simulate_continuous<D, T>(
    dynamics: &D,
    x0: &[T],
    u: &[T],
    dt: T,
    steps: usize,
) -> Trajectory<T>
where
    D: Dynamics<T>,
    T: Field,
{
    let mut states = Vec::with_capacity(steps + 1);
    let mut x = x0.to_vec();
    states.push(x.clone());
    let mut t = T::zero();
    for _ in 0..steps {
        rk4_step(dynamics, t, dt, &mut x, u);
        states.push(x.clone());
        t += dt;
    }
    Trajectory { states }
}

/// Simulates a discrete-time system `(A_d, B_d)` with a per-step input
/// sequence; the last input is held if `us` is shorter than `steps`.
pub fn simulate_discrete<T>(
    ad: &DMat<T>,
    bd: &DMat<T>,
    x0: &[T],
    us: &[Vec<T>],
    steps: usize,
) -> Trajectory<T>
where
    T: Field,
{
    let n = ad.nrows();
    let m = bd.ncols();
    let mut states = Vec::with_capacity(steps + 1);
    let mut x = DMat::from_fn(n, 1, |i, _| x0[i]);
    states.push(x0.to_vec());
    for k in 0..steps {
        let uk: Vec<T> = us
            .get(k)
            .cloned()
            .unwrap_or_else(|| us.last().cloned().unwrap_or_else(|| vec![T::zero(); m]));
        let u = DMat::from_fn(m, 1, |i, _| uk[i]);
        x = &(ad * &x) + &(bd * &u);
        states.push(x.column(0));
    }
    Trajectory { states }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rk4_integrates_integrator_exactly() {
        // x' = u with constant u: RK4 is exact for polynomials of degree <= 4.
        let d = |_: f64, x: &[f64], u: &[f64], out: &mut [f64]| {
            out[0] = x[1];
            out[1] = u[0];
        };
        let mut x = vec![0.0, 0.0];
        let u = [1.0];
        let dt = 0.01;
        let steps = 100;
        for k in 0..steps {
            rk4_step(&d, k as f64 * dt, dt, &mut x, &u);
        }
        // x(t) = t^2 / 2, v(t) = t with t = 1.0
        assert!((x[1] - 1.0).abs() < 1e-12);
        assert!((x[0] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn rk4_decays_exponentially() {
        let d = |_: f64, x: &[f64], _: &[f64], out: &mut [f64]| out[0] = -2.0 * x[0];
        let mut x = vec![1.0];
        let u: [f64; 0] = [];
        let dt = 0.001;
        for k in 0..1000 {
            rk4_step(&d, k as f64 * dt, dt, &mut x, &u);
        }
        assert!((x[0] - (-2.0_f64).exp()).abs() < 1e-9);
    }

    #[test]
    fn discrete_simulation_matches_step() {
        let ad = DMat::from_rows(&[&[0.9, 0.0], &[0.0, 0.8]]);
        let bd = DMat::from_rows(&[&[1.0], &[0.0]]);
        let traj = simulate_discrete(&ad, &bd, &[1.0, 1.0], &[vec![0.0]], 3);
        assert_eq!(traj.states.len(), 4);
        assert!((traj.states[3][0] - 0.9_f64.powi(3)).abs() < 1e-12);
        assert!((traj.states[3][1] - 0.8_f64.powi(3)).abs() < 1e-12);
    }
}
