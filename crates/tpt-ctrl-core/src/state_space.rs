//! Linear time-invariant state-space models `x' = A x + B u, y = C x + D u`
//! with compile-time dimensions, in continuous and discrete time.

use crate::discretize;
use crate::dmat::DMat;
use crate::error::Result;
use crate::field::Field;
use crate::smat::Mat;

/// The quadruple `(A, B, C, D)` of an LTI state-space model.
///
/// Dimensions are carried by the type parameters: `N` states, `M` inputs,
/// `P` outputs.
#[derive(Clone, Debug, PartialEq)]
pub struct StateSpace<T, const N: usize, const M: usize, const P: usize> {
    /// State matrix (`N x N`).
    pub a: Mat<T, N, N>,
    /// Input matrix (`N x M`).
    pub b: Mat<T, N, M>,
    /// Output matrix (`P x N`).
    pub c: Mat<T, P, N>,
    /// Feedthrough matrix (`P x M`).
    pub d: Mat<T, P, M>,
}

impl<T: Field, const N: usize, const M: usize, const P: usize> StateSpace<T, N, M, P> {
    /// Builds a model from its four matrices.
    pub const fn new(
        a: Mat<T, N, N>,
        b: Mat<T, N, M>,
        c: Mat<T, P, N>,
        d: Mat<T, P, M>,
    ) -> StateSpace<T, N, M, P> {
        StateSpace { a, b, c, d }
    }

    /// Dynamic (runtime-dimensioned) copy, ready for the numeric solvers.
    pub fn to_dynamic(&self) -> DynSystem<T> {
        DynSystem {
            a: self.a.to_dynamic(),
            b: self.b.to_dynamic(),
            c: self.c.to_dynamic(),
            d: self.d.to_dynamic(),
        }
    }
}

/// A continuous-time LTI system wrapping a [`StateSpace`].
#[derive(Clone, Debug, PartialEq)]
pub struct Continuous<T, const N: usize, const M: usize, const P: usize> {
    /// The underlying `(A, B, C, D)` quadruple.
    pub ss: StateSpace<T, N, M, P>,
}

impl<T: Field, const N: usize, const M: usize, const P: usize> Continuous<T, N, M, P> {
    /// Wraps a state-space quadruple as a continuous-time system.
    pub const fn new(ss: StateSpace<T, N, M, P>) -> Continuous<T, N, M, P> {
        Continuous { ss }
    }

    /// Exact zero-order-hold discretization at sample time `dt`
    /// (Van Loan: `expm([[A, B], [0, 0]] * dt)`).
    pub fn discretize_zoh(&self, dt: T) -> Result<Discrete<T, N, M, P>> {
        let (ad, bd) = discretize::zoh(&self.ss.a.to_dynamic(), &self.ss.b.to_dynamic(), dt)?;
        Ok(Discrete {
            ss: StateSpace {
                a: Mat::from_dynamic(&ad)?,
                b: Mat::from_dynamic(&bd)?,
                c: self.ss.c.clone(),
                d: self.ss.d.clone(),
            },
            dt,
        })
    }

    /// Tustin (bilinear) discretization at sample time `dt`.
    pub fn discretize_tustin(&self, dt: T) -> Result<Discrete<T, N, M, P>> {
        let (ad, bd, cd, dd) = discretize::tustin(
            &self.ss.a.to_dynamic(),
            &self.ss.b.to_dynamic(),
            &self.ss.c.to_dynamic(),
            &self.ss.d.to_dynamic(),
            dt,
        )?;
        Ok(Discrete {
            ss: StateSpace {
                a: Mat::from_dynamic(&ad)?,
                b: Mat::from_dynamic(&bd)?,
                c: Mat::from_dynamic(&cd)?,
                d: Mat::from_dynamic(&dd)?,
            },
            dt,
        })
    }
}

/// A discrete-time LTI system with sample time `dt`.
#[derive(Clone, Debug, PartialEq)]
pub struct Discrete<T, const N: usize, const M: usize, const P: usize> {
    /// The underlying `(A, B, C, D)` quadruple.
    pub ss: StateSpace<T, N, M, P>,
    /// Sample time.
    pub dt: T,
}

impl<T: Field, const N: usize, const M: usize, const P: usize> Discrete<T, N, M, P> {
    /// Wraps a state-space quadruple with its sample time.
    pub const fn new(ss: StateSpace<T, N, M, P>, dt: T) -> Discrete<T, N, M, P> {
        Discrete { ss, dt }
    }

    /// One propagation step `x+ = A x + B u`.
    pub fn step(&self, x: &Mat<T, N, 1>, u: &Mat<T, M, 1>) -> Mat<T, N, 1> {
        &self.ss.a * x + &self.ss.b * u
    }

    /// Output equation `y = C x + D u`.
    pub fn output(&self, x: &Mat<T, N, 1>, u: &Mat<T, M, 1>) -> Mat<T, P, 1> {
        &self.ss.c * x + &self.ss.d * u
    }
}

/// Runtime-dimensioned `(A, B, C, D)` quadruple used by the solver crates
/// (`tpt-ctrl-optimal`, `tpt-ctrl-estimation`, `tpt-ctrl-mpc`), where sizes
/// are horizon- or model-dependent.
#[derive(Clone, Debug, PartialEq)]
pub struct DynSystem<T = f64> {
    /// State matrix.
    pub a: DMat<T>,
    /// Input matrix.
    pub b: DMat<T>,
    /// Output matrix.
    pub c: DMat<T>,
    /// Feedthrough matrix.
    pub d: DMat<T>,
}

impl<T: Field> DynSystem<T> {
    /// Builds from the four dynamic matrices, checking shape consistency.
    pub fn new(a: DMat<T>, b: DMat<T>, c: DMat<T>, d: DMat<T>) -> Result<DynSystem<T>> {
        let n = a.nrows();
        let ok = a.is_square()
            && b.nrows() == n
            && c.ncols() == n
            && c.nrows() == d.nrows()
            && b.ncols() == d.ncols();
        if !ok {
            return Err(crate::error::LinalgError::DimensionMismatch {
                expected: (n, n),
                found: (b.nrows(), c.nrows()),
            });
        }
        Ok(DynSystem { a, b, c, d })
    }

    /// Number of states.
    pub fn n_states(&self) -> usize {
        self.a.nrows()
    }

    /// Number of inputs.
    pub fn n_inputs(&self) -> usize {
        self.b.ncols()
    }

    /// Number of outputs.
    pub fn n_outputs(&self) -> usize {
        self.c.nrows()
    }

    /// One discrete propagation step `x+ = A x + B u`.
    pub fn step(&self, x: &DMat<T>, u: &DMat<T>) -> DMat<T> {
        &self.a * x + &self.b * u
    }

    /// Output equation `y = C x + D u`.
    pub fn output(&self, x: &DMat<T>, u: &DMat<T>) -> DMat<T> {
        &self.c * x + &self.d * u
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12
    }

    /// Double integrator: x'' = u.
    fn double_integrator() -> Continuous<f64, 2, 1, 1> {
        Continuous::new(StateSpace::new(
            Mat::from_rows_arr(&[[0.0, 1.0], [0.0, 0.0]]),
            Mat::from_rows_arr(&[[0.0], [1.0]]),
            Mat::from_rows_arr(&[[1.0, 0.0]]),
            Mat::from_rows_arr(&[[0.0]]),
        ))
    }

    #[test]
    fn zoh_double_integrator_closed_form() {
        let dt = 0.1;
        let d = double_integrator().discretize_zoh(dt).unwrap();
        assert!(close(d.ss.a[(0, 0)], 1.0));
        assert!(close(d.ss.a[(0, 1)], dt));
        assert!(close(d.ss.a[(1, 1)], 1.0));
        assert!(close(d.ss.b[(0, 0)], dt * dt / 2.0));
        assert!(close(d.ss.b[(1, 0)], dt));
        assert_eq!(d.dt, dt);
    }

    #[test]
    fn tustin_integrator_matches_bilinear_tf() {
        // G(s) = 1/s discretized: (h/2)(z+1)/(z-1).
        let sys = Continuous::new(StateSpace::new(
            Mat::<f64, 1, 1>::from_rows_arr(&[[0.0]]),
            Mat::<f64, 1, 1>::from_rows_arr(&[[1.0]]),
            Mat::<f64, 1, 1>::from_rows_arr(&[[1.0]]),
            Mat::<f64, 1, 1>::from_rows_arr(&[[0.0]]),
        ));
        let h = 0.05;
        let d = sys.discretize_tustin(h).unwrap();
        assert!(close(d.ss.a[(0, 0)], 1.0));
        assert!(close(d.ss.b[(0, 0)], h));
        assert!(close(d.ss.c[(0, 0)], 1.0));
        assert!(close(d.ss.d[(0, 0)], h / 2.0));
        // TF check: D + C(zI - A)^{-1} B = h/2 + h/(z-1) = (h/2)(z+1)/(z-1).
        let z = 2.3;
        let tf = d.ss.d[(0, 0)] + d.ss.c[(0, 0)] / (z - d.ss.a[(0, 0)]) * d.ss.b[(0, 0)];
        let expected = (h / 2.0) * (z + 1.0) / (z - 1.0);
        assert!(close(tf, expected));
    }

    #[test]
    fn discrete_step_and_output() {
        let dt = 0.1;
        let d = double_integrator().discretize_zoh(dt).unwrap();
        let x0 = Mat::<f64, 2, 1>::from_row_slice(&[0.0, 0.0]);
        let u = Mat::<f64, 1, 1>::from_row_slice(&[2.0]);
        let x1 = d.step(&x0, &u);
        assert!(close(x1[(0, 0)], dt * dt / 2.0 * 2.0));
        assert!(close(x1[(1, 0)], dt * 2.0));
        let y = d.output(&x1, &u);
        assert!(close(y[(0, 0)], x1[(0, 0)]));
    }

    #[test]
    fn dyn_system_shape_checks() {
        let ok = DynSystem::new(
            DMat::<f64>::identity(2),
            DMat::zeros(2, 1),
            DMat::zeros(1, 2),
            DMat::zeros(1, 1),
        );
        assert!(ok.is_ok());
        let bad = DynSystem::new(
            DMat::<f64>::identity(2),
            DMat::zeros(3, 1),
            DMat::zeros(1, 2),
            DMat::zeros(1, 1),
        );
        assert!(bad.is_err());
    }
}
