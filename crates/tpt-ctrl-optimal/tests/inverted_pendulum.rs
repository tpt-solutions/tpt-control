//! Integration test (todo.md Phase 1): LQR on an inverted pendulum,
//! verified via proptest for closed-loop stability.

use proptest::prelude::*;
use tpt_ctrl_core::analysis::is_controllable;
use tpt_ctrl_core::dmat::DMat;
use tpt_ctrl_optimal::lqr;

/// Linearized cart-pole about the upright equilibrium.
///
/// States: `[cart position, cart velocity, pole angle, pole angular rate]`,
/// input: horizontal force on the cart. Classic massless-pole model.
fn cart_pole(m: f64, m_cart: f64, l: f64, g: f64) -> (DMat<f64>, DMat<f64>) {
    let a = DMat::from_rows(&[
        &[0.0, 1.0, 0.0, 0.0],
        &[0.0, 0.0, -(m * g) / m_cart, 0.0],
        &[0.0, 0.0, 0.0, 1.0],
        &[0.0, 0.0, ((m_cart + m) * g) / (m_cart * l), 0.0],
    ]);
    let b = DMat::from_rows(&[&[0.0], &[1.0 / m_cart], &[0.0], &[-1.0 / (m_cart * l)]]);
    (a, b)
}

fn closed_loop_simulation(a: &DMat<f64>, k: &DMat<f64>, x0: &[f64], t_end: f64) -> Vec<f64> {
    // x' = (A - B K) x via RK4; returns the trajectory of the pole angle.
    let cl = &(a - &{
        let b = DMat::from_rows(&[&[0.0], &[1.0], &[0.0], &[-2.0]]);
        b * k
    });
    let dynamics = |_t: f64, x: &[f64], _u: &[f64], out: &mut [f64]| {
        for i in 0..4 {
            let mut acc = 0.0;
            for j in 0..4 {
                acc += cl[(i, j)] * x[j];
            }
            out[i] = acc;
        }
    };
    let dt = 0.01;
    let steps = (t_end / dt) as usize;
    let mut x = x0.to_vec();
    let mut angles = Vec::with_capacity(steps + 1);
    angles.push(x[2]);
    let mut t = 0.0;
    let u: [f64; 0] = [];
    for _ in 0..steps {
        tpt_ctrl_core::simulate::rk4_step(&dynamics, t, dt, &mut x, &u);
        t += dt;
        angles.push(x[2]);
    }
    angles
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    #[test]
    fn lqr_regulates_inverted_pendulum(
        theta0 in -0.35f64..0.35,
        q_state in 0.5f64..20.0,
        q_angle in 1.0f64..50.0,
        r in 0.5f64..5.0,
    ) {
        let (a, b) = cart_pole(0.1, 1.0, 0.5, 9.8);
        prop_assert!(is_controllable(&a, &b, 1e-9).unwrap());

        let q = DMat::diagonal(&[q_state, q_state, q_angle, q_state]);
        let r = DMat::from_rows(&[&[r]]);
        let sol = lqr(&a, &b, &q, &r).unwrap();

        // Closed loop must be Hurwitz.
        let cl = &a - &(&b * &sol.k);
        prop_assert!(tpt_ctrl_core::analysis::is_stable_continuous(&cl, 1e-9).unwrap());

        // Riccati solution positive definite (strictly, for detectable Q).
        prop_assert!(sol.x.is_positive_definite());

        // Regulate from a perturbed upright angle: linear dynamics.
        let angles = closed_loop_simulation(&a, &sol.k, &[0.0, 0.0, theta0, 0.0], 12.0);
        let last = *angles.last().unwrap();
        let peak = angles.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        prop_assert!(last.abs() < 1e-3, "final angle {last}");
        prop_assert!(peak < 0.5, "divergent transient peak {peak}");
    }
}

#[test]
fn lqr_stabilizes_reference_case() {
    let (a, b) = cart_pole(0.1, 1.0, 0.5, 9.8);
    let q = DMat::diagonal(&[1.0, 1.0, 10.0, 1.0]);
    let r = DMat::from_rows(&[&[1.0]]);
    let sol = lqr(&a, &b, &q, &r).unwrap();
    let cl = &a - &(&b * &sol.k);
    assert!(tpt_ctrl_core::analysis::is_stable_continuous(&cl, 1e-9).unwrap());
    // Cost on a tilted start is finite and positive.
    let x0 = DMat::from_rows(&[&[0.0], &[0.0], &[0.2], &[0.0]]);
    let v = &(&x0.transpose() * &sol.x) * &x0;
    assert!(v[(0, 0)] > 0.0);
    // Angle actually regulated.
    let angles = closed_loop_simulation(&a, &sol.k, &[0.0, 0.0, 0.2, 0.0], 12.0);
    assert!(angles.last().unwrap().abs() < 1e-3);
}
