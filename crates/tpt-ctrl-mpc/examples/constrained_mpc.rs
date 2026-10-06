//! Worked example: constrained MPC on a double integrator.
//!
//! The actuator is bounded to |u| <= 0.3; the MPC regulates a large
//! initial state while never violating the bound, and the closed loop is
//! compared against the (constraint-blind) DLQR baseline.

use tpt_ctrl_core::dmat::DMat;
use tpt_ctrl_mpc::MpcBuilder;
use tpt_ctrl_optimal::dlqr;

fn main() {
    let a = DMat::from_rows(&[&[1.0, 0.1], &[0.0, 1.0]]);
    let b = DMat::from_rows(&[&[0.005], &[0.1]]);
    let q = DMat::identity(2);
    let r = DMat::from_rows(&[&[0.5]]);

    let mpc = MpcBuilder::new(a.clone(), b.clone(), 20)
        .stage_costs(&q, &r)
        .terminal_cost(&(DMat::identity(2) * 10.0))
        .input_bounds(vec![-0.3], vec![0.3])
        .build();

    let dlqr = dlqr(&a, &b, &q, &r).unwrap();

    let x0 = DMat::from_rows(&[&[2.0], &[0.0]]);
    let (mut x_mpc, mut x_lqr) = (x0.clone(), x0.clone());
    let mut max_u_lqr = 0.0_f64;
    for k in 0..60 {
        let sol = mpc.solve(&x_mpc).expect("MPC solve");
        x_mpc = &(&a * &x_mpc) + &(&b * sol.u0[0]);
        if k < 5 {
            println!(
                "t = {:4.1}  u_mpc = {:+.4}  x = [{:+.4}, {:+.4}]",
                k as f64 * 0.1,
                sol.u0[0],
                x_mpc[(0, 0)],
                x_mpc[(1, 0)]
            );
        }
        let u_lqr = -(&dlqr.k * &x_lqr)[(0, 0)];
        max_u_lqr = max_u_lqr.max(u_lqr.abs());
        x_lqr = &(&a * &x_lqr) + &(&b * u_lqr);
    }
    println!(
        "MPC final state:  [{:+.6}, {:+.6}]",
        x_mpc[(0, 0)],
        x_mpc[(1, 0)]
    );
    println!(
        "LQR final state:  [{:+.6}, {:+.6}] (peak |u| = {max_u_lqr:.3}, bound 0.3)",
        x_lqr[(0, 0)],
        x_lqr[(1, 0)]
    );
}
