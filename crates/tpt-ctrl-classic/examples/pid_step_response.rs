//! Worked example: tuning a PI and a PID controller on a first-order plant.
//!
//! The plant is `tau * y' = -y + K u` with `K = 2`, `tau = 1 s`, integrated
//! with forward Euler. Each controller tracks a unit setpoint step; the
//! output limits show how back-calculation anti-windup keeps the integrator
//! bounded when the actuator saturates.

use tpt_ctrl_classic::pid::{Pid, PidGains};

/// Simulates the closed loop for `t_end` seconds and returns the final
/// output, the peak overshoot, and the peak integrator value.
fn run(label: &str, gains: PidGains, limits: (f64, f64)) {
    let (k_plant, tau) = (2.0_f64, 1.0_f64);
    let dt = 0.01_f64;
    let mut pid = Pid::new(gains, limits.0, limits.1);
    let (mut y, mut peak_y, mut peak_i) = (0.0_f64, 0.0_f64, 0.0_f64);

    for _ in 0..1000 {
        let u = pid.step(1.0, y, dt);
        y += dt * (-y + k_plant * u) / tau;
        peak_y = peak_y.max(y);
        peak_i = peak_i.max(pid.integral().abs());
    }
    println!(
        "{label:<22} y(10s) = {y:.4}   overshoot = {:5.1}%   peak |I| = {peak_i:.3}",
        (peak_y - 1.0).max(0.0) * 100.0
    );
}

fn main() {
    // Unconstrained actuator (wide limits).
    run("P only (kp = 1)", PidGains::p(1.0), (-100.0, 100.0));
    run(
        "PI (kp=1, ki=1)",
        PidGains::pi(1.0, 1.0, 0.0),
        (-100.0, 100.0),
    );
    run(
        "PID (kp=1,ki=1,kd=0.1)",
        PidGains::pid(1.0, 1.0, 0.1, 0.0, 0.05),
        (-100.0, 100.0),
    );

    println!();
    // Tight actuator limits: compare with and without anti-windup.
    run(
        "PI, |u| <= 0.5, no aw",
        PidGains::pi(3.0, 2.0, 0.0),
        (-0.5, 0.5),
    );
    run(
        "PI, |u| <= 0.5, aw",
        PidGains::pi(3.0, 2.0, 1.0),
        (-0.5, 0.5),
    );
}
