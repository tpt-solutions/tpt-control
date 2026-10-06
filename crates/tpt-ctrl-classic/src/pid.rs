//! PID control with back-calculation anti-windup.
//!
//! The derivative acts on the *measurement* (not the error) so setpoint
//! steps do not produce derivative kicks, and an optional first-order
//! filter limits derivative noise amplification.
//!
//! Note (todo.md Phase 1 checkpoint): `tpt-dsp-control` hosts an
//! embedded/motor-control PID. This crate's PID is deliberately scoped to
//! the state-space-integrated control-theory side (error-in/error-out with
//! configurable saturation and anti-windup, composable with the transfer
//! functions and margins here); the two are not merged to avoid coupling
//! the audio-real-time `tpt-dsp` release cycle to control-theory changes.

/// Tuning and limits for a PID controller.
#[derive(Debug, Clone, PartialEq)]
pub struct PidGains {
    /// Proportional gain.
    pub kp: f64,
    /// Integral gain (units of output per unit-error per second).
    pub ki: f64,
    /// Derivative gain.
    pub kd: f64,
    /// Anti-windup back-calculation gain `[1/s]`; 0 disables anti-windup,
    /// typical values are `1 / T_i` to `sqrt(ki / kd)`.
    pub kaw: f64,
    /// Derivative low-pass filter time constant `[s]` (0 = unfiltered).
    pub d_filter_tau: f64,
}

impl PidGains {
    /// Pure proportional.
    pub fn p(kp: f64) -> PidGains {
        PidGains {
            kp,
            ki: 0.0,
            kd: 0.0,
            kaw: 0.0,
            d_filter_tau: 0.0,
        }
    }

    /// PI with anti-windup back-calculation at `1 / (kp / ki)` style tuning.
    pub fn pi(kp: f64, ki: f64, kaw: f64) -> PidGains {
        PidGains {
            kp,
            ki,
            kd: 0.0,
            kaw,
            d_filter_tau: 0.0,
        }
    }

    /// Full PID.
    pub fn pid(kp: f64, ki: f64, kd: f64, kaw: f64, d_filter_tau: f64) -> PidGains {
        PidGains {
            kp,
            ki,
            kd,
            kaw,
            d_filter_tau,
        }
    }
}

/// Stateful PID controller with output limits.
#[derive(Debug, Clone)]
pub struct Pid {
    gains: PidGains,
    out_min: f64,
    out_max: f64,
    integral: f64,
    prev_measurement: f64,
    d_state: f64,
    initialized: bool,
}

impl Pid {
    /// Creates a controller with output clamped to `[out_min, out_max]`.
    pub fn new(gains: PidGains, out_min: f64, out_max: f64) -> Pid {
        assert!(out_min < out_max, "pid: out_min must be below out_max");
        Pid {
            gains,
            out_min,
            out_max,
            integral: 0.0,
            prev_measurement: 0.0,
            d_state: 0.0,
            initialized: false,
        }
    }

    /// Clears all internal state.
    pub fn reset(&mut self) {
        self.integral = 0.0;
        self.prev_measurement = 0.0;
        self.d_state = 0.0;
        self.initialized = false;
    }

    /// Current integrator value (for diagnostics / tests).
    pub fn integral(&self) -> f64 {
        self.integral
    }

    /// One control step: returns the commanded output for
    /// error `setpoint - measurement` at sample time `dt`.
    pub fn step(&mut self, setpoint: f64, measurement: f64, dt: f64) -> f64 {
        assert!(dt > 0.0, "pid: dt must be positive");
        let error = setpoint - measurement;

        // Derivative on measurement with optional first-order filter.
        let d_term = if self.gains.kd != 0.0 {
            let raw = if self.initialized {
                -(measurement - self.prev_measurement) / dt
            } else {
                0.0
            };
            let alpha = if self.gains.d_filter_tau > 0.0 {
                dt / (self.gains.d_filter_tau + dt)
            } else {
                1.0
            };
            self.d_state += alpha * (raw - self.d_state);
            self.gains.kd * self.d_state
        } else {
            0.0
        };
        self.prev_measurement = measurement;
        self.initialized = true;

        // Raw (unsaturated) output with the current integral estimate.
        let u_unsat = self.gains.kp * error + self.integral + d_term;
        let u = u_unsat.clamp(self.out_min, self.out_max);

        // Integrator update with back-calculation anti-windup: feed the
        // saturation mismatch back into the integrator so it unwinds.
        self.integral +=
            dt * (self.gains.ki * error + self.gains.kaw * (u - u_unsat) * self.gains.kp);

        u
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simulate(gains: PidGains, setpoint: f64, steps: usize, dt: f64) -> Vec<f64> {
        let mut pid = Pid::new(gains, 0.0, 5.0);
        let mut y = 0.0;
        let mut out = Vec::with_capacity(steps);
        for _ in 0..steps {
            let u = pid.step(setpoint, y, dt);
            // First-order plant 1/(s + 1).
            y += dt * (u - y);
            out.push(y);
        }
        out
    }

    #[test]
    fn pi_tracks_setpoint() {
        let gains = PidGains::pi(2.0, 4.0, 2.0);
        let resp = simulate(gains, 1.0, 4000, 0.01);
        let last = *resp.last().unwrap();
        assert!((last - 1.0).abs() < 1e-3, "final y = {last}");
    }

    #[test]
    fn anti_windup_limits_integrator_windup() {
        // Saturating setpoint step: with back-calculation the integrator
        // stays near the saturation boundary; without it, it winds up.
        let with_aw = PidGains::pid(1.0, 8.0, 0.0, 8.0, 0.0);
        let without_aw = PidGains {
            kaw: 0.0,
            ..with_aw.clone()
        };
        let mut pid_on = Pid::new(with_aw, 0.0, 5.0);
        let mut pid_off = Pid::new(without_aw, 0.0, 5.0);
        let mut y = 0.0;
        for _ in 0..150 {
            let u_on = pid_on.step(3.0, y, 0.01);
            let u_off = pid_off.step(3.0, y, 0.01);
            // Same plant driven by the saturated outputs: the windup test
            // is about the controller state, so track both plants.
            let y_on = y + 0.01 * (u_on - y);
            let y_off = y + 0.01 * (u_off - y);
            y = y_on; // drive both from the AW plant for a controlled comparison
            let _ = y_off;
        }
        assert!(
            pid_on.integral() < pid_off.integral(),
            "integrator with AW ({}) should stay below unwound ({})",
            pid_on.integral(),
            pid_off.integral()
        );
        // And the AW integrator stays bounded near the actuator limit.
        assert!(pid_on.integral() < 10.0, "integrator {}", pid_on.integral());
    }

    #[test]
    fn derivative_on_measurement_no_setpoint_kick() {
        // At the first step after a setpoint change, the output must not
        // include a large derivative spike.
        let mut pid = Pid::new(PidGains::pid(1.0, 0.0, 10.0, 0.0, 0.0), -100.0, 100.0);
        let u0 = pid.step(1.0, 0.0, 0.01);
        assert!(
            u0.abs() < 2.0,
            "first-step output {u0} shows derivative kick"
        );
    }

    #[test]
    fn reset_clears_state() {
        let mut pid = Pid::new(PidGains::pi(1.0, 1.0, 0.0), 0.0, 1.0);
        for _ in 0..100 {
            pid.step(1.0, 0.0, 0.01);
        }
        assert!(pid.integral() > 0.0);
        pid.reset();
        assert_eq!(pid.integral(), 0.0);
    }
}
