//! `tpt-ctrl-classic` — classical (frequency-domain, SISO) control tools.
//!
//! * [`tf::TransferFunction`] — rational transfer-function algebra with
//!   poles/zeros via the core real-Schur eigenvalue solver (companion
//!   matrices, no external polynomial library).
//! * [`pid::Pid`] — PID with back-calculation anti-windup and derivative
//!   filtering.
//! * [`leadlag`] — lead / lag compensator constructors (unity DC gain).
//! * [`freq`] — frequency response, Bode/Nyquist data, gain and phase
//!   margins.
//!
//! # Example
//!
//! ```
//! use tpt_ctrl_classic::freq::margins;
//! use tpt_ctrl_classic::tf::TransferFunction;
//!
//! // G(s) = 1 / (s (s + 1))
//! let g = TransferFunction::new(&[1.0], &[1.0, 1.0, 0.0]).unwrap();
//! let m = margins(&g, 1e-3, 1e3, 4000).unwrap();
//! // Classic textbook result: phase margin ~= 51.8 degrees, GM infinite.
//! assert!((m.phase_margin_deg - 51.8).abs() < 0.5, "{}", m.phase_margin_deg);
//! assert!(m.gain_margin.is_infinite());
//! ```

#![forbid(unsafe_code)]

pub mod freq;
pub mod leadlag;
pub mod pid;
pub mod tf;
