# Changelog

All notable changes to `tpt-ctrl-optimal` are documented here.
This project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `kalman_gain_continuous`: steady-state Kalman gain for continuous-time models, from the dual CARE.
- Runnable example `lqg_regulator`.

### Fixed
- `lqg` (continuous-time) computed its observer gain from the discrete DARE, which is wrong for a continuous plant and could produce an unstable observer. It now uses `kalman_gain_continuous`. `dlqg` and `kalman_gain` are unchanged.

## [0.1.0] — 2026-10-06

### Added
- DARE via the structure-preserving doubling algorithm.
- CARE via Newton–Kleinman, initialized from the DLQR gain of a fine ZOH discretization.
- LQR/DLQR, steady-state Kalman gain (dual DARE), LQG and discrete LQG.

[0.1.0]: https://github.com/tpt-solutions/tpt-control/releases/tag/v0.1.0
