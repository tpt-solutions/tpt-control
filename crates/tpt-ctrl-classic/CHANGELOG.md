# Changelog

All notable changes to `tpt-ctrl-classic` are documented here.
This project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Runnable examples `pid_step_response` (PI/PID tracking and anti-windup under saturation) and `bode_margins` (margins before and after lead compensation).

## [0.1.0] — 2026-10-06

### Added
- Rational transfer functions with companion-matrix poles and zeros, series / parallel / closed-loop algebra, and controllable-canonical realization.
- PID with back-calculation anti-windup, derivative-on-measurement and derivative filtering.
- Lead, lag and notch compensator constructors (unity DC gain).
- Frequency response, Bode and Nyquist data, and gain and phase margins on a logarithmic grid with crossover interpolation.

[0.1.0]: https://github.com/tpt-solutions/tpt-control/releases/tag/v0.1.0
