# Changelog

All notable changes to `tpt-ctrl-verify` are documented here.
This project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Runnable example `lyapunov_contract`: an LQR Riccati solution as a Lyapunov function, checked over a sampled box.

## [0.1.0] — 2026-10-06

### Added
- proptest strategies: SPD matrices, Hurwitz and Schur matrices, LTI systems, bounded noise.
- Lyapunov contract checking over sampled boxes.
- Kani bounded-model-checking harnesses (LU, Cholesky, ZOH, unrolling, DARE, QP) under `#[cfg(kani)]`.

[0.1.0]: https://github.com/tpt-solutions/tpt-control/releases/tag/v0.1.0
