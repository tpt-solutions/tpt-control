# Changelog

All notable changes to `tpt-ctrl-mpc` are documented here.
This project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Crate metadata: README with a quick start, and keywords and categories suited to crates.io.

## [0.1.0] — 2026-10-06

### Added
- Strictly convex active-set QP solver (exact KKT sub-solves, add-before-drop pivoting, dependency fallback).
- Horizon condensing (`Sx`, `Su`), condensed MPC with input bounds and linear state constraints, receding-horizon controller.
- Explicit-MPC nearest-neighbour lookup tables.

[0.1.0]: https://github.com/tpt-solutions/tpt-control/releases/tag/v0.1.0
