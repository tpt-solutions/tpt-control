# tpt-control — developer task runner.
#
# Run `just` to list recipes. Common entry points:
#   just ci        — the full local gate CI runs (fmt, build, test, clippy, deny, doc)
#   just test      — run the whole workspace test suite (all features)
#   just no-std    — build tpt-ctrl-core for a bare-metal target
#   just kani      — run the Kani proof harnesses (requires kani; Linux-only)

# Default recipe: list everything.
default:
    @just --list

# --- Lint / format --------------------------------------------------------

# Check formatting without modifying files (CI uses this).
fmt:
    cargo fmt --all -- --check

# Auto-format the workspace.
fmt-fix:
    cargo fmt --all

# --- Build / test / verify ------------------------------------------------

# Build every crate with all features.
build:
    cargo build --workspace --all-features

# Run the full workspace test suite (all features).
test:
    cargo test --workspace --all-features

# Clippy across all targets, denying warnings.
clippy:
    cargo clippy --workspace --all-targets --all-features -- -D warnings

# License / advisory / source audit (default features).
deny:
    cargo deny check

# Build API docs.
doc:
    cargo doc --workspace --no-deps

# Build tpt-ctrl-core for a bare-metal Cortex-M target (no_std + alloc).
no-std:
    cargo build -p tpt-ctrl-core --target thumbv6m-none-eabi --no-default-features --features alloc

# Run Kani bounded model-checking harnesses (Linux only; see .github/workflows/kani.yml).
kani:
    cargo kani -p tpt-ctrl-verify --harness verify

# --- Meta -----------------------------------------------------------------

# Everything CI runs, in CI order.
ci: fmt build test clippy deny doc
