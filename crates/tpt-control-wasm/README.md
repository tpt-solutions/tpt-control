# tpt-control-wasm

WebAssembly bindings for the tpt-control playground. The crate exposes one function, `design_lqr`, which designs an LQR (continuous) or DLQR (discrete) gain in the browser and certifies it. The page itself is in [`playground/`](https://github.com/tpt-solutions/tpt-control/tree/master/playground).

This crate is not published (`publish = false`). It exists to build the playground.

## Build and run the playground

You need the `wasm32-unknown-unknown` target and a `wasm-bindgen` CLI matching the crate's pinned version (`=0.2.128`):

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.128
just playground
```

`just playground` writes the JavaScript glue to `playground/pkg/`. Browsers refuse to load WebAssembly from `file://`, so serve the folder over HTTP:

```sh
python -m http.server -d playground 8000
```

Then open <http://localhost:8000>.

## JavaScript API

```js
import init, { design_lqr } from "./pkg/tpt_control_wasm.js";

await init();

// continuous = true, n = 2 states, m = 1 input.
// Matrices are flat, row-major. Pass typed arrays: plain JS arrays are rejected.
const json = design_lqr(
  true, 2, 1,
  new Float64Array([0, 1, 0, 0]),   // A (n x n)
  new Float64Array([0, 1]),         // B (n x m)
  new Float64Array([1, 0, 0, 1]),   // Q (n x n)
  new Float64Array([1]),            // R (m x m)
);
const { k, riccati_residual, closed_loop_margin, certified } = JSON.parse(json);
```

`design_lqr` returns a JSON string with these fields:

| Field | Meaning |
|-------|---------|
| `k` | the gain, flattened row-major (`u = -K x`) |
| `riccati_residual` | relative Riccati residual; certified when at most `1e-8` |
| `closed_loop_margin` | stability margin of the closed loop; must be positive |
| `certified` | `true` when both checks pass |

Bad input, such as a matrix with the wrong number of entries, or a solver failure throws a JavaScript `Error` with a message.

## Tests

```sh
cargo test -p tpt-control-wasm
```

These run on the host, through the plain-Rust `design` function, so they need no browser.

## Status

Pre-1.0 and not published. The JavaScript signature may change. See [CHANGELOG.md](CHANGELOG.md).

Dual-licensed under MIT OR Apache-2.0.
