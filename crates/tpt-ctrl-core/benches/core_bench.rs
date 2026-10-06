//! Benchmarks for the tpt-ctrl-core numerical hot paths.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tpt_ctrl_core::dmat::DMat;

fn bench_group(c: &mut Criterion) {
    let mut group = c.benchmark_group("core");

    let a6 = DMat::from_rows(&[
        &[0.3, 1.2, -0.4, 0.1, 0.5, -0.2],
        &[-0.1, 0.2, 0.3, 0.7, -0.3, 0.1],
        &[0.2, -0.5, 0.1, 0.2, 0.4, 0.3],
        &[0.6, 0.1, -0.2, 0.1, 0.2, -0.4],
        &[-0.3, 0.4, 0.2, -0.1, 0.3, 0.5],
        &[0.1, -0.2, 0.4, 0.3, -0.5, 0.2],
    ]);

    group.bench_function("expm_6x6", |bch| {
        bch.iter(|| tpt_ctrl_core::expm::expm(black_box(&a6)).unwrap())
    });

    group.bench_function("eigenvalues_6x6", |bch| {
        bch.iter(|| black_box(&a6).eigenvalues().unwrap())
    });

    group.bench_function("solve_20x20", |bch| {
        // Well-conditioned system: A = I + 0.01 * M.
        let m = DMat::from_fn(20, 20, |i, j| ((i * 7 + j * 13) % 21) as f64 / 21.0 - 0.5);
        let a = &DMat::identity(20) + &(&m * 0.02);
        let rhs = DMat::from_fn(20, 1, |i, _| (i % 5) as f64);
        bch.iter(|| black_box(&a).solve(black_box(&rhs)).unwrap())
    });

    group.bench_function("lyapunov_6x6", |bch| {
        let q = DMat::identity(6);
        bch.iter(|| black_box(&a6).solve_lyapunov(black_box(&q)).unwrap())
    });

    group.finish();
}

criterion_group!(benches, bench_group);
criterion_main!(benches);
