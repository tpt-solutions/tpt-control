//! Benchmarks for the Riccati solvers.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tpt_ctrl_core::dmat::DMat;
use tpt_ctrl_optimal::{dare, lqr};

fn bench_group(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimal");

    let a = DMat::from_rows(&[&[1.0, 0.1], &[0.0, 1.0]]);
    let b = DMat::from_rows(&[&[0.005], &[0.1]]);
    let q = DMat::identity(2);
    let r = DMat::from_rows(&[&[1.0]]);

    group.bench_function("dare_2x2", |bch| {
        bch.iter(|| dare(black_box(&a), black_box(&b), black_box(&q), black_box(&r)).unwrap())
    });

    group.bench_function("lqr_2x2", |bch| {
        bch.iter(|| lqr(black_box(&a), black_box(&b), black_box(&q), black_box(&r)).unwrap())
    });

    group.finish();
}

criterion_group!(benches, bench_group);
criterion_main!(benches);
