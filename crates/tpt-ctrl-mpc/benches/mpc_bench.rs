//! Benchmarks for the QP solver and a full MPC step.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tpt_ctrl_core::dmat::DMat;
use tpt_ctrl_mpc::{qp, MpcBuilder};

fn bench_group(c: &mut Criterion) {
    let mut group = c.benchmark_group("mpc");

    // Small QP with box constraints on a 2-variable problem.
    let g = DMat::from_rows(&[&[4.0, 1.0], &[1.0, 3.0]]);
    let gt = DMat::from_rows(&[&[-1.0], &[-2.0]]);
    let c = DMat::from_rows(&[&[-1.0, 0.0], &[0.0, -1.0], &[1.0, 0.0], &[0.0, 1.0]]);
    let bvec = [-1.5, -0.5, 0.5, 0.2];
    group.bench_function("qp_2d_4c", |bch| {
        bch.iter(|| {
            qp::solve(
                black_box(&g),
                black_box(&gt),
                black_box(&c),
                black_box(&bvec),
            )
            .unwrap()
        })
    });

    // Full receding-horizon MPC step (double integrator, N = 15).
    let a = DMat::from_rows(&[&[1.0, 0.1], &[0.0, 1.0]]);
    let bmat = DMat::from_rows(&[&[0.005], &[0.1]]);
    let mpc = MpcBuilder::new(a.clone(), bmat.clone(), 15)
        .stage_costs(&DMat::identity(2), &DMat::from_rows(&[&[0.5]]))
        .terminal_cost(&(DMat::identity(2) * 5.0))
        .input_bounds(vec![-0.3], vec![0.3])
        .build();
    let x0 = DMat::from_rows(&[&[1.5], &[0.2]]);
    group.bench_function("mpc_step_n15", |bch| {
        bch.iter(|| black_box(&mpc).solve(black_box(&x0)).unwrap())
    });

    group.finish();
}

criterion_group!(benches, bench_group);
criterion_main!(benches);
