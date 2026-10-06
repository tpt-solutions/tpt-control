use tpt_ctrl_core::dmat::DMat;
use tpt_ctrl_optimal::dare::dare;
fn main() {
    // dual DARE case
    let a = DMat::from_rows(&[&[0.9, 0.4], &[-0.2, 0.7]]);
    let c = DMat::from_rows(&[&[1.0, 0.0]]);
    let qn = DMat::identity(2);
    let rn = DMat::from_rows(&[&[0.5]]);
    let sol = dare(&a.transpose(), &c.transpose(), &qn, &rn).unwrap();
    println!("P = {:?}", sol.x);
    println!("K(=L^T-ish) = {:?}", sol.k);
    let l = &(&sol.x * &c.transpose()) * &rn.inverse().unwrap();
    println!("L = {:?}", l);
    let err_dyn = &(&(&DMat::identity(2) - &(&l * &c)) * &a);
    println!("err eig = {:?}", err_dyn.eigenvalues().unwrap());

    // lqg LQR half
    let a2 = DMat::from_rows(&[&[1.0, 0.1], &[0.0, 1.0]]);
    let b2 = DMat::from_rows(&[&[0.005], &[0.1]]);
    let q = DMat::identity(2);
    let r = DMat::from_rows(&[&[1.0]]);
    let sol2 = dare(&a2, &b2, &q, &r).unwrap();
    println!("X2 = {:?}", sol2.x);
    println!("K2 = {:?}", sol2.k);
    let cl = &a2 - &(&b2 * &sol2.k);
    println!("cl eig = {:?}", cl.eigenvalues().unwrap());
    println!(
        "cl is stable: {:?}",
        tpt_ctrl_core::analysis::is_stable_discrete(&cl, 1e-9)
    );
}
