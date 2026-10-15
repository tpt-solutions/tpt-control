//! Worked example: stability margins and lead compensation.
//!
//! Computes gain and phase margins for a plant, then adds a lead
//! compensator and shows how the phase margin improves.

use tpt_ctrl_classic::freq::margins;
use tpt_ctrl_classic::leadlag::lead;
use tpt_ctrl_classic::tf::TransferFunction;

fn report(label: &str, g: &TransferFunction) {
    match margins(g, 1e-3, 1e3, 4000) {
        Some(m) => println!(
            "{label:<28} phase margin = {:6.2} deg   gain margin = {:8.3}",
            m.phase_margin_deg, m.gain_margin
        ),
        None => println!("{label:<28} margins not found on the grid"),
    }
}

fn main() {
    // Plant: G(s) = 4 / (s (s + 1)(s/5 + 1))   (lightly damped loop).
    let plant = TransferFunction::new(&[4.0], &[0.2, 1.2, 1.0, 0.0]).unwrap();
    report("plant G(s)", &plant);

    // Textbook check: 1 / (s (s + 1)) has phase margin ~ 51.8 deg.
    let reference = TransferFunction::new(&[1.0], &[1.0, 1.0, 0.0]).unwrap();
    report("reference 1/(s(s+1))", &reference);

    // Lead compensator with alpha = 0.2 (max phase lead near 1/(T sqrt(alpha))).
    let comp = lead(1.0, 0.2).unwrap();
    let compensated = plant.series(&comp);
    report("G(s) with lead", &compensated);

    println!("\nDC gain of compensator (expect 1): {:.3}", comp.dc_gain());
    println!("closed-loop poles of compensated loop:");
    for p in compensated.closed_loop().poles() {
        println!("  {:+.4} {:+.4}i", p.re, p.im);
    }
}
