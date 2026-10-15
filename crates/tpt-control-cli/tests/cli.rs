use std::path::PathBuf;
use std::process::Command;

fn plant(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("plants")
        .join(name)
}

fn tpt(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_tpt"))
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn certifies_continuous_plant() {
    let out = tpt(&["lqr", plant("double_integrator.toml").to_str().unwrap()]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(0), "{stdout}");
    assert!(stdout.contains("CERTIFIED"));
}

#[test]
fn certifies_discrete_plant() {
    let out = tpt(&[
        "lqr",
        plant("sampled_double_integrator.toml").to_str().unwrap(),
    ]);
    assert_eq!(out.status.code(), Some(0));
}

#[test]
fn unstabilizable_plant_is_refused_with_exit_status_2() {
    let out = tpt(&["lqr", plant("unstable_unobservable.toml").to_str().unwrap()]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "{stderr}");
    assert!(stderr.contains("not stabilizable"), "{stderr}");
}

#[test]
fn bad_usage_exits_2() {
    assert_eq!(tpt(&[]).status.code(), Some(2));
    assert_eq!(
        tpt(&["lqr", "/nonexistent/plant.toml"]).status.code(),
        Some(2)
    );
}

#[test]
fn help_exits_0() {
    assert_eq!(tpt(&["--help"]).status.code(), Some(0));
}
