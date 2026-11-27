//! Build explicitly requested native precision kernels.
use std::{env, path::PathBuf, process::Command};

fn run(command: &mut Command) {
    let status = command.status().expect("start native precision tool");
    assert!(
        status.success(),
        "native precision tool failed: {command:?}"
    );
}

fn build_correlation() {
    println!("cargo:rerun-if-changed=native/correlation_x87.c");
    println!("cargo:rerun-if-env-changed=CC");
    println!("cargo:rerun-if-env-changed=AR");
    println!("cargo:rustc-check-cfg=cfg(audioscopy_x87)");
    if env::var_os("CARGO_FEATURE_PRECISION_X87").is_none() {
        return;
    }
    let target = env::var("TARGET").unwrap();
    // Initially qualified ABI only. Explicit X87 requests on other targets
    // fail; Extended80/Auto can select the portable double-double backend.
    if target != "x86_64-unknown-linux-gnu" || env::var("HOST").unwrap() != target {
        return;
    }
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let object = out.join("correlation_x87.o");
    let archive = out.join("libaudioscopy_x87.a");
    let cc = env::var_os("CC").unwrap_or_else(|| "cc".into());
    let ar = env::var_os("AR").unwrap_or_else(|| "ar".into());
    run(Command::new(cc)
        .args([
            "-std=c11",
            "-O3",
            "-fPIC",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-frounding-math",
            "-fno-lto",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-c",
            "native/correlation_x87.c",
            "-o",
        ])
        .arg(&object));
    run(Command::new(ar).arg("crs").arg(&archive).arg(&object));
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=audioscopy_x87");
    println!("cargo:rustc-link-lib=m");
    println!("cargo:rustc-cfg=audioscopy_x87");
}

fn build_wide() {
    for file in [
        "spectral_precision.c",
        "spectral_precision.h",
        "spectral_fft.h",
        "spectral_cross.h",
        "spectral_noise.h",
        "noise_constants.h",
        "loudness_wide.h",
        "loudness_constants.h",
        "spectral_dd.h",
        "spectral_scalar.h",
        "spectral_constants.h",
        "spectral_vector.h",
    ] {
        println!("cargo:rerun-if-changed=native/{file}");
    }
    println!("cargo:rustc-check-cfg=cfg(audioscopy_wide_spectrum)");
    println!("cargo:rustc-check-cfg=cfg(audioscopy_wide_loudness)");
    let spectrum = env::var_os("CARGO_FEATURE_PRECISION_SPECTRUM").is_some();
    let loudness = env::var_os("CARGO_FEATURE_PRECISION_LOUDNESS").is_some();
    if !spectrum && !loudness {
        return;
    }
    let target = env::var("TARGET").unwrap();
    // The optional C11 backend supports native GNU/Linux and Apple builds.
    // Other targets keep the ordinary analyzer and CP7 portable correlation.
    if env::var("HOST").unwrap() != target
        || !matches!(
            target.as_str(),
            "x86_64-unknown-linux-gnu"
                | "aarch64-unknown-linux-gnu"
                | "x86_64-apple-darwin"
                | "aarch64-apple-darwin"
        )
    {
        return;
    }
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let object = out.join("spectral_precision.o");
    let archive = out.join("libaudioscopy_spectral_precision.a");
    let cc = env::var_os("CC").unwrap_or_else(|| "cc".into());
    let ar = env::var_os("AR").unwrap_or_else(|| "ar".into());
    let mut command = Command::new(cc);
    command.args([
        "-std=c11",
        "-O3",
        "-fPIC",
        "-fno-fast-math",
        "-ffp-contract=off",
        "-frounding-math",
        "-fno-lto",
        "-Wall",
        "-Wextra",
        "-Werror",
    ]);
    if target == "x86_64-unknown-linux-gnu" && env::var_os("CARGO_FEATURE_PRECISION_X87").is_some()
    {
        command.arg("-DAUDIOSCOPY_ENABLE_X87");
    }
    run(command
        .args(["-c", "native/spectral_precision.c", "-o"])
        .arg(&object));
    run(Command::new(ar).arg("crs").arg(&archive).arg(&object));
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=audioscopy_spectral_precision");
    println!("cargo:rustc-link-lib=m");
    if spectrum {
        println!("cargo:rustc-cfg=audioscopy_wide_spectrum");
    }
    if loudness {
        println!("cargo:rustc-cfg=audioscopy_wide_loudness");
    }
}

fn main() {
    build_correlation();
    build_wide();
}
