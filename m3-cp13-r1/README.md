# CP13-R1 qualification on the M3

This folder runs the full CP13 native qualification on an Apple Silicon Mac.
It is the first native ARM run of CP13: NEON dispatch, the DD NEON precision
mode (8) and Clippy on the aarch64-only files.

## Contents

- `Audioscopy_CP13_R1_NATIVE_OPERATOR_KIT_2026-11-27.tar.gz`: the delivered kit,
  unchanged (SHA-256 `daa2631387be36e55790693cc1ad4c584b8c22592a0e761b1b8c2b870037ac6d`).
- `flake/`: every tool the runner needs. Rust 1.99.1 with rustfmt and Clippy,
  Python 3, and the Nix stdenv Clang and `ar`. `flake.lock` is the same lock as
  the CP7/CP13 kits and the audioscopy repository.
- `run-m3.sh`: checks, extraction and the run.

macOS itself supplies only `/usr/bin/time` (BSD time, for peak RSS) and
`sysctl` (CPU identity). The runner requires both at those places.

All aarch64-darwin packages in the flake were checked against cache.nixos.org,
and the Rust 1.99.1 aarch64-darwin downloads against static.rust-lang.org.
Nothing needs to be built from source. Cargo downloads crates from crates.io on
the first build.

## Steps

1. Open a native arm64 terminal (not Rosetta). Connect mains power. Close
   other apps.
2. Extract this package anywhere, for example in `~/audioscopy-qualification/`.
3. Run the checks:

       cd audioscopy-cp13-r1-m3
       ./run-m3.sh --check

   This checks the kit SHA-256, extracts the kit into `kit/`, prints the tool
   versions, compiles a C probe, runs `--verify-only` (expect `MANIFEST 226`,
   `CP7_BASELINE 168`) and the two runner self-tests.
4. Run the qualification:

       ./run-m3.sh

   This is the command from the kit's `OPERATOR.md`:
   `qualify_cp13.py --work ~/audioscopy-qualification/cp13-r1-m3-portable --layout-control --long`,
   inside the flake shell and under `caffeinate`. No `--cpu` (Linux only), no
   `--perf` (Linux only), no `--format-candidate` (the formatting is already in
   the R1 source).

   If the run stops, run the same command again. The runner reuses passed
   stages. Do not change anything between the two runs.

## What to bring back

The whole `return/` folder:

- `cp13-r1-m3-portable-evidence-<timestamp>.tar.gz` and its `.sha256`
- `cp13-r1-m3-portable.log`

Bring it back also if stages fail. A failed gate is evidence.
