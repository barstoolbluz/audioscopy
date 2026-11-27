# CP13-R1 native aarch64 evidence

Two full CP13 qualification runs on 2026-11-27, both on the same Mac, with the
flake in `../flake/` and the command from the kit's `OPERATOR.md`:

    qualify_cp13.py --work <new dir> --layout-control --long

The Mac is an Apple M4 (Mac mini), not an M3. The runs were native arm64, not
Rosetta, on mains power, with rustc host `aarch64-apple-darwin`. The work
directories and evidence names still say `m3-portable`.

| Folder | Kit | Result | Evidence SHA-256 |
| --- | --- | --- | --- |
| `r1-as-delivered/` | R1 as delivered (`daa26313…`), MANIFEST 226 | 53 PASS, 1 FAIL (`05-hygiene`), exit 1 | `b6c68cdf…3d2866` |
| `local-clippy-fix/` | R1 plus a local Clippy fix, MANIFEST 228 | 54 PASS, 0 FAIL, `status.json` PASS | `6ec491d4…f12f0` |

## Rebuilding the tarballs

Each evidence tarball is about 105.9 MB, which is over GitHub's per-file limit.
It is split into three parts with `split -b 45m`. To rebuild them and check
their hashes:

    cd r1-as-delivered
    cat cp13-r1-m3-portable-evidence-20261127-133550-37908.tar.gz.part-* \
      > cp13-r1-m3-portable-evidence-20261127-133550-37908.tar.gz
    shasum -a 256 -c cp13-r1-m3-portable-evidence-20261127-133550-37908.tar.gz.sha256

    cd ../local-clippy-fix
    cat cp13-r1-local-clippy-m3-portable-evidence-20261127-135627-68019.tar.gz.part-* \
      > cp13-r1-local-clippy-m3-portable-evidence-20261127-135627-68019.tar.gz
    shasum -a 256 -c cp13-r1-local-clippy-m3-portable-evidence-20261127-135627-68019.tar.gz.sha256

On Linux, use `sha256sum -c` instead of `shasum -a 256 -c`.

## Run 1: R1 as delivered

The only failure is `05-hygiene`:
`cargo clippy --all-targets --all-features -- -D warnings` fails on aarch64 for
both crates, because x86-only code becomes dead or trivial there.

- true-peak `src/loudness/simd.rs:383`: `accumulate_window_scalar` is never
  used. Its only callers are the SSE2/AVX paths.
- true-peak `src/loudness/simd.rs:91`: `Backend::best_available` (`cfg(test)`)
  is never used. Its only callers are x86_64-only tests.
- audioscopy `src/precision/mod.rs:55`: `clippy::match_like_matches_macro`.
  `cfg!(audioscopy_x87)` is always false on aarch64.

Every other stage passed, including the NEON dispatch, DD NEON mode 8 parity
and the `--long` cohorts.

## Run 2: the local Clippy fix

This kit was modified locally. It is not R1 as delivered, and the evidence
identity records the modified manifest (`989d6f55…`). The files that describe
the change are:

- `LOCAL-CLIPPY-FIX.patch`: the source fix. It adds `cfg(target_arch = "x86_64")`
  to the two true-peak items and changes `Backend::available` to
  `!matches!(self, Self::X87) || cfg!(audioscopy_x87)`, which returns the same
  results.
- `kit-r1-to-local.diff`: the complete diff from the delivered kit. Besides the
  source fix, it covers the runner change, the manifest and the note. Because
  `simd.rs` belongs to the frozen CP7 baseline, `qualify_cp13.py` was changed to
  accept either the baseline `simd.rs` or the exact patched file (SHA-256
  `68a8b186…`), and nothing else.
- `LOCAL-MODIFICATION.md`: the note that was shipped inside the modified kit.

The log's last line, `CP13_EXIT=`, is empty because the operator's wrapper did
not capture the exit code under zsh. The evidence archive's `status.json`
reads `{"failed": [], "status": "PASS"}`.

## Open points

- The fix changes a frozen CP7 file, so it should come back as a proper
  corrective round (R2) from the kit author.
- Clippy has not been run on x86_64 for the changed files.
