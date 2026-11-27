# Local modification of the CP13-R1 kit

This is NOT the kit as delivered. It was derived on 2026-11-27 from
Audioscopy_CP13_R1_NATIVE_OPERATOR_KIT_2026-11-27.tar.gz
(SHA-256 daa2631387be36e55790693cc1ad4c584b8c22592a0e761b1b8c2b870037ac6d)
after the delivered R1 run on native aarch64 (Apple M4) failed only 05-hygiene
(Clippy -D warnings, both crates).

Changes (LOCAL-CLIPPY-FIX.patch):
- true-peak/src/loudness/simd.rs: cfg(target_arch = "x86_64") on
  accumulate_window_scalar (callers are SSE2/AVX only) and on the test helper
  Backend::best_available (callers are x86_64-only tests). No x86_64 change.
  This file is part of the frozen CP7 ordinary DSP baseline.
- audioscopy/src/precision/mod.rs: Backend::available uses
  !matches!(self, Self::X87) || cfg!(audioscopy_x87), same truth table.
- qualification-cp13/qualify_cp13.py: the frozen-baseline check accepts the
  baseline true-peak/src/loudness/simd.rs or exactly the patched file
  (SHA-256 68a8b18645b15cf311a8573089383607fdb767916371a58c9db63eeec168085c), nothing else.
- MANIFEST-SHA256.txt: hashes updated for those three files; this note and the
  patch are added.

Not verified here: Clippy on x86_64 for the changed files.
