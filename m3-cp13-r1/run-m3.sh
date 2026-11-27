#!/usr/bin/env bash
# Run the CP13-R1 native qualification on an Apple Silicon Mac (M3).
#
#   ./run-m3.sh --check   tool and kit checks only (a few minutes, first time
#                         longer while Nix fetches the toolchain)
#   ./run-m3.sh           checks, then the full qualification run
#
# Every tool comes from ./flake except /usr/bin/time and sysctl, which macOS
# supplies. Re-running the same command resumes and reuses passed stages.
# Works with the macOS system bash (3.2).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
KIT_TGZ="$HERE/Audioscopy_CP13_R1_NATIVE_OPERATOR_KIT_2026-11-27.tar.gz"
KIT_SHA="daa2631387be36e55790693cc1ad4c584b8c22592a0e761b1b8c2b870037ac6d"
KIT_DIR="$HERE/kit"
ROOT="$KIT_DIR/Audioscopy_CP13_R1"
WORK="$HOME/audioscopy-qualification/cp13-r1-m3-portable"
RETURN="$HERE/return"
LOG="$RETURN/cp13-r1-m3-portable.log"
NIX="nix --extra-experimental-features nix-command --extra-experimental-features flakes"

mode="full"
case "${1:-}" in
  "") ;;
  --check) mode="check" ;;
  *) echo "usage: $0 [--check]" >&2; exit 2 ;;
esac

say() { printf '\n== %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
sha256() {
  if command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | awk '{print $1}'
  else sha256sum "$1" | awk '{print $1}'; fi
}

say "Host"
uname -a
system="$(uname -s)"; arch="$(uname -m)"
if [ "$system" = "Darwin" ]; then
  [ "$arch" = "arm64" ] || die "expected arm64, got $arch"
  [ "$(sysctl -n sysctl.proc_translated 2>/dev/null || echo 0)" = "0" ] || die "running under Rosetta; use a native arm64 terminal"
  [ -x /usr/bin/time ] || die "/usr/bin/time missing"
  sysctl -n machdep.cpu.brand_string || true
fi
command -v nix >/dev/null 2>&1 || die "nix not on PATH"

say "Kit checksum"
got="$(sha256 "$KIT_TGZ")"
[ "$got" = "$KIT_SHA" ] || die "kit SHA-256 $got, expected $KIT_SHA"
echo "OK $KIT_SHA"

say "Extract kit"
if [ -d "$ROOT" ]; then
  echo "kit already extracted at $ROOT (the runner verifies every file)"
else
  mkdir -p "$KIT_DIR"
  tar xzf "$KIT_TGZ" -C "$KIT_DIR"
  echo "extracted to $ROOT"
fi

mkdir -p "$RETURN"
cd "$ROOT"

say "Toolchain from the flake (first run fetches it)"
$NIX develop "path:$HERE/flake" --command bash -c '
  set -e
  echo "rustc:   $(rustc --version)"
  echo "cargo:   $(cargo --version)"
  echo "rustfmt: $(rustfmt --version)"
  echo "clippy:  $(cargo clippy --version)"
  echo "python:  $(python3 --version) at $(command -v python3)"
  echo "cc:      $(command -v cc)"; cc --version | head -n 1
  echo "ar:      $(command -v ar)"
  if [ "$(uname -s)" = Darwin ]; then echo "time:    /usr/bin/time (system)"; else echo "time:    $(command -v time)"; fi
  rustc -vV | grep -q "^release: 1.99.1$" || { echo "rustc is not 1.99.1" >&2; exit 1; }
  rustc -vV | grep -q "^host: aarch64-" || [ "$(uname -s)" != Darwin ] || { echo "rustc host is not aarch64" >&2; exit 1; }
  for v in RUSTFLAGS CARGO_ENCODED_RUSTFLAGS RUSTC_WRAPPER RUSTC_WORKSPACE_WRAPPER CARGO_BUILD_TARGET CARGO_TARGET_DIR; do
    if [ -n "${!v:-}" ]; then echo "unset $v before running" >&2; exit 1; fi
  done
  if env | grep -qE "^CARGO_(PROFILE|TARGET)_"; then echo "unset CARGO_PROFILE_* / CARGO_TARGET_* before running" >&2; exit 1; fi
  t="$(mktemp -d)"
  printf "#include <math.h>\nint main(void){return sqrt(4.0)==2.0?0:1;}\n" > "$t/cc-probe.c"
  cc -std=c11 -O2 "$t/cc-probe.c" -lm -o "$t/cc-probe" && "$t/cc-probe" && echo "cc probe: OK"
  cc -std=c11 -O2 -fPIC -shared "$t/cc-probe.c" -o "$t/probe.so" 2>/dev/null || cc -std=c11 -O2 -fPIC -dynamiclib "$t/cc-probe.c" -o "$t/probe.dylib"
  echo "shared library probe: OK"
  rm -rf "$t"
'

say "Kit verify and runner self-tests"
$NIX develop "path:$HERE/flake" --command python3 -B qualification-cp13/qualify_cp13.py --verify-only
( cd qualification-cp7 && $NIX develop "path:$HERE/flake" --command python3 -B -m unittest -v test_runner )

if [ "$mode" = "check" ]; then
  say "Checks passed. Run ./run-m3.sh with no option for the full qualification."
  exit 0
fi

say "Full CP13 qualification (about 30-60 minutes). Log: $LOG"
echo "Keep the Mac awake, on mains power, with other apps closed."
awake=""; [ "$system" = "Darwin" ] && awake="caffeinate -dims"
set +e
$NIX develop "path:$HERE/flake" --command \
  $awake python3 -B qualification-cp13/qualify_cp13.py \
    --work "$WORK" --layout-control --long 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "CP13_EXIT=$rc" | tee -a "$LOG"

evidence="$(grep '^EVIDENCE ' "$LOG" | tail -n 1 | cut -d' ' -f2-)"
if [ -n "$evidence" ] && [ -f "$evidence" ]; then
  cp "$evidence" "$RETURN/"
  [ -f "$evidence.sha256" ] && cp "$evidence.sha256" "$RETURN/"
  say "Evidence"
  echo "$RETURN/$(basename "$evidence")"
  echo "SHA-256 $(sha256 "$RETURN/$(basename "$evidence")")"
  grep -E '^(PASS|FAIL|UNAVAILABLE|BLOCKED)' "$LOG" | awk '{print $1}' | sort | uniq -c
  grep -E '^(FAIL|UNAVAILABLE|BLOCKED)' "$LOG" || true
  echo "Bring back the whole $RETURN folder."
else
  echo "No EVIDENCE line found. Bring back $LOG."
fi
exit "$rc"
