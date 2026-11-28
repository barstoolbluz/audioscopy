# Against A R4 native corrective: operator transfer

This branch only carries the operator kit to the aarch64 macOS host. It is
not a source release; `main` is unchanged.

| File | SHA-256 |
| --- | --- |
| `against-a-comparison-R4-NATIVE-CORRECTIVE.tar.gz` | `a417ae6ddf32165ae34ad00f5bc4ba7ce0298575480bd25870e53ba3ee1c7e07` |
| `x86-evidence/against-a-x86_64-linux-20261128T032103Z-teXAKrLx.tar.gz` (PASS) | `846266e4cd320007372b6f0df78ef7fa0da7438fc5b0b610ea7ecac866ee49b7` |

The kit is R4 (`a2ef2555…`) plus the native corrective described in its
`R4-NATIVE-CORRECTIVE.md`. Both crates are GPL-2.0-or-later.

## Run on Apple Silicon (native arm64, not Rosetta)

Nix with flakes is the only prerequisite. The kit's flake supplies Rust
1.99.1, the C compiler, Python with numpy/scipy/mpmath, and the archive tools.

```bash
git fetch origin && git switch against-a-r4-native
mkdir -p ~/against-a-r4 && cd ~/against-a-r4
cp ~/path/to/audioscopy/against-a-r4-native/against-a-comparison-R4-NATIVE-CORRECTIVE.tar.gz* .
shasum -a 256 -c against-a-comparison-R4-NATIVE-CORRECTIVE.tar.gz.sha256
tar xzf against-a-comparison-R4-NATIVE-CORRECTIVE.tar.gz
cd against-a-comparison-R4-NATIVE-CORRECTIVE
shasum -a 256 -c MANIFEST-SHA256.txt
nix develop --no-update-lock-file path:. --command bash operator/qualify.sh aarch64-darwin
```

The run writes `against-a-aarch64-darwin-<time>-<id>.tar.gz` and its
`.sha256` into `~/against-a-r4`. `Status: PASS` and exit 0 mean the run
qualified. Any other status means it did not.
