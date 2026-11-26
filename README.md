# audioscopy

Audio measurements computed from decoded PCM samples.

The crate takes samples in and gives measurements out. It does not open or
decode files; bring your own decoder.

Status: empty. Nothing is measured yet.

## Building

With Nix:

```bash
nix develop
cargo test
```

Without Nix, any Rust toolchain at or above the version in `Cargo.toml`.

## Licence

audioscopy is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free
Software Foundation, either version 2 of the License, or (at your option)
any later version. The full text is in [COPYING](COPYING).

It is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE.
