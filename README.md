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

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in this crate by you, as defined in the Apache-2.0
licence, shall be dual licensed as above, without any additional terms or
conditions.
