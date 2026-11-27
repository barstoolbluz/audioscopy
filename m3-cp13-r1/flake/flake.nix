{
  description = "audioscopy CP13-R1 native qualification: every tool the runner needs (M3 first, Linux for checks)";

  # Same inputs and flake.lock as the audioscopy repository and the CP7/CP13
  # kits, so Rust 1.99.1 and nixpkgs resolve to the same revisions.
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay }:
    flake-utils.lib.eachSystem [ "aarch64-darwin" "aarch64-linux" "x86_64-linux" ] (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ (import rust-overlay) ];
        };
        lib = pkgs.lib;
        isDarwin = pkgs.stdenv.hostPlatform.isDarwin;

        # rustc, cargo, rustfmt and Clippy, all 1.99.1.
        rustToolchain = pkgs.rust-bin.stable."1.99.1".default.override {
          extensions = [ "rust-src" "clippy" "rustfmt" ];
        };

        # The qualification stages use only the standard library. The extra
        # modules match the CP7 kit flake, for the historical reference scripts.
        python = pkgs.python3.withPackages (ps: [ ps.mpmath ps.numpy ps.scipy ]);
      in
      {
        # mkShell's stdenv supplies the native C compiler and archiver as `cc`
        # and `ar`: Clang and cctools on macOS, GCC and binutils on Linux.
        #
        # Taken from the operating system, not from Nix, on macOS:
        #   /usr/bin/time    BSD time; the runner calls this exact path for RSS
        #   /usr/sbin/sysctl CPU identity; the runner calls `sysctl`
        devShells.default = pkgs.mkShell {
          packages = [
            rustToolchain
            python
            pkgs.gnutar
            pkgs.gzip
            pkgs.gnupatch
            pkgs.diffutils
            pkgs.cacert
          ]
          ++ lib.optionals isDarwin [
            pkgs.libiconv   # some Rust links on Darwin need -liconv
          ]
          ++ lib.optionals (!isDarwin) [
            pkgs.time       # GNU time, for peak RSS
            pkgs.util-linux # taskset, lscpu
            pkgs.binutils   # objdump
          ];

          # cargo downloads crates from crates.io on the first build.
          SSL_CERT_FILE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
        };
      });
}
