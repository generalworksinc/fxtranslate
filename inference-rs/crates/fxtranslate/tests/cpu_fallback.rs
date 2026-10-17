#![cfg(all(feature = "gemmology", not(target_arch = "wasm32")))]
//! The gemmology shim is compiled for AVX2 (x86_64) or i8mm (aarch64) regardless of
//! the machine that runs it. On a CPU without those instructions, nothing in the shim
//! may be called: the first prepared weight was an illegal instruction (SIGILL, exit
//! 132). `FXTRANSLATE_FORCE_SCALAR` stands in for such a CPU here. It is read once per
//! process, so this file holds a single test (each test file is its own process).
//!
//! The real case was seen under `qemu-x86_64 -cpu Nehalem`: before, exit 132 on the
//! first translation; after, the translation is byte-identical to a `portable` build.

use fxtranslate::gemm::{self, PreparedB};

#[test]
fn a_cpu_without_the_kernels_instructions_never_enters_the_shim() {
    // SAFETY: set before anything in this test process reads the environment.
    unsafe { std::env::set_var("FXTRANSLATE_FORCE_SCALAR", "1") };
    assert_eq!(gemm::backend(), "scalar");
    // A shape the SIMD kernel would take (k a multiple of every register width) is
    // declined, so the weight stays raw for the scalar kernel.
    assert!(PreparedB::new(&[1i8; 64 * 64], 64, 64).is_none());
    // Nor is what the engine reports about its kernel read from the shim.
    assert_eq!((gemm::gemm_threads(), gemm::prepared_bytes()), (1, 0));
}
