// src/py/mod.rs
//! Consolidated Python helpers split into submodules for clarity.
#![allow(non_local_definitions)]

#[cfg(feature = "jit")]
pub mod jit;
#[cfg(not(feature = "jit"))]
pub mod jit_stub;
pub mod mailbox;
pub mod pool;
pub mod runtime;
pub mod utils;
#[cfg(feature = "vortex")]
pub mod vortex;
pub mod wrappers;

// re-export a few helpers for external callers (tests, build scripts, etc.)
pub use wrappers::{init, make_module};
