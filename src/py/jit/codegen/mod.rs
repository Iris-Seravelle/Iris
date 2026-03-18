// src/py/jit/codegen.rs
//! Core JIT compilation logic, including registry and Cranelift codegen.

use std::collections::{HashMap, HashSet};
use std::ffi::CStr;
use std::time::Instant;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use crate::py::jit::simd;

// cranelift imports
use cranelift::prelude::*;
use cranelift_module::{Linkage, Module};

use pyo3::AsPyPointer;
use pyo3::IntoPy;

const BREAK_SENTINEL_BITS: u64 = 0x7ff8_0000_0000_0b01;
const CONTINUE_SENTINEL_BITS: u64 = 0x7ff8_0000_0000_0c01;

mod jit_types;
mod registry;
mod quantum;
mod lowering;
mod gen_expr;
mod compiler;
mod jit_module;
use jit_module::TLS_JIT_TYPE_PROFILE;

pub use jit_types::*;
pub use registry::*;
pub use quantum::*;
pub use lowering::*;
pub use compiler::*;

fn simd_math_mode_from_env() -> SimdMathMode {
    match std::env::var("IRIS_JIT_SIMD_MATH") {
        Ok(v) => {
            let key = v.trim().to_ascii_lowercase();
            if matches!(key.as_str(), "fast" | "approx" | "poly") {
                SimdMathMode::FastApprox
            } else {
                SimdMathMode::Accurate
            }
        }
        Err(_) => SimdMathMode::Accurate,
    }
}

fn jit_dump_clif_enabled() -> bool {
    match std::env::var("IRIS_JIT_DUMP_CLIF") {
        Ok(v) => matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        Err(_) => false,
    }
}


impl JitEntry {
    /// Check if this entry has a valid, compiled function pointer.
    pub fn is_valid(&self) -> bool {
        self.func_ptr != 0
    }

    fn with_strategy(mut self, strategy: QuantumVariantStrategy) -> Self {
        self.variant_strategy = strategy;
        self
    }

    fn allows_lowered_kernel(&self) -> bool {
        self.variant_strategy != QuantumVariantStrategy::ScalarFallback
    }

    fn loop_unroll_for_entry(&self) -> usize {
        match self.variant_strategy {
            QuantumVariantStrategy::ScalarFallback => 1,
            _ => simd_unroll_factor(),
        }
    }

    fn math_mode_for_entry(&self) -> SimdMathMode {
        match self.variant_strategy {
            QuantumVariantStrategy::FastTrigExperiment => SimdMathMode::FastApprox,
            _ => simd_math_mode_from_env(),
        }
    }
}

static LOWERED_EXEC_LOGGED: once_cell::sync::OnceCell<std::sync::Mutex<HashSet<usize>>> =
    once_cell::sync::OnceCell::new();
static UNROLL_EXEC_LOGGED: once_cell::sync::OnceCell<std::sync::Mutex<HashSet<usize>>> =
    once_cell::sync::OnceCell::new();
static SIMD_MATH_EXEC_LOGGED: once_cell::sync::OnceCell<std::sync::Mutex<HashSet<usize>>> =
    once_cell::sync::OnceCell::new();

fn log_lowered_exec_once(entry: &JitEntry, len: usize) {
    let Some(kernel) = entry.lowered_kernel.as_ref() else {
        return;
    };

    let set = LOWERED_EXEC_LOGGED.get_or_init(|| std::sync::Mutex::new(HashSet::new()));
    let mut guard = set.lock().unwrap();
    if !guard.insert(entry.func_ptr) {
        return;
    }

    let math_mode = entry.math_mode_for_entry();
    crate::py::jit::jit_log(|| {
        format!(
            "[Iris][jit][lower] execute kernel={:?} reduction={:?} len={} math_mode={:?} variant={:?}",
            kernel, entry.reduction, len, math_mode, entry.variant_strategy
        )
    });
}

fn log_unroll_exec_once(entry: &JitEntry, len: usize, unroll: usize, path: &'static str) {
    if unroll <= 1 || len < unroll {
        return;
    }
    let set = UNROLL_EXEC_LOGGED.get_or_init(|| std::sync::Mutex::new(HashSet::new()));
    let mut guard = set.lock().unwrap();
    if !guard.insert(entry.func_ptr) {
        return;
    }
    crate::py::jit::jit_log(|| {
        format!(
            "[Iris][jit][unroll] execute path={} len={} unroll={} variant={:?}",
            path, len, unroll, entry.variant_strategy
        )
    });
}

fn log_simd_math_exec_once(entry: &JitEntry, op: LoweredUnaryKernel) {
    let set = SIMD_MATH_EXEC_LOGGED.get_or_init(|| std::sync::Mutex::new(HashSet::new()));
    let mut guard = set.lock().unwrap();
    if !guard.insert(entry.func_ptr) {
        return;
    }

    crate::py::jit::jit_log(|| {
        format!(
            "[Iris][jit][simd-math] backend=neon-f64x2 op={:?} variant={:?}",
            op, entry.variant_strategy
        )
    });
}

// Quantum variant / profiling helpers are implemented in `quantum.rs`.
// See `crate::py::jit::codegen::quantum` for the full implementation.

#[cfg(feature = "pyo3")]
pub fn execute_registered_jit(
    py: pyo3::Python,
    func_key: usize,
    args: &pyo3::types::PyTuple,
) -> Option<pyo3::PyResult<pyo3::PyObject>> {
    if crate::py::jit::quantum_speculation_enabled() {
        if let Some(map) = QUANTUM_REGISTRY.get() {
            let (entry, idx, fallback_entries, should_use_quantum, active_count) = {
                let mut guard = map.lock().unwrap();
                if let Some(state) = guard.get_mut(&func_key) {
                    if state.entries.is_empty() {
                        (None, 0usize, Vec::new(), false, 0usize)
                    } else {
                        // Decide whether to use multi-variant quantum dispatch.
                        // If the observed runtime remains below threshold, stick to the
                        // baseline and avoid speculative overhead.
                        let speculation_threshold_ns =
                            crate::py::jit::quantum_speculation_threshold_ns();
                        let best_ewma = state
                            .stats
                            .iter()
                            .enumerate()
                            .filter(|(idx, _)| state.active.get(*idx).copied().unwrap_or(false))
                            .map(|(_, s)| s)
                            .filter(|s| s.runs > 0)
                            .map(|s| s.ewma_ns)
                            .fold(f64::MAX, |a: f64, b| a.min(b));
                        let best_ewma = if best_ewma == f64::MAX {
                            0.0
                        } else {
                            best_ewma
                        };
                        let stability = quantum_stability_score(state);
                        let min_stability = crate::py::jit::quantum_stability_min_score();
                        let active_count = active_indices(state).len();
                        let has_unrun = has_unrun_active_variant(state);
                        let use_quantum = should_use_quantum_dispatch(
                            active_count,
                            best_ewma,
                            stability,
                            min_stability,
                            speculation_threshold_ns,
                            has_unrun,
                        );

                        if use_quantum {
                            let idx = choose_quantum_index(state);
                            let entry = state.entries[idx].clone();
                            // Check if the selected entry is valid; if not, skip quantum and use baseline.
                            if !entry.is_valid() {
                                if idx < state.active.len() {
                                    state.active[idx] = false;
                                }
                                let baseline_idx = state.baseline_idx.min(state.entries.len() - 1);
                                let baseline_entry = state.entries[baseline_idx].clone();
                                if !baseline_entry.is_valid() {
                                    // No valid entries at all; skip JIT
                                    (None, 0usize, Vec::new(), false, 0usize)
                                } else {
                                    (
                                        Some(baseline_entry),
                                        baseline_idx,
                                        Vec::new(),
                                        false,
                                        active_count,
                                    )
                                }
                            } else {
                                let mut fallbacks = Vec::new();
                                for (i, e) in state.entries.iter().enumerate() {
                                    if i != idx
                                        && state.active.get(i).copied().unwrap_or(false)
                                        && e.is_valid()
                                    {
                                        fallbacks.push((i, e.clone()));
                                    }
                                }
                                (Some(entry), idx, fallbacks, true, active_count)
                            }
                        } else {
                            // Baseline execution only
                            let baseline_idx = state.baseline_idx.min(state.entries.len() - 1);
                            let baseline_entry = state.entries[baseline_idx].clone();
                            if !baseline_entry.is_valid() {
                                // Baseline is invalid; skip JIT
                                (None, 0usize, Vec::new(), false, active_count)
                            } else {
                                (
                                    Some(baseline_entry),
                                    baseline_idx,
                                    Vec::new(),
                                    false,
                                    active_count,
                                )
                            }
                        }
                    }
                } else {
                    (None, 0usize, Vec::new(), false, 0usize)
                }
            };

            if let Some(entry) = entry {
                let quantum_total = active_count.max(1);
                let chosen_strategy = entry.variant_strategy;
                let log_threshold_ns = crate::py::jit::quantum_log_threshold_ns();

                let log_if_slow = |chosen: usize, used: usize, elapsed_ns: u64, note: &str| {
                    if crate::py::jit::jit_logging_enabled() && elapsed_ns >= log_threshold_ns {
                        crate::py::jit::jit_log(|| {
                            format!(
                                "[Iris][jit][quantum] func_key={} chosen={}/{} used={} elapsed={}ns {} active={} strategy={:?}",
                                func_key, chosen, quantum_total, used, elapsed_ns, note, active_count, chosen_strategy
                            )
                        });
                    }
                };

                let start = Instant::now();
                match execute_jit_func(py, &entry, args) {
                    Ok(obj) => {
                        let elapsed = start.elapsed().as_nanos() as u64;
                        update_quantum_stats(func_key, idx, elapsed, true);
                        if !should_use_quantum {
                            let _ = crate::py::jit::maybe_rearm_quantum_compile(
                                func_key,
                                elapsed,
                                active_count,
                            );
                        }
                        log_if_slow(
                            idx,
                            idx,
                            elapsed,
                            if should_use_quantum {
                                "success"
                            } else {
                                "baseline"
                            },
                        );
                        return Some(Ok(obj));
                    }
                    Err(primary_err) => {
                        let elapsed = start.elapsed().as_nanos() as u64;
                        update_quantum_stats(func_key, idx, elapsed, false);
                        if !should_use_quantum {
                            let _ = crate::py::jit::maybe_rearm_quantum_compile(
                                func_key,
                                elapsed,
                                active_count,
                            );
                        }
                        log_if_slow(idx, idx, elapsed, "primary_failed");
                        if should_use_quantum {
                            for (fb_idx, fb_entry) in fallback_entries {
                                let start_fb = Instant::now();
                                match execute_jit_func(py, &fb_entry, args) {
                                    Ok(obj) => {
                                        let elapsed_fb = start_fb.elapsed().as_nanos() as u64;
                                        update_quantum_stats(func_key, fb_idx, elapsed_fb, true);
                                        log_if_slow(idx, fb_idx, elapsed_fb, "fallback_success");
                                        return Some(Ok(obj));
                                    }
                                    Err(_) => {
                                        let elapsed_fb = start_fb.elapsed().as_nanos() as u64;
                                        update_quantum_stats(func_key, fb_idx, elapsed_fb, false);
                                        log_if_slow(idx, fb_idx, elapsed_fb, "fallback_failed");
                                    }
                                }
                            }
                        }
                        return Some(Err(primary_err));
                    }
                }
            }
        }
    }

    lookup_jit(func_key).map(|entry| execute_jit_func(py, &entry, args))
}

pub use jit_module::*;


pub(crate) use gen_expr::gen_expr;

// helper for zero-copy buffer access used by the JIT runner
struct BufferView {
    view: pyo3::ffi::Py_buffer,
    elem_type: BufferElemType,
    len: usize,
}

impl BufferView {
    #[inline(always)]
    fn as_ptr_u8(&self) -> *const u8 {
        self.view.buf as *const u8
    }

    #[inline(always)]
    fn as_ptr_f64(&self) -> *const f64 {
        self.view.buf as *const f64
    }

    #[inline(always)]
    fn is_aligned_for_f64(&self) -> bool {
        (self.view.buf as usize) % std::mem::align_of::<f64>() == 0
    }
}

impl Drop for BufferView {
    fn drop(&mut self) {
        unsafe { pyo3::ffi::PyBuffer_Release(&mut self.view) };
    }
}

#[cfg(feature = "pyo3")]
unsafe fn parse_buffer_elem_type(view: &pyo3::ffi::Py_buffer) -> Option<BufferElemType> {
    if view.itemsize <= 0 {
        return None;
    }
    let itemsize = view.itemsize as usize;

    fn expected_size_for_code(code: char) -> Option<usize> {
        match code {
            'd' => Some(std::mem::size_of::<f64>()),
            'f' => Some(std::mem::size_of::<f32>()),
            'q' => Some(std::mem::size_of::<i64>()),
            'i' => Some(std::mem::size_of::<i32>()),
            'h' => Some(std::mem::size_of::<i16>()),
            'b' => Some(std::mem::size_of::<i8>()),
            'Q' => Some(std::mem::size_of::<u64>()),
            'I' => Some(std::mem::size_of::<u32>()),
            'H' => Some(std::mem::size_of::<u16>()),
            'B' => Some(std::mem::size_of::<u8>()),
            '?' => Some(std::mem::size_of::<u8>()),
            _ => None,
        }
    }

    fn to_elem_type(code: char, itemsize: usize) -> Option<BufferElemType> {
        if code == 'l' {
            return match itemsize {
                8 => Some(BufferElemType::I64),
                4 => Some(BufferElemType::I32),
                _ => None,
            };
        }
        if code == 'L' {
            return match itemsize {
                8 => Some(BufferElemType::U64),
                4 => Some(BufferElemType::U32),
                _ => None,
            };
        }
        if let Some(expected) = expected_size_for_code(code) {
            if expected != itemsize {
                return None;
            }
        }
        match code {
            'd' => Some(BufferElemType::F64),
            'f' => Some(BufferElemType::F32),
            'q' => Some(BufferElemType::I64),
            'i' => Some(BufferElemType::I32),
            'h' => Some(BufferElemType::I16),
            'b' => Some(BufferElemType::I8),
            'Q' => Some(BufferElemType::U64),
            'I' => Some(BufferElemType::U32),
            'H' => Some(BufferElemType::U16),
            'B' => Some(BufferElemType::U8),
            '?' => Some(BufferElemType::Bool),
            _ => None,
        }
    }

    if view.format.is_null() {
        return match itemsize {
            8 => Some(BufferElemType::F64),
            4 => Some(BufferElemType::F32),
            2 => Some(BufferElemType::I16),
            1 => Some(BufferElemType::U8),
            _ => None,
        };
    }

    let fmt = CStr::from_ptr(view.format).to_str().ok()?;
    let code = fmt
        .chars()
        .rev()
        .find(|ch| ch.is_ascii_alphabetic() || *ch == '?')?;
    to_elem_type(code, itemsize)
}

#[cfg(feature = "pyo3")]
unsafe fn open_typed_buffer(obj: &pyo3::PyAny) -> Option<BufferView> {
    let mut view: pyo3::ffi::Py_buffer = std::mem::zeroed();
    let flags = pyo3::ffi::PyBUF_C_CONTIGUOUS | pyo3::ffi::PyBUF_FORMAT;
    if pyo3::ffi::PyObject_GetBuffer(obj.as_ptr(), &mut view, flags) != 0 {
        pyo3::ffi::PyErr_Clear();
        return None;
    }

    let itemsize = view.itemsize as usize;
    if itemsize == 0 {
        pyo3::ffi::PyBuffer_Release(&mut view);
        return None;
    }

    let elem_type = match parse_buffer_elem_type(&view) {
        Some(elem) => elem,
        None => {
            pyo3::ffi::PyBuffer_Release(&mut view);
            return None;
        }
    };

    let total_bytes = view.len as usize;
    if total_bytes % itemsize != 0 {
        pyo3::ffi::PyBuffer_Release(&mut view);
        return None;
    }

    let len = total_bytes / itemsize;
    Some(BufferView {
        view,
        elem_type,
        len,
    })
}

#[cfg(feature = "pyo3")]
#[inline(always)]
unsafe fn read_buffer_f64(view: &BufferView, index: usize) -> f64 {
    let base = view.as_ptr_u8();
    match view.elem_type {
        BufferElemType::F64 => {
            let p = base.add(index * std::mem::size_of::<f64>()) as *const f64;
            std::ptr::read_unaligned(p)
        }
        BufferElemType::F32 => {
            let p = base.add(index * std::mem::size_of::<f32>()) as *const f32;
            std::ptr::read_unaligned(p) as f64
        }
        BufferElemType::I64 => {
            let p = base.add(index * std::mem::size_of::<i64>()) as *const i64;
            std::ptr::read_unaligned(p) as f64
        }
        BufferElemType::I32 => {
            let p = base.add(index * std::mem::size_of::<i32>()) as *const i32;
            std::ptr::read_unaligned(p) as f64
        }
        BufferElemType::I16 => {
            let p = base.add(index * std::mem::size_of::<i16>()) as *const i16;
            std::ptr::read_unaligned(p) as f64
        }
        BufferElemType::I8 => {
            let p = base.add(index * std::mem::size_of::<i8>()) as *const i8;
            std::ptr::read_unaligned(p) as f64
        }
        BufferElemType::U64 => {
            let p = base.add(index * std::mem::size_of::<u64>()) as *const u64;
            std::ptr::read_unaligned(p) as f64
        }
        BufferElemType::U32 => {
            let p = base.add(index * std::mem::size_of::<u32>()) as *const u32;
            std::ptr::read_unaligned(p) as f64
        }
        BufferElemType::U16 => {
            let p = base.add(index * std::mem::size_of::<u16>()) as *const u16;
            std::ptr::read_unaligned(p) as f64
        }
        BufferElemType::U8 => {
            let p = base.add(index * std::mem::size_of::<u8>()) as *const u8;
            std::ptr::read_unaligned(p) as f64
        }
        BufferElemType::Bool => {
            let p = base.add(index) as *const u8;
            if std::ptr::read_unaligned(p) == 0 {
                0.0
            } else {
                1.0
            }
        }
    }
}

#[inline(always)]
fn fast_sin_approx(x: f64) -> f64 {
    const APPROX_DOMAIN_LIMIT: f64 = 4_096.0;
    const PI: f64 = std::f64::consts::PI;
    const TAU: f64 = std::f64::consts::TAU;
    const HALF_PI: f64 = std::f64::consts::FRAC_PI_2;

    if !x.is_finite() || x.abs() > APPROX_DOMAIN_LIMIT {
        return x.sin();
    }

    let mut y = x.rem_euclid(TAU);
    if y > PI {
        y -= TAU;
    }

    let mut sign = 1.0;
    if y > HALF_PI {
        y = PI - y;
    } else if y < -HALF_PI {
        y = PI + y;
        sign = -1.0;
    }

    let z = y * y;
    let p = y
        * (1.0 + z * (-1.0 / 6.0 + z * (1.0 / 120.0 + z * (-1.0 / 5040.0 + z * (1.0 / 362880.0)))));
    sign * p
}

#[inline(always)]
fn fast_cos_approx(x: f64) -> f64 {
    const APPROX_DOMAIN_LIMIT: f64 = 4_096.0;
    if !x.is_finite() || x.abs() > APPROX_DOMAIN_LIMIT {
        return x.cos();
    }
    fast_sin_approx(x + std::f64::consts::FRAC_PI_2)
}

#[cfg(target_arch = "aarch64")]
fn fast_sin_reduce_for_poly(x: f64) -> Option<(f64, f64)> {
    const PI: f64 = std::f64::consts::PI;
    const TAU: f64 = std::f64::consts::TAU;
    const HALF_PI: f64 = std::f64::consts::FRAC_PI_2;

    if !x.is_finite() {
        return None;
    }

    let mut y = x.rem_euclid(TAU);
    if y > PI {
        y -= TAU;
    }

    let mut sign = 1.0;
    if y > HALF_PI {
        y = PI - y;
    } else if y < -HALF_PI {
        y = PI + y;
        sign = -1.0;
    }

    Some((y, sign))
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn fast_sin_approx_pair_neon(x0: f64, x1: f64) -> (f64, f64) {
    let (y0, s0) = match fast_sin_reduce_for_poly(x0) {
        Some(reduced) => reduced,
        None => return (x0.sin(), x1.sin()),
    };
    let (y1, s1) = match fast_sin_reduce_for_poly(x1) {
        Some(reduced) => reduced,
        None => return (x0.sin(), x1.sin()),
    };

    let y_arr = [y0, y1];
    let sign_arr = [s0, s1];
    let y = vld1q_f64(y_arr.as_ptr());
    let sign = vld1q_f64(sign_arr.as_ptr());
    let z = vmulq_f64(y, y);

    let c1 = vdupq_n_f64(-1.0 / 6.0);
    let c2 = vdupq_n_f64(1.0 / 120.0);
    let c3 = vdupq_n_f64(-1.0 / 5040.0);
    let c4 = vdupq_n_f64(1.0 / 362880.0);

    let p3 = vaddq_f64(c3, vmulq_f64(z, c4));
    let p2 = vaddq_f64(c2, vmulq_f64(z, p3));
    let p1 = vaddq_f64(c1, vmulq_f64(z, p2));
    let poly = vaddq_f64(vdupq_n_f64(1.0), vmulq_f64(z, p1));
    let out = vmulq_f64(sign, vmulq_f64(y, poly));

    let mut out_arr = [0.0_f64; 2];
    vst1q_f64(out_arr.as_mut_ptr(), out);
    (out_arr[0], out_arr[1])
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn fast_cos_approx_pair_neon(x0: f64, x1: f64) -> (f64, f64) {
    const APPROX_DOMAIN_LIMIT: f64 = 4_096.0;
    if !x0.is_finite()
        || !x1.is_finite()
        || x0.abs() > APPROX_DOMAIN_LIMIT
        || x1.abs() > APPROX_DOMAIN_LIMIT
    {
        return (x0.cos(), x1.cos());
    }
    fast_sin_approx_pair_neon(
        x0 + std::f64::consts::FRAC_PI_2,
        x1 + std::f64::consts::FRAC_PI_2,
    )
}

#[inline(always)]
fn lowered_unary_eval_pair(
    op: LoweredUnaryKernel,
    x0: f64,
    x1: f64,
    mode: SimdMathMode,
) -> Option<(f64, f64)> {
    if mode != SimdMathMode::FastApprox {
        return None;
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        return match op {
            LoweredUnaryKernel::Sin => Some(fast_sin_approx_pair_neon(x0, x1)),
            LoweredUnaryKernel::Cos => Some(fast_cos_approx_pair_neon(x0, x1)),
            LoweredUnaryKernel::Tan => {
                let (s0, s1) = fast_sin_approx_pair_neon(x0, x1);
                let (c0, c1) = fast_cos_approx_pair_neon(x0, x1);
                Some((s0 / c0, s1 / c1))
            }
            _ => None,
        };
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = (op, x0, x1);
        None
    }
}

#[inline(always)]
fn lowered_unary_eval(op: LoweredUnaryKernel, x: f64, mode: SimdMathMode) -> f64 {
    match op {
        LoweredUnaryKernel::Identity => x,
        LoweredUnaryKernel::Neg => -x,
        LoweredUnaryKernel::Abs => x.abs(),
        LoweredUnaryKernel::Sin => {
            if mode == SimdMathMode::FastApprox {
                fast_sin_approx(x)
            } else {
                x.sin()
            }
        }
        LoweredUnaryKernel::Cos => {
            if mode == SimdMathMode::FastApprox {
                fast_cos_approx(x)
            } else {
                x.cos()
            }
        }
        LoweredUnaryKernel::Tan => {
            if mode == SimdMathMode::FastApprox {
                fast_sin_approx(x) / fast_cos_approx(x)
            } else {
                x.tan()
            }
        }
        LoweredUnaryKernel::Exp => x.exp(),
        LoweredUnaryKernel::Log => x.ln(),
        LoweredUnaryKernel::Sqrt => x.sqrt(),
    }
}

#[inline(always)]
fn lowered_binary_eval(op: LoweredBinaryKernel, lhs: f64, rhs: f64) -> f64 {
    match op {
        LoweredBinaryKernel::Add => lhs + rhs,
        LoweredBinaryKernel::Sub => lhs - rhs,
        LoweredBinaryKernel::Mul => lhs * rhs,
        LoweredBinaryKernel::Div => lhs / rhs,
    }
}

fn lowered_expr_eval(
    expr: &LoweredExpr,
    views: &[BufferView],
    idx: usize,
    mode: SimdMathMode,
) -> Option<f64> {
    match expr {
        LoweredExpr::Const(v) => Some(*v),
        LoweredExpr::Input(input_idx) => {
            let view = views.get(*input_idx)?;
            Some(unsafe { read_buffer_f64(view, idx) })
        }
        LoweredExpr::Add(a, b) => {
            Some(lowered_expr_eval(a, views, idx, mode)? + lowered_expr_eval(b, views, idx, mode)?)
        }
        LoweredExpr::Sub(a, b) => {
            Some(lowered_expr_eval(a, views, idx, mode)? - lowered_expr_eval(b, views, idx, mode)?)
        }
        LoweredExpr::Mul(a, b) => {
            Some(lowered_expr_eval(a, views, idx, mode)? * lowered_expr_eval(b, views, idx, mode)?)
        }
        LoweredExpr::Div(a, b) => {
            Some(lowered_expr_eval(a, views, idx, mode)? / lowered_expr_eval(b, views, idx, mode)?)
        }
        LoweredExpr::Mod(a, b) => {
            Some(lowered_expr_eval(a, views, idx, mode)? % lowered_expr_eval(b, views, idx, mode)?)
        }
        LoweredExpr::Eq(a, b) => Some(
            if lowered_expr_eval(a, views, idx, mode)? == lowered_expr_eval(b, views, idx, mode)? {
                1.0
            } else {
                0.0
            },
        ),
        LoweredExpr::Ne(a, b) => Some(
            if lowered_expr_eval(a, views, idx, mode)? != lowered_expr_eval(b, views, idx, mode)? {
                1.0
            } else {
                0.0
            },
        ),
        LoweredExpr::Lt(a, b) => Some(
            if lowered_expr_eval(a, views, idx, mode)? < lowered_expr_eval(b, views, idx, mode)? {
                1.0
            } else {
                0.0
            },
        ),
        LoweredExpr::Gt(a, b) => Some(
            if lowered_expr_eval(a, views, idx, mode)? > lowered_expr_eval(b, views, idx, mode)? {
                1.0
            } else {
                0.0
            },
        ),
        LoweredExpr::Le(a, b) => Some(
            if lowered_expr_eval(a, views, idx, mode)? <= lowered_expr_eval(b, views, idx, mode)? {
                1.0
            } else {
                0.0
            },
        ),
        LoweredExpr::Ge(a, b) => Some(
            if lowered_expr_eval(a, views, idx, mode)? >= lowered_expr_eval(b, views, idx, mode)? {
                1.0
            } else {
                0.0
            },
        ),
        LoweredExpr::And(a, b) => Some(
            if lowered_expr_eval(a, views, idx, mode)? != 0.0
                && lowered_expr_eval(b, views, idx, mode)? != 0.0
            {
                1.0
            } else {
                0.0
            },
        ),
        LoweredExpr::Or(a, b) => Some(
            if lowered_expr_eval(a, views, idx, mode)? != 0.0
                || lowered_expr_eval(b, views, idx, mode)? != 0.0
            {
                1.0
            } else {
                0.0
            },
        ),
        LoweredExpr::Neg(a) => Some(-lowered_expr_eval(a, views, idx, mode)?),
        LoweredExpr::Not(a) => Some(if lowered_expr_eval(a, views, idx, mode)? == 0.0 {
            1.0
        } else {
            0.0
        }),
        LoweredExpr::Abs(a) => Some(lowered_expr_eval(a, views, idx, mode)?.abs()),
        LoweredExpr::Sin(a) => Some(lowered_unary_eval(
            LoweredUnaryKernel::Sin,
            lowered_expr_eval(a, views, idx, mode)?,
            mode,
        )),
        LoweredExpr::Cos(a) => Some(lowered_unary_eval(
            LoweredUnaryKernel::Cos,
            lowered_expr_eval(a, views, idx, mode)?,
            mode,
        )),
        LoweredExpr::Tan(a) => Some(lowered_unary_eval(
            LoweredUnaryKernel::Tan,
            lowered_expr_eval(a, views, idx, mode)?,
            mode,
        )),
        LoweredExpr::Exp(a) => Some(lowered_unary_eval(
            LoweredUnaryKernel::Exp,
            lowered_expr_eval(a, views, idx, mode)?,
            mode,
        )),
        LoweredExpr::Log(a) => Some(lowered_unary_eval(
            LoweredUnaryKernel::Log,
            lowered_expr_eval(a, views, idx, mode)?,
            mode,
        )),
        LoweredExpr::Sqrt(a) => Some(lowered_unary_eval(
            LoweredUnaryKernel::Sqrt,
            lowered_expr_eval(a, views, idx, mode)?,
            mode,
        )),
        LoweredExpr::Ternary {
            cond,
            then_expr,
            else_expr,
        } => {
            if lowered_expr_eval(cond, views, idx, mode)? != 0.0 {
                lowered_expr_eval(then_expr, views, idx, mode)
            } else {
                lowered_expr_eval(else_expr, views, idx, mode)
            }
        }
    }
}

#[inline(always)]
fn lowered_unary_eval_pair_any(
    op: LoweredUnaryKernel,
    x0: f64,
    x1: f64,
    mode: SimdMathMode,
) -> (f64, f64) {
    if let Some((y0, y1)) = lowered_unary_eval_pair(op, x0, x1, mode) {
        (y0, y1)
    } else {
        (
            lowered_unary_eval(op, x0, mode),
            lowered_unary_eval(op, x1, mode),
        )
    }
}

fn lowered_expr_eval_pair(
    expr: &LoweredExpr,
    views: &[BufferView],
    idx0: usize,
    idx1: usize,
    mode: SimdMathMode,
) -> Option<(f64, f64)> {
    match expr {
        LoweredExpr::Const(v) => Some((*v, *v)),
        LoweredExpr::Input(input_idx) => {
            let view = views.get(*input_idx)?;
            Some((unsafe { read_buffer_f64(view, idx0) }, unsafe {
                read_buffer_f64(view, idx1)
            }))
        }
        LoweredExpr::Add(a, b) => {
            let (a0, a1) = lowered_expr_eval_pair(a, views, idx0, idx1, mode)?;
            let (b0, b1) = lowered_expr_eval_pair(b, views, idx0, idx1, mode)?;
            Some((a0 + b0, a1 + b1))
        }
        LoweredExpr::Sub(a, b) => {
            let (a0, a1) = lowered_expr_eval_pair(a, views, idx0, idx1, mode)?;
            let (b0, b1) = lowered_expr_eval_pair(b, views, idx0, idx1, mode)?;
            Some((a0 - b0, a1 - b1))
        }
        LoweredExpr::Mul(a, b) => {
            let (a0, a1) = lowered_expr_eval_pair(a, views, idx0, idx1, mode)?;
            let (b0, b1) = lowered_expr_eval_pair(b, views, idx0, idx1, mode)?;
            Some((a0 * b0, a1 * b1))
        }
        LoweredExpr::Div(a, b) => {
            let (a0, a1) = lowered_expr_eval_pair(a, views, idx0, idx1, mode)?;
            let (b0, b1) = lowered_expr_eval_pair(b, views, idx0, idx1, mode)?;
            Some((a0 / b0, a1 / b1))
        }
        LoweredExpr::Mod(a, b) => {
            let (a0, a1) = lowered_expr_eval_pair(a, views, idx0, idx1, mode)?;
            let (b0, b1) = lowered_expr_eval_pair(b, views, idx0, idx1, mode)?;
            Some((a0 % b0, a1 % b1))
        }
        LoweredExpr::Eq(a, b) => {
            let (a0, a1) = lowered_expr_eval_pair(a, views, idx0, idx1, mode)?;
            let (b0, b1) = lowered_expr_eval_pair(b, views, idx0, idx1, mode)?;
            Some((
                if a0 == b0 { 1.0 } else { 0.0 },
                if a1 == b1 { 1.0 } else { 0.0 },
            ))
        }
        LoweredExpr::Ne(a, b) => {
            let (a0, a1) = lowered_expr_eval_pair(a, views, idx0, idx1, mode)?;
            let (b0, b1) = lowered_expr_eval_pair(b, views, idx0, idx1, mode)?;
            Some((
                if a0 != b0 { 1.0 } else { 0.0 },
                if a1 != b1 { 1.0 } else { 0.0 },
            ))
        }
        LoweredExpr::Lt(a, b) => {
            let (a0, a1) = lowered_expr_eval_pair(a, views, idx0, idx1, mode)?;
            let (b0, b1) = lowered_expr_eval_pair(b, views, idx0, idx1, mode)?;
            Some((
                if a0 < b0 { 1.0 } else { 0.0 },
                if a1 < b1 { 1.0 } else { 0.0 },
            ))
        }
        LoweredExpr::Gt(a, b) => {
            let (a0, a1) = lowered_expr_eval_pair(a, views, idx0, idx1, mode)?;
            let (b0, b1) = lowered_expr_eval_pair(b, views, idx0, idx1, mode)?;
            Some((
                if a0 > b0 { 1.0 } else { 0.0 },
                if a1 > b1 { 1.0 } else { 0.0 },
            ))
        }
        LoweredExpr::Le(a, b) => {
            let (a0, a1) = lowered_expr_eval_pair(a, views, idx0, idx1, mode)?;
            let (b0, b1) = lowered_expr_eval_pair(b, views, idx0, idx1, mode)?;
            Some((
                if a0 <= b0 { 1.0 } else { 0.0 },
                if a1 <= b1 { 1.0 } else { 0.0 },
            ))
        }
        LoweredExpr::Ge(a, b) => {
            let (a0, a1) = lowered_expr_eval_pair(a, views, idx0, idx1, mode)?;
            let (b0, b1) = lowered_expr_eval_pair(b, views, idx0, idx1, mode)?;
            Some((
                if a0 >= b0 { 1.0 } else { 0.0 },
                if a1 >= b1 { 1.0 } else { 0.0 },
            ))
        }
        LoweredExpr::And(a, b) => {
            let (a0, a1) = lowered_expr_eval_pair(a, views, idx0, idx1, mode)?;
            let (b0, b1) = lowered_expr_eval_pair(b, views, idx0, idx1, mode)?;
            Some((
                if a0 != 0.0 && b0 != 0.0 { 1.0 } else { 0.0 },
                if a1 != 0.0 && b1 != 0.0 { 1.0 } else { 0.0 },
            ))
        }
        LoweredExpr::Or(a, b) => {
            let (a0, a1) = lowered_expr_eval_pair(a, views, idx0, idx1, mode)?;
            let (b0, b1) = lowered_expr_eval_pair(b, views, idx0, idx1, mode)?;
            Some((
                if a0 != 0.0 || b0 != 0.0 { 1.0 } else { 0.0 },
                if a1 != 0.0 || b1 != 0.0 { 1.0 } else { 0.0 },
            ))
        }
        LoweredExpr::Neg(a) => {
            let (x0, x1) = lowered_expr_eval_pair(a, views, idx0, idx1, mode)?;
            Some((-x0, -x1))
        }
        LoweredExpr::Not(a) => {
            let (x0, x1) = lowered_expr_eval_pair(a, views, idx0, idx1, mode)?;
            Some((
                if x0 == 0.0 { 1.0 } else { 0.0 },
                if x1 == 0.0 { 1.0 } else { 0.0 },
            ))
        }
        LoweredExpr::Abs(a) => {
            let (x0, x1) = lowered_expr_eval_pair(a, views, idx0, idx1, mode)?;
            Some((x0.abs(), x1.abs()))
        }
        LoweredExpr::Sin(a) => {
            let (x0, x1) = lowered_expr_eval_pair(a, views, idx0, idx1, mode)?;
            Some(lowered_unary_eval_pair_any(
                LoweredUnaryKernel::Sin,
                x0,
                x1,
                mode,
            ))
        }
        LoweredExpr::Cos(a) => {
            let (x0, x1) = lowered_expr_eval_pair(a, views, idx0, idx1, mode)?;
            Some(lowered_unary_eval_pair_any(
                LoweredUnaryKernel::Cos,
                x0,
                x1,
                mode,
            ))
        }
        LoweredExpr::Tan(a) => {
            let (x0, x1) = lowered_expr_eval_pair(a, views, idx0, idx1, mode)?;
            Some(lowered_unary_eval_pair_any(
                LoweredUnaryKernel::Tan,
                x0,
                x1,
                mode,
            ))
        }
        LoweredExpr::Exp(a) => {
            let (x0, x1) = lowered_expr_eval_pair(a, views, idx0, idx1, mode)?;
            Some(lowered_unary_eval_pair_any(
                LoweredUnaryKernel::Exp,
                x0,
                x1,
                mode,
            ))
        }
        LoweredExpr::Log(a) => {
            let (x0, x1) = lowered_expr_eval_pair(a, views, idx0, idx1, mode)?;
            Some(lowered_unary_eval_pair_any(
                LoweredUnaryKernel::Log,
                x0,
                x1,
                mode,
            ))
        }
        LoweredExpr::Sqrt(a) => {
            let (x0, x1) = lowered_expr_eval_pair(a, views, idx0, idx1, mode)?;
            Some(lowered_unary_eval_pair_any(
                LoweredUnaryKernel::Sqrt,
                x0,
                x1,
                mode,
            ))
        }
        LoweredExpr::Ternary {
            cond,
            then_expr,
            else_expr,
        } => {
            let (c0, c1) = lowered_expr_eval_pair(cond, views, idx0, idx1, mode)?;
            let c0_true = c0 != 0.0;
            let c1_true = c1 != 0.0;
            match (c0_true, c1_true) {
                (true, true) => lowered_expr_eval_pair(then_expr, views, idx0, idx1, mode),
                (false, false) => lowered_expr_eval_pair(else_expr, views, idx0, idx1, mode),
                _ => {
                    let v0 = if c0_true {
                        lowered_expr_eval(then_expr, views, idx0, mode)?
                    } else {
                        lowered_expr_eval(else_expr, views, idx0, mode)?
                    };
                    let v1 = if c1_true {
                        lowered_expr_eval(then_expr, views, idx1, mode)?
                    } else {
                        lowered_expr_eval(else_expr, views, idx1, mode)?
                    };
                    Some((v0, v1))
                }
            }
        }
    }
}

enum LoweredVectorResult {
    Vector(Vec<f64>),
    Reduced(f64),
}

#[inline(always)]
fn try_execute_lowered_vector_kernel(
    entry: &JitEntry,
    views: &[BufferView],
    len: usize,
    unroll: usize,
) -> Option<LoweredVectorResult> {
    if !entry.allows_lowered_kernel() {
        return None;
    }
    let mode = entry.math_mode_for_entry();

    let kernel = entry.lowered_kernel.clone()?;

    let eval_unary = |op: LoweredUnaryKernel, view: &BufferView, idx: usize| {
        let x = unsafe { read_buffer_f64(view, idx) };
        lowered_unary_eval(op, x, mode)
    };
    let eval_binary =
        |op: LoweredBinaryKernel, lhs_view: &BufferView, rhs_view: &BufferView, idx: usize| {
            let l = unsafe { read_buffer_f64(lhs_view, idx) };
            let r = unsafe { read_buffer_f64(rhs_view, idx) };
            lowered_binary_eval(op, l, r)
        };

    if entry.reduction != ReductionMode::None {
        let lanes = unroll.clamp(1, 4);
        match kernel {
            LoweredKernel::Unary { op, input } => {
                let input_view = views.get(input)?;
                match entry.reduction {
                    ReductionMode::Sum => {
                        let mut lane_acc = [0.0_f64; 4];
                        let mut i = 0;
                        while i + lanes <= len {
                            let mut lane = 0;
                            while lane < lanes {
                                if lane + 1 < lanes {
                                    let x0 = unsafe { read_buffer_f64(input_view, i + lane) };
                                    let x1 = unsafe { read_buffer_f64(input_view, i + lane + 1) };
                                    if let Some((y0, y1)) =
                                        lowered_unary_eval_pair(op, x0, x1, mode)
                                    {
                                        log_simd_math_exec_once(entry, op);
                                        lane_acc[lane] += y0;
                                        lane_acc[lane + 1] += y1;
                                        lane += 2;
                                        continue;
                                    }
                                }
                                lane_acc[lane] += eval_unary(op, input_view, i + lane);
                                lane += 1;
                            }
                            i += lanes;
                        }
                        let mut acc = lane_acc[..lanes].iter().copied().sum::<f64>();
                        while i < len {
                            acc += eval_unary(op, input_view, i);
                            i += 1;
                        }
                        return Some(LoweredVectorResult::Reduced(acc));
                    }
                    ReductionMode::Any => {
                        let mut lane_any = [false; 4];
                        let mut i = 0;
                        while i + lanes <= len {
                            for lane in 0..lanes {
                                lane_any[lane] |= eval_unary(op, input_view, i + lane) != 0.0;
                            }
                            if lane_any[..lanes].iter().any(|v| *v) {
                                return Some(LoweredVectorResult::Reduced(1.0));
                            }
                            i += lanes;
                        }
                        while i < len {
                            if eval_unary(op, input_view, i) != 0.0 {
                                return Some(LoweredVectorResult::Reduced(1.0));
                            }
                            i += 1;
                        }
                        return Some(LoweredVectorResult::Reduced(0.0));
                    }
                    ReductionMode::All => {
                        let mut lane_all = [true; 4];
                        let mut i = 0;
                        while i + lanes <= len {
                            for lane in 0..lanes {
                                lane_all[lane] &= eval_unary(op, input_view, i + lane) != 0.0;
                            }
                            if lane_all[..lanes].iter().any(|v| !*v) {
                                return Some(LoweredVectorResult::Reduced(0.0));
                            }
                            i += lanes;
                        }
                        while i < len {
                            if eval_unary(op, input_view, i) == 0.0 {
                                return Some(LoweredVectorResult::Reduced(0.0));
                            }
                            i += 1;
                        }
                        return Some(LoweredVectorResult::Reduced(1.0));
                    }
                    ReductionMode::None => {}
                }
            }
            LoweredKernel::Binary { op, lhs, rhs } => {
                let lhs_view = views.get(lhs)?;
                let rhs_view = views.get(rhs)?;
                match entry.reduction {
                    ReductionMode::Sum => {
                        let mut lane_acc = [0.0_f64; 4];
                        let mut i = 0;
                        while i + lanes <= len {
                            for lane in 0..lanes {
                                lane_acc[lane] += eval_binary(op, lhs_view, rhs_view, i + lane);
                            }
                            i += lanes;
                        }
                        let mut acc = lane_acc[..lanes].iter().copied().sum::<f64>();
                        while i < len {
                            acc += eval_binary(op, lhs_view, rhs_view, i);
                            i += 1;
                        }
                        return Some(LoweredVectorResult::Reduced(acc));
                    }
                    ReductionMode::Any => {
                        let mut lane_any = [false; 4];
                        let mut i = 0;
                        while i + lanes <= len {
                            for lane in 0..lanes {
                                lane_any[lane] |=
                                    eval_binary(op, lhs_view, rhs_view, i + lane) != 0.0;
                            }
                            if lane_any[..lanes].iter().any(|v| *v) {
                                return Some(LoweredVectorResult::Reduced(1.0));
                            }
                            i += lanes;
                        }
                        while i < len {
                            if eval_binary(op, lhs_view, rhs_view, i) != 0.0 {
                                return Some(LoweredVectorResult::Reduced(1.0));
                            }
                            i += 1;
                        }
                        return Some(LoweredVectorResult::Reduced(0.0));
                    }
                    ReductionMode::All => {
                        let mut lane_all = [true; 4];
                        let mut i = 0;
                        while i + lanes <= len {
                            for lane in 0..lanes {
                                lane_all[lane] &=
                                    eval_binary(op, lhs_view, rhs_view, i + lane) != 0.0;
                            }
                            if lane_all[..lanes].iter().any(|v| !*v) {
                                return Some(LoweredVectorResult::Reduced(0.0));
                            }
                            i += lanes;
                        }
                        while i < len {
                            if eval_binary(op, lhs_view, rhs_view, i) == 0.0 {
                                return Some(LoweredVectorResult::Reduced(0.0));
                            }
                            i += 1;
                        }
                        return Some(LoweredVectorResult::Reduced(1.0));
                    }
                    ReductionMode::None => {}
                }
            }
            LoweredKernel::Expr(expr) => match entry.reduction {
                ReductionMode::Sum => {
                    let mut lane_acc = [0.0_f64; 4];
                    let mut i = 0;
                    while i + lanes <= len {
                        let mut lane = 0;
                        while lane < lanes {
                            if lane + 1 < lanes {
                                let idx0 = i + lane;
                                let idx1 = i + lane + 1;
                                let (v0, v1) =
                                    lowered_expr_eval_pair(&expr, views, idx0, idx1, mode)?;
                                lane_acc[lane] += v0;
                                lane_acc[lane + 1] += v1;
                                lane += 2;
                                continue;
                            }
                            lane_acc[lane] += lowered_expr_eval(&expr, views, i + lane, mode)?;
                            lane += 1;
                        }
                        i += lanes;
                    }
                    let mut acc = lane_acc[..lanes].iter().copied().sum::<f64>();
                    while i < len {
                        acc += lowered_expr_eval(&expr, views, i, mode)?;
                        i += 1;
                    }
                    return Some(LoweredVectorResult::Reduced(acc));
                }
                ReductionMode::Any => {
                    let mut lane_any = [false; 4];
                    let mut i = 0;
                    while i + lanes <= len {
                        for lane in 0..lanes {
                            lane_any[lane] |=
                                lowered_expr_eval(&expr, views, i + lane, mode)? != 0.0;
                        }
                        if lane_any[..lanes].iter().any(|v| *v) {
                            return Some(LoweredVectorResult::Reduced(1.0));
                        }
                        i += lanes;
                    }
                    while i < len {
                        if lowered_expr_eval(&expr, views, i, mode)? != 0.0 {
                            return Some(LoweredVectorResult::Reduced(1.0));
                        }
                        i += 1;
                    }
                    return Some(LoweredVectorResult::Reduced(0.0));
                }
                ReductionMode::All => {
                    let mut lane_all = [true; 4];
                    let mut i = 0;
                    while i + lanes <= len {
                        for lane in 0..lanes {
                            lane_all[lane] &=
                                lowered_expr_eval(&expr, views, i + lane, mode)? != 0.0;
                        }
                        if lane_all[..lanes].iter().any(|v| !*v) {
                            return Some(LoweredVectorResult::Reduced(0.0));
                        }
                        i += lanes;
                    }
                    while i < len {
                        if lowered_expr_eval(&expr, views, i, mode)? == 0.0 {
                            return Some(LoweredVectorResult::Reduced(0.0));
                        }
                        i += 1;
                    }
                    return Some(LoweredVectorResult::Reduced(1.0));
                }
                ReductionMode::None => {}
            },
        }
        return None;
    }

    let mut results = Vec::with_capacity(len);

    match kernel {
        LoweredKernel::Unary { op, input } => {
            let input_view = views.get(input)?;
            let mut i = 0;
            while i + unroll <= len {
                let mut lane = 0;
                while lane < unroll {
                    if lane + 1 < unroll {
                        let idx = i + lane;
                        let x0 = unsafe { read_buffer_f64(input_view, idx) };
                        let x1 = unsafe { read_buffer_f64(input_view, idx + 1) };
                        if let Some((y0, y1)) = lowered_unary_eval_pair(op, x0, x1, mode) {
                            log_simd_math_exec_once(entry, op);
                            results.push(y0);
                            results.push(y1);
                            lane += 2;
                            continue;
                        }
                    }
                    let idx = i + lane;
                    results.push(eval_unary(op, input_view, idx));
                    lane += 1;
                }
                i += unroll;
            }
            while i < len {
                results.push(eval_unary(op, input_view, i));
                i += 1;
            }
            Some(LoweredVectorResult::Vector(results))
        }
        LoweredKernel::Binary { op, lhs, rhs } => {
            let lhs_view = views.get(lhs)?;
            let rhs_view = views.get(rhs)?;
            let mut i = 0;
            while i + unroll <= len {
                for lane in 0..unroll {
                    let idx = i + lane;
                    results.push(eval_binary(op, lhs_view, rhs_view, idx));
                }
                i += unroll;
            }
            while i < len {
                results.push(eval_binary(op, lhs_view, rhs_view, i));
                i += 1;
            }
            Some(LoweredVectorResult::Vector(results))
        }
        LoweredKernel::Expr(expr) => {
            let mut i = 0;
            while i + unroll <= len {
                let mut lane = 0;
                while lane < unroll {
                    if lane + 1 < unroll {
                        let idx0 = i + lane;
                        let idx1 = i + lane + 1;
                        let (v0, v1) = lowered_expr_eval_pair(&expr, views, idx0, idx1, mode)?;
                        results.push(v0);
                        results.push(v1);
                        lane += 2;
                        continue;
                    }
                    results.push(lowered_expr_eval(&expr, views, i + lane, mode)?);
                    lane += 1;
                }
                i += unroll;
            }
            while i < len {
                results.push(lowered_expr_eval(&expr, views, i, mode)?);
                i += 1;
            }
            Some(LoweredVectorResult::Vector(results))
        }
    }
}

#[cfg(feature = "pyo3")]
#[inline(always)]
fn lookup_exec_profile(func_ptr: usize) -> Option<JitExecProfile> {
    TLS_JIT_TYPE_PROFILE.with(|m| m.borrow().get(&func_ptr).cloned())
}

#[cfg(feature = "pyo3")]
#[inline(always)]
fn store_exec_profile(func_ptr: usize, profile: JitExecProfile) {
    TLS_JIT_TYPE_PROFILE.with(|m| {
        m.borrow_mut().insert(func_ptr, profile);
    });
}

#[cfg(feature = "pyo3")]
#[inline(always)]
fn reduction_identity(mode: ReductionMode) -> f64 {
    match mode {
        ReductionMode::None => 0.0,
        ReductionMode::Sum => 0.0,
        ReductionMode::Any => 0.0,
        ReductionMode::All => 1.0,
    }
}

#[cfg(feature = "pyo3")]
#[inline(always)]
fn reduction_step(mode: ReductionMode, acc: &mut f64, value: f64) -> bool {
    let bits = value.to_bits();
    if bits == BREAK_SENTINEL_BITS {
        return true;
    }
    if bits == CONTINUE_SENTINEL_BITS {
        return false;
    }

    match mode {
        ReductionMode::None => false,
        ReductionMode::Sum => {
            *acc += value;
            false
        }
        ReductionMode::Any => {
            if value != 0.0 {
                *acc = 1.0;
                true
            } else {
                false
            }
        }
        ReductionMode::All => {
            if value == 0.0 {
                *acc = 0.0;
                true
            } else {
                false
            }
        }
    }
}

#[inline(always)]
fn simd_unroll_factor_for_plan(plan: simd::SimdPlan) -> usize {
    if !plan.auto_vectorize {
        return 1;
    }

    let lanes = (plan.lane_bytes / std::mem::size_of::<f64>()).max(1);
    lanes.min(4)
}

#[inline(always)]
fn simd_unroll_factor() -> usize {
    simd_unroll_factor_for_plan(simd::auto_vectorization_plan())
}

/// Highly optimized helper to execute a JIT compiled function.
/// Handles zero-copy buffers (including multi-argument vectorization) and scalar argument unpacking via stack.
#[cfg(feature = "pyo3")]
#[inline(always)]
fn jit_return_to_py(py: pyo3::Python, value: f64, return_type: JitReturnType) -> pyo3::PyObject {
    match return_type {
        JitReturnType::Float => value.into_py(py),
        JitReturnType::Int => (value as i64).into_py(py),
        JitReturnType::Bool => (value != 0.0).into_py(py),
    }
}

#[cfg(feature = "pyo3")]
#[inline(always)]
fn vec_f64_to_py(py: pyo3::Python, values: &[f64], return_type: JitReturnType) -> pyo3::PyObject {
    match return_type {
        JitReturnType::Float => {
            let byte_slice = unsafe {
                std::slice::from_raw_parts(
                    values.as_ptr() as *const u8,
                    values.len() * std::mem::size_of::<f64>(),
                )
            };
            let py_bytes = pyo3::types::PyBytes::new(py, byte_slice);
            let array_mod = py.import("array").unwrap();
            let array_obj = array_mod
                .getattr("array")
                .unwrap()
                .call1(("d", py_bytes))
                .unwrap();
            array_obj.into_py(py)
        }
        JitReturnType::Int => {
            let list = pyo3::types::PyList::new(py, values.iter().map(|v| *v as i64));
            list.into_py(py)
        }
        JitReturnType::Bool => {
            let list = pyo3::types::PyList::new(py, values.iter().map(|v| *v != 0.0));
            list.into_py(py)
        }
    }
}

#[cfg(feature = "pyo3")]
#[inline(always)]
pub fn execute_jit_func(
    py: pyo3::Python,
    entry: &JitEntry,
    args: &pyo3::types::PyTuple,
) -> pyo3::PyResult<pyo3::PyObject> {
    // Safety check: if func_ptr is zero (or invalid), cannot execute JIT.
    // This can happen if quantum speculation is enabled but variants haven't been compiled yet.
    if entry.func_ptr == 0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "JIT function pointer is invalid or not yet compiled",
        ));
    }

    let arg_count = args.len();
    let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry.func_ptr) };
    let loop_unroll = entry.loop_unroll_for_entry();

    // 0. Trailing-count vectorization mode: call with N scalar args matching
    // the compiled signature, plus one final integer count to request a
    // vectorized return of repeated evaluations.
    if arg_count == entry.arg_count + 1 {
        if let Ok(count_item) = args.get_item(arg_count - 1) {
            if let Ok(count_i64) = count_item.extract::<i64>() {
                if count_i64 < 0 {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "count must be non-negative",
                    ));
                }
                let count = count_i64 as usize;

                const MAX_FAST_ARGS: usize = 8;
                let mut results = Vec::with_capacity(count);
                log_unroll_exec_once(entry, count, loop_unroll, "trailing-count");
                if entry.arg_count <= MAX_FAST_ARGS {
                    let mut stack_args: [f64; MAX_FAST_ARGS] = [0.0; MAX_FAST_ARGS];
                    for i in 0..entry.arg_count {
                        let item =
                            unsafe { pyo3::ffi::PyTuple_GET_ITEM(args.as_ptr(), i as isize) };
                        let val = unsafe { pyo3::ffi::PyFloat_AsDouble(item) };
                        if val == -1.0 && !unsafe { pyo3::ffi::PyErr_Occurred() }.is_null() {
                            return Err(pyo3::PyErr::fetch(py));
                        }
                        stack_args[i] = val;
                    }
                    let mut produced = 0;
                    while produced + loop_unroll <= count {
                        for _ in 0..loop_unroll {
                            results.push(f(stack_args.as_ptr()));
                        }
                        produced += loop_unroll;
                    }
                    while produced < count {
                        results.push(f(stack_args.as_ptr()));
                        produced += 1;
                    }
                } else {
                    let mut heap_args = Vec::with_capacity(entry.arg_count);
                    for i in 0..entry.arg_count {
                        let item =
                            unsafe { pyo3::ffi::PyTuple_GET_ITEM(args.as_ptr(), i as isize) };
                        let val = unsafe { pyo3::ffi::PyFloat_AsDouble(item) };
                        if val == -1.0 && !unsafe { pyo3::ffi::PyErr_Occurred() }.is_null() {
                            return Err(pyo3::PyErr::fetch(py));
                        }
                        heap_args.push(val);
                    }
                    let mut produced = 0;
                    while produced + loop_unroll <= count {
                        for _ in 0..loop_unroll {
                            results.push(f(heap_args.as_ptr()));
                        }
                        produced += loop_unroll;
                    }
                    while produced < count {
                        results.push(f(heap_args.as_ptr()));
                        produced += 1;
                    }
                }

                let byte_slice = unsafe {
                    std::slice::from_raw_parts(
                        results.as_ptr() as *const u8,
                        results.len() * std::mem::size_of::<f64>(),
                    )
                };
                let py_bytes = pyo3::types::PyBytes::new(py, byte_slice);
                let array_mod = py.import("array")?;
                let array_obj = array_mod.getattr("array")?.call1(("d", py_bytes))?;
                return Ok(array_obj.into_py(py));
            }
        }
    }

    // Speculative fast path using cached type profile.
    if let Some(profile) = lookup_exec_profile(entry.func_ptr) {
        match profile {
            JitExecProfile::PackedBuffer {
                arg_count: expected,
                elem,
            } => {
                if arg_count == 1 && entry.arg_count == expected {
                    if let Ok(item) = args.get_item(0) {
                        if let Some(view) = unsafe { open_typed_buffer(item) } {
                            if view.elem_type == elem && view.len == expected {
                                if elem == BufferElemType::F64 {
                                    if view.is_aligned_for_f64() {
                                        let res = f(view.as_ptr_f64());
                                        return Ok(jit_return_to_py(py, res, entry.return_type));
                                    }
                                }
                                let mut converted = Vec::with_capacity(expected);
                                for i in 0..expected {
                                    converted.push(unsafe { read_buffer_f64(&view, i) });
                                }
                                let res = f(converted.as_ptr());
                                return Ok(jit_return_to_py(py, res, entry.return_type));
                            }
                        }
                    }
                }
            }
            JitExecProfile::VectorizedBuffers {
                arg_count: expected,
                elem_types,
            } => {
                if arg_count == expected && expected == elem_types.len() {
                    let mut views = Vec::with_capacity(expected);
                    let mut common_len: Option<usize> = None;
                    let mut matched = true;
                    for i in 0..expected {
                        let Ok(item) = args.get_item(i) else {
                            matched = false;
                            break;
                        };
                        let Some(view) = (unsafe { open_typed_buffer(item) }) else {
                            matched = false;
                            break;
                        };
                        if view.elem_type != elem_types[i] {
                            matched = false;
                            break;
                        }
                        if let Some(len) = common_len {
                            if view.len != len {
                                matched = false;
                                break;
                            }
                        } else {
                            common_len = Some(view.len);
                        }
                        views.push(view);
                    }

                    if matched {
                        let len = common_len.unwrap_or(0);
                        log_unroll_exec_once(entry, len, loop_unroll, "profiled-vector-buffers");
                        if let Some(lowered) =
                            try_execute_lowered_vector_kernel(entry, &views, len, loop_unroll)
                        {
                            log_lowered_exec_once(entry, len);
                            match lowered {
                                LoweredVectorResult::Vector(results) => {
                                    return Ok(vec_f64_to_py(py, &results, entry.return_type));
                                }
                                LoweredVectorResult::Reduced(value) => {
                                    return Ok(jit_return_to_py(py, value, entry.return_type));
                                }
                            }
                        }
                        if entry.reduction != ReductionMode::None {
                            let mut acc = reduction_identity(entry.reduction);
                            const MAX_FAST_ARGS: usize = 8;
                            if expected <= MAX_FAST_ARGS {
                                let mut i = 0;
                                let mut should_break = false;
                                while i + loop_unroll <= len {
                                    for lane in 0..loop_unroll {
                                        let idx = i + lane;
                                        let mut iter_args = [0.0_f64; MAX_FAST_ARGS];
                                        for j in 0..expected {
                                            iter_args[j] =
                                                unsafe { read_buffer_f64(&views[j], idx) };
                                        }
                                        let val = f(iter_args.as_ptr());
                                        if reduction_step(entry.reduction, &mut acc, val) {
                                            should_break = true;
                                            break;
                                        }
                                    }
                                    if should_break {
                                        break;
                                    }
                                    i += loop_unroll;
                                }
                                while i < len {
                                    let mut iter_args = [0.0_f64; MAX_FAST_ARGS];
                                    for j in 0..expected {
                                        iter_args[j] = unsafe { read_buffer_f64(&views[j], i) };
                                    }
                                    let val = f(iter_args.as_ptr());
                                    if reduction_step(entry.reduction, &mut acc, val) {
                                        break;
                                    }
                                    i += 1;
                                }
                            } else {
                                let mut i = 0;
                                let mut should_break = false;
                                while i + loop_unroll <= len {
                                    for lane in 0..loop_unroll {
                                        let idx = i + lane;
                                        let mut iter_args = Vec::with_capacity(expected);
                                        for j in 0..expected {
                                            iter_args
                                                .push(unsafe { read_buffer_f64(&views[j], idx) });
                                        }
                                        let val = f(iter_args.as_ptr());
                                        if reduction_step(entry.reduction, &mut acc, val) {
                                            should_break = true;
                                            break;
                                        }
                                    }
                                    if should_break {
                                        break;
                                    }
                                    i += loop_unroll;
                                }
                                while i < len {
                                    let mut iter_args = Vec::with_capacity(expected);
                                    for j in 0..expected {
                                        iter_args.push(unsafe { read_buffer_f64(&views[j], i) });
                                    }
                                    let val = f(iter_args.as_ptr());
                                    if reduction_step(entry.reduction, &mut acc, val) {
                                        break;
                                    }
                                    i += 1;
                                }
                            }
                            return Ok(jit_return_to_py(py, acc, entry.return_type));
                        }

                        let mut results = Vec::with_capacity(len);
                        const MAX_FAST_ARGS: usize = 8;
                        if expected <= MAX_FAST_ARGS {
                            let mut i = 0;
                            while i + loop_unroll <= len {
                                for lane in 0..loop_unroll {
                                    let idx = i + lane;
                                    let mut iter_args = [0.0_f64; MAX_FAST_ARGS];
                                    for j in 0..expected {
                                        iter_args[j] = unsafe { read_buffer_f64(&views[j], idx) };
                                    }
                                    results.push(f(iter_args.as_ptr()));
                                }
                                i += loop_unroll;
                            }
                            while i < len {
                                let mut iter_args = [0.0_f64; MAX_FAST_ARGS];
                                for j in 0..expected {
                                    iter_args[j] = unsafe { read_buffer_f64(&views[j], i) };
                                }
                                results.push(f(iter_args.as_ptr()));
                                i += 1;
                            }
                        } else {
                            let mut i = 0;
                            while i + loop_unroll <= len {
                                for lane in 0..loop_unroll {
                                    let idx = i + lane;
                                    let mut iter_args = Vec::with_capacity(expected);
                                    for j in 0..expected {
                                        iter_args.push(unsafe { read_buffer_f64(&views[j], idx) });
                                    }
                                    results.push(f(iter_args.as_ptr()));
                                }
                                i += loop_unroll;
                            }
                            while i < len {
                                let mut iter_args = Vec::with_capacity(expected);
                                for j in 0..expected {
                                    iter_args.push(unsafe { read_buffer_f64(&views[j], i) });
                                }
                                results.push(f(iter_args.as_ptr()));
                                i += 1;
                            }
                        }
                        return Ok(vec_f64_to_py(py, &results, entry.return_type));
                    }
                }
            }
            JitExecProfile::ScalarArgs => {
                if arg_count == entry.arg_count {
                    const MAX_FAST_ARGS: usize = 8;
                    if arg_count <= MAX_FAST_ARGS {
                        let mut stack_args = [0.0_f64; MAX_FAST_ARGS];
                        let mut scalar_mismatch = false;
                        for i in 0..arg_count {
                            let item =
                                unsafe { pyo3::ffi::PyTuple_GET_ITEM(args.as_ptr(), i as isize) };
                            let val = unsafe { pyo3::ffi::PyFloat_AsDouble(item) };
                            if val == -1.0 && !unsafe { pyo3::ffi::PyErr_Occurred() }.is_null() {
                                unsafe { pyo3::ffi::PyErr_Clear() };
                                scalar_mismatch = true;
                                break;
                            }
                            stack_args[i] = val;
                        }
                        if !scalar_mismatch {
                            let res = f(stack_args.as_ptr());
                            return Ok(jit_return_to_py(py, res, entry.return_type));
                        }
                    }
                }
            }
        }
    }

    // 1. Single buffer acting as the entire argument array for a multi-argument function
    if arg_count == 1 && entry.arg_count > 1 {
        if let Ok(item) = args.get_item(0) {
            if let Some(view) = unsafe { open_typed_buffer(item) } {
                let len = view.len;
                if len == entry.arg_count {
                    let res = if view.elem_type == BufferElemType::F64 {
                        if view.is_aligned_for_f64() {
                            f(view.as_ptr_f64())
                        } else {
                            let mut converted = Vec::with_capacity(len);
                            for i in 0..len {
                                converted.push(unsafe { read_buffer_f64(&view, i) });
                            }
                            f(converted.as_ptr())
                        }
                    } else {
                        let mut converted = Vec::with_capacity(len);
                        for i in 0..len {
                            converted.push(unsafe { read_buffer_f64(&view, i) });
                        }
                        f(converted.as_ptr())
                    };
                    store_exec_profile(
                        entry.func_ptr,
                        JitExecProfile::PackedBuffer {
                            arg_count: entry.arg_count,
                            elem: view.elem_type,
                        },
                    );
                    return Ok(jit_return_to_py(py, res, entry.return_type));
                }
            }
        }
    }

    // 2. Vectorization: 1 or more arguments, all of which must be buffers of the same length
    if arg_count == entry.arg_count && arg_count > 0 {
        let mut views = Vec::with_capacity(arg_count);
        let mut common_len = None;
        let mut all_buffers = true;

        for i in 0..arg_count {
            if let Ok(item) = args.get_item(i) {
                if let Some(view) = unsafe { open_typed_buffer(item) } {
                    let len = view.len;
                    if let Some(c_len) = common_len {
                        if len != c_len {
                            all_buffers = false;
                            break;
                        }
                    } else {
                        common_len = Some(len);
                    }
                    views.push(view);
                } else {
                    all_buffers = false;
                    break;
                }
            } else {
                all_buffers = false;
                break;
            }
        }

        if all_buffers {
            if let Some(len) = common_len {
                log_unroll_exec_once(entry, len, loop_unroll, "generic-vector-buffers");
                if let Some(lowered) =
                    try_execute_lowered_vector_kernel(entry, &views, len, loop_unroll)
                {
                    log_lowered_exec_once(entry, len);
                    store_exec_profile(
                        entry.func_ptr,
                        JitExecProfile::VectorizedBuffers {
                            arg_count,
                            elem_types: views.iter().map(|v| v.elem_type).collect::<Vec<_>>(),
                        },
                    );
                    match lowered {
                        LoweredVectorResult::Vector(results) => {
                            return Ok(vec_f64_to_py(py, &results, entry.return_type));
                        }
                        LoweredVectorResult::Reduced(value) => {
                            return Ok(jit_return_to_py(py, value, entry.return_type));
                        }
                    }
                }
                let unroll = loop_unroll;
                if entry.reduction != ReductionMode::None {
                    let mut acc = reduction_identity(entry.reduction);
                    const MAX_FAST_ARGS: usize = 8;
                    if arg_count <= MAX_FAST_ARGS {
                        let mut i = 0;
                        let mut should_break = false;
                        while i + unroll <= len {
                            for lane in 0..unroll {
                                let idx = i + lane;
                                let mut iter_args: [f64; MAX_FAST_ARGS] = [0.0; MAX_FAST_ARGS];
                                for j in 0..arg_count {
                                    iter_args[j] = unsafe { read_buffer_f64(&views[j], idx) };
                                }
                                let val = f(iter_args.as_ptr());
                                if reduction_step(entry.reduction, &mut acc, val) {
                                    should_break = true;
                                    break;
                                }
                            }
                            if should_break {
                                break;
                            }
                            i += unroll;
                        }

                        while i < len {
                            let mut iter_args: [f64; MAX_FAST_ARGS] = [0.0; MAX_FAST_ARGS];
                            for j in 0..arg_count {
                                iter_args[j] = unsafe { read_buffer_f64(&views[j], i) };
                            }
                            let val = f(iter_args.as_ptr());
                            if reduction_step(entry.reduction, &mut acc, val) {
                                break;
                            }
                            i += 1;
                        }
                    } else {
                        let mut i = 0;
                        let mut should_break = false;
                        while i + unroll <= len {
                            for lane in 0..unroll {
                                let idx = i + lane;
                                let mut iter_args = Vec::with_capacity(arg_count);
                                for j in 0..arg_count {
                                    iter_args.push(unsafe { read_buffer_f64(&views[j], idx) });
                                }
                                let val = f(iter_args.as_ptr());
                                if reduction_step(entry.reduction, &mut acc, val) {
                                    should_break = true;
                                    break;
                                }
                            }
                            if should_break {
                                break;
                            }
                            i += unroll;
                        }

                        while i < len {
                            let mut iter_args = Vec::with_capacity(arg_count);
                            for j in 0..arg_count {
                                iter_args.push(unsafe { read_buffer_f64(&views[j], i) });
                            }
                            let val = f(iter_args.as_ptr());
                            if reduction_step(entry.reduction, &mut acc, val) {
                                break;
                            }
                            i += 1;
                        }
                    }

                    store_exec_profile(
                        entry.func_ptr,
                        JitExecProfile::VectorizedBuffers {
                            arg_count,
                            elem_types: views.iter().map(|v| v.elem_type).collect::<Vec<_>>(),
                        },
                    );
                    return Ok(jit_return_to_py(py, acc, entry.return_type));
                }

                let mut results = Vec::with_capacity(len);
                let elem_types = views.iter().map(|v| v.elem_type).collect::<Vec<_>>();
                let all_f64 = elem_types.iter().all(|k| *k == BufferElemType::F64);

                const MAX_FAST_ARGS: usize = 8;
                if arg_count <= MAX_FAST_ARGS {
                    let mut i = 0;
                    while i + unroll <= len {
                        for lane in 0..unroll {
                            let idx = i + lane;
                            let mut iter_args: [f64; MAX_FAST_ARGS] = [0.0; MAX_FAST_ARGS];
                            for j in 0..arg_count {
                                iter_args[j] = if all_f64 {
                                    unsafe { read_buffer_f64(&views[j], idx) }
                                } else {
                                    unsafe { read_buffer_f64(&views[j], idx) }
                                };
                            }
                            results.push(f(iter_args.as_ptr()));
                        }
                        i += unroll;
                    }

                    while i < len {
                        let mut iter_args: [f64; MAX_FAST_ARGS] = [0.0; MAX_FAST_ARGS];
                        for j in 0..arg_count {
                            iter_args[j] = if all_f64 {
                                unsafe { read_buffer_f64(&views[j], i) }
                            } else {
                                unsafe { read_buffer_f64(&views[j], i) }
                            };
                        }
                        results.push(f(iter_args.as_ptr()));
                        i += 1;
                    }
                } else {
                    let mut i = 0;
                    while i + unroll <= len {
                        for lane in 0..unroll {
                            let idx = i + lane;
                            let mut iter_args = Vec::with_capacity(arg_count);
                            for j in 0..arg_count {
                                let val = if all_f64 {
                                    unsafe { read_buffer_f64(&views[j], idx) }
                                } else {
                                    unsafe { read_buffer_f64(&views[j], idx) }
                                };
                                iter_args.push(val);
                            }
                            results.push(f(iter_args.as_ptr()));
                        }
                        i += unroll;
                    }

                    while i < len {
                        let mut iter_args = Vec::with_capacity(arg_count);
                        for j in 0..arg_count {
                            let val = if all_f64 {
                                unsafe { read_buffer_f64(&views[j], i) }
                            } else {
                                unsafe { read_buffer_f64(&views[j], i) }
                            };
                            iter_args.push(val);
                        }
                        results.push(f(iter_args.as_ptr()));
                        i += 1;
                    }
                }

                store_exec_profile(
                    entry.func_ptr,
                    JitExecProfile::VectorizedBuffers {
                        arg_count,
                        elem_types,
                    },
                );

                return Ok(vec_f64_to_py(py, &results, entry.return_type));
            }
        }
    }

    // 2.25 Generic sequence fallback for single-arg kernels when buffer fast paths miss.
    // This keeps scalar-profiled functions robust when later called with vector-like inputs
    // that don't satisfy the typed-buffer fast path.
    if arg_count == 1 && entry.arg_count == 1 {
        if let Ok(item) = args.get_item(0) {
            let is_text_like = item.is_instance_of::<pyo3::types::PyString>()
                || item.is_instance_of::<pyo3::types::PyBytes>()
                || item.is_instance_of::<pyo3::types::PyByteArray>();

            if !is_text_like {
                if let Ok(len) = item.len() {
                    log_unroll_exec_once(entry, len, loop_unroll, "sequence-fallback");
                    if entry.reduction != ReductionMode::None {
                        let mut acc = reduction_identity(entry.reduction);
                        let mut i = 0;
                        while i + loop_unroll <= len {
                            for lane in 0..loop_unroll {
                                let idx = i + lane;
                                let elem = item.get_item(idx)?;
                                let val: f64 = elem.extract()?;
                                let arg0 = [val];
                                let out = f(arg0.as_ptr());
                                if reduction_step(entry.reduction, &mut acc, out) {
                                    return Ok(jit_return_to_py(py, acc, entry.return_type));
                                }
                            }
                            i += loop_unroll;
                        }
                        while i < len {
                            let elem = item.get_item(i)?;
                            let val: f64 = elem.extract()?;
                            let arg0 = [val];
                            let out = f(arg0.as_ptr());
                            if reduction_step(entry.reduction, &mut acc, out) {
                                break;
                            }
                            i += 1;
                        }
                        return Ok(jit_return_to_py(py, acc, entry.return_type));
                    }

                    let mut results = Vec::with_capacity(len);
                    let mut i = 0;
                    while i + loop_unroll <= len {
                        for lane in 0..loop_unroll {
                            let idx = i + lane;
                            let elem = item.get_item(idx)?;
                            let val: f64 = elem.extract()?;
                            let arg0 = [val];
                            results.push(f(arg0.as_ptr()));
                        }
                        i += loop_unroll;
                    }
                    while i < len {
                        let elem = item.get_item(i)?;
                        let val: f64 = elem.extract()?;
                        let arg0 = [val];
                        results.push(f(arg0.as_ptr()));
                        i += 1;
                    }
                    return Ok(vec_f64_to_py(py, &results, entry.return_type));
                }
            }
        }
    }

    // 2.5 Sequence fallback for reduction-style single-arg kernels
    if arg_count == 1 && entry.arg_count == 1 && entry.reduction != ReductionMode::None {
        if let Ok(item) = args.get_item(0) {
            if let Ok(iter) = item.iter() {
                let mut acc = reduction_identity(entry.reduction);
                let mut buf = [0.0_f64; 1];
                for obj_res in iter {
                    let obj = obj_res?;
                    let val: f64 = obj.extract()?;
                    buf[0] = val;
                    let out = f(buf.as_ptr());
                    if reduction_step(entry.reduction, &mut acc, out) {
                        break;
                    }
                }
                return Ok(jit_return_to_py(py, acc, entry.return_type));
            }
        }
    }

    if arg_count != entry.arg_count {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "wrong argument count for JIT function",
        ));
    }

    // 3. Fast path for small number of standard Python scalar arguments
    const MAX_FAST_ARGS: usize = 8;
    if arg_count <= MAX_FAST_ARGS {
        let mut stack_args: [f64; MAX_FAST_ARGS] = [0.0; MAX_FAST_ARGS];
        for i in 0..arg_count {
            let item = unsafe { pyo3::ffi::PyTuple_GET_ITEM(args.as_ptr(), i as isize) };
            let val = unsafe { pyo3::ffi::PyFloat_AsDouble(item) };
            if val == -1.0 && !unsafe { pyo3::ffi::PyErr_Occurred() }.is_null() {
                return Err(pyo3::PyErr::fetch(py));
            }
            stack_args[i] = val;
        }
        store_exec_profile(entry.func_ptr, JitExecProfile::ScalarArgs);
        let res = f(stack_args.as_ptr());
        return Ok(jit_return_to_py(py, res, entry.return_type));
    }

    // 4. Fallback for > 8 scalar args: heap allocation
    let mut heap_args = Vec::with_capacity(arg_count);
    for i in 0..arg_count {
        let item = unsafe { pyo3::ffi::PyTuple_GET_ITEM(args.as_ptr(), i as isize) };
        let val = unsafe { pyo3::ffi::PyFloat_AsDouble(item) };
        if val == -1.0 && !unsafe { pyo3::ffi::PyErr_Occurred() }.is_null() {
            return Err(pyo3::PyErr::fetch(py));
        }
        heap_args.push(val);
    }
    store_exec_profile(entry.func_ptr, JitExecProfile::ScalarArgs);
    let res = f(heap_args.as_ptr());
    Ok(jit_return_to_py(py, res, entry.return_type))
}

#[cfg(test)]
mod simd_unroll_tests {
    use super::*;
    use crate::py::jit::parser;

    #[test]
    fn scalar_plan_uses_unroll_1() {
        let plan = simd::SimdPlan {
            backend: simd::SimdBackend::Scalar,
            lane_bytes: 8,
            auto_vectorize: false,
        };
        assert_eq!(simd_unroll_factor_for_plan(plan), 1);
    }

    #[test]
    fn neon_plan_uses_unroll_2() {
        let plan = simd::SimdPlan {
            backend: simd::SimdBackend::ArmNeon,
            lane_bytes: 16,
            auto_vectorize: true,
        };
        assert_eq!(simd_unroll_factor_for_plan(plan), 2);
    }

    #[test]
    fn wide_simd_plans_cap_at_unroll_4() {
        let sve = simd::SimdPlan {
            backend: simd::SimdBackend::ArmSve,
            lane_bytes: 32,
            auto_vectorize: true,
        };
        let avx2 = simd::SimdPlan {
            backend: simd::SimdBackend::X86Avx2,
            lane_bytes: 32,
            auto_vectorize: true,
        };
        assert_eq!(simd_unroll_factor_for_plan(sve), 4);
        assert_eq!(simd_unroll_factor_for_plan(avx2), 4);
    }

    #[test]
    fn narrow_vector_plan_stays_unroll_1() {
        let plan = simd::SimdPlan {
            backend: simd::SimdBackend::WasmSimd128,
            lane_bytes: 8,
            auto_vectorize: true,
        };
        assert_eq!(simd_unroll_factor_for_plan(plan), 1);
    }

    #[test]
    fn detect_lowered_kernel_for_trig_unary() {
        let mut p = parser::Parser::new(parser::tokenize("math.sin(x)"));
        let expr = p.parse_expr().expect("parse trig");
        let args = vec!["x".to_string()];
        let kernel = detect_lowered_kernel(&expr, &args);
        assert_eq!(
            kernel,
            Some(LoweredKernel::Unary {
                op: LoweredUnaryKernel::Sin,
                input: 0,
            })
        );
    }

    #[test]
    fn detect_lowered_kernel_for_binary_arith() {
        let mut p = parser::Parser::new(parser::tokenize("a * b"));
        let expr = p.parse_expr().expect("parse binary");
        let args = vec!["a".to_string(), "b".to_string()];
        let kernel = detect_lowered_kernel(&expr, &args);
        assert_eq!(
            kernel,
            Some(LoweredKernel::Binary {
                op: LoweredBinaryKernel::Mul,
                lhs: 0,
                rhs: 1,
            })
        );
    }

    #[test]
    fn detect_lowered_kernel_for_complex_trig_exp_expr() {
        let src = "((math.sin(x) * 0.5 + x * 1.2) / (1.0 + math.exp(-abs(x) * 0.001)))";
        let mut p = parser::Parser::new(parser::tokenize(src));
        let expr = p.parse_expr().expect("parse complex lowered expr");
        let args = vec!["x".to_string()];
        let kernel = detect_lowered_kernel(&expr, &args);
        match kernel {
            Some(LoweredKernel::Expr(_)) => {}
            other => panic!("expected LoweredKernel::Expr(_), got {:?}", other),
        }
    }

    #[test]
    fn compile_sumover_complex_expr_selects_lowered_kernel() {
        let src = "sum(((math.sin(x) * 0.5 + x * 1.2) / (1.0 + math.exp(-abs(x) * 0.001)) for x in data))";
        let args = vec!["data".to_string()];
        let entry = compile_jit_impl(src, &args, true, JitReturnType::Float)
            .expect("compile sumover expression");
        match entry.lowered_kernel {
            Some(LoweredKernel::Expr(_)) => {}
            other => panic!("expected lowered expression kernel, got {:?}", other),
        }
    }

    #[test]
    fn detect_lowered_kernel_for_filtered_ternary_expr() {
        let src = "((x % 2 == 0 or x > 75.0) and (not x < 10.0)) ? (x > 50.0 ? x * math.sin(x) : x * math.cos(x)) : 0.0";
        let mut p = parser::Parser::new(parser::tokenize(src));
        let expr = p.parse_expr().expect("parse filtered ternary expr");
        let args = vec!["x".to_string()];
        let kernel = detect_lowered_kernel(&expr, &args);
        match kernel {
            Some(LoweredKernel::Expr(_)) => {}
            other => panic!("expected LoweredKernel::Expr(_), got {:?}", other),
        }
    }

    #[test]
    fn compile_sumover_filtered_ternary_selects_lowered_kernel() {
        let src = "sum((x * math.sin(x) if x > 50.0 else x * math.cos(x) for x in data if (x % 2 == 0 or x > 75.0) and (not x < 10.0)))";
        let args = vec!["data".to_string()];
        let entry = compile_jit_impl(src, &args, true, JitReturnType::Float)
            .expect("compile filtered ternary sumover expression");
        match entry.lowered_kernel {
            Some(LoweredKernel::Expr(_)) => {}
            other => panic!("expected lowered expression kernel, got {:?}", other),
        }
    }

    #[test]
    fn simd_math_mode_defaults_to_accurate() {
        std::env::remove_var("IRIS_JIT_SIMD_MATH");
        assert_eq!(simd_math_mode_from_env(), SimdMathMode::Accurate);
    }

    #[test]
    fn simd_math_mode_fast_parse() {
        std::env::set_var("IRIS_JIT_SIMD_MATH", "fast");
        assert_eq!(simd_math_mode_from_env(), SimdMathMode::FastApprox);
        std::env::remove_var("IRIS_JIT_SIMD_MATH");
    }

    #[test]
    fn fast_trig_approx_is_reasonable_near_common_angles() {
        let x = 0.7_f64;
        let sin_err = (fast_sin_approx(x) - x.sin()).abs();
        let cos_err = (fast_cos_approx(x) - x.cos()).abs();
        assert!(
            sin_err < 1e-3,
            "sin approximation error too high: {}",
            sin_err
        );
        assert!(
            cos_err < 1e-3,
            "cos approximation error too high: {}",
            cos_err
        );
    }

    #[test]
    fn fast_trig_approx_preserves_negative_quadrant_sign() {
        let x = -2.0_f64;
        let approx = fast_sin_approx(x);
        assert!(
            approx < 0.0,
            "expected sin(-2.0) approximation to be negative, got {}",
            approx
        );
    }

    #[test]
    fn fast_trig_approx_stays_bounded_for_large_inputs() {
        let samples = [10_000.0_f64, -10_000.0_f64, 100_000.0_f64, -100_000.0_f64];
        for &x in &samples {
            let s = fast_sin_approx(x);
            let c = fast_cos_approx(x);
            assert!(
                s.is_finite() && c.is_finite(),
                "expected finite fast trig outputs for x={}, got sin={} cos={}",
                x,
                s,
                c
            );
            assert!(
                (s - x.sin()).abs() < 5e-3,
                "sin approximation drift too large for x={}: approx={} exact={}",
                x,
                s,
                x.sin()
            );
            assert!(
                (c - x.cos()).abs() < 5e-3,
                "cos approximation drift too large for x={}: approx={} exact={}",
                x,
                c,
                x.cos()
            );
        }
    }

    #[test]
    fn quantum_dispatch_explores_unrun_variants_even_below_threshold() {
        let use_quantum = should_use_quantum_dispatch(3, 10.0, 0.0, 0.9, 1000, true);
        assert!(use_quantum);
    }

    #[test]
    fn quantum_dispatch_respects_threshold_when_fully_profiled() {
        let use_quantum = should_use_quantum_dispatch(3, 10.0, 0.95, 0.9, 1000, false);
        assert!(!use_quantum);

        let use_quantum_hot = should_use_quantum_dispatch(3, 5000.0, 0.95, 0.9, 1000, false);
        assert!(use_quantum_hot);
    }

    #[test]
    fn lowered_unary_eval_pair_any_matches_scalar_in_accurate_mode() {
        let x0 = 0.123_f64;
        let x1 = -1.234_f64;

        let (sin0, sin1) =
            lowered_unary_eval_pair_any(LoweredUnaryKernel::Sin, x0, x1, SimdMathMode::Accurate);
        let (exp0, exp1) =
            lowered_unary_eval_pair_any(LoweredUnaryKernel::Exp, x0, x1, SimdMathMode::Accurate);

        let eps = 1e-12;
        assert!((sin0 - x0.sin()).abs() < eps);
        assert!((sin1 - x1.sin()).abs() < eps);
        assert!((exp0 - x0.exp()).abs() < eps);
        assert!((exp1 - x1.exp()).abs() < eps);
    }
}
