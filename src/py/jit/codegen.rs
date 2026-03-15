// src/py/jit/codegen.rs
//! Core JIT compilation logic, including registry and Cranelift codegen.

use std::collections::{HashMap, HashSet};
use std::ffi::CStr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use crate::py::jit::parser::Expr;
use crate::py::jit::heuristics;
use crate::py::jit::simd;

// cranelift imports
use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use pyo3::AsPyPointer;
use pyo3::IntoPy;

const BREAK_SENTINEL_BITS: u64 = 0x7ff8_0000_0000_0b01;
const CONTINUE_SENTINEL_BITS: u64 = 0x7ff8_0000_0000_0c01;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SymbolAlias {
    Identity,
    Rename(&'static str),
}

pub(crate) fn resolve_symbol_alias(symbol: &str, arg_count: usize) -> Option<SymbolAlias> {
    match (symbol, arg_count) {
        ("float", 1) => Some(SymbolAlias::Identity),
        ("int", 1) => Some(SymbolAlias::Rename("trunc")),
        ("round", 1) => Some(SymbolAlias::Rename("round")),
        _ => None,
    }
}

/// A compiled function entry returned by the JIT.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReductionMode {
    None,
    Sum,
    Any,
    All,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JitReturnType {
    Float,
    Int,
    Bool,
}

impl Default for JitReturnType {
    fn default() -> Self {
        JitReturnType::Float
    }
}

#[derive(Clone)]
pub struct JitEntry {
    pub func_ptr: usize,
    pub arg_count: usize,
    pub reduction: ReductionMode,
    pub return_type: JitReturnType,
    lowered_kernel: Option<LoweredKernel>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LoweredUnaryKernel {
    Identity,
    Neg,
    Abs,
    Sin,
    Cos,
    Tan,
    Exp,
    Log,
    Sqrt,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LoweredBinaryKernel {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LoweredKernel {
    Unary { op: LoweredUnaryKernel, input: usize },
    Binary {
        op: LoweredBinaryKernel,
        lhs: usize,
        rhs: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SimdMathMode {
    Accurate,
    FastApprox,
}

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

fn arg_index_of_var(arg_names: &[String], var: &str) -> Option<usize> {
    arg_names.iter().position(|n| n == var)
}

fn normalize_intrinsic_name(raw: &str) -> &str {
    raw.rsplit('.').next().unwrap_or(raw)
}

fn detect_lowered_kernel(expr: &Expr, arg_names: &[String]) -> Option<LoweredKernel> {
    match expr {
        Expr::Var(name) => {
            let input = arg_index_of_var(arg_names, name)?;
            Some(LoweredKernel::Unary {
                op: LoweredUnaryKernel::Identity,
                input,
            })
        }
        Expr::UnaryOp('-', sub) => {
            let Expr::Var(name) = sub.as_ref() else {
                return None;
            };
            let input = arg_index_of_var(arg_names, name)?;
            Some(LoweredKernel::Unary {
                op: LoweredUnaryKernel::Neg,
                input,
            })
        }
        Expr::Call(name, args) => {
            if args.len() != 1 {
                return None;
            }
            let Expr::Var(var_name) = &args[0] else {
                return None;
            };
            let input = arg_index_of_var(arg_names, var_name)?;
            let op = match normalize_intrinsic_name(name).to_ascii_lowercase().as_str() {
                "abs" | "fabs" => LoweredUnaryKernel::Abs,
                "sin" => LoweredUnaryKernel::Sin,
                "cos" => LoweredUnaryKernel::Cos,
                "tan" => LoweredUnaryKernel::Tan,
                "exp" => LoweredUnaryKernel::Exp,
                "log" | "ln" => LoweredUnaryKernel::Log,
                "sqrt" => LoweredUnaryKernel::Sqrt,
                _ => return None,
            };
            Some(LoweredKernel::Unary { op, input })
        }
        Expr::BinOp(lhs, op, rhs) => {
            let Expr::Var(lhs_name) = lhs.as_ref() else {
                return None;
            };
            let Expr::Var(rhs_name) = rhs.as_ref() else {
                return None;
            };
            let lhs_idx = arg_index_of_var(arg_names, lhs_name)?;
            let rhs_idx = arg_index_of_var(arg_names, rhs_name)?;
            let op = match op.as_str() {
                "+" => LoweredBinaryKernel::Add,
                "-" => LoweredBinaryKernel::Sub,
                "*" => LoweredBinaryKernel::Mul,
                "/" => LoweredBinaryKernel::Div,
                _ => return None,
            };
            Some(LoweredKernel::Binary {
                op,
                lhs: lhs_idx,
                rhs: rhs_idx,
            })
        }
        _ => None,
    }
}

impl JitEntry {
    /// Check if this entry has a valid, compiled function pointer.
    pub fn is_valid(&self) -> bool {
        self.func_ptr != 0
    }
}

static JIT_FUNC_COUNTER: once_cell::sync::Lazy<AtomicUsize> =
    once_cell::sync::Lazy::new(|| AtomicUsize::new(0));

static JIT_REGISTRY: once_cell::sync::OnceCell<std::sync::Mutex<HashMap<usize, JitEntry>>> =
    once_cell::sync::OnceCell::new();

static NAMED_JIT_REGISTRY: once_cell::sync::OnceCell<std::sync::Mutex<HashMap<String, JitEntry>>> =
    once_cell::sync::OnceCell::new();

#[derive(Clone)]
struct QuantumStats {
    ewma_ns: f64,
    runs: u64,
    failures: u64,
}

#[derive(Clone, Debug)]
pub struct QuantumProfilePoint {
    pub index: usize,
    pub ewma_ns: f64,
    pub runs: u64,
    pub failures: u64,
}

#[derive(Clone, Debug)]
pub struct QuantumProfileSeed {
    pub index: usize,
    pub ewma_ns: f64,
    pub runs: u64,
    pub failures: u64,
}

impl Default for QuantumStats {
    fn default() -> Self {
        Self {
            ewma_ns: 0.0,
            runs: 0,
            failures: 0,
        }
    }
}

#[derive(Clone)]
struct QuantumState {
    entries: Vec<JitEntry>,
    stats: Vec<QuantumStats>,
    active: Vec<bool>,
    baseline_idx: usize,
    round_robin: usize,
    total_runs: u64,
}

static QUANTUM_REGISTRY: once_cell::sync::OnceCell<std::sync::Mutex<HashMap<usize, QuantumState>>> =
    once_cell::sync::OnceCell::new();
static QUANTUM_PENDING_SEEDS: once_cell::sync::OnceCell<std::sync::Mutex<HashMap<usize, Vec<QuantumProfileSeed>>>> =
    once_cell::sync::OnceCell::new();
static LOWERED_EXEC_LOGGED: once_cell::sync::OnceCell<std::sync::Mutex<HashSet<usize>>> =
    once_cell::sync::OnceCell::new();

fn log_lowered_exec_once(entry: &JitEntry, len: usize) {
    let Some(kernel) = entry.lowered_kernel else {
        return;
    };

    let set = LOWERED_EXEC_LOGGED.get_or_init(|| std::sync::Mutex::new(HashSet::new()));
    let mut guard = set.lock().unwrap();
    if !guard.insert(entry.func_ptr) {
        return;
    }

    let math_mode = simd_math_mode_from_env();
    crate::py::jit::jit_log(|| {
        format!(
            "[Iris][jit][lower] execute kernel={:?} reduction={:?} len={} math_mode={:?}",
            kernel, entry.reduction, len, math_mode
        )
    });
}

fn apply_quantum_seeds(state: &mut QuantumState, seeds: &[QuantumProfileSeed]) {
    for seed in seeds {
        if let Some(stats) = state.stats.get_mut(seed.index) {
            stats.ewma_ns = if seed.ewma_ns.is_finite() && seed.ewma_ns >= 0.0 {
                seed.ewma_ns
            } else {
                0.0
            };
            stats.runs = seed.runs;
            stats.failures = seed.failures;
        }
    }
}

pub fn register_jit(func_key: usize, entry: JitEntry) {
    let map = JIT_REGISTRY.get_or_init(|| std::sync::Mutex::new(HashMap::new()));
    let mut guard = map.lock().unwrap();
    guard.insert(func_key, entry);
}

pub fn lookup_jit(func_key: usize) -> Option<JitEntry> {
    JIT_REGISTRY
        .get()
        .and_then(|map| map.lock().unwrap().get(&func_key).cloned())
}

pub fn register_named_jit(name: &str, entry: JitEntry) {
    let map = NAMED_JIT_REGISTRY.get_or_init(|| std::sync::Mutex::new(HashMap::new()));
    let mut guard = map.lock().unwrap();
    guard.insert(name.to_string(), entry);
}

pub fn lookup_named_jit(name: &str) -> Option<JitEntry> {
    NAMED_JIT_REGISTRY
        .get()
        .and_then(|map| map.lock().unwrap().get(name).cloned())
}

fn invoke_named_entry(func_ptr: i64, args: &[f64]) -> f64 {
    let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(func_ptr as usize) };
    f(args.as_ptr())
}

macro_rules! make_invoke {
    ($name:ident) => {
        #[no_mangle]
        pub extern "C" fn $name(func_ptr: i64) -> f64 {
            let args: [f64; 0] = [];
            invoke_named_entry(func_ptr, &args)
        }
    };
    ($name:ident, $($arg:ident),+) => {
        #[no_mangle]
        pub extern "C" fn $name(func_ptr: i64, $($arg: f64),+) -> f64 {
            let args = [$($arg),+];
            invoke_named_entry(func_ptr, &args)
        }
    };
}

make_invoke!(iris_jit_invoke_0);
make_invoke!(iris_jit_invoke_1, a0);
make_invoke!(iris_jit_invoke_2, a0, a1);
make_invoke!(iris_jit_invoke_3, a0, a1, a2);
make_invoke!(iris_jit_invoke_4, a0, a1, a2, a3);
make_invoke!(iris_jit_invoke_5, a0, a1, a2, a3, a4);
make_invoke!(iris_jit_invoke_6, a0, a1, a2, a3, a4, a5);
make_invoke!(iris_jit_invoke_7, a0, a1, a2, a3, a4, a5, a6);
make_invoke!(iris_jit_invoke_8, a0, a1, a2, a3, a4, a5, a6, a7);
make_invoke!(iris_jit_invoke_9, a0, a1, a2, a3, a4, a5, a6, a7, a8);
make_invoke!(iris_jit_invoke_10, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9);
make_invoke!(iris_jit_invoke_11, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
make_invoke!(iris_jit_invoke_12, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11);
make_invoke!(iris_jit_invoke_13, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12);
make_invoke!(iris_jit_invoke_14, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13);
make_invoke!(iris_jit_invoke_15, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14);
make_invoke!(iris_jit_invoke_16, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15);

pub fn register_quantum_jit(func_key: usize, mut entries: Vec<JitEntry>) {
    if entries.is_empty() {
        return;
    }
    let entry_count = entries.len();
    // prefer optimized candidate (first) as baseline fallback mapping
    register_jit(func_key, entries[0].clone());
    let stats = vec![QuantumStats::default(); entries.len()];
    let mut state = QuantumState {
        entries: std::mem::take(&mut entries),
        stats,
        active: vec![true; entry_count],
        baseline_idx: 0,
        round_robin: 0,
        total_runs: 0,
    };

    if let Some(pending_map) = QUANTUM_PENDING_SEEDS.get() {
        if let Some(seeds) = pending_map.lock().unwrap().remove(&func_key) {
            apply_quantum_seeds(&mut state, &seeds);
        }
    }

    let map = QUANTUM_REGISTRY.get_or_init(|| std::sync::Mutex::new(HashMap::new()));
    map.lock().unwrap().insert(func_key, state);
}

pub fn quantum_has_seed_hint(func_key: usize) -> bool {
    QUANTUM_PENDING_SEEDS
        .get()
        .and_then(|map| map.lock().unwrap().get(&func_key).map(|rows| !rows.is_empty()))
        .unwrap_or(false)
}

fn active_indices(state: &QuantumState) -> Vec<usize> {
    state
        .active
        .iter()
        .enumerate()
        .filter_map(|(idx, is_active)| if *is_active { Some(idx) } else { None })
        .collect()
}

fn quantum_score(stats: &QuantumStats) -> f64 {
    let penalty = 1.0 + (stats.failures as f64 * 0.25);
    if stats.ewma_ns > 0.0 {
        stats.ewma_ns * penalty
    } else {
        f64::MAX / 4.0
    }
}

fn quantum_stability_score(state: &QuantumState) -> f64 {
    let active = active_indices(state);
    if active.len() <= 1 {
        return 1.0;
    }

    let mut total_runs = 0u64;
    let mut total_failures = 0u64;
    let mut min_ewma: f64 = f64::MAX;
    let mut max_ewma: f64 = 0.0;
    let mut ewma_count = 0u64;

    for idx in active {
        let s = &state.stats[idx];
        total_runs = total_runs.saturating_add(s.runs);
        total_failures = total_failures.saturating_add(s.failures);
        if s.runs > 0 && s.ewma_ns > 0.0 {
            ewma_count = ewma_count.saturating_add(1);
            min_ewma = min_ewma.min(s.ewma_ns);
            max_ewma = max_ewma.max(s.ewma_ns);
        }
    }

    let min_runs = crate::py::jit::quantum_stability_min_runs();
    if total_runs < min_runs {
        return 0.0;
    }

    let denom = (total_runs.saturating_add(total_failures)) as f64;
    let failure_rate = if denom > 0.0 {
        (total_failures as f64 / denom).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let dispersion = if ewma_count >= 2 && max_ewma > 0.0 && min_ewma < f64::MAX {
        ((max_ewma - min_ewma) / max_ewma).clamp(0.0, 1.0)
    } else {
        0.0
    };

    (1.0 - failure_rate) * (1.0 - dispersion)
}

fn reconcile_quantum_lifecycle_state(state: &mut QuantumState) {
    if state.entries.is_empty() {
        return;
    }

    let fail_limit = crate::py::jit::quantum_variant_failure_limit();
    let promotion_min_runs = crate::py::jit::quantum_variant_promotion_min_runs();
    let baseline = state.baseline_idx.min(state.entries.len() - 1);
    state.baseline_idx = baseline;

    for idx in 0..state.entries.len() {
        if idx == baseline {
            state.active[idx] = true;
            continue;
        }
        if !state.active[idx] {
            continue;
        }
        let s = &state.stats[idx];
        if s.failures >= fail_limit && s.runs < promotion_min_runs {
            state.active[idx] = false;
        }
    }

    let baseline_score = quantum_score(&state.stats[baseline]);
    let mut best_idx = baseline;
    let mut best_score = baseline_score;
    for idx in 0..state.entries.len() {
        if !state.active[idx] {
            continue;
        }
        let s = &state.stats[idx];
        if s.runs < promotion_min_runs {
            continue;
        }
        let score = quantum_score(s);
        if score < best_score * 0.90 {
            best_score = score;
            best_idx = idx;
        }
    }
    state.baseline_idx = best_idx;
    state.active[state.baseline_idx] = true;
}

#[cfg(test)]
pub(crate) fn reconcile_quantum_lifecycle(func_key: usize) -> bool {
    let Some(map) = QUANTUM_REGISTRY.get() else {
        return false;
    };
    let mut guard = map.lock().unwrap();
    let Some(state) = guard.get_mut(&func_key) else {
        return false;
    };
    reconcile_quantum_lifecycle_state(state);
    true
}

#[cfg(test)]
pub(crate) fn quantum_active_variant_count(func_key: usize) -> Option<usize> {
    QUANTUM_REGISTRY.get().and_then(|map| {
        map.lock().unwrap().get(&func_key).map(|state| {
            state.active.iter().filter(|a| **a).count()
        })
    })
}

#[cfg(test)]
pub(crate) fn quantum_stability_for(func_key: usize) -> Option<f64> {
    QUANTUM_REGISTRY.get().and_then(|map| {
        map.lock().unwrap().get(&func_key).map(quantum_stability_score)
    })
}

pub fn quantum_profile_snapshot(func_key: usize) -> Option<Vec<QuantumProfilePoint>> {
    QUANTUM_REGISTRY.get().and_then(|map| {
        map.lock().unwrap().get(&func_key).map(|state| {
            state
                .stats
                .iter()
                .enumerate()
                .map(|(index, stats)| QuantumProfilePoint {
                    index,
                    ewma_ns: stats.ewma_ns,
                    runs: stats.runs,
                    failures: stats.failures,
                })
                .collect()
        })
    })
}

pub fn seed_quantum_profile(func_key: usize, seeds: &[QuantumProfileSeed]) -> bool {
    if let Some(map) = QUANTUM_REGISTRY.get() {
        let mut guard = map.lock().unwrap();
        if let Some(state) = guard.get_mut(&func_key) {
            apply_quantum_seeds(state, seeds);
            return true;
        }
    }

    if seeds.is_empty() {
        return false;
    }

    let pending = QUANTUM_PENDING_SEEDS.get_or_init(|| std::sync::Mutex::new(HashMap::new()));
    pending.lock().unwrap().insert(func_key, seeds.to_vec());
    true
}

fn choose_quantum_index(state: &mut QuantumState) -> usize {
    for (idx, s) in state.stats.iter().enumerate() {
        if !state.active.get(idx).copied().unwrap_or(false) {
            continue;
        }
        if s.runs == 0 {
            return idx;
        }
    }

    let active = active_indices(state);
    if active.is_empty() {
        return state.baseline_idx.min(state.entries.len().saturating_sub(1));
    }

    state.total_runs = state.total_runs.saturating_add(1);
    if state.total_runs % 16 == 0 {
        let rr = state.round_robin % active.len();
        state.round_robin = (rr + 1) % active.len();
        return active[rr];
    }
    let mut best_idx = state.baseline_idx.min(state.entries.len().saturating_sub(1));
    let mut best_score = f64::MAX;
    for idx in active {
        let s = &state.stats[idx];
        let score = quantum_score(s);
        if score < best_score {
            best_score = score;
            best_idx = idx;
        }
    }
    best_idx
}

fn update_quantum_stats(func_key: usize, idx: usize, elapsed_ns: u64, success: bool) {
    if let Some(map) = QUANTUM_REGISTRY.get() {
        if let Some(state) = map.lock().unwrap().get_mut(&func_key) {
            if let Some(stats) = state.stats.get_mut(idx) {
                if success {
                    stats.runs = stats.runs.saturating_add(1);
                    let sample = elapsed_ns as f64;
                    if stats.ewma_ns <= 0.0 {
                        stats.ewma_ns = sample;
                    } else {
                        stats.ewma_ns = stats.ewma_ns * 0.80 + sample * 0.20;
                    }
                } else {
                    stats.failures = stats.failures.saturating_add(1);
                }
            }
                reconcile_quantum_lifecycle_state(state);
        }
    }
}

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
                            .fold(f64::MAX, |a, b| a.min(b));
                        let best_ewma = if best_ewma == f64::MAX { 0.0 } else { best_ewma };
                        let stability = quantum_stability_score(state);
                        let min_stability = crate::py::jit::quantum_stability_min_score();
                        let active_count = active_indices(state).len();
                        let use_quantum = active_count > 1
                            && best_ewma >= (speculation_threshold_ns as f64)
                            && stability >= min_stability;

                        if use_quantum {
                            let idx = choose_quantum_index(state);
                            let entry = state.entries[idx].clone();
                            // Check if the selected entry is valid; if not, skip quantum and use baseline.
                            if !entry.is_valid() {
                                let baseline_idx = state.baseline_idx.min(state.entries.len() - 1);
                                let baseline_entry = state.entries[baseline_idx].clone();
                                if !baseline_entry.is_valid() {
                                    // No valid entries at all; skip JIT
                                    (None, 0usize, Vec::new(), false, 0usize)
                                } else {
                                    (Some(baseline_entry), baseline_idx, Vec::new(), false, active_count)
                                }
                            } else {
                                let mut fallbacks = Vec::new();
                                for (i, e) in state.entries.iter().enumerate() {
                                    if i != idx && state.active.get(i).copied().unwrap_or(false) && e.is_valid() {
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
                let quantum_total = 1 + fallback_entries.len();
                let log_threshold_ns = crate::py::jit::quantum_log_threshold_ns();

                let log_if_slow = |chosen: usize,
                                   used: usize,
                                   elapsed_ns: u64,
                                   note: &str| {
                    if crate::py::jit::jit_logging_enabled() && elapsed_ns >= log_threshold_ns {
                        crate::py::jit::jit_log(|| {
                            format!(
                                "[Iris][jit][quantum] func_key={} chosen={}/{} used={} elapsed={}ns {}",
                                func_key, chosen, quantum_total, used, elapsed_ns, note
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
                            let _ = crate::py::jit::maybe_rearm_quantum_compile(func_key, elapsed, active_count);
                        }
                        log_if_slow(idx, idx, elapsed, if should_use_quantum { "success" } else { "baseline" });
                        return Some(Ok(obj));
                    }
                    Err(primary_err) => {
                        let elapsed = start.elapsed().as_nanos() as u64;
                        update_quantum_stats(func_key, idx, elapsed, false);
                        if !should_use_quantum {
                            let _ = crate::py::jit::maybe_rearm_quantum_compile(func_key, elapsed, active_count);
                        }
                        log_if_slow(idx, idx, elapsed, "primary_failed");
                        if should_use_quantum {
                            for (fb_idx, fb_entry) in fallback_entries {
                                let start_fb = Instant::now();
                                match execute_jit_func(py, &fb_entry, args) {
                                    Ok(obj) => {
                                        let elapsed_fb = start_fb.elapsed().as_nanos() as u64;
                                        update_quantum_stats(
                                            func_key,
                                            fb_idx,
                                            elapsed_fb,
                                            true,
                                        );
                                        log_if_slow(
                                            idx,
                                            fb_idx,
                                            elapsed_fb,
                                            "fallback_success",
                                        );
                                        return Some(Ok(obj));
                                    }
                                    Err(_) => {
                                        let elapsed_fb = start_fb.elapsed().as_nanos() as u64;
                                        update_quantum_stats(
                                            func_key,
                                            fb_idx,
                                            elapsed_fb,
                                            false,
                                        );
                                        log_if_slow(
                                            idx,
                                            fb_idx,
                                            elapsed_fb,
                                            "fallback_failed",
                                        );
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

thread_local! {
    static TLS_JIT_MODULE: std::cell::RefCell<Option<JITModule>> =
        std::cell::RefCell::new(None);
}

thread_local! {
    static TLS_JIT_TYPE_PROFILE: std::cell::RefCell<HashMap<usize, JitExecProfile>> =
        std::cell::RefCell::new(HashMap::new());
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BufferElemType {
    F64,
    F32,
    I64,
    I32,
    I16,
    I8,
    U64,
    U32,
    U16,
    U8,
    Bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum JitExecProfile {
    ScalarArgs,
    PackedBuffer {
        arg_count: usize,
        elem: BufferElemType,
    },
    VectorizedBuffers {
        arg_count: usize,
        elem_types: Vec<BufferElemType>,
    },
}

fn with_jit_module<F, R>(f: F) -> R
where
    F: FnOnce(&mut JITModule) -> R,
{
    TLS_JIT_MODULE.with(|cell| {
        let mut opt = cell.borrow_mut();
        if opt.is_none() {
            let build_isa = |plan: simd::SimdPlan| {
                let mut flag_builder = settings::builder();
                flag_builder.set("use_colocated_libcalls", "false").unwrap();
                if cfg!(target_arch = "aarch64") {
                    flag_builder.set("is_pic", "false").unwrap();
                } else {
                    flag_builder.set("is_pic", "true").unwrap();
                }
                simd::apply_cranelift_simd_flags(&mut flag_builder, plan);
                let _ = flag_builder.set("opt_level", "speed");

                let isa_builder = cranelift_native::builder().unwrap_or_else(|msg| {
                    panic!("host machine is not supported: {}", msg);
                });
                isa_builder.finish(settings::Flags::new(flag_builder))
            };

            let requested_plan = simd::auto_vectorization_plan();
            let (isa, active_plan) = match build_isa(requested_plan) {
                Ok(isa) => (isa, requested_plan),
                Err(err) if requested_plan.auto_vectorize => {
                    let fallback_plan = simd::SimdPlan::default();
                    crate::py::jit::jit_log(|| {
                        format!(
                            "[Iris][jit][simd] host rejected SIMD plan {:?} (err={:?}); falling back to scalar",
                            requested_plan, err
                        )
                    });
                    match build_isa(fallback_plan) {
                        Ok(isa) => (isa, fallback_plan),
                        Err(fallback_err) => {
                            panic!(
                                "failed to create ISA with SIMD plan ({:?}) and scalar fallback ({:?})",
                                err, fallback_err
                            )
                        }
                    }
                }
                Err(err) => panic!("failed to create ISA: {:?}", err),
            };

            crate::py::jit::jit_log(|| {
                format!(
                    "[Iris][jit][simd] backend={:?} lane_bytes={} auto_vectorize={}",
                    active_plan.backend,
                    active_plan.lane_bytes,
                    active_plan.auto_vectorize
                )
            });

            let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
            builder.symbol("iris_jit_invoke_0", iris_jit_invoke_0 as *const u8);
            builder.symbol("iris_jit_invoke_1", iris_jit_invoke_1 as *const u8);
            builder.symbol("iris_jit_invoke_2", iris_jit_invoke_2 as *const u8);
            builder.symbol("iris_jit_invoke_3", iris_jit_invoke_3 as *const u8);
            builder.symbol("iris_jit_invoke_4", iris_jit_invoke_4 as *const u8);
            builder.symbol("iris_jit_invoke_5", iris_jit_invoke_5 as *const u8);
            builder.symbol("iris_jit_invoke_6", iris_jit_invoke_6 as *const u8);
            builder.symbol("iris_jit_invoke_7", iris_jit_invoke_7 as *const u8);
            builder.symbol("iris_jit_invoke_8", iris_jit_invoke_8 as *const u8);
            builder.symbol("iris_jit_invoke_9", iris_jit_invoke_9 as *const u8);
            builder.symbol("iris_jit_invoke_10", iris_jit_invoke_10 as *const u8);
            builder.symbol("iris_jit_invoke_11", iris_jit_invoke_11 as *const u8);
            builder.symbol("iris_jit_invoke_12", iris_jit_invoke_12 as *const u8);
            builder.symbol("iris_jit_invoke_13", iris_jit_invoke_13 as *const u8);
            builder.symbol("iris_jit_invoke_14", iris_jit_invoke_14 as *const u8);
            builder.symbol("iris_jit_invoke_15", iris_jit_invoke_15 as *const u8);
            builder.symbol("iris_jit_invoke_16", iris_jit_invoke_16 as *const u8);
            *opt = Some(JITModule::new(builder));
        }
        let module = opt.as_mut().unwrap();
        f(module)
    })
}

fn compile_jit_impl(
    expr_str: &str,
    arg_names: &[String],
    optimize_ast: bool,
    return_type: JitReturnType,
) -> Option<JitEntry> {
    // tokenize and parse
    let tokens = crate::py::jit::parser::tokenize(expr_str);
    let mut parser = crate::py::jit::parser::Parser::new(tokens);
    let mut expr = parser.parse_expr()?;
    crate::py::jit::jit_log(|| format!("[Iris][jit] parsed AST for '{}': {:?}", expr_str, expr));
    // detect generator-style loop over a container and convert to body-only
    // expression.  Python wrapper will pass the container buffer and the
    // JIT runtime will vectorize across it; the compiled function gets a
    // single scalar argument representing each element.
    let mut adjusted_args = arg_names.to_vec();
    let mut reduction = ReductionMode::None;
    // use a cloned copy when destructuring to release borrow immediately
    if let Expr::SumOver { iter_var, container, body, pred } = expr.clone() {
        if let Expr::Var(ref cont_name) = *container {
            if adjusted_args.len() == 1 && adjusted_args[0] == *cont_name {
                crate::py::jit::jit_log(|| {
                    format!("[Iris][jit] converting SumOver '{}' in {}", iter_var, cont_name)
                });
                expr = if let Some(p) = pred {
                    Expr::Ternary(p, body, Box::new(Expr::Const(0.0)))
                } else {
                    *body.clone()
                };
                adjusted_args = vec![iter_var.clone()];
                reduction = ReductionMode::Sum;
            }
        }
    }
    if let Expr::AnyOver { iter_var, container, body, pred } = expr.clone() {
        if let Expr::Var(ref cont_name) = *container {
            if adjusted_args.len() == 1 && adjusted_args[0] == *cont_name {
                crate::py::jit::jit_log(|| {
                    format!("[Iris][jit] converting AnyOver '{}' in {}", iter_var, cont_name)
                });
                expr = if let Some(p) = pred {
                    Expr::Ternary(p, body, Box::new(Expr::Const(0.0)))
                } else {
                    *body.clone()
                };
                adjusted_args = vec![iter_var.clone()];
                reduction = ReductionMode::Any;
            }
        }
    }
    if let Expr::AllOver { iter_var, container, body, pred } = expr.clone() {
        if let Expr::Var(ref cont_name) = *container {
            if adjusted_args.len() == 1 && adjusted_args[0] == *cont_name {
                crate::py::jit::jit_log(|| {
                    format!("[Iris][jit] converting AllOver '{}' in {}", iter_var, cont_name)
                });
                expr = if let Some(p) = pred {
                    Expr::Ternary(p, body, Box::new(Expr::Const(1.0)))
                } else {
                    *body.clone()
                };
                adjusted_args = vec![iter_var.clone()];
                reduction = ReductionMode::All;
            }
        }
    }
    if optimize_ast {
        expr = heuristics::optimize(expr);
        crate::py::jit::jit_log(|| format!("[Iris][jit] optimized AST: {:?}", expr));
    } else {
        crate::py::jit::jit_log(|| format!("[Iris][jit] baseline AST (no-opt): {:?}", expr));
    }
    let arg_count = adjusted_args.len();
    let lowered_kernel = detect_lowered_kernel(&expr, &adjusted_args);
    if let Some(kernel) = lowered_kernel {
        crate::py::jit::jit_log(|| format!("[Iris][jit][lower] selected kernel={:?}", kernel));
    }

    // perform compilation using the thread-local module instance;
    // the closure returns the resulting pointer so we can pass it back.
    with_jit_module(|module| {
        let mut ctx = module.make_context();
        ctx.func.signature.params.push(AbiParam::new(types::I64));
        ctx.func.signature.returns.push(AbiParam::new(types::F64));

        let mut func_ctx = FunctionBuilderContext::new();
        {
            let mut fb = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);
            let entry = fb.create_block();
            fb.append_block_params_for_function_params(entry);
            fb.switch_to_block(entry);
            fb.seal_block(entry);
            let ptr_val = fb.block_params(entry)[0];
            let locals = HashMap::new();
            let val = gen_expr(&expr, &mut fb, ptr_val, arg_names, module, &locals);
            fb.ins().return_(&[val]);
            fb.finalize();
        }

        let idx = JIT_FUNC_COUNTER.fetch_add(1, Ordering::SeqCst);
        let func_name = format!("jit_func_{}", idx);
        let id = module
            .declare_function(&func_name, Linkage::Local, &ctx.func.signature)
            .ok();
        if id.is_none() {
            return None;
        }
        let id = id.unwrap();
        if let Err(err) = module.define_function(id, &mut ctx) {
            crate::py::jit::jit_log(|| format!("[Iris][jit] define_function failed: {:?}", err));
            return None;
        }
        module.clear_context(&mut ctx);
        module.finalize_definitions();

        let code_ptr = module.get_finalized_function(id) as usize;
        Some(JitEntry {
            func_ptr: code_ptr,
            arg_count,
            reduction,
            return_type,
            lowered_kernel,
        })
    })
}

/// Compile an expression string into a native function entry.
#[allow(dead_code)]
pub fn compile_jit(expr_str: &str, arg_names: &[String]) -> Option<JitEntry> {
    compile_jit_with_return_type(expr_str, arg_names, JitReturnType::Float)
}

pub fn compile_jit_with_return_type(
    expr_str: &str,
    arg_names: &[String],
    return_type: JitReturnType,
) -> Option<JitEntry> {
    compile_jit_impl(expr_str, arg_names, true, return_type)
}

/// Compile multiple speculative versions of the same expression in parallel.
pub fn compile_jit_quantum(expr_str: &str, arg_names: &[String], return_type: JitReturnType) -> Vec<JitEntry> {
    let optimized = compile_jit_impl(expr_str, arg_names, true, return_type);
    let baseline = compile_jit_impl(expr_str, arg_names, false, return_type);

    let mut out = Vec::new();
    if let Some(e) = optimized {
        out.push(e);
    }
    if let Some(e) = baseline {
        out.push(e);
    }
    out
}

fn gen_expr(
    expr: &Expr,
    fb: &mut FunctionBuilder,
    ptr: Value,
    arg_names: &[String],
    module: &mut JITModule,
    locals: &HashMap<String, Value>,
) -> Value {

    #[derive(Clone, Copy)]
    enum LoopControl<'a> {
        None,
        Break { cond: &'a Expr, value: &'a Expr, invert_cond: bool },
        Continue { cond: &'a Expr, value: &'a Expr, invert_cond: bool },
    }

    fn detect_loop_control(expr: &Expr) -> LoopControl<'_> {
        match expr {
            Expr::Call(name, args) if args.len() == 2 => {
                let symbol = name.rsplit('.').next().unwrap_or(name.as_str());
                match symbol {
                    "break_if" | "loop_break_if" | "break_when" | "loop_break_when" => LoopControl::Break {
                        cond: &args[0],
                        value: &args[1],
                        invert_cond: false,
                    },
                    "break_unless" | "loop_break_unless" => LoopControl::Break {
                        cond: &args[0],
                        value: &args[1],
                        invert_cond: true,
                    },
                    "continue_if" | "loop_continue_if" | "continue_when" | "loop_continue_when" => LoopControl::Continue {
                        cond: &args[0],
                        value: &args[1],
                        invert_cond: false,
                    },
                    "continue_unless" | "loop_continue_unless" => LoopControl::Continue {
                        cond: &args[0],
                        value: &args[1],
                        invert_cond: true,
                    },
                    _ => LoopControl::None,
                }
            }
            _ => LoopControl::None,
        }
    }

    match expr {
        Expr::Const(n) => fb.ins().f64const(*n),
        Expr::Var(name) => {
            if let Some(v) = locals.get(name) {
                return *v;
            }
            if let Some(idx) = arg_names.iter().position(|n| n == name) {
                let offset = (idx as i64) * 8;
                let offset_const = fb.ins().iconst(types::I64, offset);
                let addr1 = fb.ins().iadd(ptr, offset_const);
                return fb.ins().load(types::F64, MemFlags::new(), addr1, 0);
            }
            // treat named constants
            match name.as_str() {
                "pi" => return fb.ins().f64const(std::f64::consts::PI),
                "e" => return fb.ins().f64const(std::f64::consts::E),
                _ => {}
            }
            let idx = 0;
            let offset = (idx as i64) * 8;
            let offset_const = fb.ins().iconst(types::I64, offset);
            let addr1 = fb.ins().iadd(ptr, offset_const);
            fb.ins().load(types::F64, MemFlags::new(), addr1, 0)
        }
        Expr::BinOp(lhs, op, rhs) => {
            let l = gen_expr(lhs, fb, ptr, arg_names, module, locals);
            let r = gen_expr(rhs, fb, ptr, arg_names, module, locals);
            match op.as_str() {
                "+" => fb.ins().fadd(l, r),
                "-" => fb.ins().fsub(l, r),
                "*" => fb.ins().fmul(l, r),
                "/" => fb.ins().fdiv(l, r),
                "%" => {
                    let mut sig = module.make_signature();
                    sig.params.push(AbiParam::new(types::F64));
                    sig.params.push(AbiParam::new(types::F64));
                    sig.returns.push(AbiParam::new(types::F64));
                    let fid = module
                        .declare_function("fmod", Linkage::Import, &sig)
                        .expect("failed to declare fmod");
                    let local = module.declare_func_in_func(fid, &mut fb.func);
                    let call = fb.ins().call(local, &[l, r]);
                    fb.inst_results(call)[0]
                }
                "**" => {
                    // existing exponent handling (unchanged)
                    if let Expr::Const(n) = **rhs {
                        if n == 1.0 {
                            return l;
                        }
                        if n == -1.0 {
                            let one = fb.ins().f64const(1.0);
                            return fb.ins().fdiv(one, l);
                        }
                        if n == 0.5 {
                            let mut sig = module.make_signature();
                            sig.params.push(AbiParam::new(types::F64));
                            sig.returns.push(AbiParam::new(types::F64));
                            let fid = module
                                .declare_function("sqrt", Linkage::Import, &sig)
                                .expect("failed to declare sqrt");
                            let local = module.declare_func_in_func(fid, &mut fb.func);
                            let call = fb.ins().call(local, &[l]);
                            return fb.inst_results(call)[0];
                        }
                        if n.fract() == 0.0 {
                            let exp = n as i64;
                            if exp == 0 {
                                return fb.ins().f64const(1.0);
                            } else if exp > 0 {
                                let mut e = exp as u64;
                                let mut base_val = l;
                                let mut acc = fb.ins().f64const(1.0);
                                while e > 0 {
                                    if e & 1 == 1 {
                                        acc = fb.ins().fmul(acc, base_val);
                                    }
                                    e >>= 1;
                                    if e > 0 {
                                        base_val = fb.ins().fmul(base_val, base_val);
                                    }
                                }
                                return acc;
                            }
                        }
                    }
                    let mut sig = module.make_signature();
                    sig.params.push(AbiParam::new(types::F64));
                    sig.params.push(AbiParam::new(types::F64));
                    sig.returns.push(AbiParam::new(types::F64));
                    let fid = module
                        .declare_function("pow", Linkage::Import, &sig)
                        .expect("failed to declare pow");
                    let local = module.declare_func_in_func(fid, &mut fb.func);
                    let call = fb.ins().call(local, &[l, r]);
                    fb.inst_results(call)[0]
                }
                "<" | ">" | "<=" | ">=" | "==" | "!=" => {
                    // comparison produces 1.0/0.0
                    let cc = match op.as_str() {
                        "<" => FloatCC::LessThan,
                        ">" => FloatCC::GreaterThan,
                        "<=" => FloatCC::LessThanOrEqual,
                        ">=" => FloatCC::GreaterThanOrEqual,
                        "==" => FloatCC::Equal,
                        "!=" => FloatCC::NotEqual,
                        _ => unreachable!(),
                    };
                    let boolv = fb.ins().fcmp(cc, l, r);
                    let intv = fb.ins().bint(types::I64, boolv);
                    fb.ins().fcvt_from_sint(types::F64, intv)
                }
                "and" | "or" => {
                    let zero = fb.ins().f64const(0.0);
                    let l_true = fb.ins().fcmp(FloatCC::NotEqual, l, zero);
                    let r_true = fb.ins().fcmp(FloatCC::NotEqual, r, zero);
                    let boolv = if op == "and" {
                        fb.ins().band(l_true, r_true)
                    } else {
                        fb.ins().bor(l_true, r_true)
                    };
                    let intv = fb.ins().bint(types::I64, boolv);
                    fb.ins().fcvt_from_sint(types::F64, intv)
                }
                _ => fb.ins().fadd(l, r),
            }
        }
        Expr::UnaryOp(op, sub) => {
            let v = gen_expr(sub, fb, ptr, arg_names, module, locals);
            match op {
                '-' => {
                    let zero = fb.ins().f64const(0.0);
                    fb.ins().fsub(zero, v)
                }
                '!' => {
                    let zero = fb.ins().f64const(0.0);
                    let boolv = fb.ins().fcmp(FloatCC::Equal, v, zero);
                    let intv = fb.ins().bint(types::I64, boolv);
                    fb.ins().fcvt_from_sint(types::F64, intv)
                }
                '+' => v,
                _ => v,
            }
        }
        Expr::Call(name, args) => {
            let symbol = name.rsplit('.').next().unwrap().to_string();
            if (symbol == "any_while" || symbol == "all_while") && args.len() == 5 {
                if let Expr::Var(iter_name) = &args[0] {
                    let init_val = gen_expr(&args[1], fb, ptr, arg_names, module, locals);
                    let cond_expr = &args[2];
                    let step_expr = &args[3];
                    let body_expr_raw = &args[4];

                    let zero = fb.ins().f64const(0.0);
                    let one = fb.ins().f64const(1.0);
                    let false_mask = fb.ins().fcmp(FloatCC::Equal, zero, one);
                    let true_mask = fb.ins().fcmp(FloatCC::Equal, zero, zero);
                    let acc_init = if symbol == "any_while" { zero } else { one };

                    let loop_block = fb.create_block();
                    let body_block = fb.create_block();
                    let continue_block = fb.create_block();
                    let exit_block = fb.create_block();
                    fb.append_block_param(loop_block, types::F64); // iter
                    fb.append_block_param(loop_block, types::F64); // acc
                    fb.append_block_param(loop_block, types::I64); // budget
                    fb.append_block_param(continue_block, types::F64); // next_iter
                    fb.append_block_param(continue_block, types::F64); // next_acc
                    fb.append_block_param(continue_block, types::I64); // next_budget
                    fb.append_block_param(exit_block, types::F64); // result

                    let max_iters = fb.ins().iconst(types::I64, 1_000_000);
                    fb.ins().jump(loop_block, &[init_val, acc_init, max_iters]);

                    fb.switch_to_block(loop_block);
                    let iter_val = fb.block_params(loop_block)[0];
                    let acc_val = fb.block_params(loop_block)[1];
                    let budget_val = fb.block_params(loop_block)[2];
                    let budget_exhausted = fb.ins().icmp_imm(IntCC::Equal, budget_val, 0);
                    let budget_ok_block = fb.create_block();
                    let budget_exit_block = fb.create_block();
                    fb.ins().brnz(budget_exhausted, budget_exit_block, &[]);
                    fb.ins().jump(budget_ok_block, &[]);

                    fb.switch_to_block(budget_exit_block);
                    fb.ins().jump(exit_block, &[acc_val]);

                    fb.switch_to_block(budget_ok_block);
                    let mut while_locals = locals.clone();
                    while_locals.insert(iter_name.clone(), iter_val);
                    let cond_val = gen_expr(cond_expr, fb, ptr, arg_names, module, &while_locals);
                    let cond_true = fb.ins().fcmp(FloatCC::NotEqual, cond_val, zero);
                    fb.ins().brnz(cond_true, body_block, &[]);
                    fb.ins().jump(exit_block, &[acc_val]);

                    fb.switch_to_block(body_block);
                    let ctrl = detect_loop_control(body_expr_raw);
                    let body_expr = match ctrl {
                        LoopControl::Break { value, .. } | LoopControl::Continue { value, .. } => value,
                        LoopControl::None => body_expr_raw,
                    };

                    let break_true = if let LoopControl::Break { cond, .. } = ctrl {
                        let break_cond_val = gen_expr(cond, fb, ptr, arg_names, module, &while_locals);
                        fb.ins().fcmp(FloatCC::NotEqual, break_cond_val, zero)
                    } else {
                        false_mask
                    };
                    let break_true = if let LoopControl::Break { invert_cond, .. } = ctrl {
                        if invert_cond {
                            fb.ins().bnot(break_true)
                        } else {
                            break_true
                        }
                    } else {
                        break_true
                    };

                    let continue_true = if let LoopControl::Continue { cond, .. } = ctrl {
                        let continue_cond_val =
                            gen_expr(cond, fb, ptr, arg_names, module, &while_locals);
                        fb.ins().fcmp(FloatCC::NotEqual, continue_cond_val, zero)
                    } else {
                        false_mask
                    };
                    let continue_true = if let LoopControl::Continue { invert_cond, .. } = ctrl {
                        if invert_cond {
                            fb.ins().bnot(continue_true)
                        } else {
                            continue_true
                        }
                    } else {
                        continue_true
                    };

                    let body_val = gen_expr(body_expr, fb, ptr, arg_names, module, &while_locals);
                    let step_val = gen_expr(step_expr, fb, ptr, arg_names, module, &while_locals);
                    let budget_next = fb.ins().iadd_imm(budget_val, -1);
                    let body_bits = fb.ins().bitcast(types::I64, body_val);
                    let is_break_sentinel =
                        fb.ins().icmp_imm(IntCC::Equal, body_bits, BREAK_SENTINEL_BITS as i64);
                    let is_continue_sentinel =
                        fb.ins().icmp_imm(IntCC::Equal, body_bits, CONTINUE_SENTINEL_BITS as i64);
                    let stop_now = fb.ins().bor(break_true, is_break_sentinel);
                    let skip_body = fb.ins().bor(continue_true, is_continue_sentinel);

                    let body_true = fb.ins().fcmp(FloatCC::NotEqual, body_val, zero);
                    if symbol == "any_while" {
                        let effective_true = fb.ins().select(skip_body, false_mask, body_true);
                        let acc_true = fb.ins().fcmp(FloatCC::NotEqual, acc_val, zero);
                        let next_true = fb.ins().bor(acc_true, effective_true);
                        let stop_any = fb.ins().bor(stop_now, next_true);
                        let any_exit_block = fb.create_block();
                        fb.ins().brnz(stop_any, any_exit_block, &[]);
                        fb.ins().jump(continue_block, &[step_val, zero, budget_next]);

                        fb.switch_to_block(any_exit_block);
                        let exit_val = fb.ins().select(stop_now, acc_val, one);
                        fb.ins().jump(exit_block, &[exit_val]);
                        fb.seal_block(any_exit_block);
                    } else {
                        let effective_true = fb.ins().select(skip_body, true_mask, body_true);
                        let acc_true = fb.ins().fcmp(FloatCC::NotEqual, acc_val, zero);
                        let next_true = fb.ins().band(acc_true, effective_true);
                        let next_false = fb.ins().bnot(next_true);
                        let stop_all = fb.ins().bor(stop_now, next_false);
                        let all_exit_block = fb.create_block();
                        fb.ins().brnz(stop_all, all_exit_block, &[]);
                        fb.ins().jump(continue_block, &[step_val, one, budget_next]);

                        fb.switch_to_block(all_exit_block);
                        let exit_val = fb.ins().select(stop_now, acc_val, zero);
                        fb.ins().jump(exit_block, &[exit_val]);
                        fb.seal_block(all_exit_block);
                    }

                    fb.switch_to_block(continue_block);
                    let next_iter = fb.block_params(continue_block)[0];
                    let next_acc = fb.block_params(continue_block)[1];
                    let next_budget = fb.block_params(continue_block)[2];
                    fb.ins().jump(loop_block, &[next_iter, next_acc, next_budget]);

                    fb.seal_block(body_block);
                    fb.seal_block(budget_exit_block);
                    fb.seal_block(budget_ok_block);
                    fb.seal_block(continue_block);
                    fb.seal_block(loop_block);
                    fb.switch_to_block(exit_block);
                    fb.seal_block(exit_block);
                    return fb.block_params(exit_block)[0];
                }
            }

            if symbol == "sum_while" && args.len() == 5 {
                if let Expr::Var(iter_name) = &args[0] {
                    let init_val = gen_expr(&args[1], fb, ptr, arg_names, module, locals);
                    let cond_expr = &args[2];
                    let step_expr = &args[3];
                    let body_expr_raw = &args[4];

                    let zero = fb.ins().f64const(0.0);
                    let one = fb.ins().f64const(1.0);
                    let false_mask = fb.ins().fcmp(FloatCC::Equal, zero, one);
                    let acc_init = zero;

                    let loop_block = fb.create_block();
                    let body_block = fb.create_block();
                    let continue_block = fb.create_block();
                    let exit_block = fb.create_block();
                    fb.append_block_param(loop_block, types::F64); // iter
                    fb.append_block_param(loop_block, types::F64); // acc
                    fb.append_block_param(loop_block, types::I64); // budget
                    fb.append_block_param(continue_block, types::F64); // next_iter
                    fb.append_block_param(continue_block, types::F64); // next_acc
                    fb.append_block_param(continue_block, types::I64); // next_budget
                    fb.append_block_param(exit_block, types::F64); // result

                    let max_iters = fb.ins().iconst(types::I64, 1_000_000);
                    fb.ins().jump(loop_block, &[init_val, acc_init, max_iters]);

                    fb.switch_to_block(loop_block);
                    let iter_val = fb.block_params(loop_block)[0];
                    let acc_val = fb.block_params(loop_block)[1];
                    let budget_val = fb.block_params(loop_block)[2];
                    let budget_exhausted = fb.ins().icmp_imm(IntCC::Equal, budget_val, 0);
                    let budget_ok_block = fb.create_block();
                    let budget_exit_block = fb.create_block();
                    fb.ins().brnz(budget_exhausted, budget_exit_block, &[]);
                    fb.ins().jump(budget_ok_block, &[]);

                    fb.switch_to_block(budget_exit_block);
                    fb.ins().jump(exit_block, &[acc_val]);

                    fb.switch_to_block(budget_ok_block);
                    let mut while_locals = locals.clone();
                    while_locals.insert(iter_name.clone(), iter_val);
                    let cond_val = gen_expr(cond_expr, fb, ptr, arg_names, module, &while_locals);
                    let cond_true = fb.ins().fcmp(FloatCC::NotEqual, cond_val, zero);
                    fb.ins().brnz(cond_true, body_block, &[]);
                    fb.ins().jump(exit_block, &[acc_val]);

                    fb.switch_to_block(body_block);
                    let ctrl = detect_loop_control(body_expr_raw);
                    let body_expr = match ctrl {
                        LoopControl::Break { value, .. } | LoopControl::Continue { value, .. } => value,
                        LoopControl::None => body_expr_raw,
                    };

                    let break_true = if let LoopControl::Break { cond, .. } = ctrl {
                        let break_cond_val = gen_expr(cond, fb, ptr, arg_names, module, &while_locals);
                        fb.ins().fcmp(FloatCC::NotEqual, break_cond_val, zero)
                    } else {
                        false_mask
                    };
                    let break_true = if let LoopControl::Break { invert_cond, .. } = ctrl {
                        if invert_cond {
                            fb.ins().bnot(break_true)
                        } else {
                            break_true
                        }
                    } else {
                        break_true
                    };

                    let continue_true = if let LoopControl::Continue { cond, .. } = ctrl {
                        let continue_cond_val =
                            gen_expr(cond, fb, ptr, arg_names, module, &while_locals);
                        fb.ins().fcmp(FloatCC::NotEqual, continue_cond_val, zero)
                    } else {
                        false_mask
                    };
                    let continue_true = if let LoopControl::Continue { invert_cond, .. } = ctrl {
                        if invert_cond {
                            fb.ins().bnot(continue_true)
                        } else {
                            continue_true
                        }
                    } else {
                        continue_true
                    };

                    let body_val = gen_expr(body_expr, fb, ptr, arg_names, module, &while_locals);
                    let step_val = gen_expr(step_expr, fb, ptr, arg_names, module, &while_locals);
                    let budget_next = fb.ins().iadd_imm(budget_val, -1);
                    let body_bits = fb.ins().bitcast(types::I64, body_val);
                    let is_break_sentinel =
                        fb.ins().icmp_imm(IntCC::Equal, body_bits, BREAK_SENTINEL_BITS as i64);
                    let is_continue_sentinel =
                        fb.ins().icmp_imm(IntCC::Equal, body_bits, CONTINUE_SENTINEL_BITS as i64);
                    let stop_now = fb.ins().bor(break_true, is_break_sentinel);
                    let skip_body = fb.ins().bor(continue_true, is_continue_sentinel);

                    let effective_body = fb.ins().select(skip_body, zero, body_val);
                    let next_acc = fb.ins().fadd(acc_val, effective_body);
                    let sum_break_block = fb.create_block();
                    fb.ins().brnz(stop_now, sum_break_block, &[]);
                    fb.ins().jump(continue_block, &[step_val, next_acc, budget_next]);
                    fb.switch_to_block(sum_break_block);
                    fb.ins().jump(exit_block, &[acc_val]);
                    fb.seal_block(sum_break_block);

                    fb.switch_to_block(continue_block);
                    let next_iter = fb.block_params(continue_block)[0];
                    let next_acc = fb.block_params(continue_block)[1];
                    let next_budget = fb.block_params(continue_block)[2];
                    fb.ins().jump(loop_block, &[next_iter, next_acc, next_budget]);

                    fb.seal_block(body_block);
                    fb.seal_block(budget_exit_block);
                    fb.seal_block(budget_ok_block);
                    fb.seal_block(continue_block);
                    fb.seal_block(loop_block);
                    fb.switch_to_block(exit_block);
                    fb.seal_block(exit_block);
                    return fb.block_params(exit_block)[0];
                }
            }

            if (symbol == "break_on_nan" || symbol == "loop_break_on_nan"
                || symbol == "continue_on_nan" || symbol == "loop_continue_on_nan")
                && args.len() == 1
            {
                let value_val = gen_expr(&args[0], fb, ptr, arg_names, module, locals);
                let is_nan = fb.ins().fcmp(FloatCC::Unordered, value_val, value_val);
                let sentinel = if symbol == "break_on_nan" || symbol == "loop_break_on_nan" {
                    fb.ins().f64const(f64::from_bits(BREAK_SENTINEL_BITS))
                } else {
                    fb.ins().f64const(f64::from_bits(CONTINUE_SENTINEL_BITS))
                };
                return fb.ins().select(is_nan, sentinel, value_val);
            }

            if symbol == "let_bind" && args.len() == 3 {
                if let Expr::Var(var_name) = &args[0] {
                    let v = gen_expr(&args[1], fb, ptr, arg_names, module, locals);
                    let mut new_locals = locals.clone();
                    new_locals.insert(var_name.clone(), v);
                    return gen_expr(&args[2], fb, ptr, arg_names, module, &new_locals);
                }
            }

            if symbol == "if_else" && args.len() == 3 {
                let cond_val = gen_expr(&args[0], fb, ptr, arg_names, module, locals);
                let then_val = gen_expr(&args[1], fb, ptr, arg_names, module, locals);
                let else_val = gen_expr(&args[2], fb, ptr, arg_names, module, locals);
                let zero = fb.ins().f64const(0.0);
                let cond_true = fb.ins().fcmp(FloatCC::NotEqual, cond_val, zero);
                return fb.ins().select(cond_true, then_val, else_val);
            }

            if let Some(alias) = resolve_symbol_alias(&symbol, args.len()) {
                match alias {
                    SymbolAlias::Identity => {
                        return gen_expr(&args[0], fb, ptr, arg_names, module, locals);
                    }
                    SymbolAlias::Rename(target) => {
                        let mut arg_vals = Vec::with_capacity(args.len());
                        for a in args {
                            arg_vals.push(gen_expr(a, fb, ptr, arg_names, module, locals));
                        }
                        let mut sig = module.make_signature();
                        for _ in 0..arg_vals.len() {
                            sig.params.push(AbiParam::new(types::F64));
                        }
                        sig.returns.push(AbiParam::new(types::F64));
                        let func_id = module
                            .declare_function(target, Linkage::Import, &sig)
                            .expect("failed to declare external function");
                        let local = module.declare_func_in_func(func_id, &mut fb.func);
                        let call = fb.ins().call(local, &arg_vals);
                        return fb.inst_results(call)[0];
                    }
                }
            }

            if symbol == "min" && args.len() == 2 {
                let a = gen_expr(&args[0], fb, ptr, arg_names, module, locals);
                let b = gen_expr(&args[1], fb, ptr, arg_names, module, locals);
                let cond = fb.ins().fcmp(FloatCC::LessThanOrEqual, a, b);
                return fb.ins().select(cond, a, b);
            }

            if symbol == "max" && args.len() == 2 {
                let a = gen_expr(&args[0], fb, ptr, arg_names, module, locals);
                let b = gen_expr(&args[1], fb, ptr, arg_names, module, locals);
                let cond = fb.ins().fcmp(FloatCC::GreaterThanOrEqual, a, b);
                return fb.ins().select(cond, a, b);
            }

            if let Some(named) = lookup_named_jit(&symbol) {
                if named.arg_count == args.len() {
                    let helper_name = match args.len() {
                        0 => Some("iris_jit_invoke_0"),
                        1 => Some("iris_jit_invoke_1"),
                        2 => Some("iris_jit_invoke_2"),
                        3 => Some("iris_jit_invoke_3"),
                        4 => Some("iris_jit_invoke_4"),
                        5 => Some("iris_jit_invoke_5"),
                        6 => Some("iris_jit_invoke_6"),
                        7 => Some("iris_jit_invoke_7"),
                        8 => Some("iris_jit_invoke_8"),
                        9 => Some("iris_jit_invoke_9"),
                        10 => Some("iris_jit_invoke_10"),
                        11 => Some("iris_jit_invoke_11"),
                        12 => Some("iris_jit_invoke_12"),
                        13 => Some("iris_jit_invoke_13"),
                        14 => Some("iris_jit_invoke_14"),
                        15 => Some("iris_jit_invoke_15"),
                        16 => Some("iris_jit_invoke_16"),
                        _ => None,
                    };

                    if let Some(helper_name) = helper_name {
                        let mut arg_vals = Vec::with_capacity(args.len() + 1);
                        arg_vals.push(fb.ins().iconst(types::I64, named.func_ptr as i64));
                        for a in args {
                            arg_vals.push(gen_expr(a, fb, ptr, arg_names, module, locals));
                        }

                        let mut sig = module.make_signature();
                        sig.params.push(AbiParam::new(types::I64));
                        for _ in 0..args.len() {
                            sig.params.push(AbiParam::new(types::F64));
                        }
                        sig.returns.push(AbiParam::new(types::F64));
                        let func_id = module
                            .declare_function(helper_name, Linkage::Import, &sig)
                            .expect("failed to declare named jit invoke helper");
                        let local = module.declare_func_in_func(func_id, &mut fb.func);
                        let call = fb.ins().call(local, &arg_vals);
                        return fb.inst_results(call)[0];
                    }
                }
            }

            if (symbol == "break_if" || symbol == "loop_break_if"
                || symbol == "break_when" || symbol == "loop_break_when"
                || symbol == "break_unless" || symbol == "loop_break_unless"
                || symbol == "continue_if" || symbol == "loop_continue_if"
                || symbol == "continue_when" || symbol == "loop_continue_when"
                || symbol == "continue_unless" || symbol == "loop_continue_unless")
                && args.len() == 2
            {
                let cond_val = gen_expr(&args[0], fb, ptr, arg_names, module, locals);
                let value_val = gen_expr(&args[1], fb, ptr, arg_names, module, locals);
                let zero = fb.ins().f64const(0.0);
                let cond_true = if symbol == "break_unless" || symbol == "loop_break_unless"
                    || symbol == "continue_unless" || symbol == "loop_continue_unless"
                {
                    fb.ins().fcmp(FloatCC::Equal, cond_val, zero)
                } else {
                    fb.ins().fcmp(FloatCC::NotEqual, cond_val, zero)
                };
                let sentinel = if symbol == "break_if" || symbol == "loop_break_if"
                    || symbol == "break_when" || symbol == "loop_break_when"
                    || symbol == "break_unless" || symbol == "loop_break_unless"
                {
                    fb.ins().f64const(f64::from_bits(BREAK_SENTINEL_BITS))
                } else {
                    fb.ins().f64const(f64::from_bits(CONTINUE_SENTINEL_BITS))
                };
                return fb.ins().select(cond_true, sentinel, value_val);
            }

            let mut arg_vals = Vec::with_capacity(args.len());
            for a in args {
                arg_vals.push(gen_expr(a, fb, ptr, arg_names, module, locals));
            }
            let mut symbol = symbol;
            if symbol == "abs" {
                symbol = "fabs".to_string();
            }
            let mut sig = module.make_signature();
            for _ in 0..arg_vals.len() {
                sig.params.push(AbiParam::new(types::F64));
            }
            sig.returns.push(AbiParam::new(types::F64));
            let func_id = module
                .declare_function(&symbol, Linkage::Import, &sig)
                .expect("failed to declare external function");
            let local = module.declare_func_in_func(func_id, &mut fb.func);
            let call = fb.ins().call(local, &arg_vals);
            fb.inst_results(call)[0]
        }
        Expr::Ternary(cond, then_expr, else_expr) => {
            let cond_val = gen_expr(cond, fb, ptr, arg_names, module, locals);
            let zero = fb.ins().f64const(0.0);
            let cond_bool = fb.ins().fcmp(FloatCC::NotEqual, cond_val, zero);
            let then_val = gen_expr(then_expr, fb, ptr, arg_names, module, locals);
            let else_val = gen_expr(else_expr, fb, ptr, arg_names, module, locals);
            fb.ins().select(cond_bool, then_val, else_val)
        }
        Expr::SumOver { .. } => {
            panic!("SumOver should have been transformed before codegen");
        }
        Expr::AnyOver { .. } => {
            panic!("AnyOver should have been transformed before codegen");
        }
        Expr::AllOver { .. } => {
            panic!("AllOver should have been transformed before codegen");
        }
        Expr::AnyFor {
            iter_var,
            start,
            end,
            step,
            body,
            pred,
        } => {
            let start_val = gen_expr(start, fb, ptr, arg_names, module, locals);
            let end_val = gen_expr(end, fb, ptr, arg_names, module, locals);
            let step_val = if let Some(st) = step {
                gen_expr(st, fb, ptr, arg_names, module, locals)
            } else {
                fb.ins().f64const(1.0)
            };
            let zero = fb.ins().f64const(0.0);
            let one = fb.ins().f64const(1.0);

            let loop_block = fb.create_block();
            let body_block = fb.create_block();
            let short_true_block = fb.create_block();
            let continue_block = fb.create_block();
            let exit_block = fb.create_block();
            fb.append_block_param(loop_block, types::F64); // i
            fb.append_block_param(loop_block, types::F64); // acc (0/1)
            fb.append_block_param(continue_block, types::F64); // next acc (0/1)
            fb.append_block_param(exit_block, types::F64); // result
            fb.ins().jump(loop_block, &[start_val, zero]);

            fb.switch_to_block(loop_block);
            let i_val = fb.block_params(loop_block)[0];
            let acc_val = fb.block_params(loop_block)[1];
            let step_pos = fb.ins().fcmp(FloatCC::GreaterThan, step_val, zero);
            let step_neg = fb.ins().fcmp(FloatCC::LessThan, step_val, zero);
            let cond_lt = fb.ins().fcmp(FloatCC::LessThan, i_val, end_val);
            let cond_gt = fb.ins().fcmp(FloatCC::GreaterThan, i_val, end_val);
            let run_pos = fb.ins().band(step_pos, cond_lt);
            let run_neg = fb.ins().band(step_neg, cond_gt);
            let cond = fb.ins().bor(run_pos, run_neg);
            fb.ins().brnz(cond, body_block, &[]);
            fb.ins().jump(exit_block, &[acc_val]);

            fb.switch_to_block(body_block);
            let mut body_locals = locals.clone();
            body_locals.insert(iter_var.clone(), i_val);
            let ctrl = detect_loop_control(body);
            let body_expr = match ctrl {
                LoopControl::Break { value, .. } | LoopControl::Continue { value, .. } => value,
                LoopControl::None => body,
            };
            let break_true = if let LoopControl::Break { cond, .. } = ctrl {
                let break_cond_val = gen_expr(cond, fb, ptr, arg_names, module, &body_locals);
                fb.ins().fcmp(FloatCC::NotEqual, break_cond_val, zero)
            } else {
                fb.ins().fcmp(FloatCC::Equal, zero, one)
            };
            let break_true = if let LoopControl::Break { invert_cond, .. } = ctrl {
                if invert_cond {
                    fb.ins().bnot(break_true)
                } else {
                    break_true
                }
            } else {
                break_true
            };
            let continue_true = if let LoopControl::Continue { cond, .. } = ctrl {
                let cont_cond_val = gen_expr(cond, fb, ptr, arg_names, module, &body_locals);
                fb.ins().fcmp(FloatCC::NotEqual, cont_cond_val, zero)
            } else {
                fb.ins().fcmp(FloatCC::Equal, zero, one)
            };
            let continue_true = if let LoopControl::Continue { invert_cond, .. } = ctrl {
                if invert_cond {
                    fb.ins().bnot(continue_true)
                } else {
                    continue_true
                }
            } else {
                continue_true
            };

            let body_val = gen_expr(body_expr, fb, ptr, arg_names, module, &body_locals);
            let body_true = fb.ins().fcmp(FloatCC::NotEqual, body_val, zero);
            let effective_true = if let Some(pred_expr) = pred {
                let pred_val = gen_expr(pred_expr, fb, ptr, arg_names, module, &body_locals);
                let pred_true = fb.ins().fcmp(FloatCC::NotEqual, pred_val, zero);
                fb.ins().band(pred_true, body_true)
            } else {
                body_true
            };
            let false_mask = fb.ins().fcmp(FloatCC::Equal, zero, one);
            let effective_true = fb.ins().select(continue_true, false_mask, effective_true);
            let acc_true = fb.ins().fcmp(FloatCC::NotEqual, acc_val, zero);
            let next_true = fb.ins().bor(acc_true, effective_true);
            let next_acc_base = fb.ins().select(next_true, one, zero);
            let next_acc = fb.ins().select(break_true, acc_val, next_acc_base);
            fb.ins().brnz(next_true, short_true_block, &[]);
            fb.ins().jump(continue_block, &[next_acc]);

            fb.switch_to_block(continue_block);
            let next_acc = fb.block_params(continue_block)[0];
            let next_i_base = fb.ins().fadd(i_val, step_val);
            let next_i = fb.ins().select(break_true, end_val, next_i_base);
            fb.ins().jump(loop_block, &[next_i, next_acc]);

            fb.switch_to_block(short_true_block);
            fb.ins().jump(exit_block, &[one]);

            fb.seal_block(body_block);
            fb.seal_block(short_true_block);
            fb.seal_block(continue_block);
            fb.seal_block(loop_block);
            fb.switch_to_block(exit_block);
            fb.seal_block(exit_block);
            fb.block_params(exit_block)[0]
        }
        Expr::AllFor {
            iter_var,
            start,
            end,
            step,
            body,
            pred,
        } => {
            let start_val = gen_expr(start, fb, ptr, arg_names, module, locals);
            let end_val = gen_expr(end, fb, ptr, arg_names, module, locals);
            let step_val = if let Some(st) = step {
                gen_expr(st, fb, ptr, arg_names, module, locals)
            } else {
                fb.ins().f64const(1.0)
            };
            let zero = fb.ins().f64const(0.0);
            let one = fb.ins().f64const(1.0);
            let true_mask = fb.ins().fcmp(FloatCC::Equal, zero, zero);

            let loop_block = fb.create_block();
            let body_block = fb.create_block();
            let short_false_block = fb.create_block();
            let continue_block = fb.create_block();
            let exit_block = fb.create_block();
            fb.append_block_param(loop_block, types::F64); // i
            fb.append_block_param(loop_block, types::F64); // acc (0/1)
            fb.append_block_param(continue_block, types::F64); // next acc (0/1)
            fb.append_block_param(exit_block, types::F64); // result
            fb.ins().jump(loop_block, &[start_val, one]);

            fb.switch_to_block(loop_block);
            let i_val = fb.block_params(loop_block)[0];
            let acc_val = fb.block_params(loop_block)[1];
            let step_pos = fb.ins().fcmp(FloatCC::GreaterThan, step_val, zero);
            let step_neg = fb.ins().fcmp(FloatCC::LessThan, step_val, zero);
            let cond_lt = fb.ins().fcmp(FloatCC::LessThan, i_val, end_val);
            let cond_gt = fb.ins().fcmp(FloatCC::GreaterThan, i_val, end_val);
            let run_pos = fb.ins().band(step_pos, cond_lt);
            let run_neg = fb.ins().band(step_neg, cond_gt);
            let cond = fb.ins().bor(run_pos, run_neg);
            fb.ins().brnz(cond, body_block, &[]);
            fb.ins().jump(exit_block, &[acc_val]);

            fb.switch_to_block(body_block);
            let mut body_locals = locals.clone();
            body_locals.insert(iter_var.clone(), i_val);
            let ctrl = detect_loop_control(body);
            let body_expr = match ctrl {
                LoopControl::Break { value, .. } | LoopControl::Continue { value, .. } => value,
                LoopControl::None => body,
            };
            let break_true = if let LoopControl::Break { cond, .. } = ctrl {
                let break_cond_val = gen_expr(cond, fb, ptr, arg_names, module, &body_locals);
                fb.ins().fcmp(FloatCC::NotEqual, break_cond_val, zero)
            } else {
                fb.ins().fcmp(FloatCC::Equal, zero, one)
            };
            let break_true = if let LoopControl::Break { invert_cond, .. } = ctrl {
                if invert_cond {
                    fb.ins().bnot(break_true)
                } else {
                    break_true
                }
            } else {
                break_true
            };
            let continue_true = if let LoopControl::Continue { cond, .. } = ctrl {
                let cont_cond_val = gen_expr(cond, fb, ptr, arg_names, module, &body_locals);
                fb.ins().fcmp(FloatCC::NotEqual, cont_cond_val, zero)
            } else {
                fb.ins().fcmp(FloatCC::Equal, zero, one)
            };
            let continue_true = if let LoopControl::Continue { invert_cond, .. } = ctrl {
                if invert_cond {
                    fb.ins().bnot(continue_true)
                } else {
                    continue_true
                }
            } else {
                continue_true
            };

            let body_val = gen_expr(body_expr, fb, ptr, arg_names, module, &body_locals);
            let body_true = fb.ins().fcmp(FloatCC::NotEqual, body_val, zero);
            let effective_true = if let Some(pred_expr) = pred {
                let pred_val = gen_expr(pred_expr, fb, ptr, arg_names, module, &body_locals);
                let pred_true = fb.ins().fcmp(FloatCC::NotEqual, pred_val, zero);
                fb.ins().select(pred_true, body_true, true_mask)
            } else {
                body_true
            };
            let effective_true = fb.ins().select(continue_true, true_mask, effective_true);
            let acc_true = fb.ins().fcmp(FloatCC::NotEqual, acc_val, zero);
            let next_true = fb.ins().band(acc_true, effective_true);
            let next_acc_base = fb.ins().select(next_true, one, zero);
            let next_acc = fb.ins().select(break_true, acc_val, next_acc_base);
            fb.ins().brz(next_true, short_false_block, &[]);
            fb.ins().jump(continue_block, &[next_acc]);

            fb.switch_to_block(continue_block);
            let next_acc = fb.block_params(continue_block)[0];
            let next_i_base = fb.ins().fadd(i_val, step_val);
            let next_i = fb.ins().select(break_true, end_val, next_i_base);
            fb.ins().jump(loop_block, &[next_i, next_acc]);

            fb.switch_to_block(short_false_block);
            fb.ins().jump(exit_block, &[zero]);

            fb.seal_block(body_block);
            fb.seal_block(short_false_block);
            fb.seal_block(continue_block);
            fb.seal_block(loop_block);
            fb.switch_to_block(exit_block);
            fb.seal_block(exit_block);
            fb.block_params(exit_block)[0]
        }
        Expr::SumFor {
            iter_var,
            start,
            end,
            step,
            body,
            pred,
        } => {
            let start_val = gen_expr(start, fb, ptr, arg_names, module, locals);
            let end_val = gen_expr(end, fb, ptr, arg_names, module, locals);
            let step_val = if let Some(st) = step {
                gen_expr(st, fb, ptr, arg_names, module, locals)
            } else {
                fb.ins().f64const(1.0)
            };
            let zero_acc = fb.ins().f64const(0.0);
            let loop_block = fb.create_block();
            let body_block = fb.create_block();
            let exit_block = fb.create_block();
            fb.append_block_param(loop_block, types::F64); // i
            fb.append_block_param(loop_block, types::F64); // acc
            fb.append_block_param(exit_block, types::F64); // result
            fb.ins().jump(loop_block, &[start_val, zero_acc]);
            fb.switch_to_block(loop_block);
            let i_val = fb.block_params(loop_block)[0];
            let acc_val = fb.block_params(loop_block)[1];
            // runtime-aware step direction:
            // step > 0 => i < end
            // step < 0 => i > end
            // step == 0 => do not enter loop
            let zero = fb.ins().f64const(0.0);
            let step_pos = fb.ins().fcmp(FloatCC::GreaterThan, step_val, zero);
            let step_neg = fb.ins().fcmp(FloatCC::LessThan, step_val, zero);
            let cond_lt = fb.ins().fcmp(FloatCC::LessThan, i_val, end_val);
            let cond_gt = fb.ins().fcmp(FloatCC::GreaterThan, i_val, end_val);
            let run_pos = fb.ins().band(step_pos, cond_lt);
            let run_neg = fb.ins().band(step_neg, cond_gt);
            let cond = fb.ins().bor(run_pos, run_neg);
            fb.ins().brnz(cond, body_block, &[]);
            fb.ins().jump(exit_block, &[acc_val]);
            fb.switch_to_block(body_block);
            let mut body_locals = locals.clone();
            body_locals.insert(iter_var.clone(), i_val);
            let ctrl = detect_loop_control(body);
            let body_expr = match ctrl {
                LoopControl::Break { value, .. } | LoopControl::Continue { value, .. } => value,
                LoopControl::None => body,
            };
            let break_true = if let LoopControl::Break { cond, .. } = ctrl {
                let break_cond_val = gen_expr(cond, fb, ptr, arg_names, module, &body_locals);
                fb.ins().fcmp(FloatCC::NotEqual, break_cond_val, zero)
            } else {
                let one_const = fb.ins().f64const(1.0);
                fb.ins().fcmp(FloatCC::Equal, zero, one_const)
            };
            let break_true = if let LoopControl::Break { invert_cond, .. } = ctrl {
                if invert_cond {
                    fb.ins().bnot(break_true)
                } else {
                    break_true
                }
            } else {
                break_true
            };
            let continue_true = if let LoopControl::Continue { cond, .. } = ctrl {
                let cont_cond_val = gen_expr(cond, fb, ptr, arg_names, module, &body_locals);
                fb.ins().fcmp(FloatCC::NotEqual, cont_cond_val, zero)
            } else {
                let one_const = fb.ins().f64const(1.0);
                fb.ins().fcmp(FloatCC::Equal, zero, one_const)
            };
            let continue_true = if let LoopControl::Continue { invert_cond, .. } = ctrl {
                if invert_cond {
                    fb.ins().bnot(continue_true)
                } else {
                    continue_true
                }
            } else {
                continue_true
            };

            let body_val = gen_expr(body_expr, fb, ptr, arg_names, module, &body_locals);
            let body_val = if let Some(pred_expr) = pred {
                let cond_val = gen_expr(pred_expr, fb, ptr, arg_names, module, &body_locals);
                let zero = fb.ins().f64const(0.0);
                let mask = fb.ins().fcmp(FloatCC::NotEqual, cond_val, zero);
                fb.ins().select(mask, body_val, zero)
            } else {
                body_val
            };
            let body_val = fb.ins().select(continue_true, zero, body_val);
            let next_acc_base = fb.ins().fadd(acc_val, body_val);
            let next_acc = fb.ins().select(break_true, acc_val, next_acc_base);
            let next_i_base = fb.ins().fadd(i_val, step_val);
            let next_i = fb.ins().select(break_true, end_val, next_i_base);
            fb.ins().jump(loop_block, &[next_i, next_acc]);
            fb.seal_block(body_block);
            fb.seal_block(loop_block);
            fb.switch_to_block(exit_block);
            fb.seal_block(exit_block);
            fb.block_params(exit_block)[0]
        }
    }
}

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
    const PI: f64 = std::f64::consts::PI;
    const TAU: f64 = std::f64::consts::TAU;
    const HALF_PI: f64 = std::f64::consts::FRAC_PI_2;

    let mut y = x % TAU;
    if y > PI {
        y -= TAU;
    } else if y < -PI {
        y += TAU;
    }

    let mut sign = 1.0;
    if y > HALF_PI {
        y = PI - y;
    } else if y < -HALF_PI {
        y = -PI - y;
        sign = -1.0;
    }

    let z = y * y;
    let p = y * (1.0
        + z * (-1.0 / 6.0
        + z * (1.0 / 120.0
        + z * (-1.0 / 5040.0
        + z * (1.0 / 362880.0)))));
    sign * p
}

#[inline(always)]
fn fast_cos_approx(x: f64) -> f64 {
    fast_sin_approx(x + std::f64::consts::FRAC_PI_2)
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
    let mode = simd_math_mode_from_env();

    let kernel = entry.lowered_kernel?;

    let eval_unary = |op: LoweredUnaryKernel, view: &BufferView, idx: usize| {
        let x = unsafe { read_buffer_f64(view, idx) };
        lowered_unary_eval(op, x, mode)
    };
    let eval_binary = |op: LoweredBinaryKernel, lhs_view: &BufferView, rhs_view: &BufferView, idx: usize| {
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
                            for lane in 0..lanes {
                                lane_acc[lane] += eval_unary(op, input_view, i + lane);
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
                                lane_any[lane] |= eval_binary(op, lhs_view, rhs_view, i + lane) != 0.0;
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
                                lane_all[lane] &= eval_binary(op, lhs_view, rhs_view, i + lane) != 0.0;
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
        }
        return None;
    }

    let mut results = Vec::with_capacity(len);

    match kernel {
        LoweredKernel::Unary { op, input } => {
            let input_view = views.get(input)?;
            let mut i = 0;
            while i + unroll <= len {
                for lane in 0..unroll {
                    let idx = i + lane;
                    results.push(eval_unary(op, input_view, idx));
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
            let array_obj = array_mod.getattr("array").unwrap().call1(("d", py_bytes)).unwrap();
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
pub fn execute_jit_func(py: pyo3::Python, entry: &JitEntry, args: &pyo3::types::PyTuple) -> pyo3::PyResult<pyo3::PyObject> {
    // Safety check: if func_ptr is zero (or invalid), cannot execute JIT.
    // This can happen if quantum speculation is enabled but variants haven't been compiled yet.
    if entry.func_ptr == 0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "JIT function pointer is invalid or not yet compiled"
        ));
    }

    let arg_count = args.len();
    let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry.func_ptr) };
    let loop_unroll = simd_unroll_factor();

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
                if entry.arg_count <= MAX_FAST_ARGS {
                    let mut stack_args: [f64; MAX_FAST_ARGS] = [0.0; MAX_FAST_ARGS];
                    for i in 0..entry.arg_count {
                        let item = unsafe { pyo3::ffi::PyTuple_GET_ITEM(args.as_ptr(), i as isize) };
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
                        let item = unsafe { pyo3::ffi::PyTuple_GET_ITEM(args.as_ptr(), i as isize) };
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
            JitExecProfile::PackedBuffer { arg_count: expected, elem } => {
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
            JitExecProfile::VectorizedBuffers { arg_count: expected, elem_types } => {
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
                        if let Some(lowered) = try_execute_lowered_vector_kernel(entry, &views, len, loop_unroll) {
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
                            let item = unsafe { pyo3::ffi::PyTuple_GET_ITEM(args.as_ptr(), i as isize) };
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
                if let Some(lowered) = try_execute_lowered_vector_kernel(entry, &views, len, loop_unroll) {
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
            let is_text_like = item
                .is_instance_of::<pyo3::types::PyString>()
                || item.is_instance_of::<pyo3::types::PyBytes>()
                || item.is_instance_of::<pyo3::types::PyByteArray>();

            if !is_text_like {
                if let Ok(len) = item.len() {
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
        assert!(sin_err < 1e-3, "sin approximation error too high: {}", sin_err);
        assert!(cos_err < 1e-3, "cos approximation error too high: {}", cos_err);
    }
}
