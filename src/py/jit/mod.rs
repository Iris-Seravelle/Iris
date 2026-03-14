// src/py/jit/mod.rs
//! Python JIT/offload support for the Iris runtime.
//!
//! This module provides the low-level bindings that power the `@iris.offload`
//! decorator in Python.  It also exposes the public API surface used by the
//! various JIT submodules (parser, codegen, heuristics) which live in their
//! own files for clarity.

#![allow(non_local_definitions)]

use std::sync::Arc;
use std::sync::atomic::{AtomicI8, Ordering};
use std::sync::{OnceLock, RwLock};
use std::collections::HashMap;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use std::{any::Any, panic::{catch_unwind, AssertUnwindSafe}};

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
#[cfg(feature = "pyo3")]
use pyo3::types::{PyDict, PyTuple};

use pyo3::AsPyPointer;

pub(crate) mod parser;
pub(crate) mod codegen;
pub(crate) mod heuristics;

// re-export helpers for convenience within this module
use crate::py::jit::codegen::{
    compile_jit_quantum,
    compile_jit_with_return_type,
    execute_registered_jit,
    lookup_jit,
    quantum_has_seed_hint,
    quantum_profile_snapshot,
    register_jit,
    register_quantum_jit,
    register_named_jit,
    seed_quantum_profile as seed_quantum_profile_state,
    JitReturnType,
    QuantumProfileSeed,
};

static JIT_LOG_OVERRIDE: AtomicI8 = AtomicI8::new(-1); // -1 env, 0 off, 1 on
static JIT_LOG_ENV_VAR: OnceLock<RwLock<String>> = OnceLock::new();
static JIT_LOG_HOOK: OnceLock<std::sync::Mutex<Option<Box<dyn Fn(String) + Send + Sync>>>> =
    OnceLock::new();

static JIT_QUANTUM_OVERRIDE: AtomicI8 = AtomicI8::new(-1); // -1 env, 0 off, 1 on
static JIT_QUANTUM_ENV_VAR: OnceLock<RwLock<String>> = OnceLock::new();
static JIT_QUANTUM_LOG_ENV_VAR: OnceLock<RwLock<String>> = OnceLock::new();
static JIT_QUANTUM_SPECULATION_ENV_VAR: OnceLock<RwLock<String>> = OnceLock::new();
static JIT_QUANTUM_COMPILE_BUDGET_ENV_VAR: OnceLock<RwLock<String>> = OnceLock::new();
static JIT_QUANTUM_COMPILE_WINDOW_ENV_VAR: OnceLock<RwLock<String>> = OnceLock::new();
static JIT_QUANTUM_COOLDOWN_BASE_ENV_VAR: OnceLock<RwLock<String>> = OnceLock::new();
static JIT_QUANTUM_COOLDOWN_MAX_ENV_VAR: OnceLock<RwLock<String>> = OnceLock::new();
static JIT_QUANTUM_STABILITY_MIN_SCORE_ENV_VAR: OnceLock<RwLock<String>> = OnceLock::new();
static JIT_QUANTUM_STABILITY_MIN_RUNS_ENV_VAR: OnceLock<RwLock<String>> = OnceLock::new();
static JIT_QUANTUM_VARIANT_FAILURE_LIMIT_ENV_VAR: OnceLock<RwLock<String>> = OnceLock::new();
static JIT_QUANTUM_VARIANT_PROMOTION_MIN_RUNS_ENV_VAR: OnceLock<RwLock<String>> = OnceLock::new();

#[derive(Default)]
struct QuantumCompileBudgetState {
    window_start_ns: u64,
    consumed_ns: u64,
}

#[derive(Default, Clone, Copy)]
struct QuantumCooldownState {
    failures: u32,
    cooldown_until_ns: u64,
}

static QUANTUM_COMPILE_BUDGET_STATE: OnceLock<std::sync::Mutex<QuantumCompileBudgetState>> =
    OnceLock::new();
static QUANTUM_COOLDOWN_STATE: OnceLock<std::sync::Mutex<HashMap<usize, QuantumCooldownState>>> =
    OnceLock::new();

fn jit_log_env_var() -> &'static RwLock<String> {
    JIT_LOG_ENV_VAR.get_or_init(|| RwLock::new("IRIS_JIT_LOG".to_string()))
}

fn jit_quantum_env_var() -> &'static RwLock<String> {
    JIT_QUANTUM_ENV_VAR.get_or_init(|| RwLock::new("IRIS_JIT_QUANTUM".to_string()))
}

fn jit_quantum_log_env_var() -> &'static RwLock<String> {
    JIT_QUANTUM_LOG_ENV_VAR.get_or_init(|| RwLock::new("IRIS_JIT_QUANTUM_LOG_NS".to_string()))
}

fn jit_quantum_speculation_env_var() -> &'static RwLock<String> {
    JIT_QUANTUM_SPECULATION_ENV_VAR
        .get_or_init(|| RwLock::new("IRIS_JIT_QUANTUM_SPECULATION_NS".to_string()))
}

fn jit_quantum_compile_budget_env_var() -> &'static RwLock<String> {
    JIT_QUANTUM_COMPILE_BUDGET_ENV_VAR
        .get_or_init(|| RwLock::new("IRIS_JIT_QUANTUM_COMPILE_BUDGET_NS".to_string()))
}

fn jit_quantum_compile_window_env_var() -> &'static RwLock<String> {
    JIT_QUANTUM_COMPILE_WINDOW_ENV_VAR
        .get_or_init(|| RwLock::new("IRIS_JIT_QUANTUM_COMPILE_WINDOW_NS".to_string()))
}

fn jit_quantum_cooldown_base_env_var() -> &'static RwLock<String> {
    JIT_QUANTUM_COOLDOWN_BASE_ENV_VAR
        .get_or_init(|| RwLock::new("IRIS_JIT_QUANTUM_COOLDOWN_BASE_NS".to_string()))
}

fn jit_quantum_cooldown_max_env_var() -> &'static RwLock<String> {
    JIT_QUANTUM_COOLDOWN_MAX_ENV_VAR
        .get_or_init(|| RwLock::new("IRIS_JIT_QUANTUM_COOLDOWN_MAX_NS".to_string()))
}

fn jit_quantum_stability_min_score_env_var() -> &'static RwLock<String> {
    JIT_QUANTUM_STABILITY_MIN_SCORE_ENV_VAR
        .get_or_init(|| RwLock::new("IRIS_JIT_QUANTUM_STABILITY_MIN_SCORE".to_string()))
}

fn jit_quantum_stability_min_runs_env_var() -> &'static RwLock<String> {
    JIT_QUANTUM_STABILITY_MIN_RUNS_ENV_VAR
        .get_or_init(|| RwLock::new("IRIS_JIT_QUANTUM_STABILITY_MIN_RUNS".to_string()))
}

fn jit_quantum_variant_failure_limit_env_var() -> &'static RwLock<String> {
    JIT_QUANTUM_VARIANT_FAILURE_LIMIT_ENV_VAR
        .get_or_init(|| RwLock::new("IRIS_JIT_QUANTUM_VARIANT_FAILURE_LIMIT".to_string()))
}

fn jit_quantum_variant_promotion_min_runs_env_var() -> &'static RwLock<String> {
    JIT_QUANTUM_VARIANT_PROMOTION_MIN_RUNS_ENV_VAR
        .get_or_init(|| RwLock::new("IRIS_JIT_QUANTUM_VARIANT_PROMOTION_MIN_RUNS".to_string()))
}

fn now_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

fn parse_bool_env(v: &str) -> bool {
    matches!(
        v.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on" | "debug"
    )
}

fn parse_return_type(rt: Option<&str>) -> JitReturnType {
    match rt {
        Some("int") => JitReturnType::Int,
        Some("bool") => JitReturnType::Bool,
        _ => JitReturnType::Float,
    }
}

pub(crate) fn jit_logging_enabled() -> bool {
    match JIT_LOG_OVERRIDE.load(Ordering::Relaxed) {
        0 => false,
        1 => true,
        _ => {
            let env_name = jit_log_env_var().read().unwrap().clone();
            std::env::var(env_name)
                .ok()
                .map(|v| parse_bool_env(&v))
                .unwrap_or(false)
        }
    }
}

#[cfg(test)]
pub(crate) fn jit_log_hook<F>(hook: F)
where
    F: Fn(String) + Send + Sync + 'static,
{
    let lock = JIT_LOG_HOOK.get_or_init(|| std::sync::Mutex::new(None));
    let mut guard = lock.lock().unwrap();
    *guard = Some(Box::new(hook));
}

#[cfg(test)]
pub(crate) fn jit_log_clear_hook() {
    if let Some(lock) = JIT_LOG_HOOK.get() {
        let mut guard = lock.lock().unwrap();
        *guard = None;
    }
}

pub(crate) fn jit_log<F>(msg: F)
where
    F: FnOnce() -> String,
{
    if !jit_logging_enabled() {
        return;
    }

    if let Some(lock) = JIT_LOG_HOOK.get() {
        let guard = lock.lock().unwrap();
        if let Some(ref hook) = *guard {
            hook(msg());
            return;
        }
    }

    eprintln!("{}", msg());
}

pub(crate) fn quantum_log_threshold_ns() -> u64 {
    const DEFAULT: u64 = 1_000_000; // 1ms
    let env_name = jit_quantum_log_env_var().read().unwrap().clone();
    std::env::var(env_name)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(DEFAULT)
}

pub(crate) fn quantum_speculation_threshold_ns() -> u64 {
    const DEFAULT: u64 = 1_000_000; // 1ms
    let env_name = jit_quantum_speculation_env_var().read().unwrap().clone();
    std::env::var(env_name)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(DEFAULT)
}

pub(crate) fn quantum_compile_budget_ns() -> u64 {
    const DEFAULT: u64 = 50_000_000; // 50ms compile budget per window
    let env_name = jit_quantum_compile_budget_env_var().read().unwrap().clone();
    std::env::var(env_name)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(DEFAULT)
}

pub(crate) fn quantum_compile_window_ns() -> u64 {
    const DEFAULT: u64 = 1_000_000_000; // 1s
    let env_name = jit_quantum_compile_window_env_var().read().unwrap().clone();
    std::env::var(env_name)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(DEFAULT)
}

pub(crate) fn quantum_cooldown_base_ns() -> u64 {
    const DEFAULT: u64 = 5_000_000; // 5ms
    let env_name = jit_quantum_cooldown_base_env_var().read().unwrap().clone();
    std::env::var(env_name)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(DEFAULT)
}

pub(crate) fn quantum_cooldown_max_ns() -> u64 {
    const DEFAULT: u64 = 1_000_000_000; // 1s
    let env_name = jit_quantum_cooldown_max_env_var().read().unwrap().clone();
    std::env::var(env_name)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(DEFAULT)
}

pub(crate) fn quantum_stability_min_score() -> f64 {
    const DEFAULT: f64 = 0.35;
    let env_name = jit_quantum_stability_min_score_env_var().read().unwrap().clone();
    std::env::var(env_name)
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .map(|v| v.clamp(0.0, 1.0))
        .unwrap_or(DEFAULT)
}

pub(crate) fn quantum_stability_min_runs() -> u64 {
    const DEFAULT: u64 = 8;
    let env_name = jit_quantum_stability_min_runs_env_var().read().unwrap().clone();
    std::env::var(env_name)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(DEFAULT)
}

pub(crate) fn quantum_variant_failure_limit() -> u64 {
    const DEFAULT: u64 = 8;
    let env_name = jit_quantum_variant_failure_limit_env_var().read().unwrap().clone();
    std::env::var(env_name)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(DEFAULT)
}

pub(crate) fn quantum_variant_promotion_min_runs() -> u64 {
    const DEFAULT: u64 = 8;
    let env_name = jit_quantum_variant_promotion_min_runs_env_var().read().unwrap().clone();
    std::env::var(env_name)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(DEFAULT)
}

pub(crate) fn quantum_compile_may_run(func_key: usize, now_ns: u64) -> bool {
    let cooldown_map = QUANTUM_COOLDOWN_STATE.get_or_init(|| std::sync::Mutex::new(HashMap::new()));
    if let Some(state) = cooldown_map.lock().unwrap().get(&func_key).copied() {
        if now_ns < state.cooldown_until_ns {
            return false;
        }
    }

    let window_ns = quantum_compile_window_ns();
    let budget_ns = quantum_compile_budget_ns();
    let budget_state = QUANTUM_COMPILE_BUDGET_STATE
        .get_or_init(|| std::sync::Mutex::new(QuantumCompileBudgetState::default()));
    let mut budget = budget_state.lock().unwrap();
    if budget.window_start_ns == 0 || now_ns.saturating_sub(budget.window_start_ns) >= window_ns {
        budget.window_start_ns = now_ns;
        budget.consumed_ns = 0;
    }
    budget.consumed_ns < budget_ns
}

pub(crate) fn record_quantum_compile_attempt(func_key: usize, now_ns: u64, elapsed_ns: u64, success: bool) {
    let window_ns = quantum_compile_window_ns();
    let budget_state = QUANTUM_COMPILE_BUDGET_STATE
        .get_or_init(|| std::sync::Mutex::new(QuantumCompileBudgetState::default()));
    {
        let mut budget = budget_state.lock().unwrap();
        if budget.window_start_ns == 0 || now_ns.saturating_sub(budget.window_start_ns) >= window_ns {
            budget.window_start_ns = now_ns;
            budget.consumed_ns = 0;
        }
        budget.consumed_ns = budget.consumed_ns.saturating_add(elapsed_ns);
    }

    let cooldown_map = QUANTUM_COOLDOWN_STATE.get_or_init(|| std::sync::Mutex::new(HashMap::new()));
    let mut guard = cooldown_map.lock().unwrap();
    let state = guard.entry(func_key).or_default();
    if success {
        state.failures = 0;
        state.cooldown_until_ns = 0;
        return;
    }

    state.failures = state.failures.saturating_add(1);
    let base = quantum_cooldown_base_ns();
    let max = quantum_cooldown_max_ns();
    let shift = (state.failures.saturating_sub(1)).min(20);
    let mult = 1u64.checked_shl(shift).unwrap_or(u64::MAX);
    let cooldown = base.saturating_mul(mult).min(max);
    state.cooldown_until_ns = now_ns.saturating_add(cooldown);
}

#[cfg(test)]
pub(crate) fn reset_quantum_control_state() {
    if let Some(state) = QUANTUM_COMPILE_BUDGET_STATE.get() {
        let mut guard = state.lock().unwrap();
        guard.window_start_ns = 0;
        guard.consumed_ns = 0;
    }
    if let Some(state) = QUANTUM_COOLDOWN_STATE.get() {
        state.lock().unwrap().clear();
    }
}

fn panic_payload_to_string(payload: Box<dyn Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic payload".to_string()
    }
}

#[cfg(feature = "pyo3")]
fn execute_registered_jit_guarded(
    py: Python,
    func_key: usize,
    args: &PyTuple,
) -> Option<PyResult<PyObject>> {
    match catch_unwind(AssertUnwindSafe(|| execute_registered_jit(py, func_key, args))) {
        Ok(res) => res,
        Err(payload) => {
            let msg = panic_payload_to_string(payload);
            Some(Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "jit panic: {}",
                msg
            ))))
        }
    }
}

pub(crate) fn quantum_speculation_enabled() -> bool {
    match JIT_QUANTUM_OVERRIDE.load(Ordering::Relaxed) {
        0 => false,
        1 => true,
        _ => {
            let env_name = jit_quantum_env_var().read().unwrap().clone();
            std::env::var(env_name)
                .ok()
                .map(|v| parse_bool_env(&v))
                .unwrap_or(false)
        }
    }
}

// Offload actor pool ---------------------------------------------------------

/// A task describing a Python call to execute.
struct OffloadTask {
    func: Py<PyAny>,
    args: Py<PyTuple>,
    kwargs: Option<Py<PyDict>>,
    resp: std::sync::mpsc::Sender<Result<PyObject, PyErr>>,
}

struct OffloadPool {
    sender: crossbeam_channel::Sender<OffloadTask>,
}

impl OffloadPool {
    fn new(size: usize) -> Self {
        let (tx, rx) = crossbeam_channel::unbounded::<OffloadTask>();

        for _ in 0..size {
            let rx = rx.clone();
            std::thread::spawn(move || {
                loop {
                    match rx.recv() {
                        Ok(task) => {
                            if unsafe { pyo3::ffi::Py_IsInitialized() } == 0 {
                                break;
                            }
                            Python::with_gil(|py| {
                                let func = task.func.as_ref(py);
                                let args = task.args.as_ref(py);
                                let kwargs = task
                                    .kwargs
                                    .as_ref()
                                    .map(|k: &Py<PyDict>| k.as_ref(py));
                                let result = func.call(args, kwargs).map(|obj| obj.into_py(py));
                                let _ = task.resp.send(result);
                            });
                        }
                        Err(_) => break,
                    }
                }
            });
        }

        OffloadPool { sender: tx }
    }
}

// shared singleton
static OFFLOAD_POOL: once_cell::sync::OnceCell<Arc<OffloadPool>> =
    once_cell::sync::OnceCell::new();

fn get_offload_pool() -> Arc<OffloadPool> {
    OFFLOAD_POOL
        .get_or_init(|| Arc::new(OffloadPool::new(num_cpus::get())))
        .clone()
}

// Python bindings -----------------------------------------------------------

/// Initialize the Python submodule (called from `wrappers.populate_module`).
#[cfg(feature = "pyo3")]
pub(crate) fn init_py(m: &PyModule) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(register_offload, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(offload_call, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(call_jit, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(call_jit_step_loop_f64, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(configure_jit_logging, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(is_jit_logging_enabled, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(configure_quantum_speculation, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(is_quantum_speculation_enabled, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(configure_quantum_speculation_threshold, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(get_quantum_speculation_threshold, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(set_quantum_speculation_threshold, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(configure_quantum_log_threshold, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(get_quantum_log_threshold, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(set_quantum_log_threshold, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(configure_quantum_compile_budget, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(get_quantum_compile_budget, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(configure_quantum_cooldown, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(get_quantum_cooldown, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(get_quantum_profile, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(seed_quantum_profile, m)?)?;
    Ok(())
}

#[cfg(feature = "pyo3")]
#[pyfunction]
fn configure_jit_logging(enabled: Option<bool>, env_var: Option<String>) -> PyResult<bool> {
    if let Some(name) = env_var {
        *jit_log_env_var().write().unwrap() = name;
    }
    match enabled {
        Some(true) => JIT_LOG_OVERRIDE.store(1, Ordering::Relaxed),
        Some(false) => JIT_LOG_OVERRIDE.store(0, Ordering::Relaxed),
        None => JIT_LOG_OVERRIDE.store(-1, Ordering::Relaxed),
    }
    Ok(jit_logging_enabled())
}

#[cfg(feature = "pyo3")]
#[pyfunction]
fn is_jit_logging_enabled() -> PyResult<bool> {
    Ok(jit_logging_enabled())
}

#[cfg(feature = "pyo3")]
#[pyfunction]
fn configure_quantum_speculation(enabled: Option<bool>, env_var: Option<String>) -> PyResult<bool> {
    if let Some(name) = env_var {
        *jit_quantum_env_var().write().unwrap() = name;
    }
    match enabled {
        Some(true) => JIT_QUANTUM_OVERRIDE.store(1, Ordering::Relaxed),
        Some(false) => JIT_QUANTUM_OVERRIDE.store(0, Ordering::Relaxed),
        None => JIT_QUANTUM_OVERRIDE.store(-1, Ordering::Relaxed),
    }
    Ok(quantum_speculation_enabled())
}

#[cfg(feature = "pyo3")]
#[pyfunction]
fn is_quantum_speculation_enabled() -> PyResult<bool> {
    Ok(quantum_speculation_enabled())
}

#[cfg(feature = "pyo3")]
#[pyfunction]
fn configure_quantum_speculation_threshold(threshold_ns: Option<u64>, env_var: Option<String>) -> PyResult<u64> {
    if let Some(name) = env_var {
        *jit_quantum_speculation_env_var().write().unwrap() = name;
    }
    if let Some(ns) = threshold_ns {
        let env_name = jit_quantum_speculation_env_var().read().unwrap().clone();
        std::env::set_var(env_name, ns.to_string());
    }
    Ok(quantum_speculation_threshold_ns())
}

#[cfg(feature = "pyo3")]
#[pyfunction]
fn get_quantum_speculation_threshold() -> PyResult<u64> {
    Ok(quantum_speculation_threshold_ns())
}

#[cfg(feature = "pyo3")]
#[pyfunction]
fn set_quantum_speculation_threshold(threshold_ns: Option<u64>, env_var: Option<String>) -> PyResult<u64> {
    configure_quantum_speculation_threshold(threshold_ns, env_var)
}

#[cfg(feature = "pyo3")]
#[pyfunction]
fn configure_quantum_log_threshold(threshold_ns: Option<u64>, env_var: Option<String>) -> PyResult<u64> {
    if let Some(name) = env_var {
        *jit_quantum_log_env_var().write().unwrap() = name;
    }
    if let Some(ns) = threshold_ns {
        let env_name = jit_quantum_log_env_var().read().unwrap().clone();
        std::env::set_var(env_name, ns.to_string());
    }
    Ok(quantum_log_threshold_ns())
}

#[cfg(feature = "pyo3")]
#[pyfunction]
fn get_quantum_log_threshold() -> PyResult<u64> {
    Ok(quantum_log_threshold_ns())
}

#[cfg(feature = "pyo3")]
#[pyfunction]
fn set_quantum_log_threshold(threshold_ns: Option<u64>, env_var: Option<String>) -> PyResult<u64> {
    configure_quantum_log_threshold(threshold_ns, env_var)
}

#[cfg(feature = "pyo3")]
#[pyfunction]
fn configure_quantum_compile_budget(
    budget_ns: Option<u64>,
    window_ns: Option<u64>,
    budget_env_var: Option<String>,
    window_env_var: Option<String>,
) -> PyResult<(u64, u64)> {
    if let Some(name) = budget_env_var {
        *jit_quantum_compile_budget_env_var().write().unwrap() = name;
    }
    if let Some(name) = window_env_var {
        *jit_quantum_compile_window_env_var().write().unwrap() = name;
    }
    if let Some(ns) = budget_ns {
        let env_name = jit_quantum_compile_budget_env_var().read().unwrap().clone();
        std::env::set_var(env_name, ns.to_string());
    }
    if let Some(ns) = window_ns {
        let env_name = jit_quantum_compile_window_env_var().read().unwrap().clone();
        std::env::set_var(env_name, ns.to_string());
    }
    Ok((quantum_compile_budget_ns(), quantum_compile_window_ns()))
}

#[cfg(feature = "pyo3")]
#[pyfunction]
fn get_quantum_compile_budget() -> PyResult<(u64, u64)> {
    Ok((quantum_compile_budget_ns(), quantum_compile_window_ns()))
}

#[cfg(feature = "pyo3")]
#[pyfunction]
fn configure_quantum_cooldown(
    base_ns: Option<u64>,
    max_ns: Option<u64>,
    base_env_var: Option<String>,
    max_env_var: Option<String>,
) -> PyResult<(u64, u64)> {
    if let Some(name) = base_env_var {
        *jit_quantum_cooldown_base_env_var().write().unwrap() = name;
    }
    if let Some(name) = max_env_var {
        *jit_quantum_cooldown_max_env_var().write().unwrap() = name;
    }
    if let Some(ns) = base_ns {
        let env_name = jit_quantum_cooldown_base_env_var().read().unwrap().clone();
        std::env::set_var(env_name, ns.to_string());
    }
    if let Some(ns) = max_ns {
        let env_name = jit_quantum_cooldown_max_env_var().read().unwrap().clone();
        std::env::set_var(env_name, ns.to_string());
    }
    Ok((quantum_cooldown_base_ns(), quantum_cooldown_max_ns()))
}

#[cfg(feature = "pyo3")]
#[pyfunction]
fn get_quantum_cooldown() -> PyResult<(u64, u64)> {
    Ok((quantum_cooldown_base_ns(), quantum_cooldown_max_ns()))
}

#[cfg(feature = "pyo3")]
#[pyfunction]
fn get_quantum_profile(func: PyObject) -> PyResult<Vec<(usize, f64, u64, u64)>> {
    let key = func.as_ptr() as usize;
    let points = quantum_profile_snapshot(key).unwrap_or_default();
    Ok(points
        .into_iter()
        .map(|point| (point.index, point.ewma_ns, point.runs, point.failures))
        .collect())
}

#[cfg(feature = "pyo3")]
#[pyfunction]
fn seed_quantum_profile(func: PyObject, rows: Vec<(usize, f64, u64, u64)>) -> PyResult<bool> {
    let key = func.as_ptr() as usize;
    let seeds = rows
        .into_iter()
        .map(|(index, ewma_ns, runs, failures)| QuantumProfileSeed {
            index,
            ewma_ns,
            runs,
            failures,
        })
        .collect::<Vec<_>>();
    Ok(seed_quantum_profile_state(key, &seeds))
}

/// Register a Python function for offloading.
#[cfg(feature = "pyo3")]
#[pyfunction]
fn register_offload(
    func: PyObject,
    strategy: Option<String>,
    return_type: Option<String>,
    source_expr: Option<String>,
    arg_names: Option<Vec<String>>,
) -> PyResult<PyObject> {
    if let Some(ref s) = strategy {
        if s == "actor" {
            let _ = get_offload_pool();
        } else if s == "jit" {
            if let (Some(expr), Some(args)) = (source_expr.clone(), arg_names.clone()) {
                let key = func.as_ptr() as usize;
                let func_name = Python::with_gil(|py| {
                    func.as_ref(py)
                        .getattr("__name__")
                        .ok()
                        .and_then(|n| n.extract::<String>().ok())
                });
                let return_type = parse_return_type(return_type.as_deref());
                let warm_seed_hint = quantum_has_seed_hint(key);

                let compile_single = || {
                    let maybe_entry = match catch_unwind(AssertUnwindSafe(|| {
                        compile_jit_with_return_type(&expr, &args, return_type)
                    })) {
                        Ok(entry) => entry,
                        Err(payload) => {
                            let msg = panic_payload_to_string(payload);
                            jit_log(|| format!("[Iris][jit] panic while compiling expr '{}': {}", expr, msg));
                            None
                        }
                    };
                    if let Some(entry) = maybe_entry {
                        if let Some(name) = func_name.as_deref() {
                            register_named_jit(name, entry.clone());
                        }
                        register_jit(key, entry);
                        jit_log(|| format!("[Iris][jit] compiled JIT for function ptr={}", key));
                    } else {
                        jit_log(|| format!("[Iris][jit] failed to compile expr: {}", expr));
                    }
                };

                let compile_single_quantum = || {
                    let started = Instant::now();
                    let maybe_entry = match catch_unwind(AssertUnwindSafe(|| {
                        compile_jit_with_return_type(&expr, &args, return_type)
                    })) {
                        Ok(entry) => entry,
                        Err(payload) => {
                            let msg = panic_payload_to_string(payload);
                            jit_log(|| format!("[Iris][jit] panic while compiling warm quantum expr '{}': {}", expr, msg));
                            None
                        }
                    };

                    let elapsed = started.elapsed().as_nanos() as u64;
                    let success = maybe_entry.is_some();
                    record_quantum_compile_attempt(key, now_ns(), elapsed, success);

                    if let Some(entry) = maybe_entry {
                        if let Some(name) = func_name.as_deref() {
                            register_named_jit(name, entry.clone());
                        }
                        register_jit(key, entry.clone());
                        register_quantum_jit(key, vec![entry]);
                        jit_log(|| format!("[Iris][jit] warm-seeded single-variant compile for ptr={}", key));
                    } else {
                        jit_log(|| format!("[Iris][jit] failed warm-seeded single-variant compile: {}", expr));
                        compile_single();
                    }
                };

                if quantum_speculation_enabled() {
                    if warm_seed_hint {
                        compile_single_quantum();
                    } else {
                        let now = now_ns();
                        if quantum_compile_may_run(key, now) {
                            let started = Instant::now();
                            let entries = match catch_unwind(AssertUnwindSafe(|| {
                                compile_jit_quantum(&expr, &args, return_type)
                            })) {
                                Ok(entries) => entries,
                                Err(payload) => {
                                    let msg = panic_payload_to_string(payload);
                                    jit_log(|| format!("[Iris][jit] panic while compiling quantum variants '{}': {}", expr, msg));
                                    Vec::new()
                                }
                            };
                            let elapsed = started.elapsed().as_nanos() as u64;
                            let success = !entries.is_empty();
                            record_quantum_compile_attempt(key, now, elapsed, success);

                            if success {
                                if let Some(name) = func_name.as_deref() {
                                    register_named_jit(name, entries[0].clone());
                                }
                                register_quantum_jit(key, entries);
                                jit_log(|| format!("[Iris][jit] compiled quantum JIT variants for ptr={}", key));
                            } else {
                                jit_log(|| format!("[Iris][jit] failed to compile quantum variants: {}", expr));
                                compile_single();
                            }
                        } else {
                            jit_log(|| format!("[Iris][jit] quantum compile gated by cooldown/budget for ptr={}", key));
                            compile_single();
                        }
                    }
                } else {
                    compile_single();
                }
            }
        }
    }
    jit_log(|| {
        format!(
            "[Iris][jit] register_offload called strategy={:?} return_type={:?} source={:?} args={:?}",
            strategy, return_type, source_expr, arg_names
        )
    });
    Ok(func)
}

/// Execute a Python callable on the offload actor pool, blocking until result.
#[cfg(feature = "pyo3")]
#[pyfunction]
fn offload_call(
    py: Python,
    func: PyObject,
    args: &PyTuple,
    kwargs: Option<&PyDict>,
) -> PyResult<PyObject> {
    let key = func.as_ptr() as usize;
    if let Some(res) = execute_registered_jit_guarded(py, key, args) {
        if let Ok(obj) = res {
            return Ok(obj);
        }
        if let Err(err) = &res {
            jit_log(|| format!("[Iris][jit] guarded execution failed in offload_call; falling back: {}", err));
        }
    }

    let pool = get_offload_pool();

    let (tx, rx) = std::sync::mpsc::channel();
    let task = OffloadTask {
        func: func.into_py(py),
        args: args.into_py(py),
        kwargs: kwargs.map(|d: &PyDict| d.into_py(py)),
        resp: tx,
    };

    pool.sender
        .send(task)
        .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("offload queue closed"))?;

    let result = py.allow_threads(move || match rx.recv() {
        Ok(res) => res,
        Err(_) => Err(pyo3::exceptions::PyRuntimeError::new_err(
            "offload task canceled",
        )),
    });

    result
}

/// Directly invoke the JIT-compiled version of a Python function.
#[cfg(feature = "pyo3")]
#[pyfunction]
fn call_jit(
    py: Python,
    func: PyObject,
    args: &PyTuple,
    _kwargs: Option<&PyDict>,
) -> PyResult<PyObject> {
    let key = func.as_ptr() as usize;
    if let Some(res) = execute_registered_jit_guarded(py, key, args) {
        return res;
    }
    Err(pyo3::exceptions::PyRuntimeError::new_err("no JIT entry found"))
}

/// Execute a registered scalar 2-arg JIT step function in a Rust loop.
///
/// This is used by Python wrappers for recurrence kernels to avoid Python↔Rust
/// crossing overhead on each iteration.
#[cfg(feature = "pyo3")]
#[pyfunction]
fn call_jit_step_loop_f64(func: PyObject, seed: f64, count: usize) -> PyResult<f64> {
    let key = func.as_ptr() as usize;
    let entry = lookup_jit(key).ok_or_else(|| {
        pyo3::exceptions::PyRuntimeError::new_err("no JIT entry found")
    })?;

    if entry.arg_count != 2 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "step loop requires a 2-argument scalar JIT entry",
        ));
    }

    let run = || {
        let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry.func_ptr) };
        let mut values = [seed, 0.0_f64];
        let mut state = seed;
        for i in 0..count {
            values[0] = state;
            values[1] = i as f64;
            state = f(values.as_ptr());
        }
        state
    };

    match catch_unwind(AssertUnwindSafe(run)) {
        Ok(out) => Ok(out),
        Err(payload) => {
            let msg = panic_payload_to_string(payload);
            Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "jit panic: {}",
                msg
            )))
        }
    }
}

// ------- tests ------------------------------------------------------------

#[cfg(test)]
mod tests;
