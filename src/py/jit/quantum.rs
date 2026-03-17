// src/py/jit/quantum.rs
//! Quantum compilation control plane.
//!
//! Tracks compile budgets, cooldowns, and rearm plans for quantum JIT variants.

use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use crate::py::jit::codegen::JitReturnType;
use crate::py::jit::codegen::{compile_jit_quantum, register_quantum_jit};
use crate::py::jit::config::{
    jit_log, panic_payload_to_string, quantum_compile_budget_ns, quantum_compile_window_ns,
    quantum_cooldown_base_ns, quantum_cooldown_max_ns, quantum_rearm_interval_ns,
    quantum_rearm_max_volatility, quantum_rearm_min_observed_ns, quantum_rearm_min_samples,
    quantum_speculation_enabled, quantum_speculation_threshold_ns,
};

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

#[derive(Clone)]
struct QuantumRearmPlan {
    expr: String,
    args: Vec<String>,
    return_type: JitReturnType,
}

#[derive(Default, Clone, Copy)]
struct QuantumRearmObserved {
    samples: u64,
    ewma_ns: f64,
    ewma_abs_delta_ns: f64,
}

static QUANTUM_COMPILE_BUDGET_STATE: OnceLock<Mutex<QuantumCompileBudgetState>> = OnceLock::new();
static QUANTUM_COOLDOWN_STATE: OnceLock<Mutex<HashMap<usize, QuantumCooldownState>>> =
    OnceLock::new();
static QUANTUM_REARM_PLANS: OnceLock<Mutex<HashMap<usize, QuantumRearmPlan>>> = OnceLock::new();
static QUANTUM_REARM_LAST_ATTEMPT: OnceLock<Mutex<HashMap<usize, u64>>> = OnceLock::new();
static QUANTUM_REARM_OBSERVED: OnceLock<Mutex<HashMap<usize, QuantumRearmObserved>>> =
    OnceLock::new();

fn record_quantum_rearm_observation(func_key: usize, observed_ns: u64) -> (u64, f64) {
    const ALPHA: f64 = 0.25;
    let observed = observed_ns as f64;
    let state = QUANTUM_REARM_OBSERVED.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = state.lock().unwrap();
    let slot = guard.entry(func_key).or_default();

    if slot.samples == 0 {
        slot.samples = 1;
        slot.ewma_ns = observed;
        slot.ewma_abs_delta_ns = 0.0;
    } else {
        let delta = (observed - slot.ewma_ns).abs();
        slot.ewma_ns = (1.0 - ALPHA) * slot.ewma_ns + ALPHA * observed;
        slot.ewma_abs_delta_ns = (1.0 - ALPHA) * slot.ewma_abs_delta_ns + ALPHA * delta;
        slot.samples = slot.samples.saturating_add(1);
    }

    let volatility = if slot.ewma_ns > 0.0 {
        (slot.ewma_abs_delta_ns / slot.ewma_ns).clamp(0.0, 4.0)
    } else {
        0.0
    };
    (slot.samples, volatility)
}

pub(crate) fn register_quantum_rearm_plan(
    func_key: usize,
    expr: &str,
    args: &[String],
    return_type: JitReturnType,
) {
    let plans = QUANTUM_REARM_PLANS.get_or_init(|| Mutex::new(HashMap::new()));
    plans.lock().unwrap().insert(
        func_key,
        QuantumRearmPlan {
            expr: expr.to_string(),
            args: args.to_vec(),
            return_type,
        },
    );
}

pub(crate) fn maybe_rearm_quantum_compile(
    func_key: usize,
    observed_ns: u64,
    active_variants: usize,
) -> bool {
    if active_variants > 1 || !quantum_speculation_enabled() {
        return false;
    }

    let (samples, volatility) = record_quantum_rearm_observation(func_key, observed_ns);
    let min_samples = quantum_rearm_min_samples();
    if samples < min_samples {
        return false;
    }
    if volatility > quantum_rearm_max_volatility() {
        jit_log(|| {
            format!(
                "[Iris][jit] quantum rearm skipped for ptr={} due to volatility {:.3}",
                func_key, volatility
            )
        });
        return false;
    }

    let volatility_factor = (1.0 + volatility).clamp(1.0, 3.0);
    let observed_gate = ((quantum_speculation_threshold_ns().max(quantum_rearm_min_observed_ns())
        as f64)
        * volatility_factor) as u64;
    if observed_ns < observed_gate {
        return false;
    }

    let plan = {
        let plans = QUANTUM_REARM_PLANS.get_or_init(|| Mutex::new(HashMap::new()));
        plans.lock().unwrap().get(&func_key).cloned()
    };
    let Some(plan) = plan else {
        return false;
    };

    let now = super::config::now_ns();
    let interval_scale = if samples < min_samples.saturating_mul(2) {
        2.0
    } else {
        1.0
    };
    let interval =
        ((quantum_rearm_interval_ns() as f64) * interval_scale * volatility_factor) as u64;
    {
        let attempts = QUANTUM_REARM_LAST_ATTEMPT.get_or_init(|| Mutex::new(HashMap::new()));
        let mut guard = attempts.lock().unwrap();
        if let Some(last) = guard.get(&func_key).copied() {
            if now.saturating_sub(last) < interval {
                return false;
            }
        }
        guard.insert(func_key, now);
    }

    if !quantum_compile_may_run(func_key, now) {
        jit_log(|| {
            format!(
                "[Iris][jit] quantum rearm gated by cooldown/budget for ptr={}",
                func_key
            )
        });
        return false;
    }

    let started = Instant::now();
    let entries = match catch_unwind(AssertUnwindSafe(|| {
        compile_jit_quantum(&plan.expr, &plan.args, plan.return_type)
    })) {
        Ok(entries) => entries,
        Err(payload) => {
            let msg = panic_payload_to_string(payload);
            jit_log(|| {
                format!(
                    "[Iris][jit] panic during quantum rearm for ptr={}: {}",
                    func_key, msg
                )
            });
            Vec::new()
        }
    };

    let elapsed = started.elapsed().as_nanos() as u64;
    let success = entries.len() > 1;
    record_quantum_compile_attempt(func_key, now, elapsed, success);

    if success {
        register_quantum_jit(func_key, entries);
        jit_log(|| format!("[Iris][jit] quantum rearmed variants for ptr={}", func_key));
        true
    } else {
        jit_log(|| {
            format!(
                "[Iris][jit] quantum rearm skipped (insufficient variants) for ptr={}",
                func_key
            )
        });
        false
    }
}

#[cfg(test)]
pub(crate) fn register_quantum_rearm_plan_for_test(
    func_key: usize,
    expr: &str,
    args: &[String],
    return_type: JitReturnType,
) {
    register_quantum_rearm_plan(func_key, expr, args, return_type);
}

#[cfg(test)]
pub(crate) fn clear_quantum_rearm_plan_for_test(func_key: usize) {
    if let Some(plans) = QUANTUM_REARM_PLANS.get() {
        plans.lock().unwrap().remove(&func_key);
    }
    if let Some(attempts) = QUANTUM_REARM_LAST_ATTEMPT.get() {
        attempts.lock().unwrap().remove(&func_key);
    }
    if let Some(observed) = QUANTUM_REARM_OBSERVED.get() {
        observed.lock().unwrap().remove(&func_key);
    }
}

pub(crate) fn quantum_compile_may_run(func_key: usize, now_ns: u64) -> bool {
    let cooldown_map = QUANTUM_COOLDOWN_STATE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(state) = cooldown_map.lock().unwrap().get(&func_key).copied() {
        if now_ns < state.cooldown_until_ns {
            return false;
        }
    }

    let window_ns = quantum_compile_window_ns();
    let budget_ns = quantum_compile_budget_ns();
    let budget_state = QUANTUM_COMPILE_BUDGET_STATE
        .get_or_init(|| Mutex::new(QuantumCompileBudgetState::default()));
    let mut budget = budget_state.lock().unwrap();
    if budget.window_start_ns == 0 || now_ns.saturating_sub(budget.window_start_ns) >= window_ns {
        budget.window_start_ns = now_ns;
        budget.consumed_ns = 0;
    }
    budget.consumed_ns < budget_ns
}

pub(crate) fn record_quantum_compile_attempt(
    func_key: usize,
    now_ns: u64,
    elapsed_ns: u64,
    success: bool,
) {
    let window_ns = quantum_compile_window_ns();
    let budget_state = QUANTUM_COMPILE_BUDGET_STATE
        .get_or_init(|| Mutex::new(QuantumCompileBudgetState::default()));
    {
        let mut budget = budget_state.lock().unwrap();
        if budget.window_start_ns == 0 || now_ns.saturating_sub(budget.window_start_ns) >= window_ns
        {
            budget.window_start_ns = now_ns;
            budget.consumed_ns = 0;
        }
        budget.consumed_ns = budget.consumed_ns.saturating_add(elapsed_ns);
    }

    let cooldown_map = QUANTUM_COOLDOWN_STATE.get_or_init(|| Mutex::new(HashMap::new()));
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
    if let Some(state) = QUANTUM_REARM_LAST_ATTEMPT.get() {
        state.lock().unwrap().clear();
    }
    if let Some(state) = QUANTUM_REARM_OBSERVED.get() {
        state.lock().unwrap().clear();
    }
}
