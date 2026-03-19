use crate::py::vortex_bytecode::{
    decode_wordcode, encode_wordcode, instrument_with_probe, opcode_meta, probe_instructions,
    quickening_support, evaluate_rewrite_compatibility, validate_probe_compatibility,
    verify_cache_layout,
};
use once_cell::sync::Lazy;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyBytes, PyDict};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

pyo3::create_exception!(iris, VortexSuspend, pyo3::exceptions::PyException);

static BUDGET: AtomicUsize = AtomicUsize::new(0);
const MAX_PATCHED_CODE_BYTES: usize = 8 * 1024 * 1024;

#[derive(Debug, Clone)]
struct GuardTelemetry {
    mode: String,
    reason: String,
    py_minor: i32,
    rewrite_attempted: bool,
    rewrite_applied: bool,
}

impl Default for GuardTelemetry {
    fn default() -> Self {
        GuardTelemetry {
            mode: "unset".to_string(),
            reason: "none".to_string(),
            py_minor: -1,
            rewrite_attempted: false,
            rewrite_applied: false,
        }
    }
}

static GUARD_TELEMETRY: Lazy<Mutex<GuardTelemetry>> = Lazy::new(|| Mutex::new(GuardTelemetry::default()));

fn set_guard_telemetry(mode: &str, reason: &str, py_minor: i32, attempted: bool, applied: bool) {
    if let Ok(mut g) = GUARD_TELEMETRY.lock() {
        g.mode = mode.to_string();
        g.reason = reason.to_string();
        g.py_minor = py_minor;
        g.rewrite_attempted = attempted;
        g.rewrite_applied = applied;
    }
}

#[pyfunction]
pub fn get_guard_status(py: Python) -> PyResult<PyObject> {
    let g = GUARD_TELEMETRY
        .lock()
        .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("vortex/guard-status: lock poisoned"))?
        .clone();

    let d = PyDict::new(py);
    d.set_item("mode", g.mode)?;
    d.set_item("reason", g.reason)?;
    d.set_item("py_minor", g.py_minor)?;
    d.set_item("rewrite_attempted", g.rewrite_attempted)?;
    d.set_item("rewrite_applied", g.rewrite_applied)?;
    Ok(d.into())
}

#[pyfunction]
pub fn _vortex_check() -> PyResult<()> {
    let current = BUDGET.load(Ordering::Relaxed);
    if current == 0 {
        return Err(VortexSuspend::new_err("budget exhausted"));
    }
    BUDGET.store(current - 1, Ordering::Relaxed);
    Ok(())
}

#[pyfunction]
pub fn set_budget(budget: usize) {
    BUDGET.store(budget, Ordering::Relaxed);
}

#[pyfunction]
pub fn transmute_function(py: Python, py_func: &PyAny) -> PyResult<PyObject> {
    let py_minor: i32 = py
        .eval("__import__('sys').version_info.minor", None, None)
        .and_then(|v| v.extract())
        .unwrap_or(99);

    let code = py_func
        .getattr("__code__")
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("vortex/code: {e}")))?;
    let raw: &[u8] = code
        .getattr("co_code")
        .and_then(|v| v.extract())
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("vortex/co_code: {e}")))?;

    let globals_any = py_func
        .getattr("__globals__")
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("vortex/globals: {e}")))?;
    let globals = globals_any
        .downcast::<PyDict>()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("vortex/globals-cast: {e}")))?;
    let local_mod = match py
        .import("sys")
        .and_then(|s| s.getattr("modules"))
        .and_then(|mods| mods.get_item("iris"))
    {
        Ok(m) => m,
        Err(_) => globals
            .get_item("iris")
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("vortex/module-lookup: iris missing"))?,
    };
    let check_fn = local_mod
        .getattr("_vortex_check")
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("vortex/check-fn: {e}")))?;
    globals
        .set_item("_vortex_check", check_fn)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("vortex/globals-inject: {e}")))?;

    // Primary RFC path: bytecode-level shadow clone with capability checks.
    let meta = match opcode_meta(py) {
        Ok(m) => m,
        Err(_) => {
            set_guard_telemetry("fallback", "opcode_metadata_unavailable", py_minor, false, false);
            return fallback_shadow(py, py_func, "opcode metadata unavailable");
        }
    };

    let quickening = match quickening_support(py) {
        Ok(q) => q,
        Err(_) => {
            set_guard_telemetry("fallback", "quickening_metadata_unavailable", py_minor, false, false);
            return fallback_shadow(py, py_func, "quickening metadata unavailable");
        }
    };

    if let Err(reason) = evaluate_rewrite_compatibility(raw, meta.extended_arg, &quickening) {
        set_guard_telemetry("fallback", reason, py_minor, false, false);
        return fallback_shadow(py, py_func, reason);
    }

    set_guard_telemetry("rewrite", "attempt", py_minor, true, false);
    let original = decode_wordcode(raw, meta.extended_arg);
    if let Ok(probe) = probe_instructions(py, meta.extended_arg) {
        if let Err(reason) = validate_probe_compatibility(&probe, &quickening) {
            set_guard_telemetry("fallback", reason, py_minor, true, false);
            return fallback_shadow(py, py_func, reason);
        }

        if let Ok(patched) = instrument_with_probe(&original, &probe, &meta) {
            if verify_cache_layout(&patched, &quickening).is_err() {
                set_guard_telemetry("fallback", "patched_cache_layout_invalid", py_minor, true, false);
                return fallback_shadow(py, py_func, "patched cache layout invalid");
            }

            let patched_raw = encode_wordcode(&patched, meta.extended_arg);
            if patched_raw.len() <= MAX_PATCHED_CODE_BYTES {
                let kwargs = [("co_code", PyBytes::new(py, &patched_raw))].into_py_dict(py);
                if let Ok(new_code) = code.call_method("replace", (), Some(kwargs)) {
                    if let Ok(types_mod) = py.import("types") {
                        if let Ok(shadow) = types_mod.getattr("FunctionType").and_then(|ctor| {
                            ctor.call1((
                                new_code,
                                globals,
                                py_func.getattr("__name__")?,
                                py_func.getattr("__defaults__")?,
                                py_func.getattr("__closure__")?,
                            ))
                        }) {
                            if let Ok(kwdefaults) = py_func.getattr("__kwdefaults__") {
                                let _ = shadow.setattr("__kwdefaults__", kwdefaults);
                            }
                            set_guard_telemetry("rewrite", "applied", py_minor, true, true);
                            return Ok(shadow.into());
                        }
                    }
                }
            } else {
                set_guard_telemetry("fallback", "patched_code_too_large", py_minor, true, false);
                return fallback_shadow(py, py_func, "patched code too large");
            }
        }
    }

    set_guard_telemetry("fallback", "rewrite_pipeline_failed", py_minor, true, false);

    fallback_shadow(py, py_func, "guard fallback")
}

fn fallback_shadow(py: Python, py_func: &PyAny, _reason: &str) -> PyResult<PyObject> {
    let globals_any = py_func
        .getattr("__globals__")
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("vortex/globals: {e}")))?;
    let globals = globals_any
        .downcast::<PyDict>()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("vortex/globals-cast: {e}")))?;

    let locals = PyDict::new(py);
    locals.set_item("fn", py_func)?;
    py.run(
        r#"
def _iris_make_shadow(fn):
    import sys
    target_code = fn.__code__

    def _trace(frame, event, arg):
        if frame.f_code is not target_code:
            return _trace
        if event == "call":
            return _trace
        if event == "line":
            _vortex_check()
        return _trace

    def _wrapped(*a, **k):
        old = sys.gettrace()
        sys.settrace(_trace)
        try:
            return fn(*a, **k)
        finally:
            sys.settrace(old)

    return _wrapped

shadow = _iris_make_shadow(fn)
"#,
        Some(globals),
        Some(locals),
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("vortex/shadow-fallback: {e}")))?;
    let shadow = locals
        .get_item("shadow")
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("vortex/shadow-fallback: missing shadow"))?;

    Ok(shadow.into())
}

pub fn init_py(m: &PyModule) -> PyResult<()> {
    m.add("VortexSuspend", m.py().get_type::<VortexSuspend>())?;
    m.add_function(wrap_pyfunction!(_vortex_check, m)?)?;
    m.add_function(wrap_pyfunction!(set_budget, m)?)?;
    m.add_function(wrap_pyfunction!(get_guard_status, m)?)?;
    m.add_function(wrap_pyfunction!(transmute_function, m)?)?;
    Ok(())
}
