use crate::py::vortex_bytecode::{
    probe_raw_bytes,
};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::atomic::{AtomicUsize, Ordering};

pyo3::create_exception!(iris, VortexSuspend, pyo3::exceptions::PyException);

static BUDGET: AtomicUsize = AtomicUsize::new(0);

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

    // Keep parser/probe generation exercised for hardening while using guarded fallback.
    let _ = probe_raw_bytes(py)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("vortex/probe-build: {e}")))?;

    // Vortex Guard fallback: isolate execution into a shadow trampoline and
    // use opcode-level tracing to enforce reduction checks without mutating
    // shared code objects while reassembler hardening continues.
    let locals = PyDict::new(py);
    locals.set_item("fn", py_func)?;
    py.run(
        r#"
def _iris_make_shadow(fn):
    import sys

    def _trace(frame, event, arg):
        if event == "call":
            frame.f_trace_opcodes = True
            return _trace
        if event == "opcode" or event == "line":
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
    m.add_function(wrap_pyfunction!(transmute_function, m)?)?;
    Ok(())
}
