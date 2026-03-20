#![cfg(all(feature = "pyo3", feature = "vortex"))]

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use tokio::time::{sleep, timeout, Duration};

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_vortex_preemption_on_while_true() {
    Python::with_gil(|py| {
        let m = iris::py::make_module(py).unwrap();

        let globals = PyDict::new(py);
        globals.set_item("iris", &m).unwrap();

        // Define an endless loop
        let code = r#"
def endless():
    while True:
        pass
"#;
        py.run(code, Some(globals), None).unwrap();

        let endless_func = globals.get_item("endless").unwrap();
        let original_code = endless_func
            .getattr("__code__")
            .unwrap()
            .getattr("co_code")
            .unwrap()
            .to_object(py);

        m.getattr(py, "set_budget").unwrap().call1(py, (5,)).unwrap();

        // Shadow clone transmutation should not mutate the original function object.
        let shadow = m
            .getattr(py, "transmute_function")
            .unwrap()
            .call1(py, (endless_func,))
            .unwrap();
        assert!(!shadow.as_ref(py).is(endless_func));

        let guard = m
            .getattr(py, "get_guard_status")
            .unwrap()
            .call0(py)
            .unwrap();
        let guard_dict = guard.downcast::<PyDict>(py).unwrap();
        let mode: String = guard_dict.get_item("mode").unwrap().extract().unwrap();
        let rewrite_attempted: bool = guard_dict
            .get_item("rewrite_attempted")
            .unwrap()
            .extract()
            .unwrap();
        assert!(mode == "rewrite" || mode == "fallback");
        assert!(rewrite_attempted || mode == "fallback");

        let current_original_code = endless_func
            .getattr("__code__")
            .unwrap()
            .getattr("co_code")
            .unwrap()
            .to_object(py);
        assert!(original_code.as_ref(py).eq(current_original_code.as_ref(py)).unwrap());

        // Run transmuted shadow function. It should suspend by budget.
        let res = shadow.call0(py);

        assert!(res.is_err());
        let err = res.unwrap_err();
        assert!(err.is_instance_of::<iris::py::vortex::VortexSuspend>(py));
    });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_vortex_fallback_reports_opcode_metadata_unavailable() {
    Python::with_gil(|py| {
        let m = iris::py::make_module(py).unwrap();

        let globals = PyDict::new(py);
        globals.set_item("iris", &m).unwrap();
        py.run(
            r#"
def sample():
    return 42
"#,
            Some(globals),
            None,
        )
        .unwrap();

        let sample = globals.get_item("sample").unwrap();

        // Ensure dis is present in sys.modules before monkeypatching it.
        py.import("dis").unwrap();
        let sys = py.import("sys").unwrap();
        let modules = sys.getattr("modules").unwrap().downcast::<PyDict>().unwrap();
        let original_dis = modules.get_item("dis").unwrap().to_object(py);

        modules.set_item("dis", py.eval("object()", None, None).unwrap()).unwrap();

        let shadow = m
            .getattr(py, "transmute_function")
            .unwrap()
            .call1(py, (sample,))
            .unwrap();
        assert!(!shadow.as_ref(py).is(sample));

        let guard = m
            .getattr(py, "get_guard_status")
            .unwrap()
            .call0(py)
            .unwrap();
        let guard_dict = guard.downcast::<PyDict>(py).unwrap();
        let mode: String = guard_dict.get_item("mode").unwrap().extract().unwrap();
        let reason: String = guard_dict.get_item("reason").unwrap().extract().unwrap();
        let rewrite_attempted: bool = guard_dict
            .get_item("rewrite_attempted")
            .unwrap()
            .extract()
            .unwrap();
        let rewrite_applied: bool = guard_dict
            .get_item("rewrite_applied")
            .unwrap()
            .extract()
            .unwrap();

        assert_eq!(mode, "fallback");
        assert_eq!(reason, "opcode_metadata_unavailable");
        assert!(!rewrite_attempted);
        assert!(!rewrite_applied);

        modules.set_item("dis", original_dis).unwrap();
    });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_vortex_fallback_reports_quickening_metadata_unavailable() {
    Python::with_gil(|py| {
        let m = iris::py::make_module(py).unwrap();

        let globals = PyDict::new(py);
        globals.set_item("iris", &m).unwrap();
        py.run(
            r#"
def sample2():
    return 7
"#,
            Some(globals),
            None,
        )
        .unwrap();

        let sample = globals.get_item("sample2").unwrap();
        let dis = py.import("dis").unwrap();
        let had_inline = dis.hasattr("_inline_cache_entries").unwrap();
        let original_inline = if had_inline {
            Some(dis.getattr("_inline_cache_entries").unwrap().to_object(py))
        } else {
            None
        };

        // Force quickening metadata extraction to fail.
        dis.setattr("_inline_cache_entries", py.eval("object()", None, None).unwrap())
            .unwrap();

        let shadow_res = m
            .getattr(py, "transmute_function")
            .unwrap()
            .call1(py, (sample,));

        if had_inline {
            dis.setattr("_inline_cache_entries", original_inline.unwrap())
                .unwrap();
        } else {
            let _ = dis.delattr("_inline_cache_entries");
        }

        let shadow = shadow_res.unwrap();
        assert!(!shadow.as_ref(py).is(sample));

        let guard = m
            .getattr(py, "get_guard_status")
            .unwrap()
            .call0(py)
            .unwrap();
        let guard_dict = guard.downcast::<PyDict>(py).unwrap();
        let mode: String = guard_dict.get_item("mode").unwrap().extract().unwrap();
        let reason: String = guard_dict.get_item("reason").unwrap().extract().unwrap();
        let rewrite_attempted: bool = guard_dict
            .get_item("rewrite_attempted")
            .unwrap()
            .extract()
            .unwrap();
        let rewrite_applied: bool = guard_dict
            .get_item("rewrite_applied")
            .unwrap()
            .extract()
            .unwrap();

        assert_eq!(mode, "fallback");
        assert_eq!(reason, "quickening_metadata_unavailable");
        assert!(!rewrite_attempted);
        assert!(!rewrite_applied);
    });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_vortex_fallback_reports_original_cache_layout_invalid() {
    Python::with_gil(|py| {
        let m = iris::py::make_module(py).unwrap();

        let globals = PyDict::new(py);
        globals.set_item("iris", &m).unwrap();
        py.run(
            r#"
def sample3():
    return 99
"#,
            Some(globals),
            None,
        )
        .unwrap();

        let sample = globals.get_item("sample3").unwrap();
        let dis = py.import("dis").unwrap();
        let locals = PyDict::new(py);
        locals.set_item("dis", dis).unwrap();
        let cache_opcode: i32 = py
            .eval("dis.opmap.get('CACHE', -1)", Some(locals), None)
            .unwrap()
            .extract()
            .unwrap();

        if cache_opcode < 0 {
            // Python runtime has no inline cache opcode, so this path is not applicable.
            return;
        }

        let had_inline = dis.hasattr("_inline_cache_entries").unwrap();
        let original_inline = if had_inline {
            Some(dis.getattr("_inline_cache_entries").unwrap().to_object(py))
        } else {
            None
        };

        // Force a clearly incompatible table: every non-CACHE opcode expects a cache slot.
        let mut entries = vec![1u16; 256];
        entries[cache_opcode as usize] = 0;
        dis.setattr("_inline_cache_entries", PyList::new(py, entries))
            .unwrap();

        let shadow_res = m
            .getattr(py, "transmute_function")
            .unwrap()
            .call1(py, (sample,));

        if had_inline {
            dis.setattr("_inline_cache_entries", original_inline.unwrap())
                .unwrap();
        } else {
            let _ = dis.delattr("_inline_cache_entries");
        }

        let shadow = shadow_res.unwrap();
        assert!(!shadow.as_ref(py).is(sample));

        let guard = m
            .getattr(py, "get_guard_status")
            .unwrap()
            .call0(py)
            .unwrap();
        let guard_dict = guard.downcast::<PyDict>(py).unwrap();
        let mode: String = guard_dict.get_item("mode").unwrap().extract().unwrap();
        let reason: String = guard_dict.get_item("reason").unwrap().extract().unwrap();
        let rewrite_attempted: bool = guard_dict
            .get_item("rewrite_attempted")
            .unwrap()
            .extract()
            .unwrap();
        let rewrite_applied: bool = guard_dict
            .get_item("rewrite_applied")
            .unwrap()
            .extract()
            .unwrap();

        assert_eq!(mode, "fallback");
        assert_eq!(reason, "original_cache_layout_invalid");
        assert!(!rewrite_attempted);
        assert!(!rewrite_applied);
    });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_vortex_fallback_reports_stack_depth_invariant_failed() {
    Python::with_gil(|py| {
        let m = iris::py::make_module(py).unwrap();

        let globals = PyDict::new(py);
        globals.set_item("iris", &m).unwrap();
        py.run(
            r#"
def sample_small_stack():
    return 1
"#,
            Some(globals),
            None,
        )
        .unwrap();

        let sample = globals.get_item("sample_small_stack").unwrap();
        let original_code = sample.getattr("__code__").unwrap().to_object(py);
        let locals = PyDict::new(py);
        locals.set_item("fn", sample).unwrap();
        py.run(
            r#"
fn.__code__ = fn.__code__.replace(co_stacksize=1)
"#,
            Some(locals),
            Some(locals),
        )
        .unwrap();

        let dis = py.import("dis").unwrap();
        let had_inline = dis.hasattr("_inline_cache_entries").unwrap();
        let original_inline = if had_inline {
            Some(dis.getattr("_inline_cache_entries").unwrap().to_object(py))
        } else {
            None
        };
        dis.setattr("_inline_cache_entries", PyList::new(py, vec![0u16; 256]))
            .unwrap();

        let shadow_res = m
            .getattr(py, "transmute_function")
            .unwrap()
            .call1(py, (sample,));

        sample.setattr("__code__", original_code).unwrap();
        if had_inline {
            dis.setattr("_inline_cache_entries", original_inline.unwrap())
                .unwrap();
        } else {
            let _ = dis.delattr("_inline_cache_entries");
        }

        let shadow = shadow_res.unwrap();
        assert!(!shadow.as_ref(py).is(sample));

        let guard = m
            .getattr(py, "get_guard_status")
            .unwrap()
            .call0(py)
            .unwrap();
        let guard_dict = guard.downcast::<PyDict>(py).unwrap();
        let mode: String = guard_dict.get_item("mode").unwrap().extract().unwrap();
        let reason: String = guard_dict.get_item("reason").unwrap().extract().unwrap();
        let rewrite_attempted: bool = guard_dict
            .get_item("rewrite_attempted")
            .unwrap()
            .extract()
            .unwrap();
        let rewrite_applied: bool = guard_dict
            .get_item("rewrite_applied")
            .unwrap()
            .extract()
            .unwrap();

        assert_eq!(mode, "fallback");
        assert_eq!(reason, "stack_depth_invariant_failed");
        assert!(!rewrite_attempted);
        assert!(!rewrite_applied);
    });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_vortex_fallback_reports_exception_table_invalid() {
    Python::with_gil(|py| {
        let m = iris::py::make_module(py).unwrap();

        let globals = PyDict::new(py);
        globals.set_item("iris", &m).unwrap();
        py.run(
            r#"
def sample_exc():
    a = 1
    b = 2
    return a + b
"#,
            Some(globals),
            None,
        )
        .unwrap();

        let sample = globals.get_item("sample_exc").unwrap();
        let dis = py.import("dis").unwrap();
        let had_inline = dis.hasattr("_inline_cache_entries").unwrap();
        let original_inline = if had_inline {
            Some(dis.getattr("_inline_cache_entries").unwrap().to_object(py))
        } else {
            None
        };
        dis.setattr("_inline_cache_entries", PyList::new(py, vec![0u16; 256]))
            .unwrap();

        let original_bytecode = dis.getattr("Bytecode").unwrap().to_object(py);
        let locals = PyDict::new(py);
        locals.set_item("dis", dis).unwrap();
        py.run(
            r#"
import types

class _IrisBadEntry:
    def __init__(self):
        self.start = 0
        self.end = 999999
        self.depth = 0

class _IrisBadBytecode:
    def __init__(self, _code):
        self.exception_entries = [_IrisBadEntry()]

dis.Bytecode = _IrisBadBytecode
"#,
            Some(locals),
            Some(locals),
        )
        .unwrap();

        let shadow_res = m
            .getattr(py, "transmute_function")
            .unwrap()
            .call1(py, (sample,));

        dis.setattr("Bytecode", original_bytecode).unwrap();
        if had_inline {
            dis.setattr("_inline_cache_entries", original_inline.unwrap())
                .unwrap();
        } else {
            let _ = dis.delattr("_inline_cache_entries");
        }

        let shadow = shadow_res.unwrap();
        assert!(!shadow.as_ref(py).is(sample));

        let guard = m
            .getattr(py, "get_guard_status")
            .unwrap()
            .call0(py)
            .unwrap();
        let guard_dict = guard.downcast::<PyDict>(py).unwrap();
        let mode: String = guard_dict.get_item("mode").unwrap().extract().unwrap();
        let reason: String = guard_dict.get_item("reason").unwrap().extract().unwrap();
        let rewrite_attempted: bool = guard_dict
            .get_item("rewrite_attempted")
            .unwrap()
            .extract()
            .unwrap();
        let rewrite_applied: bool = guard_dict
            .get_item("rewrite_applied")
            .unwrap()
            .extract()
            .unwrap();

        assert_eq!(mode, "fallback");
        assert_eq!(reason, "exception_table_invalid");
        assert!(!rewrite_attempted);
        assert!(!rewrite_applied);
    });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_vortex_fallback_reports_exception_table_metadata_unavailable() {
    Python::with_gil(|py| {
        let m = iris::py::make_module(py).unwrap();

        let globals = PyDict::new(py);
        globals.set_item("iris", &m).unwrap();
        py.run(
            r#"
def sample_exc_meta_unavailable():
    a = 1
    b = 2
    return a + b
"#,
            Some(globals),
            None,
        )
        .unwrap();

        let sample = globals.get_item("sample_exc_meta_unavailable").unwrap();
        let dis = py.import("dis").unwrap();
        let had_inline = dis.hasattr("_inline_cache_entries").unwrap();
        let original_inline = if had_inline {
            Some(dis.getattr("_inline_cache_entries").unwrap().to_object(py))
        } else {
            None
        };
        dis.setattr("_inline_cache_entries", PyList::new(py, vec![0u16; 256]))
            .unwrap();

        let original_bytecode = dis.getattr("Bytecode").unwrap().to_object(py);
        let locals = PyDict::new(py);
        locals.set_item("dis", dis).unwrap();
        py.run(
            r#"
class _IrisFailBytecode:
    def __init__(self, _code):
        raise RuntimeError("forced bytecode metadata failure")

dis.Bytecode = _IrisFailBytecode
"#,
            Some(locals),
            Some(locals),
        )
        .unwrap();

        let shadow_res = m
            .getattr(py, "transmute_function")
            .unwrap()
            .call1(py, (sample,));

        dis.setattr("Bytecode", original_bytecode).unwrap();
        if had_inline {
            dis.setattr("_inline_cache_entries", original_inline.unwrap())
                .unwrap();
        } else {
            let _ = dis.delattr("_inline_cache_entries");
        }

        let shadow = shadow_res.unwrap();
        assert!(!shadow.as_ref(py).is(sample));

        let guard = m
            .getattr(py, "get_guard_status")
            .unwrap()
            .call0(py)
            .unwrap();
        let guard_dict = guard.downcast::<PyDict>(py).unwrap();
        let mode: String = guard_dict.get_item("mode").unwrap().extract().unwrap();
        let reason: String = guard_dict.get_item("reason").unwrap().extract().unwrap();
        let rewrite_attempted: bool = guard_dict
            .get_item("rewrite_attempted")
            .unwrap()
            .extract()
            .unwrap();
        let rewrite_applied: bool = guard_dict
            .get_item("rewrite_applied")
            .unwrap()
            .extract()
            .unwrap();

        assert_eq!(mode, "fallback");
        assert_eq!(reason, "exception_table_metadata_unavailable");
        assert!(!rewrite_attempted);
        assert!(!rewrite_applied);
    });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_vortex_fallback_reports_patched_exception_table_invalid() {
    Python::with_gil(|py| {
        let m = iris::py::make_module(py).unwrap();

        let globals = PyDict::new(py);
        globals.set_item("iris", &m).unwrap();
        py.run(
            r#"
def sample_patched_exc():
    a = 1
    b = 2
    return a + b
"#,
            Some(globals),
            None,
        )
        .unwrap();

        let sample = globals.get_item("sample_patched_exc").unwrap();
        let dis = py.import("dis").unwrap();
        let had_inline = dis.hasattr("_inline_cache_entries").unwrap();
        let original_inline = if had_inline {
            Some(dis.getattr("_inline_cache_entries").unwrap().to_object(py))
        } else {
            None
        };
        dis.setattr("_inline_cache_entries", PyList::new(py, vec![0u16; 256]))
            .unwrap();

        let os = py.import("os").unwrap();
        let environ = os.getattr("environ").unwrap();
        environ
            .set_item("IRIS_VORTEX_TEST_FORCE_PATCHED_EXCEPTION_TABLE_INVALID", "1")
            .unwrap();

        let shadow_res = m
            .getattr(py, "transmute_function")
            .unwrap()
            .call1(py, (sample,));

        let _ = environ.del_item("IRIS_VORTEX_TEST_FORCE_PATCHED_EXCEPTION_TABLE_INVALID");
        if had_inline {
            dis.setattr("_inline_cache_entries", original_inline.unwrap())
                .unwrap();
        } else {
            let _ = dis.delattr("_inline_cache_entries");
        }

        let shadow = shadow_res.unwrap();
        assert!(!shadow.as_ref(py).is(sample));

        let guard = m
            .getattr(py, "get_guard_status")
            .unwrap()
            .call0(py)
            .unwrap();
        let guard_dict = guard.downcast::<PyDict>(py).unwrap();
        let mode: String = guard_dict.get_item("mode").unwrap().extract().unwrap();
        let reason: String = guard_dict.get_item("reason").unwrap().extract().unwrap();
        let rewrite_attempted: bool = guard_dict
            .get_item("rewrite_attempted")
            .unwrap()
            .extract()
            .unwrap();
        let rewrite_applied: bool = guard_dict
            .get_item("rewrite_applied")
            .unwrap()
            .extract()
            .unwrap();

        assert_eq!(mode, "fallback");
        assert_eq!(reason, "patched_exception_table_invalid");
        assert!(rewrite_attempted);
        assert!(!rewrite_applied);
    });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_vortex_fallback_reports_probe_extraction_failed() {
    Python::with_gil(|py| {
        let m = iris::py::make_module(py).unwrap();

        let globals = PyDict::new(py);
        globals.set_item("iris", &m).unwrap();
        py.run(
            r#"
def sample_probe_fail():
    a = 1
    b = 2
    return a + b
"#,
            Some(globals),
            None,
        )
        .unwrap();

        let sample = globals.get_item("sample_probe_fail").unwrap();
        let dis = py.import("dis").unwrap();

        let had_inline = dis.hasattr("_inline_cache_entries").unwrap();
        let original_inline = if had_inline {
            Some(dis.getattr("_inline_cache_entries").unwrap().to_object(py))
        } else {
            None
        };
        dis.setattr("_inline_cache_entries", PyList::new(py, vec![0u16; 256]))
            .unwrap();

        let original_get_instructions = dis.getattr("get_instructions").unwrap().to_object(py);
        let locals = PyDict::new(py);
        locals.set_item("dis", dis).unwrap();
        py.run(
            r#"
def _iris_fail_get_instructions(*_args, **_kwargs):
    raise RuntimeError("forced probe extraction failure")

dis.get_instructions = _iris_fail_get_instructions
"#,
            Some(locals),
            Some(locals),
        )
        .unwrap();

        let shadow_res = m
            .getattr(py, "transmute_function")
            .unwrap()
            .call1(py, (sample,));

        dis.setattr("get_instructions", original_get_instructions).unwrap();
        if had_inline {
            dis.setattr("_inline_cache_entries", original_inline.unwrap())
                .unwrap();
        } else {
            let _ = dis.delattr("_inline_cache_entries");
        }

        let shadow = shadow_res.unwrap();
        assert!(!shadow.as_ref(py).is(sample));

        let guard = m
            .getattr(py, "get_guard_status")
            .unwrap()
            .call0(py)
            .unwrap();
        let guard_dict = guard.downcast::<PyDict>(py).unwrap();
        let mode: String = guard_dict.get_item("mode").unwrap().extract().unwrap();
        let reason: String = guard_dict.get_item("reason").unwrap().extract().unwrap();
        let rewrite_attempted: bool = guard_dict
            .get_item("rewrite_attempted")
            .unwrap()
            .extract()
            .unwrap();
        let rewrite_applied: bool = guard_dict
            .get_item("rewrite_applied")
            .unwrap()
            .extract()
            .unwrap();

        assert_eq!(mode, "fallback");
        assert_eq!(reason, "probe_extraction_failed");
        assert!(rewrite_attempted);
        assert!(!rewrite_applied);
    });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_vortex_fallback_reports_probe_instrumentation_failed() {
    Python::with_gil(|py| {
        let m = iris::py::make_module(py).unwrap();

        let globals = PyDict::new(py);
        globals.set_item("iris", &m).unwrap();
        py.run(
            r#"
def sample_probe_instrumentation_fail():
    a = 1
    b = 2
    return a + b
"#,
            Some(globals),
            None,
        )
        .unwrap();

        let sample = globals
            .get_item("sample_probe_instrumentation_fail")
            .unwrap();
        let dis = py.import("dis").unwrap();
        let had_inline = dis.hasattr("_inline_cache_entries").unwrap();
        let original_inline = if had_inline {
            Some(dis.getattr("_inline_cache_entries").unwrap().to_object(py))
        } else {
            None
        };
        dis.setattr("_inline_cache_entries", PyList::new(py, vec![0u16; 256]))
            .unwrap();

        let os = py.import("os").unwrap();
        let environ = os.getattr("environ").unwrap();
        environ
            .set_item("IRIS_VORTEX_TEST_FORCE_PROBE_INSTRUMENTATION_FAILED", "1")
            .unwrap();

        let shadow_res = m
            .getattr(py, "transmute_function")
            .unwrap()
            .call1(py, (sample,));

        let _ = environ.del_item("IRIS_VORTEX_TEST_FORCE_PROBE_INSTRUMENTATION_FAILED");
        if had_inline {
            dis.setattr("_inline_cache_entries", original_inline.unwrap())
                .unwrap();
        } else {
            let _ = dis.delattr("_inline_cache_entries");
        }

        let shadow = shadow_res.unwrap();
        assert!(!shadow.as_ref(py).is(sample));

        let guard = m
            .getattr(py, "get_guard_status")
            .unwrap()
            .call0(py)
            .unwrap();
        let guard_dict = guard.downcast::<PyDict>(py).unwrap();
        let mode: String = guard_dict.get_item("mode").unwrap().extract().unwrap();
        let reason: String = guard_dict.get_item("reason").unwrap().extract().unwrap();
        let rewrite_attempted: bool = guard_dict
            .get_item("rewrite_attempted")
            .unwrap()
            .extract()
            .unwrap();
        let rewrite_applied: bool = guard_dict
            .get_item("rewrite_applied")
            .unwrap()
            .extract()
            .unwrap();

        assert_eq!(mode, "fallback");
        assert_eq!(reason, "probe_instrumentation_failed");
        assert!(rewrite_attempted);
        assert!(!rewrite_applied);
    });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_vortex_fallback_reports_patched_stack_metadata_unavailable() {
    Python::with_gil(|py| {
        let m = iris::py::make_module(py).unwrap();

        let globals = PyDict::new(py);
        globals.set_item("iris", &m).unwrap();
        py.run(
            r#"
def sample_patched_stack_metadata_unavailable():
    a = 1
    b = 2
    return a + b
"#,
            Some(globals),
            None,
        )
        .unwrap();

        let sample = globals
            .get_item("sample_patched_stack_metadata_unavailable")
            .unwrap();
        let dis = py.import("dis").unwrap();
        let had_inline = dis.hasattr("_inline_cache_entries").unwrap();
        let original_inline = if had_inline {
            Some(dis.getattr("_inline_cache_entries").unwrap().to_object(py))
        } else {
            None
        };
        dis.setattr("_inline_cache_entries", PyList::new(py, vec![0u16; 256]))
            .unwrap();

        let os = py.import("os").unwrap();
        let environ = os.getattr("environ").unwrap();
        environ
            .set_item("IRIS_VORTEX_TEST_FORCE_PATCHED_STACK_METADATA_UNAVAILABLE", "1")
            .unwrap();

        let shadow_res = m
            .getattr(py, "transmute_function")
            .unwrap()
            .call1(py, (sample,));

        let _ = environ.del_item("IRIS_VORTEX_TEST_FORCE_PATCHED_STACK_METADATA_UNAVAILABLE");
        if had_inline {
            dis.setattr("_inline_cache_entries", original_inline.unwrap())
                .unwrap();
        } else {
            let _ = dis.delattr("_inline_cache_entries");
        }

        let shadow = shadow_res.unwrap();
        assert!(!shadow.as_ref(py).is(sample));

        let guard = m
            .getattr(py, "get_guard_status")
            .unwrap()
            .call0(py)
            .unwrap();
        let guard_dict = guard.downcast::<PyDict>(py).unwrap();
        let mode: String = guard_dict.get_item("mode").unwrap().extract().unwrap();
        let reason: String = guard_dict.get_item("reason").unwrap().extract().unwrap();
        let rewrite_attempted: bool = guard_dict
            .get_item("rewrite_attempted")
            .unwrap()
            .extract()
            .unwrap();
        let rewrite_applied: bool = guard_dict
            .get_item("rewrite_applied")
            .unwrap()
            .extract()
            .unwrap();

        assert_eq!(mode, "fallback");
        assert_eq!(reason, "patched_stack_metadata_unavailable");
        assert!(rewrite_attempted);
        assert!(!rewrite_applied);
    });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_vortex_fallback_reports_patched_exception_table_metadata_unavailable() {
    Python::with_gil(|py| {
        let m = iris::py::make_module(py).unwrap();

        let globals = PyDict::new(py);
        globals.set_item("iris", &m).unwrap();
        py.run(
            r#"
def sample_patched_exception_table_metadata_unavailable():
    a = 1
    b = 2
    return a + b
"#,
            Some(globals),
            None,
        )
        .unwrap();

        let sample = globals
            .get_item("sample_patched_exception_table_metadata_unavailable")
            .unwrap();
        let dis = py.import("dis").unwrap();
        let had_inline = dis.hasattr("_inline_cache_entries").unwrap();
        let original_inline = if had_inline {
            Some(dis.getattr("_inline_cache_entries").unwrap().to_object(py))
        } else {
            None
        };
        dis.setattr("_inline_cache_entries", PyList::new(py, vec![0u16; 256]))
            .unwrap();

        let os = py.import("os").unwrap();
        let environ = os.getattr("environ").unwrap();
        environ
            .set_item(
                "IRIS_VORTEX_TEST_FORCE_PATCHED_EXCEPTION_TABLE_METADATA_UNAVAILABLE",
                "1",
            )
            .unwrap();

        let shadow_res = m
            .getattr(py, "transmute_function")
            .unwrap()
            .call1(py, (sample,));

        let _ = environ.del_item("IRIS_VORTEX_TEST_FORCE_PATCHED_EXCEPTION_TABLE_METADATA_UNAVAILABLE");
        if had_inline {
            dis.setattr("_inline_cache_entries", original_inline.unwrap())
                .unwrap();
        } else {
            let _ = dis.delattr("_inline_cache_entries");
        }

        let shadow = shadow_res.unwrap();
        assert!(!shadow.as_ref(py).is(sample));

        let guard = m
            .getattr(py, "get_guard_status")
            .unwrap()
            .call0(py)
            .unwrap();
        let guard_dict = guard.downcast::<PyDict>(py).unwrap();
        let mode: String = guard_dict.get_item("mode").unwrap().extract().unwrap();
        let reason: String = guard_dict.get_item("reason").unwrap().extract().unwrap();
        let rewrite_attempted: bool = guard_dict
            .get_item("rewrite_attempted")
            .unwrap()
            .extract()
            .unwrap();
        let rewrite_applied: bool = guard_dict
            .get_item("rewrite_applied")
            .unwrap()
            .extract()
            .unwrap();

        assert_eq!(mode, "fallback");
        assert_eq!(reason, "patched_exception_table_metadata_unavailable");
        assert!(rewrite_attempted);
        assert!(!rewrite_applied);
    });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_vortex_fallback_reports_code_replace_failed() {
    Python::with_gil(|py| {
        let m = iris::py::make_module(py).unwrap();

        let globals = PyDict::new(py);
        globals.set_item("iris", &m).unwrap();
        py.run(
            r#"
def sample_code_replace_fail():
    a = 1
    b = 2
    return a + b
"#,
            Some(globals),
            None,
        )
        .unwrap();

        let sample = globals.get_item("sample_code_replace_fail").unwrap();
        let dis = py.import("dis").unwrap();
        let had_inline = dis.hasattr("_inline_cache_entries").unwrap();
        let original_inline = if had_inline {
            Some(dis.getattr("_inline_cache_entries").unwrap().to_object(py))
        } else {
            None
        };
        dis.setattr("_inline_cache_entries", PyList::new(py, vec![0u16; 256]))
            .unwrap();

        let os = py.import("os").unwrap();
        let environ = os.getattr("environ").unwrap();
        environ
            .set_item("IRIS_VORTEX_TEST_FORCE_CODE_REPLACE_FAILED", "1")
            .unwrap();

        let shadow_res = m
            .getattr(py, "transmute_function")
            .unwrap()
            .call1(py, (sample,));

        let _ = environ.del_item("IRIS_VORTEX_TEST_FORCE_CODE_REPLACE_FAILED");
        if had_inline {
            dis.setattr("_inline_cache_entries", original_inline.unwrap())
                .unwrap();
        } else {
            let _ = dis.delattr("_inline_cache_entries");
        }

        let shadow = shadow_res.unwrap();
        assert!(!shadow.as_ref(py).is(sample));

        let guard = m
            .getattr(py, "get_guard_status")
            .unwrap()
            .call0(py)
            .unwrap();
        let guard_dict = guard.downcast::<PyDict>(py).unwrap();
        let mode: String = guard_dict.get_item("mode").unwrap().extract().unwrap();
        let reason: String = guard_dict.get_item("reason").unwrap().extract().unwrap();
        let rewrite_attempted: bool = guard_dict
            .get_item("rewrite_attempted")
            .unwrap()
            .extract()
            .unwrap();
        let rewrite_applied: bool = guard_dict
            .get_item("rewrite_applied")
            .unwrap()
            .extract()
            .unwrap();

        assert_eq!(mode, "fallback");
        assert_eq!(reason, "code_replace_failed");
        assert!(rewrite_attempted);
        assert!(!rewrite_applied);
    });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_vortex_fallback_reports_types_module_unavailable() {
    Python::with_gil(|py| {
        let m = iris::py::make_module(py).unwrap();

        let globals = PyDict::new(py);
        globals.set_item("iris", &m).unwrap();
        py.run(
            r#"
def sample_types_unavailable():
    a = 3
    b = 4
    return a + b
"#,
            Some(globals),
            None,
        )
        .unwrap();

        let sample = globals.get_item("sample_types_unavailable").unwrap();
        let dis = py.import("dis").unwrap();
        let had_inline = dis.hasattr("_inline_cache_entries").unwrap();
        let original_inline = if had_inline {
            Some(dis.getattr("_inline_cache_entries").unwrap().to_object(py))
        } else {
            None
        };
        dis.setattr("_inline_cache_entries", PyList::new(py, vec![0u16; 256]))
            .unwrap();

        let os = py.import("os").unwrap();
        let environ = os.getattr("environ").unwrap();
        environ
            .set_item("IRIS_VORTEX_TEST_FORCE_TYPES_MODULE_UNAVAILABLE", "1")
            .unwrap();

        let shadow_res = m
            .getattr(py, "transmute_function")
            .unwrap()
            .call1(py, (sample,));

        let _ = environ.del_item("IRIS_VORTEX_TEST_FORCE_TYPES_MODULE_UNAVAILABLE");
        if had_inline {
            dis.setattr("_inline_cache_entries", original_inline.unwrap())
                .unwrap();
        } else {
            let _ = dis.delattr("_inline_cache_entries");
        }

        let shadow = shadow_res.unwrap();
        assert!(!shadow.as_ref(py).is(sample));

        let guard = m
            .getattr(py, "get_guard_status")
            .unwrap()
            .call0(py)
            .unwrap();
        let guard_dict = guard.downcast::<PyDict>(py).unwrap();
        let mode: String = guard_dict.get_item("mode").unwrap().extract().unwrap();
        let reason: String = guard_dict.get_item("reason").unwrap().extract().unwrap();
        let rewrite_attempted: bool = guard_dict
            .get_item("rewrite_attempted")
            .unwrap()
            .extract()
            .unwrap();
        let rewrite_applied: bool = guard_dict
            .get_item("rewrite_applied")
            .unwrap()
            .extract()
            .unwrap();

        assert_eq!(mode, "fallback");
        assert_eq!(reason, "types_module_unavailable");
        assert!(rewrite_attempted);
        assert!(!rewrite_applied);
    });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_vortex_fallback_reports_shadow_function_construction_failed() {
    Python::with_gil(|py| {
        let m = iris::py::make_module(py).unwrap();

        let globals = PyDict::new(py);
        globals.set_item("iris", &m).unwrap();
        py.run(
            r#"
def sample_shadow_construction_fail():
    a = 5
    b = 6
    return a + b
"#,
            Some(globals),
            None,
        )
        .unwrap();

        let sample = globals.get_item("sample_shadow_construction_fail").unwrap();
        let dis = py.import("dis").unwrap();
        let had_inline = dis.hasattr("_inline_cache_entries").unwrap();
        let original_inline = if had_inline {
            Some(dis.getattr("_inline_cache_entries").unwrap().to_object(py))
        } else {
            None
        };
        dis.setattr("_inline_cache_entries", PyList::new(py, vec![0u16; 256]))
            .unwrap();

        let os = py.import("os").unwrap();
        let environ = os.getattr("environ").unwrap();
        environ
            .set_item("IRIS_VORTEX_TEST_FORCE_SHADOW_CONSTRUCTION_FAILED", "1")
            .unwrap();

        let shadow_res = m
            .getattr(py, "transmute_function")
            .unwrap()
            .call1(py, (sample,));

        let _ = environ.del_item("IRIS_VORTEX_TEST_FORCE_SHADOW_CONSTRUCTION_FAILED");
        if had_inline {
            dis.setattr("_inline_cache_entries", original_inline.unwrap())
                .unwrap();
        } else {
            let _ = dis.delattr("_inline_cache_entries");
        }

        let shadow = shadow_res.unwrap();
        assert!(!shadow.as_ref(py).is(sample));

        let guard = m
            .getattr(py, "get_guard_status")
            .unwrap()
            .call0(py)
            .unwrap();
        let guard_dict = guard.downcast::<PyDict>(py).unwrap();
        let mode: String = guard_dict.get_item("mode").unwrap().extract().unwrap();
        let reason: String = guard_dict.get_item("reason").unwrap().extract().unwrap();
        let rewrite_attempted: bool = guard_dict
            .get_item("rewrite_attempted")
            .unwrap()
            .extract()
            .unwrap();
        let rewrite_applied: bool = guard_dict
            .get_item("rewrite_applied")
            .unwrap()
            .extract()
            .unwrap();

        assert_eq!(mode, "fallback");
        assert_eq!(reason, "shadow_function_construction_failed");
        assert!(rewrite_attempted);
        assert!(!rewrite_applied);
    });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_pyruntime_vortex_auto_policy_and_telemetry() {
    let rt_py = Python::with_gil(|py| {
        let module = iris::py::make_module(py).expect("make_module");
        let runtime_type = module
            .as_ref(py)
            .getattr("PyRuntime")
            .expect("no PyRuntime type");
        runtime_type.call0().expect("construct PyRuntime").into_py(py)
    });

    Python::with_gil(|py| {
        rt_py
            .as_ref(py)
            .call_method0("vortex_reset_auto_telemetry")
            .unwrap();

        let counts: (u64, u64) = rt_py
            .as_ref(py)
            .call_method0("vortex_get_auto_resolution_counts")
            .unwrap()
            .extract()
            .unwrap();
        let replay: u64 = rt_py
            .as_ref(py)
            .call_method0("vortex_get_auto_replay_count")
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(counts, (0, 0));
        assert_eq!(replay, 0);

        let ok: bool = rt_py
            .as_ref(py)
            .call_method1("vortex_set_auto_ghost_policy", ("PreferPrimary",))
            .unwrap()
            .extract()
            .unwrap();
        assert!(ok);

        let current: String = rt_py
            .as_ref(py)
            .call_method0("vortex_get_auto_ghost_policy")
            .unwrap()
            .extract::<Option<String>>()
            .unwrap()
            .unwrap();
        assert_eq!(current, "PreferPrimary");
    });

    let cb = Python::with_gil(|py| {
        py.eval("lambda _msg: None", None, None)
            .unwrap()
            .to_object(py)
    });

    let pid: u64 = Python::with_gil(|py| {
        rt_py
            .as_ref(py)
            .call_method1("spawn_py_handler", (cb, 8usize, false))
            .unwrap()
            .extract()
            .unwrap()
    });

    Python::with_gil(|py| {
        for _ in 0..1400u32 {
            let _ = rt_py.as_ref(py).call_method1(
                "send",
                (pid, pyo3::types::PyBytes::new(py, b"tick")),
            );
        }
    });

    timeout(Duration::from_secs(3), async {
        loop {
            let replay: u64 = Python::with_gil(|py| {
                rt_py
                    .as_ref(py)
                    .call_method0("vortex_get_auto_replay_count")
                    .unwrap()
                    .extract()
                    .unwrap()
            });

            if replay > 0 {
                break;
            }

            sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("auto replay telemetry should increase");

    Python::with_gil(|py| {
        let counts: (u64, u64) = rt_py
            .as_ref(py)
            .call_method0("vortex_get_auto_resolution_counts")
            .unwrap()
            .extract()
            .unwrap();
        assert!(counts.0 > 0);
        assert_eq!(counts.1, 0);

        let replay: u64 = rt_py
            .as_ref(py)
            .call_method0("vortex_get_auto_replay_count")
            .unwrap()
            .extract()
            .unwrap();
        assert!(replay > 0);

        rt_py
            .as_ref(py)
            .call_method0("vortex_reset_auto_telemetry")
            .unwrap();
        let counts_after: (u64, u64) = rt_py
            .as_ref(py)
            .call_method0("vortex_get_auto_resolution_counts")
            .unwrap()
            .extract()
            .unwrap();
        let replay_after: u64 = rt_py
            .as_ref(py)
            .call_method0("vortex_get_auto_replay_count")
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(counts_after, (0, 0));
        assert_eq!(replay_after, 0);
    });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_pyruntime_vortex_auto_policy_rejects_invalid_value() {
    let rt_py = Python::with_gil(|py| {
        let module = iris::py::make_module(py).expect("make_module");
        let runtime_type = module
            .as_ref(py)
            .getattr("PyRuntime")
            .expect("no PyRuntime type");
        runtime_type.call0().expect("construct PyRuntime").into_py(py)
    });

    Python::with_gil(|py| {
        let res = rt_py
            .as_ref(py)
            .call_method1("vortex_set_auto_ghost_policy", ("not-a-policy",));
        assert!(res.is_err());
        let err = res.unwrap_err();
        assert!(err.is_instance_of::<pyo3::exceptions::PyValueError>(py));
    });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_pyruntime_vortex_genetic_budgeting_toggle() {
    let rt_py = Python::with_gil(|py| {
        let module = iris::py::make_module(py).expect("make_module");
        let runtime_type = module
            .as_ref(py)
            .getattr("PyRuntime")
            .expect("no PyRuntime type");
        runtime_type.call0().expect("construct PyRuntime").into_py(py)
    });

    Python::with_gil(|py| {
        let initial: bool = rt_py
            .as_ref(py)
            .call_method0("vortex_get_genetic_budgeting")
            .unwrap()
            .extract()
            .unwrap();
        assert!(!initial);

        let ok_true: bool = rt_py
            .as_ref(py)
            .call_method1("vortex_set_genetic_budgeting", (true,))
            .unwrap()
            .extract()
            .unwrap();
        assert!(ok_true);

        let after_true: bool = rt_py
            .as_ref(py)
            .call_method0("vortex_get_genetic_budgeting")
            .unwrap()
            .extract()
            .unwrap();
        assert!(after_true);

        let ok_false: bool = rt_py
            .as_ref(py)
            .call_method1("vortex_set_genetic_budgeting", (false,))
            .unwrap()
            .extract()
            .unwrap();
        assert!(ok_false);

        let after_false: bool = rt_py
            .as_ref(py)
            .call_method0("vortex_get_genetic_budgeting")
            .unwrap()
            .extract()
            .unwrap();
        assert!(!after_false);
    });
}
