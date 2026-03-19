#![cfg(all(feature = "pyo3", feature = "vortex"))]

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

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
