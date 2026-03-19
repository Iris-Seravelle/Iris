#![cfg(all(feature = "pyo3", feature = "vortex"))]

use pyo3::prelude::*;
use pyo3::types::PyDict;

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
