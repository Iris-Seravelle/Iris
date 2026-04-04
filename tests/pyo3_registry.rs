#![cfg(feature = "pyo3")]

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use std::time::Duration;

#[tokio::test]
async fn test_name_registration_and_resolution() {
    Python::with_gil(|py| {
        let module = iris::py::make_module(py).expect("make_module");
        let rt_type = module.as_ref(py).getattr("PyRuntime").unwrap();
        let rt = rt_type.call0().unwrap();

        let results = PyList::empty(py);
        let locals = PyDict::new(py);
        locals.set_item("results", results).unwrap();

        py.run(
            r#"def named_handler(msg, results=results):
    results.append(msg.decode())
"#,
            None,
            Some(locals),
        )
        .unwrap();

        let handler = locals.get_item("named_handler").unwrap();

        let pid: u64 = rt
            .call_method1("spawn_py_handler", (handler, 10usize))
            .unwrap()
            .extract()
            .unwrap();

        rt.call_method1("register", ("my_service", pid)).unwrap();

        let resolved_pid: Option<u64> = rt
            .call_method1("resolve", ("my_service",))
            .unwrap()
            .extract()
            .unwrap();

        assert_eq!(resolved_pid, Some(pid));

        let msg = PyBytes::new(py, b"hello registry");
        rt.call_method1("send", (resolved_pid.unwrap(), msg))
            .unwrap();
    });

    tokio::time::sleep(Duration::from_millis(50)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_registry_lifecycle() {
    Python::with_gil(|py| {
        let module = iris::py::make_module(py).unwrap();
        let rt = module.getattr(py, "PyRuntime").unwrap().call0(py).unwrap();
        let locals = PyDict::new(py);
        locals.set_item("rt", rt).unwrap();
        locals
            .set_item("__builtins__", py.import("builtins").unwrap())
            .unwrap();

        py.run(
            r#"import time

def dummy_service(mailbox):
    mailbox.recv()

pid = rt.spawn_with_mailbox(dummy_service, 100)
assert pid > 0

rt.register("my_service", pid)

found_pid = rt.resolve("my_service")
assert found_pid == pid, f"Resolve failed: expected {pid}, got {found_pid}"

alias_pid = rt.whereis("my_service")
assert alias_pid == pid, f"Whereis failed: expected {pid}, got {alias_pid}"

rt.register("my_service", pid)

rt.unregister("my_service")

gone = rt.resolve("my_service")
assert gone is None, f"Expected None after unregister, got {gone}"

rt.stop(pid)
"#,
            Some(locals),
            Some(locals),
        )
        .unwrap();
    });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_communication_via_name() {
    Python::with_gil(|py| {
        let module = iris::py::make_module(py).unwrap();
        let rt = module.getattr(py, "PyRuntime").unwrap().call0(py).unwrap();
        let locals = PyDict::new(py);
        locals.set_item("rt", rt).unwrap();
        locals
            .set_item("__builtins__", py.import("builtins").unwrap())
            .unwrap();

        py.run(
            r#"import time

def logger_actor(mailbox):
    msg = mailbox.recv(timeout=1.0)
    global log_result
    log_result = msg

logger_pid = rt.spawn_with_mailbox(logger_actor, 100)

rt.register("central_logger", logger_pid)

target = rt.resolve("central_logger")
if target:
    rt.send(target, b"log_this_data")
else:
    raise Exception("Could not find central_logger")

time.sleep(0.2)
"#,
            Some(locals),
            Some(locals),
        )
        .unwrap();

        let result_any = locals
            .get_item("log_result")
            .unwrap()
            .unwrap();
        let result: Vec<u8> = result_any.extract().unwrap();

        assert_eq!(result, b"log_this_data");
    });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_register_non_existent_returns_none() {
    Python::with_gil(|py| {
        let module = iris::py::make_module(py).unwrap();
        let rt = module.getattr(py, "PyRuntime").unwrap().call0(py).unwrap();
        let locals = PyDict::new(py);
        locals.set_item("rt", rt).unwrap();

        py.run(
            r#"assert rt.resolve("ghost_service") is None
assert rt.whereis("ghost_service") is None

rt.unregister("ghost_service")
"#,
            Some(locals),
            Some(locals),
        )
        .unwrap();
    });
}
