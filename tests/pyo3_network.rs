#![cfg(feature = "pyo3")]

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use std::time::Duration;

#[tokio::test]
async fn test_distributed_messaging() {
    let addr = "127.0.0.1:9999";

    // 1. Setup Node A (The Receiver)
    let (rt_a, pid_a, results): (PyObject, u64, PyObject) = Python::with_gil(|py| {
        let module = iris::py::make_module(py).expect("make_module");
        let rt_type = module.as_ref(py).getattr("PyRuntime").unwrap();
        let rt = rt_type.call0().unwrap();

        let results = PyList::empty(py);
        let locals = PyDict::new(py);
        locals.set_item("results", results).unwrap();

        rt.call_method1("listen", (addr,)).unwrap();

        py.run(
            r#"
def remote_handler(msg, results=results):
    results.append(msg.decode())
"#,
            None,
            Some(locals),
        )
        .unwrap();

        let handler = locals.get_item("remote_handler").unwrap();
        let pid: u64 = rt
            .call_method1("spawn_py_handler", (handler, 10usize))
            .unwrap()
            .extract()
            .unwrap();

        (rt.into_py(py), pid, results.into_py(py))
    });

    tokio::time::sleep(Duration::from_millis(100)).await;

    Python::with_gil(|py| {
        let module = iris::py::make_module(py).expect("make_module");
        let rt_type = module.as_ref(py).getattr("PyRuntime").unwrap();
        let rt_b = rt_type.call0().unwrap();

        let payload = PyBytes::new(py, b"Hello from Node B");
        rt_b.call_method1("send_remote", (addr, pid_a, payload))
            .unwrap();
    });

    let mut success = false;
    for _ in 0..20 {
        tokio::time::sleep(Duration::from_millis(50)).await;
        success = Python::with_gil(|py| {
            let res: Vec<String> = results.extract(py).unwrap();
            res.contains(&"Hello from Node B".to_string())
        });
        if success {
            break;
        }
    }

    assert!(success, "Remote message never arrived at Node A");

    Python::with_gil(|py| {
        rt_a.call_method1(py, "stop", (pid_a,)).unwrap();
    });
}

#[test]
fn test_remote_name_discovery() {
    let addr = "127.0.0.1:9095";

    let (rt_a, pid_a, results): (PyObject, u64, PyObject) = Python::with_gil(|py| {
        let module = iris::py::make_module(py).unwrap();
        let rt = module
            .as_ref(py)
            .getattr("PyRuntime")
            .unwrap()
            .call0()
            .unwrap();

        rt.call_method1("listen", (addr,)).unwrap();

        let results = PyList::empty(py);
        let locals = PyDict::new(py);
        locals.set_item("results", results).unwrap();

        py.run(
            r#"
def auth_handler(msg, results=results):
    results.append(f"Auth:{msg.decode()}")
"#,
            None,
            Some(locals),
        )
        .unwrap();

        let handler = locals.get_item("auth_handler").unwrap();
        let pid: u64 = rt
            .call_method1("spawn_py_handler", (handler, 10usize))
            .unwrap()
            .extract()
            .unwrap();

        rt.call_method1("register", ("auth-service", pid)).unwrap();
        (rt.into_py(py), pid, results.into_py(py))
    });

    std::thread::sleep(Duration::from_millis(150));

    Python::with_gil(|py| {
        let module = iris::py::make_module(py).unwrap();
        let rt_b = module
            .as_ref(py)
            .getattr("PyRuntime")
            .unwrap()
            .call0()
            .unwrap();

        let resolved_pid: Option<u64> = rt_b
            .call_method1("resolve_remote", (addr, "auth-service"))
            .unwrap()
            .extract()
            .unwrap();

        assert!(
            resolved_pid.is_some(),
            "Node B failed to resolve Node A's service name"
        );
        let proxy = resolved_pid.unwrap();

        let payload = PyBytes::new(py, b"login_request");
        rt_b.call_method1("send", (proxy, payload)).unwrap();
    });

    let mut success = false;
    for _ in 0..15 {
        std::thread::sleep(Duration::from_millis(100));
        success = Python::with_gil(|py| {
            let res: Vec<String> = results.extract(py).unwrap();
            res.contains(&"Auth:login_request".to_string())
        });
        if success {
            break;
        }
    }

    assert!(success, "Remote message via discovered name never arrived");

    Python::with_gil(|py| {
        rt_a.call_method1(py, "stop", (pid_a,)).unwrap();
    });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_remote_monitoring_is_node_level_not_pid_level() {
    let addr = "127.0.0.1:9998";

    let (rt_a, pid_a) = Python::with_gil(|py| {
        let module = iris::py::make_module(py).unwrap();
        let rt = module
            .as_ref(py)
            .getattr("PyRuntime")
            .unwrap()
            .call0()
            .unwrap();

        rt.call_method1("listen", (addr,)).unwrap();

        py.run("def target(msg): pass", None, None).unwrap();
        let handler = py.eval("target", None, None).unwrap();

        let pid: u64 = rt
            .call_method1("spawn_py_handler", (handler, 10usize))
            .unwrap()
            .extract()
            .unwrap();

        rt.call_method1("register", ("target", pid)).unwrap();
        (rt.into_py(py), pid)
    });

    let mut proxy_pid: u64 = 0;
    let rt_b = Python::with_gil(|py| {
        let module = iris::py::make_module(py).unwrap();
        let rt = module
            .as_ref(py)
            .getattr("PyRuntime")
            .unwrap()
            .call0()
            .unwrap();

        let resolved: Option<u64> = rt
            .call_method1("resolve_remote", (addr, "target"))
            .unwrap()
            .extract()
            .unwrap();
        assert!(resolved.is_some());
        proxy_pid = resolved.unwrap();

        rt.call_method1("monitor_remote", (addr, proxy_pid))
            .unwrap();
        rt.into_py(py)
    });

    Python::with_gil(|py| {
        rt_a.call_method1(py, "stop", (pid_a,)).unwrap();
    });

    tokio::time::sleep(Duration::from_millis(1200)).await;

    Python::with_gil(|py| {
        let alive: bool = rt_b
            .call_method1(py, "is_alive", (proxy_pid,))
            .unwrap()
            .extract(py)
            .unwrap();
        assert!(
            alive,
            "proxy should remain alive while remote node is reachable"
        );
    });
}
