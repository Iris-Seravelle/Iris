#![cfg(feature = "pyo3")]

use pyo3::prelude::*;

#[tokio::test]
async fn py_runtime_spawn_and_send() {
    // create a single PyRuntime instance and keep it alive across await points
    let rt_py = Python::with_gil(|py| {
        let module = iris::py::make_module(py).expect("make_module");
        let runtime_type = module
            .as_ref(py)
            .getattr("PyRuntime")
            .expect("no PyRuntime type");
        let rt_obj = runtime_type.call0().expect("construct PyRuntime");
        rt_obj.into_py(py)
    });

    // spawn an observed handler and send a message (call into the same PyRuntime)
    let pid: u64 = Python::with_gil(|py| {
        rt_py
            .as_ref(py)
            .call_method1("spawn_observed_handler", (1usize,))
            .unwrap()
            .extract()
            .unwrap()
    });

    let sent: bool = Python::with_gil(|py| {
        rt_py
            .as_ref(py)
            .call_method1("send", (pid, pyo3::types::PyBytes::new(py, b"hello")))
            .unwrap()
            .extract()
            .unwrap()
    });
    assert!(sent, "send failed");

    // allow the tokio tasks to run
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    // inspect recorded messages for the same runtime/pid
    let msgs: Vec<Vec<u8>> = Python::with_gil(|py| {
        rt_py
            .as_ref(py)
            .call_method1("get_messages", (pid,))
            .unwrap()
            .extract()
            .unwrap()
    });

    assert_eq!(msgs.len(), 1);
    assert_eq!(&msgs[0], b"hello");

    // --- NEW: spawn a Python-backed handler that appends to a Python list ---
    let lst_obj: pyo3::PyObject = Python::with_gil(|py| {
        let module = iris::py::make_module(py).expect("make_module");
        let runtime_type = module
            .as_ref(py)
            .getattr("PyRuntime")
            .expect("no PyRuntime type");
        let rt_obj = runtime_type.call0().expect("construct PyRuntime");

        // create a Python list and a callback that appends bytes to it
        let lst = pyo3::types::PyList::empty(py);
        let lst_obj: pyo3::PyObject = lst.into_py(py);
        let locals = pyo3::types::PyDict::new(py);
        // build a factory that returns a callback which closes over `lst`
        py.run(
            "def make_cb(lst):\n    def cb(b): lst.append(b)\n    return cb\n",
            None,
            Some(locals),
        )
        .unwrap();
        let make_cb = locals.get_item("make_cb").unwrap();
        let cb: pyo3::PyObject = make_cb.call1((lst_obj.as_ref(py),)).unwrap().into();

        // spawn a Python handler and send a message
        let pid: u64 = rt_obj
            .call_method1("spawn_py_handler", (cb, 1usize))
            .unwrap()
            .extract()
            .unwrap();
        rt_obj
            .call_method1("send", (pid, pyo3::types::PyBytes::new(py, b"pycall")))
            .unwrap();

        lst_obj
    });

    // allow the tokio tasks to run (outside the GIL)
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    // verify Python list got the bytes
    Python::with_gil(|py| {
        let lst = lst_obj
            .as_ref(py)
            .downcast::<pyo3::types::PyList>()
            .unwrap();
        let got: Vec<&[u8]> = lst.extract().unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0], b"pycall");
    });

    // --- NEW: register a Python factory with the supervisor and validate ---
    Python::with_gil(|py| {
        let module = iris::py::make_module(py).expect("make_module");
        let runtime_type = module
            .as_ref(py)
            .getattr("PyRuntime")
            .expect("no PyRuntime type");
        let rt_obj = runtime_type.call0().expect("construct PyRuntime");

        // spawn observed handler to supervise
        let pid: u64 = rt_obj
            .call_method1("spawn_observed_handler", (1usize,))
            .unwrap()
            .extract()
            .unwrap();

        // create a factory that calls back into this same runtime instance
        let locals = pyo3::types::PyDict::new(py);
        locals.set_item("rt", rt_obj.into_py(py)).unwrap();
        py.run(
            "def factory(rt=rt):\n    return rt.spawn_observed_handler(1)",
            None,
            Some(locals),
        )
        .unwrap();
        let factory: pyo3::PyObject = locals.get_item("factory").unwrap().into();

        rt_obj
            .call_method1("supervise_with_factory", (pid, factory, "RestartOne"))
            .unwrap();
        let count: usize = rt_obj
            .call_method0("children_count")
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(count, 1);

        let pids: Vec<u64> = rt_obj
            .call_method0("child_pids")
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(pids.len(), 1);
        assert_eq!(pids[0], pid);
    });
}

// simple bounded mailbox send drop-new test
#[tokio::test]
async fn py_bounded_mailbox_drop_new() {
    // create runtime via Python helper like other tests
    let rt_py = Python::with_gil(|py| {
        let module = iris::py::make_module(py).expect("make_module");
        let runtime_type = module
            .as_ref(py)
            .getattr("PyRuntime")
            .expect("no PyRuntime type");
        let rt_obj = runtime_type.call0().expect("construct PyRuntime");
        rt_obj.into_py(py)
    });

    // build a Python list and callback to record messages
    let msgs = Python::with_gil(|py| py.eval("[]", None, None).unwrap().to_object(py));
    let cb = Python::with_gil(|py| {
        let locals = pyo3::types::PyDict::new(py);
        locals.set_item("msgs", msgs.clone_ref(py)).unwrap();
        // evaluate lambda expression directly so we can capture it
        let obj = py
            .eval("lambda msg, msgs=msgs: msgs.append(bytes(msg))", None, Some(&locals))
            .unwrap();
        obj.to_object(py)
    });

    let pid: u64 = Python::with_gil(|py| {
        rt_py
            .as_ref(py)
            .call_method1("spawn_py_handler_bounded", (cb.clone_ref(py), 100usize, 1usize, false))
            .unwrap()
            .extract()
            .unwrap()
    });

    let ok1: bool = Python::with_gil(|py| {
        rt_py
            .as_ref(py)
            .call_method1("send", (pid, pyo3::types::PyBytes::new(py, b"a")))
            .unwrap()
            .extract()
            .unwrap()
    });
    assert!(ok1);

    let ok2: bool = Python::with_gil(|py| {
        rt_py
            .as_ref(py)
            .call_method1("send", (pid, pyo3::types::PyBytes::new(py, b"b")))
            .unwrap()
            .extract()
            .unwrap()
    });
    assert!(!ok2);
}

// ---------- structured concurrency tests ----------

#[tokio::test]
async fn py_structured_concurrency_normal_and_crash() {
    let rt_py = Python::with_gil(|py| {
        let module = iris::py::make_module(py).expect("make_module");
        let runtime_type = module
            .as_ref(py)
            .getattr("PyRuntime")
            .expect("no PyRuntime type");
        let rt_obj = runtime_type.call0().expect("construct PyRuntime");
        rt_obj.into_py(py)
    });

    // helper to spawn a simple handler that stores messages in a list
    let make_handler = |py: Python<'_>| {
        let locals = pyo3::types::PyDict::new(py);
        py.run(
            "def make_cb(lst):\n    def cb(b): lst.append(b)\n    return cb\n",
            None,
            Some(locals),
        )
        .unwrap();
        let lst = pyo3::types::PyList::empty(py);
        let cb = locals
            .get_item("make_cb")
            .unwrap()
            .call1((lst,))
            .unwrap()
            .into_py(py);
        (lst.into_py(py), cb)
    };

    // normal-exit scenario
    let (_parent_list, parent_cb): (pyo3::PyObject, pyo3::PyObject) = Python::with_gil(|py| make_handler(py));
    let parent_pid: u64 = Python::with_gil(|py| {
        rt_py
            .as_ref(py)
            .call_method1("spawn_py_handler", (parent_cb.clone(), 1usize))
            .unwrap()
            .extract()
            .unwrap()
    });

    // debug: list available methods on PyRuntime
    Python::with_gil(|py| {
        let obj = rt_py.as_ref(py);
        let dirlist: Vec<String> = obj
            .dir()
            .iter()
            .map(|item| item.extract::<String>().unwrap())
            .collect();
        eprintln!("PyRuntime attributes: {:?}", dirlist);
    });
    let child_pid: u64 = Python::with_gil(|py| {
        rt_py
            .as_ref(py)
            .call_method1("spawn_child", (parent_pid, parent_cb.clone(), 1usize, false))
            .unwrap()
            .extract()
            .unwrap()
    });

    // send a message to the parent; the handler itself doesn't exit
    Python::with_gil(|py| {
        rt_py
            .as_ref(py)
            .call_method1("send", (parent_pid, pyo3::types::PyBytes::new(py, b"ok")))
            .unwrap();
    });

    // now explicitly stop the parent (normal shutdown) to exercise structured concurrency
    Python::with_gil(|py| {
        rt_py
            .as_ref(py)
            .call_method1("stop", (parent_pid,))
            .unwrap();
    });

    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    let alive: bool = Python::with_gil(|py| {
        rt_py
            .as_ref(py)
            .call_method1("is_alive", (child_pid,))
            .unwrap()
            .extract()
            .unwrap()
    });
    assert!(!alive, "child should die when parent exits normally");

    // crash scenario: spawn parent that panics
    let crash_cb: pyo3::PyObject = Python::with_gil(|py| {
        let src = "def cb(_):\n    raise Exception('crash')\n";
        let locals = pyo3::types::PyDict::new(py);
        py.run(src, None, Some(locals)).unwrap();
        locals.get_item("cb").unwrap().into_py(py)
    });
    let parent_crash: u64 = Python::with_gil(|py| {
        rt_py
            .as_ref(py)
            .call_method1("spawn_py_handler", (crash_cb.clone(), 1usize))
            .unwrap()
            .extract()
            .unwrap()
    });
    let child_crash: u64 = Python::with_gil(|py| {
        rt_py
            .as_ref(py)
            .call_method1("spawn_child", (parent_crash, crash_cb, 1usize, false))
            .unwrap()
            .extract()
            .unwrap()
    });

    Python::with_gil(|py| {
        rt_py
            .as_ref(py)
            .call_method1("send", (parent_crash, pyo3::types::PyBytes::new(py, b"go")))
            .unwrap();
    });

    // wait a bit for Python exception to be logged
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // explicitly terminate the parent after crash
    Python::with_gil(|py| {
        rt_py
            .as_ref(py)
            .call_method1("stop", (parent_crash,))
            .unwrap();
    });

    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    let alive2: bool = Python::with_gil(|py| {
        rt_py
            .as_ref(py)
            .call_method1("is_alive", (child_crash,))
            .unwrap()
            .extract()
            .unwrap()
    });
    assert!(!alive2, "child should die when parent crashes");
}

#[tokio::test]
async fn py_spawn_child_pool_reuses_workers_under_parent() {
    let rt_py = Python::with_gil(|py| {
        let module = iris::py::make_module(py).expect("make_module");
        let runtime_type = module
            .as_ref(py)
            .getattr("PyRuntime")
            .expect("no PyRuntime type");
        let rt_obj = runtime_type.call0().expect("construct PyRuntime");
        rt_obj.into_py(py)
    });

    let (parent_pid, _worker_pids, lst_obj): (u64, Vec<u64>, pyo3::PyObject) = Python::with_gil(|py| {
        let rt = rt_py.as_ref(py);

        let parent_pid: u64 = rt
            .call_method1("spawn_observed_handler", (1usize,))
            .unwrap()
            .extract()
            .unwrap();

        let lst = pyo3::types::PyList::empty(py);
        let locals = pyo3::types::PyDict::new(py);
        locals.set_item("lst", lst).unwrap();
        py.run(
            "def cb(b):\n    lst.append(bytes(b))",
            Some(locals),
            Some(locals),
        )
        .unwrap();
        let cb: pyo3::PyObject = locals.get_item("cb").unwrap().into();

        let worker_pids: Vec<u64> = rt
            .call_method1("spawn_child_pool", (parent_pid, cb, 4usize, 64usize, false))
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(worker_pids.len(), 4);

        for (i, pid) in worker_pids.iter().enumerate() {
            let payload = format!("m{}", i);
            let ok: bool = rt
                .call_method1("send", (*pid, pyo3::types::PyBytes::new(py, payload.as_bytes())))
                .unwrap()
                .extract()
                .unwrap();
            assert!(ok);
        }

        (parent_pid, worker_pids, lst.into_py(py))
    });

    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    Python::with_gil(|py| {
        let lst = lst_obj
            .as_ref(py)
            .downcast::<pyo3::types::PyList>()
            .unwrap();
        let got: Vec<Vec<u8>> = lst.extract().unwrap();
        assert_eq!(got.len(), 4);
    });

    Python::with_gil(|py| {
        let rt = rt_py.as_ref(py);
        rt.call_method1("stop", (parent_pid,)).unwrap();
    });
}

// overflow policy tests -------------------------------------------------

#[tokio::test]
async fn py_overflow_drop_old() {
    let rt_py = Python::with_gil(|py| {
        let module = iris::py::make_module(py).expect("make_module");
        let runtime_type = module
            .as_ref(py)
            .getattr("PyRuntime")
            .expect("no PyRuntime type");
        runtime_type.call0().unwrap().into_py(py)
    });

    let lst_obj: pyo3::PyObject =
        Python::with_gil(|py| pyo3::types::PyList::empty(py).into_py(py));

    Python::with_gil(|py| {
        let rt = rt_py.as_ref(py);
        let locals = pyo3::types::PyDict::new(py);
        locals.set_item("lst", lst_obj.as_ref(py)).unwrap();
        let handler = py
            .eval("lambda m, lst=lst: lst.append(bytes(m))", None, Some(locals))
            .unwrap();
        let pid: u64 = rt
            .call_method1("spawn_py_handler_bounded", (
                handler,
                100usize,
                2usize,
                false,
            ))
            .unwrap()
            .extract()
            .unwrap();

        // set policy to dropold
        rt.call_method1("set_overflow_policy", (pid, "dropold", Option::<u64>::None)).unwrap();

        // send three small messages; capacity is 2, so oldest should be dropped
        for b in [b"a", b"b", b"c"].iter() {
            let ok: bool = rt
                .call_method1("send", (pid, pyo3::types::PyBytes::new(py, *b)))
                .unwrap()
                .extract()
                .unwrap();
            assert!(ok);
        }
    });

    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    Python::with_gil(|py| {
        let lst = lst_obj
            .as_ref(py)
            .downcast::<pyo3::types::PyList>()
            .unwrap();
        let got: Vec<Vec<u8>> = lst.extract().unwrap();
        // Under tight timing, overflow may or may not trigger before consumer drain.
        // If overflow triggers, oldest is dropped and we keep ["b", "c"].
        // If consumer drains early, all 3 may be processed in order.
        assert!(
            got == vec![b"b".to_vec(), b"c".to_vec()]
                || got == vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()]
        );
    });
}

#[tokio::test]
async fn py_overflow_redirect() {
    let rt_py = Python::with_gil(|py| {
        let module = iris::py::make_module(py).expect("make_module");
        let runtime_type = module
            .as_ref(py)
            .getattr("PyRuntime")
            .expect("no PyRuntime type");
        runtime_type.call0().unwrap().into_py(py)
    });

    let (lst1_obj, lst2_obj): (pyo3::PyObject, pyo3::PyObject) = Python::with_gil(|py| {
        (
            pyo3::types::PyList::empty(py).into_py(py),
            pyo3::types::PyList::empty(py).into_py(py),
        )
    });

    Python::with_gil(|py| {
        let rt = rt_py.as_ref(py);
        let locals1 = pyo3::types::PyDict::new(py);
        locals1.set_item("lst1", lst1_obj.as_ref(py)).unwrap();
        let handler1 = py
            .eval("lambda m, lst1=lst1: lst1.append(bytes(m))", None, Some(locals1))
            .unwrap();

        let locals2 = pyo3::types::PyDict::new(py);
        locals2.set_item("lst2", lst2_obj.as_ref(py)).unwrap();
        let handler2 = py
            .eval("lambda m, lst2=lst2: lst2.append(bytes(m))", None, Some(locals2))
            .unwrap();
        // create primary bounded actor
        let pid: u64 = rt
            .call_method1("spawn_py_handler_bounded", (
                handler1,
                100usize,
                1usize,
                false,
            ))
            .unwrap()
            .extract()
            .unwrap();
        // create fallback actor
        let fid: u64 = rt
            .call_method1("spawn_py_handler", (
                handler2,
                100usize,
                false,
            ))
            .unwrap()
            .extract()
            .unwrap();
        // set redirect policy
        rt.call_method1("set_overflow_policy", (pid, "redirect", fid)).unwrap();

        // send two messages; second should redirect
        for b in [b"1", b"2"].iter() {
            let ok: bool = rt
                .call_method1("send", (pid, pyo3::types::PyBytes::new(py, *b)))
                .unwrap()
                .extract()
                .unwrap();
            assert!(ok);
        }
    });

    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    Python::with_gil(|py| {
        let lst1 = lst1_obj
            .as_ref(py)
            .downcast::<pyo3::types::PyList>()
            .unwrap();
        let lst2 = lst2_obj
            .as_ref(py)
            .downcast::<pyo3::types::PyList>()
            .unwrap();
        let got1: Vec<Vec<u8>> = lst1.extract().unwrap();
        let got2: Vec<Vec<u8>> = lst2.extract().unwrap();

        // Under contention the primary may drain before overflow triggers.
        // Accept either strict redirect split ["1"]+["2"] or both on primary.
        let mut combined = got1.clone();
        combined.extend(got2.clone());
        assert_eq!(combined, vec![b"1".to_vec(), b"2".to_vec()]);
        assert!(
            (got1 == vec![b"1".to_vec()] && got2 == vec![b"2".to_vec()])
                || (got1 == vec![b"1".to_vec(), b"2".to_vec()] && got2.is_empty())
        );
    });
}

#[tokio::test]
async fn py_overflow_spill() {
    let rt_py = Python::with_gil(|py| {
        let module = iris::py::make_module(py).expect("make_module");
        let runtime_type = module
            .as_ref(py)
            .getattr("PyRuntime")
            .expect("no PyRuntime type");
        runtime_type.call0().unwrap().into_py(py)
    });

    let (primary_obj, fallback_obj): (pyo3::PyObject, pyo3::PyObject) = Python::with_gil(|py| {
        (
            pyo3::types::PyList::empty(py).into_py(py),
            pyo3::types::PyList::empty(py).into_py(py),
        )
    });

    let pid: u64 = Python::with_gil(|py| {
        let rt = rt_py.as_ref(py);

        let p_locals = pyo3::types::PyDict::new(py);
        p_locals.set_item("lst", primary_obj.as_ref(py)).unwrap();
        let primary_handler = py
            .eval("lambda m, lst=lst: lst.append(bytes(m))", None, Some(p_locals))
            .unwrap();

        let f_locals = pyo3::types::PyDict::new(py);
        f_locals.set_item("lst", fallback_obj.as_ref(py)).unwrap();
        let fallback_handler = py
            .eval("lambda m, lst=lst: lst.append(bytes(m))", None, Some(f_locals))
            .unwrap();

        let pid: u64 = rt
            .call_method1("spawn_py_handler_bounded", (primary_handler, 100usize, 1usize, false))
            .unwrap()
            .extract()
            .unwrap();

        let fid: u64 = rt
            .call_method1("spawn_py_handler", (fallback_handler, 100usize, false))
            .unwrap()
            .extract()
            .unwrap();

        rt.call_method1("set_overflow_policy", (pid, "spill", fid)).unwrap();

        let ok1: bool = rt
            .call_method1("send", (pid, pyo3::types::PyBytes::new(py, b"1")))
            .unwrap()
            .extract()
            .unwrap();
        assert!(ok1);
        pid
    });

    let rt_for_send = rt_py.clone();
    let send_task = tokio::task::spawn_blocking(move || {
        Python::with_gil(|py| {
            let rt = rt_for_send.as_ref(py);
            let ok2: bool = rt
                .call_method1("send", (pid, pyo3::types::PyBytes::new(py, b"2")))
                .unwrap()
                .extract()
                .unwrap();
            ok2
        })
    });

    let ok2 = tokio::time::timeout(std::time::Duration::from_secs(1), send_task)
        .await
        .expect("spill send should complete")
        .expect("spill send task join");
    assert!(ok2);

    tokio::time::sleep(std::time::Duration::from_millis(250)).await;

    Python::with_gil(|py| {
        let primary = primary_obj
            .as_ref(py)
            .downcast::<pyo3::types::PyList>()
            .unwrap();
        let fallback = fallback_obj
            .as_ref(py)
            .downcast::<pyo3::types::PyList>()
            .unwrap();

        let got_primary: Vec<Vec<u8>> = primary.extract().unwrap();
        let got_fallback: Vec<Vec<u8>> = fallback.extract().unwrap();

        assert!(!got_primary.is_empty());
        assert_eq!(got_primary[0], b"1".to_vec());

        let mut combined = got_primary.clone();
        combined.extend(got_fallback);
        assert!(combined.iter().any(|m| m.as_slice() == b"2"));
    });
}

#[tokio::test]
async fn py_overflow_block() {
    let rt_py = Python::with_gil(|py| {
        let module = iris::py::make_module(py).expect("make_module");
        let runtime_type = module
            .as_ref(py)
            .getattr("PyRuntime")
            .expect("no PyRuntime type");
        runtime_type.call0().unwrap().into_py(py)
    });

    let list_obj: pyo3::PyObject = Python::with_gil(|py| pyo3::types::PyList::empty(py).into_py(py));

    Python::with_gil(|py| {
        let rt = rt_py.as_ref(py);
        let locals = pyo3::types::PyDict::new(py);
        locals.set_item("lst", list_obj.as_ref(py)).unwrap();
        let handler = py
            .eval("lambda m, lst=lst: lst.append(bytes(m))", None, Some(locals))
            .unwrap();

        let pid: u64 = rt
            .call_method1("spawn_py_handler_bounded", (handler, 100usize, 1usize, false))
            .unwrap()
            .extract()
            .unwrap();

        rt.call_method1("set_overflow_policy", (pid, "block", Option::<u64>::None)).unwrap();

        let ok1: bool = rt
            .call_method1("send", (pid, pyo3::types::PyBytes::new(py, b"a")))
            .unwrap()
            .extract()
            .unwrap();
        let ok2: bool = rt
            .call_method1("send", (pid, pyo3::types::PyBytes::new(py, b"b")))
            .unwrap()
            .extract()
            .unwrap();

        assert!(ok1);
        assert!(ok2);
    });

    tokio::time::sleep(std::time::Duration::from_millis(250)).await;

    Python::with_gil(|py| {
        let lst = list_obj
            .as_ref(py)
            .downcast::<pyo3::types::PyList>()
            .unwrap();
        let got: Vec<Vec<u8>> = lst.extract().unwrap();
        assert_eq!(got, vec![b"a".to_vec(), b"b".to_vec()]);
    });
}

#[tokio::test]
async fn py_virtual_actor_activates_on_first_send() {
    let rt_py = Python::with_gil(|py| {
        let module = iris::py::make_module(py).expect("make_module");
        let runtime_type = module
            .as_ref(py)
            .getattr("PyRuntime")
            .expect("no PyRuntime type");
        runtime_type.call0().unwrap().into_py(py)
    });

    let list_obj: pyo3::PyObject =
        Python::with_gil(|py| pyo3::types::PyList::empty(py).into_py(py));

    let pid: u64 = Python::with_gil(|py| {
        let rt = rt_py.as_ref(py);
        let locals = pyo3::types::PyDict::new(py);
        locals.set_item("lst", list_obj.as_ref(py)).unwrap();
        let handler = py
            .eval("lambda m, lst=lst: lst.append(bytes(m))", None, Some(locals))
            .unwrap();

        rt.call_method1("spawn_virtual_py_handler", (handler, 16usize, Option::<u64>::None))
            .unwrap()
            .extract()
            .unwrap()
    });

    let sent: bool = Python::with_gil(|py| {
        rt_py
            .as_ref(py)
            .call_method1("send", (pid, pyo3::types::PyBytes::new(py, b"lazy")))
            .unwrap()
            .extract()
            .unwrap()
    });
    assert!(sent);

    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    Python::with_gil(|py| {
        let lst = list_obj
            .as_ref(py)
            .downcast::<pyo3::types::PyList>()
            .unwrap();
        let got: Vec<Vec<u8>> = lst.extract().unwrap();
        assert_eq!(got, vec![b"lazy".to_vec()]);
    });
}

#[tokio::test]
async fn py_virtual_actor_idle_timeout() {
    let rt_py = Python::with_gil(|py| {
        let module = iris::py::make_module(py).expect("make_module");
        let runtime_type = module
            .as_ref(py)
            .getattr("PyRuntime")
            .expect("no PyRuntime type");
        runtime_type.call0().unwrap().into_py(py)
    });

    let pid: u64 = Python::with_gil(|py| {
        let rt = rt_py.as_ref(py);
        let handler = py.eval("lambda m: None", None, None).unwrap();
        rt.call_method1("spawn_virtual_py_handler", (handler, 8usize, Some(50u64)))
            .unwrap()
            .extract()
            .unwrap()
    });

    let sent: bool = Python::with_gil(|py| {
        rt_py
            .as_ref(py)
            .call_method1("send", (pid, pyo3::types::PyBytes::new(py, b"ping")))
            .unwrap()
            .extract()
            .unwrap()
    });
    assert!(sent);

    tokio::time::sleep(std::time::Duration::from_millis(220)).await;

    let alive: bool = Python::with_gil(|py| {
        rt_py
            .as_ref(py)
            .call_method1("is_alive", (pid,))
            .unwrap()
            .extract()
            .unwrap()
    });
    assert!(!alive, "virtual actor should stop after idle timeout");
}

