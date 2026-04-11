#![cfg(feature = "pyo3")]

use pyo3::prelude::*;
use pyo3::types::PyDict;

#[test]
fn bench_send_vs_send_many() {
    Python::with_gil(|py| {
        let module = iris::py::make_module(py).expect("make_module");
        let rt = module
            .as_ref(py)
            .getattr("PyRuntime")
            .expect("PyRuntime missing")
            .call0()
            .expect("construct PyRuntime failed");

        let locals = PyDict::new(py);
        locals.set_item("rt", rt).unwrap();
        locals
            .set_item("__builtins__", py.import("builtins").unwrap())
            .unwrap();

        py.run(
            r#"
import statistics
import time

TOTAL_MESSAGES = 100_000
BATCH_SIZE = 256
ROUNDS = 5
PAYLOAD = b"ping"
MAX_BENCH_SECONDS = 25.0
MISSING_PID = (1 << 63) - 1


def run_send():
    t0 = time.perf_counter()
    for _ in range(TOTAL_MESSAGES):
        # Missing PID isolates Python->Rust API dispatch overhead and avoids
        # queue drain time from dominating benchmark runtime.
        _ = rt.send(MISSING_PID, PAYLOAD)
    return time.perf_counter() - t0


def run_send_many():
    full = TOTAL_MESSAGES // BATCH_SIZE
    rem = TOTAL_MESSAGES % BATCH_SIZE
    batch = [PAYLOAD] * BATCH_SIZE

    t0 = time.perf_counter()
    for _ in range(full):
        _ = rt.send_many(MISSING_PID, batch)
    if rem:
        _ = rt.send_many(MISSING_PID, [PAYLOAD] * rem)
    return time.perf_counter() - t0


# Warmup
run_send()
run_send_many()

send_times = []
send_many_times = []
bench_start = time.perf_counter()
for _ in range(ROUNDS):
    if time.perf_counter() - bench_start > MAX_BENCH_SECONDS:
        break
    send_times.append(run_send())
    send_many_times.append(run_send_many())

if not send_times or not send_many_times:
    raise RuntimeError("benchmark did not complete any rounds")

send_med = statistics.median(send_times)
many_med = statistics.median(send_many_times)
speedup = send_med / many_med if many_med > 0 else float("inf")

print("[bench] send median:      %.6fs (%.0f msg/s)" % (send_med, TOTAL_MESSAGES / send_med))
print("[bench] send_many median: %.6fs (%.0f msg/s)" % (many_med, TOTAL_MESSAGES / many_med))
print("[bench] speedup: %.2fx" % speedup)

bench_send_med = send_med
bench_many_med = many_med
bench_speedup = speedup
"#,
            Some(locals),
            Some(locals),
        )
        .expect("python benchmark run failed");

        let send_med: f64 = locals
            .get_item("bench_send_med")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let many_med: f64 = locals
            .get_item("bench_many_med")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let speedup: f64 = locals
            .get_item("bench_speedup")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        assert!(send_med.is_finite() && send_med > 0.0);
        assert!(many_med.is_finite() && many_med > 0.0);
        assert!(speedup.is_finite() && speedup > 0.0);
    });
}
