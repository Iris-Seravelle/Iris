# JIT & Compute Offload

This document describes Iris JIT behavior, supported language surface, intrinsics, and runtime controls.

## Current optimization model (2026-03)

Iris JIT acceleration now has four layers that can stack:

1. Cranelift-native expression JIT compilation.
2. SIMD capability planning (`aarch64/arm`, `x86/x86_64`, `wasm32`, scalar fallback).
3. SIMD-aware loop unrolling in hot runtime vector-buffer execution paths.
4. Direct lowered-kernel execution for common elementwise math kernels (including trig), bypassing per-element JIT dispatch overhead.

## Decorator API

```python
@iris.offload(strategy="jit", return_type="float")
def kernel(...):
    ...
```

- `strategy="jit"`: compile eligible code paths to native code (Cranelift).
- `strategy="actor"`: execute through Rust actor offload workers.
- `return_type`: `"float"` (default), `"int"`, or `"bool"`.

## Supported JIT Surface

### Expressions and control

- Arithmetic: `+ - * / % **`.
- Unary ops and parentheses.
- Comparisons including chained comparisons.
- Boolean logic (`and`, `or`, `not`).
- Ternary expressions (`a if cond else b`).
- Builtin-style casts/aliases: `float(x)`, `int(x)`, `round(x)`.

### Math intrinsics

Supported math calls include:

- `sin`, `cos`, `tan`
- `sinh`, `cosh`, `tanh`
- `exp`, `log`, `sqrt`
- `abs`, `min`, `max`, `pow`

### Lowered kernel coverage (fast vector loop execution)

When expressions match simple elementwise forms, Iris may run a lowered vector loop directly:

- Unary kernels:
  - identity (`x`), negation (`-x`), `abs(x)`
  - `sin(x)`, `cos(x)`, `tan(x)`
  - `exp(x)`, `log(x)` / `ln(x)`, `sqrt(x)`
- Binary kernels:
  - `a + b`, `a - b`, `a * b`, `a / b`

These lowered paths are used on vector-buffer call paths (profiled and generic), and support both map-style outputs and reductions (`sum`, `any`, `all`).

> `math.<fn>` style usage is normalized by the frontend when extractable to JIT expressions.

### Reductions and generators

- `sum(...)`, `any(...)`, `all(...)` over `range(...)` and container generators.
- Predicated reductions in generator bodies.
- Positive and negative `range` step handling (including dynamic step expressions).

### While-style reduction intrinsics

- `sum_while(...)`
- `any_while(...)`
- `all_while(...)`

### Loop-control intrinsics

Supported forms (aliases included):

- Break family:
  - `break_if`, `break_when`, `loop_break_if`, `loop_break_when`
  - `break_unless`, `loop_break_unless`
- Continue family:
  - `continue_if`, `continue_when`, `loop_continue_if`, `loop_continue_when`
  - `continue_unless`, `loop_continue_unless`
- NaN control:
  - `break_on_nan`, `loop_break_on_nan`
  - `continue_on_nan`, `loop_continue_on_nan`

## Type and data support

### Scalars

- Inputs: Python `float`, `int`, `bool`.
- Internal ABI: lowered through native `f64` argument path.
- Output conversion follows `return_type` (`float`/`int`/`bool`).

### Vectorized buffers

Supported buffer element types include:

- `f64`, `f32`
- signed integers (`i64`, `i32`, `i16`, `i8`)
- unsigned integers (`u64`, `u32`, `u16`, `u8`)
- `bool`

## SIMD planner and loop optimization

### Architecture-aware backend selection

Iris selects a SIMD backend from host capabilities at runtime:

- ARM/aarch64: prefers `SVE/SVE2`, falls back to `NEON`, then scalar.
- x86/x86_64: prefers `AVX2`, then `AVX`, then `SSE2`, then scalar.
- wasm32: uses `simd128` when available, else scalar.

### Loop unrolling in practical paths

Unrolling is applied to hot loops in:

- profiled vector-buffer execution,
- generic all-buffer vector execution,
- trailing-count vectorized mode,
- indexable sequence fallback loops.

Unroll factor is derived from SIMD lane width (with scalar-safe tail handling), improving throughput even before explicit vector intrinsics are emitted for every operation.

### Horizontal-style reduction combine

For lowered reductions, Iris uses lane-wise partial accumulation (horizontal-style combine) for:

- `sum` (lane accumulators reduced at the end),
- `any` (lane OR style early success),
- `all` (lane AND style early failure).

This is primarily beneficial for reduction workloads; pure map kernels do not need horizontal combine.

## Quantum speculation behavior

When enabled, Iris may compile multiple variants and select adaptively using runtime telemetry.

- Selection is runtime-driven from per-variant execution stats.
- Warm-seeded startup may intentionally compile a single variant first, then rearm to multi-variant when observed latency degrades.
- If speculation is gated by runtime controls, Iris falls back to single-variant compile.
- Failures or mismatches fall back safely to Python execution path.

## Runtime Controls

### SIMD controls

- Enable/disable SIMD planning:
  - Env: `IRIS_JIT_SIMD`
  - Values: truthy (`1/true/yes/on/...`) to enable, falsy to force scalar planning.

- SIMD math mode for lowered unary trig kernels:
  - Env: `IRIS_JIT_SIMD_MATH`
  - `accurate` (default): standard libm-backed behavior.
  - `fast` / `approx` / `poly`: fast approximation mode for lower latency with possible precision tradeoffs.

### Logging

- Enable: `IRIS_JIT_LOG=1` or `iris.jit.set_jit_logging(...)`
- Read status: `iris.jit.get_jit_logging()`

When logging is enabled, SIMD planner details are emitted, for example:

- `[Iris][jit][simd] backend=ArmNeon lane_bytes=16 auto_vectorize=true`

### Quantum enable/disable

- Env: `IRIS_JIT_QUANTUM=1`
- API: `iris.jit.set_quantum_speculation(...)`, `iris.jit.get_quantum_speculation()`

### Quantum thresholds

- Speculation threshold (ns):
  - Env: `IRIS_JIT_QUANTUM_SPECULATION_NS`
  - API: `iris.jit.set_quantum_speculation_threshold(...)`, `iris.jit.get_quantum_speculation_threshold()`
- Quantum log threshold (ns):
  - Env: `IRIS_JIT_QUANTUM_LOG_NS`
  - API: `iris.jit.set_quantum_log_threshold(...)`, `iris.jit.get_quantum_log_threshold()`

### Compile governance (budget + cooldown)

- Compile budget/window (ns):
  - Env: `IRIS_JIT_QUANTUM_COMPILE_BUDGET_NS`, `IRIS_JIT_QUANTUM_COMPILE_WINDOW_NS`
  - API: `iris.jit.set_quantum_compile_budget(...)`, `iris.jit.get_quantum_compile_budget()`
- Cooldown backoff bounds (ns):
  - Env: `IRIS_JIT_QUANTUM_COOLDOWN_BASE_NS`, `IRIS_JIT_QUANTUM_COOLDOWN_MAX_NS`
  - API: `iris.jit.set_quantum_cooldown(...)`, `iris.jit.get_quantum_cooldown()`
- Rearm cadence/trigger (ns):
  - Env: `IRIS_JIT_QUANTUM_REARM_INTERVAL_NS`, `IRIS_JIT_QUANTUM_REARM_MIN_OBSERVED_NS`
  - Behavior: controls how often a single-variant warm state may attempt multi-variant rearm, and the minimum observed latency required to trigger rearm.
- Rearm sensitivity controls:
  - Env: `IRIS_JIT_QUANTUM_REARM_MIN_SAMPLES`, `IRIS_JIT_QUANTUM_REARM_MAX_VOLATILITY`
  - Behavior: requires enough observations before rearm and suppresses rearm when per-run latency is too volatile.

## Warm-start metadata cache

Quantum telemetry is persisted for restart-time warm-up.

- Path: `__pycache__/.iris.meta.bin`
- Format: binary framed payload (`magic + flags + msgpack-bytes`), optional compression
- Persistence controls:
  - `IRIS_JIT_META_TTL_NS`
  - `IRIS_JIT_META_MAX_ENTRIES`
  - `IRIS_JIT_META_FLUSH_MIN`, `IRIS_JIT_META_FLUSH_MAX`
  - `IRIS_JIT_META_COMPRESS_MIN_BYTES`
  - `IRIS_JIT_META_REFRESH_NS`

Metadata lifecycle notes:

- Warm seeds are loaded during registration and may be staged before full quantum state is initialized.
- Aggressive-source registration paths use effective extracted source for seed/register/persist flow.
- Writes are adaptive/deferred during execution and force-flushed at process exit for short-lived runs.
- Unchanged profile *shape* may skip rewrite within the refresh window to reduce churn/noise.

## Failure model

Iris prioritizes correctness over acceleration:

- Unsupported syntax or compile misses: Python fallback.
- Runtime panic/mismatch in JIT path: guarded fallback.
- Quantum variant errors: fallback variant or Python path.
- Lowered kernel mismatch or unsupported expression shape: falls back to normal JIT execution path.

## Fallback behavior highlights

- Scalar fast path remains preferred for stable scalar workloads.
- Single-arg kernels now include generic sequence fallback when typed-buffer fast paths do not apply.
- Loop-step wrappers compile lowered runtime expressions (including `let_bind`) and evaluate safely in Python fallback mode when needed.

## Rearm tuning guidance

- One-shot / bursty workloads: increase `IRIS_JIT_QUANTUM_REARM_MIN_SAMPLES` and/or lower `IRIS_JIT_QUANTUM_REARM_MAX_VOLATILITY` to avoid speculative churn.
- Long-running stable workloads: keep `IRIS_JIT_QUANTUM_REARM_MIN_SAMPLES` low (or default) and raise `IRIS_JIT_QUANTUM_REARM_MAX_VOLATILITY` moderately if rearm feels too conservative.
- If runs differ heavily call-to-call, prefer stricter volatility gating before reducing rearm interval.

## Minimal example

```python
import iris

iris.set_quantum_speculation(True)
iris.set_quantum_speculation_threshold(0)
iris.set_quantum_compile_budget(10_000_000, 1_000_000_000)
iris.set_quantum_cooldown(0, 0)

@iris.offload(strategy="jit", return_type="float")
def heavy(a: float, b: float, c: float) -> float:
    return (a * a + b * b + c * c) / (a + b + c + 1.0)
```

> [!NOTE]
> On aarch64 targets, Iris adjusts JIT module flags to avoid unsupported relocation paths.

> [!TIP]
> For Android ARM devices, SIMD planning is supported when built for ARM targets (`aarch64`/`arm`). Backend selection still depends on runtime CPU feature availability.
