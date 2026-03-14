# JIT & Compute Offload

This document describes Iris JIT behavior, supported language surface, intrinsics, and runtime controls.

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

## Quantum speculation behavior

When enabled, Iris may compile multiple variants and select adaptively using runtime telemetry.

- Selection is runtime-driven from per-variant execution stats.
- If speculation is gated by runtime controls, Iris falls back to single-variant compile.
- Failures or mismatches fall back safely to Python execution path.

## Runtime Controls

### Logging

- Enable: `IRIS_JIT_LOG=1` or `iris.jit.set_jit_logging(...)`
- Read status: `iris.jit.get_jit_logging()`

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

## Warm-start metadata cache

Quantum telemetry is persisted for restart-time warm-up.

- Path: `__pycache__/.iris.meta.bin`
- Format: binary framed payload (`magic + flags + msgpack-bytes`), optional compression
- Persistence controls:
  - `IRIS_JIT_META_TTL_NS`
  - `IRIS_JIT_META_MAX_ENTRIES`
  - `IRIS_JIT_META_FLUSH_MIN`, `IRIS_JIT_META_FLUSH_MAX`
  - `IRIS_JIT_META_COMPRESS_MIN_BYTES`

## Failure model

Iris prioritizes correctness over acceleration:

- Unsupported syntax or compile misses: Python fallback.
- Runtime panic/mismatch in JIT path: guarded fallback.
- Quantum variant errors: fallback variant or Python path.

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
