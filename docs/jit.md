# JIT & Compute Offload

Iris exposes one decorator API for native acceleration:

```python
@iris.offload(strategy="jit", return_type="float")
def kernel(...):
    ...
```

- `strategy="jit"`: compiles eligible code paths with [Cranelift](https://github.com/bytecodealliance/wasmtime/tree/main/cranelift).
- `strategy="actor"`: executes on a dedicated Rust offload pool.

### Quick start
```python
import iris

@iris.offload(strategy="jit", return_type="float")
def vector_magnitude(x: float, y: float, z: float) -> float:
    return (x*x + y*y + z*z) ** 0.5

result = vector_magnitude(1.0, 2.0, 3.0)
print(result)
```

### What gets accelerated
- Scalar expression kernels with arithmetic/logic/comparisons/ternaries and common math calls.
- Generator reductions (`sum/any/all`) over `range(...)` and runtime containers.
- While-style reduction helpers (`sum_while`, `any_while`, `all_while`) and loop-control intrinsics.
- Scalar recurrence loops recognized by the frontend (for/while patterns), including inlined helper calls.

### Runtime behavior and safety
- JIT execution profiles are specialized by observed input shapes and data layouts.
- Supported scalar inputs: Python `float`, `int`, `bool` (lowered to native `f64` ABI).
- Supported vectorized buffers: `f64`, `f32`, signed/unsigned integers, and bool.
- On unsupported syntax, compile miss, panic, or profile mismatch, Iris falls back safely to Python.

### Observability and controls
- Enable JIT logs: `IRIS_JIT_LOG=1` or `iris.jit.set_jit_logging(...)`.
- Query logging status: `iris.jit.get_jit_logging()`.
- Optional multi-variant quantum speculation:
  - env: `IRIS_JIT_QUANTUM=1`
  - API: `iris.jit.set_quantum_speculation(...)` / `iris.jit.get_quantum_speculation()`

### Architecture Comparison

| Feature | Approach A (AST JIT) | Approach B (Actor Routing) |
| --- | --- | --- |
| **Performance** | Maximum (Native machine code) | High (Rust FFI + Zero-copy overhead) |
| **Supported Operations** | Strict subset (Math/Variables only) | Limited to pre-compiled Rust library |
| **Implementation Complexity** | Extremely High (Cranelift engineering) | Low (Reuses existing Iris architecture) |
| **GIL Behavior** | Completely bypassed | Bypassed via Actor messaging |

> [!NOTE]
> On aarch64 targets, Iris automatically adjusts JIT module flags to avoid unsupported relocation paths.
