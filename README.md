<div align="center">

![Iris Banner](https://svg-banners.vercel.app/api?type=luminance&text1=Iris%20🌸&width=800&height=200&color=FFB6C1)

![Version](https://img.shields.io/badge/version-0.4.0-blue.svg?style=for-the-badge)
![Language](https://img.shields.io/badge/language-Rust%20%7C%20Python%20%7C%20Node.js-orange.svg?style=for-the-badge&logo=rust)
![License](https://img.shields.io/badge/license-AGPL_3.0-green.svg?style=for-the-badge)

**Hybrid distributed runtime fabric for actors, native compute offload, and cross-language services.**

[Architecture](docs/architecture.md) • [Usage Guide](docs/usage.md) • [JIT & Offload](docs/jit.md) • [Distributed Mesh](docs/distributed.md)

</div>

---

## Overview

**Iris** is a hybrid distributed runtime built in Rust with first-class **Python** and **Node.js** bindings. It combines three execution styles:
- **Actor Mesh:** Stateful, message-driven workflows with high concurrency.
- **Native Offload/JIT:** CPU-heavy hot paths accelerated via Cranelift.
- **Cross-Language API:** Service-oriented apps mixing Rust, Python, and Node.js.

Iris uses a **cooperative reduction-based scheduler** for fairness, providing built-in supervision, hot swapping, discovery, and location-transparent messaging across nodes.

> [!NOTE]
> Node.js bindings are currently in Alpha and reaching feature parity with Python.

---

## Core Capabilities

- **Hybrid Concurrency:** Mix "Push" green-thread actors with "Pull" OS-thread actors.
- **Atomic Hot-Swap:** Update live application logic (Python/Node) without zero downtime.
- **Global Discovery:** Register and resolve named services locally or over the network.
- **Self-Healing:** Path-scoped supervisors and structured `EXIT` reasons for fault tolerance.
- **JIT Acceleration:** Transparently compile Python math functions to native machine code.
    - **Quantum Speculation:** Optional multi-variant JIT selection with runtime telemetry, bounded by compile budget and cooldown controls (see [JIT Internals & Configuration](docs/jit.md)).

---

## Quick Start

### Installation

#### 🐍 Python
```bash
pip install maturin
maturin develop --release
```

#### 📦 Node.js
```bash
npm install
npm run build
```

### Basic Example (Python)

```python
import iris
rt = iris.Runtime()

# 1. Spawn a high-performance actor
def worker(msg):
    print(f"Got: {msg}")

pid = rt.spawn(worker, budget=50)

# 2. Transparently offload math to JIT
@iris.offload(strategy="jit", return_type="float")
def fast_math(x: float):
    return x * 1.5 + 42.0

# 3. Message the actor
rt.send(pid, b"hello world")
print(fast_math(10.0))
```

---

## Learn More

- [Full Architecture Reference](docs/architecture.md)
- [Usage Examples & API Guide](docs/usage.md)
- [JIT Internals & Configuration](docs/jit.md)
- [Distributed Mesh & Discovery](docs/distributed.md)

---

## Disclaimer

> [!IMPORTANT]
> **Production Status:** Iris is currently in **Beta**. The JIT/offload APIs are experimental.
>
> **Performance (v0.3.0):**
> - **Push Actors:** 100k+ concurrent actors, ~1.2M+ msgs/sec.
> - **Pull Actors:** 100k+ concurrent instances, ~1.5M+ msgs/sec.
> - **Hot-Swapping:** ~136k swaps/sec under load.
> - **See more at:** [v0.3.0 Benchmarks](docs/benchmarks/BENCHMARKS.md)

---

<div align="center">

**Author:** Seuriin ([SSL-ACTX](https://github.com/SSL-ACTX))

</div>