# Vortex-Transmuter Guide

This document tracks the Vortex-Transmuter execution model in Iris and maps implementation status against RFC #0003.

It is intended to answer three questions clearly:
- What Vortex does today.
- What guardrails exist for safety and forward compatibility.
- What remains to reach full RFC behavior.

---

## 1. Scope and Runtime Model

Vortex is an experimental execution subsystem for deterministic preemption and transactional speculative recovery.

Current architecture uses three layers:
- Rust runtime integration (`Runtime` + `VortexEngine`) for preemption, staging, and transaction orchestration.
- Python transmutation path (`iris.transmute_function`) that attempts bytecode instrumentation on a shadow function and falls back safely when compatibility checks fail.
- Guard and verifier layer that validates bytecode shape and cache layout before applying patches.

Primary modules:
- `src/vortex/engine.rs`
- `src/vortex/transaction.rs`
- `src/vortex/transmuter.rs`
- `src/py/vortex.rs`
- `src/py/vortex_bytecode.rs`

---

## 2. Implemented Capabilities

### 2.1 Deterministic preemption checks

Implemented:
- Reduction-based preemption ticks in the runtime actor handling loop.
- Suspend-path handling that detaches and replenishes budget.
- Automatic suspend hook that triggers ghost checkpoint/race/replay flow in runtime preemption branches.

Key points:
- Preemption is exercised in the actual actor execution loop (`spawn_handler_with_budget`).
- Automatic hook increments replay telemetry (`vortex_auto_replay_count`) when staged ghost V-IO is committed and replayed.

### 2.2 Python transmutation and safety fallback

Implemented:
- Shadow-function transmutation with guard telemetry.
- Capability-based compatibility checks (not simple Python-version switches).
- Structured fallback reasons for unsupported or unsafe layouts.
- Stage-specific rewrite failure telemetry (probe extraction, instrumentation, code replace, and shadow construction).

Typical telemetry reasons:
- `opcode_metadata_unavailable`
- `quickening_metadata_unavailable`
- `invalid_wordcode_shape`
- `inline_cache_entries_incomplete`
- `original_cache_layout_invalid`
- `patched_cache_layout_invalid`
- `patched_code_too_large`
- `stack_depth_invariant_failed`
- `exception_table_metadata_unavailable`
- `exception_table_invalid`
- `patched_stack_metadata_unavailable`
- `patched_exception_table_metadata_unavailable`
- `patched_exception_table_invalid`
- `probe_extraction_failed`
- `probe_instrumentation_failed`
- `code_replace_failed`
- `types_module_unavailable`
- `shadow_function_construction_failed`

Note:
- Test-only deterministic hooks exist for selected late-stage fallback branches to keep CI behavior stable across CPython runtime variance.

### 2.3 Bytecode verifier and compatibility checks

Implemented:
- Wordcode shape verification and maximum size limits.
- Jump target and relative jump validation.
- Inline-cache layout verification using runtime quickening metadata.
- Probe compatibility validation prior to instrumentation.
- Exception-table invariant checks (range/depth, handler-target bounds, ordering, and duplicate-entry rejection).
- Quickening-aware handler-target validation rejects exception handlers that land on `CACHE` opcode slots.
- Stack-size minimum gate for safe probe injection assumptions.

Design intent:
- Continue operating on instruction-level IR and verifier checks to stay resilient to CPython bytecode format evolution.

### 2.4 Transactional ghosting primitives

Implemented:
- Checkpoint capture for primary and ghost transactions.
- Staged V-IO recording and commit/abort semantics.
- Ghost race resolution with policy control:
  - `FirstSafePointWins`
  - `PreferPrimary`
- Replay executor that can stop on failure.

### 2.5 Quiescence-gated swap support

Implemented in engine:
- Staged code swap queue.
- Swap application at safe conditions:
  - idle
  - quiescent stack conditions
  - completion (`ctx.done`)

### 2.6 Runtime-level Vortex APIs

Implemented wrappers in `Runtime` (feature `vortex`):
- Transaction lifecycle: start/stage/commit/take committed V-IO.
- Ghost lifecycle: start/stage/resolve race/replay.
- Auto policy controls: set/get automatic ghost arbitration policy.
- Auto telemetry accessors: replay count and resolution counts `(primary_wins, ghost_wins)`.
- Auto telemetry reset: clear counters to deterministic baseline for repeated runs.

Python `PyRuntime` wrappers expose:
- `vortex_set_auto_ghost_policy(...)`
- `vortex_get_auto_ghost_policy()`
- `vortex_get_auto_resolution_counts()`
- `vortex_get_auto_replay_count()`
- `vortex_reset_auto_telemetry()`
- `vortex_set_genetic_budgeting(bool)`
- `vortex_get_genetic_budgeting()`

This allows exercising Vortex behavior from runtime boundaries, not only from direct engine tests.

### 2.7 Genetic budgeting primitive

Implemented (runtime primitive):
- Optional runtime toggle for adaptive budgeting (`vortex_set_genetic_budgeting` / `vortex_genetic_budgeting_enabled`).
- Adaptive budget policy in the Vortex preemption loop:
  - Shrinks budget on suspend events.
  - Gradually grows budget on clean cycles.
  - Clamps within safe min/max bounds derived from base budget.

Scope note:
- This is an initial scheduler primitive and not yet a full historical-learning policy.

---

## 3. Roadmap Status (RFC #0003 Mapping)

Legend:
- Implemented: available in code with tests.
- Partial: available primitives, not yet full end-to-end behavior.
- Planned: not yet implemented.

| RFC Area | Status | Notes |
| :--- | :--- | :--- |
| 3.1 DIBP instruction-bound preemption | Partial | Runtime preemption and suspend hooks exist. Python opcode injection path is guarded and may fallback. |
| 3.2 Ghosting with transactional V-IO | Partial | Checkpoint, staged V-IO, race resolution, replay are implemented. Full production actor-flow policy orchestration still evolving. |
| 4.1 Quiescence-gated hot-swap | Implemented (engine level) | Staging and safe-point apply behavior implemented and tested in engine. |
| 4.2 Rescue pool detached stalling | Implemented (core primitive) | Rescue pool APIs and tests are present; broader operational policy tuning remains iterative. |
| 5.1 High-level IR future-proofing | Partial | Instruction IR + compatibility gates are present; continuous adaptation for new CPython internals remains ongoing. |
| 5.2 Vortex static verifier | Partial | Verifier checks now cover shape/jumps/cache layout plus exception-table and stack-depth gates; exception-handler semantics are still being expanded. |
| Genetic budgeting | Partial | Runtime adaptive budget primitive is implemented behind an explicit toggle; full historical/policy tuning is still pending. |
| Watchdog forced interrupt path | Planned | Not implemented yet. |
| Bytecode-level isolation rewrites | Planned | Not implemented yet. |

---

## 4. Tests and Verification Coverage

Current verification includes:
- Engine-level tests for transactions, ghost race resolution, replay behavior, and staged swap semantics.
- Runtime-level tests for Vortex wrapper lifecycle and automatic suspend hook replay.
- Runtime + PyRuntime policy/telemetry tests (including invalid policy and reset behavior).
- PyO3 integration tests for real Python execution and stage-specific fallback telemetry reasons.
- Bytecode utility tests for verifier behavior and compatibility rejection cases.

Targeted commands:

```bash
cargo test --lib runtime_vortex_ --no-default-features --features vortex -- --nocapture
cargo test --lib runtime_vortex_auto_ghost_hook_triggers_on_preempt_suspend --no-default-features --features vortex -- --nocapture
cargo test --test pyo3_vortex --no-default-features --features "pyo3 vortex" -- --nocapture
```

---

## 5. Guardrails and Failure Model

Vortex prioritizes safety over aggressive rewrite behavior.

Operational rules:
- If metadata is unavailable or incompatible, transmutation falls back to shadow tracing mode.
- Original function code objects are not mutated directly in the guarded path.
- Replay can be bounded by executor return value to stop on first failed side effect.
- Unsupported/unsafe conditions are exposed through explicit guard telemetry reasons.

This keeps behavior deterministic and debuggable while compatibility support expands.

---

## 6. Known Gaps and Next Milestones

Short-term milestones:
1. Complete verifier follow-up work for exception-handler semantics and stack-preservation coverage beyond current range/depth/min-stack gates.
2. Increase direct rewrite success on modern quickening-heavy runtimes without relaxing safety gates.
3. Push ghost race policies deeper into default runtime scheduling decisions.
4. Expose richer telemetry for automatic suspend hook (counts by reason/policy).

Mid-term milestones:
1. Integrate watchdog/escalation strategy for severe stalls.
2. Introduce adaptive quantum tuning (genetic budgeting style).
3. Explore actor memory-isolation rewrite policies under strict guard mode.

Later-phase milestones:
1. Add asyncio-aligned execution interop/mirroring goals after verifier and scheduler policy maturity.
2. Reduce Python function-color boundaries where safe, so transmuted flows can feel more uniform across sync/async call paths.
3. Validate these changes behind strict guard telemetry before making them default runtime behavior.

---

## 7. Practical Usage Notes

For users enabling Vortex paths:
- Treat APIs as experimental and validate on your Python/runtime version.
- Use targeted tests for your deployment feature set (`pyo3`, `vortex`, optional `jit`).
- Check guard telemetry to understand whether rewrite or fallback executed.

The current trajectory is incremental hardening with strict safety and test-first expansion.
