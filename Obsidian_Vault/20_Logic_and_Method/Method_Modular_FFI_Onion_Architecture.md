---
type: Logic
status: Open
related_pillar: "[[Ch3_Methodology]]"
tags: [thesis, rust, architecture, ffi, pyo3]
created: 2026-05-01
---
# Method: Modular FFI and the Onion Architecture

> [!info] Quick Summary
> Decoupling core scientific logic from Foreign Function Interface (FFI) bindings to improve testability, modularity, and maintainability.

## 1. Initial Goal
De-bloat the monolithic `experiments.rs` file, which previously handled everything from FFI translation to low-level matrix math.

## 2. Complexities Faced
- **FFI Entanglement**: Logic trapped inside `#[pyfunction]` wrappers was untestable without a Python Global Interpreter Lock (GIL).
- **Primitive Obsession**: Passing massive, unnamed tuples of arguments across boundary layers made signatures brittle.
- **Ownership Conflicts**: Simultaneous access to disjoint parts of a struct (e.g., a worker's state and its RNG) caused friction with the borrow checker.

## 3. Core Concepts Learned

### The Onion (Boundary) Pattern
Separate the codebase into layers based on dependency and concern:
1. **The FFI Boundary (`ffi.rs`)**: Translates Python types (`PyReadonlyArray`) into Rust types. It holds the GIL and handles translation boilerplate.
2. **The Orchestrator (`experiments.rs`)**: Coordinates concurrency (Rayon) and data flow. It is "Pure Rust" and knows nothing about Python.
3. **The Domain Core (`core.rs`)**: Defines the "Business Logic" and data structures. It is independent of FFI and Concurrency.

### Context Structs
Instead of passing 7+ arguments to a function, group related state into a `WorkerContext` or `ParameterConfig` struct. This reduces cognitive load and allows attaching logic via `impl` methods.

### Disjoint Field Borrowing
Rust allows borrowing separate fields of the same struct simultaneously. For example: `self.runner.execute(&mut self.state, &mut self.optimizer)` is valid because `runner`, `state`, and `optimizer` occupy non-overlapping memory regions.

## 4. Implementation Approach

1. **Thin FFI Layer**: Implement a thin wrapper in `ffi.rs` that extracts views and calls a core Rust function.
2. **Method Extraction**: Move free-floating functions into `impl` blocks of the relevant context structs.
3. **Project Structure**:
    - `lib.rs`: front-desk for module declarations.
    - `ffi.rs`: Python-facing API.
    - `experiments.rs`: Parallel simulations.
    - `core.rs`: Fundamental types (`GameState`, `GameResult`).

## 5. Backlinks
- [[Doc_Python_Optigame]]
- [[Method_Parallel_Monte_Carlo_Worker_Pool]]
- [[Impl_Testing_Setup]]
