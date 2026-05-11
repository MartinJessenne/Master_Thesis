---
type: Logic
status: Open
related_pillar: "[[Ch3_Methodology]]"
tags: [thesis, rust, concurrency, rayon]
created: 2026-05-01
---
# Method: Parallel Monte Carlo Worker Pool with Rayon

> [!info] Quick Summary
> Scaling stochastic simulations (Monte Carlo) using a high-performance worker pool pattern to manage stateful resources like RNGs and pre-allocated buffers.

## 1. Initial Goal
Run thousands of independent matrix game experiments (neighborhood exploration) concurrently while maintaining a low memory footprint and avoiding the overhead of repeated thread spawning.

## 2. Complexities Faced
- **The "Memory Wall"**: Pre-allocating all random matrices (e.g., in a 3D array) before the loop caused OOM (Out Of Memory) crashes at scale.
- **Shared Mutability Constraints**: Random Number Generators (RNGs) in Rust are stateful and `!Sync`. They cannot be safely shared between threads without expensive Mutex locking.
- **State Recycling**: Experiments need to "forget" the previous run's results (reset strategies and gradients) to ensure statistical independence.

## 3. Core Concepts Learned

### Worker Pool Pattern (`map_init`)
Instead of one-task-per-thread, Rayon uses a fixed pool of threads (usually matching CPU cores). `map_init` allows creating a **Worker Context** once per thread.
- **Init Phase**: Allocate heavy resources (RNG, dummy matrix buffer, Optimizer).
- **Map Phase**: Reuse those resources for hundreds of sequential tasks assigned to that thread.

### Scoped Borrowing
Since the parallel loop finishes before the orchestrating function returns, we use **lexical references** (`&T`) instead of `Arc<T>`. This eliminates atomic reference counting overhead.

### Explicit Re-borrowing (`&mut *ptr`)
When passing a mutable reference received from a closure into an API that expects a new mutable borrow, the `&mut *ptr` syntax allows "re-borrowing" the underlying data without losing ownership of the primary pointer.

## 4. Implementation Approach

1. **On-the-fly Generation**: Generate noise matrices inside the thread worker rather than pre-allocating.
2. **Resource Encapsulation**: Wrap `GameState`, `Optimizer`, and `ThreadRng` into a `WorkerContext` struct.
3. **The Pipeline**:
    - **Reset**: Wipe auxiliary state (optimizer memory).
    - **Randomize**: Use `mapv_inplace` with the thread-local RNG.
    - **Compute**: Perform in-place math (`+=`, `*=`).
    - **Run**: Execute the runner and collect the scalar result.

## 5. Backlinks
- [[Random_Neighborhood_Exploration]]
- [[Method_Chaos_Metrics]]
- [[Doc_Python_Optigame]]
