---
type: Logic
status: Closed
related_pillar: "[[Ch3_Methodology]]"
tags: [thesis, chapter_3, implementation, rust, rayon, concurrency]
---
# Methodology Output Structure

## Conceptual Logic
When translating a sequential Python loop into a parallelized Rust function using `rayon`, we encounter a fundamental paradigm shift required by Rust's ownership and concurrency model. In Python, you iteratively mutate a shared state (appending to a list). In Rust, multiple threads cannot safely mutate the same dynamically sized collection (`Vec`) concurrently without explicit synchronization tools (like a `Mutex`), which would bottleneck performance. 

Instead of an imperative loop that mutates a shared state (`list_of_results.push`), we must adopt a **functional mapping approach**. By using `map`, each thread takes inputs (the values of `lambda` and `gamma`), runs the experiment independently, and yields a `GameResult`. Once all threads finish, the results are assembled into a single vector using `collect()`.

Furthermore, objects shared across parallel closures must either be thread-safe references or individually owned by the thread. The `Optimizer` in `optigame` implements `Clone`, so we must clone it for each iteration to give every thread its own independent optimizer instance. Finally, NumPy arrays passed via PyO3 (`PyReadonlyArray1`) do not directly implement Rayon's parallel iteration traits. We must first convert them into Rust slices to unlock `.par_iter()`.

## API Reference Table
| Concept | Struct / Type | Method | Usage Context |
| :--- | :--- | :--- | :--- |
| **NumPy interop** | `numpy::PyReadonlyArray1<f64>` | `as_slice()` | Converts the readonly NumPy array into a Rust slice `&[f64]` to allow idiomatic parallel iteration. |
| **Rayon parallel iter** | `&[f64]` (slice) | `par_iter()` | Converts the slice into a parallel iterator to distribute the workload across CPU cores. |
| **Functional mapping** | `rayon::iter::ParallelIterator` | `zip()`, `map()`, `collect()` | `zip` joins two parallel iterators, `map` applies a closure to each pair yielding a `GameResult`, and `collect` gathers them into a `Vec<GameResult>`. |
| **Data ownership** | `Optimizer` | `clone()` | Creates an owned copy of the optimizer for the current parallel closure, satisfying the borrow checker. |
| **Simplex mapping** | `S` | `from_projected()` | Initializes the `x` and `y` strategies by projecting them onto the Simplex. |

## Logical Checklist
- [ ] Convert `P_lambda` and `Q_gamma` into standard Rust slices using the `as_slice().unwrap()` method.
- [ ] Create parallel iterators on both slices using `.par_iter()` and stitch them together using `.zip()`.
- [ ] Replace the imperative `.iter_par(mut |...| {...})` syntax with a functional `.map(|(&lambda, &gamma)| { ... })`.
- [ ] Clone the `optimizer` locally inside the closure.
- [ ] Chain a `.collect::<Vec<GameResult>>()` at the end.

## Structural Outline

```rust
#[pyfunction]
pub fn neighborhood_exploration(
    P_lambda: numpy::PyReadonlyArray1<f64>,
    Q_gamma: numpy::PyReadonlyArray1<f64>,
    optimizer: Optimizer,
    num_steps: usize,
    Normalize_matrix: bool
) -> Vec<GameResult> {
    
    let p_slice = P_lambda.as_slice().unwrap();
    let q_slice = Q_gamma.as_slice().unwrap();

    p_slice.par_iter().zip(q_slice.par_iter()).map(|(&lambda, &gamma)| {
        
        // # Calculate game matrix components
        // let a = ...
        // let matrix = array![[...]];
        
        // # Create the GameState
        let game_state = GameState { ... };
        
        let thread_optimizer = optimizer.clone();
        
        let mut experiment = Experiment { 
            state: game_state, 
            optimizer: thread_optimizer, 
            num_steps 
        };
        
        experiment.run_experiment_until_convergence_in_place()

    }).collect()
}
```
