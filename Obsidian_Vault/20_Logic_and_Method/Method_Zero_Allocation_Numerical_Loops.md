---
type: Logic
status: Open
related_pillar: "[[Ch3_Methodology]]"
tags: [thesis, rust, performance, ndarray]
created: 2026-05-01
---
# Method: Zero-Allocation Numerical Loops in Rust

> [!info] Quick Summary
> Optimizing iterative numerical simulations by eliminating heap allocations through memory recycling and in-place mutation.

## 1. Initial Goal
Eliminate the $O(N)$ allocation bottleneck where every iteration of a game simulation or perturbation experiment created a new `Array2` on the heap.

## 2. Complexities Faced
- **Immutable Slices**: Array "Views" (slices) cannot be converted to owned matrices without a new allocation.
- **Consuming Operators**: Standard math operators (`+`, `-`) in libraries like `ndarray` typically consume their inputs and return a newly allocated result.
- **State Preservation**: Reusing memory requires disciplined "Reset" logic to prevent state pollution between independent experiments.

## 3. Core Concepts Learned

### Memory Recycling (`.assign()`)
Instead of `a = b`, which might rebind or reallocate, `.assign(&b)` performs a high-speed memory copy (`memcpy`) into the existing buffer. The heap address remains identical.

### The 4 Flavors of `ndarray` Addition
| Syntax | Performance | Memory Impact |
| :--- | :--- | :--- |
| `A + B` | Fast | Consumes both, reuses one's memory |
| `A + &B` | Very Fast | Consumes A, reuses its memory |
| `&A + B` | Very Fast | Consumes B, reuses its memory |
| `&A + &B` | Slow | Allocates a brand new Array |

### In-Place Operators (`+=`, `*=`)
Mapped to `AddAssign` and `MulAssign` traits. These are the most explicit way to ensure no hidden allocations occur.

### Array Slicing (`s!`)
Using the `s!` macro creates a zero-allocation window (View) into a sub-region of a matrix. This is used to compute metrics on the "tail" of an iteration history without copying the data.

## 4. Implementation Approach

1. **Mutating APIs**: Refactor functions from "Consuming" (`fn(self)`) to "Mutating" (`fn(&mut self)`).
2. **Buffer Hoisting**: Allocate the "dummy" matrices once in a worker setup (e.g., Rayon's `map_init`) and pass them into the loop as mutable references.
3. **Reset -> Modify -> Run**:
    - **Reset**: Call `.assign()` to restore the initial state template.
    - **Modify**: Use `+=`, `*=`, and `mapv_inplace` to update the noise.
    - **Run**: Pass the recycled buffer to the execution engine.

## 5. Backlinks
- [[Method_Parallel_Monte_Carlo_Worker_Pool]]
- [[Doc_Python_Optigame]]
- [[Impl_Overflow_Handling]]
