# Finding Extrema in Rust Iterators

## Conceptual Logic
When iterating over collections of `f64` floats in Rust, finding the minimum or maximum is not as straightforward as calling `.min()` or `.max()`. Because `f64` implements `PartialOrd` but not `Ord` (due to the presence of `NaN`), iterators of floats do not automatically satisfy the trait bounds for standard extrema methods.

There are two primary ways to find extrema in Rust for `f64` arrays, which we use throughout the codebase (e.g., in our `optigame` implementation):
1. **Using `.fold()`:** You can start with an initial value (like `f64::INFINITY` or `f64::NEG_INFINITY`) and fold over the iterator using the `.max()` or `.min()` methods specifically implemented on `f64`.
2. **Using `.min_by()` or `.max_by()`:** You can provide a custom comparison closure using `.partial_cmp().unwrap()` or the more modern and safer `.total_cmp()`.

## API Reference Table

| Method / Approach | Example Usage | Description |
| :--- | :--- | :--- |
| `fold` with `f64::max` | `.fold(f64::NEG_INFINITY, \|a, &b\| a.max(b))` | Accumulates the maximum value explicitly using `f64`'s intrinsic `max` method. Safe and ignores NaNs appropriately. |
| `min_by` with `total_cmp` | `.min_by(\|a, b\| a.total_cmp(b)).unwrap()` | Finds the minimum element using `total_cmp`, which enforces a strict total ordering even on NaNs. |

## Logical Checklist
- [x] If using `.fold()`, ensure your initial accumulator is `f64::NEG_INFINITY` for finding a maximum, or `f64::INFINITY` for finding a minimum.
- [x] If using `.min_by()` or `.max_by()`, pass a closure using `a.total_cmp(b)`.
- [x] Do NOT use `.cmp()` on `f64` (e.g., `b.cmp(acc)`), because `f64` does not implement `Ord`. Attempting to do so will result in a compiler error.

## Structural Outline
```rust
// Finding max using fold (e.g., in optimizers.rs step calculations)
let step_x: Array1<f64> = ...;
let max_step_x = step_x.iter().fold(f64::NEG_INFINITY, |a: f64, &b| a.max(b));

// Finding min/max using total_cmp (e.g., in experiments.rs duality gap)
let grad_x: Vec<f64> = ...;
let min_x: f64 = *grad_x.into_iter().min_by(|a, b| a.total_cmp(b)).unwrap();

// Incorrect usage (will not compile)
let matrix: Array2<f64> = ...;
let max_component = matrix.iter().fold(f64::NEG_INFINITY, |acc, b| b.cmp(acc)); // ERROR: no method named `cmp` found for `&f64`
```

See also: [[Ord and PartialOrd Traits in Rust]], [[Finding Min and Max of f64 Arrays in Rust]]

