# Ord and PartialOrd Traits in Rust

## Conceptual Logic
In Rust, `PartialOrd` and `Ord` dictate how values can be compared. 
- `PartialOrd` is for types that can *sometimes* be compared, returning an `Option<Ordering>` (`Some(Less)`, `Some(Equal)`, `Some(Greater)`, or `None`). The float types `f32` and `f64` only implement `PartialOrd` because comparing a regular number with `NaN` (Not a Number) returns `None`.
- `Ord` is for types that have a strict, total order. It returns an `Ordering` directly.

This separation is crucial when attempting to sort arrays. Standard sorting functions like `slice::sort()` require the elements to implement `Ord`. Since `f64` does not implement `Ord`, calling `.sort()` on an array of floats will fail to compile.

To sort `f64` arrays, you must use `.sort_by()` and provide a closure that handles the `None` case (e.g., using `.expect("NaN encountered")` or using the `total_cmp` method).

## API Reference Table

| Method        | Trait Required         | Description                                                              | Example                                            |
| :------------ | :--------------------- | :----------------------------------------------------------------------- | :------------------------------------------------- |
| `sort()`      | `Ord`                  | Sorts the slice in-place.                                                | `vec.sort();` (Fails for `f64`)                    |
| `sort_by()`   | None (takes a closure) | Sorts the slice with a custom comparator function.                       | `vec.sort_by(\|a, b\| a.partial_cmp(b).unwrap());` |
| `total_cmp()` | N/A (method on `f64`)  | Provides a strict total order for `f64`, sorting NaNs deterministically. | `vec.sort_by(\|a, b\| a.total_cmp(b));`            |

## Logical Checklist
- [x] Use `sort_by` rather than `sort` when dealing with arrays or vectors of `f64`.
- [x] Inside the closure, if you are sure there are no `NaN`s, you can use `a.partial_cmp(b).expect("NaN encountered")`.
- [x] If sorting descending, reverse the arguments in the closure: `b.partial_cmp(a)`.
- [x] Alternatively, use the newer `.total_cmp(b)` method for a panic-free total ordering of floats.

## Structural Outline
```rust
// Sorting an array of f64 in descending order (e.g., in math.rs)
let mut my_vec: Vec<f64> = vec![3.14, 1.59, 2.65];
my_vec.sort_by(|a, b| b.partial_cmp(a).expect("NaN encountered"));

// Sorting using total_cmp (ascending)
let mut another_vec: Vec<f64> = vec![3.14, f64::NAN, 1.59];
another_vec.sort_by(|a, b| a.total_cmp(b));
```

See also: [[Finding Extrema in Rust Iterators]], [[Finding Min and Max of f64 Arrays in Rust]]

