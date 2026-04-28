# Methodology Output Structure

## Conceptual Logic
In Rust, comparison operations (`<`, `>`, `==`) are backed by traits. 
`PartialEq` and `Eq` handle equality.
`PartialOrd` allows for partial ordering, meaning some values might not be comparable. When you call `.partial_cmp()`, it returns an `Option<Ordering>`, where `None` signifies that the two values cannot be compared.
`Ord` requires a strict total order, meaning every two values can be definitively compared. It returns an `Ordering` directly (either `Less`, `Equal`, or `Greater`).

This distinction prevents logic errors at compile time. Sorting algorithms and min/max functions often require `Ord` because they need a strict total order to function correctly. Back to hub: [[Finding Min and Max of f64 Arrays in Rust]].

## API Reference Table
| Trait / Method | Description | Example |
| :--- | :--- | :--- |
| `std::cmp::PartialOrd` | Trait for values that can be partially ordered. | `impl PartialOrd for MyType` |
| `partial_cmp` | Compares two values, returning an `Option`. | `let ord = a.partial_cmp(&b);` |
| `std::cmp::Ord` | Trait for values with a total order. Requires `Eq` and `PartialOrd`. | `impl Ord for MyType` |
| `cmp` | Compares two values, returning a strict `Ordering`. | `let ord = a.cmp(&b);` |

## Logical Checklist
- [ ] Identify if your type has a strict total order (all pairs of values can be compared).
- [ ] Implement `PartialOrd` for types where comparisons make sense, even if some values are incomparable.
- [ ] Implement `Ord` only when every value can be definitively compared to any other value without ambiguity.

## Structural Outline
```rust
// Implementing Ord and PartialOrd for a custom struct
#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct PlayerScore {
    score: i32,
}

// Comparing values with Ord
let p1 = PlayerScore { score: 10 };
let p2 = PlayerScore { score: 20 };
let ordering = p1.cmp(&p2); // Returns std::cmp::Ordering::Less

// Comparing values with PartialOrd
let f1: f64 = 10.0;
let f2: f64 = f64::NAN;
let partial_ordering = f1.partial_cmp(&f2); // Returns None
```


