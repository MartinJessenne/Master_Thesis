---
type: #Logic
status: Open
related_pillar: "[[Method_Zero_Allocation_Numerical_Loops]]"
tags: [thesis, rust, memory, performance, architecture, optimization]
created: 2026-05-03
---
# Cost of Copying in Rust: Hardware Reality vs. OS Allocation

> [!info] Quick Summary
> Clarifies the fundamental difference between an expensive "Copy" (an OS-level heap allocation of $O(N)$ complexity) and a zero-cost "Copy" (a CPU register load or a 24-byte fat pointer duplication of $O(1)$ complexity). Essential for understanding why dereferencing primitive types (`&f64`) or copying `ArrayView`s is the theoretical maximum limit of performance.

## 1. The Costly Copy (Heap Allocation)

In high-level languages or when explicitly calling `.clone()` on owned collections in Rust (like `Vec<T>` or `Array1<T>`), a "copy" refers to an **OS-Level Heap Allocation**.

*   **Mechanism**: The CPU asks the Operating System (OS) for a new, contiguous block of RAM. Once the OS finds and locks this memory, the CPU must iterate over the original data in RAM, read it across the system memory bus, and write it to the new location.
*   **The Cost**: This is an $O(N)$ operation. For an array of 1 million elements, it requires moving 8 Megabytes of data. This incurs massive latency, triggers OS kernel locks, and completely destroys the CPU cache lines.
*   **Rust Context**: Rust explicitly prevents this from happening implicitly. If a type manages heap memory, it does not implement the `Copy` trait. You must explicitly opt-in to this massive performance hit by calling `.clone()`.

## 2. The Zero-Cost "Copy" (Fat Pointers & Views)

When dealing with slices (`&[T]`) or ndarray views (`ArrayView1<T>`), copying the variable does *not* copy the underlying data.

*   **The "Fat Pointer"**: An `ArrayView1<f64>` is not an array; it is a tiny metadata struct (a "signpost" or "binoculars") living on the stack. It typically contains:
    1. A pointer to the first element in RAM (`*const f64`).
    2. The length of the view (`usize`).
    3. The stride (the byte jump to the next element).
*   **The Cost**: Regardless of whether the view points to 10 elements or 1 billion elements, the `ArrayView` struct is always exactly 24 bytes (on a 64-bit architecture). 
*   **Rust Context**: Copying an `ArrayView` simply copies these 24 bytes from one CPU register to another. It is an $O(1)$ operation taking less than a nanosecond. It does not allocate memory on the heap. Because it is trivially cheap, `ArrayView` implements the `Copy` trait.
*   **Conclusion**: Calling `.into_iter()` on an `ArrayView` consumes the 24-byte signpost, not the 1 million elements.

## 3. The Physical Limit "Copy" (CPU Register Load)

When iterating over an array and performing math (e.g., `acc.max(elem)` where `elem` is a dereferenced `&f64`), the computer *must* process each element.

*   **Mechanism**: To add or compare two numbers, the CPU cannot perform math directly in the RAM sticks. It must issue a **Register Load** instruction. It tells the RAM to send those specific 8 bytes (the `f64`) across the bus into the ultra-fast L1 cache, and then directly into a CPU execution register (like an XMM or YMM register in the ALU).
*   **The Cost**: This is the absolute physical limit of computation. You cannot process data without loading it into the CPU.
*   **Rust Context**: This is why primitive types like `f64`, `usize`, and `bool` implement the `Copy` trait. Moving 8 bytes from RAM into a CPU register is a bitwise copy. There is no OS allocation, no heap interaction, and no pointer chasing. The CPU's hardware pre-fetcher optimizes this sequential read to be as fast as physically possible.

## 4. Synthesis: Dereferencing is Not a Clone

When you write:
```rust
let tail: ArrayView1<f64> = gaps.slice(s![cutoff_idx..]);
let max_val = tail.into_iter().fold(std::f64::NEG_INFINITY, |acc, &elem| acc.max(elem));
```

1. `tail.into_iter()`: Destroys a 24-byte fat pointer (Zero-cost).
2. `&elem`: Instructs the CPU to look at the RAM address of the current element.
3. `acc.max(elem)`: Triggers a CPU Register Load, moving 8 bytes of raw data into the ALU to perform the math (Physical limit of speed).

There are zero heap allocations. This represents the pinnacle of performant data processing.

---
## Related Notes
- [[Method_Zero_Allocation_Numerical_Loops]]: Context on how we avoid allocations in the `WorkerContext` hot loop.
- [[Doc_Python_Optigame]]: How these zero-cost abstractions cross the FFI boundary to Python.
- [[Ch3_Methodology]]: The broader architectural strategy for fast simulations.
