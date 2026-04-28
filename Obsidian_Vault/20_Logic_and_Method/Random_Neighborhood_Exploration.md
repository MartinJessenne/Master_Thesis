---
type:
status: Open
related_pillar: "[[Ch4_Empirical_Study]]"
tags:
  - thesis
created: 2026-04-27 12:24
---
> [!info] Quick Summary
> Implementations options, path and reflection for the random neighborhood exploration
# Description
We're going to opt for a pure rust implementation
The pipeline is going to be the following : 
1. We're going to make a grid search for several values of $\epsilon$ 
2. for each value of $\epsilon$ we're going to have a vector of random duality gap results
3. for each vector, reduce it to one scalar using one of the three [[Method_Chaos_Metrics]]
4. produce a scatter of those scalar for each value of $\epsilon$, and maybe use box and whiskers plots to plot the "uncertainty" (the variance in fact)

> [!abstract] **Algorithm: `random_neighborhood_exploration`**
> ---
> **Input:**  `f64: epsilon`, `usize: num_exploration`, `Array2<f64>: A_delta`
> **Output:** `Array1<f64>: duality_gap_history`
> ---					
> 1. **Initialize** the `num_exploration` random matrices in `random_matrices`
> 2. **For** $t = 1, \dots, \text{num\_exploration}$ **do**:
>    1.  `duality_gap`  = `Omwu`(`A_delta` + $\epsilon \times$  `random_matrices[t]`)
>    2.  `result.append(duality_gap)`
> 3. **Return** result





