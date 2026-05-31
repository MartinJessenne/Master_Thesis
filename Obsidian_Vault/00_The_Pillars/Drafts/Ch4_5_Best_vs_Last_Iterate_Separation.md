---
type: Draft
status: Open
related_pillar: "[[Ch4_Empirical_Study]]"
tags: [thesis, chapter_4, draft]
created: 2026-05-31
---

# Section 4.5: Best vs. Last Iterate Separation

## 1. The Three Non-Ergodic Convergence Modes
To rigorously evaluate the convergence of learning dynamics in ill-conditioned games, we must distinguish between three distinct non-ergodic convergence modes. Let $z^t = (x^t, y^t)$ denote the strategy profile played at iteration $t$, and let $\text{Gap}(z)$ be the standard zero-sum duality gap. For a total horizon of $T$ iterations, we analyze:

1. **Last-Iterate Convergence:** Evaluates the quality of the final strategy profile produced at the very end of the learning horizon:
   $$ \text{Last-Gap}(T) = \text{Gap}(z^T) $$
2. **Random-Iterate Convergence:** Evaluates the expected quality of a strategy profile selected uniformly at random from the history. This is equivalent to the time-averaged duality gap:
   $$ \text{Random-Gap}(T) = \mathbb{E}_{t \sim \mathcal{U}(1, T)} [\text{Gap}(z^t)] = \frac{1}{T} \sum_{t=1}^T \text{Gap}(z^t) $$
3. **Best-Iterate Convergence:** Evaluates the highest-quality strategy profile achieved at any single step during the entire trajectory:
   $$ \text{Best-Gap}(T) = \min_{t \in [1, T]} \text{Gap}(z^t) $$

---

## 2. Theoretical Separation and Uniformity Bounds
The core theoretical result established by Cai et al. (2025) is a stark **complexity separation** between these three modes for OMWU in $2 \times 2$ zero-sum matrix games:

* **Last-Iterate (Non-Uniform / $\Omega(1)$):** OMWU fails to achieve a uniform last-iterate convergence rate. As $\delta \to 0$ (where $\delta$ is the minimum equilibrium probability at the boundary), the last-iterate duality gap remains bounded away from zero by a constant:
   $$ \text{Last-Gap}(T) = \Omega(1) $$
   for a number of iterations proportional to $1/\delta$.
* **Random-Iterate (Non-Uniform / $\Omega(1/\log T)$):** The average-iterate also suffers from a non-uniform barrier, failing to achieve polynomial convergence and exhibiting a logarithmic lower bound of $\Omega(1/\log T)$.
* **Best-Iterate (Uniform / $O(T^{-1/6})$):** In stark contrast, the best-iterate duality gap is guaranteed to converge to zero at a uniform rate that is **completely independent of the boundary proximity $\delta$**:
   $$ \text{Best-Gap}(T) \le O\left(T^{-1/6}\right) $$

---

## 3. Demystifying the Big-$O$ Bound: How to Validate $O(T^{-1/6})$ Empirically
A common source of confusion in empirical studies is how to "validate" a theoretical Big-$O$ bound. Because a Big-$O$ statement incorporates an implicit, game-dependent multiplying constant $C > 0$:
$$ \text{Best-Gap}(T) \le C \cdot T^{-1/6} $$
evaluating a single raw gap value at a specific iteration (e.g., noting that at $T=10,000$, the gap is $0.05$, which is less than $10000^{-1/6} \approx 0.21$) does not formally prove or disprove the bound. 

To rigorously validate the $O(T^{-1/6})$ convergence behavior, we must analyze two key properties in our empirical data:

### A. The Uniformity Test (Independence from $\delta$)
This is the most critical test. If a bound is **uniform**, its decay rate must not degrade as the Nash Equilibrium approaches the boundary of the probability simplex ($\delta \to 0$). 
* **The Pathology of the Last/Random Iterates:** In our parametric sweeps, as we push the equilibrium closer to the boundary, the last-iterate and random-iterate gaps explode toward $\Omega(1)$ for a fixed horizon $T$. Their curves "flatten out," showing that they require exponentially more time to converge.
* **The Robustness of the Best Iterate:** In contrast, the best-iterate curve remains extremely small and decays consistently, regardless of whether the equilibrium is in the center of the simplex or pushed right against the boundary wall. Plotting the best-iterate curves for different values of $\delta$ reveals that they lie on top of one another, empirically validating that the rate is **uniform** (parameter-independent).

### B. The Log-Log Decay Rate (The Power-Law Slope)
A power-law relationship of the form $y \le C \cdot T^{-\alpha}$ becomes a straight line on a logarithmic scale:
$$ \log(\text{Best-Gap}) \le \log(C) - \alpha \log(T) $$
By plotting the empirical best-iterate duality gap against the iteration index $T$ on a **log-log scale**, the rate is represented by the slope of the curve:
* The theoretical worst-case bound guarantees a slope of at least $-\alpha = -1/6 \approx -0.167$.
* If the empirical curve exhibits a slope that is steeper (e.g., a slope of $-0.5$ or $-1.0$), then it strictly dominates and validates the theoretical upper bound of $-1/6$. Empirically, the best-iterate often converges much faster than $O(T^{-1/6})$ because the discrete steps happen to land extremely close to the Nash Equilibrium during its periodic dips.

---

## 4. Geometric Orbit Interpretation of the Separation
Our 2D strategy-space trajectories (Section 4.4) provide a beautiful, intuitive geometric mechanism that explains exactly why this separation occurs:

* **Last-Iterate wild oscillations:** OMWU is trapped in a permanent closed orbit (a stable limit cycle) around the Nash Equilibrium. Because it never contracts, the last-iterate gap simply oscillates indefinitely, never converging.
* **Random-Iterate slow decay:** The strategy trajectory spends most of its time in the outer, boundary-adjacent segments of the orbit (the wide "peaks" of the $L^2$ distance). Because it spends so little time near the solution, the time-averaged (random-iterate) gap converges extremely slowly.
* **Best-Iterate rapid convergence:** The orbit is eccentric and periodically "dips" extremely close to the Nash Equilibrium. During these brief proximity events, the duality gap drops sharply to near-zero. The best-iterate is a monotonically decreasing step function that records the **"deepest dip"** achieved so far. 

As the simulation progresses, the discrete strategy updates land at different points on the continuous orbit. Over thousands of iterations, these discrete hits eventually sample a point that is exceptionally close to the exact coordinates of the equilibrium. The best-iterate metric captures this single closest event, yielding a robust, uniform convergence guarantee despite the permanent global instability of the system.
