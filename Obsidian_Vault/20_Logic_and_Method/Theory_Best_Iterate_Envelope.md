# Theoretical Best-Iterate Convergence Envelope for OMWU

This note provides a detailed mathematical explanation of the $O(T^{-1/6})$ best-iterate convergence bound for the Optimistic Multiplicative Weights Update (OMWU) algorithm in zero-sum games, analyzes why the current plotting logic in `optigame/analysis.py` fails to act as a proper upper bound, and proposes principled alternatives for setting the constant $C$.

---

## 1. Mathematical Logic Behind the $O(T^{-1/6})$ Best-Iterate Bound

The separation paper (*On Separation Between Best-Iterate, Random-Iterate, and Last-Iterate Convergence of Learning in Games*) establishes a unique landscape for OMWU in $2 \times 2$ zero-sum games with fully mixed Nash equilibria:
1. **Last-Iterate**: $\Omega(1)$ duality gap for a number of iterations proportional to $1/\delta$ (where $\delta$ is the minimum equilibrium probability). No game-independent last-iterate rate exists.
2. **Random-Iterate**: Fails to achieve polynomial uniform convergence, exhibiting a logarithmic lower bound of $\Omega(1/\log T)$.
3. **Best-Iterate**: Achieves a uniform, game-independent convergence rate of $O(T^{-1/6})$.

### The Two-Phase Analysis
The proof of the $O(T^{-1/6})$ best-iterate bound relies on splitting the OMWU trajectory into two phases depending on the boundary proximity parameter $\delta$:

1. **Initial Phase ($t \le T_1$)**:
   The iterates start at the uniform initialization $(1/2, 1/2)$, which is in the interior of the simplex. During this phase, before the strategies get close to the boundary, OMWU converges extremely rapidly. The best-iterate duality gap is bounded by:
   $$ \min_{k \le t} \text{DualityGap}(z^k) \le O\left( \frac{\log^2 t}{\eta t} \right) \approx \tilde{O}(t^{-1}) $$
   At the end of this phase ($t = T_1$, the first step where $x^{T_1}[1] \ge 1 - \delta$), the duality gap is guaranteed to be small:
   $$ \text{DualityGap}(z^{T_1}) \le 2\delta $$

2. **Global Phase ($t \ge T_1$)**:
   For larger time scales, we can use the global $\delta$-dependent random-iterate convergence rate (derived via a new connection to dynamic regret and interval regret):
   $$ \frac{1}{T} \sum_{t=1}^T \text{DualityGap}(z^t) = O\left( T^{-1/4}\delta^{-1/2} \right) $$

### Combining the Phases to get $O(T^{-1/6})$
Since the best-iterate is a non-increasing running minimum, for any $T \ge T_1$, it must be at most the gap at the boundary transition $T_1$, and also at most the running average over all steps $T$:
$$ \min_{t \in [1, T]} \text{DualityGap}(z^t) \le \min \left\{ 2\delta, C_0 T^{-1/4}\delta^{-1/2} \right\} $$
where $C_0$ is the game-independent constant from the global phase.

To find the uniform, $\delta$-independent rate, we find the worst-case boundary proximity $\delta$ where these two bounds intersect:
$$ 2\delta = C_0 T^{-1/4}\delta^{-1/2} \implies \delta^{3/2} = \frac{C_0}{2} T^{-1/4} \implies \delta = \left(\frac{C_0}{2}\right)^{2/3} T^{-1/6} $$
Substituting this worst-case $\delta$ back into $2\delta$ yields the uniform, parameter-independent rate:
$$ \min_{t \in [1, T]} \text{DualityGap}(z^t) \le O\left( T^{-1/6} \right) $$

---

## 2. Why the Current Plotting Code Fails (The Anchoring Bug)

In `optigame/analysis.py`, the envelope is plotted using:
```python
C = best_iterate[100] * (100 ** (1/6))
theoretical_bound = C * (steps ** (-1/6))
```

This code forces the theoretical envelope $C \cdot t^{-1/6}$ to intersect the empirical `best_iterate` curve exactly at index $100$ ($t = 101$). 

This choice causes the green `best_iterate` curve to remain **almost always above** the dashed envelope line due to two physical/mathematical phenomena:

1. **Transient vs. Asymptotic Scaling**:
   At $t = 101$, the trajectory is still in its very early, central stages close to the uniform initialization, where the duality gap is exceptionally small. Thus, the value `best_iterate[100]` is very small, which forces the scaling constant $C$ to be extremely small.
2. **Outward Spiraling and Flat Staircase Pattern**:
   OMWU in zero-sum games with fully mixed equilibria *diverges* from the Nash equilibrium, spiraling outward toward the simplex boundary. As it spirals outward, the orbit size expands and the last-iterate duality gap oscillates widely. The running minimum (`best_iterate`) remains flat for a very long time before making a closer approach to the NE in a later orbit.
   Meanwhile, the theoretical curve $C \cdot t^{-1/6}$ decays continuously. Because it was anchored to the tiny transient value at $t = 100$, it decays below the flat green `best_iterate` staircase almost instantly.
3. **Slope Discrepancy**:
   As your thesis states, the empirical decay rate of the best-iterate is much faster (approaching $O(T^{-1/2})$) than the worst-case uniform bound ($O(T^{-1/6})$). Going backward in time from $t=100$, the steeper empirical curve (slope $\approx -0.5$) rises much faster than the conservative theoretical curve (slope $\approx -0.167$). Thus, for $t < 100$, the green curve is above the dashed line. For $t > 100$, the outward spiraling delays the next best-iterate drop, keeping the green curve flat and above the decaying dashed line.
   
This results in a major incoherency with Section 4.5 of your thesis, where you state: *"...the best-iterate gap (green) converges rapidly, remaining strictly bounded by the theoretical $O(T^{-1/6})$ worst-case uniform envelope (black dashed)."*

---

## 3. Alternative and Principled Ways to Set the Constant $C$

To restore coherence to your thesis and ensure that the green curve remains strictly below the dashed envelope, you can choose one of the following three methods:

### Method A: The Maximum-Scaling Method (Empirical Upper Envelope) — *RECOMMENDED*
Instead of anchoring at an arbitrary single step like $t=100$, we find the mathematically smallest constant $C$ that ensures the theoretical bound is a valid upper envelope for the *entire* plotted range:
$$ C = \max_{t \in [1, T]} \left( \text{best\_iterate}[t-1] \cdot t^{1/6} \right) $$

**Code Implementation:**
```python
# Mathematically sound upper envelope over the entire simulation range
C = np.max(best_iterate * (steps ** (1/6)))
theoretical_bound = C * (steps ** (-1/6))
```

**Pros:**
* **Perfect Coherence**: Mathematically guarantees that the green curve never crosses above the dashed line (`theoretical_bound[t] >= best_iterate[t]` for all $t$).
* **Tight Visual Fit**: The dashed line will act as a tight, elegant upper boundary, touching the staircase corners of the best-iterate curve but never letting it escape.

---

### Method B: The Initial-Value Anchoring Method (Anchoring at $t=1$)
Anchor the theoretical envelope at the very first step, matching the initial duality gap of the uniform starting strategies. Since $t^{-1/6} = 1$ at $t=1$, we set $C = \text{best\_iterate}[0]$.

**Code Implementation:**
```python
# Anchor at the initial duality gap
C = best_iterate[0]
theoretical_bound = C * (steps ** (-1/6))
```

**Pros:**
* **Natural Starting Point**: Shows how the worst-case rate decays starting from the actual initial state of the game.
* **Visualizes Conservatism**: The green curve will immediately plunge far below the dashed line, visually demonstrating that the $O(T^{-1/6})$ worst-case bound is extremely conservative compared to OMWU's typical best-iterate performance in practice.

---

### Method C: The Fixed Representative Value (Analytical Constant)
Set $C$ to a fixed representative value based on the dimension-scale of the game (which typically starts at a duality gap of $\le 0.5$ for $2 \times 2$ normalized games).

**Code Implementation:**
```python
# Fixed analytical reference scale
C = 0.5
theoretical_bound = C * (steps ** (-1/6))
```

**Pros:**
* **Consistent Across Sweep Points**: Using the same $C$ across all four plot subfigures allows for an honest, direct comparison of how the convergence rates differ under different boundary conditions.
* **Simple and Transparent**: Avoids empirical fitting, serving as a pure mathematical rate reference.
