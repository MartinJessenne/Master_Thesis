---
type: Draft
status: Open
related_pillar: "[[Ch4_Empirical_Study]]"
tags: [thesis, chapter_4, draft]
created: 2026-05-30
---

# Section 4.1: Baseline Replication and Neighborhood Exploration

## 1. Baseline Replication of $A_\delta$
To establish a rigorous empirical foundation, we first replicate the pathological behavior documented by Cai et al. (2025). The baseline hard instance is defined by the parameterized $2 \times 2$ zero-sum matrix game:
$$ A_\delta = \begin{pmatrix} \frac{1}{2} + \delta & \frac{1}{2} \\ 0 & 1 \end{pmatrix} $$
where $\delta > 0$ represents the minimum probability assigned to any action in the unique Nash Equilibrium (NE) profile $(x^*, y^*)$. 

In this game, the theoretical Nash Equilibrium is given by:
$$ x^* = \left( \frac{1}{1+\delta}, \frac{\delta}{1+\delta} \right) \quad \text{and} \quad y^* = \left( \frac{1}{2}, \frac{1}{2} \right) $$
As $\delta \to 0$, the $x$-player's equilibrium strategy is forced arbitrarily close to the boundary of the probability simplex $\Delta^2$. Cai et al. (2025) proved that for any Legendre-regularized Optimistic Follow-the-Regularized-Leader (OFTRL) algorithm—including Optimistic Multiplicative Weights Update (OMWU)—the duality gap remains bounded away from zero by a constant for at least $\Omega(1/\eta L \delta)$ iterations, where $\eta$ is the step size and $L$ is the Lipschitz constant of the update map. Because $\delta$ can be chosen arbitrarily close to zero, this lower bound precludes uniform last-iterate convergence.

Our computational framework successfully reproduces this behavior. When running OMWU on $A_\delta$ with small values of $\delta$, the strategies fail to stabilize, instead entering a persistent cyclic phase with a large, non-decaying last-iterate duality gap.

---

## 2. Robustness and Locality: The Neighborhood Analysis
While the theoretical lower bound is mathematically robust, a natural question arises: *Is this pathology a fragile mathematical artifact of the exact matrix coefficients in $A_\delta$, or is it a stable physical phenomenon that persists under numerical noise?*

To investigate this, we perform a robust neighborhood analysis by applying uniform random perturbations to the baseline game $A_\delta$. We define a perturbed game matrix:
$$ A_{\text{perturbed}} = A_\delta + \epsilon U $$
where $U \in [-1, 1]^{2 \times 2}$ is a perturbation matrix sampled uniformly, and $\epsilon > 0$ scales the magnitude of the perturbation. We evaluate OMWU's convergence dynamics across a large ensemble of these perturbed matrices using two distinct experimental methodologies to isolate the spatial characteristics of this instability.

### Experimental Design 1: Concentric Shell Exploration
To map the boundary of the pathology without overlapping experiments, we partition the perturbation space into concentric $L_\infty$ shells. For a set of discrete perturbation boundaries $0 = \epsilon_0 < \epsilon_1 < \dots < \epsilon_n$, the $i$-th concentric shell corresponds to the domain where the perturbation magnitude resides within $[\epsilon_{i-1}, \epsilon_i]$. 

```mermaid
radialChart
    title "L_inf concentric shells exploration domain"
    theme default
    "Shell 1: [0, e_1]": 20
    "Shell 2: [e_1, e_2]": 40
    "Shell 3: [e_2, e_3]": 60
    "Shell 4: [e_3, e_4]": 80
```

> [!IMPORTANT]
> **Methodological Nuance on Sampling Density vs. Volume:**
> Because the geometric volume of an $L_\infty$ shell scales rapidly with its radius, maintaining a constant sampling density (matrices per unit volume) would require exponentially more samples in the outer layers. 
> 
> Instead, our experimental engine (`concentric_exploration` in `optigame`) is designed to sample the **exact same number of matrices** (e.g., 1,000 runs) in each concentric band. This ensures that every layer has the same statistical weight in our visual representations, preventing the high-volume outer bands from numerically dominating the data set, and allowing us to observe the clean, uncorrupted transition of metrics across shells.

By isolating the perturbation magnitudes into distinct, non-overlapping bands, we prevent the "propagation of chaos" from larger bounds into smaller ones. The maximum duality gap over the last 10% of iterations (e.g., the last 1,000 steps of a 10,000-step simulation) is recorded for each run. 

### Experimental Design 2: Scattered Exploration
In parallel, we execute a "scattered" exploration to obtain a continuous representation of the transition. Here, we sample perturbation matrices $U$ globally and uniformly within the entire ball $B_\infty(0, \epsilon_{\text{max}})$. A posteriori, we compute the exact $L_\infty$ norm of the actual perturbation matrix:
$$ \|U\|_\infty = \max_{i,j} |U_{i,j}| $$
and plot the resulting duality gap metrics directly against this norm as a scatter plot. This continuous mapping serves to validate the discrete shell boundaries and exposes the continuous transition from unstable orbits to smooth convergence.

---

## 3. Empirical Results: Robustness, Locality, and the Contrast with OGDA
The results of our neighborhood exploration reveal two profound characteristics of OMWU's boundary pathology:

1. **Robustness:** For small perturbation magnitudes ($\epsilon \to 0$), the median of the last-iterate duality gap remains significantly elevated, matching the baseline $A_\delta$ game. This demonstrates that the pathology is not a fragile mathematical coincidence; the instability is highly robust to environmental noise.
2. **Locality:** As the perturbation magnitude $\epsilon$ increases, the median and variance of the last-iterate duality gap begin to decay. This decay occurs because larger perturbations push the unique Nash Equilibrium away from the extreme boundaries of the simplex toward the interior, where the OMWU dynamics are geometrically stable. Thus, the pathology is shown to be a **strictly local phenomenon** confined to the boundary regions.

```
       Duality Gap
          ^
          |   *  *  (Local Pathology: Wild oscillations, high gap)
          |  * * * *
          | * * * * *
          |------------------->  Threshold (e_crit)
          |                    *
          |                      *  *  (Interior: Smooth decay to NE)
          |                            *   *
          +---------------------------------------> Perturbation Magnitude (epsilon)
```

### The Contrast with OGDA
To confirm that this pathological behavior is an intrinsic structural property of OMWU rather than a general limitation of game-theoretic learning, we run the identical concentric and scattered explorations using Optimistic Gradient Descent-Ascent (OGDA). 

OGDA—which utilizes a standard quadratic Euclidean regularizer—exhibits **no pathological anomalies**. Across all concentric shells and perturbation magnitudes, the last-iterate of OGDA converges smoothly to the perturbed Nash Equilibria, maintaining a near-zero last-iterate duality gap. 

This stark contrast strongly nudges us to dive deeper. The robust yet highly localized boundary pathology of OMWU must stem from its underlying regularizer structure (the negative entropy Legendre regularizer), which accumulates cumulative historical gradients without a forgetting mechanism (OFTRL). 

To map this boundary behavior with mathematical precision and systematically explore the entire simplex interior, we require a parameterized family of games that can control the Nash Equilibrium's position directly from first principles. This motivates the parametric game design developed in the next section.
