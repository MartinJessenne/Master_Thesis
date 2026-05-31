---
type: Draft
status: Open
related_pillar: "[[Ch4_Empirical_Study]]"
tags: [thesis, chapter_4, draft]
created: 2026-05-31
---

# Section 4.4: Strategy Profiles and Orbital Dynamics

## 1. Deterministic Parametric Sweeps
The random concentric shell explorations presented in Section 4.1 establish the robustness and local nature of OMWU's boundary pathology. However, random perturbations do not allow for a systematic mapping of the dynamics as a function of the equilibrium's exact position relative to the simplex boundaries. 

To analyze the spatial layout of this boundary pathology with high precision, we transition from random perturbations to a deterministic sweep along a parameterized curve in the strategy space. The sweep is designed as a circle centered at the baseline $A_\delta$ Nash Equilibrium (with $\delta = 0.01$). The Nash Equilibrium coordinates $(p_0, q_0)$ at the center of the sweep are:
$$ p_0 = \frac{1}{1+\delta} \quad \text{and} \quad q_0 = \frac{1}{2(1+\delta)} $$

To ensure that the circular sweep remains strictly within the interior of the joint strategy space $\Delta^2 \times \Delta^2$ (geometrically represented by the unit square $[0, 1] \times [0, 1]$), the sweep radius $r$ must be strictly smaller than the distance from the center $(p_0, q_0)$ to the nearest boundary. The distances to the four boundaries of the unit square are:
1. Distance to $p = 0$: $d_{p,0} = \frac{1}{1+\delta}$
2. Distance to $p = 1$: $d_{p,1} = \frac{\delta}{1+\delta}$
3. Distance to $q = 0$: $d_{q,0} = \frac{1}{2(1+\delta)}$
4. Distance to $q = 1$: $d_{q,1} = \frac{1+2\delta}{2(1+\delta)}$

Comparing these four values for small $\delta > 0$, the minimum distance is governed by the $p = 1$ boundary strategy of the $x$-player:
$$ d_{\text{min}} = \frac{\delta}{1+\delta} $$
To guarantee that the circle does not cross the boundaries of the strategy space and degenerate into pure strategies, we must enforce $r < d_{\text{min}}$. We select the sweep radius:
$$ r = 0.5 \frac{\delta}{1+\delta} $$
This choice provides a safety clearance factor of exactly $0.5$ from the closest boundary wall. 

For each point $(\lambda, \gamma)$ along this circle, we construct the canonical game matrix $A_{\lambda, \gamma}$ defined in Section 4.2. Because the canonical matrix $A_{\lambda, \gamma}$ maintains a linear, decoupled mapping to the equilibrium coordinates ($p = 1-\gamma$, $q = 1-\lambda$), sweeping a perfect geometric circle in the parameter space $(\lambda, \gamma)$ generates an undistorted geometric circle of Nash Equilibria in the strategy space $\Delta^2 \times \Delta^2$. This linear correspondence prevents coordinate distortion, ensuring that the concentric radial distance from the boundary is controlled uniformly throughout the sweep.

---

## 2. 1D Strategy Tracking Analysis
To contrast the behavioral dynamics of OMWU and OGDA along this boundary-adjacent curve, we first evaluate the evolution of the strategies in a 1D tracking projection. Figure 3 presents the computed last-iterate strategy coordinates played by both algorithms at the end of $T = 10,000$ steps against the theoretical Nash Equilibrium values across the circular parameter sweep.

The results show a clear separation between the two algorithms:
* **OGDA Strategy Tracking:** The computed last-iterate strategies of OGDA track the theoretical Nash Equilibrium coordinates precisely across the entire parameter sweep. This confirms that OGDA's last-iterate converges to the target equilibrium even when positioned close to the simplex boundaries.
* **OMWU Strategy Tracking:** The computed last-iterate strategies of OMWU fail to track the theoretical equilibrium, exhibiting large-amplitude, persistent oscillations away from the optimal coordinates. 

This tracking failure demonstrates that last-iterate convergence for OMWU is systematically compromised in boundary-adjacent regimes.

---

## 3. 2D Phase-Space Representation
To analyze the geometric structure of OMWU's tracking failure, we transition from the one-dimensional tracking projection to the joint strategy space $\Delta^2 \times \Delta^2$. For a $2 \times 2$ zero-sum game, each player's strategy simplex is 1-dimensional ($x_2 = 1 - x_1$ and $y_2 = 1 - y_1$). Consequently, the joint strategy space is represented geometrically by the 2D unit square $[0, 1] \times [0, 1]$ with coordinates $(x_1, y_1)$.

Figure 4 (left) illustrates the trajectory of OMWU's strategies in this 2D plane for an ill-conditioned parametric instance. The trajectory does not contract toward the theoretical Nash Equilibrium. Instead, it enters a stable limit cycle, forming a closed orbital path around the equilibrium. 

In continuous-time replicator dynamics (the continuous limit of Multiplicative Weights Update), the Kullback-Leibler (KL) divergence to the Nash Equilibrium is a strict invariant of motion, analogous to the conservation of energy in a Hamiltonian physical system. The persistent limit cycle observed here represents the discrete-time numerical manifestation of this conservation property. Because the discrete OMWU updates fail to contract the divergence to the equilibrium in the boundary region, the strategies are forced into permanent, non-decaying orbits.

---

## 4. $L^2$ Distance Dynamics
To quantify this orbital behavior over the course of the simulation, we evaluate the $L^2$ distance between the strategy profile $z^t = (x^t, y^t)$ and the unique Nash Equilibrium $z^* = (x^*, y^*)$ over time:
$$ \|z^t - z^*\|_2 = \sqrt{(x^t_1 - x^*_1)^2 + (y^t_1 - y^*_1)^2} $$

The evolution of the $L^2$ distance, presented in Figure 4 (right), exhibits periodic, non-decaying fluctuations:
* **Periodic Drops:** The sharp drops in the $L^2$ distance occur during the segments of the orbit where the strategy trajectory passes directly through the immediate neighborhood of the Nash Equilibrium.
* **Broad Peaks:** The wide peaks correspond to the outer segments of the orbit furthest from the solution, where the strategy is pushed close to the simplex boundaries.

This periodic proximity provides the geometric explanation for the separation of convergence modes: the trajectory repeatedly passes close to the Nash Equilibrium (yielding a small best-iterate gap) but fails to stabilize, entering a stable limit cycle (yielding a large, oscillating last-iterate gap).

---

## 5. Diagnostic Verification of the Limit Cycle
To confirm mathematically that the observed orbit is a stable, persistent limit cycle rather than a slow, decaying transient phase, we compute the variance of the duality gap over the final 10% of iterations (iterations $9,000$ to $10,000$). 

For OMWU, the variance remains strictly bounded away from zero ($\text{Var} > C > 0$) as $T \to \infty$, demonstrating that the oscillation amplitude does not decay. This persistent variance confirms that the boundary pathology is a robust convergence anomaly linked to OMWU's Legendre-regularized non-forgetful updates, which accumulate historical gradients without a forgetting mechanism.
