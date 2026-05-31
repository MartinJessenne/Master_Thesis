#set document(title: "Master Thesis: Convergence in Ill-Conditioned Matrix Games", author: "Martin Jessenne")

#set page(
  paper: "a4",
  margin: (x: 2.5cm, y: 3cm),
  header: align(right)[_Master Thesis - Martin Jessenne_],
  numbering: "1",
)

#set text(
  font: "New Computer Modern",
  size: 11pt,
  lang: "en",
)

#set heading(numbering: "1.1")

// --- FRONT PAGE ---
#align(center)[
  #v(4cm)
  #text(size: 20pt, weight: "bold", fill: rgb("1f3c88"))[Master Thesis]
  #v(1cm)
  #text(size: 26pt, weight: "black")[On the Convergence Modes of Learning in Games: 
  
  Best, Random, and Last-Iterate Convergence]
  #v(1cm)
  #text(size: 16pt, style: "italic")[An Empirical and Theoretical Analysis in Ill-Conditioned Settings]
  
  #v(4cm)
  #text(size: 16pt)[*Author:* Martin Jessenne]

  #text(size: 16pt)[*Supervisor:* Julien Grand-Clément]
  #v(1cm)
  #text(size: 14pt)[*Date:* April 2026]
  
]

#pagebreak()

// --- TABLE OF CONTENTS ---
#outline(
  title: "Table of Contents",
  depth: 3,
  indent: auto,
)

#pagebreak()

// --- THESIS CONTENT START ---
#set page(numbering: "1")
#counter(page).update(1)

= Introduction

== Context
In applications ranging from economics to artificial intelligence, decision-makers, or "agents", interact within shared environments. When agents compete or cooperate, these interactions are modeled as games. A central concept in game theory is the Nash Equilibrium, defined as a state where no player can improve their outcome by unilaterally changing their strategy. In a two-player zero-sum game, this equilibrium corresponds to a pair of strategies $(x^*, y^*)$ that form a stable saddle point.

To identify these equilibria, learning dynamics are employed, where agents repeatedly adjust their strategies based on observed outcomes. One class of such algorithms is "no-regret learning," which provides long-term performance guarantees relative to the best fixed strategy in hindsight. Historically, the convergence of these algorithms has been analyzed in an ergodic (or average) sense, where the average of the strategies played over time approaches the Nash Equilibrium $(x^*, y^*)$. 

In recent years, research focus has shifted toward the non-ergodic behavior of these dynamics, specifically last-iterate convergence. This mode of convergence requires the step-by-step trajectory of the strategies to approach the equilibrium directly. Establishing last-iterate convergence is theoretically relevant for preventing phenomena such as recurrence, cyclic trajectories, or non-convergent behavior in multi-agent learning systems.

== Motivation
The requirement for non-ergodic convergence is observed in machine learning applications, such as the training of Generative Adversarial Networks (GANs) and multi-agent Reinforcement Learning (MARL). In these settings, agents are represented by non-linear models, such as deep neural networks. 

The resolution of competitive scenarios in optimization and game theory is fundamentally structured around identifying the saddle point of an objective function $op("min")_(x in cal(X)) op("max")_(y in cal(Y)) Phi(x, y)$. Historically, the development of no-regret learning dynamics provided convergence guarantees primarily in an ergodic sense, where the time-averaged sequence of strategies, $overline(z)^T = 1/T sum_(t=1)^T (x^t, y^t)$, asymptotically approaches a Nash Equilibrium. This ergodic paradigm is theoretically sufficient in convex-concave landscapes because Jensen’s inequality ensures that the performance of the averaged parameters is rigorously bounded by the average functional performance:
$ Phi(1/T sum_(t=1)^T x^t, y^*) <= 1/T sum_(t=1)^T Phi(x^t, y^*) $
In such settings, deploying the mathematical average of the iterates yields a theoretically robust and functionally stable equilibrium solution.

However, modern architectures in Generative Adversarial Networks (GANs) and multi-agent reinforcement learning (MARL) introduce non-convex objective surfaces where the lack of convexity invalidates the foundational premise of Jensen's inequality. In these high-dimensional regimes, the mathematical average of two high-performing parameter sets: $theta_1$ and $theta_2$ such that $L(theta_1)$ and $L(theta_2)$ are both small, does not correspond to a functional "average" model; instead, the resulting average $(theta_1 + theta_2)/2$ typically resides in a region of the parameter space characterized by high loss. This geometric reality necessitates a transition toward last-iterate (iteration-wise) convergence. Ensuring that the actual sequence of iterates $(x^T, y^T)$ converges directly to the local saddle point is the only viable mechanism for deploying models with guaranteed functional properties in non-convex environments. Additionally, iteration-wise analysis is required to detect dynamic instabilities, such as limit cycles or oscillatory orbits, which may be concealed by ergodic averages.

While algorithms like Optimistic Gradient Descent-Ascent (OGDA) have been shown to achieve uniform last-iterate convergence, other algorithms, such as Optimistic Multiplicative Weights Update (OMWU), can exhibit slow last-iterate convergence. This raises the question of whether OMWU can satisfy alternative criteria under weaker convergence modes. We investigate three convergence modes:
- *Last-iterate convergence:* The proximity of the final state produced at the end of the training process to the equilibrium.
- *Random-iterate convergence:* The expected duality gap of a strategy chosen uniformly at random from the training history.
- *Best-iterate convergence:* The existence of at least one state during the training process that is close to the equilibrium.

== Goal
This thesis aims to empirically evaluate and theoretically analyze the separation between last-iterate, random-iterate, and best-iterate convergence modes, with a focus on the OMWU algorithm. Through the study of $2 times 2$ matrix games where the Nash equilibrium is located near the boundary of the probability simplex, we will examine the stability of OMWU's last iterate relative to its best-iterate convergence rate. We introduce a computational framework in Rust and Python designed for numerical stability and parallelized exploration. The objective is to characterize the non-convergent behaviors of OMWU.

= Theoretical Framework & Literature Review

== The Shift in Convergence Studies
Historically, the analysis of learning dynamics such as Optimistic Multiplicative Weights Update (OMWU) and Optimistic Gradient Descent-Ascent (OGDA) centered on their ergodic properties. Both algorithms are noted for achieving $O(1/T)$ convergence rates to a Nash Equilibrium in the average iterate. 

The requirements for training non-linear models have led to an increased interest in *last-iterate* properties to ensure the final state is close to an equilibrium. 

In the analysis of non-ergodic convergence, OGDA has been shown to achieve a $O(1/sqrt(T))$ uniform last-iterate convergence rate @cai2025fastconvergence. Conversely, OMWU's performance in this context has been subject to limitations. Recent results by @cai2025fastconvergence indicate that a class of Optimistic Follow-The-Regularized-Leader (OFTRL) algorithms, including OMWU, do not achieve a uniform last-iterate convergence rate. Specifically, in parameterized $2 times 2$ matrix games, OMWU exhibits a constant duality gap for a number of iterations proportional to $1/delta$, where $delta$ is the minimum non-zero probability in the Nash Equilibrium. Since $delta$ can be arbitrarily small, this dependency precludes uniform last-iterate convergence.

This leads to the investigation of whether OMWU satisfies weaker convergence criteria. A study by @cai2025separation established a separation between convergence modes, showing that while OMWU does not achieve polynomial *random-iterate* convergence (exhibiting an $Omega(1/(log T))$ lower bound), it achieves a uniform $O(T^(-1/6))$ *best-iterate* convergence rate in $2 times 2$ matrix games. This separation forms the motivation for our work: to empirically analyze these behaviors and quantify the conditions that prevent last-iterate convergence while evaluating the performance of the best iterate.

== Algorithms
The primary algorithms examined in this thesis are implemented in the `optigame` Rust library and Python framework.

*Optimistic Follow-The-Regularized-Leader (OFTRL)* 
OFTRL is a family of no-regret online learning algorithms defined by a regularizer $R$. Given cumulative loss vectors $L_x^(t-1)$ and per-step losses $l_x^(t-1)$, the update rule for the $x$-player with step size $eta$ is:
$ x^t = op("argmin")_(x in Delta^(d_1)) { chevron.l x, L_x^(t-1) + l_x^(t-1) chevron.r + 1/eta R(x) } $
The dependence on the history of cumulative losses ($L_x^(t-1)$) results in the "non-forgetfulness" property, which can lead to extended periods of sub-optimal strategy selection @cai2025fastconvergence.

*Optimistic Multiplicative Weights Update (OMWU)* \
OMWU is an instance of OFTRL (and Optimistic Online Mirror Descent) using the negative entropy regularizer $R(x) = sum_i x[i] log x[i]$. In theoretical settings involving Legendre regularizers, the OFTRL and OOMD updates are equivalent. The `optigame` framework provides the two implementations, `OmwuOftrl` and `OmwuOomd`.

*Optimistic Gradient Descent-Ascent (OGDA)* \
OGDA uses the squared Euclidean norm regularizer $R(x) = 1/2 sum_i x[i]^2$. OGDA achieves a $O(1/sqrt(T))$ uniform random-iterate and last-iterate convergence rate @cai2025fastconvergence. It serves as a baseline for comparison with OMWU's oscillatory behavior.

== Formal Definitions
To evaluate these algorithms, the following definitions are used.

*Game Setup & Nash Equilibrium* \
We consider two-player zero-sum matrix games defined by a loss matrix $A in [0, 1]^(d_1 times d_2)$. Players select strategies $x in Delta^(d_1)$ and $y in Delta^(d_2)$ from the probability simplex $Delta^d$, representing probability distributions over actions. A Nash Equilibrium $(x^*, y^*)$ is a state where neither player can improve their outcome by unilaterally changing their strategy. We denote $delta > 0$ as the minimum probability assigned to any action in the equilibrium.

*Duality Gap* \
The proximity of a strategy profile $(x, y)$ to the Nash equilibrium is measured by the duality gap:
$ op("Gap")(x,y) = max_(y' in Delta^(d_2)) (x^T A y') - min_(x' in Delta^(d_1)) (x'^T A y) $
The duality gap is non-negative and equals zero if and only if $(x, y)$ is a Nash equilibrium.

*Modes of Convergence* \
We consider three non-ergodic convergence modes, where $f(T)$ is a rate function approaching zero as $T$ increases:
- *Last-iterate convergence:* The final state satisfies $op("Gap")(x^T, y^T) <= O(f(T))$.
- *Random-iterate convergence:* The expected duality gap of a uniformly selected iterate satisfies $1/T sum_(t=1)^T op("Gap")(x^t, y^t) <= O(f(T))$.
- *Best-iterate convergence:* The minimum observed duality gap satisfies $min_(t in [1,T]) op("Gap")(x^t, y^t) <= O(f(T))$.

*Uniform vs. Universal Rates* \
A uniform rate bounds convergence speed as a function of $T$ and game dimensions, independent of specific game properties. A universal rate may depend on game-specific constants, such as $delta$ @cai2025separation.

= Methodology & Computational Framework

== Implementation & Performance analysis
To evaluate OMWU's behavior across parameterized matrices, we developed `optigame`, a high-performance framework. Pure Python implementations for sweeping dense grids of game parameters (e.g., $10,000$ steps across a $500 times 500$ grid) were found to be inefficient due to interpreter overhead and the Global Interpreter Lock (GIL). 

`optigame` uses a Rust core with a Python interface via PyO3. This architecture permits safe mathematical operations and thread-level parallelism. Concurrency techniques are used to distribute neighborhood exploration across available CPU cores. This approach results in significant speedups relative to pure Python implementations, enabling efficient analysis and plotting in Python environments.

#v(4cm)
#align(center)[
#figure(
  image("perf_benchmark.png", width: 80%),
  caption: [
  Benchmark results showing performance improvements after migrating the computation logic to Rust in the setting of 500 runs of 10000 iterations each.
  ],
)
]
#v(2cm)

== Numerical Stability & Implementation Details
OMWU and OFTRL use exponential updates determined by the step size $eta$ and cumulative gradients. In settings where strategies approach the simplex boundary, these terms can lead to numerical overflow. 

We implemented two safeguards for stability:
1. *Shifted Exponentials:* During multiplicative updates, we subtract the maximum component of the gradient vector from all elements before exponentiation (e.g., $exp(s - max(s))$). This manipulation is known as the Log-Sum-Exp trick in other machine learning set up, such as cross-entropy computation in deep-learning.   
2. *Simplex Projections:* For algorithms that do not natively maintain probability distributions, such as OGDA, we use an $O(n log n)$ projection onto the probability simplex to ensure iterates satisfy $x in Delta^(d_1)$ and $y in Delta^(d_2)$.

=== Zero-Allocation Hot Loop
To optimize throughput during large-scale simulations, the `random_neighborhood_exploration` and grid search routines utilize a zero-allocation design in the core loop. Initializing matrices and state vectors in memory is a computationally expensive operation when performed repeatedly across thousands of experiments. Our implementation utilizes pre-allocated buffers for game matrices and strategy states, which are overwritten in place for each trial. By passing matrix views (references) instead of copying the matrix data, we reduce memory allocation overhead and improve cache locality, resulting in performance improvements over naive allocation strategies.

== Quantifying Divergence
To evaluate convergence behavior and distinguish between slow decay and periodic behavior, we introduce three trajectory-based metrics:
1. *Maximum Value of the Last 10% Iterates:* Measures the peak amplitude of oscillations near the end of the simulation.
2. *Variance of the Last 10% Iterates:* Quantifies the stability of the final iterates; high variance indicates persistent oscillations.
3. *Total Variation:* Computed as $sum_t |op("Gap")^(t) - op("Gap")^(t-1)|$, this metric quantifies the total oscillation amplitude across the trajectory. 

These metrics provide a quantitative basis for mapping the stability of the parameter space and characterizing OMWU's last-iterate behavior.


= Empirical Study: Convergence in Ill-Conditioned Matrix Games

In this chapter, we deploy our computational framework to empirically investigate the theoretical bounds discussed previously. We aim to map the non-convergent behavior of OMWU in ill-conditioned games and validate the theoretical separation between last-iterate convergence and other modes (best/random).

== The Baseline Hard Instance & Random Neighborhood Exploration

To establish an empirical baseline, we replicate the non-convergent behavior documented by @cai2025fastconvergence. The baseline game is defined by a $2 times 2$ zero-sum matrix:
$ A_delta = mat(1/2 + delta, 1/2; 0, 1) $
where $delta > 0$ represents the minimum probability assigned to any action in the unique Nash Equilibrium $(x^*, y^*)$, given by $x^* = (1/(1+delta), delta/(1+delta))$ and $y^* = (1/2, 1/2)$. As $delta -> 0$, the optimal strategy for the $x$-player is positioned arbitrarily close to the boundary of the probability simplex $Delta^2$. In this setting, the last-iterate duality gap of OMWU fails to converge to zero, instead entering a persistent cyclic phase.

To determine whether this pathological behavior is a fragile artifact of the exact matrix coefficients or a robust phenomenon, we perform a neighborhood analysis. We apply uniform random perturbations to the baseline matrix $A_delta$, defining:
$ A_("perturbed") = A_delta + epsilon U $
where $U$ is a perturbation matrix sampled uniformly in $B_infinity (0,1)$, and $epsilon > 0$ scales the perturbation magnitude, for this entire part, we set $delta = 0.01$.

To map the spatial characteristics of this instability, we design a concentric shell exploration. For $(epsilon_i)_(i in [1, n])$ such that $0 < epsilon_1 < ... < epsilon_n = epsilon$, the perturbation domain is partitioned into concentric $L_infinity$ shells $[epsilon_(i-1), epsilon_i]$. Within each concentric band, we sample the same number of matrices, in the @concentric_exploration_OMWU set-up we sampled 1,000 independent matrix for each shell, and for each matrix the algorithm ran for 10,000 steps. 

#v(1cm)
#align(center)[
#figure(
  image("images\mean_concentric_results_max_last_10_OMWU.svg", width: 80%),
  caption: [Distribution of the maximum duality gap over the last 10% of iterations for OMWU, across different perturbation magnitudes $epsilon$.]
)<concentric_exploration_OMWU>
]
#v(1cm)

The results, presented in the meanplot above, indicate that adding a uniform perturbation to $A_delta$ does not immediately induce convergence. For small values of $epsilon$, the median of the last-iterate duality gap remains elevated, matching the baseline $A_delta$ behavior. This confirms that the non-convergent behavior is robust to small numerical perturbations. 

However, as $epsilon$ increases, the median duality gap decays, indicating that the pathology is a local anomaly. Larger perturbations shift the Nash Equilibrium away from the extreme boundaries of the simplex toward the interior, restoring convergence.

To verify that this behavior is specific to OMWU, we execute the same concentric exploration using Optimistic Gradient Descent-Ascent (OGDA).

#v(1cm)
#align(center)[
#figure(
  image("images\mean_concentric_results_max_last_10_OGDA.svg", width: 80%),
  caption: [Distribution of the maximum duality gap over the last 10% of iterations for OMWU, across different perturbation magnitudes $epsilon$.]
)<concentric_exploration_OGDA>
]
#v(1cm)

 In contrast to OMWU, OGDA, which employs a quadratic Euclidean regularizer, exhibits no non-convergent instabilities. Across all perturbation shells, the last-iterate of OGDA converges to the perturbed equilibrium, maintaining a near-zero last-iterate duality gap. This comparison suggests that the instability is an intrinsic property of OMWU's regularizer structure, rather than a general property of learning in zero-sum games. This localized boundary instability motivates a systematic study using a custom parametric family of games to map the entire simplex.

== Parametric Game Design ($A_(lambda, gamma)$) & Normalization

To exhaustively test OMWU's sensitivity to the minimum equilibrium probability $delta$, we require the ability to construct arbitrary game matrices with a known, predetermined Nash Equilibrium $(x^*, y^*)$. 

Given a $2 times 2$ matrix $A = mat(a, b; c, d)$, the Nash equilibrium $(x^*, y^*)$ must satisfy the indifference conditions. For the $y$-player facing $x^*$, and the $x$-player facing $y^*$, these conditions are respectively:
$ (x^*)^T A mat(1; 0) = (x^*)^T A mat(0; 1) quad "and" quad mat(1, 0) A y^* = mat(0, 1) A y^* $

Letting $x^* = (p, 1-p)$ and $y^* = (q, 1-q)$, this yields the system of equations:
$ cases(
  a p + c(1-p) = b p + d(1-p),
  a q + b(1-q) = c q + d(1-q)
) $

Solving for $p$ and $q$, we obtain explicit expressions for the Nash equilibrium based on the matrix coefficients:
$ p = (d-c)/(a-b-c+d) quad "and" quad q = (d-b)/(a-b-c+d) $

To generate a parameterized family of games whose Nash equilibria can be freely moved, we map the equilibrium probabilities to independent parameters $lambda$ and $gamma$. Letting $S = a - b - c + d$ be the scaling factor, we have:
$ p = (d-c)/S quad "and" quad q = (d-b)/S $

We have one remaining degree of freedom. By arbitrarily fixing $d=1$, we can map the off-diagonal elements directly to our parameters by setting $b = lambda$ and $c = gamma$. Substituting these into the scaling factor equation yields $a = S - 1 + lambda + gamma$. This gives the formulation for the parameterized matrix family $A_(lambda, gamma, S)$:
$ A_(lambda, gamma, S) = mat(
  S - 1 + lambda + gamma, lambda;
  gamma, 1
) $

This construction guarantees that the unique Nash equilibrium is exactly $(p, 1-p)$ and $(q, 1-q)$, where:
$ p = (1-gamma)/S quad "and" quad q = (1-lambda)/S $

=== The Degrees of Freedom in $2 times 2$ Zero-Sum Games
Although the matrix $A_(lambda, gamma, S)$ is defined using three parameters $(lambda, gamma, S)$, game-theoretic dynamics dictate that a $2 times 2$ zero-sum game possesses exactly two strategic degrees of freedom. This is due to two structural invariances under positive affine transformations of the payoffs:

1. *Translation Invariance:* Adding a constant to all entries in a column alters the expected payoff of the column player but preserves the relative payoff differences. The Nash Equilibrium is invariant under column translations.
2. *Scaling Invariance:* Multiplying all matrix coefficients by a positive scalar $alpha > 0$ scales the expected utilities but preserves all relative preference inequalities.

Because of these invariances, one of the three parameters in the family $A_(lambda, gamma, S)$ is strategically redundant. The coordinates of the Nash Equilibrium $(p, q)$ occupy a two-dimensional space inside the simplex interior, meaning they can be fully parameterized using only two variables.

=== Canonical Form ($S = 1$)
To resolve this scaling redundancy, we enforce a scaling normalization constraint by setting:
$ S = 1 $

Enforcing $S = 1$ resolves the redundancy and yields the canonical $2 times 2$ matrix family defined purely by the two strategic parameters $lambda$ and $gamma$:
$ A_(lambda, gamma) = mat(
  lambda + gamma, lambda;
  gamma, 1
) $

This canonical matrix provides a linear, decoupled mapping directly to the Nash Equilibrium coordinates:
$ p = 1 - gamma quad "and" quad q = 1 - lambda $

=== Generalization to the $A_delta$ Baseline
The general 3-parameter derivation is sufficiently robust to recover the baseline game $A_delta$ studied in the literature. 

Instead of setting $S = 1$, the $A_delta$ baseline is recovered by freezing the off-diagonal coordinates at $gamma = 0$ and $lambda = 1/2$. In this coordinate projection, the scale $S$ is retained as the single free variable. By writing $S = 1 + delta$, the general matrix $A_(lambda, gamma, S)$ reduces to:
$ A_(1/2, 0, 1+delta) = mat(
  (1+delta) - 1 + 1/2 + 0, 1/2;
  0, 1
) = mat(
  1/2 + delta, 1/2;
  0, 1
) = A_delta $

This demonstrates that both $A_delta$ (a 1-parameter family sweeping one boundary) and our canonical $A_(lambda, gamma)$ (a 2-parameter family sweeping both boundaries simultaneously) are mathematically rigorous projections of the same general parameterized game family.

== Comparative Analysis of Parameterizations
The $A_(lambda, gamma)$ parameterization was developed to facilitate a systematic exploration of the simplex boundaries while maintaining numerical robustness. To validate this approach, its performance is compared with the matrix $A_("sep")$, derived from the construction presented in Lemma 5 of @cai2025separation:
$ A_("sep") = mat(
  (1 - delta_y) / (1 - delta_x), (1 - delta_x - delta_y) / (1 - delta_x);
  0, 1
) $
where $delta_x$ and $delta_y$ denote the minimum action probabilities at equilibrium. This matrix serves as a standard benchmark for evaluating the separation between convergence modes in $2 times 2$ games.

#v(2cm)
#align(center)[
#figure(
  image("images/Comparaison_A_lambda_gamma_lemma5.png", width: 80%),
  caption: [Comparison of the $L^2$ distance to the Nash Equilibrium over time for the $A_(lambda, gamma)$ parameterization versus the $A_("sep")$ benchmark.]
)
]
The results demonstrate that both $A_(lambda, gamma)$ and $A_("sep")$ exhibit qualitatively similar non-ergodic profiles, characterized by persistent oscillations and periodic spikes in the distance to the Nash Equilibrium. This confirmation supports the use of $A_(lambda, gamma)$ as a valid generalization of the literature baseline.

However, the comparison reveals numerical discrepancies during the optimization of $A_("sep")$. Specifically, as the equilibrium approaches the boundary ($delta_x, delta_y -> 0$), the coefficients of $A_("sep")$ can lead to numerical instability, occasionally resulting in floating-point overflows or undefined values (NaNs). In contrast, the $A_(lambda, gamma)$ parameterization, when used in conjunction with the affine normalization described in Section 4.2, preserves the non-convergent behavior while ensuring reliable computation across a broader range of the parameter space. This analysis indicates that the observed discrepancies are artifacts of the numerical implementation of the baseline matrix rather than fundamental differences in game dynamics.

== Strategy Profiles and Orbital Dynamics
The random concentric shell explorations in Section 4.1 establish the robustness and local nature of OMWU's boundary anomaly. However, random perturbations do not provide a systematic way to map how game dynamics change as the equilibrium approaches the boundaries.

=== Deterministic Parametric Sweeps
To analyze the spatial layout of this boundary anomaly with high precision, we transition from random perturbations to a deterministic sweep along a parameterized curve in the strategy space. The sweep is designed as a circle centered at the baseline $A_delta$ Nash Equilibrium (with $delta = 0.01$). The Nash Equilibrium coordinates $(p_0, q_0)$ at the center of the sweep are:
$ p_0 = 1/(1+delta) quad "and" quad q_0 = 1/(2(1+delta)) $

To ensure that the circular sweep remains strictly within the interior of the joint strategy space $Delta^2 times Delta^2$ (represented geometrically by the unit square $K = [0, 1] times [0, 1]$ with coordinates $z = (p, q)$), the sweep radius $r$ must be strictly smaller than the Euclidean ($L^2$) distance from the sweep center $z_0 = (p_0, q_0)$ to the boundary $partial K$:
$ d_2(z_0, partial K) = inf_(w in partial K) norm(z_0 - w)_2 $

As derived in @appendix:boundary_distances, we find that the Nash Equilibrium is at a minimum distance $d_("min") = delta/(1+delta) $ from the boundaries of the strategy space. 

To guarantee that the parametric sweep circle does not cross the boundaries and degenerate into pure strategies, we must enforce $r < d_("min")$. We select the sweep radius:
$ r = 1/2 times delta/(1+delta) $

The circular trajectory is parameterized by the polar angle $theta in [0, 2pi)$, where:
$ p(theta) = p_0 + r cos(theta) quad "and" quad q(theta) = q_0 + r sin(theta) $
For each point $(p(theta), q(theta))$ along this circle, we construct the corresponding canonical game matrix $A_(lambda, gamma)$ defined in Section 4.2 by setting $lambda = 1 - q(theta)$ and $gamma = 1 - p(theta)$. 

=== 1D Strategy Tracking Analysis
To contrast the behavioral dynamics of OMWU and OGDA along this boundary-adjacent curve, we first evaluate the evolution of the strategies in a 1D tracking projection. Figure 3 presents the computed last-iterate strategy coordinates played by both algorithms at the end of $T = 10,000$ steps against the theoretical Nash Equilibrium values across the circular parameter sweep.

The results show a clear separation between the two algorithms:
- *OGDA Strategy Tracking:* The computed last-iterate strategies of OGDA track the theoretical Nash Equilibrium coordinates precisely across the entire parameter sweep. This confirms that OGDA's last-iterate converges to the target equilibrium even when positioned close to the simplex boundaries.
- *OMWU Strategy Tracking:* The computed last-iterate strategies of OMWU fail to track the theoretical equilibrium, exhibiting large-amplitude, persistent oscillations away from the optimal coordinates. 

This tracking failure demonstrates that last-iterate convergence for OMWU is systematically compromised in boundary-adjacent regimes.

#v(1cm)
#align(center)[
#figure(
  grid(
    rows: 2,
    gutter: 1em,
    image("images/OGDA_Last_xo_vs_ne.svg", width: 100%),
    image("images/OMWU_last_xo_vs_ne.svg", width: 100%),
  ),
  caption: [1D Strategy Tracking. Evolution of computed last-iterate strategies against theoretical Nash Equilibrium probabilities for OGDA (top) and OMWU (bottom) across the circular parameter sweep.],
)<fig:1d_tracking>
]
#v(1cm)

=== 2D Strategy-Space Representation
To analyze the geometric structure of OMWU's tracking failure, we transition from the previous one-dimensional analysis where we compared the last iterate strategy to the theoretical Nash equilibrium to the two-dimensional joint strategy space $Delta^2 times Delta^2$ to study the evolution of the computed strategies along the circle. 

@fig:2d_trajectory_evolution illustrates the evolution of OMWU's strategies in this 2D plane for an ill-conditioned parametric instance. This visualization demonstrates that the oscillatory behavior is pronounced when the theoretical equilibrium is closest to the boundary of the strategy space (at $theta = pi$), while when it is positioned furthest from the boundary (at $theta = 0$), the strategies orbit closer to the Nash equilibrium. 

#v(1cm)
#align(center)[
#figure(
  grid(
    columns: 2,
    rows: 2,
    gutter: 1.5em,
    image("images/2d_profiles/2d_profile_000.svg", width: 90%),
    image("images/2d_profiles/2d_profile_125.svg", width: 90%),
    image("images/2d_profiles/2d_profile_250.svg", width: 90%),
    image("images/2d_profiles/2d_profile_375.svg", width: 90%),
  ),
  caption: [Evolution of OMWU's 2D strategy trajectories in the joint strategy space $Delta^2 times Delta^2$ (geometrically represented by the unit square $[0,1] times [0,1]$) at different points along the circular parametric sweep: (a) $theta = 0$ (top-left), (b) $theta = pi/2$ (top-right), (c) $theta = pi$ (bottom-left), and (d) $theta = 3pi/2$ (bottom-right). The black dot represents the static baseline Nash Equilibrium center, the red 'X' represents the $(0.5,0.5)$ starting strategy profile, and the green circle represents the moving theoretical Nash Equilibrium for that specific sweep parameter.],
)<fig:2d_trajectory_evolution>
]
#v(1cm)

== Best vs. Last Iterate Comparison

=== Theoretical Separation of Convergence Rates
Building upon the formal definitions established in Section 2.3, @cai2025separation proved a uniform complexity separation between the three non-ergodic convergence modes for OMWU in $2 times 2$ matrix games:
- *Last-Iterate:* Non-uniform, exhibiting an $Omega(1)$ duality gap. 
- *Random-Iterate:* Non-uniform, failing to achieve polynomial convergence and exhibiting a logarithmic lower bound of $Omega(1 / (log T))$.
- *Best-Iterate:* Uniform, achieving a convergence rate of $O(T^(-1/6))$ that is independent of the boundary proximity $delta$.

This separation establishes that while OMWU does not stabilize at the Nash Equilibrium (last-iterate), its trajectory is guaranteed to repeatedly achieve close proximity to it (best-iterate).

=== Last-iterate, Random-iterate and Best-iterate separation study
To validate these bounds, we analyze the empirical convergence of OMWU on a log-log scale across $T=10,000$ iterations. We map these results to the exact same four sweep points studied in the 2D orbital analysis (Section 4.4).

#v(1cm)
#align(center)[
#figure(
  grid(
    columns: 2,
    rows: 2,
    gutter: 1.5em,
    image("images/OMWU_Convergence_Modes_Separation_step_0.svg", width: 90%),
    image("images/OMWU_Convergence_Modes_Separation_step_125.svg", width: 90%),
    image("images/OMWU_Convergence_Modes_Separation_step_250.svg", width: 90%),
    image("images/OMWU_Convergence_Modes_Separation_step_375.svg", width: 90%),
  ),
  caption: [Empirical separation of OMWU's convergence modes on a log-log scale for the identical four circular sweep points: (a) $theta = 0$ (top-left), (b) $theta = pi/2$ (top-right), (c) $theta = pi$ (bottom-left), and (d) $theta = 3pi/2$ (bottom-right). The last-iterate gap (red) oscillates indefinitely; the random-iterate gap (orange) decays very slowly; the best-iterate gap (green) converges rapidly, remaining strictly bounded by the theoretical $O(T^(-1/6))$ worst-case uniform envelope (black dashed).],
) <fig:convergence_separation_grid>
]
#v(1cm)

The empirical profiles in @fig:convergence_separation_grid reveal a direct interpretation with the 2D orbital trajectories in @fig:2d_trajectory_evolution:
- *Last-Iterate (Red):* Oscillates indefinitely in a stable, non-decaying limit cycle. The amplitude of these oscillations is directly proportional to the eccentricity of the 2D orbit, reaching its maximum at $theta = 0$ where the orbit is stretched closest to the simplex boundaries.
- *Random-Iterate (Orange):* Exhibits a nearly flat, logarithmic decay. This is because the strategy trajectory spends the majority of its orbital period in the high-gap outer boundaries of the limit cycle.
- *Best-Iterate (Green):* Converges rapidly in a staircase pattern. Each horizontal step represents the time spent orbiting the outer cycle, while the step-like drops occur precisely when the iteration-wise trajectory makes its closest approach to the Nash Equilibrium.

To validate the theoretical upper bound, we fit and plot the theoretical envelope $C dot t^(-1/6)$ (black dashed). Across all sweep points, the best-iterate remains strictly below this worst-case boundary. Furthermore, the empirical log-log slope of the best iterate is steeper than $-1/6$, demonstrating that OMWU's practical best-iterate performance is in accordance with the claims in @cai2025separation.



= Conclusion

== Synthesis of Empirical Findings

This work investigated the iteration-wise convergence properties of Optimistic Multiplicative Weights Update (OMWU) in zero-sum games, with a particular focus on the mathematical separation between last-iterate, random-iterate, and best-iterate convergence modes. While theoretical results established by @cai2025separation prove a uniform best-iterate rate of $O(T^(-1/6))$ for $2 times 2$ matrix games, empirical study of such boundary-adjacent dynamics requires extensive numerical simulations over vast parametric spaces and numerous iterations.

The primary engineering contribution of this study is the design and implementation of a high-performance optimization framework in Rust (`optigame`), which is publicly available and installable with a simple `pip install` command for reproducibility's sake. By leveraging zero-allocation and parallelized loops and an efficient Foreign Function Interface (FFI) bridge to Python, the computational throughput was increased by several orders of magnitude compared to a pure Python implementation. This performance enhancement enabled the execution of dense deterministic sweeps and high-fidelity strategy tracking that would otherwise be computationally prohibitive. 

Using this framework, we mapped the strategy space with high resolution, confirming on the one hand that the chaotic behavior witness in OMWU for the $A_delta$ matrix is robust, and on the other hand that the empirical best-iterate convergence rate under OMWU respects the theoretical worst-case bound across all boundary-adjacent configurations.

== Practical Implications and Limitations

From an algorithmic perspective, the uniform convergence of the best-iterate provides a crucial mathematical guarantee, establishing that OMWU does not permanently diverge or stagnate far from optimal strategies. However, translating this theoretical guarantee into practical machine learning applications reveals a clear engineering trade-off.

Unlike last-iterate convergence, where the final parameters $w^T$ are directly deployed, utilizing the best-iterate requires identifying the optimal candidate from the history of iterates $\{w^t\}_{t=1}^T$. In practical settings, evaluating the duality gap or validation accuracy at every optimization step incurs substantial computational overhead, especially in large-scale datasets or high-dimensional models settings. Consequently, while the best-iterate guarantee serves as a theoretical safety net proving the existence of close-to-Nash equilibrium states along the learning trajectory, the development of computationally efficient selection heuristics remains a necessity.

== Future Perspectives

A natural extension of this research is to evaluate whether the convergence separations observed in bilinear matrix games persist in more complex, non-bilinear optimization landscapes and in dimension higher than 2. Specifically, the standard regularized logistic regression problem can be formulated as a convex-concave saddle-point problem.

Exploring whether OMWU's best-iterate convergence continues to bypass boundary-induced slowdowns under non-bilinear objectives and unconstrained domains in high-dimensional machine learning settings could be a promising extension of this work.

#pagebreak()
= Appendix: Boundary Distance Derivations <appendix:boundary_distances>

To formally derive how these distances reduce to simple absolute coordinate differences, we evaluate the definition of the Euclidean ($L^2$) distance from the point $z_0 = (p_0, q_0)$ to a subset $S subset K$:
$ d_2(z_0, S) = inf_(w in S) norm(z_0 - w)_2 = inf_((p, q) in S) sqrt((p_0 - p)^2 + (q_0 - q)^2) $

We compute this distance step-by-step for each of the four boundary line segments:

1. *Left vertical boundary segment* $L_(p=0) = \{ (0, q) : q in [0, 1] \}$:
   The distance from $z_0$ to any point $(0, q) in L_(p=0)$ is:
   $ norm(z_0 - (0, q))_2 = sqrt((p_0 - 0)^2 + (q_0 - q)^2) $
   To find the distance $d_2(z_0, L_(p=0))$ to the segment, we minimize this expression over all points on the segment, i.e., over $q in [0, 1]$. Because the horizontal component $(p_0 - 0)^2 = p_0^2$ is constant, the minimum is achieved by choosing $q = q_0$ (which lies in $[0, 1]$ since $q_0 = 1/(2(1+delta)) in (0, 1)$). This choice eliminates the vertical coordinate difference term, yielding:
   $ d_2(z_0, L_(p=0)) = sqrt(p_0^2 + (q_0 - q_0)^2) = sqrt(p_0^2 + 0) = sqrt(p_0^2) = |p_0| $
   Since $p_0 = 1/(1+delta) > 0$, the absolute value simplifies directly to:
   $ d_2(z_0, L_(p=0)) = p_0 = 1/(1+delta) $

2. *Right vertical boundary segment* $L_(p=1) = \{ (1, q) : q in [0, 1] \}$:
   The distance from $z_0$ to any point $(1, q) in L_(p=1)$ is:
   $ norm(z_0 - (1, q))_2 = sqrt((p_0 - 1)^2 + (q_0 - q)^2) $
   Similarly, the infimum over $q in [0, 1]$ is achieved at $q = q_0$, which minimizes the vertical term to zero:
   $ d_2(z_0, L_(p=1)) = sqrt((p_0 - 1)^2 + (q_0 - q_0)^2) = sqrt((p_0 - 1)^2) = |p_0 - 1| $
   Because $p_0 = 1/(1+delta) < 1$ for any $delta > 0$, we have $p_0 - 1 < 0$. Thus, $|p_0 - 1| = 1 - p_0$. Substituting $p_0$ and simplifying:
   $ d_2(z_0, L_(p=1)) = 1 - 1/(1+delta) = (1+delta - 1)/(1+delta) = delta/(1+delta) $

3. *Lower horizontal boundary segment* $L_(q=0) = \{ (p, 0) : p in [0, 1] \}$:
   The distance from $z_0$ to any point $(p, 0) in L_(q=0)$ is:
   $ norm(z_0 - (p, 0))_2 = sqrt((p_0 - p)^2 + (q_0 - 0)^2) $
   To minimize this over $p in [0, 1]$, we choose the horizontal coordinate $p = p_0$ (which is valid since $p_0 = 1/(1+delta) in (0, 1)$), eliminating the horizontal term:
   $ d_2(z_0, L_(q=0)) = sqrt((p_0 - p_0)^2 + q_0^2) = sqrt(0 + q_0^2) = sqrt(q_0^2) = |q_0| $
   Since $q_0 = 1/(2(1+delta)) > 0$, this simplifies directly to:
   $ d_2(z_0, L_(q=0)) = q_0 = 1/(2(1+delta)) $

4. *Upper horizontal boundary segment* $L_(q=1) = \{ (p, 1) : p in [0, 1] \}$:
   The distance from $z_0$ to any point $(p, 1) in L_(q=1)$ is:
   $ norm(z_0 - (p, 1))_2 = sqrt((p_0 - p)^2 + (q_0 - 1)^2) $
   The infimum over $p in [0, 1]$ is achieved at $p = p_0$, which minimizes the horizontal term to zero:
   $ d_2(z_0, L_(q=1)) = sqrt((p_0 - p_0)^2 + (q_0 - 1)^2) = sqrt((q_0 - 1)^2) = |q_0 - 1| $
   Because $q_0 = 1/(2(1+delta)) < 1$ for any $delta > 0$, we have $q_0 - 1 < 0$. Thus, $|q_0 - 1| = 1 - q_0$. Substituting $q_0$ and simplifying:
   $ d_2(z_0, L_(q=1)) = 1 - 1/(2(1+delta)) = (2(1+delta) - 1)/(2(1+delta)) = (1+2delta)/(2(1+delta)) $

#pagebreak()
#bibliography("refs.bib")