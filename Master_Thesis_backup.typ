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

The resolution of competitive scenarios in optimization and game theory is fundamentally structured around identifying the saddle point of an objective function $op("min")_(x in cal(X)) op("max")_(y in cal(Y)) Phi(x, y)$. Historically, the development of no-regret learning dynamics provided convergence guarantees primarily in an ergodic sense, where the time-averaged sequence of strategies, $overline(z)_T = 1/T sum_(t=1)^T (x_t, y_t)$, asymptotically approaches a Nash Equilibrium. This ergodic paradigm is theoretically sufficient in convex-concave landscapes because Jensen’s inequality ensures that the performance of the averaged parameters is rigorously bounded by the average functional performance:
$ Phi(1/T sum_(t=1)^T x_t, y^*) <= 1/T sum_(t=1)^T Phi(x_t, y^*) $
In such settings, deploying the mathematical average of the iterates yields a theoretically robust and functionally stable equilibrium solution.

However, modern architectures in Generative Adversarial Networks (GANs) and multi-agent reinforcement learning (MARL) introduce non-convex objective surfaces where the lack of convexity destroys the foundational premise of Jensen's inequality. In these high-dimensional regimes, the mathematical average of two high-performing parameter sets: $theta_1$ and $theta_2$ such that $L(theta_1)$ and $L(theta_2)$ are both small, does not correspond to a functional "average" model; instead, the resulting average $(theta_1 + theta_2)/2$ typically resides in a region of the parameter space characterized by high loss. This geometric reality, necessitates a transition toward last-iterate (iteration-wise) convergence. Ensuring that the actual sequence of iterates $(x_T, y_T)$ converges directly to the local saddle point is the only viable mechanism for deploying models with guaranteed functional properties in non-convex environments. Additionally, iteration-wise analysis is required to detect dynamic instabilities, such as limit cycles or chaotic orbits, which may be concealed by ergodic averages.

In these environments, the average of the parameters of two models does not necessarily result in a model with the desired properties. Consequently, practitioners typically deploy the final state produced at the end of the training process (the "last iterate"). This necessitates theoretical guarantees that this specific final state is close to an equilibrium. 

While algorithms like Optimistic Gradient Descent-Ascent (OGDA) have been shown to achieve uniform last-iterate convergence, other algorithms, such as Optimistic Multiplicative Weights Update (OMWU), can exhibit slow last-iterate convergence. This raises the question of whether OMWU can satisfy alternative criteria under weaker convergence modes. We investigate three convergence modes:
- *Last-iterate convergence:* The proximity of the final state produced at the end of the training process to the equilibrium.
- *Random-iterate convergence:* The expected duality gap of a strategy chosen uniformly at random from the training history.
- *Best-iterate convergence:* The existence of at least one state during the training process that is close to the equilibrium.

== Goal
This thesis aims to empirically evaluate and theoretically analyze the separation between last-iterate, random-iterate, and best-iterate convergence modes, with a focus on the OMWU algorithm. Through the study of $2 times 2$ matrix games where the Nash equilibrium is located near the boundary of the probability simplex, we will examine the stability of OMWU's last iterate relative to its best-iterate convergence rate. We introduce a computational framework in Rust and Python designed for numerical stability and parallelized exploration. The objective is to characterize the non-convergent behaviors of OMWU and examine whether these theoretical separations persist in optimization scenarios.

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
OGDA uses the squared Euclidean norm regularizer $R(x) = 1/2 sum_i x[i]^2$. OGDA achieves a $O(1/sqrt(T))$ uniform random-iterate and last-iterate convergence rate @cai2025fastconvergence. It serves as a baseline for comparison with OMWU's chaotic behavior.

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
  Benchmark results showing performance improvements after migrating the computation logic to Rust.
  ],
)
]
#v(2cm)

== Numerical Stability & Implementation Details
OMWU and OFTRL use exponential updates determined by the step size $eta$ and cumulative gradients. In settings where strategies approach the simplex boundary, these terms can lead to numerical overflow. 

We implemented two safeguards for stability:
1. *Shifted Exponentials:* During multiplicative updates, we subtract the maximum component of the gradient vector from all elements before exponentiation (e.g., $exp(s - max(s))$). This factors out the growth without altering the normalized distribution, which prevents overflow.
2. *Simplex Projections:* For algorithms that do not natively maintain probability distributions, such as OGDA, we use an $O(n log n)$ projection onto the probability simplex to ensure iterates satisfy $x in Delta^(d_1)$ and $y in Delta^(d_2)$.

=== Zero-Allocation Hot Loop
To optimize throughput during large-scale simulations, the `random_neighborhood_exploration` and grid search routines utilize a zero-allocation design in the core loop. Initializing matrices and state vectors in memory is a costly operation when performed repeatedly across thousands of experiments. Our implementation utilizes pre-allocated buffers for game matrices and strategy states, which are overwritten in place for each trial. By passing matrix views (references) instead of copying the matrix data, we reduce memory allocation overhead and improve cache locality, resulting in performance improvements over naive allocation strategies.

== Quantifying Divergence
To evaluate convergence behavior and distinguish between slow decay and periodic behavior, we introduce three trajectory-based metrics:
1. *Maximum Value of the Last 10% Iterates:* Measures the peak amplitude of oscillations near the end of the simulation.
2. *Variance of the Last 10% Iterates:* Quantifies the stability of the final iterates; high variance indicates persistent oscillations.
3. *Total Variation:* Computed as $sum_t |op("Gap")^(t) - op("Gap")^(t-1)|$, this metric quantifies the total oscillation amplitude across the trajectory. 

These metrics provide a quantitative basis for mapping the stability of the parameter space and characterizing OMWU's last-iterate behavior.


= Empirical Study: Convergence in Ill-Conditioned Matrix Games

In this chapter, we deploy our computational framework to empirically investigate the theoretical bounds discussed previously. We aim to map the non-convergent behavior of OMWU in ill-conditioned games and validate the theoretical separation between last-iterate convergence and other modes (best/random).

== The Baseline Hard Instance & Random Neighborhood Exploration
Before evaluating specific parameterizations, we test the robustness of OMWU's non-convergent behavior. We do this by applying uniform random perturbations to the $A_delta$ baseline game introduced by @cai2025fastconvergence, defined as:
$ A_delta = mat(1/2 + delta, 1/2; 0, 1) $
The objective is to determine whether the observed instability is an artifact of the exact matrix structure or a robust phenomenon under small perturbations. We define a perturbation matrix $U in [-1, 1]^(2 times 2)$ scaled by a parameter $epsilon > 0$, and evaluate the OMWU dynamics on the matrix $A_delta + epsilon U$. To thoroughly explore the neighborhood, we conducted a grid search for each value of $epsilon$, running 1,000 independent experiments, each for 10,000 steps.

#v(2cm)
#align(center)[
#figure(
  image("images/BoxPlot.png", width: 80%),
  caption: [Distribution of the maximum duality gap over the last 10% of iterations for OMWU, across different perturbation magnitudes $epsilon$.]
)
]
#v(2cm)

The results, presented in the boxplot above, indicate that adding a uniform perturbation to $A_delta$ does not induce convergence. While the variance of the metric increases significantly with larger values of $epsilon$, the median remains elevated across all perturbation magnitudes. This observation confirms that the non-convergent behavior is robust, which motivates the study of specific parametric lines to carefully control the equilibrium's position relative to the simplex boundaries.

== Parametric Game Design ($A_(lambda, gamma)$) & Normalization
To exhaustively test OMWU's sensitivity to the minimum equilibrium probability $delta$, we require the ability to construct arbitrary game matrices with a known, predetermined Nash Equilibrium $(x^*, y^*)$. 

Given a $2 times 2$ matrix $A = mat(a, b; c, d)$, the // A nash equilibrium 
 Nash equilibrium $(x^*, y^*)$ must satisfy the indifference conditions. For the $y$-player facing $x^*$, and the $x$-player facing $y^*$, these conditions are respectively:
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

We have one remaining degree of freedom. By arbitrarily fixing $d=1$, we can map the off-diagonal elements directly to our parameters by setting $b = lambda$ and $c = gamma$. Substituting these into the scaling factor equation yields $a = S - 1 + lambda + gamma$. This gives us the exact, elegant formulation for our parameterized matrix family $A_(lambda, gamma)$:
$ A_(lambda, gamma) = mat(
  S - 1 + lambda + gamma, lambda;
  gamma, 1
) $

This construction guarantees that the unique Nash equilibrium is exactly $(p, 1-p)$ and $(q, 1-q)$, where $p = (1-gamma)/S$ and $q = (1-lambda)/S$.

Crucially, we can show that $A_(lambda, gamma)$ is a direct generalization of the $A_delta$ matrix studied in the literature. By setting $gamma = 0$ and $lambda = 1/2$, and choosing the scaling factor $S = 1 + delta$, the matrix simplifies to:
$ A_(1/2, 0) = mat(
  (1+delta) - 1 + 1/2 + 0, 1/2;
  0, 1
) = mat(
  1/2 + delta, 1/2;
  0, 1
) $

This recovers the $A_delta$ baseline game. While $A_delta$ only pushes the $x$-player's strategy towards the boundary (as $p = 1/(1+delta) -> 1$), our generalized $A_(lambda, gamma)$ allows us to push *both* strategies toward the boundaries simultaneously. By tuning $lambda -> 1$ (forcing $q -> 0$) and $gamma -> 0$ (forcing $p -> 1$), we can explore the pathological behaviors that arise when both players face ill-conditioned optima.

However, theoretical guarantees for learning algorithms traditionally require the matrix entries to be bounded in $[0,1]$. We mathematically prove that applying an affine normalization $A' = (A - min(A)) / (max(A) - min(A))$ preserves the underlying Nash Equilibrium. This allows us to sweep $lambda$ and $gamma$ arbitrarily close to the simplex boundaries while ensuring our matrices strictly satisfy the required theoretical constraints.

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
#v(2cm)

The results demonstrate that both $A_(lambda, gamma)$ and $A_("sep")$ exhibit qualitatively similar non-ergodic profiles, characterized by persistent oscillations and periodic spikes in the distance to the Nash Equilibrium. This confirmation supports the use of $A_(lambda, gamma)$ as a valid generalization of the literature baseline.

However, the comparison reveals numerical discrepancies during the optimization of $A_("sep")$. Specifically, as the equilibrium approaches the boundary ($delta_x, delta_y -> 0$), the coefficients of $A_("sep")$ can lead to numerical instability, occasionally resulting in floating-point overflows or undefined values (NaNs). In contrast, the $A_(lambda, gamma)$ parameterization, when used in conjunction with the affine normalization described in Section 4.2, preserves the pathological behavior while ensuring reliable computation across a broader range of the parameter space. This analysis indicates that the observed discrepancies are artifacts of the numerical implementation of the baseline matrix rather than fundamental differences in game dynamics.

== Strategy Profiles and Orbital Dynamics
To analyze the behavioral dynamics of the algorithms, we evaluate the evolution of the strategies in both 1D and 2D projections. 

// #v(2cm)
// #align(center)[
// #figure(
//   grid(
//     rows: 2,
//     gutter: 1em,
//     image("images/Ogda_Last_xo_vs_ne.png", width: 100%),
//     image("images/Omwu_last_it_vs_ne.png", width: 100%),
//   ),
//   caption: [Evolution of computed last-iterate strategies against theoretical Nash Equilibrium probabilities for OGDA (top) and OMWU (bottom).]
// )
// ]

The strategy tracking results indicate a clear separation between the algorithms. OGDA consistently identifies and tracks the theoretical Nash Equilibrium as the game parameters vary. In contrast, OMWU's last-iterate computation exhibits sustained oscillations, failing to converge to the theoretical values.

To further characterize this non-convergent behavior, we visualize the strategy profile in the 2D plane $(x_1, y_1)$ and evaluate the $L^2$ distance to the Nash Equilibrium over time for a representative ill-conditioned instance.

#v(2cm)
#align(center)[
#figure(
  grid(
    columns: 2,
    gutter: 1em,
    image("images/2D_Profile_Strategies.png", width: 80%),
    image("images/L2_dist_to_NE.png", width: 120%),
  ),
  caption: [2D strategy profile (left) showing orbital behavior around the Nash Equilibrium, and $L^2$ distance to the NE over time (right) for OMWU.]
)
]

The 2D strategy profile illustrates that OMWU does not converge to a static point but instead enters a limit cycle or orbital trajectory around the theoretical equilibrium. This rotational dynamics is reflected in the $L^2$ distance plot, which exhibits periodic fluctuations. The sharp drops in the distance occur when the trajectory passes through the neighborhood of the Nash Equilibrium, while the spikes correspond to the segments of the orbit furthest from the solution. This periodic proximity explains how OMWU can maintain a small best-iterate gap despite the lack of last-iterate convergence.

== Best vs. Last Iterate Comparison

= From Matrix Games to Optimization
== Application: 
Formulating Regularized Logistic Regression as a Min-Max game.
== Experiments: 
Applying OMWU and OGDA to this model.
== Analysis: 
Checking if the separation of convergence modes persists in practical optimization settings.

= Conclusion
== Synthesis of empirical findings and future perspectives.

#bibliography("refs.bib")