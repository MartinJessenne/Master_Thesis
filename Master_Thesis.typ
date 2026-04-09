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
  
  #v(4cm)
  // #image("logo.png", width: 50%) // Placeholder for university logo
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
In many modern applications, from economics to artificial intelligence, multiple decision-makers, or "agents", interact within a shared environment. When these agents compete or cooperate, their interactions are mathematically modeled as games. A central concept in game theory is the *Nash Equilibrium*. Intuitively, it is a state where no player can improve their outcome by unilaterally changing their strategy. In a two-player zero-sum game, this equilibrium is often written as a pair of strategies $(x^*, y^*)$, where player $x$ minimizes their loss and player $y$ maximizes it, reaching a stable saddle point.

To find these equilibria, researchers often use *learning dynamics*, where agents repeatedly play the game and adjust their strategies based on past experiences. A popular class of such algorithms is "no-regret learning," which guarantees that, in the long run, an agent's performance is at least as good as the best fixed strategy in hindsight. Traditionally, the convergence of these learning algorithms is guaranteed only in an *ergodic* (or average) sense. This means that if we take the average of all the strategies played over time, this average approaches the Nash Equilibrium $(x^*, y^*)$. 

However, in recent years, there has been a profound paradigm shift towards understanding the *non-ergodic* behavior of these dynamics. Specifically, the focus has moved towards *last-iterate convergence*. In last-iterate convergence, we do not look at the historical average; instead, we expect the actual, step-by-step trajectory of the players' strategies to directly approach the equilibrium. This shift is theoretically crucial, as algorithms with converging iterates prevent undesirable phenomena such as recurrence, cyclic trajectories, and chaotic behavior commonly observed when multiple agents learn simultaneously.

== Motivation
The motivation for establishing non-ergodic convergence is deeply rooted in modern machine learning applications, particularly in the training of Generative Adversarial Networks (GANs) and multi-agent Reinforcement Learning (RL). In these practical settings, agents are often represented by complex, non-linear models like deep neural networks. 

In such complex landscapes, the traditional approach of "averaging the iterates" fails. To put it simply: taking the average of the parameters of two good neural networks will yield a network that is often completely broken and performs poorly. Practitioners therefore deploy the final model produced at the very end of the training process (the "last iterate"). This practical reality necessitates theoretical guarantees that this specific final state is close to an equilibrium. 

While certain algorithms, like Optimistic Gradient Descent-Ascent (OGDA), have been shown to enjoy fast and reliable last-iterate convergence, other premier algorithms such as Optimistic Multiplicative Weights Update (OMWU) can be arbitrarily slow to converge in their last iterate. This discrepancy prompts a critical question: if reliable last-iterate convergence is unattainable for OMWU, can it still achieve fast convergence under slightly weaker, yet practical criteria? We specifically investigate two alternatives:
- *Random-iterate convergence:* If we stop the algorithm at a uniformly random training step, is the strategy likely to be close to the equilibrium?
- *Best-iterate convergence:* Throughout the entire training process, does the algorithm at least occasionally hit a state that is very close to the equilibrium?

== Goal
This thesis aims to empirically verify and theoretically contextualize the separation between these distinct convergence modes, last-iterate, random-iterate, and best-iterate, focusing specifically on the OMWU algorithm. Through the study of ill-conditioned $2 times 2$ matrix games where the Nash equilibrium approaches the boundary of the probability simplex, we will demonstrate how OMWU's behavior leads to severe instability in the last iterate, while still preserving a fast best-iterate convergence rate. To achieve this, we introduce a robust computational framework in Rust and Python, engineered to resolve numerical instabilities and parallelize large-scale explorations. Ultimately, our goal is to construct a comprehensive cartography of OMWU's pathological behaviors, quantifying its chaos mathematically, and examining whether these theoretical separations persist in practical optimization scenarios.

= Theoretical Framework & Literature Review

== The Paradigm Shift in Convergence Studies
Historically, the analysis of learning dynamics such as Optimistic Multiplicative Weights Update (OMWU) and Optimistic Gradient Descent-Ascent (OGDA) centered on their ergodic properties. Both algorithms were celebrated for achieving $O(1/T)$ convergence rates to a Nash Equilibrium in the average iterate. 

As established previously, the practical demands of training complex non-linear models have forced a paradigm shift away from these historical averages. Theoretical guarantees must now focus on the *last iterate* to ensure that the final deployed model is close to an equilibrium. 

In the quest for non-ergodic convergence, OGDA has been shown to enjoy a reliable $O(1/sqrt(T))$ uniform last-iterate convergence rate @cai2024fast. Conversely, OMWU, despite its widespread use and superior theoretical properties in other contexts, has struggled to provide similar guarantees. Recent work by @cai2024fast revealed a fundamental limitation: a broad class of Optimistic Follow-The-Regularized-Leader (OFTRL) algorithms, including OMWU, cannot achieve a uniform last-iterate convergence rate. The authors show that because these algorithms do not "forget" their past cumulative losses quickly enough, they can get trapped. Specifically, in a simple parameterized $2 times 2$ matrix game, OMWU exhibits an $Omega(1)$ duality gap for $Omega(1/delta)$ iterations. Here, $delta$ represents a conditioning measure of the game matrix, typically corresponding to the minimum non-zero probability assigned to any action in the Nash Equilibrium. Because $delta$ can be arbitrarily small, this dependency prevents any uniform last-iterate convergence.

This limitation prompts a natural question: if OMWU fails to converge in the last iterate, does it succeed under weaker criteria? A subsequent study by @cai2025separation investigated this by establishing a separation between different convergence modes. They proved that while OMWU does not achieve polynomial *random-iterate* convergence (showing an $Omega(1/(log T))$ lower bound), it surprisingly achieves a uniform $O(T^(-1/6))$ *best-iterate* convergence rate in $2 times 2$ matrix games. This separation forms the foundational motivation for our work: to empirically analyze these pathological behaviors and quantify the chaos that prevents last-iterate convergence, while confirming the resilience of the best iterate.

== Algorithms
To ground our analysis, we formally introduce the primary algorithms examined in this thesis, which are implemented in our `optigame` framework and python library for reproducibilty's sake.

*Optimistic Follow-The-Regularized-Leader (OFTRL)* 

OFTRL is a family of no-regret online learning algorithms defined by a regularizer $R$. Given cumulative loss vectors $L_x^(t-1)$ and per-step losses $l_x^(t-1)$, the update rule for the $x$-player with step size $eta$ is given by:
$ x^t = op("argmin")_(x in Delta^(d_1)) { chevron.l x, L_x^(t-1) + l_x^(t-1) chevron.r + 1/eta R(x) } $
It is this dependence on the full history of cumulative losses ($L_x^(t-1)$) that causes the "non-forgetfulness" phenomenon, ultimately trapping the algorithm in sub-optimal strategies for extended periods @cai2024fast.

*Optimistic Multiplicative Weights Update (OMWU)* \
OMWU is a specific instance of OFTRL (and Optimistic Online Mirror Descent) instantiated with the negative entropy regularizer $R(x) = sum_i x[i] log x[i]$. In precise theoretical settings involving Legendre regularizers like negative entropy, the OFTRL and OOMD updates are mathematically equivalent. We empirically demonstrate this equivalence in our computational framework by providing two distinct implementations, `OmwuOftrl` and `OmwuOomd`, which follow these respective update rules while offering different numerical stability properties during execution.

*Optimistic Gradient Descent-Ascent (OGDA)* \
In contrast, OGDA uses the squared Euclidean norm regularizer $R(x) = 1/2 sum_i x[i]^2$. Unlike OMWU, OGDA successfully achieves an $O(1/sqrt(T))$ uniform random-iterate and last-iterate convergence rate @cai2024fast. We include OGDA in our study as a stable baseline to compare against OMWU's chaotic non-ergodic behavior.

== Formal Definitions
To empirically evaluate these algorithms, we rely on the following rigorous definitions.

*Game Setup & Nash Equilibrium* \
We study two-player zero-sum matrix games defined by a loss matrix $A in [0, 1]^(d_1 times d_2)$. Players choose mixed strategies $x in Delta^(d_1)$ and $y in Delta^(d_2)$ from the probability simplex. The probability simplex $Delta^d$ is defined as the set of vectors in $RR^d$ whose components are non-negative and sum to one, representing a valid probability distribution over the available actions. A fully mixed Nash Equilibrium $(x^*, y^*)$ is a state where neither player can unilaterally improve their outcome. We denote $delta > 0$ as the minimum probability assigned to any action in the Nash Equilibrium.

*Duality Gap* \
The proximity of a strategy profile $(x, y)$ to the Nash equilibrium is measured by the duality gap, defined as:
$ op("Gap")(x,y) = max_(y' in Delta^(d_2)) (x^T A y') - min_(x' in Delta^(d_1)) (x'^T A y) $
By the linearity of the expected loss, the maximum and minimum over the mixed strategies $y'$ and $x'$ are achieved at pure strategies. Thus, in practice, the duality gap is efficiently computed by taking the minimum and maximum components of the vectors $-A^T x$ and $A y$. In our Rust implementation, this calculation perfectly coincides with our use of the pre-calculated gradient vectors $"grad"_y = -A^T x$ and $"grad"_x = A y$. The duality gap is non-negative and equals zero if and only if $(x, y)$ is a Nash equilibrium.

*Modes of Convergence* \
We distinguish between three non-ergodic convergence modes, listed from strongest to weakest. In each case, $f(T)$ represents a convergence rate function that approaches zero as the number of iterations $T$ approaches infinity (e.g., $f(T) = 1/sqrt(T)$):
- *Last-iterate convergence:* The algorithm's final state approaches the equilibrium, formally $op("Gap")(x^T, y^T) <= O(f(T))$.
- *Random-iterate convergence:* The expected duality gap of an iterate chosen uniformly at random approaches zero, formally $1/T sum_(t=1)^T op("Gap")(x^t, y^t) <= O(f(T))$. This is closely tied to the average Social Dynamic Regret.
- *Best-iterate convergence:* The minimum duality gap observed over the training process approaches zero, formally $min_(t in [1,T]) op("Gap")(x^t, y^t) <= O(f(T))$.

*Uniform vs. Universal Rates* \
A critical distinction in the literature is between *uniform* and *universal* convergence rates @cai2025separation. A uniform rate bounds the convergence speed solely as a function of $T$ and the dimensions of the game, independent of specific game properties. A universal rate, however, may depend on game-specific constants, such as the minimum equilibrium probability $delta$. Distinguishing between these rates is essential when analyzing algorithms like OMWU that suffer from arbitrarily bad dependencies on $delta$.

= Methodology & Computational Framework

== Implementation & Performance analysis
To investigate the pathological behavior of OMWU at scale, particularly its divergence across thousands of parameterized matrices, we developed `optigame`, a high-performance computational framework. Early prototypes in pure Python demonstrated that sweeping over a dense grid of game parameters (e.g., $10,000$ steps across a $500 times 500$ grid) was prohibitively slow due to Python's Global Interpreter Lock (GIL) and inherent interpreter overhead. 

To overcome this, `optigame` is implemented as a Rust core wrapped in a Python interface via PyO3. This architecture allows us to express complex mathematical operations safely in Rust, leveraging its zero-cost abstractions, strict typing for simplex projections, and most importantly, thread-level parallelism. By utilizing the `rayon` crate, the algorithm automatically distributes the neighborhood exploration across all available CPU cores. This hybrid approach yields orders-of-magnitude speedups, reducing hours of computation to miliseconds, while allowing the results to be natively analyzed and plotted in interactive Python environments like Jupyter or Marimo.

#v(4cm)
#align(center)[
[INSERT PLOT HERE: Benchmark comparing Python, Mixed Rust, and Full Rust execution times]
]
#v(4cm)

== Numerical Stability & Constraints
As demonstrated in previous sections, OMWU and OFTRL rely on exponential updates parameterized by the step size $eta$ and the cumulative gradients. In ill-conditioned settings where gradients push strategies towards the boundary of the simplex, these exponential terms are prone to severe numerical overflow. 

To guarantee stability over tens of thousands of iterations, we implemented two critical safeguards:
1. *Shifted Exponentials:* During the multiplicative update, we subtract the maximum component of the gradient vector from all elements before exponentiation (e.g., $exp(s - max(s))$). This factors out the largest exponential growth without altering the final normalized distribution, entirely neutralizing overflow risks.
2. *Simplex Projections:* While OMWU natively maintains probability distributions via normalization, baseline algorithms like OGDA do not. For these, we implemented an efficient $O(n log n)$ exact projection onto the probability simplex, ensuring all iterates rigorously satisfy $x in Delta^(d_1)$ and $y in Delta^(d_2)$.

== Game Parameterization & Normalization
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

This perfectly recovers the $A_delta$ baseline game. While $A_delta$ only pushes the $x$-player's strategy towards the boundary (as $p = 1/(1+delta) -> 1$), our generalized $A_(lambda, gamma)$ allows us to push *both* strategies toward the boundaries simultaneously. By tuning $lambda -> 1$ (forcing $q -> 0$) and $gamma -> 0$ (forcing $p -> 1$), we can explore the severe pathological behaviors that arise when both players face ill-conditioned optima.

However, theoretical guarantees for learning algorithms traditionally require the matrix entries to be bounded in $[0,1]$. We mathematically prove that applying an affine normalization $A' = (A - min(A)) / (max(A) - min(A))$ preserves the underlying Nash Equilibrium. This allows us to sweep $lambda$ and $gamma$ arbitrarily close to the simplex boundaries while ensuring our matrices strictly satisfy the required theoretical constraints.

#v(4cm)
#align(center)[
[INSERT PLOT HERE: Graph showing the constructed (computed) NE perfectly matching the theoretical p and q lines]
]
#v(4cm)

== Quantifying Divergence
Traditional literature defines non-ergodic convergence using the simple Duality Gap. However, the duality gap alone is insufficient to capture the *dynamic nature* of OMWU's failure. When OMWU fails to converge in the last iterate, it does not simply stall; it frequently enters chaotic, expanding orbits around the equilibrium.

To rigorously distinguish between an algorithm that converges slowly and one that exhibits pathological cycles, we introduce two new trajectory-based metrics:
- *Variance of the Last 10% Iterates:* This measures the amplitude of the oscillations at the end of training. A high variance indicates that the algorithm is trapped in a limit cycle rather than converging to a point.
- *Total Variation:* Computed as $sum |op("Gap")^(t) - op("Gap")^(t-1)|$, this metric quantifies the total "distance" the duality gap travels. Algorithms that oscillate wildly will show a massive Total Variation compared to stable algorithms like OGDA.

These metrics allow us to automatically map the stability of the parameter space, pinpointing exactly where and how OMWU's last iterate breaks down.

#v(4cm)
#align(center)[
[INSERT PLOT HERE: Heatmap or scatter plot demonstrating the Variance or Total Variation across different values of lambda and gamma]
]
#v(4cm)

= Empirical Study: Convergence in Ill-Conditioned Matrix Games

In this chapter, we deploy our computational framework to empirically investigate the theoretical bounds discussed previously. We begin by reproducing the baseline pathological behavior of OMWU on a known hard instance. We then extend this analysis by introducing a novel, parameterized game matrix that pushes both players' strategies toward the simplex boundaries simultaneously, allowing us to deeply map OMWU's failure modes.

== Baseline Reproduction: The $A_delta$ Game
To establish a baseline, we first replicate the hard instance introduced by @cai2024fast. They define a parameterized $2 times 2$ game matrix $A_delta$ where the Nash equilibrium for the $x$-player is pushed arbitrarily close to the boundary of the probability simplex as $delta -> 0$, while the $y$-player's equilibrium remains strictly mixed.

By running OMWU on $A_delta$ over thousands of iterations, we empirically confirm the failure of the last-iterate convergence. The duality gap does not smoothly decay to zero; instead, it exhibits a sustained, non-vanishing oscillation. We quantify this chaos using our Total Variation and Variance metrics introduced in Chapter 3, providing a numerical foundation for the algorithm's instability.

#v(4cm)
#align(center)[
[INSERT PLOT HERE: Duality gap over time for A_delta showing non-vanishing oscillations]
]
#v(4cm)

== Novel Exploration: Simultaneous Boundaries ($A_{lambda, gamma}$)
While the $A_delta$ game exposes OMWU's vulnerability to a single poorly conditioned strategy, real-world min-max optimization problems often involve simultaneous ill-conditioning across all players. To test if this degrades the convergence rates further, we introduce a novel matrix family, $A_{lambda, gamma}$, utilizing the parameterization technique detailed in Chapter 3.

In this game, we define the target Nash equilibrium explicitly as $x^* = (p(lambda), 1-p(lambda))$ and $y^* = (q(gamma), 1-q(gamma))$, where both $p(lambda)$ and $q(gamma)$ are pushed towards the simplex boundaries simultaneously as $lambda -> 0$ and $gamma -> 0$. By applying our affine normalization to keep the matrix bounded in $[0, 1]$, we ensure the theoretical constraints of OMWU are respected.

This dense neighborhood exploration allows us to test a critical hypothesis: does simultaneous ill-conditioning fundamentally break the resilience of OMWU, or does the best-iterate convergence survive this extreme pathological setting?

#v(4cm)
#align(center)[
[INSERT PLOT HERE: Heatmap or 3D surface plot mapping the chaos metrics (e.g., Variance of last 10%) across the lambda and gamma grid]
]
#v(4cm)

== Trajectory Analysis: Strategy Profiles in 2D
Analyzing convergence solely through the 1D duality gap obscures the geometric reality of OMWU's dynamics. In zero-sum games, learning algorithms frequently exhibit rotational dynamics. Rather than pointing directly towards the Nash equilibrium, the strategies "orbit" the optimal point, creating limit cycles or diverging spirals.

To physically understand the algorithm's behavior, we project the full state of the $2 times 2$ game at iteration $t$ into a 2D Cartesian plane, tracking the coordinates $(x_1^t, y_1^t)$ within the unit square $[0, 1] times [0, 1]$. 

This visual representation clearly distinguishes between slow convergence (a spiral tightening inward) and a true limit cycle (a stable, continuous orbit). As we push $lambda$ and $gamma$ closer to zero, we observe how the equilibrium point is squeezed into the corner of the simplex, forcing the OMWU trajectories into highly skewed, high-velocity orbits that graze the boundaries.

#v(4cm)
#align(center)[
[INSERT PLOT HERE: 2D Cartesian plot of the strategy profiles (x_1^t vs y_1^t) showing the orbital trajectories around the Nash Equilibrium]
]
#v(4cm)

== Best vs. Last Iterate Comparison
The ultimate test of our hypothesis lies in isolating and comparing the Best-Iterate convergence against the failing Last-Iterate. The Best-Iterate duality gap is mathematically defined as the cumulative minimum of the error up to iteration $T$: $min_{s <= T} op("Gap")(x_s, y_s)$.

We plot this Best-Iterate sequence for both the single-boundary baseline ($A_delta$) and our simultaneous-boundary matrix ($A_{lambda, gamma}$, also denoted $M_delta$). Superimposing the theoretical baseline curves (e.g., $O(1/T)$ or $O(T^(-1/6))$) on a log-log scale allows us to visually and empirically verify the theorems proposed by @cai2025separation. 

Our results demonstrate the stark divergence between the Best-Iterate and Last-Iterate trajectories. While the Last-Iterate oscillates wildly, the Best-Iterate manages to reliably drop as the algorithm's orbit periodically swings close to the equilibrium. Furthermore, comparing $A_delta$ and $M_delta$ reveals the precise impact of multi-dimensional ill-conditioning on the empirical convergence rate.

#v(4cm)
#align(center)[
[INSERT PLOT HERE: Log-Log plot comparing Best-Iterate and Last-Iterate for A_delta and M_delta, with the theoretical baselines superimposed]
]
#v(4cm)

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