---
type:
status: Open
related_pillar: "[[Thesis_Master_Plan]]"
tags:
  - thesis
created: 2026-05-01 14:49
---


# The Shift from Ergodic to Iteration-Wise Convergence in Min-Max Optimization and Deep Learning


## Introduction to Min-Max Optimization and the Ergodic Paradigm

The mathematical foundation of competitive systems, adversarial machine learning, and game theory rests upon the resolution of min-max optimization problems. Formally, these problems are defined as finding the saddle point of an objective function $\min_{x \in \mathcal{X}} \max_{y \in \mathcal{Y}} \Phi(x, y)$, where $\Phi(x, y)$ represents a smooth, continuous objective function, $x$ represents the parameters controlled by the minimizing agent, and $y$ represents the parameters controlled by the maximizing agent. Historically, the optimization and game theory communities approached the resolution of these competitive scenarios through the lens of no-regret learning dynamics. Foundational algorithms such as Fictitious Play, Follow-The-Regularized-Leader (FTRL), and the Multiplicative Weights Update (MWU) algorithm were developed under the premise that if players continuously minimize their regret over time, their aggregate behavior will stabilize. The central theoretical guarantee provided by these classical methods is ergodic convergence, which posits that the time-averaged history of the players' strategies—the ergodic sequence defined as $\frac{1}{T}\sum_{t=1}^T (x_t, y_t)$—will asymptotically converge to a Nash Equilibrium as $T \to \infty$.

In purely convex-concave landscapes, such as linear programming, zero-sum bimatrix games, and classic optimal transport formulations, ergodic convergence serves as an elegant, mathematically sound, and entirely sufficient solution. Because the objective function is strictly convex with respect to the minimizing variable $x$ and strictly concave with respect to the maximizing variable $y$, Jensen's inequality guarantees that the functional performance of the mathematically averaged parameters is rigorously bounded by the average of the functional performances observed across the training timeline. Consequently, deploying the historical average of the iterates yields a highly effective, stable, and theoretically robust equilibrium solution. In classical algorithmic game theory, analyzing the time-average of a system's trajectory was not merely a convenience but the fundamental definition of solving the game.

However, the advent of modern deep learning architectures—most notably Generative Adversarial Networks (GANs) and multi-agent reinforcement learning (MARL) systems—introduced highly non-convex, non-concave objective landscapes into the domain of min-max optimization. In these high-dimensional, parameterized applications, the foundational assumptions that govern the utility of ergodic convergence completely break down. The optimization algorithms are no longer simply searching for equilibrium distributions over discrete sets of actions; they are optimizing millions of continuous parameters within deep neural networks whose loss surfaces exhibit extreme non-linearities, sharp valleys, and isolated manifolds.

### The Core Hypothesis: The Failure of Parameter Averaging in Non-Convexity

The central hypothesis driving the recent and aggressive paradigm shift in optimization literature is that ergodic convergence is practically useless in non-convex settings because the mathematical average of two high-performing parameter sets does not yield a functional "average" model. This profound failure is fundamentally rooted in the complex geometry of the loss landscape inherent to deep neural networks. In a deep learning architecture, the loss function $L(\theta)$ is highly non-linear and non-convex. Therefore, given two distinct parameter configurations $\theta_1$ and $\theta_2$ that both independently achieve exceptionally low loss, their arithmetic mean $\frac{\theta_1 + \theta_2}{2}$ is practically guaranteed to achieve an exceptionally high loss. Mathematically, the lack of convexity destroys the foundational premise of Jensen's inequality, dictating that $L(\frac{x+y}{2}) \not\leq \frac{L(x)+L(y)}{2}$.

This geometric reality translates directly into catastrophic functional failures during model deployment. If a GAN generator's weights oscillate between two distinct modes of data generation during the adversarial training process, the ergodic average of those historical weights will fall into a region of the high-dimensional parameter space that has never been optimized. The resulting aggregated network will produce meaningless, distorted outputs, a phenomenon colloquially referred to in the literature as generating a "garbage model". Similarly, in multi-agent reinforcement learning, if an autonomous vehicle agent learns to navigate an obstacle by swerving sharply left in one training episode and swerving sharply right in another, the mathematical average of those two successful policies will instruct the agent to drive directly forward into the obstacle.

Because contemporary deep learning systems require the deployment of the final, physical set of neural network weights rather than an abstract historical distribution of states, the optimization community has been forced to pivot aggressively toward ensuring last-iterate, or iteration-wise, convergence. Last-iterate convergence is a strict guarantee that the actual, specific parameters at the terminus of the training process—the last iterates $(x_T, y_T)$—converge directly to the local min-max solution. This shift requires entirely new classes of algorithms, novel spectral analysis techniques, and a fundamental rethinking of how adversarial learning dynamics interact with non-convex manifolds.

## The Taxonomy of Convergence in Optimization Literature

As the community recognized the necessity of moving beyond ergodic averages, the literature evolved to precisely categorize the behavior of individual iterates during the training sequence. Recent comprehensive analyses, particularly those by Cai, Farina, and colleagues, have established a rigorous taxonomy separating three specific modes of iterate convergence. In all these modes, the proximity of the parameter state to an exact Nash Equilibrium is measured by the duality gap, a non-negative scalar that equals zero if and only if the strategy profile constitutes a mathematically perfect equilibrium.

The strongest and most desirable form of convergence is last-iterate convergence. This property mandates that the actual chronological sequence of the learner's strategies asymptotically approaches the exact set of equilibrium strategies as time progresses to infinity. If an algorithm guarantees last-iterate convergence, practitioners can confidently halt the training process at iteration $T$ and deploy the resulting parameter set, knowing it represents an optimal or near-optimal state.

The intermediate form is defined as random-iterate convergence. This property asserts that if a practitioner samples an iteration uniformly at random from the entire history of the training run, the expected strategy profile will be close to the equilibrium. Mathematically, bounding the random-iterate convergence rate is equivalent to establishing an upper bound on the algorithm's average social dynamic regret over the training timeline. While weaker than last-iterate convergence, random-iterate convergence still implies that the vast majority of the algorithm's trajectory resides in close proximity to the optimal solution.

The weakest form within this new taxonomy is best-iterate convergence. This condition merely guarantees that there exists at least one subsequence of the learner's strategies that converges to the set of equilibrium strategies. In practical terms, best-iterate convergence means that the algorithm periodically passes exceptionally close to the true equilibrium during its traversal of the parameter space, even if it immediately diverges away from that equilibrium in subsequent steps.

Beyond these three behavioral modes, the optimization literature rigorously distinguishes between the scope of the convergence rates provided by mathematical proofs. Uniform convergence results provide an absolute upper bound on the convergence rate that applies universally to any game instance, completely independent of the internal mathematical condition numbers of the game. Conversely, universal convergence results apply broadly to any game instance but contain an explicit mathematical dependence on game-specific condition numbers, such as $\delta$, which represents the smallest nonzero probability used within the true Nash Equilibrium. Because $\delta$ can be arbitrarily small, universal convergence rates can degrade catastrophically in complex games, making uniform convergence the ultimate gold standard for algorithm design.

## Theoretical Drivers for Prioritizing Last-Iterate Convergence

The realization that ergodic averages are functionally insufficient for deploying functional neural networks has spurred a wave of deep theoretical investigations into the continuous and discrete-time limit points of gradient-based algorithms. Leading theoreticians, including Daskalakis, Gidel, and Letcher, have identified profound mathematical drivers that transcend mere non-convexity, revealing structural reasons why last-iterate convergence is so notoriously difficult to achieve in saddle-point problems.

### Spurious Attractors and Internally Chain-Transitive Sets

A critical theoretical driver behind the prioritization of last-iterate research is the unsettling discovery that standard gradient-based optimization methods in min-max games do not merely oscillate harmlessly in the vicinity of true equilibria; they can be actively repelled by them or permanently attracted to functionally meaningless regions of the parameter space. Hsieh, Mertikopoulos, and Cevher conducted an exhaustive analysis mapping the long-run behavior of generalized Robbins-Monro stochastic approximation algorithms to deterministic mean-field dynamical systems. Their theoretical framework revealed that the asymptotic limit points of all generalized Robbins-Monro schemes necessarily belong to the Internally Chain-Transitive (ICT) sets of these corresponding mean-field dynamics.

In standard single-loss minimization problems, these ICT sets are highly predictable and benign, consisting almost entirely of the objective function's critical points. Therefore, gradient descent reliably converges to functional local minima. However, moving into the domain of general min-max problems, the topology of the dynamics' ICT sets becomes dangerously complex. Hsieh et al. demonstrated that in adversarial games, an ICT set can form a "spurious attractor"—a region of the parameter space that acts as a gravitational sink for the algorithm's trajectory but does not contain a single legitimate stationary or critical point of the problem under study.

Specifically, these spurious attractors can manifest as globally attracting limit cycles. The algorithms under consideration, even advanced variance-reduced methods, become permanently trapped in these cycles and are entirely incapable of escaping. Furthermore, the theoretical analysis uncovered scenarios where a highly desirable, stable min-max point possesses a basin of attraction that is completely "shielded" by a surrounding unstable ICT set. Consequently, when algorithms are run with standard stochastic gradient noise, the noise repeatedly pushes the trajectory into the repulsive field of the unstable ICT set, which aggressively violently repels the iterates away from the true solution and forces them toward the spurious, non-critical limit cycles. This topological reality underscores why last-iterate convergence cannot be assumed and must be explicitly engineered into the algorithm's update rule.

### The Impossibility of Global Convergence in Multi-Loss Optimization

Further cementing the need for specialized last-iterate algorithms, Alistair Letcher provided a rigorous mathematical proof regarding the impossibility of achieving global convergence in multi-loss optimization. In single-loss optimization, the property of coercivity—where the loss function approaches infinity as the parameters approach the boundaries of the space—guarantees the existence of a global minimum, ensuring that standard gradient descent will eventually find a critical point from almost any initialization. Letcher sought to determine if any algorithm could provide similar global guarantees in competitive differentiable games.

Letcher defined the concept of a "reasonable" algorithm using two exceedingly weak and universally desirable criteria. First, the algorithm's fixed points must correspond strictly to the critical points of the game, meaning the algorithm will only halt its execution if the simultaneous gradient vector is exactly zero. Second, the algorithm must almost surely avoid strict maxima. A strict maximum is defined geometrically as a critical point where the game's Hessian matrix is strictly negative-definite, meaning that all participating players could simultaneously decrease their respective losses by moving away from the point in any direction. Because converging to a strict maximum represents the absolute antithesis of minimization, any viable machine learning algorithm must inherently satisfy this avoidance criterion.

Letcher's groundbreaking impossibility theorem proved that global convergence is fundamentally incompatible with these two criteria. By constructing a specific two-player zero-sum game characterized by losses that are both perfectly analytic and strictly coercive, Letcher demonstrated a topology where the only simultaneous critical point in the entire landscape is a strict maximum. Consequently, any "reasonable" algorithm deployed in this environment faces a mathematical paradox: it cannot stop anywhere except at a critical point, but it is explicitly designed to avoid the only critical point that exists. Therefore, the algorithm is mathematically forced to either diverge to infinite losses or enter a state of perpetual, bounded non-convergence, manifesting as infinite limit cycles.

This theoretical driver definitively proved that the failures observed in GAN training are not merely the result of poor hyperparameter tuning or insufficient training time; they are the result of fundamental mathematical limits within multi-loss optimization. Because global convergence from arbitrary initializations is mathematically impossible, the optimization community was forced to abandon the search for a universal solver and instead prioritize the development of algorithms that guarantee strong, localized last-iterate convergence within the specific basins of attraction surrounding desirable Nash Equilibria.

### Spectral Analysis and the Antisymmetric Jacobian

Gidel, Azizian, Mitliagkas, and Lacoste-Julien further expanded the theoretical justification for specialized last-iterate methods by applying rigorous spectral analysis to the Jacobian of the vector field governing differentiable games. In standard minimization, the dynamics are governed by the Hessian matrix, which is inherently symmetric and therefore possesses entirely real eigenvalues. Real eigenvalues dictate that the gradient flow will move monotonically down the loss surface toward a local minimum.

However, in min-max games, the vector field is governed by a game Jacobian that can be decomposed into two distinct matrices: a symmetric component representing the "potential" nature of the game, and an antisymmetric component representing the "Hamiltonian" or purely competitive nature of the game. The antisymmetric component introduces complex eigenvalues into the spectral profile of the system.

These complex eigenvalues are the mathematical engine behind rotational divergence. Standard Simultaneous Gradient Descent/Ascent (SGDA) processes these complex eigenvalues poorly. In continuous time, SGDA causes the parameters to orbit the equilibrium in perfectly closed, conservative loops. In discrete time—which is how all neural networks are practically trained—the discretization error acts as an expansive force, causing the radius of the orbit to increase with every step, ensuring that the last iterate provably diverges toward infinity. Gidel's spectral framework demonstrated that achieving last-iterate convergence requires the implementation of algorithmic update rules that explicitly manipulate the spectral shape of the Jacobian, effectively dampening the imaginary components of the eigenvalues to force the trajectory to contract inward toward the saddle point.

## Algorithmic Solutions: Proximal Point, Extra-Gradient, and OGDA

To overcome the rotational divergence caused by the antisymmetric game Jacobian and successfully achieve last-iterate convergence, the optimization literature converged on three deeply interconnected algorithmic frameworks: the Proximal Point Method (PPM), Extra-gradient (EG) methods, and Optimistic Gradient Descent Ascent (OGDA).

### The Proximal Point Method (PPM)

The Proximal Point Method serves as the theoretical gold standard and the mathematical foundation for understanding last-iterate contraction in monotone inclusion problems and saddle-point optimization. Given a joint parameter state $z_t$ and a vector field $V(z)$, the implicit PPM update rule is defined by the equation: $z_{t+1} = z_t - \eta V(z_{t+1})$

The critical characteristic of PPM is its implicit nature; the update vector is determined by evaluating the gradient at the _future_ point $z_{t+1}$ rather than the current point $z_t$. By solving for the future state, PPM acts as an implicit Euler discretization of the continuous-time dynamics. This implicit formulation heavily penalizes large steps and mathematically dampens the oscillatory behavior caused by the imaginary eigenvalues of the game Jacobian, ensuring unconditional stability and extraordinarily rapid last-iterate convergence. However, calculating $V(z_{t+1})$ requires solving a complex implicit equation or executing an inner optimization loop at every single training step. While theoretically flawless, this requirement renders PPM computationally infeasible for high-dimensional deep learning applications involving millions of parameters, such as modern GANs and deep MARL networks.

### Extra-Gradient (EG) Methods

To approximate the unparalleled stability of PPM without bearing the impossible computational burden of solving implicit equations, Korpelevich introduced the Extra-gradient method. EG explicitly approximates the future point required by PPM by taking a preliminary "exploration" step, and then computing the actual parameter update based on the gradient evaluated at that explored location. The algorithm proceeds in two distinct phases per iteration:

1. The Exploration Step: $z_{t+1/2} = z_t - \eta V(z_t)$
    
2. The Update Step: $z_{t+1} = z_t - \eta V(z_{t+1/2})$
    

Gidel, Gorbunov, and Loizou demonstrated through tight spectral and potential-function analyses that EG achieves an optimal $O(1/T)$ last-iterate convergence rate for unconstrained monotone variational inequalities, and a $O(1/\sqrt{T})$ rate in constrained settings. By utilizing the gradient of the extrapolated mid-point $z_{t+1/2}$, the EG update effectively mimics the implicit regularization properties of PPM. The extrapolated gradient perfectly anticipates the rotational curvature of the vector field, creating a corrective force that pulls the trajectory inward, causing it to spiral tightly toward the saddle point rather than diverging outward.

The primary drawback of the Extra-gradient method remains computational overhead. Because EG requires computing $V(z_t)$ for the exploration step and $V(z_{t+1/2})$ for the update step, it demands two distinct gradient oracle calls, effectively doubling the time required for a forward-backward pass through a deep neural network.

### Optimistic Gradient Descent Ascent (OGDA)

To alleviate the heavy computational burden of the two-call Extra-gradient method, Popov introduced the Optimistic Gradient method, widely referred to in modern literature as Past Extra-gradient or Optimistic Gradient Descent Ascent (OGDA). OGDA ingeniously approximates the future gradient by recycling the gradient computed during the previous time step, effectively requiring only a single new oracle call per iteration. The update rule is typically formulated as: $z_{t+1} = z_t - 2\eta V(z_t) + \eta V(z_{t-1})$

This "extrapolation from the past" introduces a powerful negative momentum term into the optimization trajectory. Daskalakis, Panageas, and others established that this negative momentum acts as mathematical friction against the cyclic dynamics of the min-max game. If the parameters begin to oscillate, the subtraction of the past gradient $\eta V(z_{t-1})$ acts as a dampener, bleeding energy out of the Hamiltonian cycles and forcing strict last-iterate convergence to the saddle point in unconstrained convex-concave landscapes.

|**Algorithm**|**Gradient Oracle Calls per Iteration**|**Approximation Mechanism**|**Last-Iterate Convergence Guarantee (Unconstrained)**|
|---|---|---|---|
|**PPM**|Implicit (Requires Inner Solver)|Exact Future Evaluation|Optimal, unconditional stability|
|**Extra-Gradient (EG)**|2|Explicit Lookahead ($z_{t+1/2}$)|$O(1/T)$|
|**OGDA**|1|Past Extrapolation ($z_{t-1}$)|$O(1/T)$|

Both EG and OGDA successfully bend the trajectories of the vector field to force last-iterate convergence. However, as recent literature has shown, their theoretical guarantees diverge sharply when analyzing the specific regularizers used to generate the updates, leading to the discovery of the "forgetfulness" paradigm.

## Limit Cycles, Oscillations, and the Rock-Paper-Scissors Dynamic

To deeply understand why algorithms fail to achieve last-iterate convergence and why ergodic averaging presents a mathematical illusion of success, researchers frequently model learning dynamics using the Rock-Paper-Scissors (RPS) framework. RPS represents a quintessential zero-sum, non-cooperative game characterized by perfect non-transitive dominance: Rock beats Scissors, Scissors beats Paper, and Paper beats Rock.

### The Illusion of the Ergodic Average

In the RPS game, the unique Nash Equilibrium dictates that players should randomize their actions completely, playing each of the three options with a probability of exactly $1/3$. This strategy ensures that neither player can be systematically exploited. However, when autonomous agents update their strategies using standard gradient descent or the classical Multiplicative Weights Update (MWU) algorithm, their dynamic trajectory does not converge to this central point. Because the payoff matrix of RPS is perfectly skew-symmetric, the eigenvalues of the system's Jacobian are purely imaginary.

This purely imaginary spectral profile creates a perfectly conservative Hamiltonian system. Consequently, the algorithms exhibit persistent cyclic motions, tracing closed orbits known as limit cycles or Shapley polygons around the Nash Equilibrium without ever spiraling inward. As the discrete step size of the algorithm increases, MWU exhibits extreme instability and Hamiltonian chaos, causing the trajectory to spiral violently outward to the very boundaries of the probability simplex.

The mathematical illusion of the ergodic paradigm becomes starkly apparent when applying time-averaging to these unstable orbits. Because the trajectory forms a perfectly symmetric limit cycle around the central equilibrium, the arithmetic mean of all the iterates across the timeline evaluates exactly to the $(1/3, 1/3, 1/3)$ Nash Equilibrium. Therefore, the algorithm technically achieves "ergodic convergence." The theoretical proofs of FTRL and MWU hold true: the historical average solves the game.

Yet, in any given iteration $T$, the actual physical state of the algorithm—the last iterate—is located at the extreme edge of the simplex, playing a nearly pure, deterministic strategy that is highly exploitable by an opponent. In the context of Deep RL and GANs, this phenomenon directly mirrors neural network parameters orbiting a saddle point. The mathematical average of the cyclic weights represents a theoretical equilibrium, but the actual network deployed at any given time step is highly unstable, oscillating wildly between extreme modes of behavior. This disconnect proves that in dynamic systems governed by imaginary eigenvalues, ergodic convergence hides severe instability.

## The Necessity of "Forgetfulness" in Algorithm Design

Recent breakthroughs, specifically documented in Cai, Farina, and colleagues' 2024 and 2025 analyses on last-iterate convergence, have pinpointed exactly why certain advanced algorithms achieve rapid iteration-wise convergence while others fail catastrophically, even within the highly-regarded family of optimistic methods. The defining algorithmic characteristic required to break limit cycles and achieve uniform last-iterate convergence is termed "forgetfulness".

### Non-Forgetful Algorithms: The Failure of OFTRL and OMWU

Optimistic Multiplicative Weights Update (OMWU) has long been considered a premier algorithm due to its logarithmic dependence on the payoff matrix size and its ability to guarantee sublinear regret even in complex general-sum games. OMWU belongs to the broader class of Optimistic Follow-the-Regularized-Leader (OFTRL) algorithms.

The defining structural trait of the OFTRL update rule is its absolute reliance on the cumulative loss vector $L_x^{t-1} = \sum_{k=1}^{t-1} l_x^k$. The algorithm selects its next parameter state by solving the optimization problem: $x^t = \arg\min_{x \in \mathcal{X}} \{ \langle x, L_x^{t-1} + l_x^{t-1} \rangle + \frac{1}{\eta} R(x) \}$

This heavy reliance on the entirety of the cumulative history renders all OFTRL algorithms strictly non-forgetful. To demonstrate the fatal flaw of this mathematical property, Cai et al. constructed a highly adversarial $2 \times 2$ zero-sum game matrix $A_\delta$ parameterized by a small constant $\delta \in (0, 1/2)$. This specific matrix possesses a unique Nash Equilibrium that is heavily skewed toward the extreme boundary of the probability simplex.

When OMWU operates on this $A_\delta$ landscape, it undergoes distinct, highly problematic stages. Initially, the algorithm behaves as expected; the gradients correctly push the dynamic trajectory toward the equilibrium. However, because the equilibrium lies so close to the boundary, the $x$-player must maintain a specific, extreme strategy for a long duration to reach it. In doing so, the OFTRL framework builds up a massive "memory" of cumulative losses indicating that one specific action is vastly superior to the other.

When the dynamics finally cross the threshold of the Nash Equilibrium, the local gradient field reverses direction, signaling that the current strategy is no longer optimal. In a forgetful algorithm, the iterate would immediately respond to this new gradient and stabilize at the equilibrium point. But OMWU cannot stop. Its massive cumulative historical memory entirely overrides the subtle signals of the local gradient. Because it cannot quickly forget the past, it takes thousands of subsequent iterations (scaling proportionally with $1/\delta$) for the new, opposite gradients to slowly chip away at and cancel out the historical memory. This latency causes the algorithm to massively overshoot the Nash Equilibrium, propelling the parameters away from the solution and forcing them into a wide, semi-elliptical cycle.

Because of this inherent non-forgetfulness, the duality gap of OMWU remains at a constant $\Omega(1)$ even after an arbitrary number of rounds. Cai et al. rigorously proved that there is absolutely no function $f(d_1, d_2, T)$ that can guarantee a game-independent uniform last-iterate convergence rate for OMWU. This negative result is not limited to OMWU; it extends mathematically to any OFTRL algorithm utilizing non-forgetful regularizers, including negative entropy, Tsallis entropy, and log barrier regularizers.

### Forgetful Algorithms: The Success of OGDA

In stark contrast, Optimistic Gradient Descent Ascent (OGDA) is formulated as an instance of Optimistic Online Mirror Descent (OOMD) utilizing a squared Euclidean norm regularizer. The OOMD update rule is structurally distinct from FTRL; it relies exclusively on the most recent gradients to perturb the current state, rather than optimizing over the cumulative sum of all past losses: $x^t = \arg\min_{x \in \mathcal{X}} \{ \eta \langle x, l_x^{t-1} \rangle + D_R(x, \hat{x}^t) \}$

This specific formulation inherently allows OGDA to "forget" the distant past immediately. When OGDA approaches the highly skewed equilibrium of the $A_\delta$ matrix, it reacts instantly to the reversing local vector field. Unburdened by historical momentum, it does not overshoot the target. Consequently, OGDA successfully spirals inward, guaranteeing a uniform, game-independent last-iterate convergence rate of $O(1/\sqrt{T})$. This analysis establishes definitively that the lack of forgetfulness—the act of building up too much accumulated negative regret—is mathematically fatal for last-iterate convergence in multi-agent learning.

## The Taxonomy and Separation of Convergence Modes

The revelation that premier algorithms like OMWU fundamentally fail at uniform last-iterate convergence prompted a highly granular reevaluation of how convergence is measured and reported in literature. As previously defined, convergence is categorized into Last-Iterate, Random-Iterate, and Best-Iterate modes, with guarantees classified as either Uniform or Universal.

Historically, for highly forgetful algorithms like OGDA, there was no theoretical necessity to distinguish between these modes. OGDA achieves a uniform polynomial rate of $O(T^{-1/2})$ across best, random, and last-iterate metrics simultaneously. However, the 2025 analysis by Cai et al. formally proved a rigorous mathematical separation between these modes for OMWU, shattering the conventional wisdom that random-iterate convergence can serve as a reliable proxy for best-iterate convergence.

### The Separation Theorems for OMWU

**1. Separation Between Last-Iterate and Best-Iterate:** While OMWU has a proven universal last-iterate convergence bound, its uniform last-iterate convergence is strictly bounded below by $\Omega(1)$ due to the severe overshooting caused by its non-forgetful nature. Despite this absolute failure at the last iterate, the authors mathematically proved that OMWU achieves a strict, uniform best-iterate convergence rate of $O(T^{-1/6})$ in $2 \times 2$ games. This proves a critical reality: even when a parameter sequence as a whole fails to converge uniformly and oscillates wildly, the algorithm still periodically passes extremely close to the Nash Equilibrium, satisfying the best-iterate criteria.

**2. Separation Between Random-Iterate and Best-Iterate:** Surprisingly, while OMWU achieves this polynomial best-iterate rate, it completely fails to achieve a uniform polynomial random-iterate rate. The theoretical lower bound for OMWU's uniform random-iterate convergence is established at $\Omega(1/\log T)$. This separation occurs because the vast majority of the algorithm's time is spent orbiting in regions with a very high duality gap (the wide semi-ellipses caused by overshooting). These long durations of high error vastly outweigh the fleeting moments the algorithm spends passing near the equilibrium, driving up the expected average error and destroying the random-iterate bound.

To mathematically prove the uniform best-iterate rate of $O(T^{-1/6})$ despite the complete failure of the random-iterate convergence, researchers utilized a highly novel two-phase analysis :

- **The Initial Phase:** OMWU is analyzed during its earliest iterations and is shown to exhibit exceptionally fast uniform convergence to an iterate possessing a tight duality gap of $O(\delta)$.
    
- **The Global Phase:** The researchers establish a connection between random-iterate convergence and dynamic interval regret, yielding a _universal_ random-iterate rate of $O(T^{-1/4}\delta^{-1/2})$ that is highly dependent on $\delta$.
    
- By taking the mathematical minimum of the bounds achieved in these two distinct phases ($\min\{\delta, T^{-1/4}\delta^{-1/2}\}$), the dependence on the game-specific condition number $\delta$ cancels out entirely, yielding the robust, purely uniform $O(T^{-1/6})$ best-iterate guarantee.
    

|**Convergence Mode**|**OGDA (Uniform Rate)**|**OMWU (Uniform Rate)**|**OMWU (Universal Rate)**|
|---|---|---|---|
|**Last-Iterate**|$O(T^{-1/2})$|$\Omega(1)$ (Fails)|$O(\exp(-T/C)C)$|
|**Random-Iterate**|$O(T^{-1/2})$|$\Omega(1/\log T)$ (Fails)|$O(T^{-1/4}\delta^{-1/2})$|
|**Best-Iterate**|$O(T^{-1/2})$|**$O(T^{-1/6})$** (Succeeds)|$O(T^{-1/6})$|

## Empirical Validation: Failures of Ergodic Averaging in GANs and MARL

The theoretical impossibility of relying on ergodic convergence in non-convex environments is vividly corroborated by severe empirical failures in state-of-the-art deep learning architectures.

### Multi-Agent Reinforcement Learning (MARL)

In MARL, particularly in extensive-form games modeled by imperfect information, algorithms such as Counterfactual Regret Minimization (CFR) and Regret Matching (RM) heavily rely on iterate averaging to secure $O(1/\sqrt{T})$ convergence to Nash equilibria. However, when MARL is deployed in complex, high-dimensional, and safety-critical environments—such as StarCraft II micromanagement tasks, simulated autonomous driving, or decentralized sensor networks—deploying an averaged policy leads to catastrophic functional failures.

The underlying issue is environmental non-stationarity and the cascading nature of domino effects within multi-agent interactions. If a MARL agent learns a highly specialized evasion maneuver at training iteration 1000 and a completely different, cooperative formation maneuver at iteration 2000, averaging the neural network weights of these two distinct policies does not produce a coherent hybrid capability; it produces an incoherent, fractured action space that ignores the specific transition dynamics required to execute either maneuver successfully.

Furthermore, empirical forensics on cascading failures in MARL demonstrate the fragility of these averaged policies. Shefin et al. (2026) developed a two-stage gradient-based framework to analyze failure detection, utilizing Taylor-remainder analysis on policy-gradient costs to detect "Patient-0" anomalies and tracing directional second-order curvature to map contagion graphs. Their analysis reveals that uncoordinated, averaged actions trigger massive domino effects that amplify upstream deviations across the network, leading to rapid system collapse. Because actual, real-world deployment evaluates a single trajectory executed by the final policy network, the expected theoretical value of an infinitely long, averaged training trajectory is mathematically uninformative and practically dangerous. Thus, safety-critical MARL explicitly requires last-iterate convergence provided by advanced, forgetful algorithms.

### Generative Adversarial Networks (GANs)

GAN training represents the quintessential non-convex-non-concave min-max game that shattered the utility of ergodic convergence. The generator attempts to minimize the divergence between generated distributions and real data, while the discriminator maximizes its ability to distinguish between them. When standard SGDA is applied to this landscape, the parameters frequently enter limit cycles, manifesting empirically as the notorious "mode collapse" phenomenon. The generator continuously cycles through producing specific, narrow subsets of the data distribution (e.g., generating only images of the digit '1', then shifting to only generating '7's).

If practitioners attempt to apply the classical ergodic solution to stop the cycling by simply averaging the weights of the generator across these epochs, the resulting parameter set $\theta_{avg}$ maps to a region of the neural network's loss landscape that possesses massive loss. Because the landscape lacks convexity—$L_{GAN}(\frac{\theta_1 + \theta_2}{2}) \not\leq \frac{L_{GAN}(\theta_1) + L_{GAN}(\theta_2)}{2}$—the averaged generator produces meaningless static and noise, completely failing the image generation task. This empirical reality definitively validates the necessity of last-iterate optimization techniques, such as Extra-gradient and OGDA, which actively manipulate the vector field to shrink the limit cycles and stabilize the final, localized generator parameters.

## The "Weight Averaging" Exception: SWA and EMA

Given the catastrophic failure of ergodic averaging across non-convex neural network boundaries, a prominent and fascinating paradox exists in modern deep learning literature: the extraordinary empirical success of localized averaging techniques like Stochastic Weight Averaging (SWA) and Exponential Moving Average (EMA). If averaging parameters is mathematically unsound in non-convex landscapes, why do these specific variations of averaging yield state-of-the-art generalization?

### Stochastic Weight Averaging (SWA) and Linear Mode Connectivity

SWA operates by running a standard Stochastic Gradient Descent (SGD) process with a modified, cyclical, or unusually high-constant learning rate. It then computes an equal, simple average of the weights traversed strictly during these specific, late-stage training epochs. The theoretical justification for why SWA succeeds spectacularly where standard ergodic averaging fails lies in the topological concept of Linear Mode Connectivity.

In standard ergodic averaging, the average is taken across the entire training history, from $t=1$ to $T$. Because the initial parameters are entirely random and far from the optimum, the long-term trajectory crosses multiple distinct, isolated high-loss barriers. The global mathematical average of this entire path inevitably falls into a high-loss barrier region between optima.

SWA, conversely, begins its averaging process only _after_ the neural network has successfully converged to a general, low-loss basin. Recent discoveries by Garipov, Izmailov, and colleagues regarding loss landscape geometry reveal that local optima in heavily over-parameterized neural networks are not isolated, sharp points. Rather, they form connected manifolds or "flat basins" where paths between various optima maintain a near-constant, exceptionally low loss.

By utilizing a high-constant learning rate at the end of training, SWA intentionally prevents the network from settling down into the sharp, specific boundaries of the loss basin. Instead, it forces the SGD trajectory to aggressively explore the perimeter of the connected low-loss manifold. When SWA averages these specific, perimeter weights, it acts as a highly effective geometric interpolator. Because the sampled points all lie on the boundary of the exact same convex-like basin, their arithmetic average falls directly into the geometric center of the flat region. This central, flat minimum provides massive variance reduction and is highly robust to the inevitable data shifts between training and testing sets. Thus, SWA achieves superior generalization by averaging _locally_ within a connected mode, entirely avoiding the violation of non-convex constraints that plagues global ergodic averaging.

|**Feature**|**Global Ergodic Averaging**|**Stochastic Weight Averaging (SWA)**|
|---|---|---|
|**Averaging Window**|Entire training history ($t=1$ to $T$)|Late-stage history ($t_{start}$ to $T$)|
|**Learning Rate Profile**|Standard decaying learning rate|Cyclical or High-Constant learning rate|
|**Landscape Topology**|Crosses multiple non-convex barriers|Interpolates strictly within a flat, connected mode|
|**Resulting Model Efficacy**|High loss, "garbage" functionality|Highly robust, centered flat minimum with high generalization|

### Exponential Moving Average (EMA) in Min-Max Games

While SWA is primarily leveraged to enhance generalization in minimization problems, Exponential Moving Average (EMA) serves a distinctly different mathematical function to stabilize the adversarial min-max setting of GANs. EMA computes an exponentially discounted sum of past parameters, heavily weighting the most recent iterations and rapidly forgetting the distant past.

In simple bilinear min-max games, standard Moving Average (MA) is sufficient to guarantee convergence to the equilibrium. However, as established, MA fails miserably when applied to the non-convex settings of deep neural networks. Yazici et al. provided a rigorous theoretical proof demonstrating that EMA fundamentally alters the game dynamics in a way that MA does not. While EMA does not perfectly converge to a static point in bilinear games, it mathematically forces the trajectory to converge to limit cycles around the equilibrium with _vanishing amplitude_ as the discount parameter approaches one.

In the highly non-convex-concave setting of actual GAN training, this vanishing amplitude property allows EMA to act as a powerful stabilizer. By acting as a sophisticated low-pass filter on the oscillating trajectories of the generator and discriminator, EMA aggressively dampens the high-frequency chaotic rotations that lead to mode collapse. Unlike standard ergodic averaging, which pulls the parameters far away from the optimum into high-loss space, EMA’s rapid exponential decay ensures the parameters remain highly localized to the most recent, functional low-loss subspace. Consequently, EMA produces significantly more stable and balanced support regions, preventing mode collapse and generating state-of-the-art Inception and FID scores without explicitly requiring the expensive double-oracle calls associated with Extra-gradient methods.

## Conclusion

The profound transition from ergodic to iteration-wise (last-iterate) convergence marks one of the most critical evolutions in modern optimization theory. This shift was driven entirely by the mathematical realities of the high-dimensional, non-convex-non-concave landscapes found in contemporary deep learning architectures and multi-agent systems. Classical algorithms that rely on historical averaging, such as standard FTRL and MWU, are mathematically incompatible with the deployment requirements of GANs and MARL due to the non-linear geometry of their loss functions, where $L(\frac{x+y}{2}) \not\leq \frac{L(x)+L(y)}{2}$.

Rigorous theoretical frameworks established by Daskalakis, Gidel, and Letcher demonstrated that achieving global convergence in multi-loss optimization is mathematically impossible, as standard algorithms are inherently vulnerable to spurious attractors, strict maxima avoidance paradoxes, and the rotational divergence induced by the antisymmetric components of game Jacobians. Escaping these limit cycles requires specialized vector field modifications. Algorithms must be explicitly engineered with "forgetfulness"—such as Optimistic Gradient Descent Ascent (OGDA) or Extra-gradient (EG) methods—to ensure they react instantly to local curvature rather than being weighed down by massive historical momentum. The rigorous separation of convergence modes further highlights that while non-forgetful algorithms like OMWU may occasionally hit optimal states to satisfy best-iterate convergence, they fundamentally lack the mechanism required for reliable uniform last-iterate convergence.

While parameter averaging generally fails catastrophically across non-convex boundaries, highly localized, late-stage averaging techniques like SWA and EMA successfully circumvent this failure by exploiting the topological phenomenon of mode connectivity. By intelligently averaging parameters strictly within flat, continuous low-loss basins, these techniques provide immense variance reduction and optimal generalization without triggering the divergence associated with global ergodic averaging. Ultimately, ensuring last-iterate convergence through predictive, forgetful gradients remains the absolute cornerstone for reliably scaling complex adversarial and safety-critical multi-agent AI architectures into the future.