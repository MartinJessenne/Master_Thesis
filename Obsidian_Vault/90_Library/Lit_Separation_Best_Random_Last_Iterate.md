# On Separation Between Best-Iterate, Random-Iterate, and Last-Iterate Convergence of Learning in Games

Yang Cai¹, Gabriele Farina², Julien Grand-Clément³, Christian Kroer´, Chung-Wei Leeµ, Haipeng Luoµ, and Weiqiang Zheng¹

¹ Yale University, {yang.cai, weiqiang.zheng}@yale.edu
² MIT, gfarina@mit.edu
³ HEC Paris, grand-clement@hec.fr
´ Columbia University, christian.kroer@columbia.edu
µ University of Southern California, {haipeng.luo, lee.chung}@usc.edu

March 5, 2025

### Abstract

Non-ergodic convergence of learning dynamics in games is widely studied recently because of its importance in both theory and practice. Recent work (Cai et al., 2024) showed that a broad class of learning dynamics, including Optimistic Multiplicative Weights Update (OMWU), can exhibit arbitrarily slow last-iterate convergence even in simple $2\times2$ matrix games, despite many of these dynamics being known to converge asymptotically in the last iterate. It remains unclear, however, whether these algorithms achieve fast non-ergodic convergence under weaker criteria, such as best-iterate convergence. We show that for $2\times2$ matrix games, OMWU achieves an $O(T^{-1/6})$ best-iterate convergence rate, in stark contrast to its slow last-iterate convergence in the same class of games. Furthermore, we establish a lower bound showing that OMWU does not achieve any polynomial random-iterate convergence rate, measured by the expected duality gaps across all iterates. This result challenges the conventional wisdom that random-iterate convergence is essentially equivalent to best-iterate convergence, with the former often used as a proxy for establishing the latter. Our analysis uncovers a new connection to dynamic regret and presents a novel two-phase approach to best-iterate convergence, which could be of independent interest.

---

### 1 Introduction

No-regret learning dynamics provide one of the premier ways of computing equilibria in multiplayer interactions (Cesa-Bianchi & Lugosi, 2006). They have been successfully deployed at scale across a wide variety of games and desired notions of equilibrium, and have been an integral part of superhuman AI for poker (Brown & Sandholm, 2018, 2019; Moravčík et al., 2017), human-level AI for Stratego (Perolat et al., 2022) and Diplomacy (Meta et al., 2022), as well as other uses such as alignment of large language models (Munos et al., 2023; Jacob et al., 2023).

In general, learning dynamics guarantee convergence to equilibrium in an ergodic sense. In other words, it is not the actual behavior of the dynamics that converges to equilibrium, but rather their average behavior. Overcoming this limitation and establishing convergence in iterates in the two-player zero-sum setting has been an important direction of research in the past decade. The reasons for this endeavor are multifaceted. First, dynamics that exhibit iterate convergence properties ensure that the learning agents will eventually play according to an equilibrium strategy. This is a desirable requirement for deploying learning in an online setting, where it is important that agents eventually sample actions from an optimal strategy. Second, algorithms with converging iterates rule out undesirable phenomena such as recurrence and even formally chaotic behavior (Sato et al., 2002; Mertikopoulos et al., 2018). Finally, the construction of learning algorithms with iterate convergence guarantees is important for applications in nonconvex optimization where averaging the iterates is not possible (Daskalakis et al., 2017).

In principle, at least three notions of iterate convergence can be identified, listed next from strongest to weakest.

* **Last-iterate convergence**, meaning that the learner's strategies approach the set of equilibrium strategies over time.
* **Random-iterate convergence**, meaning that by sampling an iteration uniformly at random, the learner's strategy is close to equilibrium.
* **Best-iterate convergence**, meaning that there exists a subsequence of strategies used by the learner that converges to the set of equilibrium strategies.

In these definitions, the proximity of the strategies to equilibrium is measured by the duality gap.

Orthogonal to the mode of convergence above, is the speed (i.e., non-asymptotic rate) of convergence, particularly regarding the dependence on possible condition numbers of the payoff matrix of the two-player zero-sum game. In this regard, we identify two types of results.

* **Uniform convergence results** give a uniform upper bound on the convergence rate that applies to any game instance.
* **Universal convergence results** also apply to any game instance, but it can include a possibly arbitrarily bad dependence on some form of condition number of the game for example, the smallest nonzero probability used in equilibrium.

Examples of universal and uniform convergence rates are given in Table 1. Clearly, universal convergence guarantees for learning dynamics are easier to establish than uniform convergence guarantees. In fact, most modern optimistic learning algorithms are known to enjoy some form of universal last-iterate convergence, albeit with a dependence on some form of condition numbers of the game, which can typically be arbitrarily large even for $2\times2$ games (Tseng, 1995; Mordukhovich et al., 2010; Wei et al., 2021).

| Convergence | OGDA (uniform) | OMWU (uniform) | OMWU (universal) |
| :--- | :--- | :--- | :--- |
| Last iterate | | $\Omega(1)$ | $O(exp(-\frac{T}{C})C)^{\top}$ |
| Random iterate | $O(T^{-1/2})$ | $\Omega(\frac{1}{\log T})$ | $O(T^{-\frac{1}{4}}δ^{-\frac{1}{2}})$ |
| Best iterate | | | $O(T^{-\frac{1}{6}})^{\pm}$ |

**Table 1:** Uniform and universal convergence rates of OGDA and OMWU in zero-sum games with a fully mixed equilibrium. Here $δ > 0$ is the minimum probability in the Nash equilibrium. Universal convergence rates allow for dependence on $δ$ while uniform convergence rates are $δ$-independent (the constants only depend on the dimensions of the problems and the norm of the payoff matrix). Our new results are highlighted in gray. $\dagger: C := \Omega(exp(\frac{1}{δ}))$. $\pm$: This upper bound only holds for $2\times2$ games.

The situation regarding uniform convergence results is significantly less understood. So far in the literature, there has never been an incentive to treat the three notions of uniform iterate convergence separately. For example, while $O(T^{-\frac{1}{2}})$ uniform random-iterate convergence guarantees for Optimistic Gradient Descent Ascent (OGDA) (Popov, 1980) have appeared earlier (Wei et al., 2021; Anagnostides et al., 2022), they have been eventually strengthened to hold in the last iterate sense (Cai et al., 2022; Gorbunov et al., 2022). For OGDA, the uniform convergence rates for last-, random-, and best-iterate are polynomial and essentially identical. Furthermore, prior to our work, no algorithm was known to exhibit a separation between these three convergence modes.

However, cracks in the above state of affairs have started to emerge recently. In an unexpected turn of events, a recent paper by Cai et al. (2024) has shown that optimistic multiplicative weights update (OMWU)€”a premier no-regret algorithm with otherwise best-in-class theoretical guarantees (Rakhlin & Sridharan, 2013; Daskalakis et al., 2021; Farina et al., 2022)€”does not enjoy any uniform last-iterate convergence guarantees, despite the existing known universal last-iterate convergence rates (Wei et al., 2021). Crucially, that result only applies to uniform last-iterate convergence, and does not preclude the possibility that OMWU might still enjoy uniform random-iterate or best-iterate convergence guarantees. This suggests the following question:

*Does OMWU enjoy uniform random- or best-iterate convergence, despite the recently established negative result regarding its lack of uniform last-iterate convergence?*

Answering this question is the main contribution of this paper.

### Contributions and Techniques

In this paper, we show a separation between the best-iterate and the random-iterate in terms of uniform convergence for OMWU in two-player zero-sum games, which also implies a separation between best- and last-iterate uniform convergence in the same setting. The separation is established by (i) a new lower bound for the uniform random-iterate convergence and (ii) a new upper bound for the uniform best-iterate convergences. Our results are summarized in Table 1.

**Lower bounds** On the negative side, we show that OMWU does not enjoy a polynomial uniform random-iterate convergence guarantee (Theorem 1). Our analysis also extends to the broader family of Optimistic Follow-The-Regularized-Leader (OFTRL) algorithms with the most popular regularizers and results in new lower bounds (Theorem 2). These lower bound results hold even for $2\times2$ games with a fully-mixed Nash equilibrium, and we give a numerical illustration in Figure 1.

**Upper bounds** On the positive side, we prove that OMWU has an $O(T^{-\frac{1}{6}})$ uniform best-iterate convergence rate (Theorem 3) for $2\times2$ games with a fully-mixed Nash equilibrium. We note that for the same class of games, OMWU has no uniform last-iterate convergence (Cai et al., 2024) and no uniform polynomial random-iterate convergence, as we discussed above. This result has some important consequences:

* It partially counters the negative narrative on OMWU's convergence properties from (Cai et al., 2024), by offering a positive, if slightly weaker, result that a uniform polynomial best-iterate convergence rate is possible. Extending our positive result beyond the 2-by-2 case is an interesting future direction.
* It shows, for the first time and on one of the most important algorithms in online learning, that uniform best-, random-, and last-iterate convergence properties need not go hand-in-hand, in contrast to all existing results in the literature, and thus that different techniques are necessary to study random-iterate and best-iterate convergence.

**Techniques** As mentioned above, our positive result on the uniform polynomial best-iterate convergence rate of OMWU does not follow the common approach of showing uniform polynomial random-iterate convergence since our negative result precludes the latter. To sidestep this obstacle, we develop a novel two-phase approach that we believe might be of independent interest for the study of best-iterate convergence properties beyond OMWU.

In the global phase, we establish a global universal random-iterate convergence rate $O(T^{-\frac{1}{4}}δ^{-\frac{1}{2}})$ that has dependence on the minimum probability $δ > 0$ among actions in the support of Nash equilibrium (Theorem 4). To prove the result, we leverage the connection between random-iterate convergence, dynamic regret, and interval regret, which is new for proving random-iterate convergence. We remark that this result also holds for the general $d_{1} \times d_{2}$ case and provides an exponential improvement to the best known bound on the last-iterate convergence rate of $O(exp(\frac{1}{δ}) \cdot (1+exp(-\frac{1}{δ}))^{-T})$ (Wei et al., 2021) in the dependence on the condition number $δ$.

In the initial phase, we show that OMWU has fast uniform convergence to one iterate with duality gap $O(δ)$ (Theorem 5). Combining results in the two phases and the definition of the best-iterate convergence, we get a uniform $\min\{δ, T^{-\frac{1}{4}}δ^{-\frac{1}{2}}\} = T^{-\frac{1}{6}}$ best-iterate convergence rate that is independent of $δ$.

### 2 Preliminaries

Let $δ^{d} \subseteq \mathbb{R}^{d}$ be the d-dimension probability simplex. For a strictly convex regularizer $R: \mathcal{X} \rightarrow \mathbb{R}$ we denote its Bregman divergence as $D_{R}(x, x') = R(x) - R(x') - \langle \nabla R(x'), x - x' \rangle$.

We study online learning dynamics in a two-player zero-sum matrix game $\min_{x \in δ^{d_{1}}} \max_{y \in δ^{d_{2}}} x^{\top}Ay$ with loss matrix $A \in [0, 1]^{d_{1} \times d_{2}}$. In each iteration $t \ge 1$, the x-player chooses a mixed strategy $x^{t} \in \mathcal{X} = δ^{d_{1}}$ and the y-player chooses a mixed strategy $y^{t} \in \mathcal{Y} = δ^{d_{2}}$. Then the x-player receives loss vector $l_{x}^{t} = Ay^{t}$ while the y-player receives loss vector $l_{y}^{t} = -A^{\top}x^{t}$. The goal is convergence to a Nash equilibrium $(x^{*}, y^{*})$ where $x^{*} \in \text{argmin}_{x \in \mathcal{X}} \max_{y \in \mathcal{Y}} x^{\top}Ay$ and $y^{*} \in \text{argmax}_{y \in \mathcal{Y}} \min_{x \in \mathcal{X}} x^{\top}Ay$. A Nash equilibrium $(x^{*}, y^{*})$ is fully mixed if $x^{*}$ and $y^{*}$ both have full support. The proximity of a strategy profile $(x, y)$ to Nash equilibrium is measured by its duality gap:

$$DualityGap(x, y) = \max_{y' \in \mathcal{Y}} x^{\top}Ay' - \min_{x' \in \mathcal{X}} x'^{\top}Ay.$$

The duality gap $DualityGap(x, y)$ is nonnegative and equals zero if and only if $(x, y)$ is a Nash equilibrium of the game A.

**Online learning dynamics** We denote by $L_{x}^{t} = \sum_{k=1}^{t} l_{x}^{k}$ and $L_{y}^{t} = \sum_{k=1}^{t} l_{y}^{k}$ the cumulative loss vectors. The update rule of the Optimistic Follow-the-Regularized-Leader (OFTRL) algorithm (Syrgkanis et al., 2015) with regularizer R and step size $\eta > 0$ is: initialize $(x^{1}, y^{1})$ both as the uniform distribution, then for each $t > 2$,

$$x^{t} = \text{argmin}_{x \in \mathcal{X}} \{ \langle x, L_{x}^{t-1} + l_{x}^{t-1} \rangle + \frac{1}{\eta} R(x) \}$$
$$y^{t} = \text{argmin}_{y \in \mathcal{Y}} \{ \langle y, L_{y}^{t-1} + l_{y}^{t-1} \rangle + \frac{1}{\eta} R(y) \}.$$
(OFTRL)

Commonly-studied regularizers include the following.
* Negative entropy, $R(x) := \sum_{i=1}^{d} x[i] \log x[i]$. The resulting OFTRL algorithm is OMWU.
* Squared Euclidean norm, $R(x) := \frac{1}{2} \sum_{i=1}^{d} x[i]^{2}$.
* Log barrier, $R(x) := \sum_{i=1}^{d} -\log(x[i])$, also called the "log regularizer".
* Negative Tsallis entropy family of regularizers, $R(x) := (1 - \sum_{i=1}^{d} (x[i])^{\beta}) / (1 - \beta)$, parameterized by $\beta \in (0, 1)$.

Another popular family of online learning algorithms is the Optimistic Online Mirror Descent (OOMD) algorithm (Rakhlin & Sridharan, 2013). We introduce some notations to simplify the presentation. We let $z = (x, y) \in \mathcal{Z} := δ^{d_{1}} \times δ^{d_{2}}$ and define the gradient operator $F(z^{t}) = (l_{x}^{t}, l_{y}^{t}) = (Ay^{t}, -A^{\top}x^{t})$. We also let $D_{R}(z, z') = D_{R}(x, x') + D_{R}(y, y')$. The update rule of OOMD is to initialize $z^{1} = \hat{z}^{1}$ and for all $t \ge 2$:

$$\hat{z}^{t} = \text{argmin}_{z \in \mathcal{Z}} \{ \eta \langle z, F(z^{t-1}) \rangle + D_{R}(z, \hat{z}^{t-1}) \}$$
$$z^{t} = \text{argmin}_{z \in \mathcal{Z}} \{ \eta \langle z, F(z^{t}) \rangle + D_{R}(z, \hat{z}^{t}) \}$$
(OOMD)

We remark that when $R(x) = \frac{1}{2} \|x\|^{2}$ the resulting OOMD algorithm is also called the Optimistic Gradient Descent Ascent (OGDA) algorithm, and is different from the OFTRL algorithm instantiated with the same regularizer. However, for Legendre regularizers, including the entropy regularizer, the log regularizer, and the family of Tsallis entropy regularizers, their OFTRL and OOMD versions coincide in the sense that the iterates $\{z^{t} = (x^{t}, y^{t})\}$ produced by OFTRL and OOMD are the same.

Throughout the paper, we assume all the algorithms are initialized with the uniform distribution. This paper's main focus is the Optimistic Multiplicative Weights Update (OMWU) algorithm, though we also give several results for the broader class of OFTRL algorithms. OMWU is OFTRL/OOMD instantiated with the negative entropy regularizer $R = \sum_{i=1}^{d} x[i] \log x[i]$. OMWU admits the following closed-form update:

$$x^{t}[i] \propto x^{t-1}[i] \cdot \exp(-\eta L_{x}^{t-1}[i] - \eta l_{x}^{t-1}[i]), i \in [d_{1}]$$
$$y^{t}[j] \propto y^{t-1}[j] \cdot \exp(-\eta L_{y}^{t-1}[j] - \eta l_{y}^{t-1}[j]), j \in [d_{2}].$$

**Notions of convergence** We focus on three types of convergence rates. For zero-sum games $A \in [0, 1]^{d_{1} \times d_{2}}$, we say an algorithm has a uniform last-iterate, random-iterate, or best-iterate convergence rate $O(f(T))$ (we omit dependence on $d_{1}, d_{2}$ here) if there exists a constant $c > 0$ such that for any game, any time $T \ge 1$ we have

* Last-iterate: $DualityGap(z^{T}) \le cf(T)$
* Random-iterate: $\mathbb{E}_{t \sim Uni[1,T]} [DualityGap(z^{t})] \le cf(T)$
* Best-iterate: $\min_{t \in [1,T]} [DualityGap(z^{t})] \le cf(T)$

for $z^{T} := (x^{T}, y^{T})$. By definition, a last-iterate convergence rate upper bounds the duality gap for every iterate $t \in [1, T]$; a random-iterate convergence rate upper bounds the average duality gap during time $[1, T]$, which is also the average social dynamic regret (see Proposition 1); and a best-iterate convergence rate upper bounds the smallest duality gap in time $[1, T]$. Clearly, last-iterate convergence is stronger than random-iterate convergence, and random-iterate convergence is stronger than best-iterate convergence. Yet, for OGDA, these three types of convergence admit similar rates (see Table 1). Establishing a separation between these types of convergence for OMWU is the main focus of our paper.

### 3 Lower Bounds for Random-Iterate Convergence

In this section, we establish lower bounds for the random-iterate convergence rate of OFTRL dynamics in two-player zero-sum games. For OMWU, we present an impossibility result for a uniform polynomial convergence rate by establishing an $\Omega(\frac{1}{\ln T})$ lower bound, even in a simple class of $2\times2$ matrix games.

**Theorem 1.** *For two-player zero-sum games with loss matrix $A \in [0, 1]^{2\times2}$, the uniform random-iterate convergence rate of OMWU with any constant step size $\eta \le \frac{1}{2}$ is $\Omega(\frac{1}{\log T})$. This result continues to hold if we restrict the space of loss matrices to games with fully-mixed Nash equilibria.*

**Remark 1.** *We note that the lower bound holds for general $d_{1} \times d_{2}$ games by a reduction given in Cai et al. (2024, Theorem 3).*

Theorem 1 has the following implications:
1. For any $T > 1$, we can find a game instance $A \in [0, 1]^{2\times2}$ such that OMWU on that game has nearly linear in T social dynamic regret: $\sum_{t=1}^{T} DualityGap(x^{t}, y^{t}) = \Omega(\frac{T}{\log T})$.
2. For any $\epsilon > 0$, we can find a game instance $A \in [0, 1]^{2\times2}$ such that OMWU on that game suffers $\frac{1}{T} \sum_{t=1}^{T} DualityGap(x^{t}, y^{t}) \ge \epsilon$ even when $T = \Omega(\exp(\frac{1}{\epsilon}))$.

Next we present lower bounds on the random-iterate convergence rate for OFTRL instantiated with the squared Euclidean norm, the log barrier, and the family of Tsallis entropy regularizers.

**Theorem 2.** *For two-player zero-sum games $[0, 1]^{2\times2}$, the following lower bounds hold for the uniform random-iterate convergence rate of OFTRL with constant step size:*
1. $\Omega(1)$ *for the squared Euclidean norm regularizer*
2. $\Omega(T^{-\frac{1-\beta}{2-\beta}})$ *for the Tsallis entropy regularizer parameterized by $\beta \in (0, 1)$*
3. $\Omega(T^{-\frac{1}{2}})$ *for the log regularizer*
*These results continue to hold if we restrict the space of loss matrices to games with fully-mixed Nash equilibria.*

To the best of our knowledge, Theorem 1 and Theorem 2 are the first lower bound results for the random-iterate convergence rate of learning in games. Our lower bounds also offer insights into the relation between the random-iterate and the last-iterate convergence. By definition, we know that random-iterate convergence is a weaker requirement than last-iterate convergence. However, the $\Omega(1)$ lower bound for OFTRL with squared Euclidean norm shows that the uniform random-iterate convergence can still be arbitrarily slow. This matches the lower bound on its last-iterate convergence proved in Cai et al. (2024), thereby demonstrating that random-iterate convergence can be as hard as last-iterate convergence.


#### 3.1 Proof Overview

We present a high-level overview of the proofs of Theorem 1 and Theorem 2 here. The full proofs appear in Appendix A. We focus on a class of hard instances introduced in Cai et al. (2024) that has a fully-mixed Nash equilibrium:

$$A_{δ} := \begin{pmatrix} \frac{1}{2} + δ & \frac{1}{2} \\ 0 & 1 \end{pmatrix} \quad \forall δ \in (0, \frac{1}{2}). \quad (1)$$

Cai et al. (2024) establish an $\Omega(1)$ lower bound on the uniform last-iterate convergence rate for OFTRL dynamics with all the aforementioned regularizers. Specifically, they show that for any sufficiently-small $δ > 0$ the OFTRL dynamics on $A_{δ}$ always have at least one iterate $T = \Omega(1/δ)$ such that $DualityGap(x^{T}, y^{T}) = \Omega(1)$. Therefore, a uniform $o(1)$ last-iterate convergence rate is impossible.

Finding one iterate with a large duality gap is sufficient for the lower bound on the last-iterate convergence. However, that is not sufficient for the weaker notion of random-iterate convergence, which measures the average duality gap:

$$\frac{1}{T} \sum_{t=1}^{T} DualityGap(x^{t}, y^{t}). \quad (2)$$

Instead, to show that (2) is large, we must prove that a substantial proportion of iterations in $\{1, \dots, T\}$ all have large duality gap. Building upon the analysis of the class of games (1) presented in Cai et al. (2024), we show that for some $T = \Omega(1/δ)$ iterations, there will be a block of $\Theta(\frac{1}{δ})$ iterations each with a constant duality gap:

$$DualityGap(x^{t}, y^{t}) = \Omega(1), \quad \forall t \in [T, T + \Theta(\frac{1}{δ})].$$

As a result, the average duality gap at time $T + \Theta(\frac{1}{δ})$ is at least

$$\frac{1}{T + \Theta(\frac{1}{δ})} \sum_{t=1}^{T + \Theta(\frac{1}{δ})} DualityGap(x^{t}, y^{t}) \ge \frac{\Theta(\frac{1}{δ})}{T + \Theta(\frac{1}{δ})}.$$

If we provide an upper bound on T then we get a lower bound for the random-iterate convergence rate. Providing an upper bound on T requires a careful analysis of the trajectory of the OFTRL learning dynamics and is presented as Theorem 7 in Appendix A. The resulting upper bound is $T = O(\frac{1}{δ} \cdot f_{R}(δ))$ where $f_{R}(δ)$ is a quantity that is related to the stability of the regularizer R. In Lemma 8 in Appendix A, we establish upper bounds for $f_{R}(δ)$. For the entropy regularizer, we have $f_{R}(δ) \le \log \frac{1}{δ}$ which implies the $\Omega(\frac{1}{\log T})$ lower bound. For the squared Euclidean norm, $f_{R}(δ) \le 1$; for the log regularizer, $f_{R}(δ) \le \frac{1}{δ}$; for Tsallis entropy with $\beta \in (0, 1)$, $f_{R}(δ) \le \frac{2\beta}{1-\beta} (\frac{1}{δ})^{1-\beta}$. See Figure 1 for a numerical illustration of the effects of different regularizers on the random-iterate convergence. Combining these results completes the proof for Theorem 1 and 2.

### 4 Best-Iterate Convergence Rate for OMWU

In this section, we establish a polynomial best-iterate convergence rate for OMWU for $2\times2$ matrix games with a fully mixed Nash equilibrium. Our main theorem in this section is the following.

**Theorem 3.** *Consider any matrix game $A \in [0, 1]^{2\times2}$ with a fully mixed Nash equilibrium. Let $\{x^{t}, y^{t}\}_{t \ge 1}$ be the iterates produced by the OMWU dynamics with uniform initialization and constant step size $\eta \le \frac{1}{10}$. Then for any $T \ge 1$, we have*

$$\min_{t \in [1,T]} DualityGap(x^{t}, y^{t}) = O(T^{-\frac{1}{6}}).$$

We note that all existing non-ergodic convergence rates of OMWU involve problem-dependent constants and can be arbitrarily slow (Wei et al., 2021). Our result is the first polynomial problem-independent best-iterate convergence rate for OMWU. An important takeaway of Theorem 3 is that it provides, to our knowledge, the first separation between best-iterate convergence and random-iterate convergence for learning in games. For the same class of $2\times2$ games with a fully mixed Nash equilibrium, our results show that OMWU has an interesting landscape of convergence rates: (1) it does not have a uniform last-iterate convergence rate (Cai et al., 2024); (2) it does not have a polynomial uniform random-iterate convergence rate (Theorem 1); (3) yet it has a polynomial $O(T^{-\frac{1}{6}})$ best-iterate convergence rate (Theorem 3). Our results show a surprising contrast with the uniform convergence results for OGDA, where the last-iterate, best-iterate, and random-iterate convergence rates are similar.

#### 4.1 Proof Overview

To prove Theorem 3, we develop a novel two-phase analysis of the best-iterate convergence rate of OMWU. In this section, we provide a high-level overview of the proof. A more detailed discussion of each phase appears in Section 4.2 and Section 4.3. The full proof is in Appendix B.

In the literature, the most common approach for proving a sublinear best-iterate convergence is using the random-iterate convergence as a proxy (since this is a stronger notion of convergence than best-iterate). Using OGDA as an example, it is known that the sum of duality gaps is sublinear $\sum_{t=1}^{T} DualityGap(x^{t}, y^{t}) = O(\sqrt{T})$. This directly implies $O(T^{-\frac{1}{2}})$ best-iterate convergence rate for OGDA. However, this approach is impossible for OMWU, since we have shown a negative result for the random-iterate convergence rates of OMWU (Theorem 1): for any T, there exists a game instance such that

$$\sum_{t=1}^{T} DualityGap(x^{t}, y^{t}) = \Omega(\frac{T}{\log T})$$

As such, our proof of Theorem 3 requires new insights fully tailored to the uniform best-iterate convergence (and independent of uniform random-iterate convergence). We come up with the following two-phase analysis. Let A be any $2\times2$ game with a fully mixed Nash equilibrium and denote by $δ > 0$ the minimum probability in the Nash equilibrium.

**Global phase** We first prove that for all T, we have a universal $δ$-dependent random-iterate convergence bound:

$$\frac{1}{T} \sum_{t=1}^{T} DualityGap(x^{t}, y^{t}) = O(T^{-\frac{1}{4}}δ^{-\frac{1}{2}}).$$

Note that this does not contradict Theorem 1 since the bound has a dependence on $δ$. It is worth noting that this bound holds for general $d_{1} \times d_{2}$ games with a fully mixed Nash equilibrium. The proof uses a connection between random-iterate convergence, dynamic regret, and interval regret. A detailed discussion is in Section 4.2.

**Initial phase** We then analyze the initial iterations of OMWU. We show that there exists an iteration $T_{1} \ge 1$ such that the following two conditions hold:
1. $DualityGap(x^{T_{1}}, y^{T_{1}}) = O(δ)$.
2. For all $T \in [1, T_{1}]$, a uniform best-iterate convergence rate holds: $\min_{t \in [1, T]} DualityGap(x^{t}, y^{t}) = \tilde{O}(\frac{1}{T})$.

Here, $\tilde{O}(\cdot)$ hides terms logarithmic in T. In summary, we show that the OMWU dynamics will reach an iterate $T_{1}$ with a duality gap of $O(δ)$, and all the initial iterates $[1, T_{1}]$ have a fast best-iterate convergence rate independent of $δ$. A detailed discussion is in Section 4.3.

**Combining the two-phase analysis** For all $T \le T_{1}$, by analysis in the initial phase, we know they have $\tilde{O}(T^{-1})$ best-iterate convergence rate. For all $T \ge T_{1}$, we can combine the analysis in the initial phase and the global phase as follows:

$$\min_{t \in [1,T]} DualityGap(x^{t}, y^{t}) \le \min \{ DualityGap(x^{T_{1}}, y^{T_{1}}), \frac{1}{T} \sum_{t=1}^{T} DualityGap(x^{t}, y^{t}) \}$$
$$\le \min \{ δ, O(T^{-\frac{1}{4}}δ^{-\frac{1}{2}}) \} \le O(T^{-\frac{1}{6}}).$$

Note that the last inequality holds since (1) if $δ \le T^{-\frac{1}{6}}$, then the inequality holds; (2) if $δ \ge T^{-\frac{1}{6}}$, then $T^{-\frac{1}{4}}δ^{-\frac{1}{2}} \le T^{-\frac{1}{6}}$ and the inequality also holds. In this way, we get a uniform $δ$-independent convergence rate.

The preceding analysis demonstrates the effectiveness of our two-phase approach, which leverages the additional flexibility inherent in best-iterate convergence compared to random-iterate convergence. We hope the insight in our approach will help analyze the best-iterate convergence rates of other algorithms.

#### 4.2 Global Phase: Convergence via Minimizing the Social Dynamic Regret

In this subsection, we present a universal random-iterate convergence rate for OMWU on all matrix games $A \in [0, 1]^{d_{1} \times d_{2}}$ that have a fully mixed Nash equilibrium. Let $(x^{*}, y^{*})$ be the fully mixed Nash equilibrium, we denote by $δ = \min \{x^{*}[i], y^{*}[j] : i \in [d_{1}], j \in [d_{2}]\}$ as the minimum probability in the Nash equilibrium. We prove a random-iterate convergence rate of $O(T^{-\frac{1}{4}}δ^{-\frac{1}{2}})$.

For simplicity of analysis, we use the OOMD-type update rule of OMWU in this section, which is equivalent to the OFTRL-type update of OMWU used in other parts of the paper. Recall that the OMWU algorithm initializes $z^{1} = \hat{z}^{1}$ as the uniform distribution and updates for iteration $t \ge 2$ with step size $\eta$:

$$\hat{z}^{t} = \text{argmin}_{z \in \mathcal{Z}} \{ \eta \langle z, F(z^{t-1}) \rangle + KL(z, \hat{z}^{t-1}) \}$$
$$z^{t} = \text{argmin}_{z \in \mathcal{Z}} \{ \eta \langle z, F(z^{t}) \rangle + KL(z, \hat{z}^{t}) \}$$

Recall that $F(z^{t}) = (l_{x}^{t}, l_{y}^{t}) = (Ay^{t}, -A^{\top}x^{t})$. Our proof provides a new connection to the dynamic regret analysis from online learning.

**Dynamic regret** Given loss functions $\{l_{x}^{t}\}$, actions $\{x^{t}\}$ produced by the x-player, and a sequence of comparators $\{u^{t}\}$, we define x-player's dynamic regret as

$$\mathcal{R}_{x}^{T}(\{u^{t}\}) := \sum_{t=1}^{T} \langle l_{x}^{t}, x^{t} - u^{t} \rangle.$$

A similar definition holds for the y-player. When $u^{1} = \dots = u^{T} = u$ the dynamic regret recovers the standard static external regret. An interesting case is when the comparator sequence is optimal: $u^{t} = x_{*}^{t} := \min_{x \in δ^{d_{1}}} x^{\top}l_{x}^{t}$ for every $t \ge 1$, which is called the worst-case dynamic regret. Observe that the sum of duality gaps is precisely the social dynamic regret, i.e., the sum of both players' dynamic regret, as shown in Proposition 1.

**Proposition 1.** *It holds that* $\sum_{t=1}^{T} DualityGap(x^{t}, y^{t}) = \sum_{t=1}^{T} \max_{z \in \mathcal{Z}} \langle F(z^{t}), z^{t} - z \rangle = \mathcal{R}_{x}^{T}(\{x_{*}^{t}\}) + \mathcal{R}_{y}^{T}(\{y_{*}^{t}\})$.

We then borrow insights from the online learning literature to upper bound the dynamic regret. The insight is that an algorithm has sublinear dynamic regret when (1) it has interval regret guarantee $O(\sqrt{|\mathcal{I}|})$ for all interval $\mathcal{I}$ and (2) the variation of loss sequence is sublinear. We formally introduce interval regret and the variation of the loss sequence below and discuss how to prove these conditions for the OMWU dynamics.

**Interval Regret** An interval $\mathcal{I} = [s, e] \subseteq [1, T]$ is a sequence of consecutive iterations. The interval regret with respect to interval $\mathcal{I}$ is the standard regret but restricted to iterations in $\mathcal{I}$:

$$\mathcal{R}^{\mathcal{I}} := \sum_{t \in \mathcal{I}} (l^{t})^{\top} x^{t} - \min_{x} \sum_{t \in \mathcal{I}} (l^{t})^{\top} x.$$

Denote by $|\mathcal{I}|$ the length of interval $\mathcal{I}$. For online learning against adversarial losses, OMWU does not guarantee $\mathcal{R}^{\mathcal{I}} = o(|\mathcal{I}|)$ for all $\mathcal{I}$. However, as we show in the following, OMWU dynamics in zero-sum matrix games with a fully mixed Nash equilibrium guarantees the interval regret is constant (that depends on $δ$): $\mathcal{R}^{\mathcal{I}} = O(1/δ)$ for all $\mathcal{I}$. We summarize some existing results of the OMWU dynamics in the following.

**Lemma 1** (Adapted from Lemma 1 in (Wei et al., 2021)). *Consider OMWU for a zero-sum game $A \in [0, 1]^{d_{1} \times d_{2}}$ with $\eta \le \frac{1}{8}$. Let $z^{*}$ be a Nash equilibrium. Define $\Theta^{t} = KL(z^{*}, \hat{z}^{t}) + \frac{1}{16} KL(\hat{z}^{t}, z^{t-1})$ and $\zeta^{t} = KL(\hat{z}^{t+1}, z^{t}) + KL(z^{t}, \hat{z}^{t})$. Then for any $t \ge 2$, we have:*
1. *For any z,* $\eta \langle F(z^{t}), z^{t} - z \rangle \le \Theta^{t} - \Theta^{t+1} - \frac{15}{16} \zeta^{t}$.
2. $\Theta^{t+1} \le \Theta^{t} - \frac{15}{16} \zeta^{t}$.
3. $\sum_{t=2}^{\infty} \|z^{t+1} - z^{t}\|_{1}^{2} = O(\log(d_{1}d_{2}))$.

First, we note that since there is a fully mixed Nash equilibrium with minimum probability $δ$, the minimum probability of any iterates is lower bounded by $\Omega(\exp(-\frac{1}{δ}))$. This is because by item 2 in Lemma 1, we know $KL(z^{*}, \hat{z}^{t}) \le \Theta^{t} \le \Theta^{2}$ is bounded by a constant that only depends on $d_{1}$ and $d_{2}$. Therefore, each coordinate of $\hat{z}^{t}$ must be lower bounded by $\Omega(\exp(-\frac{1}{δ}))$; otherwise the KL divergence $KL(z^{*}, \hat{z}^{t})$ would be too large. The guarantee for $z^{t}$ is by stability of the OMWU update. Formally, we have the next lemma.

**Lemma 2** (Adapted from Lemma 19 of (Wei et al., 2021)). *Let matrix game $A \in [0, 1]^{d_{1} \times d_{2}}$ have a fully mixed Nash equilibrium with minimum probability $δ > 0$. Let $\{z^{t}\}_{t \ge 1}$ be the iterates produced by OMWU with uniform initialization and step size $\eta \le \frac{1}{8}$, then*

$$\min_{t \ge 1, i \in [d_{1}+d_{2}]} \{z^{t}[i], \hat{z}^{t}[i]\} \ge \Omega(\frac{1}{d_{1}d_{2}} \exp(-\frac{1}{δ})).$$

Combining Lemma 1 and Lemma 2, we can bound the social interval regret by $O(\frac{1}{δ})$ for any interval $\mathcal{I} = [s, e]$. Note that this bound, although has a dependence on $δ$ is independent of the interval length $|\mathcal{I}|$ and thus is sufficient for our goal of achieving $O(\sqrt{|\mathcal{I}|})$ bound. The idea is that by item 1 in Lemma 1, the sum of regret is bounded by $\Theta^{s}$, the sum of two KL divergences, whose upper bound follows by the probability lower bounds presented in Lemma 2.

**Lemma 3 (Bounded Interval Regret).** *In the same setup as Lemma 2, we have* $\mathcal{R}_{x}^{\mathcal{I}} + \mathcal{R}_{y}^{\mathcal{I}} = O(\frac{(d_{1} + d_{2}) \log(d_{1}d_{2})}{\etaδ}), \forall \mathcal{I}$.

**Sublinear variation of loss sequences** Next, we show that the OMWU dynamic produces a stable environment, i.e., the variation of loss functions is small. Formally, we define the variation over an interval $\mathcal{I} = [s, e]$ as

$$V^{\mathcal{I}} = \sum_{t=s+1}^{e} \max_{z} |\langle F(z^{t}) - F(z^{t-1}), z \rangle|.$$

We note that OMWU has bounded second-order path length $\sum_{t=2}^{\infty} \|z^{t} - z^{t-1}\|_{1}^{2}$ (by item 3 in Lemma 1). We then have the following bound on its variation of losses.

**Lemma 4.** *For any $\mathcal{I}$, we have $V^{\mathcal{I}} = O(\sqrt{|\mathcal{I}| \log(d_{1}d_{2})})$.*

**From interval regret to dynamic regret** Now we combine the bound for the interval regret (Lemma 3) and the sublinear variation of the loss sequence bound (Lemma 4) to show a sublinear dynamic regret bound.

**Theorem 4.** *Let matrix game $A \in [0, 1]^{d_{1} \times d_{2}}$ have a fully mixed Nash equilibrium with minimum probability $δ > 0$. Let $\{x^{t}, y^{t}\}_{t \ge 1}$ be the iterates produced by OMWU with uniform initialization and step size $\eta \le \frac{1}{8}$. Then for any $T > 2$, the social dynamic regret is bounded by*

$$\mathcal{R}_{x}^{T}(\{x_{*}^{t}\}) + \mathcal{R}_{y}^{T}(\{y_{*}^{t}\}) = O(\frac{(d_{1} + d_{2}) \log(d_{1}d_{2})}{\eta} \cdot T^{\frac{3}{4}}δ^{-\frac{1}{2}}).$$

We remark that the rate $O(T^{\frac{3}{4}}δ^{-\frac{1}{2}})$ improves the existing bound $O(\exp(\frac{1}{δ}) \cdot (1 + \exp(-\frac{1}{δ}))^{-T})$ exponentially in terms of the dependence on $δ$. We give a proof sketch of Theorem 4 here and the full proof is in Appendix B.1.

Recall that $z_{*}^{t}$ is the optimal comparator in time t. The idea is that the dynamic regret for an interval $\mathcal{I} = [s, e]$ could be decomposed as follows:

$$\sum_{t \in \mathcal{I}} \langle F(z^{t}), z^{t} - z_{*}^{t} \rangle = \sum_{t \in \mathcal{I}} \langle F(z^{t}), z^{t} - z_{*}^{s} \rangle + \sum_{t \in \mathcal{I}} \langle F(z^{t}), z_{*}^{s} - z_{*}^{t} \rangle$$

where the first term is bounded by the interval regret $O(\frac{1}{δ}),$ and the second term could be bounded by the variation of the loss sequence. Then we can choose an optimal partition of T rounds that achieves the optimal rate.

#### 4.3 Initial Phase: Fast Convergence towards $O(δ)$

In this section, we return to the case of games $[0, 1]^{2\times2}$ with a fully mixed Nash equilibrium and present a fast uniform best-iterate convergence rate of OMWU for the initial iterations. Recall that our goal is to show that there exists an iteration $T_{1} \ge 1$ the following two conditions hold:
1. $DualityGap(x^{T_{1}}, y^{T_{1}}) = O(δ)$.
2. For all $T \in [1, T_{1}]$, a uniform best-iterate convergence rate holds: $\min_{t \in [1, T]} DualityGap(x^{t}, y^{t}) = \tilde{O}(\frac{1}{T})$.

**Simplification by structure** Consider the game $A \in [0, 1]^{2\times2}$ that has a fully mixed Nash equilibrium $(x^{*} = (1-δ_{x}, δ_{x}), y^{*} = (1-δ_{y}, δ_{y}))$ with $δ_{x}, δ_{y} \in (0, 1)$. To simplify the analysis, we make the following assumption.

**Assumption 1.** *The Nash equilibrium of A satisfies: $0 < δ_{x} \le δ_{y} \le 1 - δ_{x}$ and $δ_{x} \le \frac{1}{100}$.*

We remark that Assumption 1 is without loss of generality. The assumption $δ_{x} \le 1 - δ_{x}$ (thus $δ_{x} \le \frac{1}{2}$) holds without loss of generality since we can exchange the role of action 1 and action 2; the assumption $x^{*}$ is closer to the boundary than $y^{*}$, i.e., $δ_{x} \le δ_{y} \le 1 - δ_{x}$ holds without loss of generality since we can exchange the role of x-player and y-player. With Assumption 1, we let $δ := δ_{x}$ be the minimum probability in the Nash equilibrium. The assumption $δ_{x} \le \frac{1}{100}$ is without loss of generality since otherwise Theorem 4 already gives a $O(T^{-\frac{1}{4}}δ^{-\frac{1}{2}}) = O(T^{-\frac{1}{4}})$ rate. We focus on the more challenging case when $δ_{x}$ could be arbitrarily close to 0 here.

We further make an important observation on the structure of $2\times2$ games. We show that every game matrix with a fully mixed Nash equilibrium is an affine transformation of a generic matrix. Formally, we have the following lemma.

**Lemma 5.** *Let $A \in \mathbb{R}^{2\times2}$ be a matrix game such that it has a Nash equilibrium $(x^{*} = (1-δ_{x}, δ_{x}), y^{*} = (1-δ_{y}, δ_{y}))$ with $δ_{x}, δ_{y} \in (0, 1)$. Then A can be written as*

$$A = b_{1} \cdot \mathbf{1} + b_{2} \cdot \begin{pmatrix} \frac{1-δ_{y}}{1-δ_{x}} & \frac{1-δ_{x}-δ_{y}}{1-δ_{x}} \\ 0 & 1 \end{pmatrix},$$

*where $\mathbf{1}$ is the all-ones matrix and $b_{1}, b_{2} \in \mathbb{R}, b_{2} \ge 0$ are scaling constants.*

Together with Assumption 1, we have $A_{δ_{x},δ_{y}} \in [0, 1]^{2\times2}$. Considering the constraint that $A \in [0, 1]^{2\times2}$, we can assume $b_{2} \in (0, 1]$ (understanding that if $b_{2} = 0$ then the matrix is all-zero and every point is a Nash equilibrium). The next proposition further shows that we only need to prove convergence for the case $b_{1} = 0, b_{2} = 1$.

**Proposition 2.** *Let $A = b_{1} \cdot \mathbf{1} + b_{2} A_{δ_{x},δ_{y}}$ where $b_{2} \in (0, 1]$ and $\eta' > 0$ be a constant. If OMWU with any step size $0 < \eta \le \eta'$ has convergence rate $O(\frac{1}{\eta} \cdot f(T))$ on $A_{δ_{x},δ_{y}}$ then OMWU with any step size $0 < \eta \le \eta'$ also has convergence rate $O(\frac{1}{\eta} \cdot f(T))$ on A.*

**Fast convergence on $A_{δ_{x},δ_{y}}$** By previous simplification that is without loss of generality, we now focus on the class of games $A_{δ_{x},δ_{y}}$. Formally, we have the following theorem.

**Theorem 5 (Fast Convergence in Initial Iterations).** *Consider matrix game $A_{δ_{x},δ_{y}}$ that satisfies Assumption 1. Let $\{x^{t}, y^{t}\}_{t \ge 1}$ be the iterates produced by OMWU dynamics with uniform initialization and step size $\eta \le \frac{1}{10}$. Then there exists $T_{1} > 1$ such that:*
1. $DualityGap(x^{T_{1}}, y^{T_{1}}) \le 2δ_{x}$.
2. *For all $t \in [1, T_{1}]$, we have problem-independent best-iterate convergence rate:*
$$\min_{k \in [1,t]} DualityGap(x^{k}, y^{k}) = O(\frac{\log^{2}t}{\eta t}).$$

Thanks to the structure of $A_{δ_{x},δ_{y}}$, we can directly track the initial trajectory of OMWU dynamics. Specifically, we define $T_{1}$ as the first iteration when $x^{T_{1}}[1] \ge 1 - δ_{x}$ (note that $x^{1}[1] = \frac{1}{2}$) and show that the initial iterations $[1, T_{1}]$ has the desirable properties. The full proof is somewhat involved and appears in Appendix B.2.

### 5 Conclusion and Discussion

In this paper, we establish, for the first time, a separation between random-iterate and best-iterate convergence: OMWU has no polynomial random-iterate convergence for two-player zero-sum games but has a $O(T^{-\frac{1}{6}})$ best-iterate convergence rate for the case of $2\times2$ games with fully mixed Nash equilibria. Whether obtaining polynomial best-iterate convergence for OMWU in the general case is possible is an interesting open question for future works.

### Acknowledgement

Yang Cai was supported by the NSF Awards CCF-1942583 (CAREER) and CCF-2342642. Gabriele Farina was supported by the NSF Award CCF-2443068 (CAREER). Julien Grand-Clément was supported by Hi! Paris and Agence Nationale de la Recherche (Grant 11-LABX-0047). Christian Kroer was supported by the Office of Naval Research awards N00014-22-1-2530 and N00014-23-1-2374, and the National Science Foundation awards IIS-2147361 and IIS-2238960. Haipeng Luo was supported by the National Science Foundation award IIS-1943607. Weiqiang Zheng was supported by the NSF Awards CCF-1942583 (CAREER), CCF-2342642, and a Research Fellowship from the Center for Algorithms, Data, and Market Design at Yale (CADMY).

### References

* Anagnostides, I., Panageas, I., Farina, G., and Sandholm, T. On last-iterate convergence beyond zero-sum games. In International Conference on Machine Learning, pp. 536-581. PMLR, 2022.
* Brown, N. and Sandholm, T. Superhuman AI for heads-up no-limit poker: Libratus beats top professionals. Science, 359(6374):418-424, 2018.
* Brown, N. and Sandholm, T. Superhuman ai for multiplayer poker. Science, 365(6456):885-890, 2019.
* Cai, Y., Oikonomou, A., and Zheng, W. Finite-time last-iterate convergence for learning in multi-player games. In Advances in Neural Information Processing Systems (NeurIPS), 2022.
* Cai, Y., Farina, G., Grand-Clément, J., Kroer, C., Lee, C.-W., Luo, H., and Zheng, W. Fast last-iterate convergence of learning in games requires forgetful algorithms. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview.net/forum?id=hK7XTpCtBi.
* Cesa-Bianchi, N. and Lugosi, G. Prediction, learning, and games. Cambridge university press, 2006.
* Daskalakis, C., Ilyas, A., Syrgkanis, V., and Zeng, H. Training gans with optimism. arXiv preprint arXiv:1711.00141, 2017.
* Daskalakis, C., Fishelson, M., and Golowich, N. Near-optimal no-regret learning in general games. Advances in Neural Information Processing Systems (NeurIPS), 2021.
* Farina, G., Lee, C.-W., Luo, H., and Kroer, C. Kernelized multiplicative weights for 0/1-polyhedral games: Bridging the gap between learning in extensive-form and normal-form games. In International Conference on Machine Learning (ICML), pp. 6337-6357, 2022.
* Gorbunov, E., Taylor, A., and Gidel, G. Last-iterate convergence of optimistic gradient method for monotone variational inequalities. In Advances in Neural Information Processing Systems, 2022.
* Jacob, A. P., Shen, Y., Farina, G., and Andreas, J. The consensus game: Language model generation via equilibrium search. arXiv preprint arXiv:2310.09139, 2023.
* Mertikopoulos, P., Papadimitriou, C., and Piliouras, G. Cycles in adversarial regularized learning. In Proceedings of the twenty-ninth annual ACM-SIAM symposium on discrete algorithms, pp. 2703-2717. SIAM, 2018.
* Meta, Bakhtin, A., Brown, N., Dinan, E., Farina, G., Flaherty, C., Fried, D., Goff, A., Gray, J., Hu, H., et al. Human-level play in the game of diplomacy by combining language models with strategic reasoning. Science, 378(6624):1067-1074, 2022.
* Moravčík, M., Schmid, M., Burch, N., Lisá»³, V., Morrill, D., Bard, N., Davis, T., Waugh, K., Johanson, M., and Bowling, M. Deepstack: Expert-level artificial intelligence in heads-up no-limit poker. Science, 356(6337):508-513, 2017.
* Mordukhovich, B. S., Pena, J. F., and Roshchina, V. Applying metric regularity to compute a condition measure of a smoothing algorithm for matrix games. SIAM Journal on Optimization, 20(6):3490-3511, 2010.
* Munos, R., Valko, M., Calandriello, D., Azar, M. G., Rowland, M., Guo, Z. D., Tang, Y., Geist, M., Mesnard, T., Michi, A., et al. Nash learning from human feedback. arXiv preprint arXiv:2312.00886, 2023.
* Perolat, J., De Vylder, B., Hennes, D., Tarassov, E., Strub, F., de Boer, V., Muller, P., Connor, J. T., Burch, N., Anthony, T., et al. Mastering the game of stratego with model-free multiagent reinforcement learning. Science, 378(6623):990-996, 2022.
* Popov, L. D. A modification of the arrow-hurwicz method for search of saddle points. Mathematical notes of the Academy of Sciences of the USSR, 28:845-848, 1980.
* Rakhlin, S. and Sridharan, K. Optimization, learning, and games with predictable sequences. Advances in Neural Information Processing Systems, 2013.
* Sato, Y., Akiyama, E., and Farmer, J. D. Chaos in learning a simple two-person game. Proceedings of the National Academy of Sciences, 99(7):4748-4751, 2002.
* Syrgkanis, V., Agarwal, A., Luo, H., and Schapire, R. E. Fast convergence of regularized learning in games. Advances in Neural Information Processing Systems (NeurIPS), 2015.
* Tseng, P. On linear convergence of iterative methods for the variational inequality problem. Journal of Computational and Applied Mathematics, 60(1-2):237-252, 1995.
* Wei, C.-Y., Lee, C.-W., Zhang, M., and Luo, H. Linear last-iterate convergence in constrained saddle-point optimization. In International Conference on Learning Representations (ICLR), 2021.

### Contents
1 Introduction (2)
2 Preliminaries (4)
3 Lower Bounds for Random-Iterate Convergence (6)
3.1 Proof Overview (7)
4 Best-Iterate Convergence Rate for OMWU (8)
4.1 Proof Overview (9)
4.2 Global Phase: Convergence via Minimizing the Social Dynamic Regret (10)
4.3 Initial Phase: Fast Convergence towards O(Î´) (13)
5 Conclusion and Discussion (14)

Appendix A: Missing Proofs in Section 3
A.1 Existing Analysis on the Hard Instance (18)
A.2 Lower Bounds for Random-Iterate (19)
A.2.1 Proof of Theorem 1 (22)
A.2.2 Proof of Theorem 2 (22)

Appendix B: Missing Proofs in Section 4
B.1 Global Phase: Missing Proofs in Section 4.2 (23)
B.1.1 Proof of Proposition 1 (23)
B.1.2 Proof of Lemma 3 (23)
B.1.3 Proof of Lemma 4 (24)
B.1.4 Proof of Theorem 4 (24)
B.2 Initial Phase: Missing Proofs in Section 4.3 (25)
B.2.1 Proof of Proposition 2 (25)
B.2.2 Proof of Theorem 5 (25)
B.3 Combining Two-Phase Analysis: Proof of Theorem 3 (33)

### Appendix A Missing Proofs in Section 3

In this section, we prove lower bounds for the random-iterate convergence rates of OMWU and other OFTRL dynamics. Recall that we focus on the specific class of $2\times2$ zero-sum matrices parameterized by $δ \in (0, \frac{1}{2})$:

$$A_{δ} := \begin{pmatrix} \frac{1}{2} + δ & \frac{1}{2} \\ 0 & 1 \end{pmatrix} \quad \forall δ \in (0, \frac{1}{2}).$$

which is the hard instance for last-iterate convergence analyzed in (Cai et al., 2024). We first introduce some notations and present the results and analysis in (Cai et al., 2024) that will be useful in our later analysis in Appendix A.1. Then, we give the proof of the lower bound for random-iterate convergence in Appendix A.2.

#### A.1 Existing Analysis on the Hard Instance

**Additional Notations** We focus on the $d=2$ dimension case where $x = (x[1], x[2])$. We let $e_{x}^{t} = l_{x}^{t}[1] - l_{x}^{t}[2]$ be the difference between the losses of action 1 and action 2, and $E_{x}^{t} = \sum_{k=1}^{t} e_{x}^{k}$ be the cumulative differences. Define the function $F_{\eta,R} : \mathbb{R} \rightarrow [0, 1]$ as follows:

$$F_{\eta,R}(e) := \text{argmin}_{x \in [0,1]} \{ x \cdot e + \frac{1}{\eta} R(x) \}. \quad (3)$$

where we slightly abuse the notation and write $R((x, 1-x))$ as $R(x)$, where R is a strongly convex regularizer over $δ^{d}$. Then by a change of variable $x^{t}[2] = 1 - x^{t}[1]$ the optimization problem in the OFTRL update can be reduced to a 1-dimension optimization problem as follows:

$$x^{t}[1] = F_{\eta,R}(E_{x}^{t-1} + e_{x}^{t-1}), \quad x^{t}[2] = 1 - x^{t}[1].$$

**Lemma 6** (Lemma 1 in (Cai et al., 2024)). *The function $F_{\eta,R}(\cdot) : \mathbb{R} \rightarrow [0, 1]$ defined in (3) is non-increasing.*

Cai et al. (2024) introduce the following assumptions on the regularizer R and show that they are satisfied by the negative entropy, the log regularizer, the squared Euclidean norm, and the Negative Tsallis entropy regularizers.

**Assumption 2.** *We assume that the regularizer R satisfies the following properties: the function $F_{\eta,R} : \mathbb{R} \rightarrow [0, 1]$ defined in (3) is:*
1. *Unbiased:* $F_{\eta,R}(0) = \frac{1}{2}$
2. *Rational:* $\lim_{E \rightarrow -\infty} F_{\eta,R}(E) = 1$ and $\lim_{E \rightarrow +\infty} F_{\eta,R}(E) = 0$.
3. *Lipschitz continuous: There exists $L \ge 0$ such that $F_{1,R}$ is L-Lipschitz.*

**Assumption 3.** *Let L be the Lipschitzness constant of $F_{1,R}$ in Assumption 2. Define the constant $c_{1} = \frac{1}{2} - F_{1,R}(\frac{1}{20L})$. There exist universal constants $δ', c_{2} > 0$ and $c_{3} \in (0, \frac{1}{2}]$ such that for any $0 < δ \le δ'$,*
1. *For any E that satisfies $F_{1,R}(E) \ge \frac{1}{1+δ}$ we have $F_{1,R}(-\frac{c_{1}^{2}}{30Lδ} + E) \ge \frac{1+c_{3}}{1+c_{3}+δ}$*
2. *For any E that satisfies $F_{1,R}(E) \ge \frac{1}{2(1+δ)}$, we have $F_{1,R}(-\frac{c_{3}c_{1}^{2}}{120L} + \frac{δ}{4L} + E) \ge \frac{1}{2} + c_{2}$.*

**Lemma 7** (Lemma 5-8 in (Cai et al., 2024)). *Assumption 2 and Assumption 3 are satisfied by negative entropy $(L = \frac{1}{2})$, squared Euclidean norm $(L = \frac{1}{2})$, the log regularizer $(L = \frac{1}{2})$, and the Tsallis entropy regularizer parameterized with $\beta \in (0, 1)$ $(L = \frac{1}{2\beta})$.*

We summarize the analysis from (Cai et al., 2024) in the following theorem.

**Theorem 6** (Adapted from Theorem 1 in (Cai et al., 2024)). *Assume the regularizer R satisfies Assumption 2 and Assumption 3. For any $δ \in (0, \hat{δ})$, where $\hat{δ} = \min\{\frac{1}{15}, \frac{c_{1}}{6}, \frac{c_{1}^{2}}{300}, δ'\}$ is a constant depending only on the constants $c_{1}$ and $δ'$ defined in Assumption 3, let $\{x^{t}, y^{t}\}$ be iterates produced by the OFTRL dynamics on $A_{δ}$ (defined in (1)) with any step size $\eta \le \frac{1}{4L}$ and initialized at the uniform strategies. Then the following holds:*
1. *Define $T_{1}$ the smallest iteration when $x^{T_{1}}[1] \ge \frac{1}{1+δ}$. Then*
$$y^{t}[1] \le \frac{1}{2} - c_{1}, \forall t \in [\frac{1}{2\eta L}, T_{1} - 1]$$
2. *Define $T_{2} > T_{1}$ the smallest iteration when $y^{T_{2}}[1] \ge \frac{1}{2(1+δ)}$ and $T_{h} := \lfloor \frac{c_{1}}{2\eta Lδ} \rfloor \in [\frac{c_{1}}{2\eta Lδ} - 1, \frac{c_{1}}{2\eta Lδ}]$. Then*
$$x^{t}[1] \ge \frac{1+c_{3}}{1+c_{3}+δ}, \forall t \in [T_{1} + T_{h}, T_{2}]$$
$$DualityGap(x^{t}, y^{t}) \ge c_{2}, \forall t \in [T_{2} + \lceil\frac{c_{1}T_{h}}{20}\rceil, T_{2} + \lfloor\frac{c_{1}T_{h}}{10}\rfloor - 2]$$

#### A.2 Lower Bounds for Random-Iterate

In this section, we present our lower bounds for random-iterate convergence rates of OFTRL learning dynamics. The main theorem in this section is Theorem 7 that shows there exists $T = O(\frac{f_{R}(δ)}{δ})$ such that all the iterates $t \in [T, T + \Omega(\frac{1}{δ})]$ have a duality gap lower bounded by a constant. Here $f_{R}(δ) := -F_{1,R}^{-1}(\frac{1}{1+δ})$ is a quantity that depends on the regularizer R. We provide upper bounds of $f_{R}(δ)$ in Lemma 8. At the end, we combine Theorem 7 and Lemma 8 to prove Theorem 1 and Theorem 2.

**Lemma 8.** *Let $f_{R}(δ) := -F_{1,R}^{-1}(\frac{1}{1+δ})$. If $E \ge -f_{R}(δ)$, then $F_{1,R}(E) \le \frac{1}{1+δ}$. Moreover, we have the following upper bounds on $f_{R}(δ)$ for $δ \in (0, \frac{1}{2})$ and regularizer R:*
1. *Negative entropy:* $f_{R}(δ) \le \log(\frac{1}{δ})$;
2. *Squared Euclidean norm:* $f_{R}(δ) \le 1$;
3. *Log barrier:* $f_{R}(δ) \le \frac{1}{δ}$.
4. *Negative Tsallis entropy* $(\beta \in (0, 1))$: $f_{R}(δ) \le \frac{2\beta}{1-\beta} (\frac{1}{δ})^{1-\beta}$.

*Proof.* Since $f_{R}(δ) = -F_{1,R}^{-1}(\frac{1}{1+δ})$, we have $F_{1,R}(-f_{R}(δ)) = F_{1,R}(F_{1,R}^{-1}(\frac{1}{1+δ})) = \frac{1}{1+δ}$. Since $F_{1,R}$ is a non-increasing function (Lemma 6), we know if $E \ge -f_{R}(δ)$, then $F_{1,R}(E) \le \frac{1}{1+δ}$. This finishes the proof of the first claim.
We then prove upper bounds for $f_{R}(δ)$ for different regularizer R. Our proof strategy is to derive the closed-form expression of $F_{1,R}$ (by setting the gradient of Equation (3) to be 0) and that of its inverse function $F_{1,R}^{-1}$ and directly bound $f_{R}(δ)$.

*Negative entropy:* $R(x) = x \log x + (1-x) \log(1-x)$. For negative entropy, $F_{1,R}$ admits the following closed-form expression:
$$F_{1,R}(E) = \frac{1}{1+\exp(E)} \Rightarrow F_{1,R}^{-1}(\frac{1}{1+δ}) = \log δ.$$
Hence, $f_{R}(δ) = -F_{1,R}^{-1}(\frac{1}{1+δ}) = \log \frac{1}{δ}$.

*Squared Euclidean norm:* $R(x) = \frac{1}{2}(x^{2} + (1-x)^{2})$. In this case, we can verify that $F_{1,R}^{-1}(x)$ admits a closed-form for $x \in (0, 1)$:
$$F_{1,R}(E) = \frac{1-E}{2}, \forall E \in (-1, 1) \Rightarrow F_{1,R}^{-1}(x) = 1 - 2x, \forall x \in (0, 1).$$
Hence, $f_{R}(δ) = -F_{1,R}^{-1}(\frac{1}{1+δ}) = \frac{1-δ}{1+δ} \le 1$.

*Log barrier:* $R(x) = -\log x - \log(1-x)$. In this case, we can verify that $F_{1,R}^{-1}(x)$ admits a closed-form for $x \in (0, 1)$:
$$F_{1,R}^{-1}(x) = \frac{2x-1}{x^{2}-x}, x \in (0, 1).$$
Hence, we have
$$f_{R}(δ) = -F_{1,R}^{-1}(\frac{1}{1+δ}) = -\frac{2(1+δ) - (1+δ)^{2}}{\frac{1}{1+δ} - (\frac{1}{1+δ})^{2}} \cdot \frac{1}{(1+δ)^{2}} = \frac{1-δ^{2}}{δ} \le \frac{1}{δ}.$$

*Negative Tsallis entropy:* $R(x) = \frac{1}{1-\beta}(1 - x^{\beta} + 1 - (1-x)^{\beta})$. In this case, we can verify that $F_{1,R}^{-1}(x)$ admits a closed-form for $x \in (0, 1)$:
$$F_{1,R}^{-1}(x) = \frac{\beta}{1-\beta} (x^{\beta-1} - (1-x)^{\beta-1}), x \in (0, 1).$$
Hence, we have
$$f_{R}(δ) = -F_{1,R}^{-1}(\frac{1}{1+δ}) = \frac{\beta}{1-\beta} ((\frac{1}{1+δ})^{\beta-1} - (\frac{δ}{1+δ})^{\beta-1}) \le \frac{\beta}{1-\beta} (\frac{1+δ}{δ})^{1-\beta} \le \frac{2\beta}{1-\beta} (\frac{1}{δ})^{1-\beta}.$$
This completes the proof. $\square$

**Theorem 7.** *Assume the regularizer R satisfies Assumption 2 and Assumption 3. For any $δ \in (0, \hat{δ})$ where $\hat{δ}$ is a constant that depending only on the constants $c_{1}$ and $δ'$ defined in Assumption 3, the OFTRL dynamics on $A_{δ}$ (defined in (1)) with any step size $\eta \le \frac{1}{4L}$ satisfies the following: there is an iteration $T \le \frac{8 + 2Lf_{R}(δ)}{c_{1}c_{3}\eta Lδ}$ (with $f_{R}(δ)$ defined in Lemma 8) such that for all $t \in [T, T + \frac{c_{1}^{2}}{80\eta Lδ}]$*

$$DualityGap(x^{t}, y^{t}) \ge c_{2},$$

*where $c_{2} > 0$ is a constant defined in Assumption 3.*

*Proof.* We will use notations and results presented in Theorem 6. We restate some definitions and key facts:
* $T_{1}$ is the smallest iteration when $x^{T_{1}}[1] \ge \frac{1}{1+δ}$;
* $T_{2} > T_{1}$ is the smallest iteration when $y^{T_{2}}[1] \ge \frac{1}{2(1+δ)}$;
* $T_{h} := \lfloor \frac{c_{1}}{2\eta Lδ} \rfloor \in [\frac{c_{1}}{2\eta Lδ} - 1, \frac{c_{1}}{2\eta Lδ}]$;
* We have $DualityGap(x^{t}, y^{t}) \ge c_{2}$ for all $t \in [T_{2} + \lceil\frac{c_{1}T_{h}}{20}\rceil, T_{2} + \lfloor\frac{c_{1}T_{h}}{10}\rfloor - 2]$.

Notice that the interval $[T_{2} + \lceil\frac{c_{1}T_{h}}{20}\rceil, T_{2} + \lfloor\frac{c_{1}T_{h}}{10}\rfloor - 2]$ has length at least
$$\lfloor\frac{c_{1}T_{h}}{10}\rfloor - 2 - \lceil\frac{c_{1}T_{h}}{20}\rceil \ge \frac{c_{1}^{2}}{20\eta Lδ} - 4 - \frac{c_{1}^{2}}{40\eta Lδ} = \frac{c_{1}^{2}}{40\eta Lδ} - 4 \ge \frac{c_{1}^{2}}{80\eta Lδ}$$
In the above inequality, we use $x-1 \le \lfloor x \rfloor \le \lceil x \rceil \le x+1$; and the fact that $\eta L \le \frac{1}{4}$ and $δ \le \frac{c_{1}^{2}}{300}$.

Now we get $\Omega(\frac{1}{δ})$ consecutive iterations with duality gap at least $c_{2}$. It remains to show that $T_{2} = O(\frac{f_{R}(δ)}{δ})$. We proceed with two steps. We first upper bound $T_{1}$, then we use the obtained bound to further bound $T_{2}$.

**Bounding $T_{1}$** By item 1 of Theorem 6, we have $y^{t}[1] \le \frac{1}{2} - c_{1}$ for all $t \in [\frac{1}{2\eta L}, T_{1} - 1]$. Recall that $e_{x}^{t} = y^{t}[1] - δ_{y} x^{t}[1] + \dots$ (referring to Prop 3), this implies $e_{x}^{t} \le -\frac{c_{1}}{2}$ for all $t \in [\frac{1}{2\eta L}, T_{1} - 1]$. Since $T_{1}$ is the first iteration that $x^{T_{1}}[1] \ge \frac{1}{1+δ}$ it follows that
$$\frac{1}{1+δ} \ge x^{T_{1}-1}[1] = F_{\eta,R}(E_{x}^{T_{1}-2} + e_{x}^{T_{1}-2}) \ge F_{\eta,R}(\frac{1}{2\eta L} - \frac{c_{1}}{2}(T_{1}-2 - \frac{1}{2\eta L}) + 1) = F_{1,R}(\frac{1}{2L} - \frac{\eta c_{1}}{2}(T_{1}-2 - \frac{1}{2\eta L}) + \eta)$$
By definition of $f_{R}(δ)$ and monotonicity of $F_{1,R}$ we deduce
$$\frac{1}{2L} - \frac{\eta c_{1}}{2}(T_{1}-2 - \frac{1}{2\eta L}) + \eta \ge F_{1,R}^{-1}(\frac{1}{1+δ}) \ge -f_{R}(δ) \Rightarrow T_{1} \le \frac{2(\frac{1}{2L} + \eta + f_{R}(δ))}{\eta c_{1}} + 2 + \frac{1}{2\eta L}.$$
Since $\eta L \le \frac{1}{4}$, the above simplifies to $T_{1} \le \frac{6}{c_{1}\eta L} + \frac{2f_{R}(δ)}{\eta c_{1}}$.

**Bounding $T_{2}$** By item 2 of Theorem 6, we have for all $t \in [T_{1} + T_{h}, T_{2}]$
$$x^{t}[1] \ge \frac{1+c_{3}}{1+c_{3}+δ}$$
Note that $1+c_{3}+δ \le 2$. This implies $e_{y}^{t} = 1 - (1+δ)x^{t}[1] = -\frac{c_{3}δ}{1+c_{3}+δ} \le -\frac{c_{3}δ}{2}$ for all $T_{1} + T_{h} \le t \le T_{2}$. Moreover, for all $t \in [T_{1} + 1, T_{2}]$, we have $x^{t}[1] \ge \frac{1}{1+δ}$ so $e_{y}^{t} \le 0$. Combining these gives
$$\frac{1}{2} \ge y^{T_{2}-1}[1] = F_{\eta,R}(E_{y}^{T_{2}-1} + e_{y}^{T_{2}-1}) \ge F_{\eta,R}(T_{1} \cdot 1 - \frac{c_{3}δ}{2}(T_{2} - T_{h} + 1)).$$
By monotonicity of $F_{\eta,R}$ and $F_{\eta,R}(0) = \frac{1}{2}$ (Assumption 2), we have
$$T_{1} - \frac{c_{3}δ}{2}(T_{2} - T_{h} + 1) \ge 0 \Rightarrow T_{2} \le \frac{2T_{1}}{c_{3}δ} + T_{h} \Rightarrow T_{2} \le \frac{7 + 2Lf_{R}(δ)}{c_{1}c_{3}\eta Lδ}.$$
Combining the facts that $DualityGap(x^{t}, y^{t}) \ge c_{2}$ for all $t \in [T_{2} + \lceil\frac{c_{1}T_{h}}{20}\rceil, T_{2} + \lfloor\frac{c_{1}T_{h}}{10}\rfloor - 2]$, and the length of the interval is at least $\frac{c_{1}^{2}}{80\eta Lδ}$ we conclude that starting from iteration no more than $\frac{8 + 2Lf_{R}(δ)}{c_{1}c_{3}\eta Lδ}$, the following $\frac{c_{1}^{2}}{80\eta Lδ}$ iterations all have duality gap larger than the constant $c_{2} > 0$. $\square$

##### A.2.1 Proof of Theorem 1

By Theorem 7 and Lemma 8, we know that we can find $T = O(\frac{\log(1/δ)}{δ})$ such that for all $t \in [T, T_{1}]$ we have $DualityGap(x^{t}, y^{t}) \ge c_{2}$ where $T_{1} = T + \Theta(\frac{1}{δ}) = O(\frac{\log(1/δ)}{δ})$. Then we have
$$\sum_{t=1}^{T_{1}} DualityGap(x^{t}, y^{t}) \ge \sum_{t=T}^{T_{1}} DualityGap(x^{t}, y^{t}) = \Omega(\frac{1}{δ}).$$
Hence we get
$$\frac{1}{T_{1}} \sum_{t=1}^{T_{1}} DualityGap(x^{t}, y^{t}) = \Omega(\frac{1}{\log(1/δ)}) = \Omega(\frac{1}{\log T_{1}})$$
where the last equality holds since $\log T_{1} = \Theta(\log(1/δ))$. We note that $T_{1}$ can be arbitrarily large as $δ \rightarrow 0$. Therefore, the uniform random-iterate convergence rate of OMWU is $\Omega(\frac{1}{\log T})$.

##### A.2.2 Proof of Theorem 2

Theorem 2 follows in the same way as Theorem 1 by combining Theorem 7 and Lemma 8 as we discussed in Section 3.1.

### Appendix B Missing Proofs in Section 4

#### B.1 Global Phase: Missing Proofs in Section 4.2

##### B.1.1 Proof of Proposition 1

*Proof.* Recall that $l_{x}^{t} = Ay^{t}$ and $l_{y}^{t} = -A^{\top}x^{t}$. By definition, we have
$$\sum_{t=1}^{T} DualityGap(x^{t}, y^{t}) = \sum_{t=1}^{T} (\max_{y \in δ^{d_{2}}} (x^{t})^{\top}Ay - \min_{x \in δ^{d_{1}}} x^{\top}Ay^{t})$$
$$= \sum_{t=1}^{T} \max_{z \in \mathcal{Z}} \langle F(z^{t}), z^{t} - z \rangle = \mathcal{R}_{y}^{T}(\{y_{*}^{t}\}) + \mathcal{R}_{x}^{T}(\{x_{*}^{t}\}). \quad \square$$

##### B.1.2 Proof of Lemma 3

*Proof.* By Lemma 1, for any interval $\mathcal{I} = [s, e] \in [2, \infty]^{2}$ we have
$$\mathcal{R}_{x}^{\mathcal{I}} + \mathcal{R}_{y}^{\mathcal{I}} = \max_{z} \sum_{t=s}^{e} \langle F(z^{t}), z^{t} - z \rangle \le \frac{1}{\eta} (\Theta^{s} - \Theta^{e+1}) \le \frac{1}{\eta} \Theta^{s}.$$
For any $t \ge 2,$ we can bound $\Theta^{t}$ by definition and the probability lower bound in Lemma 2.
$$\Theta^{t} \le KL(z^{*}, \hat{z}^{t}) + KL(\hat{z}^{t}, z^{t-1}) \le \sum_{i \in [d_{1}+d_{2}]} \log \frac{1}{z^{t}[i]} + \log \frac{1}{z^{t-1}[i]} \le O((d_{1} + d_{2})(\log(d_{1}d_{2}) + \frac{1}{δ}))$$
Combining the above two inequalities completes the proof. Here we ignore the first iteration which contributes at most $O(1)$ regret. $\square$

##### B.1.3 Proof of Lemma 4

*Proof.* Let $\mathcal{I} = [s, e]$. Then by definition, we have
$$V^{\mathcal{I}} = \sum_{t=s+1}^{e} \max_{z} |\langle F(z^{t}) - F(z^{t-1}), z \rangle| \le 2 \sum_{t=s+1}^{e} \|F(z^{t}) - F(z^{t-1})\|_{\infty} \le 2\sqrt{|\mathcal{I}| \sum_{t=s+1}^{e} \|F(z^{t}) - F(z^{t-1})\|_{\infty}^{2}}$$
$$\le 2\sqrt{|\mathcal{I}| \sum_{t=s+1}^{e} \|z^{t} - z^{t-1}\|_{1}^{2}} \le O(\sqrt{|\mathcal{I}| \log(d_{1}d_{2})}). \quad \square$$

##### B.1.4 Proof of Theorem 4

*Proof.* Let $\mathcal{I}_{1} = [s_{1}, e_{1}], \dots, \mathcal{I}_{M} = [s_{M}, e_{M}]$ be any partition of the T rounds. We let $z_{*}^{s_{m}} \in \text{argmin}_{z} \sum_{t \in \mathcal{I}_{m}} \langle F(z^{t}), z \rangle$. Then, the social dynamic regret on $\mathcal{I}_{m}$ is
$$\sum_{t \in \mathcal{I}_{m}} \langle F(z^{t}), z^{t} - z_{*}^{t} \rangle = \sum_{t \in \mathcal{I}_{m}} \langle F(z^{t}), z^{t} - z_{*}^{s_{m}} \rangle + \sum_{t \in \mathcal{I}_{m}} \langle F(z^{t}), z_{*}^{s_{m}} - z_{*}^{t} \rangle \le \mathcal{R}_{z}^{\mathcal{I}_{m}} + 2|\mathcal{I}_{m}| V^{\mathcal{I}_{m}}$$
where the last step is by the definition of interval regret and the fact that
$$\langle F(z^{t}), z_{*}^{s_{m}} - z_{*}^{t} \rangle \le \langle F(z^{t}) - F(z^{s_{m}}), z_{*}^{s_{m}} \rangle + \langle F(z^{s_{m}}) - F(z^{t}), z_{*}^{t} \rangle = \sum_{k=s_{m}+1}^{t} \langle F(z^{k}) - F(z^{k-1}), z_{*}^{s_{m}} \rangle + \sum_{k=s_{m}+1}^{t} \langle F(z^{k-1}) - F(z^{k}), z_{*}^{t} \rangle \le 2V^{\mathcal{I}_{m}}$$
where the first inequality is by optimality of $z_{*}^{s_{m}}$. Therefore,
$$\sum_{t=1}^{T} \langle F(z^{t}), z^{t} - z_{*}^{t} \rangle \le \sum_{m=1}^{M} (\mathcal{R}_{z}^{\mathcal{I}_{m}} + 2|\mathcal{I}_{m}| V^{\mathcal{I}_{m}}) \le O((d_{1} + d_{2})\log(d_{1}d_{2}) \cdot \frac{M}{\etaδ} + \max_{m \in [1,M]} |\mathcal{I}_{m}| \sqrt{T})$$
We choose $M = \max\{1, T^{\frac{3}{4}}δ^{\frac{1}{2}}\}$ and make sure each interval has length $O(T/M)$, then we have
$$\frac{M}{δ} + \max_{m \in [1,M]} |\mathcal{I}_{m}| \sqrt{T} = O(\frac{M}{\etaδ} + \frac{T^{\frac{3}{2}}}{M}) = O(\frac{T^{\frac{3}{4}}δ^{-\frac{1}{2}}}{\eta}).$$
We note that the above holds for any T. This completes the proof. $\square$

#### B.2 Initial Phase: Missing Proofs in Section 4.3

##### B.2.1 Proof of Proposition 2

*Proof.* Fix any $\eta \in (0, \eta']$. Since $A = b_{1} \cdot \mathbf{1} + b_{2} A_{δ_{x},δ_{y}}$, we know that the following two dynamics produce exactly the same trajectory $\{x^{t}, y^{t}\}$:
1. OMWU with step size $\eta$ on A;
2. OMWU with step size $b_{2}\eta$ on $A_{δ_{x},δ_{y}}$.
The reason is (1) the update rule of OMWU concerns only the relative loss between actions so the $b_{1} \mathbf{1}$ component does not matter; (2) $\eta(b_{2} A_{δ_{x},δ_{y}}) = (b_{2}\eta) A_{δ_{x},δ_{y}}$. Therefore, if we assume that $\{x^{t}, y^{t}\}$ evaluated on $A_{δ_{x},δ_{y}}$ has a rate $O(\frac{1}{b_{2}\eta} f(T))$ then the same sequence evaluated on $A = b_{1}\mathbf{1} + b_{2} A_{δ_{x},δ_{y}}$ has a convergence rate of $O(b_{2} \cdot \frac{1}{b_{2}\eta} f(T)) = O(\frac{1}{\eta} f(T))$. $\square$

##### B.2.2 Proof of Theorem 5

By Assumption 1, Lemma 5, and Proposition 2, we only need to consider the following class of matrices without loss of generality:
$$A_{δ_{x},δ_{y}} = \begin{pmatrix} \frac{1-δ_{y}}{1-δ_{x}} & \frac{1-δ_{x}-δ_{y}}{1-δ_{x}} \\ 0 & 1 \end{pmatrix}, \frac{1}{100} \le δ_{x} \le δ_{y} \le 1 - δ_{x} \quad (4)$$
For simplicity, in the proof, we omit the subscript and denote by A the matrix $A_{δ_{x},δ_{y}}$. We first summarize properties of A that can be verified by simple algebra.

**Proposition 3.** *Given Assumption 1, we have the following:*
* *The loss vectors of A are $l_{x} = Ay = [1 - \frac{δ_{y}}{1-δ_{x}} + \frac{δ_{x}}{1-δ_{x}}y[1], 1 - y[1]], l_{y} = -A^{\top}x = -[\frac{1-δ_{y}}{1-δ_{x}}x[1], 1 - \frac{δ_{y}}{1-δ_{x}}x[1]]$*
* *The loss vectors of A satisfy: $e_{x} := l_{x}[1] - l_{x}[2] = \frac{y[1] - δ_{y}}{1-δ_{x}}, e_{y} := l_{y}[1] - l_{y}[2] = \frac{x[2] - δ_{x}}{1-δ_{x}} = \frac{1 - δ_{x} - x[1]}{1-δ_{x}}.$*
* *Moreover, $e_{x} \in [-\frac{δ_{y}}{1-δ_{x}}, 1] \subseteq [-1, 1], e_{y} \in [-\frac{δ_{x}}{1-δ_{x}}, 1] \subseteq [-2δ_{x}, 1].$*

We consider two cases: (1) $δ_{y} \ge \frac{1}{2}$; (2) $δ_{y} < \frac{1}{2}$. We prove Theorem 5 for each case in the following.

**Case 1: $δ_{y} \ge \frac{1}{2}$.**

**Lemma 9.** *Consider matrix A defined in Equation (4) that satisfies $δ_{y} \in [\frac{1}{2}, 1)$. Let $\{x^{t}, y^{t}\}_{t \ge 1}$ be the iterates produced by OMWU dynamics with uniform initialization and step size $\eta \le 1$. Let $T_{1}$ be the first iteration such that $x^{t}[1] \ge 1 - δ_{x}$. Then*
1. $DualityGap(x^{t}, y^{t}) \le \frac{x^{t}[2]}{x^{t}[1]} \le 9 \exp(-\frac{\eta t}{42})$, *for all $t \in [1, T_{1} - 1]$*;
2. $DualityGap(x^{T_{1}}, y^{T_{1}}) \le 2δ_{x}$.
3. *For all $t \in [1, T_{1}]$, there exists a universal constant $C > 0$ such that* $\min_{k \in [1,t]} DualityGap(x^{k}, y^{k}) \le \frac{C}{\eta} \frac{1}{t}$.

*Proof.* The OMWU dynamics is initialized with uniform distributions $(x^{1}, y^{1})$ and step size $\eta \le 1$. Denote by $T_{0} = \lceil \frac{1}{\eta} \rceil + 1 \le \frac{2}{\eta}$. Using the update rule, we have for all $t \in [1, T_{0}]$, $\frac{x^{t}[1]}{x^{t}[2]} = \frac{x^{1}[1]}{x^{1}[2]} \exp(-\eta E_{x}^{t-1} - \eta e_{x}^{t-1}) \le e^{\eta t} \le e^{2}$, where we use $x^{1}[1] = x^{1}[2] = \frac{1}{2}$ and $-e_{x} \le 1$ by Proposition 3. This implies $x^{t}[1] \le \frac{e^{2}}{e^{2}+1} < \frac{8}{9}$ for all $t \in [1, T_{0}]$.
We define $T_{1} > T_{0}$ the first iteration where $x^{t}[1] \ge 1 - δ_{x}$. We have the following two inequalities on $e_{y}^{t} = l_{y}^{t}[1] - l_{y}^{t}[2]$ by Proposition 3:
$$e_{y}^{t} = \frac{1 - δ_{x} - x^{t}[1]}{1 - δ_{x}} \ge 0, \forall t \in [1, T_{1} - 1], \quad (5)$$
$$e_{y}^{t} \ge \frac{1}{10}, \forall t \in [1, T_{0}]. \quad (6)$$
where in (5) we use $x^{t}[1] < 1 - δ_{x}$ for all $t \in [1, T_{1} - 1];$ in (6) we use $δ_{x} \le \frac{1}{100}$ and $x^{t}[1] \le \frac{8}{9}$ for all $t \in [1, T_{0}]$. For any $t \in [T_{0}, T_{1}]$, we have $y^{t}$ satisfies
$$\frac{y^{t}[1]}{y^{t}[2]} = \exp(-\eta E_{y}^{T_{0}-1} - \sum_{k=T_{0}}^{t-1} \eta e_{y}^{k} - \eta e_{y}^{t-1}) \le \exp(-\frac{\eta(T_{0}-1)}{10}) \le \exp(-\frac{1}{10}) < \frac{10}{11}.$$
This implies $y^{t}[1] < \frac{10}{21}$ for all $t \in [T_{0}, T_{1}]$. Moreover, for all $t \in [T_{0}, T_{1}]$, we have $e_{x}^{t} = \frac{y^{t}[1] - δ_{y}}{1 - δ_{x}} \le \frac{10/21 - 1/2}{1 - δ_{x}} \le -\frac{1}{42},$ where we use $y^{t}[1] \le \frac{10}{21}$ and $δ_{y} \ge \frac{1}{2}$. For all $t \in [T_{0}, T_{1}-1]$, using $e_{x}^{t} \in [-1, -1/42]$, we have $x^{t}$ satisfies
$$\frac{x^{t}[1]}{x^{t}[2]} = \frac{x^{1}[1]}{x^{1}[2]} \exp(-2\eta e_{x}^{t-1} - \dots) \ge \exp(\frac{\eta(t - T_{0})}{42} - \eta T_{0}) \ge \exp(\frac{\eta t}{42} - \frac{1}{21} - 2) \ge \frac{1}{9} \exp(\frac{\eta t}{42}).$$
Now we track the duality gap. Note that for $t \in [T_{0}, T_{1}-1]$, we have $x^{t}[1] \le 1 - δ_{x}$ and $y^{t}[1] \le 10/21 \le δ_{y}$. Therefore,
$DualityGap(x^{t}, y^{t}) = \frac{δ_{y}}{1-δ_{x}}(1 - x^{t}[1]) - \frac{δ_{x}}{1-δ_{x}} y^{t}[1] \le x^{t}[2] \le \frac{x^{t}[2]}{x^{t}[1]} \le 9 \exp(-\frac{\eta t}{42})$.
Since $T_{0} \le 2/\eta$ the above bounds also hold for all $t \in [1, T_{0}]$. Thus $DualityGap(x^{t}, y^{t}) \le 9 \exp(-\frac{\eta t}{42})$ for $t \in [1, T_{1}-1]$. At $T_{1}$, $DualityGap(x^{T_{1}}, y^{T_{1}}) \le 2δ_{x}$. The best-iterate rate follows by $\min_{k \in [1,t]} DualityGap(x^{k}, y^{k}) \le \frac{C}{\eta t}$. $\square$

**Case 2: $δ_{y} < \frac{1}{2}$.**

**Lemma 10.** *Consider matrix A defined in (4) that satisfies Assumption 1 and $δ_{y} \in (0, \frac{1}{2})$. Let $\{x^{t}, y^{t}\}_{t \ge 1}$ be the iterates produced by OMWU dynamics with uniform initialization and step size $\eta \le \frac{1}{10}$. Denote $T_{1}$ the first iteration that $x^{t}[1] \ge 1 - δ_{x}$. Then*
1. $DualityGap(x^{T_{1}}, y^{T_{1}}) \le 2δ_{x}$.
2. *For all $t \in [1, T_{1}]$, there exists a universal constant $C > 0$ such that* $\min_{k \in [1,t]} DualityGap(x^{k}, y^{k}) \le \frac{C}{\eta} \frac{\log^{2} t}{t}$.

*Proof.* We track the trajectory in two phases:
* Phase I: $x^{t}[1]$ decreases and $y^{t}[1]$ decreases. At the end, $y^{t}[1] \le δ_{y}$ and $DualityGap \approx O(δ_{y})$.
* Phase II: $x^{t}[1]$ increases and $y^{t}[1]$ continues to decrease. At the end, $x^{t}[1] \ge 1 - δ_{x}$ and $DualityGap \approx O(δ_{x})$.

**Phase I:** $x^{t}[1], y^{t}[1]$ both decrease. Initially $1/2$. As long as $y^{t}[1] \ge δ_{y}$, $e_{x}^{t} \ge 0$, so $x^{t}[1] \le 1/2$. This implies $e_{y}^{t} \ge 1/3$. Then $\frac{y^{t}[1]}{y^{t}[2]} \le \exp(-\frac{\eta t}{3})$. For $t \in [1, T_{y}-1]$, $DualityGap(x^{t}, y^{t}) \le y^{t}[1] \le \exp(-\frac{\eta t}{3})$. At $T_{y}$, $DualityGap \le 2δ_{y}$.

**Phase II:** $y^{t}[1]$ decreases, $x^{t}[1]$ increases. Define $T_{m} = T_{y} + \lceil 2/\eta \rceil$. For $t \in [T_{y}+1, T_{m}]$, $x^{t}[1] \le 8/9$. For $t \in [T_{m}, T_{x}-1]$, $\frac{y^{t}[1]}{y^{t}[2]} \le \frac{10}{11} \frac{δ_{y}}{1-δ_{y}}$, so $y^{t}[1] \le \frac{10}{21} δ_{y}$. Then $e_{x}^{t} \le -\frac{δ_{y}}{21}$. We show $T_{y} \le \frac{3}{\eta} \log \frac{1}{δ_{y}} + 1$. For $t \in [T_{m}, T_{x}]$, $\frac{x^{t}[1]}{x^{t}[2]} \ge δ_{y}^{8} \exp(\frac{\etaδ_{y}t}{21})$. Duality gap for $t \in [T_{m}, T_{x}-1]$ is $\le \frac{1}{δ_{y}^{8}} \exp(-\frac{\etaδ_{y}t}{21})$.

**Claim 1.** $\min_{k \in [1,t]} DualityGap(x^{k}, y^{k}) \le \frac{C}{\eta} \frac{\log^{2} t}{t}$ *for $t \in [1, T_{x}]$.*
The proof analyzes the cases $t \in [1, T_{y}]$, $t \in [T_{y}, T_{m}]$, and $t \in [T_{m}, T_{x}-1]$ separately. Combining Lemma 9 and 10 gives Theorem 5. $\square$

##### B.3 Combining Two-Phase Analysis: Proof of Theorem 3

Consider any matrix game $A \in [0, 1]^{2\times2}$ that has a fully-mixed Nash equilibrium with minimum probability $δ > 0$. By Theorem 5, there is an initial phase $[1, T_{1}]$ where $O(\frac{\log^{2} T}{T})$ best-iterate convergence holds and $DualityGap(x^{T_{1}}, y^{T_{1}}) \le 2δ$. For $t \ge T_{1}+1$, by Theorem 4, the best-iterate rate $O(T^{-\frac{1}{4}}δ^{-\frac{1}{2}})$ holds. Thus for all $T \ge T_{1}+1$:
$$\min_{t \in [1,T]} DualityGap(x^{t}, y^{t}) \le \min \{ 2δ, O(T^{-\frac{1}{4}}δ^{-\frac{1}{2}}) \} \le O(T^{-\frac{1}{6}}). \quad \square$$

