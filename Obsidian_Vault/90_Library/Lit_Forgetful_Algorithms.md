**Yang Cai**
Yale
yang.cai@yale.edu

**Gabriele Farina**
MIT
gfarina@mit.edu

**Christian Kroer**
Columbia
ck2945@columbia.edu

**Chung-Wei Lee**
USC
leechung@usc.edu

**Weiqiang Zheng**
Yale
weiqiang.zheng@yale.edu

**Julien Grand-Clément**
HEC Paris
grand-clement@hec.fr

**Haipeng Luo**
USC
haipengl@usc.edu

### Abstract
Self-play via online learning is one of the premier ways to solve large-scale two-player zero-sum games, both in theory and practice. Particularly popular algorithms include optimistic multiplicative weights update (OMWU) and optimistic gradient-descent-ascent (OGDA). While both algorithms enjoy $O(1/T)$ ergodic convergence to Nash equilibrium in two-player zero-sum games, OMWU offers several advantages including logarithmic dependence on the size of the payoff matrix and $\tilde{O}(1/T)$ convergence to coarse correlated equilibria even in general-sum games. However, in terms of last-iterate convergence in two-player zero-sum games, an increasingly popular topic in this area, OGDA guarantees that the duality gap shrinks at a rate of $(1/\sqrt{T})$, while the best existing last-iterate convergence for OMWU depends on some game-dependent constant that could be arbitrarily large. This begs the question: is this potentially slow last-iterate convergence an inherent disadvantage of OMWU, or is the current analysis too loose? Somewhat surprisingly, we show that the former is true. More generally, we prove that a broad class of algorithms that do not forget the past quickly all suffer the same issue: for any arbitrarily small $δ > 0$, there exists a $2 \times 2$ matrix game such that the algorithm admits a constant duality gap even after $1/δ$ rounds. This class of algorithms includes OMWU and other standard optimistic follow-the-regularized-leader algorithms.

---

### 1 Introduction
Self-play via online learning is one of the premier ways to solve large-scale two-player zero-sum games. Major examples include super-human AIs for Go, Poker [Brown and Sandholm, 2018], and human-level AI for Stratego [Perolat et al., 2022] and alignment of large language models [Munos et al., 2023]. In particular, Optimistic Multiplicative Weights Update (OMWU) and Optimistic Gradient Descent-Ascent (OGDA) are two of the most well-known online learning algorithms. When applied to learning a two-player zero-sum game via self-play for $T$ rounds, the average iterates of both algorithms are known to be an $O(1/T)$-approximate Nash equilibrium [Rakhlin and Sridharan, 2013, Syrgkanis et al., 2015], while other algorithms, such as vanilla Multiplicative Weights Update (MWU) and vanilla Gradient Descent-Ascent (GDA), have a slower ergodic convergence rate of $O(1/\sqrt{T})$.

For multiple practical reasons, there is growing interest in studying the last-iterate convergence of these learning dynamics [Daskalakis and Panageas, 2019, Golowich et al., 2020b, Wei et al., 2021, Lee et al., 2021]. In this regard, existing results seemingly exhibit a gap between OGDA and OMWU: the duality gap of the last iterate of OGDA is known to decrease at a rate of $O(1/\sqrt{T})$ [Cai et al., 2022, Gorbunov et al., 2022], with no dependence on constants beyond the dimension and the smoothness of the players' utility functions of the game. In contrast, the existing convergence rate for OMWU depends on some game-dependent constant that could be arbitrarily large, even after fixing the dimension and the smoothness constant of the game [Wei et al., 2021]. Given the fundamental role of OMWU in online learning and its other advantages over OGDA (such as its logarithmic dependence on the number of actions), it is natural to ask the following question:

(*) **Is the potentially slow last-iterate convergence an inherent disadvantage of OMWU?**

**Main Results.** In this work, we show that the answer to this question is yes, contrary to a common belief that better analysis and better last-iterate convergence results similar to those of OGDA are possible for OMWU. More specifically, we show the following.

**Theorem (Informal).** For OMWU with constant step size, there is no function $f$ such that the corresponding learning dynamics $\{(x^t, y^t)\}_{t \ge 1}$ in two-player zero-sum games $[0, 1]^{d_1 \times d_2}$ has a last-iterate convergence rate of $f(d_1, d_2, T)$. More specifically, no function $f$ can satisfy
1. $\text{DualityGap}(x^T, y^T) \le f(d_1, d_2, T)$ for all matrices $[0, 1]^{d_1 \times d_2}$ and $T \ge 1$.
2. $\lim_{T \to \infty} f(d_1, d_2, T) \to 0$.

Our findings show that, despite the significantly superior regret properties of OMWU compared to OGDA, its last-iterate convergence properties are remarkably worse. In turn, this counters the viewpoint that "Follow-the-Regularized-Leader (FTRL) is better than Online Mirror Descent (OMD)" [van Erven, 2021]: crucially, while OMWU is an instance of (optimistic) FTRL, OGDA is an instance of optimistic OMD that cannot be expressed in the FTRL formalism. We further show that similar negative results extend to several other standard online learning algorithms, including a close variant of OGDA. More concretely, our main results are as follows.

* We identify a broad family of Optimistic FTRL (OFTRL) algorithms that do not forget about the past quickly. We prove that, for any sufficiently small $δ > 0$, there exists a $2 \times 2$ two-player zero-sum game such that, even after $1/δ$ iterations, the duality gap of the iterate output by these algorithms is still a constant (Theorem 1). This excludes the possibility of showing a game-independent last-iterate convergence rate similar to that of OGDA.
* We prove that many standard online learning algorithms, such as OFTRL with the entropy regularizer (equivalently, OMWU), the Tsallis entropy family of regularizers, the log regularizer, and the squared Euclidean norm regularizer, all fall into this family of non-forgetful algorithms and thus all suffer from the same slow convergence. Also note that Optimistic OMD (OOMD), another well-known family of algorithms, is equivalent to OFTRL when given a Legendre regularizer. Therefore, OOMD with the entropy, Tsallis entropy, and log regularizer also suffer the same issue.
* Finally, we also generalize our negative results from $2 \times 2$ games to $2n \times 2n$ games for any positive integer $n$, strengthening our message that forgetfulness is generally needed in order to achieve fast last-iterate convergence.

---

### 2 Preliminaries and Problem Setup
We consider the standard setting of no-regret learning in a zero-sum game $A \in [0, 1]^{d_1 \times d_2}$. In each iteration $t \ge 1$, the $x$-player chooses $x^t \in \mathcal{X} := δ^{d_1}$ while the $y$-player chooses $y^t \in \mathcal{Y} := δ^{d_2}$. Then the $x$-player receives loss vector $l_x^t = Ay^t$ while the $y$-player receives loss vector $l_y^t = -A^\top x^t$. The goal is to find or approximate a Nash equilibrium $(x^*, y^*)$ to the game such that $x^* \in \arg\min_{x \in \mathcal{X}} \max_{y \in \mathcal{Y}} x^\top Ay$ and $y^* \in \arg\max_{y \in \mathcal{Y}} \min_{x \in \mathcal{X}} x^\top Ay$. The approximation error of a strategy pair $(x, y)$ is measured by its duality gap, defined as $\text{DualityGap}(x, y) = \max_{y' \in \mathcal{Y}} x^\top Ay' - \min_{x' \in \mathcal{X}} (x')^\top Ay$ which is always non-negative.

Popular no-regret algorithms for solving the game include the Optimistic Follow-the-Regularized-Leader (OFTRL) algorithm and the Optimistic Online Mirror Descent (OOMD) algorithm, both defined in terms of a certain regularizer $R : δ^d \to \mathbb{R}$ (for some general dimension $d$). The corresponding Bregman divergence of $R$ is $D_R(x, x') = R(x) - R(x') - \langle \nabla R(x'), x - x' \rangle$, and the regularizer is 1-strongly convex if $D_R(x, x') \ge \frac{1}{2} \|x - x'\|_2^2$ for all $x, x' \in δ^d$.

**Optimistic Online Mirror Descent (OOMD)**
Starting from an initial point $(x^1, y^1) = (\hat{x}^1, \hat{y}^1)$, the OOMD algorithm with regularizer $R$ and steps size $\eta > 0$ updates in each iteration $t \ge 2$,
$\hat{x}^t = \arg\min_{x \in \mathcal{X}} \{ \eta \langle x, l_x^{t-1} \rangle + D_R(x, \hat{x}^{t-1}) \}$, $x^t = \arg\min_{x \in \mathcal{X}} \{ \eta \langle x, l_x^{t-1} \rangle + D_R(x, \hat{x}^t) \}$,
$\hat{y}^t = \arg\min_{y \in \mathcal{Y}} \{ \eta \langle y, l_y^{t-1} \rangle + D_R(y, \hat{y}^{t-1}) \}$, $y^t = \arg\min_{y \in \mathcal{Y}} \{ \eta \langle y, l_y^{t-1} \rangle + D_R(y, \hat{y}^t) \}$. (OOMD)

In particular, we call OOMD with a squared Euclidean norm regularizer, that is, $R(x) = \frac{1}{2} \sum_{i=1}^d x[i]^2$ optimistic gradient-descent-ascent (OGDA). When $R$ is the negative entropy, that is, $R(x) = \sum_{i=1}^d x[i] \log x[i]$, we call the resulting OOMD algorithm optimistic multiplicative weights update (OMWU).

**Optimistic Follow-the-Regularized-Leader (OFTRL)**
Define the cumulative loss vectors $L_x^t := \sum_{k=1}^t l_x^k$ and $L_y^t := \sum_{k=1}^t l_y^k$. The update rule of OFTRL with regularizer $R$ is for each $t > 1$
$x^t = \arg\min_{x \in \mathcal{X}} \{ \langle x, L_x^{t-1} + l_x^{t-1} \rangle + \frac{1}{\eta} R(x) \}$,
$y^t = \arg\min_{y \in \mathcal{Y}} \{ \langle y, L_y^{t-1} + l_y^{t-1} \rangle + \frac{1}{\eta} R(y) \}$. (OFTRL)

Throughout the paper, we consider the following regularizers:
* Negative entropy ($R(x) = \sum_{i=1}^d x[i] \log x[i]$): the resulting OFTRL algorithm coincides with OMWU defined by the OOMD framework previously.
* Squared Euclidean norm ($R(x) = \frac{1}{2} \sum_{i=1}^d x[i]^2$): note that the resulting algorithm is different from OGDA since the squared Euclidean norm is not a Legendre regularizer. As we will show, the two algorithms behave very differently in terms of last-iterate convergence.
* Log barrier ($R(x) = \sum_{i=1}^d -\log(x[i])$): we also call it the log regularizer.
* Negative Tsallis entropy regularizers ($R(x) = \frac{1 - \sum_{i=1}^d (x[i])^\beta}{1 - \beta}$ parameterized by $\beta \in (0, 1)$).

**The 2-dimension case**
We denote $x \in \mathbb{R}^2$ as $x = [x[1], x[2]]^\top$. For $d_1 = 2$, finding $x^t$ of OFTRL reduces to the following 1-dimensional optimization problem:
$x^t[1] = \arg\min_{x \in [0, 1]} \{ x \cdot (L_x^{t-1}[1] + l_x^{t-1}[1] - L_x^{t-1}[2] - l_x^{t-1}[2]) + \frac{1}{\eta} R(x) \}$, $x^t[2] = 1 - x^t[1]$,
where we slightly abuse the notation and denote $R(x) = R([x, 1-x])$ for $x \in [0, 1]$. We introduce two notations (the case for the $y$-player is similar): let $e_x^t = l_x^t[1] - l_x^t[2]$ be the difference between the losses of the two actions, and $E_x^t = \sum_{k=1}^t e_x^k$ be the cumulative difference between the losses of the two actions. For OFTRL, it is clear that the update of $x^t$ only depends on the differences $E_x^{t-1}$, $e_x^{t-1}$, the step size $\eta$, and the regularizer $R$. For this reason, we define $F_{\eta,R} : \mathbb{R} \to [0, 1]$ as follows:
$F_{\eta,R}(e) := \arg\min_{x \in [0, 1]} \{ x \cdot e + \frac{1}{\eta} R(x) \}$. (1)

We assume the function $F_{\eta,R}$ is well-defined, i.e., the above optimization problem admits a unique solution in $[0, 1]$. This is a condition easily satisfied, for example, when the regularizer $R$ is strongly convex. Then the OFTRL algorithm can be written as
$x^t[1] = F_{\eta,R}(E_x^{t-1} + e_x^{t-1})$, $x^t[2] = 1 - x^t[1]$.

The following lemma shows that the function $F_{\eta,R}$ is non-increasing.
**Lemma 1 (Monotonicity of $F_{\eta,R}$).** The function $F_{\eta,R}(\cdot) : \mathbb{R} \to [0, 1]$ defined in (1) is non-increasing.

We present some blanket assumptions on the regularizer, which are satisfied by all the regularizers introduced before.
**Assumption 1.** We assume that the regularizer $R$ satisfies the following properties: the function $F_{\eta,R} : \mathbb{R} \to [0, 1]$ defined in (1) is,
1. Unbiased: $F_{\eta,R}(0) = \frac{1}{2}$
2. Rational: $\lim_{E \to -\infty} F_{\eta,R}(E) = 1$ and $\lim_{E \to +\infty} F_{\eta,R}(E) = 0$.
3. Lipschitz, continuous: There exists $L \ge 0$ such that $F_{1,R}$ is $L$-Lipschitz.

Item 1 in Assumption 1 shows that the initial strategy is the uniform distribution over the two actions, which is standard in practice. The rational assumption (item 2 in Assumption 1) is natural since otherwise, the algorithm could not even converge to a pure Nash equilibrium. The Lipschitzness (item 3 in Assumption 1) is implied when the regularizer is strongly convex over $[0, 1]^2$ (see Lemma 4), and it further implies Lipschitzness of $F_{\eta,R}$ for any $\eta$ as shown in the following proposition.

**Proposition 1.** The function $F_{\eta,R}$ satisfies $F_{\eta,R}(E/\eta) = F_{1,R}(E)$. If $F_{1,R}$ is $L$-Lipschitz, then $F_{\eta,R}$ is $\eta L$-Lipschitz for any $\eta > 0$.

---

### 3 Slow Convergence of OFTRL: A Hard Game Instance
We give negative results on the last-iterate convergence properties of OFTRL by studying its behavior on a surprisingly simple $2 \times 2$ two-player zero-sum games. The game's loss matrix $A_δ$ is parameterized by $δ \in (0, 1/2)$ and is defined as follows:

**3.1 Basic Properties**
$A_δ := \begin{bmatrix} \frac{1}{2} + δ & \frac{1}{2} \\ 0 & 1 \end{bmatrix}$ (2)

We summarize some useful properties of $A_δ$ in the following proposition.
**Proposition 2.** The matrix game $A_δ$ satisfies:
1. $A_δ$ has a unique Nash equilibrium $x^* = [\frac{1}{1+δ}, \frac{δ}{1+δ}]$ and $y^* = [\frac{1}{2(1+δ)}, \frac{1+2δ}{2(1+δ)}]$
2. For a strategy pair $(x^t, y^t)$ the loss vectors (i.e., gradients) for the two players are respectively:
$l_x^t = A_δ y^t = \begin{bmatrix} \frac{1}{2} + δ y^t[1] \\ 1 - y^t[1] \end{bmatrix}$, $l_y^t = -A_δ^\top x^t = -\begin{bmatrix} (\frac{1}{2} + δ)x^t[1] \\ 1 - \frac{1}{2}x^t[1] \end{bmatrix}$
Moreover,
$e_x^t = l_x^t[1] - l_x^t[2] = -\frac{1}{2} + (1 + δ)y^t[1] \in [-\frac{1}{2}, \frac{1}{2} + δ]$
$e_y^t = l_y^t[1] - l_y^t[2] = 1 - (1 + δ)x^t[1] \in [-δ, 1]$. (3)

In particular, we notice that $e_y^t \ge -δ$. It implies that if the cumulative differences between the losses of the two actions $E_y^t$ is large, then it takes $\Omega(\frac{1}{δ})$ iterations to make $E_y^t$ small (close to 0). This has important implications for non-forgetful algorithms like OFTRL that look at the whole history of losses. Since OFTRL chooses the strategy $y^t$ based on $E_y^t$, it could be trapped in a bad action for a long time even if the current gradients suggest that the other action is better. This is the key observation for our main negative results on the slow last-iterate convergence rates of OFTRL.

The following lemma shows that in a particular region of $(x, y)$, the duality gap is a constant.
**Lemma 2.** Let $δ, \epsilon \in (0, 1/2)$. For any $x, y \in δ^2$ such that $x[1] \ge \frac{1}{1+δ}$ and $y[1] \ge \frac{1}{2} + \epsilon$ the duality gap of $(x, y)$ for game $A_δ$ (defined in (2)) satisfies $\text{DualityGap}(x, y) \ge \epsilon$.

**3.2 Slow Last-Iterate Convergence**
We further require the following assumption on the regularizer $R$ (and thus the function $F_{1,R}$).
**Assumption 2.** Let $L$ be the Lipschitzness constant of $F_{1,R}$ in Assumption 1. Denote constant $c_1 = \frac{1}{2} - F_{1,R}(\frac{1}{20L})$. There exist universal constants $δ'$, $c_2 > 0$ and $c_3 \in (0, 1/2]$ such that for any $0 < δ \le δ'$
1. For any $E$ that satisfies $F_{1,R}(E) \ge \frac{1}{1+δ}$, we have $F_{1,R}(-\frac{c_1^2}{30Lδ} + E) \ge \frac{1+c_3}{1+c_3+δ}$
2. For any $E$ that satisfies $F_{1,R}(E) \ge \frac{1}{2(1+δ)}$, we have $F_{1,R}(-\frac{c_3c_1^2}{120L} + \frac{δ}{4L} + E) \ge \frac{1}{2} + c_2$.

Although Assumption 2 is technical, the idea is simple. Item 1 in Assumption 2 states that if a loss difference $E < 0$ already makes $F_{1,R}(E) \ge \frac{1}{1+δ}$ then the loss difference $E' = E - \Omega(\frac{1}{δ})$ is able to make $F_{1,R}(E')$ greater than $F_{1,R}(E)$ by a margin of $\Omega(δ)$. Item 2 in Assumption 2 states that if a loss difference $E$ already makes $F_{1,R}(E) \ge \frac{1}{2(1+δ)} \approx \frac{1}{2}$, then the loss difference $E' = E - \Omega(1)$ is able to make $F_{1,R}(E')$ greater than $\frac{1}{2}$ by a constant margin $c_2$. In Appendix C, we verify that Assumption 2 holds for the negative entropy, squared Euclidean norm, the log barrier, and the negative Tsallis entropy regularizers.

Now we present the main result of the section showing that even after $\Omega(1/δ)$ iterations, the duality gap of the iterate output by OFTRL is still a constant.
**Theorem 1.** Assume the regularizer $R$ satisfies Assumption 1 and Assumption 2. For any $δ \in (0, \hat{δ})$ where $\hat{δ}$ is a constant depending only on the constants $c_1$ and $δ'$ defined in Assumption 2, the OFTRL dynamics on $A_δ$ (defined in (2)) with any step size $\eta \le \frac{1}{4L}$ satisfies the following: there exists an iteration $t \ge \frac{c_1}{3\eta Lδ}$ with a duality gap of at least $c_2$, a strictly positive constant defined in Assumption 2.

**Proof Sketch:** We decompose the analysis into three stages as illustrated in Figure 3. We describe the three stages and the high-level ideas of our proof below and defer the full proof to Appendix B.2.
* **Stage I $[1, T_1 - 1]$:** Recall that $x^1[1] = y^1[1] = \frac{1}{2}$ by Assumption 1. We show that $x^t[1]$ increases and denote $T_1$ the first iteration that $x^{T_1}[1] \ge \frac{1}{1+δ}$. During the time $[1, T_1 - 1]$, since $x^t[1]$ is always smaller than $\frac{1}{1+δ}$ we know from Proposition 2 action 1 has larger loss than action 2 for the $y$-player, i.e., $e_y^t = l_y^t[1] - l_y^t[2] \ge 0$. Thus $y^t[1]$ decreases during stage I and we show that $y^{T_1}[1] \le \frac{1}{2} - c_1$ with $c_1$ defined in Assumption 2.
* **Stage II $[T_1, T_2]$:** Recall that $y^{T_1}[1] \le \frac{1}{2} - c_1$. We define $T_2 > T_1$ as the first iteration where $y^{T_2}[1] \ge \frac{1}{2(1+δ)} > \frac{1}{2} - c_1$. We remark that for $y^t[1]$ to increase, the loss vector must satisfy $e_y^t < 0$. However, the game matrix $A_δ$ guarantees that $e_y^t \ge -δ$ no matter what the $x$-player's strategy is (Proposition 2). Thus by the $\eta L$-Lipschitzness of $F_{\eta,R}$ (Proposition 1), the per-iteration increase in $y^t[1]$ is at most $\eta Lδ$. Therefore, we know $T_2 - T_1 = \Omega(\frac{c_1}{\eta Lδ})$. As a result, $e_x^t < 0$ during $[T_1, T_2]$ and the cumulative loss for the $x$-player decreases to $E_x^{T_2} \le E_x^{T_1} - \Omega(\frac{1}{\eta Lδ})$. Recall $x^{T_1}[1] \ge \frac{1}{1+δ}$. Thus $x^{T_2}[1] > x^{T_1}[1]$ is much closer to 1.
* **Stage III $[T_2, T_3]$:** Recall that $y^{T_2}[1] \ge \frac{1}{2(1+δ)}$. Moreover, $y^t[1]$ could keep increasing if $x^t[1] \ge \frac{1}{1+δ}$ since that implies $e_y^t \le 0$. Now the question is how long would the $x$-player stay close to the boundary, i.e, $x^t[1] \ge \frac{1}{1+δ}$. Since OFTRL-type algorithms are not forgetful, this happens only when $E_x^t \ge E_x^{T_1}$ (recall $x^{T_1}[1] \ge \frac{1}{1+δ}$). But we have at the end of stage II, $E_x^{T_2} \le E_x^{T_1} - \Omega(\frac{1}{\eta Lδ})$. Since the per-iteration loss is bounded by 1, it requires at least $\Omega(\frac{1}{\eta Lδ})$ iterations to cancel the cumulative loss of $\Omega(\frac{1}{\eta Lδ})$. Define $T_3 = T_2 + \Omega(\frac{1}{\eta Lδ})$. During $[T_2, T_3]$, the $y$-player always receives loss such that $e_y^t \le 0$ and we prove that in the end $y^{T_3}[1] \ge \frac{1}{2} + c_2$ for some constant $c_2$.
* **Conclusion:** Finally, we get one iteration $T_3 \ge \Omega(\frac{1}{\eta Lδ})$ with $x^{T_3}[1] \ge \frac{1}{1+δ}$ and $y^{T_3}[1] \ge \frac{1}{2} + c_2$. Using Lemma 2, the duality gap of $(x^{T_3}, y^{T_3})$ is at least $c_2 > 0$.

Theorem 1 immediately implies the following (proof deferred to Appendix B.3).
**Theorem 2.** For optimistic FTRL with any regularizer satisfying Assumption 1 and Assumption 2 and constant steps size $\eta \le \frac{1}{4L}$ ($L$ is defined in Assumption 1), there is no function $f$ such that the corresponding learning dynamics $\{(x^t, y^t)\}_{t \ge 1}$ in two-player zero-sum games $[0, 1]^{d_1 \times d_2}$ has a last-iterate convergence rate of $f(d_1, d_2, T)$. More specifically, no function $f$ can satisfy
1. $\text{DualityGap}(x^T, y^T) \le f(d_1, d_2, T)$ for all $[0, 1]^{d_1 \times d_2}$ and for all $T \ge 1$.
2. $\lim_{T \to \infty} f(d_1, d_2, T) \to 0$.

Theorem 1 and Theorem 2 provide impossibility results for getting a last-iterate convergence rate for OFTRL that solely depends on the bounded parameters, even in two-player zero-sum games. Moreover, they show the necessity of forgetfulness for fast last-iterate convergence in games since OGDA has a last-iterate convergence rate of $O(\frac{poly(d_1, d_2)}{\sqrt{T}})$ [Cai et al., 2022, Gorbunov et al., 2022].

---

### 4 Extension to Higher Dimensions
In this section, we extend our negative results from $2 \times 2$ matrix games to games with higher dimensions. We start by showing an equivalence result for a single player (say, the first player). We assume that a decision maker is using OFTRL with a 1-strongly convex (w.r.t. the $l_2$ norm) and separable regularizer $R(x) = R_1(x_1) + R_2(x_2)$ to choose decisions. At a given time $t$, they see a loss $l^t \in [0, 1]^2$. Now consider the following $2n$-dimensional decision problem: The player uses OFTRL using the regularizer $\hat{R}(\hat{x}) = \sum_{i=1}^n R_1(\hat{x}_i) + \sum_{i=n+1}^{2n} R_2(\hat{x}_i)$, i.e., they use $R_1$ on the first half of actions, and $R_2$ on the second half. This is again a 1-strongly convex regularizer (w.r.t. the $l_2$ norm). Suppose the decision maker sees the rescaled and duplicated version of the losses $l^1, \dots, l^T$ from the 2-dimensional case: $\hat{l}_i^t = \frac{1}{n^\alpha} l_1^t$ if $i \le n$ and $\hat{l}_i^t = \frac{1}{n^\alpha} l_2^t$ if $i > n$. The parameter $\alpha$ will be chosen later based on the regularizer. Now we wish to show that by choosing $\alpha$ in the right way, we get that the decisions for the 2-dimensional and $2n$-dimensional OFTRL algorithms are equivalent. Let $x^1, \dots, x^T$ be the 2-dimensional OFTRL decisions, and let $\hat{x}^1, \dots, \hat{x}^T$ be the $2n$-dimensional OFTRL decisions. Then, we want to show that $\sum_{i=1}^n \hat{x}_i^t = x^t[1]$ and $\sum_{i=n+1}^{2n} \hat{x}_i^t = x^t[2]$ for all $t$.

**Lemma 3.** Let the losses $\hat{l}^1, \dots, \hat{l}^T$ satisfy the duplication procedure given in the preceding paragraph. Then for any time $t$, we have $\hat{x}_1^t = \dots = \hat{x}_n^t$ and $\hat{x}_{n+1}^t = \dots = \hat{x}_{2n}^t$.
**Proof.** Suppose not and let $\hat{x}^t$ be the corresponding solution. Then the optimal solution is such that $\hat{x}_i^t \ne \hat{x}_k^t$ for some $i, k$ both less than $n$, or both greater than $n$. But then, by symmetry, we have that there is more than one optimal solution to the OFTRL optimization problem at time $t$: the objective is exactly the same if we create a new solution where we swap the values of $\hat{x}_i^t$ and $\hat{x}_k^t$. This is a contradiction due to strong convexity. $\square$

From lemma 3, we have that the OFTRL decision problem in $2n$ dimensions can equivalently be written as a 2-dimensional decision problem: Since the first $n$ entries must be the same, we can simply optimize over that one shared value, say $x^t[1]$, which we use for all $n$ entries, and similarly we use $x^t[2]$ for the second half of the entries. Let Dupl: $δ^2 \to δ^{2n}$ be a function that maps the two-dimensional solution into the corresponding duplicated $2n$-dimensional solution. The equivalent 2-dimensional problem is then:
$\hat{x}^t = \text{Dupl}[\arg\min_{x \in \frac{1}{n} \cdot δ^2} \{ \frac{n}{n^\alpha} \langle x, \sum_{\tau=1}^{t-1} l^\tau + l^{t-1} \rangle + \frac{n}{\eta} R_1(x[1]) + \frac{n}{\eta} R_2(x[2]) \}]$
$= \text{Dupl}[\frac{1}{n} \cdot \arg\min_{x \in δ^2} \{ \frac{n}{n^\alpha} \langle \frac{1}{n} x, \sum_{\tau=1}^{t-1} l^\tau + l^{t-1} \rangle + \frac{n}{\eta} R(x/n) \}]$
$= \text{Dupl}[\frac{1}{n} \cdot \arg\min_{x \in δ^2} \{ \langle x, \sum_{\tau=1}^{t-1} l^\tau + l^{t-1} \rangle + \frac{n^{\alpha+1}}{\eta} R(x/n) \}]$

The next theorem shows that we can choose $\alpha$ for different regularizers and construct $2n \times 2n$ loss matrices whose learning dynamics are equivalent to the learning dynamics in $2 \times 2$ games given in the preceding sections. We defer the proof to Appendix D.
**Theorem 3.** For any loss matrix $A \in [0, 1]^{2 \times 2}$, there exists a loss matrix $\hat{A} \in [0, n^{-\alpha}]^{2n \times 2n}$ such that for the Euclidean ($\alpha=1$), entropy ($\alpha=0$), Tsallis ($\beta \in (0, 1)$ and $\alpha = -1 + \beta$), and log ($\alpha = -1$) regularizers, the resulting OFTRL learning dynamics are equivalent in the two games.

Combining Theorem 1 and Theorem 3, we have the following:
**Corollary 1.** In the same setup as Theorem 3, under Assumption 1 and Assumption 2, there exists a game matrix $\hat{A}_δ \in [0, n^{-\alpha}]^{2n \times 2n}$ such that the OFTRL learning dynamics with any step size $\eta \le \frac{1}{4L}$ satisfies the following: there exists an iteration $t \ge \frac{c_1}{3\eta Lδ}$ with a duality gap at least $c_2 n^{-\alpha}$.

Since $\alpha = 0$ for the entropy regularizer, the same results hold more generally for games where one player has more actions than the other. In particular, we can create a $2n \times 2m$ game such that the resulting dynamics are equivalent to those in a $2 \times 2$ game. This does not work for the Euclidean and log regularizers because the rescaling factors would be different for the row and column players.

---

### 5 Conclusion and Discussions
In this paper, we study last-iterate convergence rates of OFTRL algorithms with various popular regularizers, including the popular OMWU algorithm. Our main results show that even in simple $2 \times 2$ two-player zero-sum games parametrized by $δ > 0$, the lack of forgetfulness of OFTRL leads to the duality gap remaining constant even after $1/δ$ iterations (Theorem 1). As a corollary, we show that the last-iterate convergence rate of OFTRL must depend on a problem-dependent constant that can be arbitrarily bad (Theorem 2). This highlights a stark contrast with OOMD algorithms: while OGDA with constant step size achieves a $O(\frac{1}{\sqrt{T}})$ last-iterate convergence rate, such a guarantee is impossible for OMWU or more generally OFTRL. We now discuss several interesting questions regarding the convergence guarantees of learning in games and leave them as future directions.

**Best-Iterate Convergence Rates** While we focus on the last-iterate (i.e., $\text{DualityGap}(x^T, y^T)$), the weaker notion of best-iterate (i.e., $\min_{t \in [T]} \text{DualityGap}(x^t, y^t)$) is also of both practical and theoretical interest. By definition, we know the best-iterate convergence rate is at least as good as the last-iterate convergence rate and could be much faster. This raises the following question:
**What is the best-iterate convergence rate of OMWU/OFTRL?**
To our knowledge, there are no concrete results on the best-iterate convergence rates of OMWU or other OFTRL algorithms. It is thus interesting to extend our negative results to the best-iterate convergence rates or develop fast best-iterate convergence rates of OMWU/OFTRL.

**Dynamic Step Sizes** Our negative results hold for OFTRL with fixed step sizes. We conjecture that the slow last-iterate convergence of OFTRL persists even with dynamic step sizes. In particular, we believe our counterexamples still work for OFTRL with decreasing step sizes. This is because decreasing the step size makes the players move even slower, and they may be trapped in the wrong direction for a longer time due to the lack of forgetfulness. In Appendix E, we include numerical results for OMWU with adaptive stepsize akin to Adagrad [Duchi et al., 2011], which supports our intuition. We observe the same cycling behavior as for fixed step size. While the cycle is smaller than that of fixed step sizes, the dynamics take more steps to finish each cycle. Investigating the effect of dynamic step sizes on last-iterate convergence rates is an interesting future direction.

**Slow Convergence due to Lack of Forgetfulness** Our work shows that various OFTRL-type algorithms do not have fast last-iterate convergence rates for learning in games. Our proof and hard game instance build on the intuition that these algorithms lack forgetfulness: they do not forget the past quickly. This intuition is also utilized in [Panageas et al., 2023]. In particular, they give an $d \times d$ potential game where the last-iterate convergence rate of the Fictitious Play algorithm, which is equivalent to the Follow-the-Leader (FTL) algorithm, suffers exponential dependence in the dimension $d$. One natural future direction is to formalize the intuition of non-forgetfulness further and give a general condition for algorithms under which they suffer slow last-iterate convergence. It is also interesting to show other lower-bound results for learning in games.

---

### Acknowledgements
We thank the anonymous reviewers for their constructive comments on improving the paper. Yang Cai was supported by the NSF Awards CCF-1942583 (CAREER) and CCF-2342642. Christian Kroer was supported by the Office of Naval Research awards N00014-22-1-2530 and N00014-23-1-2374, and the National Science Foundation awards IIS-2147361 and IIS-2238960. Julien Grand-Clément was supported by Hi! Paris and Agence Nationale de la Recherche (Grant 11-LABX-0047). Haipeng Luo was supported by NSF award IIS-1943607. Weiqiang Zheng was supported by the NSF Awards CCF-1942583 (CAREER), CCF-2342642, and a Research Fellowship from the Center for Algorithms, Data, and Market Design at Yale (CADMY).

---

### References
Ioannis Anagnostides, Ioannis Panageas, Gabriele Farina, and Tuomas Sandholm. On last-iterate convergence beyond zero-sum games. In *International Conference on Machine Learning*, pages 536-581. PMLR, 2022.

James P Bailey and Georgios Piliouras. Multiplicative weights update in zero-sum games. In *Proceedings of the 2018 ACM Conference on Economics and Computation*, pages 321-338, 2018.

Mark Braverman, Jieming Mao, Jon Schneider, and Matt Weinberg. Selling to a no-regret buyer. In *Proceedings of the 2018 ACM Conference on Economics and Computation*, pages 523-538, 2018.

Noam Brown and Tuomas Sandholm. Superhuman AI for heads-up no-limit poker: Libratus beats top professionals. *Science*, 359(6374):418-424, 2018.

Yang Cai, Argyris Oikonomou, and Weiqiang Zheng. Finite-time last-iterate convergence for learning in multi-player games. In *Advances in Neural Information Processing Systems (NeurIPS)*, 2022.

Yun Kuen Cheung and Georgios Piliouras. Vortices instead of equilibria in minmax optimization: Chaos and butterfly effects of online learning in zero-sum games. In *Conference on Learning Theory*, pages 807-834. PMLR, 2019.

Constantinos Daskalakis and Ioannis Panageas. The limit points of (optimistic) gradient descent in min-max optimization. *Advances in neural information processing systems (NeurIPS)*, 2018.

Constantinos Daskalakis and Ioannis Panageas. Last-iterate convergence: Zero-sum games and constrained min-max optimization. In *10th Innovations in Theoretical Computer Science Conference (ITCS)*, 2019.

Constantinos Daskalakis, Andrew Ilyas, Vasilis Syrgkanis, and Haoyang Zeng. Training GANs with optimism. In *International Conference on Learning Representations (ICLR)*, 2018.

Constantinos Daskalakis, Maxwell Fishelson, and Noah Golowich. Near-optimal no-regret learning in general games. *Advances in Neural Information Processing Systems (NeurIPS)*, 2021.

John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online learning and stochastic optimization. *Journal of machine learning research*, 12(7), 2011.

Gabriele Farina, Christian Kroer, and Tuomas Sandholm. Faster game solving via predictive blackwell approachability: Connecting regret matching and mirror descent. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 35, pages 5363-5371, 2021.

Gabriele Farina, Chung-Wei Lee, Haipeng Luo, and Christian Kroer. Kernelized multiplicative weights for 0/1-polyhedral games: Bridging the gap between learning in extensive-form and normal-form games. In *International Conference on Machine Learning (ICML)*, pages 6337-6357, 2022.

Genevieve E Flaspohler, Francesco Orabona, Judah Cohen, Soukayna Mouatadid, Miruna Oprescu, Paulo Orenstein, and Lester Mackey. Online learning with optimism and delay. In *International Conference on Machine Learning*, pages 3363-3373. PMLR, 2021.

Noah Golowich, Sarath Pattathil, and Constantinos Daskalakis. Tight last-iterate convergence rates for no-regret learning in multi-player games. *Advances in neural information processing systems (NeurIPS)*, 2020a.

Noah Golowich, Sarath Pattathil, Constantinos Daskalakis, and Asuman Ozdaglar. Last iterate is slower than averaged iterate in smooth convex-concave saddle point problems. In *Conference on Learning Theory (COLT)*, 2020b.

Eduard Gorbunov, Adrien Taylor, and Gauthier Gidel. Last-iterate convergence of optimistic gradient method for monotone variational inequalities. In *Advances in Neural Information Processing Systems*, 2022.

Sergiu Hart and Andreu Mas-Colell. A simple adaptive procedure leading to correlated equilibrium. *Econometrica*, 68(5):1127-1150, 2000.

Yu-Guan Hsieh, Franck Iutzeler, Jérôme Malick, and Panayotis Mertikopoulos. On the convergence of single-call stochastic extra-gradient methods. *Advances in Neural Information Processing Systems*, 32, 2019.

Yu-Guan Hsieh, Kimon Antonakopoulos, and Panayotis Mertikopoulos. Adaptive learning in continuous games: Optimal regret bounds and convergence to nash equilibrium. In *Conference on Learning Theory*, pages 2388-2422. PMLR, 2021.

Wouter M Koolen, Manfred K Warmuth, Jyrki Kivinen, et al. Hedging structured concepts. In *COLT*, pages 93-105. Citeseer, 2010.

Rachitesh Kumar, Jon Schneider, and Balasubramanian Sivan. Strategically-robust learning algorithms for bidding in first-price auctions. In *Proceedings of the 2024 ACM Conference on Economics and Computation*, 2024.

Chung-Wei Lee, Christian Kroer, and Haipeng Luo. Last-iterate convergence in extensive-form games. *Advances in Neural Information Processing Systems*, 34:14293-14305, 2021.

Haipeng Luo. Lecture note 2, Introduction to Online Learning. 2022. URL https://haipeng-luo.net/courses/CSCI659/2022_fall/lectures/lecture2.pdf.

Panayotis Mertikopoulos, Christos Papadimitriou, and Georgios Piliouras. Cycles in adversarial regularized learning. In *Proceedings of the twenty-ninth annual ACM-SIAM symposium on discrete algorithms*, pages 2703-2717. SIAM, 2018.

Panayotis Mertikopoulos, Bruno Lecouat, Houssam Zenati, Chuan-Sheng Foo, Vijay Chandrasekhar, and Georgios Piliouras. Optimistic mirror descent in saddle-point problems: Going the extra (gradient) mile. In *International Conference on Learning Representations (ICLR)*, 2019.

Aryan Mokhtari, Asuman E Ozdaglar, and Sarath Pattathil. Convergence rate of $O(1/k)$ for optimistic gradient and extragradient methods in smooth convex-concave saddle point problems. *SIAM Journal on Optimization*, 30(4):3230-3251, 2020.

Rémi Munos, Michal Valko, Daniele Calandriello, Mohammad Gheshlaghi Azar, Mark Rowland, Zhaohan Daniel Guo, Yunhao Tang, Matthieu Geist, Thomas Mesnard, Andrea Michi, et al. Nash learning from human feedback. *arXiv preprint arXiv:2312.00886*, 2023.

Ioannis Panageas, Nikolas Patris, Stratis Skoulakis, and Volkan Cevher. Exponential lower bounds for fictitious play in potential games. In *Thirty-seventh Conference on Neural Information Processing Systems*, 2023. URL https://openreview.net/forum?id=tkenkPYkxj.

Julien Perolat, Bart De Vylder, Daniel Hennes, Eugene Tarassov, Florian Strub, Vincent de Boer, Paul Muller, Jerome T Connor, Neil Burch, Thomas Anthony, et al. Mastering the game of stratego with model-free multiagent reinforcement learning. *Science*, 378(6623):990-996, 2022.

Sasha Rakhlin and Karthik Sridharan. Optimization, learning, and games with predictable sequences. *Advances in Neural Information Processing Systems*, 2013.

Vasilis Syrgkanis, Alekh Agarwal, Haipeng Luo, and Robert E Schapire. Fast convergence of regularized learning in games. *Advances in Neural Information Processing Systems (NeurIPS)*, 2015.

Eiji Takimoto and Manfred K Warmuth. Path kernels and multiplicative updates. *The Journal of Machine Learning Research*, 4:773-818, 2003.

Oskari Tammelin, Neil Burch, Michael Johanson, and Michael Bowling. Solving heads-up limit texas hold'em. In *Twenty-fourth international joint conference on artificial intelligence*, 2015.

Tim van Erven. Why FTRL is better than online mirror descent. https://www.timvanerven.nl/blog/ftrl-vs-omd/. 2021. Accessed: 2024-05-22.

Emmanouil-Vasileios Vlatakis-Gkaragkounis, Lampros Flokas, Thanasis Lianeas, Panayotis Mertikopoulos, and Georgios Piliouras. No-regret learning and mixed nash equilibria: They do not mix. *Advances in Neural Information Processing Systems*, 33:1380-1391, 2020.

Chen-Yu Wei, Chung-Wei Lee, Mengxiao Zhang, and Haipeng Luo. Linear last-iterate convergence in constrained saddle-point optimization. In *International Conference on Learning Representations (ICLR)*, 2021.

Martin Zinkevich, Michael Johanson, Michael Bowling, and Carmelo Piccione. Regret minimization in games with incomplete information. *Advances in neural information processing systems*, 20, 2007.

---

### Contents
1 Introduction | 1
1.1 Related Work | 4
2 Preliminaries and Problem Setup | 5
3 Slow Convergence of OFTRL: A Hard Game Instance | 6
3.1 Basic Properties | 6
3.2 Slow Last-Iterate Convergence | 7
4 Extension to Higher Dimensions | 9
5 Conclusion and Discussions | 10
A Missing Proofs in Section 2 | 14
A.1 Proof of Lemma 1 | 14
A.2 Proof of Proposition 1 | 14
B Missing Proofs in Section 3 | 14
B.1 Proof of Lemma 2 | 14
B.2 Proof of Theorem 1 | 14
B.3 Proof of Theorem 2 | 17
C Verifying Assumption 2 for Different Regularizers | 18
C.1 Negative Entropy | 18
C.2 Squared Euclidean Norm Regularizer | 19
C.3 Log Barrier | 20
C.4 Negative Tsallis Entropy | 20
D Proof of Theorem 3 | 22
E Numerical Experiments with Adaptive Stepsizes | 22

---

### A Missing Proofs in Section 2
**A.1 Proof of Lemma 1**
**Proof.** Let $e_1 < e_2$. Denote $x_1 = F_{\eta,R}(e_1)$ and $x_2 = F_{\eta,R}(e_2)$. By definition, we have
$e_2(x_2 - x_1) \le \frac{1}{\eta}(R(x_1) - R(x_2)) \le e_1(x_2 - x_1)$.
Since $e_1 < e_2$, we have $x_2 \le x_1$. $\square$

**A.2 Proof of Proposition 1**
**Proof.** By definition,
$F_{\eta,R}(E/\eta) = \arg\min_{x \in [0, 1]} \{ x \cdot \frac{E}{\eta} + \frac{1}{\eta} R(x) \} = \arg\min_{x \in [0, 1]} \{ x \cdot E + R(x) \} = F_{1,R}(E)$.
The second claim on the Lipschitzness follows directly. $\square$

### B Missing Proofs in Section 3
**B.1 Proof of Lemma 2**
**Proof.** We have
$\text{DualityGap}(x, y) = \max_{\tilde{y} \in δ^2} x^\top A_δ \tilde{y} - \min_{\tilde{x} \in δ^2} \tilde{x}^\top A_δ y$
$= \max_{i \in \{1,2\}} (A_δ^\top x)[i] - \min_{i \in \{1,2\}} (A_δ y)[i]$
$= (\frac{1}{2} + δ)x[1] - (1 - y[1])$
$\ge \frac{1}{2} \frac{1 + 2δ}{1 + δ} - \frac{1}{2} + \epsilon$
$\ge \epsilon$. ($x[1] \ge \frac{1}{1+δ}, \epsilon > 0$) $\square$

**B.2 Proof of Theorem 1**
**Proof.** Recall that $c_1 = \frac{1}{2} - F_{1,R}(\frac{1}{20L})$ defined in Assumption 2. We fix any $δ < \min \{ \frac{c_1}{6}, \frac{c_1^2}{300}, δ' \}$. Since $δ < δ'$, Assumption 2 holds. We will prove that there exists an iteration $t \ge \frac{c_1}{3\eta Lδ}$ with duality gap $c_2$.

**Proof Plan:** We decompose the analysis into three stages as shown in Figure 3. Below, we describe the three stages and the high-level ideas in our proof.
* **Stage I:** Recall that $x^1[1] = y^1[1] = \frac{1}{2}$. In Stage I, we show that $x^t[1]$ will increase and denote $T_1 \ge 1$ the first iteration where $x^t[1] \ge \frac{1}{1+δ}$. The existence of $T_1$ can be proved by contradiction (Claim 1). Since before the end of Stage I, $x^t[1] < \frac{1}{1+δ}$, the loss vector for the $y$-player satisfies $e_y^t = l_y^t[1] - l_y^t[2] \ge 0$ meaning action 1 is worse than action 2. We will prove that finally $y^{T_1}[1] \le \frac{1}{2} - c_1$.
* **Stage II:** Now we have that $y^{T_1}[1] \le \frac{1}{2} - c_1$, we denote $T_2 > T_1$ the first iteration where $y^{T_2}[1] \ge \frac{1}{2(1+δ)} > \frac{1}{2} - c_1$. We remark that in order to increase $y^t[1]$ the loss vector must satisfy $e_y^t < 0$. However, the game matrix $A_δ$ guarantees that $e_y^t \ge -δ$ no matter what the $x$-player is playing. Thus by the $\eta L$-Lipschitzness of $F_{\eta,R}$ (Lemma 4), the increase in $y^t[1]$ is at most $\eta Lδ$. Therefore, we know $T_2 - T_1 = \Omega(\frac{c_1}{\eta Lδ})$. But during $[T_1, T_2]$ for the $x$-player, we have $e_x^t < 0$ which implies its cumulative loss $E_x^{T_2} \le E_x^{T_1} - \Omega(\frac{1}{\eta Lδ})$. In other words, $x^t[1]$ is very close to 1 and the cumulative loss for action 1 is much smaller than that of action 2.
* **Stage III:** Now we have $y^{T_2}[1] \ge \frac{1}{2(1+δ)}$ and that $y^t[1]$ could keep increasing if $x^t[1] \ge \frac{1}{1+δ}$ since then the loss satisfies $e_y^t \le 0$. Now the question is how long would the $x$-player stay close to the boundary, i.e, $x^t[1] \ge \frac{1}{1+δ}$. Since OFTRL-type algorithms are not forgetful, this happens only when $E_x^t \ge E_x^{T_1}$ (recall $x^{T_1}[1] \ge \frac{1}{1+δ}$). But we have at the end of stage II, $E_x^{T_2} \le E_x^{T_1} - \Omega(\frac{1}{\eta Lδ})$. Since $e_x^t$ is bounded by a constant, we know $x^t[1] \ge \frac{1}{1+δ}$ even after $\Omega(\frac{1}{\eta Lδ})$ iterations. Define $T_3 = T_2 + \Omega(\frac{1}{\eta Lδ})$. During $[T_2, T_3]$, the $y$-player always receives loss such that $e_y^t \le 0$ and we prove that $y^{T_3}[1] \ge \frac{1}{2} + c_2$ for some constant $c_2$.
* **Conclusion:** Finally we get one iteration $T_3 \ge \Omega(\frac{1}{\eta Lδ})$ with $x^{T_3}[1] \ge \frac{1}{1+δ}$ and $y^{T_3}[1] \ge \frac{1}{2} + c_2$. Using Lemma 2, the duality gap of $(x^{T_3}, y^{T_3})$ is at least $c_2$.

**Stage I:** We know $x^1[1] = y^1[1] = \frac{1}{2}$. We define (i) $T_s > 1$ to be the smallest iteration such that $x^{T_s}[1] \ge \frac{3}{4}$ and (ii) $T_1 > T_s$ to be the smallest iteration such that $x^{T_1}[1] \ge \frac{1}{1+δ}$. Both $T_s$ and $T_1$ must exist, and the reason will become clear in the following analysis. We postpone the proof of this fact in Claim 1 at the end of this paragraph. Notice from Proposition 2, the difference $e_x^t$ is lower bounded: $e_x^t \ge -1/2$ for any $t$. Thus $E_x^{t-1} + e_x^{t-1} \ge -t/2$ for any $t \ge 1$. Since $x^{T_s}[1] \ge 3/4 > 1/2$, we know that $E_x^{T_s-1} + e_x^{T_s-1} < 0$. As $F_{\eta,R}$ is $\eta L$-Lipschitz, this implies
$1/4 \le x^{T_s}[1] - x^1[1] \le \eta L \cdot |E_x^{T_s-1} + e_x^{T_s-1}| \le \frac{L \eta T_s}{2}$
$T_s \ge \frac{1}{2\eta L}$.
Since $x^t[1] < 3/4$ for all $1 \le t \le T_s - 1$, we know that $e_y^t = l_y^t[1] - l_y^t[2] = 1 - (1 + δ)x^t[1] > \frac{1 - 3δ}{4} \ge 1/5$ (as $δ \le 1/15$) for all $1 \le t \le T_s - 1$. Moreover, for all $1 \le t \le T_1 - 1$, we know that $e_y^t \ge 0$ as $x^t[1] \le \frac{1}{1+δ}$. Since the difference $e_y^t$ is at least $1/5$ for all $t \le T_s - 1$ and remains non-negative for all $t \in [T_s, T_1 - 1]$, we can conclude that for all $T_s \le t \le T_1$
$y^t[1] = F_{\eta,R}(E_y^{t-1} + e_y^{t-1}) \le F_{\eta,R}(E_y^{t-1})$,
and moreover
$F_{\eta,R}(E_y^{t-1}) \le F_{\eta,R}(\frac{T_s-1}{5}) \le F_{\eta,R}(\frac{1}{20L\eta}) = \frac{1}{2} - c_1$. ($T_s - 1 \ge \frac{1}{2\eta L} - 1 \ge \frac{1}{4L\eta}$)
This completes the proof of Stage I, where $x^{T_1}[1] \ge \frac{1}{1+δ}$ and $y^{T_1}[1] \le \frac{1}{2} - c_1$. Before we proceed to the next stage, we prove the existence of $T_s$ and $T_1$.

**Claim 1.** $T_s$ and $T_1$ exist.
**Proof.** It suffices to prove that $T_1$ exists as it implies the existence of $T_s$. Assume for the sake of contradiction that $T_1$ does not exist, i.e., $x^t[1] < \frac{1}{1+δ}$ for all $t \ge 1$. By the same analysis as for Stage I, we get $y^t[1] \le \frac{1}{2} - c_1$ for all $t \ge \frac{1}{2\eta L}$. This implies $e_x^t = -1/2 + (1 + δ)y^t[1] \le \frac{δ}{2} - c_1 \le -c_1/2$ for all $t \ge \frac{1}{2\eta L}$. Then $E_x^t + e_x^t \to -\infty$ as $t \to +\infty$. As a consequence, $x^t[1] = F_{\eta,R}(E_x^{t-1} + e_x^{t-1}) \to 1$ as $t \to +\infty$ by item 2 in Assumption 1. But this contradicts with the assumption that $x^t[1] < \frac{1}{1+δ}$ for all $t \ge 1$. This completes the proof. $\square$

**Stage II**
We define
$T := \lfloor \frac{c_1}{2L\etaδ} \rfloor \in [ \frac{c_1}{3L\etaδ}, \frac{c_1}{2L\etaδ} ]$ (4)
where the lower bound on $T$ holds since $\frac{c_1}{6L\etaδ} \ge \frac{c_1}{6δ} \ge 1$. We note that $T = \Omega(1/δ)$ since $\eta L \le 1/4$. In Stage I, we have proved that $y^{T_1}[1] \le \frac{1}{2} - c_1$. Define $T_h = T_1 + T$. We claim that for all $t \in [T_1, T_h - 1]$, $y^t[1] \le \frac{1}{2} - \frac{c_1}{2}$. To prove the claim, we first notice that $-δ \le e_y^t \le 1$ for all $t \ge 1$. Then by the monotonicity and the $\eta L$-Lipschitzness of $F_{\eta,R}$ (Lemma 1 and Lemma 4), we get for all $t \in [T_1, T_h - 1]$,
$y^t[1] \le F_{\eta,R}(E_y^{T_1-1}) + \eta L \max \{ E_y^{T_1-1} - E_y^{t-1} - e_y^{t-1}, 0 \}$
$\le \frac{1}{2} - c_1 + \eta L \cdot (t - T_1 + 1)δ$
$\le \frac{1}{2} - c_1 + \eta LTδ$
$\le \frac{1}{2} - \frac{c_1}{2}$
where, in the second-to-last inequality, we use $t - T_1 + 1 \le T \le \frac{c_1}{2\eta Lδ}$ by Equation (4).

Now we denote $T_2 \ge T_h$ the smallest iteration when $y^{T_2}[1] \ge \frac{1}{2(1+δ)}$. The existence of $T_2$ will become clear in the following analysis, and we postpone the proof to Claim 2 at the end of the discussion. Then for all $t \in [T_s, T_2 - 1]$, we have $y^t[1] \le \frac{1}{2(1+δ)}$, which implies $e_x^t \le 0$. Moreover, for all $t \in [T_s, T_1 + T - 1]$ since $y^t[1] \le \frac{1}{2} - \frac{c_1}{2}$ we have
$e_x^t = l_x^t[1] - l_x^t[2] = -1/2 + (1 + δ)y^t[1] \le \frac{-1 + (1 + δ)(1 - c_1)}{2} \le \frac{δ - c_1}{2} \le -c_1/4$. ($δ \le c_1/2$)
Then for any $T_1 + T \le t \le T_2$, we have
$x^t[1] = F_{\eta,R}(E_x^{t-1} + e_x^{t-1})$ ($e_x^{t-1} \le 0$ for all $t \in [T_1 + T, T_2]$)
$\ge F_{\eta,R}(E_x^{T_1 + T - 1}) \ge F_{\eta,R}(-\frac{c_1T}{4} + E_x^{T_1-1}) \ge F_{\eta,R}(-\frac{c_1T}{5} + E_x^{T_1-1} + e_x^{T_1-1})$
where in the last inequality, we use the fact that $\frac{c_1 T}{20} \ge \frac{c_1^2}{60\eta Lδ} \ge 1$.

**Claim 2.** $T_2$ exists.
**Proof.** Assume for the sake of contradiction that $T_2$ does not exist, i.e., $y^t[1] < \frac{1}{2(1+δ)}$ for all $t \ge T_1$ (since we know $y^t[1] \le \frac{1}{2} - \frac{c_1}{2}$ for all $t \in [T_1, T_1 + T - 1]$). Then by the analysis of Stage II and Equation (5), we have $x^t[1] \ge \frac{4}{4+δ}$ for all $t \ge T_1$. This implies $e_y^t \le -3δ/5$ for all $t \ge T_1$. As a result, we have $E_y^{t-1} + e_y^{t-1} \to -\infty$ as $t \to \infty$. By item 2 in Assumption 1, we get $y^t[1] = F_{\eta,R}(E_y^{t-1} + e_y^{t-1}) \ge 1/2$ as $t \to \infty$. But this contradicts with the assumption that $y^t[1] < \frac{1}{2(1+δ)}$ for all $t \ge T_1$. This completes the proof. $\square$

**Stage III**
Recall that we have argued in State I that $F_{\eta,R}(E_x^{T_1-1} + e_x^{T_1-1}) = F_{1,R}(\eta(E_x^{T_1-1} + e_x^{T_1-1})) = x^{T_1}[1] \ge \frac{1}{1+δ}$. By item 1 in Assumption 2, we have that
$F_{\eta,R}(-\frac{c_1T}{10} + E_x^{T_1-1} + e_x^{T_1-1})) \ge F_{\eta,R}(-\frac{c_1^2}{30L\etaδ} + E_x^{T_1-1} + e_x^{T_1-1})) = F_{1,R}(-\frac{c_1^2}{30Lδ} + \eta(E_x^{T_1-1} + e_x^{T_1-1})))$
$\ge \frac{1+c_3}{1+c_3+δ}$. (5)
where the first inequality follows from the definition of $T$ and the monotonicity of $F_{\eta,R}$ (Lemma 1).

Now denote $T_3 = T_2 + \lfloor \frac{c_1T}{10} \rfloor - 2$. For any $T_2 \le t \le T_3$ we know that
$x^t[1] = F_{\eta,R}(E_x^{t-1} + e_x^{t-1}) = F_{\eta,R}(E_x^{T_2-1} + e_x^{T_2-1} + \sum_{k=T_2}^{t-1} e_x^k + e_x^{t-1} - e_x^{T_2-1})$
$\ge F_{\eta,R}(-\frac{c_1T}{5} + E_x^{T_1-1} + e_x^{T_1-1} + \sum_{k=T_2}^{t-1} e_x^k + e_x^{t-1} - e_x^{T_2-1})$
$\ge F_{\eta,R}(-\frac{c_1T}{5} + E_x^{T_1-1} + e_x^{T_1-1} + \frac{c_1T}{10} - 2 + 2) \ge F_{\eta,R}(-\frac{c_1T}{10} + E_x^{T_1-1} + e_x^{T_1-1}))$
$\ge \frac{1+c_3}{1+c_3+δ}$. (by (5))

Note that $1+c_3+δ \le 2$. This implies $e_y^t = 1 - (1 + δ)x^t[1] = -\frac{c_3δ}{1+c_3+δ} \le -\frac{c_3δ}{2}$ for all $T_2 \le t \le T_3$. Moreover, we know that $e_y^t \ge -δ$ for any $t$. Then
$y^{T_3}[1] = F_{\eta,R}(E_y^{T_3-1} + e_y^{T_3-1})) \ge F_{\eta,R}(E_y^{T_2-1} + e_y^{T_2-1} + \sum_{k=T_2}^{T_3-1} e_y^k + e_y^{T_3-1} - e_y^{T_2-1}))$
$\ge F_{\eta,R}(E_y^{T_2-1} + e_y^{T_2-1} - \frac{c_3δ(T_3-T_2)}{2} + δ) \ge F_{\eta,R}(E_y^{T_2-1} + e_y^{T_2-1} - \frac{c_3δ c_1T}{40} + δ)$ ($T_3 - T_2 = \lfloor \frac{c_1T}{10} \rfloor - 2 \ge \frac{c_1T}{20}$)
$\ge F_{\eta,R}(E_y^{T_2-1} + e_y^{T_2-1} - \frac{c_3c_1^2}{120\eta L} + δ)$ ($T \ge \frac{c_1}{3\eta Lδ}$)
$= F_{1,R}(\eta(E_y^{T_2-1} + e_y^{T_2-1}) - \frac{c_3c_1^2}{120L} + \etaδ) \ge F_{1,R}(\eta(E_y^{T_2-1} + e_y^{T_2-1}) - \frac{c_3c_1^2}{120L} + \frac{δ}{4L})$ ($\eta \le \frac{1}{4L}$)

Recall that $F_{1,R}(\eta(E_y^{T_2-1} + e_y^{T_2-1})) = F_{\eta,R}(E_y^{T_2-1} + e_y^{T_2-1}) = y^{T_2}[1] \ge \frac{1}{2(1+δ)}$. By item 2 in Assumption 2, we have $F_{1,R}(\eta(E_y^{T_2-1} + e_y^{T_2-1}) - \frac{c_3 c_1^2}{120L} + \frac{δ}{4L}) \ge 1/2 + c_2$ for some absolute constant $c_2 > 0$. Thus, we have $y^{T_3}[1] \ge 1/2 + c_2$.

Recall that $x^{T_3}[1] \ge \frac{1+c_3}{1+c_3+δ} \ge \frac{1}{1+δ}$. Then by Lemma 2 we can conclude that the duality gap of $(x^{T_3}, y^{T_3})$ is at least $c_2 > 0$. This completes the proof as $T_3 \ge T_2 \ge T \ge \frac{c_1}{3\eta Lδ}$. $\square$

**B.3 Proof of Theorem 2**
**Proof.** Assume for the sake of contradiction that there is a function that satisfies both conditions. Then for any $A \in [0, 1]^{2 \times 2}$ we have the OFTRL learning dynamics over $A$ satisfies
1. $\text{DualityGap}(x^T, y^T) \le f(2, 2, T)$ for all $T$.
2. $\lim_{T \to \infty} f(2, 2, T) \to 0$.
Since $\lim_{T \to \infty} f(2, 2, T) \to 0$ we know there exists $T_0 > 0$ such that for any $t \ge T_0, \text{DualityGap}(x^t, y^t) \le f(2, 2, t) < c_2$. Now let $δ \le \min \{ \hat{δ}, \frac{c_1}{3\eta LT_0} \}$. Then by Theorem 1, we know there exists an iteration $t \ge \frac{c_1}{3\eta Lδ} \ge T_0$ such that $\text{DualityGap}(x^t, y^t) \ge c_2$. This completes the proof. $\square$

### C Verifying Assumption 2 for Different Regularizers
**Lemma 4.** If the regularizer $R$ is 1-strongly convex, then $F_{1,R}$ is $1/2$-Lipschitz.
**Proof.** Notice that $R(x) + R(1-x)$ is 2-strongly convex. Thus by standard analysis (see e.g., Luo [2022, Lemma 4]) we know $F_{1,R}$ is $1/2$-Lipschitz. $\square$

By Lemma 4, we can choose $L = 1/2$ for any 1-strongly convex regularizer in Assumption 1.

**C.1 Negative Entropy**
**Lemma 5 (Assumption 2 holds for the entropy regularizer).** Consider the negative entropy regularizer $R$ defined as $R(x) = x \log x + (1 - x) \log(1 - x)$. Then $F_{1,R}$ is $L=1/2$-Lipschitz. We have $c_1$ and Assumption 2 holds with $δ' = \frac{c_1^2}{480L}$, $c_2 = F_{1,R}(-\frac{c_1^2}{480L}) - 1/2$, and $c_3 = 1/2$.
**Proof.** It is easy to verify that $F_{1,R}(x)$ has a closed-form representation $F_{1,R}(E) = \frac{1}{1+\exp(E)}$. Thus $L = 1/2$ and $c_1 = 1/2 - F_{1,R}(\frac{1}{20L})$ is a universal constant. We also choose $c_3 = 1/2$. If $F_{1,R}(E) \ge \frac{1}{1+δ} \ge \frac{1}{1+δ}$, then we have $E \le -\log(1/δ)$. We note that $\exp(-\frac{c_1^2}{30Lδ}) \le \frac{1}{1+c_3} \Rightarrow \frac{1}{1+\exp(-\frac{c_1^2}{30Lδ} - \log(1/δ))} \ge \frac{1+c_3}{1+c_3+δ}$. Thus $δ \le δ_1 = \frac{c_1^2}{30L \log(1+c_3))} = \frac{c_1^2}{30 \log(3/2)L}$ suffices for item 1 in Assumption 2.

If $F_{1,R}(E) \ge \frac{1}{2(1+δ)} = \frac{1}{1+1+2δ}$, we have $E \le \log(1+2δ)$. Note that since $\log(1+2y) \le 2y$ for $y > 0$, we have $δ \le \frac{c_3c_1^2}{480L} \Rightarrow -\frac{c_3c_1^2}{120L} + \log(1+2δ) < -\frac{c_3c_1^2}{240L} \Rightarrow F_{1,R}(-\frac{c_3c_1^2}{120L} + E) > F_{1,R}(-\frac{c_3c_1^2}{240L})$. Thus item 2 in Assumption 2 holds for any $δ \le δ_2 = \frac{c_3c_1^2}{480L} = \frac{c_1^2}{960L}$ and $c_2 = F_{1,R}(-\frac{c_3c_1^2}{240L}) - 1/2 = F_{1,R}(-\frac{c_1^2}{480L}) - 1/2$. Combining the above, we know Assumption 2 holds for the negative entropy regularizer with $δ' = \frac{c_1^2}{960L}$ and $c_2 = F_{1,R}(-\frac{c_1^2}{480L}) - 1/2$. $\square$

**C.2 Squared Euclidean Norm Regularizer**
**Lemma 6 (Assumption 2 holds for the Euclidean regularizer).** Consider the Euclidean regularizer $R$ defined as $R(x) = \frac{1}{2}(x^2 + (1 - x)^2)$. We have $L = 1/2$ and $c_1 = 1/20$. We also have Assumption 2 holds with $δ' = \frac{c_1^2}{480L}$, $c_2 = \frac{c_1^2}{960L}$, and $c_3 = 1/2$.
**Proof.** It is easy to verify that $F_{1,R}(\cdot)$ has a closed-form representation
$F_{1,R}(x) = \begin{cases} 1 & \text{if } x \le -1 \\ \frac{1-x}{2} & \text{if } x \in (-1, 1) \\ 0 & \text{if } x \ge 1 \end{cases}$
Thus $F_{1,R}$ is $L$-Lipschitz with $L = 1/2$. Moreover, $c_1 = 1/2 - F_{1,R}(\frac{1}{20L}) = 1/20$. We choose $c_3 = 1/2$. Fix any $E$ such that $F_{1,R}(E) \ge \frac{1}{1+δ}$. We have $E \le -\frac{1-δ}{1+δ} < 0$. We note that for any $δ \le \frac{c_1^2}{30L} = \frac{c_1^2}{15}$, $F_{1,R}(-\frac{c_1^2}{30Lδ} + E) \ge F_{1,R}(-1) = 1$. Thus $δ \le δ_1 = \frac{c_1^2}{30L}$ suffices for item 1 in Assumption 2.

Fix any $E$ such that $F_{1,R}(E) \ge \frac{1}{2(1+δ)}$. We have $E \le \frac{δ}{1+δ} \le δ$. Then for any $δ \le \frac{c_3c_1^2}{240L}$ we have $F_{1,R}(-\frac{c_3c_1^2}{120L} + E) \ge F_{1,R}(-\frac{c_3c_1^2}{240L}) = 1/2 + \frac{c_3c_1^2}{480L}$. Thus item 2 in Assumption 2 holds for any $δ \le δ_2 = \frac{c_1^2}{480L}$ and $c_2 = \frac{c_1^2}{960L}$. Combining the above, we know Assumption 2 holds for the squared Euclidean regularizer with $δ' = \min \{δ_1, δ_2\} = \frac{c_1^2}{480L}$ and $c_2 = \frac{c_1^2}{960L}$. $\square$

**C.3 Log Barrier**
**Lemma 7 (Assumption 2 holds for the log barrier).** Consider the log barrier regularizer $R$ defined as $R(x) = -\log(x) - \log(1 - x)$. Then Assumption 2 holds with the following choices of constants:
1. $c_1 = \sqrt{\frac{1}{4} + 400L^2} - 20L > 0$.
2. $c_3 = \frac{c_1^2}{60L}$
3. $c_2 = \sqrt{\frac{1}{4} + (\frac{c_3c_1^2}{240L})^2} - \frac{c_3c_1^2}{240L} > 0$.
4. $δ' = \frac{c_3c_1^2}{2160L}$
**Proof.** By setting the gradient of $x \cdot E + R(x)$ to 0, we get a closed-form expression of $F_{1,R}$:
$F_{1,R}(E) = \begin{cases} \frac{1}{2} + \frac{1}{E} - \sqrt{\frac{1}{4} + \frac{1}{E^2}} & \text{if } E > 0 \\ \frac{1}{2} & \text{if } E = 0 \\ \frac{1}{2} + \frac{1}{E} + \sqrt{\frac{1}{4} + \frac{1}{E^2}} & \text{if } E < 0. \end{cases}$
For $x \in (0, 1)$, the $F_{1,R}$ function admits an inverse function defined as $F_{1,R}^{-1}(x) = \frac{2x-1}{x^2-x}$. Thus we know $E_0 := F_{1,R}^{-1}(\frac{1}{1+δ}) = -\frac{1-δ^2}{δ}$ satisfies $F_{1,R}(E_0) = \frac{1}{1+δ}$. Moreover, we can calculate $F_{1,R}^{-1}(\frac{1+c_3}{1+c_3+δ}) = -\frac{(1+c_3)^2-δ^2}{(1+c_3)δ} = -\frac{1+c_3}{δ} + \frac{δ}{1+c_3} = E_0 - \frac{c_3}{δ} - \frac{c_3δ}{1+c_3}$. Thus we can choose $c_3 = \frac{c_1^2}{60L}$ so that $E_0 - \frac{c_1^2}{30Lδ} = E_0 - \frac{2c_3}{δ} \le E_0 - \frac{c_3}{δ} - \frac{c_3δ}{1+c_3}$. Thus we have $F_{1,R}(E_0 - \frac{c_1^2}{30Lδ}) \ge F_{1,R}(E_0 - \frac{c_3}{δ} - \frac{c_3δ}{1+c_3}) \ge \frac{1+c_3}{1+c_3+δ}$. (since $δ < 1/2$ and $c_3 > 0$)

We calculate $E_1 := F_{1,R}^{-1}(\frac{1}{2(1+δ)}) = \frac{4(δ+δ^2)}{1+2δ} \le 8δ$. Then we can choose $δ \le δ' := \frac{c_3c_1^2}{2160L}$. Then we have $F_{1,R}(-\frac{c_3c_1^2}{120L} + \frac{δ}{4L} + E_1) \ge F_{1,R}(-\frac{c_3c_1^2}{120L} + 9δ) \ge F_{1,R}(-\frac{c_3c_1^2}{240L}) = \frac{1}{2} + c_2$, where $c_2 = \sqrt{\frac{1}{4} + (\frac{c_3c_1^2}{240L})^2} - \frac{c_3c_1^2}{240L} > 0$ by the closed-form expression of $F_{1,R}$. $\square$

**C.4 Negative Tsallis Entropy**
For $x \in [0, 1]$ the negative Tsallis entropy is a family of regularizers parameterized by $\beta \in (0, 1)$: $R(x) = \frac{1 - x^\beta}{1 - \beta}$. The corresponding $F_{1,R}$ is defined as $F_{1,R}(E) = \arg\min_{x \in (0, 1)} \{ x \cdot E + \frac{1 - x^\beta}{1 - \beta} + \frac{1 - (1 - x)^\beta}{1 - \beta} \}$. For $x \in (0, 1)$ we note that $F_{1,R}$ has an inverse function $F_{1,R}^{-1}(x) = \frac{\beta}{1 - \beta} (x^{\beta-1} - (1 - x)^{\beta-1})$. (6)

**Lemma 8 (Assumption 2 holds for Tsallis entropy).** Consider Tsallis entropy parameterized by $\beta \in (0, 1)$. Then $L = \frac{1}{2\beta}$ and Assumption 2 holds with the following choices of constants:
1. $c_1 = \frac{1}{2} - F_{1,R}(\frac{1}{20L}) > 0$.
2. $c_3 = 1/2$.
3. $c_2 = F_{1,R}(-\frac{c_3c_1^2}{240L}) - 1/2 > 0$.
4. $δ' = \min \{ (\frac{c_1^2(1 - \beta)}{120L\beta c_3^{1-\beta}})^{1/\beta}, \frac{c_3c_1^2}{120}, \frac{1 - \beta}{8\beta} \cdot \frac{c_3c_1^2}{480L} \}$.
**Proof.** We choose $c_3 = 1/2$. We have $c_1 = \frac{1}{2} - F_{1,R}(\frac{1}{20L})$ is a constant. We note that $E_0 := F_{1,R}^{-1}(\frac{1}{1+δ}) = \frac{\beta}{1 - \beta} ((1 + δ)^{1 - \beta} - (\frac{1 + δ}{δ})^{1 - \beta})$ satisfies $F_{1,R}(E_0) = \frac{1}{1+δ}$. Similarly, we calculate $E_1 := F_{1,R}^{-1}(\frac{1+c_3}{1+c_3+δ}) = \frac{\beta}{1 - \beta} ((\frac{1+c_3+δ}{1+c_3})^{1 - \beta} - (\frac{1+c_3+δ}{δ})^{1 - \beta}) \ge \frac{\beta}{1 - \beta} ((1 + δ)^{1 - \beta} - 2 - (\frac{1+c_3+δ}{δ})^{1 - \beta}) \ge \frac{\beta}{1 - \beta} ((1 + δ)^{1 - \beta} - (\frac{1 + δ}{δ})^{1 - \beta} - (\frac{c_3}{δ})^{1 - \beta} - 2) = E_0 - \frac{\beta}{1 - \beta} ((\frac{c_3}{δ})^{1 - \beta} + 2)$, where in the first inequality we use the fact that $(1 + δ)^{1 - \beta} \le 2$ since $δ \le 1$; the second inequality we use the inequality $(x + y)^{1 - \beta} \le x^{1 - \beta} + y^{1 - \beta}$. We note that $δ \le δ_1 := (\frac{c_1^2(1 - \beta)}{120L\beta c_3^{1-\beta}})^{1/\beta} \Rightarrow -\frac{c_1^2}{30Lδ} \le -\frac{\beta}{1 - \beta} ((\frac{c_3}{δ})^{1 - \beta} + 2)$. (7)
Thus for any $δ \le δ_1$, we have for any $E$ such that $F_{1,R}(E) \ge \frac{1}{1+δ}, -\frac{c_1^2}{30Lδ} + E \le -\frac{c_1^2}{30Lδ} + E_0 \le E_0 - \frac{\beta}{1 - \beta} ((\frac{c_3}{δ})^{1 - \beta} + 2) \le E_1$. The above implies $F_{1,R}(-\frac{c_1^2}{30Lδ} + E) \ge \frac{1+c_3}{1+c_3+δ}$ and the first item in Assumption 2 is satisfied.

We define $E_2 := F_{1,R}^{-1}(\frac{1}{2(1+δ)}) = \frac{\beta}{1 - \beta} ((2 + 2δ)^{1 - \beta} - (\frac{2 + 2δ}{1 + 2δ})^{1 - \beta}) = \frac{\beta}{1 - \beta} (2 + 2δ)^{1 - \beta} \cdot (1 - (\frac{1}{1 + 2δ})^{1 - \beta}) \le \frac{4\beta}{1 - \beta} \cdot (1 - (1 - \frac{2δ}{1 + 2δ})) \le \frac{4\beta}{1 - \beta} \cdot \frac{2δ}{1 + δ} = \frac{8\betaδ}{(1 - \beta)(1 + δ)}$, where in the first inequality we use $(2 + 2δ)^{1 - \beta} \le 4$ since $0 \le δ \le 1$ and $\beta \in (0, 1)$; in the second inequality we use the basic inequality $(1 - x)^r \le 1 - rx$ for $r, x \in (0, 1)$. We define $δ_2 := \min \{ \frac{c_3c_1^2}{120}, \frac{1 - \beta}{8\beta} \cdot \frac{c_3c_1^2}{480L} \}$. Then for any $δ \le δ_2$ and $E$ such that $F_{1,R}[E] \ge \frac{1}{2(1+δ)}$, we have $-\frac{c_3c_1^2}{120L} + \frac{δ}{4L} + E \le -\frac{c_3c_1^2}{120L} + \frac{δ}{4L} + E_2 \le -\frac{c_3c_1^2}{120L} + \frac{c_3c_1^2}{480L} + \frac{8\betaδ}{(1 - \beta)(1 + δ)} \le -\frac{c_3c_1^2}{120L} + \frac{c_3c_1^2}{480L} + \frac{c_3c_1^2}{480L} = -\frac{c_3c_1^2}{240L}$. Thus we know $F_{1,R}(-\frac{c_3c_1^2}{120L} + \frac{δ}{4L} + E) \ge F_{1,R}(-\frac{c_3c_1^2}{240L})$ and item 2 in Assumption 2 is satisfied by $c_2 = F_{1,R}(-\frac{c_3c_1^2}{240L}) - 1/2 > 0$. Combining the above, we can choose $δ' = \min \{ δ_1, δ_2 \}$ so that both items in Assumption 2 hold for $δ \le δ'$. $\square$

### D Proof of Theorem 3
Recall the equivalent 2-dimensional problem:
$\hat{x}^t = \text{Dupl} [ \arg\min_{x \in \frac{1}{n} δ^2} \{ \frac{n}{n^\alpha} \langle x, \sum_{\tau=1}^{t-1} l^\tau + l^{t-1} \rangle + \frac{n}{\eta} R_1(x[1]) + \frac{n}{\eta} R_2(x[2]) \} ]$
$= \text{Dupl} [ \frac{1}{n} \cdot \arg\min_{x \in δ^2} \{ \frac{n}{n^\alpha} \langle \frac{1}{n} x, \sum_{\tau=1}^{t-1} l^\tau + l^{t-1} \rangle + \frac{n}{\eta} R(x/n) \} ]$
$= \text{Dupl} [ \frac{1}{n} \cdot \arg\min_{x \in δ^2} \{ \langle x, \sum_{\tau=1}^{t-1} l^\tau + l^{t-1} \rangle + \frac{n^{\alpha+1}}{\eta} R(x/n) \} ]$

* **Euclidean regularizer:** this regularizer is homogeneous of degree two. Choosing $\alpha=1$, the inner minimization problem is exactly the same as the one solved by OFTRL in two dimensions.
* **Entropy regularizer:** we set $\alpha=0$ to get equivalence: $nR(x/n) = \sum_{i=1}^2 x[i] \log(x[i]/n) = \sum_{i=1}^2 x[i] \log x[i] - \sum_{i=1}^2 x[i] \log n = \sum_{i=1}^2 x[i] \log x[i] - \log n$. Now we have equivalence because the last term is a constant that does not affect the argmin.
* **Log regularizer:** we set $\alpha=-1$ to get equivalence, using similar logic as for entropy: $R(x/n) = \sum_{i=1}^2 -\log(x[i]/n) = 2 \log n + \sum_{i=1}^2 -\log x[i]$.
* **Tsallis entropy regularizer:** we set $\alpha=-1+\beta$ to get equivalence, using similar logic as for entropy: $n^\beta R(x/n) = n^\beta \cdot \frac{1 - \sum_{i=1}^2 (x[i]/n)^\beta}{1 - \beta} = \frac{n^\beta - 1}{1 - \beta} + \frac{1 - \sum_{i=1}^2 x[i]^\beta}{1 - \beta}$.

### E Numerical Experiments with Adaptive Stepsizes
In this section we present our numerical results when OFTRL and OOMD are instantiated with adaptive stepsize [Duchi et al., 2011]: $\eta_t = 1/\sqrt{\epsilon + \sum_{k=1}^{t-1} \|l_k\|_k^2}$ with some constant $\epsilon > 0$. We present our numerical experiments in Figure 4, where we choose $\epsilon=0.1$.

