
IMPORTANT: The file content has been truncated.
Status: Showing lines 1-2000 of 6233 total lines.
Action: To read more of the file, you can use the 'start_line' and 'end_line' parameters in a subsequent 'read_file' call. For example, to read the next section of the file, use start_line: 2001.

--- FILE CONTENT (truncated) ---
# A Modern Introduction to Online Learning

Francesco Orabona
KAUST
francesco@orabona.com

December 23, 2025

# Contents

Abstract ix
1 What is Online Learning? 1
1.1 History Bits 5
2 Online Subgradient Descent 7
2.1 Online Learning with Convex Differentiable Losses 7
2.1.1 Convex Analysis Bits: Convexity 8
2.1.2 Online Gradient Descent 10
2.1.3 Unit Analysis Bits 13
2.2 Online Subgradient Descent 15
2.2.1 Convex Analysis Bits: Subgradients 15
2.2.2 Analysis with Subgradients 17
2.3 From Convex Losses to Linear Losses 18
2.4 History Bits 18
2.5 Exercises 19
3 Online-to-Batch Conversions 20
3.1 From Online Learning to Stochastic Optimization 20
3.1.1 Bits on Concentration Inequalities 22
3.1.2 High-Probability Guarantees for Online-to-Batch Conversion 23
3.2 Application: Agnostic PAC Learning 24
3.3 History Bits 27
3.4 Exercises 27
4 Beyond $\sqrt{T}$ Regret 28
4.1 Strong Convexity and Online Subgradient Descent 28
4.1.1 Convex Analysis Bits: Strong Convexity 28
4.1.2 Online Subgradient Descent for Strongly Convex Losses 31
4.2 Adaptive Algorithms: $L^\star$ bounds and AdaGrad 32
4.2.1 Adaptive Learning Rates for Online Subgradient Descent 32
4.2.2 Convex Analysis Bits: Dual Norms, Smooth and Self-Bounded Functions 34
4.2.3 $L^\star$ bounds 36
4.2.4 AdaGrad 37
4.3 History Bits 39
4.4 Exercises 40

# List of Definitions

2.2 Definition (Convex Set) 8
2.3 Definition (Convex Function) 9
2.17 Definition (Proper Function) 15
2.19 Definition (Subgradient) 15
2.28 Definition (Lipschitz Function) 17
3.6 Definition (Martingale) 22
3.7 Definition (Supermartingale) 22
3.16 Definition (Agnostic-PAC-learnable) 25
4.1 Definition (Strongly Convex Function) 28
4.15 Definition (Dual Norm) 34
4.21 Definition (Smooth Function) 35
4.25 Definition (Self-bounded Function) 36
6.3 Definition (Strictly Convex Function) 53
6.4 Definition (Bregman Divergence) 53
6.12 Definition (Closed Function) 57
6.14 Definition (Fenchel Conjugate) 57
7.18 Definition (Group Norm) 89
7.19 Definition (Absolutely Symmetric Function) 90
7.31 Definition (Exp-Concave Function) 96
9.1 Definition (Distance to a set $\mathcal{V}$ and generalized projection) 117
9.11 Definition (Bregman Projection) 121
11.5 Definition (Subgaussian Random Variable) 150
12.1 Definition (Saddle Point) 160
12.9 Definition (Duality Gap) 162
12.10 Definition ($\epsilon$-Saddle-Point) 163
13.2 Definition (Type of a sequence of symbols) 183
14.6 Definition (Mixable Function) 201
C.5 Definition (Gamma Function) 226
C.6 Definition (Digamma Function) 226
D.1 Definition (Bounded Set) 227
D.2 Definition (Open and Closed Sets) 227
D.4 Definition (Neighborhood) 227
D.5 Definition (Interior point and Interior of a Set) 227
D.6 Definition (Boundary points and Boundary of a Set) 227

# List of Algorithms

2.1 Projected Online Gradient Descent 10
2.2 Projected Online Subgradient Descent 17
4.1 AdaGrad for Hyperrectangles 38
6.1 Online Mirror Descent 54
6.2 Exponentiated Gradient 65
6.3 Learning with Expert Advice through Randomization 70
6.4 Combining Online Subgradient Descent (OSD) instances with Exponentiated Gradient (EG) 71
6.5 Optimistic Online Mirror Descent 73
7.1 Follow-the-Regularized-Leader Algorithm 77
7.2 Follow-the-Regularized-Leader Algorithm on Linearized Losses 81
7.3 AdaHedge Algorithm 88
7.4 FTRL with Group Norms for Linear Regression 90
7.5 Follow-the-Regularized-Leader Algorithm with "Quadratized" Losses 93
7.6 Online Newton Step Algorithm 94
7.7 Vovk-Azoury-Warmuth Forecaster 98
7.8 Optimistic Follow-the-Regularized-Leader Algorithm 100
8.1 Randomized Online Linear Classifier through FTRL 108
8.2 Perceptron Algorithm 109
8.3 The Gaptron Algorithm for Binary Classification 112
9.1 Constrained OCO through (Non-Euclidean) Projections 117
9.2 Sleeping Experts Reduction 123
9.3 Learning Magnitude and Direction Separately 124
10.1 Krichevsky-Trofimov Bettor 128
10.2 Krichevsky-Trofimov OCO Algorithm 131
10.3 OCO with Coordinate-Wise Krichevsky-Trofimov 133
10.4 Learning with Expert Advice based on Krichevsky-Trofimov (KT) Bettors 137
10.5 Sleeping Expert Algorithm based on KT Bettors 140
11.1 Exponential Weights with Explicit Exploration for Multi-Armed Bandit 144
11.2 Exp3 145
11.3 Tsallis-INF Algorithm (OMD version) 148
11.4 Explore-Then-Commit Algorithm 151
11.5 Upper Confidence Bound Algorithm 153
11.6 Tsallis-INF (FTRL version) 156
12.1 Solving Saddle-Point Problems with OCO 163
12.2 Saddle-Point Optimization with OCO and $\mathcal{Y}$-Best-Response 166
12.3 Saddle-Point Optimization with OCO and $\mathcal{X}$-Best-Response 166
12.4 Saddle-Point Optimization with OCO and Alternation 167
12.5 Boosting through OCO 172
12.6 Solving Saddle-Point Problems with Optimistic FTRL 174
12.7 Solving Saddle-Point Problems with Optimistic OMD 176
12.8 Prescient Online Mirror Descent 176
12.9 Be-the-Regularized-Leader Algorithm 178
13.1 $F$-Weighted Portfolio Selection 182
13.2 Online Newton Step for Portfolio Selection 187
14.1 Weighted Average Algorithm (WAA) 196
14.2 Aggregating Algorithm (AA) 203
14.3 Aggregating Algorithm for Improper Online Multiclass Logistic Regression 205
15.1 Adaptive learning for Dynamic EnviRonment (ADER) 211
15.2 Coin Betting for Changing Environment (CBCE) 213
15.3 Efficient Coin Betting for Changing Environment 214

# Abstract

Disclaimer: This is work in progress, I plan to add more material and/or change/reorganize the content.

In this monograph, I introduce the basic concepts of Online Learning through a modern view of Online Convex Optimization. Here, online learning refers to the framework of regret minimization under worst-case assumptions. I present first-order and second-order algorithms for online learning with convex losses, in Euclidean and non-Euclidean settings. All the algorithms are clearly presented as instantiation of Online Mirror Descent or Follow-The-Regularized-Leader and their variants. Particular attention is given to the issue of tuning the parameters of the algorithms and learning in unbounded domains, through adaptive and parameter-free online learning algorithms. Non-convex losses are dealt through convex surrogate losses and through randomization. The bandit setting is also briefly discussed, touching on the problem of adversarial and stochastic multi-armed bandits. These notes do not require prior knowledge of convex analysis and all the required mathematical tools are rigorously explained. Moreover, all the included proofs have been carefully chosen to be as simple and as short as possible.

I want to thank all the people that checked the proofs and reasonings in these notes. In particular, the students in my first class that mercilessly pointed out my mistakes, Nicolò Campolongo that found all the typos in my formulas, and Jake Abernethy for the brainstorming on presentation strategies. Other people that helped me with comments, feedback, references, and/or hunting typos (in alphabetical order): Zeyad Aljaali, Andreas Argyriou, Param Kishor Budhraja, Nicolo Cesa-Bianchi, Sahil Chaudhary, Keyi Chen, Mingyu Chen, Peiqing Chen, Ryan D'Orazio, Gerardo Durán-Martà­n, Alon Gonen, Peijia Guo, Dirk van der Hoeven, Daniel Hsu, Gergely Imreh, Andrew Jacobsen, Emmeran Johnson, Ji-Ha Kim, Andrew Christian Kroer, Joon Kwon, Kwang-Sung Jun, Michał Kempka, Pierre Laforgue, Wei-Cheng Lee, Chuang-Chieh Lin, Shashank Manjunath, David Martà­nez-Rubio, Valentina Masarotto, Aryan Mokhtari, Antoine Moulin, Gergely Neu, Ankit Pensia, Viacheslav D. Potapov, Yousef Radwan, Abed Razawy, Daniel Roy, Ludovic Schwartz, Alex Shtoff, Yanze Song, Luca Viano, Guanghui Wang, Yulian Wu, Jiujia Zhang, Peng Zhao, and Xingyu Zhou.

This material is based upon work that was supported by the National Science Foundation under grant no. 1925930 "Collaborative Research: TRIPODS Institute for Optimization and Learning".

A note on citations: it is customary in the computer science literature to only cite the journal version of a result that first appeared in a conference. The rationale is that the conference version is only a preliminary version, while the journal one is often more complete and sometimes more correct. In these notes, I will not use this custom. Instead, in the presence of the conference and journal version of the same paper, I will cite both. The reason is that I want to clearly delineate the history of the ideas, their first inventors, and the unavoidable rediscoveries. Hence, I need the exact year when some ideas were first proposed. Moreover, in some rare cases the authors changed from the conference to the journal version, so citing only the latter would erase the contribution of some key people from the history of Science.

# Chapter 1

# What is Online Learning?

Consider the following repeated game:
In each round $t = 1, \dots, T$
‚¬¢ An adversary chooses a real number $y_t \in [0, 1]$ and keeps it secret;
‚¬¢ You try to guess the real number, choosing $x_t \in [0, 1]$;
‚¬¢ The adversary's number is revealed and you pay the squared difference $(x_t - y_t)^2$.

Basically, we want to guess a sequence of numbers as precisely as possible. To make it a game, we must now define a "winning condition". Let's see what makes sense to consider as a winning condition.

First, let's make the game easier for the player. Let's assume that the adversary is drawing the numbers i.i.d. from some fixed distribution over $[0, 1]$. However, he is still free to decide which distribution at the beginning of the game. If we knew the distribution, we could just predict each round the mean of the distribution and in expectation we would pay $\sigma^2 T$, where $\sigma^2$ is the variance of the distribution. We cannot do better than that! However, given that we do not know the distribution, it is natural to benchmark our strategy with respect to the optimal one. That is, it is natural to measure the quantity
$$\mathbb{E}_Y \left[ \sum_{t=1}^T (x_t - Y)^2 \right] - \sigma^2 T, \quad (1.1)$$
or equivalently considering the average
$$\frac{1}{T} \mathbb{E}_Y \left[ \sum_{t=1}^T (x_t - Y)^2 \right] - \sigma^2. \quad (1.2)$$

Clearly these quantities are positive and they seem to be a good measure, because they are somehow normalized with respect to the "difficulty" of the numbers generated by the adversary, through the variance of the distribution. This is not the only possible measure of our "success", but it is certainly a reasonable one. It would make sense to consider a strategy "successful" if the difference in (1.1) grows sublinearly over time and, equivalently, if the difference in (1.2) goes to zero as the number of rounds $T$ goes to infinity. That is, on average on the number of rounds, we would like our algorithm to be able to approach the optimal performance.

Minimizing Regret. Given that we have converged to what it seems a good measure of success of the algorithm, let's now rewrite (1.1) in an equivalent way
$$\mathbb{E} \left[ \sum_{t=1}^T (x_t - Y)^2 \right] - \min_{x \in [0, 1]} \mathbb{E} \left[ \sum_{t=1}^T (x - Y)^2 \right].$$

Now, the last step: let's remove the assumption on how the data is generated, consider any arbitrary sequence of $y_t$, and let's keep using the same measure of success. Of course, we can remove the expectation because there is no stochasticity anymore. So, we get that we will win the game if
$$\text{Regret}_T := \sum_{t=1}^T (x_t - y_t)^2 - \min_{x \in [0, 1]} \sum_{t=1}^T (x - y_t)^2$$
grows sublinearly with $T$. The quantity above is called the Regret, because it measures how much the algorithm regrets for not sticking on all the rounds to the optimal choice in hindsight. We will denote it by $\text{Regret}_T$. When the regret is sublinear, we will say that the algorithm is no-regret.

Our reasoning should provide sufficient justification for this metric, however in the following we will see that this also makes sense from both a convex optimization and machine learning point of view.

Note that in the stochastic case the optimal strategy is given by a single best prediction, so it was natural to compare against it. Instead, with arbitrary sequences it is not clear anymore that this is a good competitor. For example, we might consider a sequence of competitors instead of a single one. Indeed, it can be done, but the single competitor is still interesting in a variety of settings and simpler to explain. So, for most of this book we will use a single competitor, while we will consider sequence of competitors in Chapter 15.

Let's now generalize the online game a bit more, considering that the algorithm outputs a vector in $x_t$ in the valid set $\mathcal{V} \subseteq \mathbb{R}^d$ and it pays a loss $\ell_t : \mathcal{V} \to \mathbb{R}$ that measures how good was the prediction of the algorithm in each round. The set $\mathcal{V}$ is called the feasible set. Also, let's consider an arbitrary predictor $\mathbf{u}$ in$^1$ $\mathcal{V} \subseteq \mathbb{R}^d$ and let's parameterize the regret with respect to it: $\text{Regret}_T(\mathbf{u})$. So, to summarize, Online Learning is nothing else than designing and analyzing algorithms to minimize the Regret over a sequence of loss functions with respect to an arbitrary competitor $\mathbf{u} \in \mathcal{V} \subseteq \mathbb{R}^d$:
$$\text{Regret}_T(\mathbf{u}) := \sum_{t=1}^T \ell_t(x_t) - \sum_{t=1}^T \ell_t(\mathbf{u}).$$
It is worth stressing that an online algorithm does not know $\mathbf{u}$ nor the value of the corresponding regret in order to guarantee an upper bound to the regret.

Remark 1.1. Strictly speaking, the regret is also a function of the losses $\ell_1, \dots, \ell_T$. However, we will suppress this dependency for simplicity of notation.

This framework is pretty powerful, and it allows to reformulate a bunch of different problems in machine learning and optimization as similar games. More in general, with the regret framework we can analyze situations in which the data are not independent and identically distributed from a distribution, yet I would like to guarantee that the algorithm is "learning" something. For example, online learning can be used to analyze
‚¬¢ Prediction of clicks on banners on webpages;
‚¬¢ Routing on a network;
‚¬¢ Convergence to equilibrium of repeated games.

It can also be used to analyze stochastic algorithms, e.g., Stochastic Gradient Descent, but the adversarial nature of the analysis might give you suboptimal results. For example, it can be used to analyze momentum algorithms, but a naive treatment of the adversarial losses might give a convergence guarantee that treats the momentum term as a vanishing disturbance that does not help the algorithm in any way.

Let's now go back to our number guessing game and let's try a strategy to win it. Of course, this is one of the simplest example of online learning, without a real application. Yet, going through it we will uncover most of the key ingredients in online learning algorithms and their analysis.

$^1$ In some cases, we can make the game easier for the algorithm letting it choose the prediction from a set $\mathcal{W} \supset \mathcal{V}$.

A Winning Strategy. Can we win the number guessing game? Note that we did not assume anything on how the adversary is deciding the numbers. In fact, the numbers can be generated in any way, even in an adaptive way based on our strategy. Indeed, they can be chosen adversarially, that is explicitly trying to make us lose the game. This is why we call the mechanism generating the numbers the adversary.

The fact that the numbers are adversarially chosen means that we can immediately rule out any strategy based on any statistical modeling of the data. In fact, it cannot work because the moment we estimate something and act on our estimate, the adversary can immediately change the way he is generating the data, ruining us. So, we have to think about something else. Surprisingly enough, many times online learning algorithms will look like classic ones from statistical estimation, even if they work for different reasons.

Now, let's try to design a strategy to make the regret provably sublinear in time, regardless of how the adversary chooses the numbers. The first thing to do is to take a look at the best strategy in hindsight, that is the argmin of the second term of the regret. It should be immediate to see that
$$x^\star_T := \text{argmin}_{x \in [0, 1]} \sum_{t=1}^T (x - y_t)^2 = \frac{1}{T} \sum_{t=1}^T y_t.$$

Now, given that we do not know the future, for sure we cannot use $x^\star_T$ as our guess in each round. However, we do know the past, so a reasonable strategy in each round could be to output the best number over the past. Why such strategy would work? For sure, the reason why it could work is not because we expect the future to be like the past, because it is not true! Instead, we want to leverage the fact that the optimal guess over time cannot change too much between rounds, so we can try to "track" it over time.

Hence, on each round $t$ our strategy is to guess $x_t = x^\star_{t-1} = \frac{1}{t-1} \sum_{i=1}^{t-1} y_i$. Such strategy is usually called Follow-the-Leader (FTL), because you are following what would have been the optimal thing to do on the past rounds (i.e., the Leader).

Let's now try to show that indeed this strategy will allow us to win the game. Given that this is a simple example, we will prove its regret guarantee using first principles. In the following, we will introduce and use very general proof methods. First, we will need a small lemma.

Lemma 1.2. Let $\mathcal{V} \subseteq \mathbb{R}^d$ and $\ell_t : \mathcal{V} \to \mathbb{R}$ an arbitrary sequence of loss functions. Assume the existence of $x^\star_t \in \text{argmin}_{x \in \mathcal{V}} \sum_{i=1}^t \ell_i(x)$, a minimizer in $\mathcal{V}$ of the cumulative loss over the previous $t$ rounds. Then, we have
$$\sum_{t=1}^T \ell_t(x^\star_t) \leq \sum_{t=1}^T \ell_t(x^\star_T).$$

Proof. We prove it by induction over $T$. The base case is
$$\ell_1(x^\star_1) \leq \ell_1(x^\star_1),$$
that is trivially true. Now, for $T \geq 2$, we assume that $\sum_{t=1}^{T-1} \ell_t(x^\star_t) \leq \sum_{t=1}^{T-1} \ell_t(x^\star_{T-1})$ is true and we must prove the stated inequality, that is
$$\sum_{t=1}^T \ell_t(x^\star_t) \leq \sum_{t=1}^T \ell_t(x^\star_T).$$

This inequality is equivalent to
$$\sum_{t=1}^{T-1} \ell_t(x^\star_t) \leq \sum_{t=1}^{T-1} \ell_t(x^\star_T), \quad (1.3)$$
where we removed the last element of the sums because they are the same. Now observe that
$$\sum_{t=1}^{T-1} \ell_t(x^\star_t) \leq \sum_{t=1}^{T-1} \ell_t(x^\star_{T-1}),$$

by induction hypothesis, and
$$\sum_{t=1}^{T-1} \ell_t(x^\star_{T-1}) \leq \sum_{t=1}^{T-1} \ell_t(x^\star_T)$$
because $x^\star_{T-1}$ is a minimizer of the left hand side in $\mathcal{V}$ and $x^\star_T \in \mathcal{V}$. Chaining these two inequalities, we have that (1.3) is true, and so the theorem is proven.

Basically, the above lemma quantifies the idea that knowing the future and being adaptive to it is typically better than not being adaptive to it.

With this lemma, we can now prove that the regret will grow sublinearly, in particular it will be at most logarithmic in time. Note that we will not prove that our strategy is minimax optimal, even if it is possible to show that the logarithmic dependency on time is unavoidable for this problem.

Theorem 1.3. Let $y_t \in [0, 1]$ for $t = 1, \dots, T$ an arbitrary sequence of numbers. Let the algorithm's output $x_t = x^\star_{t-1} := \frac{1}{t-1} \sum_{i=1}^{t-1} y_i$, where we define $x^\star_0 = 0.5$. Then, we have
$$\text{Regret}_T = \sum_{t=1}^T (x_t - y_t)^2 - \min_{x \in [0,1]} \sum_{t=1}^T (x - y_t)^2 \leq 4 + 4 \ln T.$$

Proof. We use Lemma 1.2 to upper bound the regret:
$$\sum_{t=1}^T (x_t - y_t)^2 - \min_{x \in [0,1]} \sum_{t=1}^T (x - y_t)^2 = \sum_{t=1}^T (x^\star_{t-1} - y_t)^2 - \sum_{t=1}^T (x^\star_T - y_t)^2 \leq \sum_{t=1}^T (x^\star_{t-1} - y_t)^2 - \sum_{t=1}^T (x^\star_t - y_t)^2.$$

Now, let's take a look at each difference in the sum in the last equation. For $t = 1$, we have
$$(x^\star_{t-1} - y_t)^2 - (x^\star_t - y_t)^2 \leq 0.5^2.$$

For $t \geq 2$, we have that
$$(x^\star_{t-1} - y_t)^2 - (x^\star_t - y_t)^2 = (x^\star_{t-1})^2 - 2y_tx^\star_{t-1} - (x^\star_t)^2 + 2y_tx^\star_t = (x^\star_{t-1} + x^\star_t - 2y_t)(x^\star_{t-1} - x^\star_t) \leq |x^\star_{t-1} + x^\star_t - 2y_t| |x^\star_{t-1} - x^\star_t| \leq 2|x^\star_{t-1} - x^\star_t| = 2 \left| \frac{1}{t-1} \sum_{i=1}^{t-1} y_i - \frac{1}{t} \sum_{i=1}^{t} y_i \right| = 2 \left| \left( \frac{1}{t-1} - \frac{1}{t} \right) \sum_{i=1}^{t-1} y_i - \frac{y_t}{t} \right| \leq 2 \left| \frac{1}{t(t-1)} \sum_{i=1}^{t-1} y_i \right| + \frac{2|y_t|}{t} \leq \frac{2}{t} + \frac{2|y_t|}{t} \leq \frac{4}{t}.$$

Hence, overall we have
$$\sum_{t=1}^T (x_t - y_t)^2 - \min_{x \in [0,1]} \sum_{t=1}^T (x - y_t)^2 \leq 0.25 + 4 \sum_{t=2}^T \frac{1}{t} \leq 4 \sum_{t=1}^T \frac{1}{t}.$$

To upper bound the last sum, observe that we are trying to find an upper bound to the green area in Figure 1.1. As you can see from the picture, it can be upper bounded by 1 plus the integral of $\frac{1}{t-1}$ from 2 to $T + 1$. So, we have
$$\sum_{t=1}^T \frac{1}{t} \leq 1 + \int_{2}^{T+1} \frac{1}{t-1} dt = 1 + \ln T.$$

Let's write in words the steps of the proof: Lemma 1.2 allows us to upper bound the regret against the single best guess with the regret against the competitor sequence $x^\star_1, \dots, x^\star_T$. In turn, given that we produce in each round the prediction $x^\star_{t-1}$ and $|x^\star_t - x^\star_{t-1}|$ goes to zero very fast, the total regret is sublinear in time.

There are a few things to stress on this strategy. The strategy does not have parameters to tune (e.g., learning rates, regularizers). Note that the presence of parameters does not make sense in online learning: We have only one stream of data and we cannot run our algorithm over it multiple times to select the best parameter! Also, this strategy does not need to maintain a complete record of the past, but only a "summary" of it, through the running average. This gives a computationally efficient algorithm. When we design online learning algorithms, we will strive to achieve all these characteristics. The final point to stress is that the algorithm does not use gradients: Gradients are useful and we will use them a lot, but they do not constitute the entire world of online learning.

Before going on I want to remind you that, as seen above, this is different from the classic setting in statistical machine learning. So, for example, "overfitting" has absolutely no meaning here. Same for "generalization gap" and similar ideas linked to a training/testing scenario.

In the next chapters, we will introduce several algorithms for online optimization and one of them will be a strict generalization of the strategy we used in the example above.

### 1.1 History Bits

The concept of "regret" seems to have been proposed by Savage [1951], an exposition and review of the book by Wald [1950] on a foundation of statistical decision problems based on zero-sum two-person games. Savage [1951] introduces the idea of considering the difference between the utility of the best action in a given state and the utility incurred by any action under the same state. The proposed optimal strategy was the one minimizing such regret over the worst possible state. Savage [1951] called this concept "loss" and did not like the word "regret" because "that term seems to me charged with emotion and liable to lead to such misinterpretation as that the loss necessarily becomes known" [Savage, 1954, page 163]. The name of "regret" instead seems to have suggested in Milnor [1951].

However, Savage's definition is a modification of the one proposed by Wald [1950], who instead proposed to maximize just the utility, under the assumption that the utility of the best action for any state is 0. While minimizing the regret or the minimizing the negative utility under the assumption of Wald [1950] are mathematically equivalent, Savage [pages 169‚¬€œ170 1954] explains that Wald considered the regret formulation different from what he proposed, while Savage attributed to his idea "little or no originality".

Extending the definition of Savage [1951, 1954] to a sequence of games, Hannan [1957] designed a randomized algorithm for zero-sum repeated games with fixed loss matrix with a vanishing expected average regret. Hence, the concept of regret seems to originate from game theory, but strangely enough passing through the ideas of two mathematical statisticians.

Lemma 1.2 is due to Hannan [1957].

### Exercises

Problem 1.1. Extend the previous algorithm and analysis to the case when the adversary selects a vector $y_t \in \mathbb{R}^d$ such that $\|y_t\|_2 \leq 1$, the algorithm guesses a vector $x_t \in \mathbb{R}^d$, and the loss function is $\|x_t - y_t\|_2^2$. Show an upper bound to the regret logarithmic in $T$ and that does not depend on $d$. Among the other things, you will probably need the Cauchy-Schwarz inequality: $\langle x, y \rangle \leq \|x\|_2 \|y\|_2$.

# Chapter 2

# Online Subgradient Descent

In this chapter, we will introduce the Online Subgradient Descent (OSD) algorithm: a generic online algorithm to solve online problems with convex losses. First, we will introduce Online Gradient Descent (OGD) for convex differentiable functions, then we will extend it to non-differentiable functions.

## 2.1 Online Learning with Convex Differentiable Losses

To summarize what we said in the first chapter, let's define an online learning as the following general game
‚¬¢ For $t = 1, \dots, T$
‚¬€œ Outputs $x_t \in \mathcal{V} \subseteq \mathbb{R}^d$
‚¬€œ Pay $\ell_t(x_t)$ for $\ell_t : \mathcal{V} \to \mathbb{R}$
‚¬€œ Receive some feedback on $\ell_t$
‚¬¢ End for
The aim of this game is to minimize the regret with respect to any competitor $\mathbf{u} \in \mathcal{V}$:
$$\text{Regret}_T(\mathbf{u}) := \sum_{t=1}^T \ell_t(x_t) - \sum_{t=1}^T \ell_t(\mathbf{u}).$$

We also said that the way the losses $\ell_t$ are decided is adversarial. Now, without additional assumptions we cannot hope to solve this problem. Hence, we have to understand what are the reasonable assumptions we can make. Typically, we will try to restrict the choice of the loss functions in some way. This is considered reasonable because most of the time we have some say in deciding the set from which the loss functions are picked. So, for example, we will consider only convex loss functions. However, convexity might not be enough, so we might restrict the class a bit more to, for example, Lipschitz convex functions. On the other hand, assuming to know something about the future is not considered a reasonable assumption, because we very rarely have any control on the future. In general, the stronger the assumptions the better will be guarantee on the regret we can prove. The best algorithms we will see will guarantee a sublinear regret against the weakest assumption we can make, guaranteeing at the same time a smaller regret for easy adversaries.

It is also important to remember why minimizing the regret is a good objective: Given that we do not assume anything on how the adversary generates the loss functions, minimizing the regret is a good metric that takes into account the difficulty of the problem. If an online learning algorithm is able to guarantee a sublinear regret, it means that its performance on average will approach the performance of any fixed strategy. As said, we will see that in many situations if the adversary is "weak", for example it is a fixed stochastic distribution over the loss functions, being prepared for the worst-case scenario will not preclude us to get the best guarantee anyway.

For a while, we will focus on the case that $\ell_t$ are convex, and this problem will be called Online Convex Optimization (OCO). Later, we will see how to convexify some specific non-convex online problems.

Remark 2.1. I will now introduce some math concepts. If you have a background in Convex Analysis, this will be easy stuff for you. On the other hand, if you never saw these things before they might look a bit scary. Let me tell you the right way to look at them: these are tools that will make our job easier. Without these tools, it would be basically impossible to design any online learning algorithm. And, no, it is not enough to test random algorithms on some machine learning dataset, because fixed datasets are not adversarial. Without a correct proof, you might not realize that your online algorithm fail on particular sequences of losses, as it happened to Adam [Reddi et al., 2018]. I promise you that once you understand the key mathematical concepts, online learning is actually easy.

### 2.1.1 Convex Analysis Bits: Convexity

Definition 2.2 (Convex Set). $\mathcal{V} \subset \mathbb{R}^d$ is convex if for any $x, y \in \mathcal{V}$ and any $\lambda \in (0, 1)$, we have $\lambda x + (1 - \lambda)y \in \mathcal{V}$.

In words, this means that the set $\mathcal{V}$ has no holes, see Figure 2.1.
We will make use of extended-real-valued functions, that is, functions that take value in $\mathbb{R} \cup \{-\infty, +\infty\}$. For $f$ an extended-real-valued function on $\mathbb{R}^d$, its domain is the set $\text{dom } f = \{x \in \mathbb{R}^d : f(x) < +\infty\}$.

Extended-real-valued functions allow us to easily consider constrained set and are a standard notation in Convex Optimization [see, e.g., Boyd and Vandenberghe, 2004]. For example, if I want the predictions of the algorithm $x_t$ and the competitor $\mathbf{u}$ to be in a set $\mathcal{V} \subset \mathbb{R}^d$, I can just add $i_\mathcal{V}(x)$ to all the losses, where $i_\mathcal{V} : \mathbb{R}^d \to (-\infty, +\infty]$ is the indicator function of the set $\mathcal{V}$ defined as
$$i_\mathcal{V}(x) = \begin{cases} 0, & x \in \mathcal{V}, \\ +\infty, & \text{otherwise}. \end{cases}$$

In this way, the only way for the algorithm and for the competitor to suffer finite loss is to predict inside the set $\mathcal{V}$. Also, extended-real-valued functions will make the use of Fenchel conjugates more direct, see Section 6.4.1.

Convex functions will be an essential ingredient in online learning.

Definition 2.3 (Convex Function). Let $f : \mathbb{R}^d \to [-\infty, +\infty]$. $f$ is convex if the epigraph of the function, $\{(x, y) \in \mathbb{R}^{d+1} : y \geq f(x)\}$, is convex.

We can see a visualization of this definition in Figure 2.2. Note that the definition implies that the domain of a convex function is convex. Also, observe that if $f : \mathcal{V} \subseteq \mathbb{R}^d \to \mathbb{R}$ is convex, $f + i_\mathcal{V} : \mathbb{R}^d \to (-\infty, +\infty]$ is also convex. Note that $i_\mathcal{V}(x)$ is convex iff $\mathcal{V}$ is convex, so each convex set is associated with a convex function.

The definition above gives rise to the following characterization for convex functions that do not assume the value $-\infty$.

Theorem 2.4 ([Rockafellar, 1970, Theorem 4.1]). Let $f : \mathbb{R}^d \to (-\infty, +\infty]$ and $\text{dom } f$ is a convex set. Then $f$ is convex iff, for any $0 < \lambda < 1$, we have
$$f(\lambda x + (1 - \lambda)y) \leq \lambda f(x) + (1 - \lambda)f(y), \quad \forall x, y \in \text{dom } f.$$

Example 2.5. The simplest example of convex functions are the affine functions: $f(x) = \langle z, x \rangle + b$.

Example 2.6. Norms are always convex, the proof is left as exercise.

How to recognize a convex function? In the most general case, you have to rely on its definition. However, most of the time we will recognize them as composed by operations that preserve the convexity. For example:
‚¬¢ If $f$ and $g$ convex, then their linear combination with non-negative weights is also convex.
‚¬¢ The composition with an affine transformation preserves the convexity.
‚¬¢ If $f : \mathbb{R}^d \to \mathbb{R}$ and $g : \mathbb{R} \to \mathbb{R}$ are convex functions and $g$ is non-decreasing, then $h(x) = g(f(x))$ is convex.
‚¬¢ Pointwise supremum of convex functions is convex.
The proofs are left as exercises.

A very important property of differentiable convex functions is that we can construct linear lower bound to the function.

Theorem 2.7 ([Rockafellar, 1970, Theorem 25.1 and Corollary 25.1.1]). Suppose $f : \mathbb{R}^d \to (-\infty, +\infty]$ a convex function and let $x \in \text{int dom } f$. If $f$ is differentiable at $x$ then
$$f(y) \geq f(x) + \langle \nabla f(x), y - x \rangle, \quad \forall y \in \mathbb{R}^d.$$

We will also use the first-order optimality condition for differentiable convex functions:

Theorem 2.8. Let $\mathcal{V}$ be a convex non-empty set, $x^\star \in \mathcal{V}$, and $f$ a convex function, differentiable over an open set that contains $\mathcal{V}$. Then, $x^\star \in \text{argmin}_{x \in \mathcal{V}} f(x)$ iff $\langle \nabla f(x^\star), y - x^\star \rangle \geq 0, \quad \forall y \in \mathcal{V}$.

Proof. Let first assume that $x^\star$ satisfies $\langle \nabla f(x^\star), y - x^\star \rangle \geq 0, \quad \forall y \in \mathcal{V}$. Then, by Theorem 2.7, for any $y \in \mathcal{V}$, we have that
$$f(y) \geq f(x^\star) + \langle \nabla f(x^\star), y - x^\star \rangle \geq f(x^\star),$$
that is $x^\star$ is the minimizer of $f$ over $\mathcal{V}$.

Now, assume that $x^\star$ is the minimizer of $f$ over $\mathcal{V}$ and assume that there exists $y \in \mathcal{V}$ such that $\langle \nabla f(x^\star), y - x^\star \rangle < 0$. Consider $z(\alpha) = \alpha y + (1 - \alpha)x^\star$ where $\alpha \in (0, 1)$. Note that $z(\alpha) \in \mathcal{V}$ and denote by $h(\alpha) = f(z(\alpha))$. We have that $h'(0) = \langle \nabla f(x^\star), y - x^\star \rangle < 0$. Given that $f$ is differentiable and so continuous, there exists $\alpha^\star$ sufficiently small such that $f(z(\alpha^\star)) < f(z(0)) = f(x^\star)$ that contradicts the fact that $x^\star$ is the minimizer over $\mathcal{V}$.

In words, at the constrained minimum, the gradient makes an angle of $90^\circ$ or less with all the feasible variations $y - x^\star$, hence we cannot minimize more the function moving inside $\mathcal{V}$. Moreover, if $x^\star \in \text{int } \mathcal{V}$, by choosing $\epsilon$ small enough such that $y = x^\star - \epsilon \nabla f(x^\star) \in \mathcal{V}$, we obtain that $x^\star$ is a minimum iff $\nabla f(x^\star) = 0$.

Another critical property of convex functions is Jensen's inequality.

Theorem 2.9. Let $f : \mathbb{R}^d \to (-\infty, +\infty]$ be a measurable convex function and $\mathbf{x}$ be an $\mathbb{R}^d$-valued random element on some probability space such that $\mathbb{E}[\mathbf{x}]$ exists and $\mathbf{x} \in \text{dom } f$ with probability 1. Then, we have
$$\mathbb{E}[f(\mathbf{x})] \geq f(\mathbb{E}[\mathbf{x}]).$$

We can now see our first Online Convex Optimization (OCO) algorithm in the case that the functions are convex and differentiable.

### 2.1.2 Online Gradient Descent

In the first chapter, we saw a simple strategy to obtain a logarithmic regret in the guessing game. The strategy was to use the best over the past, that is the Follow-the-Leader strategy. In formulas,
$$x_t = \text{argmin}_x \sum_{i=1}^{t-1} \ell_i(x),$$
and in the first round we can play any admissible point. One might wonder if this strategy always works, but the answer is negative!

Example 2.10 (Failure of Follow-The-Leader (FTL)). Let $\mathcal{V} = [-1, 1]$ and consider the sequence of losses $\ell_t(x) = z_t x$, where
$z_1 = -0.5$,
$z_t = 1, t = 2, 4, \dots$
$z_t = -1, t = 3, 5, \dots$

Then, apart from the first round where the prediction of FTL is arbitrary in $[-1, 1]$, the predictions of FTL will be $x_t = 1$ for $t$ even and $x_t = -1$ for $t$ odd. The cumulative loss of the FTL algorithm after $T$ rounds will therefore be $T - 1 - \frac{x_1}{2}$ while the cumulative loss of the fixed solution $u = 0$ is 0. Thus, the regret of FTL with respect to $u = 0$ is $T - 1 - \frac{x_1}{2} \geq T - \frac{3}{2}$.

Hence, we will see an alternative strategy that guarantees sublinear regret for convex Lipschitz functions. Later, we will also prove that the dependency on $T$ is optimal. The strategy is called Projected Online Gradient Descent, or just Online Gradient Descent (OGD), see Algorithm 2.1. It consists in updating the prediction of the algorithm at each time step moving in the negative direction of the gradient of the loss received and projecting back onto the feasible set. Some might see that this algorithm is similar to Stochastic Gradient Descent, but it is not the same thing: here the loss functions are different at each step and they are not drawn from a fixed distribution but adversarially chosen. We will later see that Online Gradient Descent can also be used as Stochastic Gradient Descent.

Algorithm 2.1 Projected Online Gradient Descent
Require: Non-empty closed convex set $\mathcal{V} \subseteq \mathbb{R}^d$, $x_1 \in \mathcal{V}$, $\eta_1, \dots, \eta_T > 0$
1: for $t = 1$ to $T$ do
2: Output $x_t \in \mathcal{V}$
3: Pay $\ell_t(x_t)$ for $\ell_t : \mathcal{V} \to \mathbb{R}$ differentiable in an open set containing $\mathcal{V}$
4: Set $g_t = \nabla \ell_t(x_t)$
5: $x_{t+1} = \Pi_\mathcal{V} (x_t - \eta_t g_t) = \text{argmin}_{y \in \mathcal{V}} \|x_t - \eta_t g_t - y\|_2$
6: end for

First, we show the following two Lemmas. The first lemma proves that Euclidean projections always decrease the distance with points inside the set.

Proposition 2.11. Let $x \in \mathbb{R}^d$ and $y \in \mathcal{V}$, where $\mathcal{V} \subseteq \mathbb{R}^d$ is a non-empty closed convex set and define $\Pi_\mathcal{V}(x) := \text{argmin}_{y \in \mathcal{V}} \|x - y\|_2$. Then, $\|\Pi_\mathcal{V}(x) - y\|_2 \leq \|x - y\|_2$.

Proof. First of all, observe that $\text{argmin}_{y \in \mathcal{V}} \|x - y\|_2 = \text{argmin}_{y \in \mathcal{V}} \frac{1}{2} \|x - y\|_2^2$. So, from the optimality condition of Theorem 2.8 on the function $f(y) = \frac{1}{2} \|x - y\|_2^2$, we obtain
$$\langle \Pi_\mathcal{V}(x) - x, y - \Pi_\mathcal{V}(x) \rangle \geq 0.$$
Therefore,
$$\|y - x\|_2^2 = \|y - \Pi_\mathcal{V}(x) + \Pi_\mathcal{V}(x) - x\|_2^2 = \|y - \Pi_\mathcal{V}(x)\|_2^2 + 2\langle y - \Pi_\mathcal{V}(x), \Pi_\mathcal{V}(x) - x \rangle + \|\Pi_\mathcal{V}(x) - x\|_2^2 \geq \|y - \Pi_\mathcal{V}(x)\|_2^2.$$

The next lemma upper bounds the regret in one iteration of the algorithm.

Lemma 2.12. Let $\mathcal{V} \subseteq \mathbb{R}^d$ a non-empty closed convex set and $\ell_t : \mathcal{V} \to \mathbb{R}$ a convex function differentiable in an open set that contains $\mathcal{V}$. Set $g_t = \nabla \ell_t(x_t)$. Then, $\forall \mathbf{u} \in \mathcal{V}$, the following inequality holds
$$\eta_t(\ell_t(x_t) - \ell_t(\mathbf{u})) \leq \eta_t \langle g_t, x_t - \mathbf{u} \rangle \leq \frac{1}{2} \|x_t - \mathbf{u}\|_2^2 - \frac{1}{2} \|x_{t+1} - \mathbf{u}\|_2^2 + \frac{\eta_t^2}{2} \|g_t\|_2^2.$$

Proof. From Proposition 2.11 and Theorem 2.7, we have that
$$\|x_{t+1} - \mathbf{u}\|_2^2 - \|x_t - \mathbf{u}\|_2^2 \leq \|x_t - \eta_t g_t - \mathbf{u}\|_2^2 - \|x_t - \mathbf{u}\|_2^2 = -2\eta_t \langle g_t, x_t - \mathbf{u} \rangle + \eta_t^2 \|g_t\|_2^2 \leq -2\eta_t(\ell_t(x_t) - \ell_t(\mathbf{u})) + \eta_t^2 \|g_t\|_2^2.$$
Reordering, we have the stated bound.

We can prove the following regret guarantee.

Theorem 2.13. Let $\mathcal{V} \subseteq \mathbb{R}^d$ a non-empty closed convex set with diameter $D$, i.e., $\max_{\mathbf{x}, \mathbf{y} \in \mathcal{V}} \|\mathbf{x} - \mathbf{y}\|_2 \leq D$. Let $\ell_1, \dots, \ell_T$ an arbitrary sequence of convex functions $\ell_t : \mathcal{V} \to \mathbb{R}$ differentiable in open sets containing $\mathcal{V}$. Pick any $x_1 \in \mathcal{V}$ and assume $\eta_{t+1} \leq \eta_t$, $t = 1, \dots, T$. Then, $\forall \mathbf{u} \in \mathcal{V}$, the following regret bound holds
$$\sum_{t=1}^T (\ell_t(x_t) - \ell_t(\mathbf{u})) \leq \frac{D^2}{2\eta_T} + \sum_{t=1}^T \frac{\eta_t}{2} \|g_t\|_2^2 - \frac{1}{2\eta_T} \|x_{T+1} - \mathbf{u}\|_2^2.$$

Moreover, if $\eta_t$ is constant, i.e., $\eta_t = \eta$ $\forall t = 1, \dots, T$, we have
$$\sum_{t=1}^T (\ell_t(x_t) - \ell_t(\mathbf{u})) \leq \frac{\|\mathbf{u} - x_1\|_2^2}{2\eta} + \frac{\eta}{2} \sum_{t=1}^T \|g_t\|_2^2 - \frac{1}{2\eta} \|x_{T+1} - \mathbf{u}\|_2^2.$$

Proof. Dividing the inequality in Lemma 2.12 by $\eta_t$ and summing over $t = 1, \dots, T$, we have
$$\sum_{t=1}^T (\ell_t(x_t) - \ell_t(\mathbf{u})) \leq \sum_{t=1}^T \left( \frac{1}{2\eta_t} \|x_t - \mathbf{u}\|_2^2 - \frac{1}{2\eta_t} \|x_{t+1} - \mathbf{u}\|_2^2 \right) + \sum_{t=1}^T \frac{\eta_t}{2} \|g_t\|_2^2 = \frac{1}{2\eta_1} \|x_1 - \mathbf{u}\|_2^2 - \frac{1}{2\eta_T} \|x_{T+1} - \mathbf{u}\|_2^2 + \sum_{t=1}^{T-1} \left( \frac{1}{2\eta_{t+1}} - \frac{1}{2\eta_t} \right) \|x_{t+1} - \mathbf{u}\|_2^2 + \sum_{t=1}^T \frac{\eta_t}{2} \|g_t\|_2^2 \leq \frac{1}{2\eta_1} D^2 + D^2 \sum_{t=1}^{T-1} \left( \frac{1}{2\eta_{t+1}} - \frac{1}{2\eta_t} \right) + \sum_{t=1}^T \frac{\eta_t}{2} \|g_t\|_2^2 - \frac{1}{2\eta_T} \|x_{T+1} - \mathbf{u}\|_2^2 = \frac{1}{2\eta_1} D^2 + D^2 \left( \frac{1}{2\eta_T} - \frac{1}{2\eta_1} \right) + \sum_{t=1}^T \frac{\eta_t}{2} \|g_t\|_2^2 - \frac{1}{2\eta_T} \|x_{T+1} - \mathbf{u}\|_2^2 = \frac{D^2}{2\eta_T} + \sum_{t=1}^T \frac{\eta_t}{2} \|g_t\|_2^2 - \frac{1}{2\eta_T} \|x_{T+1} - \mathbf{u}\|_2^2.$$

In the same way, when the $\eta_t$ is constant, we have
$$\sum_{t=1}^T (\ell_t(x_t) - \ell_t(\mathbf{u})) \leq \sum_{t=1}^T \left( \frac{1}{2\eta} \|x_t - \mathbf{u}\|_2^2 - \frac{1}{2\eta} \|x_{t+1} - \mathbf{u}\|_2^2 \right) + \frac{\eta}{2} \sum_{t=1}^T \|g_t\|_2^2 = \frac{1}{2\eta} \|x_1 - \mathbf{u}\|_2^2 - \frac{1}{2\eta} \|x_{T+1} - \mathbf{u}\|_2^2 + \frac{\eta}{2} \sum_{t=1}^T \|g_t\|_2^2.$$

We can immediately observe a few things.
‚¬¢ The terms $\|g_t\|_2^2$ are due to the fact that we output $x_t$ before knowing the loss $\ell_t$. Indeed, these terms become non-positive if we were allowed to observe $\ell_t$ before producing $x_t$, see Section 12.5.1.
‚¬¢ In the case of constant $\eta$, the first term in the regret depends on our initial condition, that is how far is $x_1$ from the competitor $\mathbf{u}$.
‚¬¢ The last term in both bounds is non-positive, so it is usually discarded. Yet, some advanced proofs require it, so it is good to be aware of it.
‚¬¢ If we want to use time-varying learning rates, you need a bounded domain $\mathcal{V}$ for the proof to work. However, this assumption is false in most of the machine learning applications. However, in the stochastic setting you can still use a time-varying learning rate in SGD with an unbounded domain if you use a non-uniform averaging. We will see this in Chapter 3.
‚¬¢ Another important observation is that the regret bound helps us choosing the learning rates $\eta_t$. Indeed, it is the only guideline we have. Any other choice that is not justified by a regret analysis it is not justified at all.

As we said, the presence of parameters like the learning rates make no sense in online learning. So, we have to decide a strategy to set them. A simple choice is to find the constant learning rate that minimizes the bounds for a fixed number of iterations. We have to consider the expression
$$\frac{\|\mathbf{u} - x_1\|_2^2}{2\eta} + \frac{\eta}{2} \sum_{t=1}^T \|g_t\|_2^2$$
and minimize with respect to $\eta$. It is easy to see that the optimal $\eta$ is $\frac{\|\mathbf{u} - x_1\|_2}{\sqrt{\sum_{t=1}^T \|g_t\|_2^2}}$, that would give the regret bound
$$\|\mathbf{u} - x_1\|_2 \sqrt{\sum_{t=1}^T \|g_t\|_2^2}.$$

However, we have a problem: in order to use this stepsize, we should know all the future gradients and the distance between the optimal solution and the initial point! This is clearly impossible: Remember that the adversary can choose the sequence of functions. Hence, it can observe your choice of learning rates and decide the sequence so that your learning rate is now the wrong one!

Indeed, it turns out that this kind of rate is completely impossible because it is ruled out by a lower bound. Yet, we will see that it is indeed possible to achieve very similar rates using adaptive (Section 4.2) and parameter-free algorithms (Chapter 10). For the moment, we can observe that we might be happy to minimize a loose upper bound. In particular, assume that the norm of the gradients is bounded by $L$, that is $\|g_t\|_2 \leq L$. Also, assuming a bounded diameter, we can upper bound $\|\mathbf{u} - x_1\|_2$ by $D$. Hence, we have
$$\eta^\star = \text{argmin}_\eta \frac{D^2}{2\eta} + \frac{\eta L^2 T}{2} = \frac{D}{L \sqrt{T}},$$
that gives a regret bound of
$$DL \sqrt{T}. \quad (2.1)$$

So, indeed the regret is sublinear in time.

Remark 2.14. Sometimes people justify the Lipschitz assumption by arguing that a convex function is Lipschitz on any bounded subset of its domain, but in reality this is false. For example, the convex loss $f(x) = -\sqrt{x}$ defined in $[0, \infty)$ is not Lipschitz on any bounded set $[0, a]$, where $a > 0$. The actual result is more subtle: for example, one can show that any convex function is Lipschitz on closed bounded sets that are contained in the domain of the gradient. However, one has to be careful even with this statement. Again, the convex loss $f(x) = -\sqrt{x}$ has Lipschitz constant equal to $\frac{1}{2\sqrt{\epsilon}}$ on any interval $[\epsilon, a]$, where $0 < \epsilon < a$. Hence, the Lipschitz constant can be made arbitrarily large by minimal changes of the feasible set.

Example 2.15. Consider the guessing game of the first chapter, we can solve easily it with OGD. Indeed, we just need to calculate the gradients, prove that they are bounded, and find a way to calculate the projection of a real number in $[0, 1]$ So, $\ell'_t(x) = 2(x - y_t)$, that is bounded for $x, y_t \in [0, 1]$. The projection on $[0, 1]$ is just $\Pi_{[0,1]}(x) = \min(\max(x, 0), 1)$. With the optimal learning rate, the resulting regret would be $\mathcal{O}(\sqrt{T})$, that is worse than the one we found in the first chapter.

Example 2.16. Let's consider an example of the online convex optimization setting. Consider the problem of predicting at day $t$ the opening price of a stock based on a linear combination of the opening prices of the past $d$ days, represented by a vector $z_t \in \mathbb{R}^d$. So, our prediction at time $t$ will be $\langle z_t, x_t \rangle$ and the feasible set is $\mathcal{V} = \mathbb{R}^d$. Once we make our prediction, we receive the true opening price $y_t$ and we pay the Huber loss, a convex loss function robust to outliers, on the difference between our prediction and the opening price:
$$\ell_t(x) = \begin{cases} \frac{1}{2}(\langle z_t, x \rangle - y_t)^2, & \text{for } |\langle z_t, x \rangle - y_t| \leq δ, \\ δ(|\langle z_t, x \rangle - y_t| - \frac{1}{2}δ), & \text{otherwise}. \end{cases}$$

We can use OGD for this problem, we just need to calculate the gradients:
$$\nabla \ell_t(x) = \begin{cases} (\langle z_t, x \rangle - y_t)z_t, & \text{for } |\langle z_t, x \rangle - y_t| \leq δ, \\ δ \text{sign}(\langle z_t, x \rangle - y_t)z_t, & \text{otherwise}. \end{cases}$$

Assuming that $|y_t|$ and $\|z_t\|$ are bounded, we also have that these losses are Lipschitz, satisfying the assumptions of Theorem 2.13. Hence, I can run OGD and obtain that on average the performance of the algorithm will approach the performance of the a posteriori best linear predictor.

In the Section 2.2, we will see how to remove the differentiability assumption through the use of subgradients.

### 2.1.3 Unit Analysis Bits

One might wonder why I did not "simplify the math" by removing some of the constants in the bounds, for example, by setting the Lipschitz $L$ to 1. While this is a very common in machine learning papers, this is a bad idea because it makes i) difficult to check the correctness of an equation, and ii) difficult to correctly implement the algorithm. To explain why this is the case, we will have to consider the concepts of "units".

When we look at a mathematical formula coming from a physical model, we are used to the idea that each quantity has a "unit" of measurement, e.g., meters, seconds, Joules. A fundamental principle is that any physically meaningful equation must be dimensionally consistent. That is, the units on the left-hand side of an equation must be the same as the units on the right-hand side. For example, you cannot sum a quantity measured in meters with one measured in seconds.

It turns out that this simple idea is not limited to physical models and it can be used as a sanity check for any mathematical formula, including the ones we see in online learning. Let's assign a symbolic unit to each quantity we are dealing with. For example, let's denote the units of our predictions $x$ by $[x]$ and the units of the loss $\ell$ by $[\ell]$. We can now check each formula by following a few simple rules. First, we have to make sure that each term has the same units. Another rule to keep in mind is that trascendental functions are only defined on unitless quantities. So, for example, we can take logarithms and exponentials of ratio of quantities with same units, to obtain a unitless quantity, but not of quantities with units. Finally, while probabilities are unitless, the probability density function has the units of inverse of the random variable it represents, because it represents the probability per unit of the variable it is defined over.

The importance of this check cannot be overstated. When we see an equation where the units do not match either it is plainly wrong or there is a constant with units hidden somewhere. Moreover, the dimensional analysis will immediately tell us the dependency of a quantity from the others. Let's do a practical example considering OGD.

Consider the OGD update in Algorithm 2.1:
$$x_{t+1} = \Pi_\mathcal{V} (x_t - \eta_t g_t).$$

For this equation to be coherent, the term $\eta_t g_t$ must have the same units as $x_t$. Let's see if this gives us some constraints on the units of the learning rate $\eta_t$. The units of the gradient $g_t = \nabla \ell_t(x_t)$ are the units of the loss divided by the units of the variable we are differentiating with respect to. So, we have
$$[g_t] = \frac{[\ell]}{[x]}.$$

Now, from the consistency of the update rule, we must have $[x_t] = [\eta_t g_t] = [\eta_t][g_t]$. Hence, we can derive the units of the learning rate:
$$[\eta_t] = \frac{[x]}{[g_t]} = \frac{[x]}{[\ell]/[x]} = \frac{[x]^2}{[\ell]}.$$

This shows that the learning rate is not a unitless quantity. So, even without deriving the regret guarantee, from the unit analysis we immediately know that the learning rate must depend on the problem characteristics in a certain way.

Finally, let's perform a sanity check on the regret bound for OGD with a constant learning rate from Theorem 2.13:
$$\sum_{t=1}^T (\ell_t(x_t) - \ell_t(\mathbf{u})) \leq \frac{\|\mathbf{u} - x_1\|_2^2}{2\eta} + \frac{\eta}{2} \sum_{t=1}^T \|g_t\|_2^2.$$

The left-hand side is a sum of losses, so its unit is $[\ell]$. Let's check the two terms on the right-hand side. For the first term, we have
$$\left[ \frac{\|\mathbf{u} - x_1\|_2^2}{\eta} \right] = \frac{[x]^2}{[\eta]} = \frac{[x]^2}{[x]^2/[\ell]} = [\ell].$$

For the second term, we have
$$\left[ \eta \sum_{t=1}^T \|g_t\|_2^2 \right] = [\eta][g]^2 = \frac{[x]^2}{[\ell]} \frac{[\ell]^2}{[x]^2} = [\ell].$$

Both terms have units of loss, so they can be summed together, and the overall equation has consistent units. This gives us confidence in the correctness of our theorem. This kind of analysis is a useful tool to keep in mind when designing and analyzing online learning algorithms.

Let me stress that this is not only a theoretical issue but a very practical one. Indeed, constants with hidden units in any algorithm can be problematic. Let's show it with an example. Suppose to use a learning rate schedule such as $\eta_t = 1/\sqrt{t}$, which is often suggested in machine learning practice. From the above reasoning, we know that constant "1" in the numerator must have units. This implies that if we change the units of measure of our problem, that constant must change as well to keep the behaviour of the algorithm the same. Let's consider a practical example to show this issue.

Suppose our variable $x_t$ represents a location in meters, and we use a learning rate schedule such as $\eta_t = 1/\sqrt{t}$. Now, suppose we change our unit of measurement for $x_t$ from meters to kilometers. Our new variable is $x'_t = x_t/1000$. Since this is merely a change of units, the underlying learning problem is identical, and we should expect the algorithm's behavior to be invariant. However, the gradient with respect to the new variable is $g'_t = \nabla_{x'} \ell_t(1000x'_t) = 1000\nabla_x \ell_t(x_t) = 1000g_t$. If we still use the dimensionless learning rate $\eta_t = 1/\sqrt{t}$, the update in the new coordinate system becomes $x'_{t+1} = x'_t - \eta_t g'_t = x_t/1000 - (1/\sqrt{t})(1000g_t)$. Converting back to meters by multiplying by 1000, we get $x_{t+1} = x_t - (1000^2/\sqrt{t})g_t$. The effective learning rate has been scaled by a factor of one million, which will drastically change the algorithm's path, all because of a simple change in units. Instead, by stating a learning rate with the correct units, we would have immediately seen how to rescale it.

This failure occurs because of the constant with hidden units in the learning rate, that hides the fact that we should change the learning rate when we change the units of $x_t$. In fact, the correct learning rate $\eta_t$ has units of $[x]^2/[\ell]$ and should have been scaled by a factor of $(1/1000)^2 = 1/1000000$ when we switched the unites of $x_t$ from meters to kilometers.

The fact that common learning rate schedules are not scale-invariant is a significant practical issue. This motivates the need for scale-free algorithms, such as AdaGrad which will be discussed in Section 4.2.4, which automatically adjust to the scale of the gradients and are therefore robust to such changes in units.

## 2.2 Online Subgradient Descent

In the previous section, we have introduced Projected Online Gradient Descent. However, the differentiability assumption for the $\ell_t$ is quite strong. What happens when the losses are convex but not differentiable? For example $\ell_t(x) = |x - 10|$. Note that this situation is more common than one would think. For example, the hinge loss, $\ell_t(w) = \max(1 - y\langle w, x \rangle, 0)$, and the ReLU activation function used in neural networks, $\ell_t(x) = \max(x, 0)$, are not differentiable. It turns out that we can just use OGD, substituting subgradients for the gradients. For this, we need some more convex analysis!

### 2.2.1 Convex Analysis Bits: Subgradients

First, we need a technical definition.

Definition 2.17 (Proper Function). If a function $f$ is nowhere $-\infty$ and finite somewhere, then $f$ is called proper.

In this book, we are mainly interested in convex proper functions, that basically better conform to our intuition of what a convex function looks like.

Example 2.18. The indicator function of a set $\mathcal{V} \subset \mathbb{R}^d$ is proper iff $\mathcal{V}$ is non-empty.

Let's first define formally what is a subgradient.

Definition 2.19 (Subgradient). For a proper function $f : \mathbb{R}^d \to (-\infty, +\infty]$, we define a subgradient of $f$ in $x \in \mathbb{R}^d$ as a vector $g \in \mathbb{R}^d$ that satisfies
$$f(y) \geq f(x) + \langle g, y - x \rangle, \quad \forall y \in \mathbb{R}^d.$$

Basically, a subgradient of $f$ in $x$ is any vector $g$ that allows us to construct a linear lower bound to $f$. Note that the subgradient is not unique, so we denote the set of subgradients of $f$ in $x$ by $\partial f(x)$, called subdifferential of $f$ at $x$.

Observe that if $f$ is proper and convex, then $\partial f(x)$ is empty for $x \notin \text{dom } f$, because the inequality cannot be satisfied when $f(x) = +\infty$. Also, the domain of $\partial f$, denoted by $\text{dom } \partial f$, is the set of all $x \in \mathbb{R}^d$ such that $\partial f(x)$ is non-empty; it is a subset of $\text{dom } f$. A proper convex function $f$ is always subdifferentiable in $\text{int dom } f$ [Rockafellar, 1970, Theorem 23.4].

Note that we did not assume the function $f$ to be convex in the definition of a subgradient. However, the following theorem tells us that if we have subgradients everywhere, then the function must be convex.

Theorem 2.20. Let $f : \mathcal{V} \to \mathbb{R}$, where $\mathcal{V} \subseteq \mathbb{R}^d$ is convex. If $\partial f(x) \neq \{\}$ for all $x \in \mathcal{V}$, then $f$ is convex.

Proof. For any $x_1, x_2 \in \mathcal{V}$ and for any $\lambda \in [0, 1]$, consider $y = \lambda x_1 + (1 - \lambda)x_2 \in \mathcal{V}$ and $g \in \partial f(y)$. Then, we have
$$f(x_1) \geq f(y) + \langle g, x_1 - y \rangle = f(y) + (1 - \lambda) \langle g, x_1 - x_2 \rangle,$$
$$f(x_2) \geq f(y) + \langle g, x_2 - y \rangle = f(y) + \lambda \langle g, x_2 - x_1 \rangle.$$

Multiplying the first inequality by $\lambda$ and the second one by $1 - \lambda$ and adding them together, we have
$$\lambda f(x_1) + (1 - \lambda) f(x_2) \geq f(y) = f(\lambda x_1 + (1 - \lambda)x_2),$$
that implies the convexity of $f$ in $\mathcal{V}$.

The unique subgradient of a differentiable function is just the gradient, as quantified in the next theorem.

Theorem 2.21 ([Rockafellar, 1970, Theorem 25.1]). If the function $f : \mathbb{R}^d \to [-\infty, +\infty]$ is convex and finite in $x$, it is differentiable in $x$ iff the subdifferential is composed by a unique element, that turns out to be $\nabla f(x)$.

We can also calculate subgradients of sum of functions.

Theorem 2.22. Let $f_1, \dots, f_m$ be proper functions on $\mathbb{R}^d$, and $f = f_1 + \dots + f_m$. Then, $\partial f(x) \supseteq \partial f_1(x) + \dots + \partial f_m(x), \forall x$. Moreover, if $f_1, \dots, f_m$ are also convex, closed, and $\text{dom } f_m \cap \bigcap_{i=1}^{m-1} \text{int dom } f_i \neq \{\}$, then actually $\partial f(x) = \partial f_1(x) + \dots + \partial f_m(x), \forall x$.

Proof. For any $z$, define $g_i \in \partial f_i(z)$ for $i = 1, \dots, m$. From the definition of subgradient, we have
$$f(x) = \sum_{i=1}^m f_i(x) \geq \sum_{i=1}^m (f_i(z) + \langle g_i, x - z \rangle) = f(z) + \left\langle \sum_{i=1}^m g_i, x - z \right\rangle.$$
Hence, $\sum_{i=1}^m g_i \in \partial f(z)$.
For the second statement, see Bauschke and Combettes [2017, Corollary 16.50].

Example 2.23. Let $f(x) = |x|$, then the subdifferential set $\partial f(x)$ is
$$\partial f(x) = \begin{cases} \{1\}, & x > 0, \\ [-1, 1], & x = 0, \\ \{-1\}, & x < 0. \end{cases}$$

Example 2.24. Let's calculate the subgradient of the indicator function for a non-empty convex set $\mathcal{V} \subset \mathbb{R}^d$. By definition, $g \in \partial i_\mathcal{V}(x)$ if
$$i_\mathcal{V}(y) \geq i_\mathcal{V}(x) + \langle g, y - x \rangle, \quad \forall y \in \mathbb{R}^d.$$
This condition implies that $x \in \mathcal{V}$ and $0 \geq \langle g, y - x \rangle, \forall y \in \mathcal{V}$ (because for $y \notin \mathcal{V}$ the inequality is always verified). The set of all $g$ that satisfies the above inequality is called the normal cone to $\mathcal{V}$ at $x$ and it is denoted by $N_\mathcal{V}(x)$. Note that the normal cone for any $x \in \text{int } \mathcal{V}$ is $\{0\}$ (Hint: take $y = x + \epsilon g$). For example, for $\mathcal{V} = \{x \in \mathbb{R}^d : \|x\|_2 \leq 1\}$, $N_\mathcal{V}(x) = \{\alpha x | \alpha \geq 0\}$ for all $x : \|x\|_2 = 1$.

Another useful theorem is to calculate the subdifferential of the pointwise maximum of convex functions.

Theorem 2.25 ([Bauschke and Combettes, 2017, Theorem 18.5]). Let $(f_i)_{i \in I}$ be a finite set of convex functions from $\mathbb{R}^d$ to $(-\infty, +\infty]$ and suppose $x \in \bigcap_{i \in I} \text{dom } f_i$ and $f_i$ continuous at $x$. Set $F = \max_{i \in I} f_i$ and let $A(x) = \{i \in I | f_i(x) = F(x)\}$ the set of the active functions. Then
$$\partial F(x) = \text{conv } \bigcup_{i \in A(x)} \partial f_i(x),$$
where $\text{conv}$ is the convex hull.

Example 2.26 (Subgradients of the Hinge loss). Consider the loss $\ell(x) = \max(1 - \langle z, x \rangle, 0)$ for $z \in \mathbb{R}^d$. The subdifferential set is
$$\partial \ell(x) = \begin{cases} \{0\}, & 1 - \langle z, x \rangle < 0 \\ \{-\alpha z | \alpha \in [0, 1]\}, & 1 - \langle z, x \rangle = 0 \\ \{-z\}, & \text{otherwise} \end{cases}$$

Finally, we can show a result on the subgradient of affine transformations.

Theorem 2.27. Let $f : \mathbb{R}^d \to (\infty, +\infty]$ proper. Define $h(x) = f(Ax + b)$, where $A \in \mathbb{R}^{m \times d}$ and $b \in \mathbb{R}^m$. Then, we have $A^\top \partial f(Ax + b) \subseteq \partial h(x)$.

Proof. For any $g \in \partial f(Ax + b)$, we want to show that $A^\top g$ is a subgradient of $h$ in $x$. From the definition of subgradient, for all $y \in \mathbb{R}^d$ we have
$$h(x) = f(Ay + b) \geq f(Ax + b) + \langle g, Ay + b - (Ax + b) \rangle = h(y) + \langle A^\top g, y - x \rangle,$$
that implies our stated result.

Definition 2.28 (Lipschitz Function). Let $f : \mathbb{R}^d \to (-\infty, +\infty]$ is $L$-Lipschitz over a set $\mathcal{V}$ w.r.t a norm $\|\cdot\|$ if $|f(x) - f(y)| \leq L\|x - y\|, \forall x, y \in \mathcal{V}$.

We also have this handy result that upper bounds the norm of subgradients of convex Lipschitz functions.

Theorem 2.29. Let $f : \mathbb{R}^d \to (-\infty, +\infty]$ proper. Then, $f$ is $L$-Lipschitz in $\text{int dom } f$ with respect to the $L_2$ norm iff for all $x \in \text{int dom } f$ and $g \in \partial f(x)$ we have $\|g\|_2 \leq L$.

Proof. Assume $f$ $L$-Lipschitz, then $|f(x) - f(y)| \leq L\|x - y\|_2, \forall x, y \in \text{int dom } f$. For $\epsilon > 0$ small enough $y = x + \epsilon \frac{g}{\|g\|_2} \in \text{int dom } f$, then
$$L\epsilon = L\|x - y\|_2 \geq |f(y) - f(x)| \geq f(y) - f(x) \geq \langle g, y - x \rangle = \epsilon \|g\|_2,$$
that implies that $\|g\|_2 \leq L$.

For the other implication, the definition of subgradient and Cauchy-Schwarz inequalities gives us
$$f(x) - f(y) \leq \|g\|_2 \|x - y\|_2 \leq L\|x - y\|_2,$$
for any $x, y \in \text{int dom } f$. Taking $g \in \partial f(y)$, we also get
$$f(y) - f(x) \leq L\|x - y\|_2,$$
that completes the proof.

Finally, let's dispel the common misconception that a convex function must be differentiable on its domain except possibly at countably many points. For example, consider the convex function $f : \mathbb{R}^2 \to \mathbb{R}$ defined as $f(x) = |x_1|$ that is not differentiable on the line segment between $v = (0, 0)$ and $w = (0, 1)$.

### 2.2.2 Analysis with Subgradients

As I promised you, with the proper mathematical tools, analyzing online algorithms becomes easy. Indeed, switching from gradient to subgradient comes for free! In fact, our analysis of OGD with differentiable losses holds as is using subgradients instead of gradients. The reason is that the only property of the gradients that we used in the proof of Theorem 2.13 was that
$$\ell_t(x) - \ell_t(\mathbf{u}) \leq \langle g_t, x_t - \mathbf{u} \rangle,$$
where $g_t = \nabla \ell_t(x_t)$. However, the exact same property holds when $g_t \in \partial \ell_t(x_t)$. So, we can state the Online Subgradient descent algorithm in the following way, where the only difference is line 4.

Algorithm 2.2 Projected Online Subgradient Descent
Require: Non-empty closed convex set $\mathcal{V} \subseteq \mathbb{R}^d$, $x_1 \in \mathcal{V}$, $\eta_1, \dots, \eta_T > 0$
1: for $t = 1$ to $T$ do
2: Output $x_t \in \mathcal{V}$
3: Pay $\ell_t(x_t)$ for an $\ell_t : \mathcal{V} \to \mathbb{R}$ subdifferentiable in $\mathcal{V}$
4: Set $g_t \in \partial \ell_t(x_t)$
5: $x_{t+1} = \Pi_\mathcal{V} (x_t - \eta_t g_t) = \text{argmin}_{y \in \mathcal{V}} \|x_t - \eta_t g_t - y\|_2$
6: end for

Also, the regret bounds we proved hold as well, just changing differentiability with subdifferentiability and gradients with subgradients. In particular, we have the following Lemma.

Lemma 2.30. Let $\mathcal{V} \subseteq \mathbb{R}^d$ be a non-empty closed convex set and $\ell_t : \mathcal{V} \to \mathbb{R}$ a convex function subdifferentiable in $\mathcal{V}$. Set $g_t \in \partial \ell_t(x_t)$. Then, $\forall \mathbf{u} \in \mathcal{V}$, the following inequality holds
$$\eta_t(\ell_t(x_t) - \ell_t(\mathbf{u})) \leq \eta_t \langle g_t, x_t - \mathbf{u} \rangle \leq \frac{1}{2} \|x_t - \mathbf{u}\|_2^2 - \frac{1}{2} \|x_{t+1} - \mathbf{u}\|_2^2 + \frac{\eta_t^2}{2} \|g_t\|_2^2.$$

Example 2.31. Consider again the guessing game of the first class, but now change the loss function to the absolute loss of the difference: $\ell_t(x) = |x - y_t|$. Now we will need to use Online Subgradient Descent, because the functions are non-differentiable. We can easily see that
$$\partial \ell_t(x) = \begin{cases} \{1\}, & x > y_t \\ [-1, 1], & x = y_t \\ \{-1\}, & x < y_t. \end{cases}$$
Again, running Online Subgradient Descent with the optimal learning rate on this problem will give us immediately a regret of $\mathcal{O}(\sqrt{T})$, without having to design a specific strategy for it.

## 2.3 From Convex Losses to Linear Losses

Let's take a deeper look at this step
$$\ell_t(x_t) - \ell_t(\mathbf{u}) \leq \langle g_t, x_t - \mathbf{u} \rangle, \quad \forall \mathbf{u} \in \mathbb{R}^d.$$
Summing over time, we have
$$\sum_{t=1}^T (\ell_t(x_t) - \ell_t(\mathbf{u})) \leq \sum_{t=1}^T \langle g_t, x_t - \mathbf{u} \rangle, \quad \forall \mathbf{u} \in \mathbb{R}^d.$$

Now, define the linear (and convex) losses $\tilde{\ell}_t(x) := \langle g_t, x \rangle$, so we have
$$\sum_{t=1}^T (\ell_t(x_t) - \ell_t(\mathbf{u})) \leq \sum_{t=1}^T (\tilde{\ell}_t(x_t) - \tilde{\ell}_t(\mathbf{u})).$$

This is more powerful than what it seems: We upper bounded the regret with respect to the convex losses $\ell_t$ with a regret with respect to another sequence of linear losses. This is important because it implies that we can build online algorithms that deal only with linear losses, and through the reduction above they can be seamlessly used as OCO algorithms! Note that this does not imply that this reduction is always optimal, as we saw in Example 2.15. But, it allows us to easily construct optimal OCO algorithms in many interesting cases.

So, we will often consider just the problem of minimizing the linear regret
$$\text{Regret}_T(\mathbf{u}) = \sum_{t=1}^T \langle g_t, x_t \rangle - \sum_{t=1}^T \langle g_t, \mathbf{u} \rangle, \quad \forall \mathbf{u} \in \mathcal{V} \subseteq \mathbb{R}^d.$$
This problem is called Online Linear Optimization (OLO).

## 2.4 History Bits

Shor [1964] first introduced the subgradient descent method using a related concept of generalized gradients for non-differentiable functions. Instead, the concept of subgradients and the possibility to build a calculus for them appear for the first time in Rockafellar [1963]. The Projected Online Gradient Descent with time-varying learning rate and the name "Online Convex Optimization" were introduced by Zinkevich [2003], but the OCO framework was introduced earlier by Gordon [1999a,b].

Before the online convex optimization framework, the online learning community focused on specific losses and mostly linear predictors, see Cesa-Bianchi and Lugosi [2006]. Moreover, the concept of subgradient is also a recent addition to the online learning literature, and the earlier papers using subgradients of loss functions seem to be Zhang [2004], Shalev-Shwartz and Singer [2006b]. However, subgradients were implicitly used in previous analyses too, see for example Cesa-Bianchi [1999, Theorem 4].

It is impossible not to notice that the literature on OCO is heavily biased on the use of bounded domains. Yet, there are very few practical applications of OCO where the domain is bounded. Moreover, as shown in Theorem 2.13, it is possible to have a sublinear regret even in unbounded domains. I speculate that the insistence on using bounded domains is due to two main reasons: the less general definition of regret (derived from the setting of learning with experts (Section 6.8)) as
$$\sum_{t=1}^T \ell_t(x_t) - \min_{\mathbf{u} \in \mathcal{V}} \sum_{t=1}^T \ell_t(\mathbf{u})$$
that requires the existence of the minimizer of the sum of the losses and the fact that Online Mirror Descent (OMD) with time-varying learning rates has a vacuous regret upper bound on unbounded domains. However, we now know that both these two issues can easily be fixed i) by defining the regret with respect to arbitrary competitors and ii) by using different algorithms, for example, Follow-The-Regularized-Leader (FTRL). Indeed, even in offline optimization the existence of the minimizer is not required to guarantee that the suboptimality gap goes to zero, as we will see in Example 3.2. So, it is probably time for the community to stop using the crutch of bounded domains just to avoid technical difficulties.

## 2.5 Exercises

Problem 2.1. Prove that $\sum_{t=1}^T \frac{1}{\sqrt{t}} \leq 2\sqrt{T} - 1$.

Problem 2.2. Using the inequality in the previous exercise, prove that a learning rate $\eta_t \propto \frac{1}{\sqrt{t}}$ gives rise to a regret only a constant multiplicative factor worse than the one in (2.1).

Problem 2.3. Calculate the subdifferential set of the $\epsilon$-insensitive loss: $f(x) = \max(|x - y| - \epsilon, 0)$. It is a loss used in regression problems where we do not want to penalize predictions $x$ within $\pm \epsilon$ of the correct value $y$.

Problem 2.4. Using the definition of subgradient, find the subdifferential set of $f(x) = \|x\|_2, x \in \mathbb{R}^d$.

Problem 2.5. Consider Projected Online Subgradient Descent for the Example 2.10 on the failure of Follow-the-Leader: Can we use it on that problem? Would it guarantee sublinear regret? How the behaviour of the algorithm would differ from FTL?

# Chapter 3

# Online-to-Batch Conversions

In this chapter, we take a break from online learning theory and see some application of online learning to other domains. For example, we may wonder what is the connection between online learning and stochastic optimization. Given that projected Online Subgradient Descent (OSD) looks basically the same as projected stochastic (sub)gradient descent, they must have something in common. Indeed, we can show that, for example, we can reduce stochastic optimization of convex functions to Online Convex Optimization (OCO). Let's see how.

## 3.1 From Online Learning to Stochastic Optimization

Theorem 3.1. Let $\mathcal{V}$ be a non-empty closed convex set of $\mathbb{R}^d$, $F(x) = \mathbb{E}[f(x, \xi)]$ where the expectation is with respect to $\xi$ drawn from a distribution $\rho$ over some vector space $\mathcal{X}$, and $f : \mathcal{V} \times \mathcal{X} \to \mathbb{R}$ is convex and subdifferentiable in the first argument in $\mathcal{V}$. Draw $T$ samples $\xi_1, \dots, \xi_T$ i.i.d. from $\rho$ and construct the sequence of losses $\ell_t(x) = \alpha_t f(x, \xi_t)$, where $\alpha_t > 0$ are deterministic. Run any Online Convex Optimization (OCO) algorithm over the losses $\ell_t$, to construct the sequence of predictions $x_1, \dots, x_{T+1}$. Then, we have
$$\mathbb{E} \left[ F\left( \frac{1}{\sum_{t=1}^T \alpha_t} \sum_{t=1}^T \alpha_t x_t \right) \right] \leq F(\mathbf{u}) + \frac{\mathbb{E}[\text{Regret}_T(\mathbf{u})]}{\sum_{t=1}^T \alpha_t}, \quad \forall \mathbf{u} \in \mathcal{V},$$
where the expectation is with respect to $\xi_1, \dots, \xi_T$.

Proof. We first show that
$$\mathbb{E} \left[ \sum_{t=1}^T \alpha_t F(x_t) \right] = \mathbb{E} \left[ \sum_{t=1}^T \ell_t(x_t) \right]. \quad (3.1)$$
In fact, from the linearity of the expectation we have
$$\mathbb{E} \left[ \sum_{t=1}^T \ell_t(x_t) \right] = \sum_{t=1}^T \mathbb{E} [\ell_t(x_t)].$$
Then, from the law of total expectation, we have
$$\mathbb{E}_{\xi_1, \dots, \xi_T} [\ell_t(x_t)] = \mathbb{E}_{\xi_1, \dots, \xi_{t-1}} [\mathbb{E}_{\xi_t, \dots, \xi_T} [\ell_t(x_t)|\xi_1, \dots, \xi_{t-1}]] = \mathbb{E}_{\xi_1, \dots, \xi_{t-1}} [\mathbb{E}_{\xi_t, \dots, \xi_T} [\alpha_t f(x_t, \xi_t)|\xi_1, \dots, \xi_{t-1}]] = \mathbb{E}_{\xi_1, \dots, \xi_{t-1}} [\mathbb{E}_{\xi_t} [\alpha_t f(x_t, \xi_t)|\xi_1, \dots, \xi_{t-1}]] = \mathbb{E}_{\xi_t, \dots, \xi_T} [\alpha_t F(x_t)],$$
where we used the fact that $x_t$ depends only on $\xi_1, \dots, \xi_{t-1}$. Hence, (3.1) is proved.

It remains only to use Jensen's inequality, using the fact that $F$ is convex, to have
$$F\left( \frac{1}{\sum_{t=1}^T \alpha_t} \sum_{t=1}^T \alpha_t x_t \right) \leq \frac{1}{\sum_{t=1}^T \alpha_t} \sum_{t=1}^T \alpha_t F(x_t).$$
Dividing the regret by $\sum_{t=1}^T \alpha_t$ and using the above inequalities gives the stated theorem.

Let's see now some applications of this result: Let's see how to use the above theorem to transform Online Subgradient Descent in Stochastic Subgradient Descent to minimize the training error of a classifier.

Example 3.2. Consider a problem of binary classification, with inputs $z_i \in \mathbb{R}^d$ and outputs $y_i \in \{-1, 1\}$. The loss function is the hinge loss: $f(x, (z, y)) = \max(1 - y\langle z, x \rangle, 0)$. Suppose that you want to minimize the training error over a training set of $N$ samples, $\{(z_i, y_i)\}_{i=1}^N$. Also, assume the maximum $L_2$ norm of the samples is $R$. That is, we want to minimize
$$\min_x F(x) := \frac{1}{N} \sum_{i=1}^N \max(1 - y_i\langle z_i, x \rangle, 0).$$
Run the reduction described in Theorem 3.1 for $T$ iterations using Online Gradient Descent (OGD). In each iteration, construct $\ell_t(x) = \max(1 - y_t\langle z_t, x \rangle, 0)$ sampling a training point uniformly at random from 1 to $N$. Set $x_1 = 0$ and $\eta = \frac{1}{R\sqrt{T}}$. We have that
$$\mathbb{E} \left[ F\left( \frac{1}{T} \sum_{t=1}^T x_t \right) \right] - F(x^\star) \leq R \frac{\|x^\star\|_2^2 + 1}{2\sqrt{T}},$$
for all $x^\star \in \text{argmin}_x F(x)$. In words, we used an OCO algorithm to stochastically optimize a function, transforming the regret guarantee into a convergence rate guarantee.

In this last example, we have to use a constant learning rate to be able to minimize the training error over the entire space $\mathbb{R}^d$. In the next one, we will see a different approach, that allows us to implicitly use a varying learning rate without the need of a bounded feasible set.

Example 3.3. Consider the same setting of the previous example, and let's change the way in which we construct the online losses. Now use $\ell_t(x) = \frac{1}{R\sqrt{t}} \max(1 - y_t\langle z_t, x \rangle, 0)$ and step size $\eta = 1$. Hence, we have
$$\mathbb{E} \left[ F\left( \frac{1}{\sum_{t=1}^T \frac{1}{\sqrt{t}}} \sum_{t=1}^T \frac{1}{\sqrt{t}} x_t \right) \right] - F(x^\star) \leq \frac{\|x^\star\|_2^2}{2 \sum_{t=1}^T \frac{1}{R\sqrt{t}}} + \frac{1}{2 \sum_{t=1}^T \frac{1}{R\sqrt{t}}} \sum_{t=1}^T \frac{1}{t} \leq R \frac{\|x^\star\|_2^2 + 1 + \ln T}{4\sqrt{T + 1} - 4},$$
where we used $\sum_{t=1}^T \frac{1}{\sqrt{t}} \geq 2\sqrt{T + 1} - 2$.

Remark 3.4. Using the online-to-batch conversion and online subgradient descent to minimize the expectation of convex Lipschitz functions we can obtain a convergence rate of $\mathcal{O}(\frac{1}{\sqrt{T}})$, that is optimal for this class of problems. This should dispel the common misconception that online algorithms are suboptimal in the stochastic setting because designed to work in the adversarial case. Indeed, the opposite is true: Virtually any optimal guarantee for offline optimization can be recovered using online learning algorithms.

I stressed the fact that the only meaningful way to define a regret is with respect to an arbitrary point in the feasible set. This is obvious in the case we consider unconstrained Online Linear Optimization (OLO), because the optimal competitor is unbounded. But, it is also true in unconstrained OCO. Let's see an example of this.

Example 3.5. Consider a problem of binary classification, with inputs $z_i \in \mathbb{R}^d$ and outputs $y_i \in \{-1, 1\}$. The loss function is the logistic loss: $f(x, (z, y)) = \ln(1 + \exp(-y\langle z, x \rangle))$. Suppose that you want to minimize the training error over a training set of $N$ samples, $\{(z_i, y_i)\}_{i=1}^N$. Also, assume the maximum $L_2$ norm of the samples is $R$. That is, we want to minimize
$$\min_x F(x) := \frac{1}{N} \sum_{i=1}^N \ln(1 + \exp(-y_i\langle z_i, x \rangle)).$$

So, run the reduction described in Theorem 3.1 for $T$ iterations using Online Subgradient Descent (OSD). In each iteration, construct $\ell_t(x) = \ln(1 + \exp(-y_t\langle z_t, x \rangle))$ sampling a training point uniformly at random from 1 to $N$. Set $x_1 = 0$ and $\eta = \frac{1}{R\sqrt{T}}$. We have that
$$\mathbb{E} \left[ F\left( \frac{1}{T} \sum_{t=1}^T x_t \right) \right] \leq \frac{R}{2\sqrt{T}} + \min_{\mathbf{u} \in \mathbb{R}^d} F(\mathbf{u}) + R \frac{\|\mathbf{u}\|_2^2}{2\sqrt{T}}.$$

In words, we will be $\frac{R}{2\sqrt{T}}$ away from the optimal value of regularized empirical risk minimization problem, where the weight of the regularization is $\frac{R}{2\sqrt{T}}$. Now, let's consider the case that the training set is linearly separable, this means that the infimum of $F$ is 0 and the optimal solution does not exist, i.e., it has norm equal to infinity. So, any convergence guarantee that depends on $x^\star$ would be vacuous. On the other hand, our guarantee above still makes perfectly sense.

Note that the above examples only deal with training error. However, in the next sections we show a more interesting application of the online-to-batch conversion, that is to directly minimize the generalization error. Moreover, we will see guarantees in high probability, rather than just in expectation.

### 3.1.1 Bits on Concentration Inequalities

We will use a concentration inequality to prove the high probability guarantee, but we will need to go beyond the sum of i.i.d. random variables. In particular, we will use the concept of martingales.

Definition 3.6 (Martingale). A sequence of random variables $Z_1, Z_2, \dots$ is called a martingale if for all $t \geq 1$ it satisfies:
$\mathbb{E}[|Z_t|] < \infty, \quad \mathbb{E}[Z_{t+1}|Z_1, \dots, Z_t] = Z_t.$

Definition 3.7 (Supermartingale). A sequence of random variables $Z_1, Z_2, \dots$ is called a supermartingale if for all $t \geq 1$ it satisfies:
$\mathbb{E}[|Z_t|] < \infty, \quad \mathbb{E}[Z_{t+1}|Z_1, \dots, Z_t] \leq Z_t.$

Example 3.8. Consider a fair coin $c_t$ and a betting algorithm that bets $|x_t|$ money on each round on the side of the coin equal to $\text{sign}(x_t)$. We win or lose money 1:1, so the total money we won up to round $t$ is $Z_t = \sum_{i=1}^t c_i x_i$. $Z_1, \dots, Z_t$ is a martingale. Indeed, we have
$$\mathbb{E}[Z_t|Z_1, \dots, Z_{t-1}] = \mathbb{E}[Z_{t-1} + x_t c_t|Z_1, \dots, Z_{t-1}] = Z_{t-1} + \mathbb{E}[x_t c_t|Z_1, \dots, Z_{t-1}] = Z_{t-1}.$$

If we throw away part of the wealth in each round, we obtain a supermartingale.

For bounded martingales we can prove high probability guarantees as for bounded i.i.d. random variables. The following Theorem will be the key result we will need.

Theorem 3.9 (Hoeffding-Azuma inequality). Let $Z_0, \dots, Z_T$ be a martingale of $T$ random variables that satisfy $|Z_t - Z_{t+1}| \leq B, t = 1, \dots, T-1$ almost surely. Then, we have
$$\mathbb{P}\{Z_T - Z_0 \geq \epsilon\} \leq \exp \left( -\frac{\epsilon^2}{2B^2T} \right).$$

Also, the same upper bounds hold on $\mathbb{P}\{Z_0 - Z_T \geq \epsilon\}$.

### 3.1.2 High-Probability Guarantees for Online-to-Batch Conversion

We now show how the online-to-batch conversion we introduced can be strengthen to produce guarantees in high probability.

Theorem 3.10. Let $\mathcal{V} \subset \mathbb{R}^d$, $F(x) = \mathbb{E}[f(x, \xi)]$, where the expectation is with respect to $\xi$ drawn from $\rho$ with support over some set $\mathcal{D}$, and $f : \mathcal{V} \times \mathcal{D} \to [0, 1]$. Draw $T$ samples $\xi_1, \dots, \xi_T$ i.i.d. from $\rho$ and construct the sequence of losses $\ell_t(x) = f(x, \xi_t)$. Let $\mathcal{A}$ any online learning algorithm over the losses $\ell_t$ that outputs the sequence of predictions $x_1, \dots, x_{T+1}$ and guarantees $\text{Regret}_T(\mathbf{u}) \leq R(\mathbf{u}, T)$ for all $\mathbf{u} \in \mathcal{V}$, for a function $R : \mathcal{V} \times \mathbb{N} \to \mathbb{R}$. Then, we have with probability at least $1 - δ$, it holds that
$$\frac{1}{T} \sum_{t=1}^T F(x_t) \leq \min_{\mathbf{u} \in \mathcal{V}} F(\mathbf{u}) + \frac{R(\mathbf{u}, T)}{T} + 2 \sqrt{\frac{2 \ln \frac{2}{δ}}{T}}.$$

Proof. Define $Z_t = \sum_{i=1}^t (F(x_i) - \ell_i(x_i))$ for $t = 1, \dots, T$ and $Z_0 = 0$. We claim that $Z_t$ is a martingale. In fact, we have
$$\mathbb{E}[\ell_t(x_t)|\xi_1, \dots, \xi_{t-1}] = \mathbb{E}[f(x_t, \xi_t)|\xi_1, \dots, \xi_{t-1}] = F(x_t),$$
where we used the fact that $x_t$ depends only on $\xi_1, \dots, \xi_{t-1}$. Hence, we have
$$\mathbb{E}[Z_{t+1}|Z_1, \dots, Z_t] = Z_t + \mathbb{E}[F(x_{t+1}) - \ell_{t+1}(x_{t+1})|Z_1, \dots, Z_t] = Z_t,$$
that proves our claim.

Hence, using Theorem 3.9, we have
$$\mathbb{P}\left\{ \sum_{t=1}^T (F(x_t) - \ell_t(x_t)) \geq \epsilon \right\} = \mathbb{P}\{Z_T - Z_0 \geq \epsilon\} \leq \exp \left( -\frac{\epsilon^2}{2T} \right).$$

This implies that, with probability at least $1 - δ/2$, we have
$$\sum_{t=1}^T F(x_t) \leq \sum_{t=1}^T \ell_t(x_t) + \sqrt{2T \ln \frac{2}{δ}},$$
or equivalently
$$\frac{1}{T} \sum_{t=1}^T F(x_t) \leq \frac{1}{T} \sum_{t=1}^T \ell_t(x_t) + \sqrt{\frac{2 \ln \frac{2}{δ}}{T}}.$$

We now use the definition of regret with respect to any $\mathbf{u}$, to have
$$\frac{1}{T} \sum_{t=1}^T \ell_t(x_t) = \frac{\text{Regret}_T(\mathbf{u})}{T} + \frac{1}{T} \sum_{t=1}^T \ell_t(\mathbf{u}) \leq \frac{R(\mathbf{u}, T)}{T} + \frac{1}{T} \sum_{t=1}^T \ell_t(\mathbf{u}).$$

The last step is to upper bound with high probability $\frac{1}{T} \sum_{t=1}^T \ell_t(\mathbf{u})$ with $F(\mathbf{u})$. This is easier than the previous upper bound because we set $\mathbf{u}$ to be the fixed vector that minimizes $F(x) + \frac{R(x, T)}{T}$ in $\mathcal{V}$. So, $\ell_t(\mathbf{u})$ are i.i.d. random variables and for sure $Z_t = \sum_{i=1}^t (F(\mathbf{u}) - \ell_i(\mathbf{u}))$ forms a martingale. So, reasoning as above, we have that with probability at least $1 - δ/2$ it holds that
$$\frac{1}{T} \sum_{t=1}^T \ell_t(\mathbf{u}) \leq F(\mathbf{u}) + \sqrt{\frac{2 \ln \frac{2}{δ}}{T}}.$$

Putting all together and using the union bound, we have the stated bound.

The theorem above upper bounds the average value of the $T$ different solutions, while we are interested in producing a single solution. If $F$ is a convex function and $\mathcal{V}$ is convex, than we can lower bound the l.h.s. of the inequalities in the theorem with the function evaluated on the average of the $x_t$. That is
$$F\left( \frac{1}{T} \sum_{t=1}^T x_t \right) \leq \frac{1}{T} \sum_{t=1}^T F(x_t).$$

Remark 3.11. Note that when using the online-to-batch conversion to optimize a population risk, i.e., $F(x) = \mathbb{E}_{\xi \sim \rho}[f(x, \xi)]$, by sampling $T$ random vectors $\xi_1, \dots, x_T$ is not the same as using the online-to-batch conversion to minimize the empirical risk, $\hat{F}(x) = \frac{1}{T} \sum_{t=1}^T f(x, \xi_t)$. In fact, in the second case, we would sample from the $T$ samples with replacement, while in the first case we go over the samples in order that corresponds sampling without replacement. This is not a minor difference because one can show that there are cases where sampling without replacement will not minimize the empirical risk [Vansover-Hager et al., 2025].

## 3.2 Application: Agnostic PAC Learning

In this section, we show another application of online-to-batch methods to obtain statistical learning guarantees. Here, we assume to have a prediction strategy $\phi_x$ parametrized by a vector $x$ and we want to learn the relationship between an input $z$ and its associated label $y$. Moreover, we will assume that $(z, y)$ is drawn from a joint probability distribution $\rho$. Also, we are equipped with a loss function, $\ell(\hat{y}, y)$, that measures how good our prediction $\hat{y} = \phi_x(z)$ is, compared to the true label $y$. So, learning the relationship can be cast as minimizing the expected loss of our predictor
$$\min_{x \in \mathcal{V}} \mathbb{E}_{(z, y) \sim \rho} [\ell(\phi_x(z), y)].$$

In machine learning terms, the object above is nothing else than the test error of our predictor.

Note that the above setting assumes labeled samples, but we can generalize it even more considering the Vapnik's general setting of learning, where we collapse the prediction function and the loss in a unique function. This allows, for example, to treat supervised and unsupervised learning in the same unified way. So, we want to minimize the risk
$$\min_{x \in \mathcal{V}} (\text{Risk}(x) := \mathbb{E}_{\xi \sim \rho} [f(x, \xi)]),$$
where $\rho$ is an unknown distribution over $\mathcal{D}$ and $f : \mathbb{R}^d \times \mathcal{D} \to \mathbb{R}$ is measurable with respect to the second argument. Also, the set $\mathcal{F}$ of all predictors that can be expressed by vectors $x$ in $\mathcal{V}$ is called the hypothesis class.

Example 3.12. In a linear regression task where the loss is the square loss, we have $\xi = (z, y) \in \mathbb{R}^d \times \mathbb{R}$ and $\phi_x(z) = \langle z, x \rangle$. Hence, $f(x, \xi) = (\langle z, x \rangle - y)^2$.

Example 3.13. In linear binary classification where the loss is the hinge loss, we have $\xi = (z, y) \in \mathbb{R}^d \times \{-1, 1\}$ and $\phi_x(z) = \langle z, x \rangle$. Hence, $f(x, \xi) = \max(1 - y\langle z, x \rangle, 0)$.

Example 3.14. In binary classification with a neural network with the logistic loss, we have $\xi = (x, y) \in \mathbb{R}^d \times \{-1, 1\}$ and $\phi_x$ is the network corresponding to the weights $x$. Hence, $f(x, \xi) = \ln(1 + \exp(-y\phi_x(z)))$.

The key difficulty of the above problem is that we do not know the distribution $\rho$. Hence, there is no hope to exactly solve this problem. Instead, we are interested in understanding what is the best we can do if we have access to $T$ samples drawn i.i.d. from $\rho$. More in details, we want to upper bound the excess risk
$$\text{Risk}(x_T) - \min_x \text{Risk}(x),$$
where $x_T$ is a predictor that was learned using $T$ samples.

It should be clear that this is just an optimization problem and the one above is just the suboptimality gap. In this view, the objective of machine learning can be considered as a particular optimization problem.

Remark 3.15. Note that this is not the only way to approach the problem of learning. Indeed, the regret minimization model is an alternative model to learning. Moreover, another approach would be to try to estimate the distribution $\rho$ and then solve the risk minimization problem. No approach is superior to the other and each of them has its pros and cons.

Given that we have access to the distribution $\rho$ through samples drawn from it, any procedure we might think to use to minimize the risk will be stochastic in nature. This means that we cannot assure a deterministic guarantee. Instead, we can try to prove that with high probability our minimization procedure will return a solution that is close to the minimizer of the risk. It is also intuitive that the precision and probability we can guarantee must depend on how many samples we draw from $\rho$.

Quantifying the dependency of precision and probability of failure on the number of samples used is the objective of the Agnostic Probably Approximately Correct (PAC) framework, where the keyword "agnostic" refers to the fact that we do not assume anything on the best possible predictor. More in details, given a precision parameter $\epsilon$ and a probability of failure $δ$, we are interested in characterizing the sample complexity of the hypothesis class $\mathcal{F}$ that is defined as the number of samples $T$ necessary to guarantee with probability at least $1 - δ$ that the best learning algorithm using the hypothesis class $\mathcal{F}$ outputs a solution $x_T$ that has an excess risk upper bounded by $\epsilon$. Note that the sample complexity does not depend on $\rho$, so it is a worst-case measure with respect to all the possible distributions. This makes sense if you think that we know nothing about the distribution $\rho$, so if your guarantee holds for the worst distribution it will also hold for any other distribution. Mathematically, we will say that the hypothesis class is agnostic PAC-learnable is such sample complexity function exists.

Definition 3.16 (Agnostic-PAC-learnable). We will say that a function class $\mathcal{F} = \{f(x, \cdot) : x \in \mathbb{R}^d\}$ is Agnostic PAC-learnable if there exists an algorithm $\mathcal{A}$ and a function $T(\epsilon, δ) : \mathbb{R} \times [0, 1] \to \mathbb{N}$ such that when $\mathcal{A}$ is used with $T \geq T(\epsilon, δ)$ samples drawn from $\rho$, with probability at least $1 - δ$ the solution $x_T$ returned by the algorithm has excess risk at most $\epsilon$.

Note that the Agnostic PAC learning setting does not say what is the procedure we should follow to find such sample complexity. The approach most commonly used in machine learning to solve the learning problem is the so-called Empirical Risk Minimization (ERM) procedure. It consist of drawing $T$ samples i.i.d. from $\rho$ and minimizing the empirical risk defined as
$$\widehat{\text{Risk}}(x) := \min_{x \in \mathcal{V}} \frac{1}{T} \sum_{t=1}^T f(x; \xi_t).$$

The minimizer $\hat{x}_T$ is called the empirical risk minimizer. In words, ERM is nothing else than the minimization of the some loss function on a training set. However, in many interesting cases we can have that $\text{argmin}_{x \in \mathcal{V}} \frac{1}{T} \sum_{t=1}^T f(x; \xi_t)$ can be very far from the true optimum $\text{argmin}_{x \in \mathcal{V}} \mathbb{E}[f(x; \xi)]$, even with an infinite number of samples! So, we need to modify the ERM formulation in some way, e.g., using a regularization term, a Bayesian prior of $x$, or more generally find conditions under which ERM works.

It is worth stressing that sometimes people are concerned with the difference between the training error and the test error of the trained predictor, i.e., $\text{Risk}(\hat{x}_T) - \widehat{\text{Risk}}(\hat{x}_T)$. However, this gap can be large without implying anything on the risk of the trained predictor.

The ERM approach is so widespread that machine learning itself is often wrongly identified with some kind of minimization of the training error. We now show that ERM is not the entire world of ML, showing that the existence of a no-regret algorithm, that is an online learning algorithm with sublinear regret, guarantee Agnostic-PAC learnability. More in details, we will see that an online algorithm with sublinear regret can be used to solve machine learning problems. This is not just a curiosity, for example this gives rise to computationally efficient parameter-free algorithms, that can be achieved through ERM only running a two-step procedure, i.e., running ERM with different parameters and selecting the best solution among them.

We can use Theorem 3.10, to produce a solution with small risk. In particular, if the risk is convex, we can output the average of the $x_t$, using Jensen's inequality.

If the risk is not a convex function, we need a different way. An alternative solution is to construct a stochastic classifier that samples one of the $x_t$ with uniform probability and predicts with it. For this classifier, we immediately have
$$\text{Risk}(\{x_1, \dots, x_T\}) = \frac{1}{T} \sum_{t=1}^T \text{Risk}(x_t),$$
where the expectation in the definition of the risk of the stochastic classifier is also with respect to the random index.

Yet another way, is to select among the $T$ predictors, the one with the smallest risk. This works because the average is lower bounded by the minimum. This is easily achieved using $T/2$ samples for the online learning procedure and $T/2$ samples to generate a validation set to evaluate the solution and pick the best one. The following Theorem shows that selecting the predictor with the smallest empirical risk on a validation set will give us a predictor close to the best one with high probability.

Theorem 3.17. Let $\mathcal{V} \subset \mathbb{R}^d$, $\text{Risk}(x) = \mathbb{E}[f(x, \xi)]$, where the expectation is with respect to $\xi$ drawn from $\rho$ with support over some set $\mathcal{D}$, and $f : \mathcal{V} \times \mathcal{D} \to [0, 1]$. We have a finite set of vectors $S = \{x_1, \dots, x_{|S|}\}$ and $T$ random vectors $\xi_1, \dots, \xi_T$ drawn i.i.d. from $\rho$. Denote by $\hat{x} = \text{argmin}_{x \in S} \widehat{\text{Risk}}(x)$, where $\widehat{\text{Risk}}(x) = \frac{1}{T} \sum_{t=1}^T f(x, \xi_t)$. Then, with probability at least $1 - δ$, we have
$$\text{Risk}(\hat{x}) \leq \min_{x \in S} \text{Risk}(x) + 2 \sqrt{\frac{2 \ln(2|S|/δ)}{T}}.$$

Proof. We want to calculate the probability that the hypothesis that minimizes the validation error is far from the best hypothesis in the set. We cannot do it directly because we do not have the required independence to use a concentration inequality. Instead, we will upper bound the probability that there exists at least one function whose empirical risk is far from the risk. So, using the union bound, we have
$$\mathbb{P}\left\{ \exists x \in S : |\text{Risk}(x) - \widehat{\text{Risk}}(x)| > \frac{\epsilon}{2} \right\} \leq \sum_{i=1}^{|S|} \mathbb{P}\left\{ |\text{Risk}(x_i) - \widehat{\text{Risk}}(x_i)| > \frac{\epsilon}{2} \right\} \leq 2|S| \exp \left( -\frac{\epsilon^2 T}{8} \right).$$

Hence, with probability at least $1 - δ$, we have that
$$|\text{Risk}(x) - \widehat{\text{Risk}}(x)| \leq \frac{\epsilon}{2}, \quad \forall x \in S,$$
where $\epsilon = 2\sqrt{\frac{2 \ln(2|S|/δ)}{T}}$.

We are now able to upper bound the risk of $\hat{x}$, just using the fact that the above applies to $\hat{x}$ too. Defining $x^\star = \text{argmin}_{x \in S} \text{Risk}(x)$, we have
$$\text{Risk}(\hat{x}) \leq \widehat{\text{Risk}}(\hat{x}) + \epsilon/2 \leq \widehat{\text{Risk}}(x^\star) + \epsilon/2 \leq \text{Risk}(x^\star) + \epsilon,$$
where in the second inequality we used the fact that $\hat{x}$ minimizes the empirical risk.

Using this theorem, we can use $T/2$ samples for the training and $T/2$ samples for the validation, where $T \geq 2$. Denoting by $\hat{x}_T$ the predictor with the best empirical risk on the validation set among the $T/2$ generated during the online procedure, we have with probability at least $1 - 2δ$ that
$$\text{Risk}(\hat{x}_T) \leq \min_{\mathbf{u} \in \mathcal{V}} \text{Risk}(\mathbf{u}) + \frac{2R(\mathbf{u}, T/2)}{T} + 8\sqrt{\frac{\ln(T/δ)}{T}}.$$

It is important to note that with any of the above three methods to produce one predictor from the $T$ generated ones by the online learning procedure, the sample complexity guarantee we get matches the one we would have obtained by ERM, up to polylogarithmic factors. In other words, there is nothing special about ERM compared to the online learning approach to statistical learning. Moreover, ERM implies the existence of a hypothetical procedure that perfectly minimizes the training error. In reality, we should take into account the optimization error in the analysis of ERM. On the other hand, in the online learning approach we have a guarantee directly for the computed solution.

Another important point is that the above guarantee does not imply the existence of online learning algorithms with sublinear regret for any learning problem. It just says that, if it exists, it can be used in the statistical setting too.

### 3.3 History Bits

The specific shape of Theorem 3.1 is new, but I would not be surprised if it appeared somewhere in the literature. In particular, the uniform averaging is from Cesa-Bianchi et al. [2004], but was proposed for the absolute loss in Blum et al. [1999]. The non-uniform averaging of Example 3.3 is from Zhang [2004], even if there it is not proposed explicitly as an online-to-batch conversion.

A more recent method to do online-to-batch conversion has been introduced in Cutkosky [2019a], that independently rediscovered and generalized the averaging method in Nesterov and Shikhman [2015]. This new method allows to prove the convergence of the last iterate rather than the one of the weighted average, with a small change in any online learning algorithm.

Theorem 3.10 is from Cesa-Bianchi et al. [2004], but here I used a second concentration to state it in terms of the competitor's true risk rather than its empirical risk. Theorem 3.17 is nothing else than the Agnostic PAC learning guarantee for ERM for hypothesis classes with finite cardinality. Cesa-Bianchi et al. [2004] also gives an alternative procedure to select a single hypothesis among the $T$ generated during the online procedure that does not require splitting the data in training and validation. However, the obtained guarantee matches the one we have proved.

### 3.4 Exercises

Problem 3.1. Derive an explicit rate of convergence for SGD in Example 3.5 by upper bounding the value of the minimum on the right hand side of the convergence rate guarantee. Hint: see Ji and Telgarsky [2019].

# Chapter 4

# Beyond $\sqrt{T}$ Regret

## 4.1 Strong Convexity and Online Subgradient Descent

Let's now go back to online convex optimization theory. The example in the first chapter showed us that it is possible to get logarithmic regret in time. However, we saw that we get only $\sqrt{T}$-regret with Online Subgradient Descent (OSD) on the same game. What is the reason? It turns out that the losses in the first game, $\ell_t(x) = (x - y_t)^2$ on $[0, 1]$, are not just Lipschitz. They also possess some curvature that can be exploited to achieve a better asymptotic regret. In a moment we will see that the only change we will need to Online Subgradient Descent (OSD) is a different learning rate, dictated as usual by the regret analysis.

The key concept we will need is the one of strong convexity.

### 4.1.1 Convex Analysis Bits: Strong Convexity

Here, we introduce a stronger concept of convexity, that allows to build better lower bound to a function. Instead of the linear lower bound achievable through the use of subgradients, we will make use of quadratic lower bound.

Definition 4.1 (Strongly Convex Function). Let $\lambda \geq 0$. A proper function $f : \mathcal{X} \to (-\infty, +\infty]$ is $\lambda$-strongly convex with respect to $\|\cdot\|$ over a convex set $\mathcal{V} \subseteq \text{dom } f$ if
$$f(\alpha x + (1 - \alpha)y) \leq \alpha f(x) + (1 - \alpha)f(y) - \frac{1}{2} \lambda \alpha(1 - \alpha)\|x - y\|^2,$$
for all $x, y \in \mathcal{V}$ and all $\alpha \in (0, 1)$.

We will also say that $f$ is strongly convex in $\mathcal{V}$, if there exists $\lambda > 0$ and a norm such that the above holds.

From the definition, it is clear that if a function is $\lambda$-strongly convex, it is also $\lambda'$-strongly convex for any $0 \leq \lambda' < \lambda$. Moreover, $0$-strong convexity is just the definition of convex function.

We can also obtain an equivalent characterization in terms of subgradients.

Lemma 4.2. Let $\lambda \geq 0$. We have that $f : \mathcal{X} \to (-\infty, +\infty]$ is $\lambda$-strongly convex over a convex set $\mathcal{V} \subseteq \text{dom } \partial f$ with respect to $\|\cdot\|$ iff
$$\forall x, y \in \mathcal{V}, g \in \partial f(y), \quad f(x) \geq f(y) + \langle g, x - y \rangle + \frac{\lambda}{2} \|x - y\|^2.$$

Proof. Let's first assume that $f$ is $\lambda$-strongly convex over $\mathcal{V}$ with respect to $\|\cdot\|$. Then, for any $\alpha \in (0, 1)$ and any $g \in \partial f(y)$, we have
$$\langle g, x - y \rangle \leq \frac{f(\alpha x + (1 - \alpha)y) - f(y)}{\alpha} \leq f(x) - f(y) - \frac{1}{2} \lambda(1 - \alpha)\|x - y\|^2.$$

Taking the limit for $\alpha$ to 0, we obtain the statement.

Let's now assume that the inequality in the lemma holds and let's prove that $f$ is $\lambda$-strongly convex. Setting $v = \alpha x + (1 - \alpha)y$, for any $g \in \partial f(v)$, we have
$$\langle g, x - v \rangle \leq f(x) - f(v) - \frac{\lambda}{2} \|x - v\|^2,$$
$$\langle g, y - v \rangle \leq f(y) - f(v) - \frac{\lambda}{2} \|y - v\|^2.$$

Summing these two inequalities with coefficients $\alpha$ and $1 - \alpha$, we have
$$0 = \langle g, \alpha x - \alpha v + (1 - \alpha)y - (1 - \alpha)v \rangle \leq \alpha f(x) - \alpha f(v) + (1 - \alpha) f(y) - (1 - \alpha)f(v) - \alpha \frac{\lambda}{2} \|x - v\|^2 - (1 - \alpha) \frac{\lambda}{2} \|y - v\|^2$$
$$= \alpha f(x) - f(\alpha x + (1 - \alpha)y) + (1 - \alpha)f(y) - \alpha \frac{\lambda}{2} \|x - \alpha x - (1 - \alpha)y\|^2 - (1 - \alpha) \frac{\lambda}{2} \|y - \alpha x - (1 - \alpha)y\|^2$$
$$= \alpha f(x) - f(\alpha x + (1 - \alpha)y) + (1 - \alpha)f(y) - \alpha \frac{\lambda}{2} (1 - \alpha)^2 \|x - y\|^2 - (1 - \alpha) \frac{\lambda}{2} \alpha^2 \|x - y\|^2$$
$$= \alpha f(x) - f(\alpha x + (1 - \alpha)y) + (1 - \alpha)f(y) - \alpha(1 - \alpha) \frac{\lambda}{2} \|x - y\|^2.$$

In words, the lemma above tells us that a strongly convex function can be lower bounded by a quadratic, where the linear term is the usual one constructed through the subgradient, and the quadratic term depends on the strong convexity. Hence, we have a tighter lower bound to the function with respect to simply using convexity. This is what we would expect using a Taylor expansion on a twice-differentiable convex function and lower bounding the smallest eigenvalue of the Hessian. Indeed, we have the following Theorem.

Theorem 4.3. Let $f : \mathbb{R}^d \to (-\infty, +\infty]$ be proper and convex.
‚¬¢ $f$ is $\lambda$-strongly convex with respect to $\|\cdot\|$ in $\text{dom } \partial f$ iff
$$\langle g_x - g_y, x - y \rangle \geq \lambda \|x - y\|^2, \quad \forall g_x \in \partial f(x), g_y \in \partial f(y). \quad (4.1)$$
‚¬¢ Let $x, y \in \text{dom } f$. If $f$ is continuously twice differentiable between $y$ and $x$, and it holds that
$$\langle \nabla^2 f(y)x, x \rangle \geq \lambda \|x\|^2, \quad (4.2)$$
then we have
$$f(x) \geq f(y) + \langle \nabla f(y), x - y \rangle + \frac{\lambda}{2} \|x - y\|^2.$$
If the assumption holds for any $x, y \in \text{dom } f$, then $f$ is $\lambda$-strongly convex in $\text{dom } f$ with respect to $\|\cdot\|$.

Proof. For the first statement, first assume that $f$ is $\lambda$-strongly convex with respect to $\|\cdot\|$. Then, from Lemma 4.2, we have
$$\langle g_x, y - x \rangle \leq f(x) - f(y) - \frac{\lambda}{2} \|x - y\|^2, \quad \forall g_x \in \partial f(x),$$
$$\langle g_y, x - y \rangle \leq f(y) - f(x) - \frac{\lambda}{2} \|x - y\|^2, \quad \forall g_y \in \partial f(y).$$

Summing the two inequalities, we have the stated bound.
Now, assume that the (4.1) holds. Define $h(\alpha) = f(y + \alpha(x - y))$, $w_\alpha = y + \alpha(x - y)$, and $g_{w_\alpha} \in \partial f(w_\alpha)$. From Theorem 2.27, we have $\langle g_{w_\alpha}, x - y \rangle \in \partial h(\alpha)$. Moreover, we have
$$\langle g_{w_\alpha}, x - y \rangle - \langle g_y, x - y \rangle = \frac{1}{\alpha} \langle g_{w_\alpha} - g_y, w_\alpha - y \rangle \geq \frac{\lambda}{\alpha} \|w_\alpha - y\|^2 = \lambda \alpha \|x - y\|^2,$$
where we used (4.1) in the inequality. Using the Fundamental Theorem of Calculus (Theorem A.1) and this last inequality, we have
$$f(x) - f(y) - \langle g_y, x - y \rangle = h(1) - h(0) - \langle g_y, x - y \rangle = \int_0^1 (\langle g_{w_\alpha}, x - y \rangle - \langle g_y, x - y \rangle) d\alpha \geq \frac{\lambda}{2} \|x - y\|^2.$$

Hence, by Lemma 4.2, $f$ is $\lambda$-strongly convex with respect to $\|\cdot\|$.
For the second statement, assume that $f$ is twice differentiable and (4.2) holds. Then,
$$h''(\alpha) = \langle \nabla^2 f(y + \alpha(x - y))(x - y), x - y \rangle \geq \lambda \|x - y\|^2.$$
Moreover, from the Taylor's remainder theorem, we have that $h(1) = h(0) + h'(0) + h''(\beta)/2$, where $\beta \in [0, 1]$. So, we have
$$f(x) = h(1) = h(0) + h'(0) + h''(\beta)/2 \geq f(y) + \langle \nabla f(y), x - y \rangle + \frac{\lambda}{2} \|x - y\|^2,$$
which, by Lemma 4.2, implies that $f$ is $\lambda$-strongly convex with respect to $\|\cdot\|$.

Example 4.4. Let $f(x) = \frac{1}{2} \|x\|_2^2$. Using Theorem 4.3, we have that $f$ is 1-strongly convex with respect to $\|\cdot\|_2$ in $\mathbb{R}^d$.

However, a strongly convex function does not need to be twice differentiable. Indeed, we do not even need plain differentiability. Hence, the use of the subgradient implies that the quadratic lower bound does not have to be uniquely determined, as in the next example.

Example 4.5. Consider the strongly convex function $f(x) = |x| + x^2$. In Figure 4.1, we show two possible quadratic lower bounds to the function in $x = 0$.

We also have the following useful property on the sum of strongly convex functions.

Theorem 4.6. Let $f : \mathbb{R}^d \to (-\infty, +\infty]$ be $\mu_1$-strongly convex and $g : \mathbb{R}^d \to (-\infty, +\infty]$ a $\mu_2$-strongly convex function in a non-empty convex set $\mathcal{V} \subseteq \text{int dom } f \cap \text{int dom } g$ with respect to $\|\cdot\|$. Then, $f + g$ is $\mu_1 + \mu_2$-strongly convex in $\mathcal{V}$ with respect to $\|\cdot\|$.

Proof. From the assumption on $\mathcal{V}$ and Theorem 2.22, we have that the subdifferential set of the sum is equal to the sum of the subdifferential sets. Hence, the proof is immediate from Lemma 4.2.

### 4.1.2 Online Subgradient Descent for Strongly Convex Losses

Theorem 4.7. Let $\mathcal{V}$ be a non-empty closed convex set in $\mathbb{R}^d$. Assume that the functions $\ell_t : \mathcal{V} \to \mathbb{R}$ are $\mu_t$-strongly convex w.r.t $\|\cdot\|_2$ and subdifferentiable in $\mathcal{V}$, where $\mu_t > 0$. Use OSD in Algorithm 2.2 with stepsizes equal to $\eta_t = \frac{1}{\sum_{i=1}^t \mu_i}$. Then, for any $\mathbf{u} \in \mathcal{V}$, we have the following regret guarantee
$$\sum_{t=1}^T \ell_t(x_t) - \sum_{t=1}^T \ell_t(\mathbf{u}) \leq \frac{1}{2} \sum_{t=1}^T \frac{\|g_t\|_2^2}{\sum_{i=1}^t \mu_i}.$$

Proof. From the assumption of $\mu_t$-strong convexity of the functions $\ell_t$, we have that
$$\ell_t(x_t) - \ell_t(\mathbf{u}) \leq \langle g_t, x_t - \mathbf{u} \rangle - \frac{\mu_t}{2} \|x_t - \mathbf{u}\|_2^2.$$
From the fact that $\eta_t = \frac{1}{\sum_{i=1}^t \mu_i}$, we have
$$\frac{1}{2\eta_1} - \frac{\mu_1}{2} = 0, \quad \frac{1}{2\eta_t} - \frac{\mu_t}{2} = \frac{1}{2\eta_{t-1}}, \quad t = 2, \dots, T.$$
Hence, use Lemma 2.30 and sum from $t = 1, \dots, T$, to obtain
$$\sum_{t=1}^T (\ell_t(x_t) - \ell_t(\mathbf{u})) \leq \sum_{t=1}^T \left( \frac{1}{2\eta_t} \|x_t - \mathbf{u}\|_2^2 - \frac{1}{2\eta_t} \|x_{t+1} - \mathbf{u}\|_2^2 - \frac{\mu_t}{2} \|x_t - \mathbf{u}\|_2^2 + \frac{\eta_t}{2} \|g_t\|_2^2 \right) = -\frac{1}{2\eta_1} \|x_2 - \mathbf{u}\|_2^2 + \sum_{t=2}^T \left( \frac{1}{2\eta_{t-1}} \|x_t - \mathbf{u}\|_2^2 - \frac{1}{2\eta_t} \|x_{t+1} - \mathbf{u}\|_2^2 \right) + \sum_{t=1}^T \frac{\eta_t}{2} \|g_t\|_2^2.$$
Observing that the first sum on the right hand side is a telescopic sum, we have the stated bound.

Remark 4.8. Notice that this theorem implicitly requires a bounded domain, otherwise the loss functions will not be Lipschitz given that they are also strongly convex.

Corollary 4.9. Under the assumptions of Theorem 4.7, if in addiction we have $\mu_t = \mu > 0$ and $\ell_t$ is $L$-Lipschitz with respect to $\|\cdot\|_2$, for $t = 1, \dots, T$, then we have
$$\sum_{t=1}^T \ell_t(x_t) - \sum_{t=1}^T \ell_t(\mathbf{u}) \leq \frac{L^2}{2\mu} (1 + \ln T).$$

Remark 4.10. Corollary 4.9 does not imply that for any finite $T$ the regret will be smaller than using learning rates $\propto \frac{1}{L\sqrt{t}}$. Instead, asymptotically the regret upper bound in Corollary 4.9 is always better than to one of OSD with Lipschitz losses.

Example 4.11. Consider once again the example in the first chapter: $\ell_t(x) = (x - y_t)^2$. Note that the loss functions are $2$-strongly convex with respect to $|\cdot|$. Hence, setting $\eta_t = \frac{1}{2t}$ and $\ell'_t(x) = 2(x - y_t)$ gives a regret of $\ln(T) + 1$.

We can also use the online-to-batch conversion on strongly convex stochastic problems.

Example 4.12. As done before, we can use the online-to-batch conversion to use Corollary 4.9 to obtain stochastic subgradient descent algorithms for strongly convex stochastic functions. For example, consider the classic Support Vector Machine objective
$$\min_x F(x) := \frac{\lambda}{2} \|x\|_2^2 + \frac{1}{N} \sum_{i=1}^N \max(1 - y_i\langle z_i, x \rangle, 0),$$
or any other regularized formulation like regularized logistic regression:
$$\min_x F(x) := \frac{\lambda}{2} \|x\|_2^2 + \frac{1}{N} \sum_{i=1}^N \ln(1 + \exp(-y_i\langle z_i, x \rangle)),$$
where $z_i \in \mathbb{R}^d, \|z_i\|_2 \leq R$, and $y_i \in \{-1, 1\}$. First, notice that the minimizer of both expressions has to be in the $L_2$ ball of radius proportional to $\sqrt{1/\lambda}$ (proof left as exercise). Hence, we can set $\mathcal{V}$ equal to this set. Then, setting $\ell_t(x) = \frac{\lambda}{2} \|x\|_2^2 + \max(1 - y_i\langle z_i, x \rangle, 0)$ or $\ell_t(x) = \frac{\lambda}{2} \|x\|_2^2 + \ln(1 + \exp(-y_i\langle z_i, x \rangle))$ results in $\lambda$-strongly convex loss functions. Using Corollary 4.9 and Theorem 3.1 gives immediately
$$\mathbb{E} \left[ F\left( \frac{1}{T} \sum_{t=1}^T x_t \right) \right] - \min_x F(x) = \mathcal{O} \left( \frac{\ln T}{\lambda T} \right).$$

However, we can do better! We can use non-uniform weights in Theorem 3.1 to remove the log term and obtain the optimal convergence rate for the stochastic optimization of strongly convex functions. Observe that $\ell_t(x) = \frac{\lambda t}{2} \|x\|_2^2 + t \max(1 - y_i\langle z_i, x \rangle, 0)$ or $\ell_t(x) = \frac{\lambda t}{2} \|x\|_2^2 + t \ln(1 + \exp(-y_i\langle z_i, x \rangle))$ are $\lambda t$-strongly convex loss functions. So, using Theorem 4.7, we have that $\eta_t = \frac{2}{\lambda t(t+1)}$ and Theorem 3.1 gives immediately
$$\mathbb{E} \left[ F\left( \frac{2}{T(T + 1)} \sum_{t=1}^T t x_t \right) \right] - \min_x F(x) = \mathcal{O} \left( \frac{1}{\lambda T} \right),$$
that is asymptotically better because it does not have the logarithmic term.

## 4.2 Adaptive Algorithms: $L^\star$ bounds and AdaGrad

In this section, we will explore a bit more under which conditions we can get better regret upper bounds than $\mathcal{O}(DL \sqrt{T})$ as $T \to \infty$. Also, we will obtain this improved guarantees in an automatic way. That is, the algorithm will be adaptive to characteristics of the sequence of loss functions, without having to rely on information about the future.

### 4.2.1 Adaptive Learning Rates for Online Subgradient Descent

Consider the minimization of the regret with linear losses:
$$\text{Regret}_T(\mathbf{u}) = \sum_{t=1}^T \langle g_t, x_t \rangle - \sum_{t=1}^T \langle g_t, \mathbf{u} \rangle.$$

Using Online Subgradient Descent (OSD), in Chapter 2 we said that the regret for bounded domains can be upper bounded by
$$\sum_{t=1}^T \langle g_t, x_t \rangle - \sum_{t=1}^T \langle g_t, \mathbf{u} \rangle \leq \frac{D^2}{2\eta_T} + \frac{1}{2} \sum_{t=1}^T \eta_t \|g_t\|_2^2.$$
With a fixed learning rate $\eta_t = \eta$, the learning rate that minimizes this upper bound on the regret is
$$\eta^\star = \frac{D}{\sqrt{\sum_{t=1}^T \|g_t\|_2^2}}.$$

Unfortunately, as we said, this learning rate cannot be used because it assumes the knowledge of the future rounds. However, we might be lucky and we might try to just approximate it in each round using the knowledge up to time $t$. That is, we might try to use
$$\eta_t = \frac{D}{\sqrt{\sum_{i=1}^t \|g_i\|_2^2}}, \quad (4.3)$$
and just skip the rounds in which $g_i = 0$ to avoid possible divisions by 0. Observe that $\eta_T = \eta^\star$, so the first term of the regret would be exactly what we need! For the other term, the optimal learning rate would give us
$$\frac{1}{2} \sum_{t=1}^T \eta^\star_T \|g_t\|_2^2 = \frac{1}{2} D \sqrt{\sum_{t=1}^T \|g_t\|_2^2}.$$

Now, let's see what we obtain with our approximation in the other term of the regret:
$$\frac{1}{2} \sum_{t=1}^T \eta_t \|g_t\|_2^2 = \frac{1}{2} D \sum_{t=1}^T \frac{\|g_t\|_2^2}{\sqrt{\sum_{i=1}^t \|g_i\|_2^2}}.$$

We need a way to upper bound that sum. The way to treat these sums, as we did in other cases, is to try to approximate them with integrals. So, we can use the following very handy Lemma that generalizes a lot of similar specific ones.

Lemma 4.13. Let $a_0 \geq 0$ and $f : [0, +\infty) \to [0, +\infty)$ a non-increasing continuos function. Then
$$\sum_{t=1}^T a_t f\left( a_0 + \sum_{i=1}^t a_i \right) \leq \int_{a_0}^{\sum_{t=0}^T a_t} f(x) dx.$$

Proof. Denote by $s_t = \sum_{i=0}^t a_i$.
$$a_t f\left( a_0 + \sum_{i=1}^t a_i \right) = a_t f(s_t) = \int_{s_{t-1}}^{s_t} f(s_t) dx \leq \int_{s_{t-1}}^{s_t} f(x) dx.$$

Summing over $t = 1, \dots, T$, we have the stated bound.

Using this Lemma, we have that
$$\frac{1}{2} D \sum_{t=1}^T \frac{\|g_t\|_2^2}{\sqrt{\sum_{i=1}^t \|g_i\|_2^2}} \leq D \sqrt{\sum_{t=1}^T \|g_t\|_2^2}.$$

Surprisingly, this term is only a factor of 2 worse than what we would have got from the optimal choice of $\eta^\star$. However, this learning rate can be computed without knowledge of the future and it can actually be used! Overall, with this choice we get
$$\text{Regret}_T(\mathbf{u}) = \sum_{t=1}^T \ell_t(x_t) - \sum_{t=1}^T \ell_t(\mathbf{u}) \leq \frac{3}{2} D \sqrt{\sum_{t=1}^T \|g_t\|_2^2}. \quad (4.4)$$

Note that it is possible to improve the constant in front of the bound to $\sqrt{2}$ by multiplying the learning rates by $\frac{\sqrt{2}}{2}$. So, putting all together we have the following theorem.

Theorem 4.14. Let $\mathcal{V} \subset \mathbb{R}^d$ a closed non-empty convex set with diameter $D$, i.e., $\max_{x, y \in \mathcal{V}} \|x - y\|_2 \leq D$. Let $\ell_1, \dots, \ell_T$ an arbitrary sequence of convex functions $\ell_t : \mathcal{V} \to \mathbb{R}$ subdifferentiable in $\mathcal{V}$ for $t = 1, \dots, T$. Pick any $x_1 \in \mathcal{V}$, and run OSD with $\eta_t = \frac{\sqrt{2}D}{2\sqrt{\sum_{i=1}^t \|g_i\|_2^2}}$, $t = 1, \dots, T$, and do not update on rounds when $g_t = 0$. Then, $\forall \mathbf{u} \in \mathcal{V}$, the following regret bound holds
$$\text{Regret}_T(\mathbf{u}) = \sum_{t=1}^T (\ell_t(x_t) - \ell_t(\mathbf{u})) \leq \sqrt{2} D \sqrt{\sum_{t=1}^T \|g_t\|_2^2} = \sqrt{2} \min_{\eta > 0} \frac{D^2}{2\eta} + \frac{\eta}{2} \sum_{t=1}^T \|g_t\|_2^2.$$

The second equality in the theorem clearly show the advantage of this learning rates: We obtain (almost) the same guarantee we would have got knowing the future gradients!

This is an interesting result on its own: it gives a principled way to set the learning rates with an almost optimal guarantee. Observe that this approach avoid knowing the Lipschitz constants of the functions, so we adapt to it. The downside is that it works only in the bounded case. Another important observation is that the sum of the squared gradients act as an intrinsic notion of time, better suited than $T$ to capture the dependency on time.

There are also other consequences of this simple regret upper bound: We will now specialize this result to the case that the losses are smooth or self-bounded.

### 4.2.2 Convex Analysis Bits: Dual Norms, Smooth and Self-Bounded Functions

We now consider a family of loss functions that have the characteristic of being lower bounded by the squared norm of their subgradients. We will also introduce the concept of dual norms. While dual norms are not strictly needed for this topic, they give more generality and at the same time they allows me to slowly introduce some of the concepts that will be needed for the chapter on Online Mirror Descent.

Definition 4.15 (Dual Norm). The dual norm $\|\cdot\|_\star$ of a norm $\|\cdot\|$ is defined as $\|\theta\|_\star = \max_{x : \|x\| \leq 1} \langle \theta, x \rangle$.

Example 4.16. The dual norm of the $L_2$ norm is the $L_2$ norm. We can easily prove it. First of all, if $\theta = 0$, then the dual norm is 0 too. Hence, let's assume that $\theta \neq 0$. Indeed, $\|\theta\|_\star = \max_{x : \|x\|_2 \leq 1} \langle \theta, x \rangle \leq \|\theta\|_2$ by Cauchy-Schwarz inequality. Also, set $v = \frac{\theta}{\|\theta\|_2}$, so $\max_{x : \|x\|_2 \leq 1} \langle \theta, x \rangle \geq \langle \theta, v \rangle = \|\theta\|_2$.

Example 4.17. Let $p \geq 1$. The $L_p$ norm of a vector $x \in \mathbb{R}^d$ is defined as $\|x\|_p = (\sum_{i=1}^d |x_i|^p)^{1/p}$. The dual norm is the $q$-norm where $\frac{1}{p} + \frac{1}{q} = 1$. Note that the dual of the $L_1$ norm is the $L_\infty$ norm, defined as $\|x\|_\infty = \max_{i=1,\dots,d} |x_i|$. The proof is left to the reader.

Example 4.18. Let $A$ be a positive definite matrix, then it is possible to show that $\|x\|_A := \sqrt{x^\top A x}$ is a norm. The dual norm is $\|x\|_{A^{-1}} = \sqrt{x^\top A^{-1} x}$. In fact, we have that the dual norm of $\|\cdot\|_A$ is defined as
$$\|\theta\|_\star = \max_{x : x^\top Ax \leq 1} \theta^\top x = \max_{x : x^\top Ax \leq 1} \theta^\top A^{-1/2} y = \max_{y : \|y\|_2 \leq 1} (A^{-1/2} \theta)^\top y = \|A^{-1/2} \theta\|_2 = \sqrt{\theta^\top A^{-1} \theta},$$
where we have used the change of variable $y = A^{1/2}x$ in the third equality and the dual norm norm of the $L_2$ norm from Example 4.16 in the second to last equality.

If you do not know the concept of operator norms, the concept of dual norm can be a bit weird at first. One way to understand it is that it is a way to measure how "big" are linear functionals. For example, consider the linear function $f(x) = \langle z, x \rangle$, we want to try to understand how big it is. So, we can measure $\max_{x \neq 0} \frac{\langle z, x \rangle}{\|x\|}$ that is we measure how big is the output of the linear functional compared to its input $x$, where $x$ is measured with some norm. Now, you can show that the above is equivalent to the dual norm of $z$.

Remark 4.19. The definition of dual norm immediately implies $\langle \theta, x \rangle \leq \|\theta\|_\star \|x\|$.

We also have the following characterization of subgradient of norms.

Lemma 4.20. Let $f(x) = \|x\|$ a norm on $\mathbb{R}^d$, and $g_x \in \partial f(x)$. Then, $\|x\| = \langle g_x, x \rangle$ and if $x \neq 0$ then $\|g_x\|_\star = 1$.

Proof. The first statement is true for $x = 0$, hence in the following we can assume $x$ different from the zero vector. Define $h(x) = \|x\|$. Then, for all $g_x \in \partial h(x)$ by the definition of subgradient, we have
$$\|x\| \leq \langle g_x, x \rangle \leq \|g_x\|_\star \|x\|,$$
hence $\|g_x\|_\star \geq 1$.

On the other hand, using the inverse triangle inequality and the definition of subgradient, we have
$$\|x\| - \|y\| \leq \|x - y\| \leq \|x\| - \langle g_x, y \rangle.$$
This implies
$$\langle g_x, y \rangle \leq \|y\|.$$
Hence, we have
$$\|g_x\|_\star = \max_{\|y\| \leq 1} \langle g_x, y \rangle \leq \max_{\|y\| \leq 1} \|y\| = 1.$$
We conclude that $\|g_x\|_\star = 1$.

Now, using this in the first inequality, we have
$$\|x\| \leq \langle g, x \rangle \leq \|g_x\|_\star \|x\| = \|x\|.$$
Hence, $\|x\| = \langle g_x, x \rangle$.

Now we can introduce smooth functions, using the dual norms defined above.

Definition 4.21 (Smooth Function). Let $f : \mathcal{V} \to \mathbb{R}$ differentiable in an open set containing $\mathcal{V}$. We say that $f$ is $s$-smooth with respect to $\|\cdot\|$ if $\|\nabla f(x) - \nabla f(y)\|_\star \leq s\|x - y\|$ for all $x, y \in \mathcal{V}$.

Keeping in mind the intuition above on dual norms, taking the dual norm of a gradient makes sense if you associate each gradient with the linear functional $\langle \nabla f(y), x \rangle$, that is the one needed to create a linear approximation of $f$.

Remark 4.22. Note that smoothness does not imply convexity.

Smooth functions have many properties, for example a smooth function can be upper and lower bounded by a quadratic.

Lemma 4.23. Let $f : \mathcal{V} \to \mathbb{R}$ be $s$-smooth. Then, for any $x, y \in \mathcal{V}$, we have
$$|f(y) - f(x) - \langle \nabla f(x), y - x \rangle| \leq \frac{s}{2} \|y - x\|^2.$$

Proof. First, notice that by the definition of smoothness, $\nabla f : \mathcal{V} \to \mathbb{R}^d$ is Lipschitz and so continuous. Hence, by the fundamental theorem of calculus, we have
$$f(y) = f(x) + \int_0^1 \langle \nabla f(x + \tau(y - x)), y - x \rangle d\tau = f(x) + \langle \nabla f(x), y - x \rangle + \int_0^1 \langle \nabla f(x + \tau(y - x)) - \nabla f(x), y - x \rangle d\tau.$$
Therefore,
$$|f(y) - f(x) - \langle \nabla f(x), y - x \rangle| = \left| \int_0^1 \langle \nabla f(x + \tau(y - x)) - \nabla f(x), y - x \rangle d\tau \right| \leq \int_0^1 |\langle \nabla f(x + \tau(y - x)) - \nabla f(x), y - x \rangle| d\tau \leq \int_0^1 \|\nabla f(x + \tau(y - x)) - \nabla f(x)\|_\star \|y - x\| d\tau \leq \int_0^1 \tau s \|y - x\|^2 d\tau = \frac{s}{2} \|y - x\|^2.$$

In the following, we will need the following property.

Theorem 4.24. Let $f : \mathbb{R}^d \to \mathbb{R}$ be $s$-smooth and bounded from below, then for all $x \in \mathbb{R}^d$
$$\|\nabla f(x)\|_\star^2 \leq 2s(f(x) - \inf_{y \in \mathbb{R}^d} f(y)).$$

Proof. From Lemma 4.23, for any $x, v \in \mathbb{R}^d$, we have
$$\langle -\nabla f(x), v \rangle - \frac{s}{2} \|v\|^2 \leq f(x) - f(x + v) \leq f(x) - \inf_{y \in \mathbb{R}^d} f(y).$$
Given that this holds for any $v$, we can take the supremum of the left hand side with respect to $v$. Using Example 6.21, we have
$$\frac{1}{2s} \|\nabla f(x)\|_\star^2 = \sup_v \langle -\nabla f(x), v \rangle - \frac{s}{2} \|v\|^2 \leq f(x) - \inf_{y \in \mathbb{R}^d} f(y).$$

Sometimes we do not need smoothness nor differentiability but only the property of the above theorem. We call convex functions that satisfy such inequality self-bounded.

Definition 4.25 (Self-bounded Function). Let $f : \mathbb{R}^d \to \mathbb{R}$ bounded from below, and subdifferentiable in a set $\mathcal{V}$. We say that $f$ is $s$-self-bounded in $\mathcal{V}$ with respect to $\|\cdot\|$ if
$$\|g\|_\star^2 \leq 2s(f(x) - \inf_{y \in \mathbb{R}^d} f(y)), \quad \forall x \in \mathcal{V}, \forall g \in \partial f(x).$$

Remark 4.26. Self-bounded functions are also convex in $\mathcal{V}$ because we are assuming that they are subdifferentiable in $\mathcal{V}$.

Clearly, a convex $s$-smooth function is also $s$-self-bounded, but the converse is not true, as shown in the next example.

Example 4.27. Let $f : \mathbb{R} \to \mathbb{R}$ defined as $f(x) = \frac{1}{2}x^2 + |x - 1|$. The function $f$ is not differentiable in 1, hence it is not smooth. However, it is easy to verify that it is 4-self-bounded.

### 4.2.3 $L^\star$ bounds

We now introduce the $L^\star$ bounds, that depend on the cumulative competitor loss that is usually denoted by $L^\star$.
Assume now that the loss functions $\ell_1, \dots, \ell_T$ are bounded from below and $s$-self-bounded on $\mathcal{V}$. Without loss of generality, we can assume that each of them is bounded from below by 0. Under these assumptions, we can obtain bounds that depends on the cumulative loss of the competitor rather than time.

From the regret of Online Gradient Descent (OGD) in Theorem 2.13 and Definition 4.25, for a constant learning rate $\eta$ we obtain for any
$$\sum_{t=1}^T (\ell_t(x_t) - \ell_t(\mathbf{u})) \leq \frac{\|\mathbf{u} - x_1\|_2^2}{2\eta} + \eta \sum_{t=1}^T s \ell_t(x_t), \quad \mathbf{u} \in \mathcal{V}.$$

Reordering, it implies
$$\sum_{t=1}^T (\ell_t(x_t) - \ell_t(\mathbf{u})) \leq \frac{\eta s}{1 - \eta s} \sum_{t=1}^T \ell_t(\mathbf{u}) + \frac{\|\mathbf{u} - x_1\|_2^2}{2\eta(1 - \eta s)}, \quad \mathbf{u} \in \mathcal{V}.$$

Assuming $\eta \leq \frac{1}{2s}$, we simplify this regret upper bound in
$$\sum_{t=1}^T (\ell_t(x_t) - \ell_t(\mathbf{u})) \leq 2\eta s \sum_{t=1}^T \ell_t(\mathbf{u}) + \frac{\|\mathbf{u} - x_1\|_2^2}{\eta}, \quad \mathbf{u} \in \mathcal{V}.$$

This is already an interesting result because it guarantees that a fixed learning rate that depends only on $s$ can achieve a vanishing average regret if there exists a competitor $\mathbf{u} \in \mathcal{V}$ whose cumulative loss grows sublinearly. However, we could do better. In fact, for a fixed $\mathbf{u} \in \mathcal{V}$, setting $\eta = \min \left( \sqrt{\frac{1}{2s \sum_{t=1}^T \ell_t(\mathbf{u})}}, \frac{1}{2s} \right)$, we obtain
$$\sum_{t=1}^T (\ell_t(x_t) - \ell_t(\mathbf{u})) \leq (\|\mathbf{u} - x_1\|_2^2 + 1) \max \left( \sqrt{2s \sum_{t=1}^T \ell_t(\mathbf{u})}, 2s \right).$$

Comparing this bound to the one of OGD with Lipschitz losses, we see that here the dependency is on $\sqrt{\sum_{t=1}^T \ell_t(\mathbf{u})}$ instead of $\sqrt{T}$. The cumulative loss of the competitor can be much smaller than $T$ and in particular can be even 0. In this case, the regret is upper bounded by a constant. Moreover, in this latter case we can afford to use a learning rate $\eta$ that depends only on the self-boundedness constant. However, this result is partially interesting because it requires the knowledge of the future through the cumulative loss of the competitor. In the following, we show how to easily get rid of this limitation with a different choice of the learning rate.

In fact, under the same assumptions on the losses, from the regret in Theorem 4.14 and Theorem 4.24 we immediately obtain
$$\text{Regret}_T(\mathbf{u}) = \sum_{t=1}^T \ell_t(x_t) - \sum_{t=1}^T \ell_t(\mathbf{u}) \leq 2D \sqrt{s \sum_{t=1}^T \ell_t(x_t)},$$
where $D$ is the diameter of $\mathcal{V}$, assumed to be bounded. This is an implicit bound, in the sense that $\sum_{t=1}^T \ell_t(x_t)$ appears on both sides of the inequality. To makes it explicit, we will use the following simple Lemma (proof left as an exercise).

Lemma 4.28. Let $a, c > 0, b \geq 0$, and $x \geq 0$ such that $x - \sqrt{ax + b} \leq c$. Then $x \leq a + c + 2\sqrt{b + ac}$.

So, we have the following theorem.

Theorem 4.29. Let $\mathcal{V} \subset \mathbb{R}^d$ a closed non-empty convex set with diameter $D$, i.e., $\max_{x, y \in \mathcal{V}} \|x - y\|_2 \leq D$. Let $\ell_1, \dots, \ell_T$ an arbitrary sequence of non-negative convex functions $\ell_t : \mathbb{R}^d \to \mathbb{R}$ $s$-self-bounded in $\mathbb{R}^d$ for $t = 1, \dots, T$. Pick any $x_1 \in \mathcal{V}$, run projected OGD with $\eta_t = \frac{\sqrt{2}D}{2\sqrt{\sum_{i=1}^t \|g_i\|_2^2}}$, $t = 1, \dots, T$, and do not update on rounds in which $g_t = 0$. Then, $\forall \mathbf{u} \in \mathcal{V}$, the following regret bound holds
$$\text{Regret}_T(\mathbf{u}) = \sum_{t=1}^T \ell_t(x_t) - \sum_{t=1}^T \ell_t(\mathbf{u}) \leq 4sD^2 + 4D \sqrt{s \sum_{t=1}^T \ell_t(\mathbf{u})}.$$

This regret guarantee is very interesting because in the worst case it is still of the order $\mathcal{O}(\sqrt{T})$, but in the best case scenario it becomes a constant! In fact, if there exists a $\mathbf{u} \in \mathcal{V}$ such that $\sum_{t=1}^T \ell_t(\mathbf{u}) = 0$ we get a constant regret. Basically, if the losses are "easy", the algorithm adapts to this situation and gives us a better regret.

Example 4.30. Consider an online linear classification problem with the squared hinge loss. So, each loss is defined as $\ell_t(x) = \max(1 - y_t \langle z_t, x \rangle, 0)^2$ for labels $y_t \in \{-1, 1\}$ and features $z_t \in \mathbb{R}^d$. If we assume that $\|z_t\|_2$ is bounded for all $t$, then the losses are self-bounded (and even smooth) (proof left as an exercise). Under this assumption, if the problem is linearly separable in $\mathcal{V}$, i.e., there exists $\mathbf{u} \in \mathcal{V}$ such that $\ell_t(\mathbf{u}) = 0$ for all $t$, then $L^\star$ is equal to 0 and Theorem 4.29 will guarantee a constant regret.

### 4.2.4 AdaGrad

We now present another application of the regret bound in (4.4). AdaGrad, that stands for Adaptive Gradient, is an Online Convex Optimization (OCO) algorithm proposed independently by McMahan and Streeter [2010] and Duchi et al. [2010]. It aims at being adaptive to the sequence of gradients. It is usually known as a stochastic optimization algorithm, but in reality it was proposed for the online setting. To use it as a stochastic algorithm, you should use an online-to-batch conversion, otherwise you do not have any guarantee of convergence.

We will present a proof that only allows hyperrectangles as feasible sets $\mathcal{V}$, on the other hand the restriction makes the proof almost trivial. Let's see how it works.

AdaGrad has key ingredients:
‚¬¢ A coordinate-wise learning process;
‚¬¢ The adaptive learning rates in (4.3).

For the first ingredient, as we said, the regret of any OCO problem can be upper bounded by the regret of the Online Linear Optimization (OLO) problem. That is,
$$\sum_{t=1}^T \ell_t(x_t) - \sum_{t=1}^T \ell_t(\mathbf{u}) \leq \sum_{t=1}^T \langle g_t, x_t \rangle - \sum_{t=1}^T \langle g_t, \mathbf{u} \rangle.$$

Now, the essential observation is to explicitly write the inner product as a sum of product over the single coordinates:
$$\sum_{t=1}^T \langle g_t, x_t \rangle - \sum_{t=1}^T \langle g_t, \mathbf{u} \rangle = \sum_{t=1}^T \left( \sum_{i=1}^d g_{t,i} x_{t,i} - \sum_{i=1}^d g_{t,i} u_i \right) = \sum_{i=1}^d \left( \sum_{t=1}^T g_{t,i} x_{t,i} - \sum_{t=1}^T g_{t,i} u_i \right) = \sum_{i=1}^d \text{Regret}_{T,i}(u_i),$$
where we denoted by $\text{Regret}_{T,i}(u_i)$ the regret of the 1-dimensional Online Linear Optimization (OLO) problem over coordinate $i$, that is $\sum_{t=1}^T g_{t,i} x_{t,i} - \sum_{t=1}^T g_{t,i} u_i$. In words, we can decompose the original regret as the sum of $d$ OLO regret minimization problems and we can try to focus on each one of them separately.

A good candidate for the 1-dimensional problems is OSD with the learning rates in (4.3). We can specialize the regret in (4.4) to the 1-dimensional case for linear losses, so we get for each coordinate $i$
$$\sum_{t=1}^T g_{t,i} x_{t,i} - \sum_{t=1}^T g_{t,i} u_i \leq \sqrt{2} D_i \sqrt{\sum_{t=1}^T g_{t,i}^2}.$$

This choice gives us the AdaGrad algorithm in Algorithm 4.1.

Algorithm 4.1 AdaGrad for Hyperrectangles
Require: $\mathcal{V} = \{x \in \mathbb{R}^d : a_i \leq x_i \leq b_i\}, x_1 \in \mathcal{V}$
1: for $t = 1$ to $T$ do
2: Output $x_t$
3: Pay $\ell_t(x_t)$ for a $\ell_t : \mathcal{V} \to \mathbb{R}$ subdifferentiable in $\mathcal{V}$
4: Set $g_t \in \partial \ell_t(x_t)$
5: for $i = 1$ to $d$ do
6: if $g_{t,i} \neq 0$ then
7: Set $\eta_{t,i} = \frac{\sqrt{2}D_i}{2\sqrt{\sum_{j=1}^t g_{j,i}^2}}$
8: $x_{t+1,i} = \max(\min(x_{t,i} - \eta_{t,i} g_{t,i}, b_i), a_i)$
9: else
10: $x_{t+1,i} = x_{t,i}$
11: end if
12: end for
13: end for

Putting all together, we have immediately the following regret guarantee.

Theorem 4.31. Let $\mathcal{V} = \{x \in \mathbb{R}^d : a_i \leq x_i \leq b_i\}$ with diameters along each coordinate equal to $D_i = b_i - a_i$. Let $\ell_1, \dots, \ell_T$ an arbitrary sequence of convex functions $\ell_t : \mathcal{V} \to \mathbb{R}$ subdifferentiable in $\mathcal{V}$ for $t = 1, \dots, T$. Pick any $x_1 \in \mathcal{V}$ and $\eta_{t,i} = \frac{\sqrt{2}D_i}{2\sqrt{\sum_{j=1}^t g_{j,i}^2}}, t = 1, \dots, T$. Then, $\forall \mathbf{u} \in \mathcal{V}$, Algorithm 4.1 guarantees
$$\text{Regret}_T(\mathbf{u}) = \sum_{t=1}^T \ell_t(x_t) - \sum_{t=1}^T \ell_t(\mathbf{u}) \leq \sqrt{2} \sum_{i=1}^d D_i \sqrt{\sum_{t=1}^T g_{t,i}^2}.$$

Is this a better regret bound compared to the one in Theorem 4.14? It depends! To compare the two, let's first consider the case that $\mathcal{V}$ is a hyperrectangle. Then, we have to compare
$$D \sqrt{\sum_{t=1}^T \|g_t\|_2^2} \quad \text{versus} \quad \sum_{i=1}^d D_i \sqrt{\sum_{t=1}^T g_{t,i}^2}.$$

From Cauchy-Schwarz, we have that $\sum_{i=1}^d D_i \sqrt{\sum_{t=1}^T g_{t,i}^2} \leq D \sqrt{\sum_{t=1}^T \|g_t\|_2^2}$. So, assuming the same sequence of subgradients, AdaGrad has never a worse regret on hyperrectangles. For a more precise quantification of the gain of AdaGrad, let's now assume that $\mathcal{V}$ is a hypersquare. Also, note that
$$\sqrt{\sum_{t=1}^T \|g_t\|_2^2} \leq \sum_{i=1}^d \sqrt{\sum_{t=1}^T g_{t,i}^2} \leq \sqrt{d} \sqrt{\sum_{t=1}^T \|g_t\|_2^2}, \quad (4.5)$$
where the lower bound is by the fact that the $L_1$ norm is bigger than the $L_2$ norm, and the upper bound is given by Cauchy-Schwarz. So, in the case that $\mathcal{V}$ is a hypercube we have $D_i = D_\infty = \max_{x, y} \|x - y\|_\infty$ and $D = \sqrt{d}D_\infty$, the bound of AdaGrad is between $1/\sqrt{d}$ and 1 times the bound of Theorem 4.14. In other words, if we are lucky with the subgradients, we might save us a factor of $\sqrt{d}$ in the guarantee.

However, what does happen if the domain is a $L_2$ ball? First of all, it is possible to generalize AdaGrad to work on $L_2$ balls and the guarantee remains the same. Hence, in this case, we have $D_i = D_\infty = D$, so from (4.5), we have
$$\frac{1}{\sqrt{d}} \sum_{i=1}^d D_i \sqrt{\sum_{t=1}^T g_{t,i}^2} \leq D \sqrt{\sum_{t=1}^T \|g_t\|_2^2} \leq \sum_{i=1}^d D_i \sqrt{\sum_{t=1}^T g_{t,i}^2}.$$

Hence, for an $L_2$ ball the opposite happens: the bound in Theorem 4.14 is never worse than the one of AdaGrad and, depending on the subgradients, we can gain a $\sqrt{d}$ factor. Overall, the shape of the domain determines the potential gain of one approach over the other and the specific sequence of subgradients determines the actual gain. It is possible to show that hyperrectangles are indeed the best domains for AdaGrad. We will explore this issue of choosing the online algorithm based on the shape of the feasible set $\mathcal{V}$ when we will introduce Online Mirror Descent in Chapter 6.

Another big advantage of AdaGrad is the property of being coordinate-wise scale-free. That is, if each coordinate of the gradients are multiplied by different constants, the learning rate will automatically scale them back. Another way to say it is that the update of AdaGrad is invariant to the units of each coordinate of the subgradients. This fact is not immediately apparent from the regret because by scaling the coordinate of the gradients the optimal solution $\mathbf{u}$ would also scale accordingly, but the fixed diameters of the feasible set hide it. This might be useful in the case the ranges of coordinates of the gradients are vastly different one from the other. Indeed, this does happen in many machine learning problems, for example, in the stochastic optimization of deep neural networks, where the first layers have different magnitude of the gradients compared to the last layers.

### 4.3 History Bits

The concept of strong convexity is defined for the first time in Polyak [1966].
The logarithmic regret in Corollary 4.9 was shown for the first time in the seminal paper Hazan et al. [2006, 2007]. The general statement in Theorem 4.7 was proven by Hazan et al. [2008].

The non-uniform averaging for the online-to-batch conversion of Example 4.12 is from Lacoste-Julien et al. [2012], but there it is not proposed as an online-to-batch conversion. The basic idea of solving the Support Vector Machine (SVM) problem with OSD and online-to-batch conversion of Example 4.12 was the Pegasos algorithm [Shalev-Shwartz et al., 2007], for many years the most used optimizer for SVMs.

The adaptive learning rate in (4.3) first appeared in Streeter and McMahan [2010]. However, similar methods were used long time before. Indeed, the key observation to approximate oracle quantities with estimates up to time $t$ was first proposed in the self-confident algorithms [Auer et al., 2002c], where the learning rate is inversely proportional to the square root of the cumulative loss of the algorithm, and for self-bounded losses it implies the $L^\star$ bounds similar to the one in Theorem 4.29. The $L^\star$ bound for the square loss and linear predictors was introduced by Cesa-Bianchi et al. [1996]. In the past, several work focused on obtaining regret upper bounds depending on constant times $L^\star$ [see, e.g., Kivinen and Warmuth, 1997], however these guarantees are meaningful only if $L^\star$ is sublinear in $T$. Zhang [2004] explored the use of $L^\star$ bounds in stochastic optimization. The observation that the cumulative sum of the squared gradients act as an intrinsic notion of time comes from the statistics literature, see the discussion in Blackwell and Freedman [1973].

AdaGrad was proposed in basically identically form independently by two groups at the same conference: McMahan and Streeter [2010] and Duchi et al. [2010]. The analysis presented here is the one in Streeter and McMahan [2010] that does not handle generic feasible sets and does not support the "full-matrices" proposed in Duchi et al. [2010], i.e., full-matrix learning rates instead of diagonal ones. However, in machine learning applications AdaGrad is rarely used with a projection step (even if doing so provably destroys the worst-case performance [Orabona and Pál, 2018]). Also, in the adversarial setting full-matrices do not seem to offer advantages in terms of regret compared to diagonal ones, see the discussion in Cutkosky [2020b, Section 5].

Note that the AdaGrad learning rate is usually written as
$$\eta_{t,i} = \frac{D_i}{\epsilon + \sqrt{\sum_{j=1}^t g_{j,i}^2}},$$
where $\epsilon > 0$ is a small constant used to prevent division by zero. In reality, $\epsilon$ is not necessary: there should be no update when the coordinate of the gradient is 0 [Orabona and Pál, 2015, 2018, Agarwal et al., 2020]. Moreover, removing $\epsilon$ makes the updates scale-free, as stressed in Orabona and Pál [2015, 2018]. Scale-freeness in online learning has been introduced in Cesa-Bianchi et al. [2005, 2007] for the setting of Learning with Expert Advice (LEA) and in Orabona and Pál [2015, 2018] for OCO.

AdaGrad inspired an incredible number of clones, most of them with similar, worse, or no regret guarantees. The keyword "adaptive" itself has shifted its meaning over time. It used to denote the ability of the algorithm to obtain the same guarantee as it knew in advance a particular property of the data (i.e., adaptive to the gradients/noise/scale = (almost) same performance as it knew the gradients/noise/scale in advance). Indeed, in Statistics this keyword is used with the same meaning. Nowadays, instead "adaptive" seems to denote any kind of coordinate-wise learning rates that does not guarantee anything in particular.

### 4.4 Exercises

Problem 4.1. Prove that OSD in Example 4.11 with $x_1 = 0$ is exactly the Follow-the-Leader strategy for that particular problem.

Problem 4.2. Prove that $\ell_t(x) = \|x - z_t\|_2^2$ is 2-strongly convex with respect to $\|\cdot\|_2$, derive the OSD update for it, and its regret guarantee.

Problem 4.3. Prove that the dual norm of $\|\cdot\|_p$ is $\|\cdot\|_q$, where $\frac{1}{p} + \frac{1}{q} = 1$ and $p, q \geq 1$.

Problem 4.4. Show that using online subgradient descent on a bounded domain $\mathcal{V}$ with the learning rates $\eta_t = \mathcal{O}(1/t)$ with Lipschitz, self-bounded, and strongly convex functions you can get $\mathcal{O}(\ln(1 + L^\star))$ bounds.

Problem 4.5. Prove that the logistic loss $\ell(x) = \ln(1 + \exp(-y\langle z, x \rangle))$, where $\|z\|_2 \leq 1$ and $y \in \{-1, 1\}$ is $\frac{1}{4}$-smooth with respect to $\|\cdot\|_2$.

# Chapter 5

# Lower bounds for Online Linear
Optimization

In this chapter we will present some lower bounds for Online Linear Optimization (OLO). Remembering that linear losses are convex, this immediately gives us lower bounds for Online Convex Optimization (OCO). We will consider both the constrained and the unconstrained case. The lower bounds are important because they inform us on what are the optimal algorithms and where are the gaps in our knowledge.

## 5.1 Lower Bounds for Bounded Online Linear Optimization (OLO)

We will first consider the bounded constrained case. Finding a lower bound accounts to find a strategy for the adversary that forces a certain regret onto the algorithm, no matter what the algorithm does. We will use the probabilistic method to construct our lower bound.

The basic method relies on the fact that if for a given $K \in \mathbb{R}$ we can construct a sequence of random vectors $\tilde{g}_1, \dots, \tilde{g}_T$ such that
$$\mathbb{E} \left[ \sum_{t=1}^T \langle \tilde{g}_t, x_t \rangle \right] \geq K,$$
this implies that there exists a sequence $g_1, \dots, g_T$ among all the possible random sequences such that
$$\sum_{t=1}^T \langle g_t, x_t \rangle \geq K.$$

It is easy to convice oneself that this is true: if for all sequences we would have $\sum_{t=1}^T \langle \tilde{g}_t, x_t \rangle < K$, then the expectation should also be strictly less than $K$, contradicting our assumption.

For us, it means that we prove the existence of "difficult" sequence of functions through a result on the expectation with respect to a distribution over stochastic functions. Why do you rely on expectations rather than actually constructing an adversarial sequence? Because the use of stochastic loss functions makes very easy to deal with arbitrary algorithms. In particular, we will choose a distribution over stochastic loss functions that makes the expected loss of the algorithm equal to 0, independently from the strategy of the algorithm.

Theorem 5.1. Let $\mathcal{V} \subset \mathbb{R}^d$ be any non-empty bounded closed convex subset. Let $D = \sup_{v, w \in \mathcal{V}} \|v - w\|_2$ be the diameter of $\mathcal{V}$. Let $\mathcal{A}$ be any (possibly randomized) algorithm for OLO on $\mathcal{V}$. Let $T$ be any non-negative integer. Then, there exists a sequence of vectors $g_1, \dots, g_T$ with $\|g_t\|_2 \leq L$ and $\mathbf{u} \in \mathcal{V}$ such that the regret of algorithm $\mathcal{A}$ satisfies
$$\text{Regret}_T(\mathbf{u}) = \sum_{t=1}^T \langle g_t, x_t \rangle - \sum_{t=1}^T \langle g_t, \mathbf{u} \rangle \geq \frac{\sqrt{2}LD\sqrt{T}}{4}.$$

Proof. Let's denote by $\text{Regret}_T = \max_{\mathbf{u} \in \mathcal{V}} \text{Regret}_T(\mathbf{u})$. Let $v, w \in \mathcal{V}$ such that $\|v - w\|_2 = D$. Let $z = \frac{v - w}{\|v - w\|_2}$, so that $\langle z, v - w \rangle = D$. Let $\epsilon_1, \dots, \epsilon_T$ be i.i.d. Rademacher random variables, that is $\mathbb{P}\{\epsilon_t = 1\} = \mathbb{P}\{\epsilon_t = -1\} = 1/2$ and set the vector of the stochastic linear losses $\tilde{g}_t = L \epsilon_t z$.

So, we have
$$\mathbb{E}_{\tilde{g}_1, \dots, \tilde{g}_T} \left[ \sum_{t=1}^T \langle \tilde{g}_t, x_t \rangle - \min_{\mathbf{u} \in \mathcal{V}} \sum_{t=1}^T \langle \tilde{g}_t, \mathbf{u} \rangle \right] = \mathbb{E}_{\epsilon_1, \dots, \epsilon_T} \left[ \sum_{t=1}^T L \epsilon_t \langle z, x_t \rangle - \min_{\mathbf{u} \in \mathcal{V}} \sum_{t=1}^T L \epsilon_t \langle z, \mathbf{u} \rangle \right] = \mathbb{E}_{\epsilon_1, \dots, \epsilon_T} \left[ - \min_{\mathbf{u} \in \mathcal{V}} \sum_{t=1}^T L \epsilon_t \langle z, \mathbf{u} \rangle \right] = \mathbb{E}_{\epsilon_1, \dots, \epsilon_T} \left[ \max_{\mathbf{u} \in \mathcal{V}} \sum_{t=1}^T -L \epsilon_t \langle z, \mathbf{u} \rangle \right] = \mathbb{E}_{\epsilon_1, \dots, \epsilon_T} \left[ \max_{\mathbf{u} \in \mathcal{V}} \sum_{t=1}^T L \epsilon_t \langle z, \mathbf{u} \rangle \right] \geq \mathbb{E}_{\epsilon_1, \dots, \epsilon_T} \left[ \max_{\mathbf{u} \in \{v, w\}} \sum_{t=1}^T L \epsilon_t \langle z, \mathbf{u} \rangle \right] = \mathbb{E}_{\epsilon_1, \dots, \epsilon_T} \left[ \frac{1}{2} \sum_{t=1}^T L \epsilon_t \langle z, v + w \rangle + \frac{1}{2} \left| \sum_{t=1}^T L \epsilon_t \langle z, v - w \rangle \right| \right] = \frac{L}{2} \mathbb{E}_{\epsilon_1, \dots, \epsilon_T} \left[ \left| \sum_{t=1}^T \epsilon_t \langle z, v - w \rangle \right| \right] = \frac{LD}{2} \mathbb{E}_{\epsilon_1, \dots, \epsilon_T} \left[ \left| \sum_{t=1}^T \epsilon_t \right| \right] \geq \frac{\sqrt{2}LD\sqrt{T}}{4}.$$

where in the first equality we used $\mathbb{E}[\epsilon_t] = 0$ and the independence of $\epsilon_t$ and $x_t$, the fact that $\epsilon_t$ and $-\epsilon_t$ follow the same distribution in the fourth equality, $\max(a, b) = \frac{a+b}{2} + \frac{|a-b|}{2}$ in the fifth equality, and Khintchine inequality (Theorem B.1) in the last inequality.

Now, given that the expectation is lower bound by a positive constant, there exists a sequence of realizations of the random variables that gives the same lower bound.

Remark 5.2. Differently from similar proofs, in the above proof we do not assume that $\mathcal{V}$ is symmetric with respect to 0.

We see that the lower bound is a constant multiplicative factor from the upper bound we proved for Online Subgradient Descent (OSD) with learning rates $\eta_t = \frac{D}{L\sqrt{t}}$ or $\eta = \frac{D}{L\sqrt{T}}$. This means that Online Subgradient Descent (OSD) is asymptotically optimal with both settings of the learning rate.

At this point there is an important consideration to do: How can this be the optimal regret when we managed to prove a better regret, for example, with adaptive learning rates in Section 4.2? The subtlety is that, constraining the adversary to play $L$-Lipschitz losses, the adversary could always force on the algorithm at least the regret in Theorem 5.1. However, we can design algorithms that take advantage of suboptimal plays of the adversary. Indeed, for example, if the adversary plays in a way that all the subgradients have the same norm equal to $L$, there is nothing to adapt to!

## 5.2 Lower Bound for Unconstrained Online Subgradient Descent

Here, we will focus on a specific algorithm, that is, Online Subgradient Descent (OSD). We want to show that the limitation of OSD with time-varying stepsizes to be used only on bounded domains is real. In fact, we can prove the following lower bound.

Theorem 5.3. Let $\alpha \in (0, 1)$, $\phi : (0, 1) \to (0, 1 - \ln 2)$ defined as $\phi(\alpha) := \frac{1}{2-\alpha} + \frac{(1/2)^{1-\alpha-1}}{1-\alpha}$, and $T \geq \frac{2}{(1-\alpha)f(\alpha)}$. For unprojected OSD with stepsizes $\eta_t = t^{-\alpha}$ and $x_1 = 0$, there exists a sequence of $T$ convex and 1-Lipschitz losses such that
$$\text{Regret}_T(0) \geq \frac{1}{2} \phi(\alpha) T^{2-\alpha}.$$

Also, we have that $\lim_{\alpha \to 1} \phi(\alpha) = 1 - \ln 2 \geq 0.3$.

Proof. We assume $d = 1$. For $d \geq 2$, we simply embed the one-dimensional loss vectors into the first coordinate of $\mathbb{R}^d$. Note that the condition on $T$ implies $T \geq 2$. Consider the sequence
$$(\ell_1(x), \dots, \ell_T(x)) = (\underbrace{-x, \dots, -x}_{\lceil T/2 \rceil}, \underbrace{x, \dots, x}_{\lfloor T/2 \rfloor}).$$

That is, the first half consists of $-x$'s, the second of $+x$'s. For $t \leq \lceil T/2 \rceil$, we have $x_{t+1} = x_t + \frac{1}{\sqrt{t}}$. Unrolling the recurrence and using $x_1 = 0$ we get
$$x_t = \sum_{i=1}^{t-1} \frac{1}{\sqrt{i}}, \quad t \leq \lceil T/2 \rceil + 1.$$

On the other hand, for $t \geq \lceil T/2 \rceil + 1$, we have $x_{t+1} = x_t - t^{-\alpha}$. Unrolling the recurrence up to $x_{\lceil T/2 \rceil+1}$ we get
$$x_t = x_{\lceil T/2 \rceil+1} - \sum_{i=\lceil T/2 \rceil+1}^{t-1} i^{-\alpha} = \sum_{i=1}^{\lceil T/2 \rceil} i^{-\alpha} - \sum_{i=\lceil T/2 \rceil+1}^{t-1} i^{-\alpha}, \quad t \geq \lceil T/2 \rceil + 1.$$

We are ready to lower bound the regret:
$$\text{Regret}_T(0) = -\sum_{t=1}^{\lceil T/2 \rceil} x_t + \sum_{t=\lceil T/2 \rceil+1}^T x_t = -\sum_{t=1}^{\lceil T/2 \rceil} \sum_{i=1}^{t-1} i^{-\alpha} + \sum_{t=\lceil T/2 \rceil+1}^T \left( \sum_{i=1}^{\lceil T/2 \rceil} i^{-\alpha} - \sum_{i=\lceil T/2 \rceil+1}^{t-1} i^{-\alpha} \right) = -\sum_{i=1}^{\lceil T/2 \rceil} \frac{\lceil T/2 \rceil - i}{i^\alpha} + \lfloor T/2 \rfloor \sum_{i=1}^{\lceil T/2 \rceil} i^{-\alpha} - \sum_{t=\lceil T/2 \rceil+1}^T \frac{T - i}{i^\alpha} = -\sum_{i=1}^{\lceil T/2 \rceil} \frac{\lceil T/2 \rceil - \lfloor T/2 \rfloor}{i^\alpha} + \sum_{i=1}^T i^{1-\alpha} - T \sum_{i=\lceil T/2 \rceil+1}^T i^{-\alpha} \geq -\sum_{i=1}^{\lceil T/2 \rceil} i^{-\alpha} + \sum_{i=1}^T i^{1-\alpha} - T \sum_{i=\lceil T/2 \rceil+1}^T i^{-\alpha} \geq -1 - \int_1^{\lceil T/2 \rceil} x^{-\alpha} dx + \int_0^T x^{1-\alpha} dx - T \int_{\lceil T/2 \rceil}^T x^{-\alpha} dx = -1 - \frac{\lceil T/2 \rceil^{1-\alpha} - 1}{1 - \alpha} + \frac{T^{2-\alpha}}{2 - \alpha} - T \frac{T^{1-\alpha} - \lceil T/2 \rceil^{1-\alpha}}{1 - \alpha} \geq -1 + \frac{1}{1 - \alpha} - \frac{\lceil T/2 \rceil^{1-\alpha}}{1 - \alpha} + \frac{T^{2-\alpha}}{2 - \alpha} - \frac{T^{2-\alpha} - T(\lceil T/2 \rceil)^{1-\alpha}}{1 - \alpha} \geq -\frac{T^{1-\alpha}}{1 - \alpha} + \left( \frac{1}{2 - \alpha} + \frac{(1/2)^{1-\alpha}}{1 - \alpha} - \frac{1}{1 - \alpha} \right) T^{2-\alpha} = -\frac{T^{1-\alpha}}{1 - \alpha} + \phi(\alpha) T^{2-\alpha} \geq \frac{1}{2} \phi(\alpha) T^{2-\alpha}.$$

This lower bound tells us that OSD does fail in unbounded domains when used with a polynomially decreasing stepsize. However, it does not rule out the possibility for another algorithm to work in the same setting. Indeed, we will see that in Chapter 7 that Follow-the-Regularized-Leader achieves sublinear regret on unbounded domains with a time-varying regularizers. Yet, its dependency on the other quantities will be still suboptimal and we will obtain the optimal bound only with parameter-free algorithms in Chapter 10. Indeed, in the next section, we prove that unconstrained OLO is actually more difficult than OLO in bounded domains, for any algorithm.

## 5.3 Lower Bounds for Unconstrained OLO

The previous lower bound applies only to the constrained setting. In the unconstrained setting, we proved that OSD with $x_1 = 0$ and constant learning rate of $\eta = \frac{1}{L\sqrt{T}}$ gives a regret of $\frac{1}{2} L(\|u\|_2^2 + 1) \sqrt{T}$ for any $\mathbf{u} \in \mathbb{R}^d$. Is this regret optimal? It is clear that the regret must be at least linear in $\|\mathbf{u}\|_2$. In fact, we could select a specific $\mathbf{u}$, pass the information of $\|\mathbf{u}\|_2$ to the online learning algorithm and make the problem constrained in $\mathcal{V} = \{x : \|x\|_2 \leq \|\mathbf{u}\|_2\}$, so that the lower bound for bounded OLO would hold. However, we now show that the correct dependency in $\|\mathbf{u}\|_2$ is more than linear, so the unconstrained setting is strictly more difficult than the bounded one.

The approach I will follow is to reduce the OLO game to the online game of betting on a coin, where the lower bounds are easier to prove. So, let's introduce the coin-betting online game:
‚¬¢ Start with an initial amount of money $\epsilon > 0$.
‚¬¢ In each round, the algorithm bets a fraction of its current wealth on the outcome of a coin.
‚¬¢ The outcome of the coin is revealed and the algorithm wins or loses its bet, 1 to 1.

The aim of this online game is to win as much money as possible. Also, as in all the online games we consider, we do not assume anything on how the outcomes of the coin are decided. Note that this game can also be written as Online Convex Optimization (OCO) using the log loss.

We will denote by $c_t \in \{-1, 1\}, t = 1, \dots, T$ the outcomes of the coin. We will use the absolute value of $\beta_t \in [-1, 1]$ to denote the fraction of money to bet and its sign to denote on which side we are betting. The money the algorithm has won from the beginning of the game till the end of round $t$ will be denoted by $r_t$ and given that the money are won or lost 1 to 1, we have
$$\overbrace{r_t + \epsilon}^{\text{Money at the end of round } t} = \overbrace{r_{t-1} + \epsilon}^{\text{Money at the beginning of round } t} + \overbrace{c_t \beta_t (r_{t-1} + \epsilon)}^{\text{Money won or lost}} = \epsilon \prod_{i=1}^t (1 + \beta_i c_i),$$
where we used the fact that $r_0 = 0$. We will also denote by $x_t = \beta_t(\epsilon + r_{t-1})$ the bet of the algorithm on round $t$.

If we got all the outcomes of the coin correct, we would double our money in each round, so that $\epsilon + r_T = \epsilon 2^T$. On the other hand, if the adversary can always select a coin outcome that is the opposite of our bet. Hence, we are interested in the best any algorithm can do when the adversary is constrained to give us a sequence of coins where one side is appearing more often the other one.

To facilitate the use of this theorem in the next pages, we will also slightly generalize the problem assuming that the coins are in $\{-L, L\}$ instead of $\{-1, 1\}$, where $L > 0$.

Theorem 5.4. Let $T \geq 1$ even and $0 \leq q \leq \frac{T}{2}$. Then, for any online betting algorithm that guarantees non-negative wealth on any sequence of coins $T$ in $\{-L, L\}$ and starts with initial wealth $\epsilon$, there exists a sequence of coins such that $|\sum_{t=1}^T c_t| \geq 2qL$ and the wealth of the algorithm is upper bounded by
$$\frac{3}{2} \left( \frac{2q}{\sqrt{T}} + 1 \right) \epsilon \exp \left( T \cdot D \left( \frac{1}{2} + \frac{q}{T} \middle\| \frac{1}{2} \right) \right) \leq \frac{3}{2} \left( \frac{2q}{\sqrt{T}} + 1 \right) \epsilon \exp \left( 2 \frac{q^2}{T} + 3.1 \frac{q^4}{T^3} \right),$$
where $D(p\|q)$ denotes the KL divergence between two Bernoulli distributions with parameters $p$ and $q$:
$$D(p\|q) = p \ln \frac{p}{q} + (1 - p) \ln \frac{1 - p}{1 - q}.$$

Proof. Let $Y_t$ independent random variable that assume the value of 1 with probability 0.5 and -1 with probability 0.5. Hence, we have that $\mathbb{E}[\sum_{t=1}^T x_t L Y_t] = 0$, and also $\sum_{t=1}^T x_t L Y_t \geq -\epsilon$ for the hypothesis on the betting algorithm.

For any $q > 0$, it follows that
$$0 = \mathbb{E} \left[ \sum_{t=1}^T x_t Y_t \right] = \mathbb{E} \left[ \sum_{t=1}^T x_t Y_t \middle| \left| \sum_{t=1}^T Y_t \right| < 2q \right] \mathbb{P} \left\{ \left| \sum_{t=1}^T Y_t \right| < 2q \right\} + \mathbb{E} \left[ \sum_{t=1}^T x_t Y_t \middle| \left| \sum_{t=1}^T Y_t \right| \geq 2q \right] \mathbb{P} \left\{ \left| \sum_{t=1}^T Y_t \right| \geq 2q \right\} \geq -\frac{\epsilon}{L} + \left( \frac{\epsilon}{L} + \mathbb{E} \left[ \sum_{t=1}^T x_t Y_t \middle| \left| \sum_{t=1}^T Y_t \right| \geq 2q \right] \right) \mathbb{P} \left\{ \left| \sum_{t=1}^T Y_t \right| \geq 2q \right\},$$
hence
$$\mathbb{E} \left[ \sum_{t=1}^T x_t Y_t \middle| \left| \sum_{t=1}^T Y_t \right| \geq 2q \right] \leq \frac{\epsilon}{L \mathbb{P} \left\{ \left| \sum_{t=1}^T Y_t \right| \geq 2q \right\}} - \frac{\epsilon}{L} = \frac{\epsilon}{2L \mathbb{P} \left\{ \sum_{t=1}^T Y_t \geq 2q \right\}} - \frac{\epsilon}{L}.$$

Using the fact that $\mathbb{P} \left\{ \sum_{t=1}^T Y_t \geq 2q \right\} = \mathbb{P} \left\{ \frac{\sum_{t=1}^T Y_t + 1}{2} \geq \frac{1}{2} T + q \right\}$, where $\frac{Y_t+1}{2}$ are Bernoulli random variables, we can apply Lemma B.4, to obtain
$$\mathbb{P} \left\{ \sum_{t=1}^T Y_t \geq 2q \right\} \geq \frac{1}{3} \frac{1}{\frac{2q}{\sqrt{T}} + 1} \exp \left( -T \cdot D \left( \frac{1}{2} + \frac{q}{T} \middle\| \frac{1}{2} \right) \right).$$

Hence, we have
$$\mathbb{E} \left[ \sum_{t=1}^T x_t L Y_t \middle| \left| \sum_{t=1}^T Y_t \right| \geq 2q \right] \leq \frac{3}{2} \epsilon \left( \frac{2q}{\sqrt{T}} + 1 \right) \exp \left( T \cdot D \left( \frac{1}{2} + \frac{q}{T} \middle\| \frac{1}{2} \right) \right) - \epsilon.$$

Given that the minimum over a set is smaller than or equal to the expectation with respect to any distribution over the set, there exists a sequence of $c_1, \dots, c_T \in \{-L, L\}^T$ such that $\frac{1}{L} |\sum_{t=1}^T c_t| \geq 2q$ and the wealth of the algorithm is deterministically upper bounded by
$$\frac{3}{2} \epsilon \left( \frac{2q}{\sqrt{T}} + 1 \right) \exp \left( T \cdot D \left( \frac{1}{2} + \frac{q}{T} \middle\| \frac{1}{2} \right) \right).$$

For the second upper bound, it is enough to use the elementary inequality
$$D \left( \frac{1}{2} + x \middle\| \frac{1}{2} \right) \leq 2x^2 + 3.1x^4, \quad |x| \leq \frac{1}{2}.$$

Remark 5.5. It is also possible to upper bound the left hand side of the bound in the previous Theorem with the wealth of the best constant betting fraction.
For the expression of the optimal wealth on the $c_1, \dots, c_T$, consider the wealth of strategy that bets a constant amount of money $\beta$. Starting with initial money $\epsilon$, after $T$ rounds the wealth is $\epsilon \prod_{t=1}^T (1 + \beta c_t)$. By taking the derivative of the logarithm of the wealth, it is immediate to verify that the $\beta^\star$ that maximizes the above quantity is $\frac{\sum_{t=1}^T c_t}{L^2 T}$. Denote by $k = |\{c_t : c_t = L\}|$, hence we have $\beta^\star = \frac{2k-T}{L^2 T}$. Hence, the optimal wealth is
$$\epsilon (1 + \beta^\star L)^k (1 - \beta^\star L)^{T-k} = \epsilon \exp \left( k \ln \frac{2k}{T} + (T - k) \ln 2 \left( 1 - \frac{k}{T} \right) \right) = \epsilon \exp \left( T \cdot D \left( \frac{k}{T} \middle\| \frac{1}{2} \right) \right).$$

Equivalently, given that $\frac{1}{L} \sum_{t=1}^T c_t = 2k - T$, we also have that
$$\max_{-1 \leq \beta \leq 1} \prod_{t=1}^T (1 + \beta c_t) = \exp \left( T \cdot D \left( \frac{\sum_{t=1}^T c_t}{2LT} + \frac{1}{2} \middle\| \frac{1}{2} \right) \right) = \exp \left( T \cdot D \left( \frac{|\sum_{t=1}^T c_t|}{2LT} + \frac{1}{2} \middle\| \frac{1}{2} \right) \right),$$
where in the second equality we used the fact that $D(x + \frac{1}{2} \| \frac{1}{2}) = D(-x + \frac{1}{2} \| \frac{1}{2})$. Hence, we have
$$\frac{3}{2} \epsilon \left( \frac{2q}{\sqrt{T}} + 1 \right) \exp \left( T \cdot D \left( \frac{1}{2} + \frac{q}{T} \middle\| \frac{1}{2} \right) \right) \leq \frac{3}{2} \epsilon \left( \frac{2q}{\sqrt{T}} + 1 \right) \exp \left( T \cdot D \left( \frac{1}{2} + \frac{|\sum_{t=1}^T c_t|}{2LT} \middle\| \frac{1}{2} \right) \right) = \frac{3}{2} \left( \frac{2q}{\sqrt{T}} + 1 \right) \max_\beta \epsilon \prod_{t=1}^T (1 + c_t \beta).$$

Now, let's connect the coin-betting game with OLO, thanks to the next Theorem.

Theorem 5.6. Let $\epsilon_t$ a non-decreasing sequence and $\mathcal{A}$ an OLO algorithm that guarantees $\text{Regret}_t(0) \leq \epsilon_t$ for any sequence of $g_1, \dots, g_t \in \mathbb{R}^d$ with $\|g_i\|_2 \leq L$. Then, there exists $\beta_t$ such that $x_t = \frac{\beta_t}{L} (\epsilon_T - \sum_{i=1}^{t-1} \langle g_i, x_i \rangle)$ and $\|\beta_t\|_2 \leq 1$ for $t = 1, \dots, T$.

Proof. Define $r_t = -\sum_{i=1}^t \langle g_i, x_i \rangle$ the "reward" of the algorithm. So, we have
$$\text{Regret}_t(u) = \sum_{i=1}^t \langle g_i, x_i \rangle - \sum_{i=1}^t \langle g_i, u \rangle = -r_t + \sum_{i=1}^t \langle g_i, u \rangle.$$

Since, we assumed that $\text{Regret}_t(0) \leq \epsilon_t$, we always have $r_t \geq -\epsilon_t$. Using this, we claim that $L \|x_t\|_2 \leq r_{t-1} + \epsilon_t$ for all $t = 1, \dots, T$. To see this, assume that there is a sequence $g_1, \dots, g_{t-1}$ that gives $L \|x_t\|_2 > r_{t-1} + \epsilon_t$. We then set $g_t = L \frac{x_t}{\|x_t\|_2}$. For this sequence, we would have $r_t = r_{t-1} - L\|x_t\|_2 < -\epsilon_t$, that contradicts the observation that $r_t \geq -\epsilon_t$.

So, from the fact that $L \|x_t\|_2 \leq r_{t-1} + \epsilon_t \leq r_{t-1} + \epsilon_T$ we have that there exists $\beta_t$ such that $x_t = \frac{\beta_t}{L} (\epsilon_T + r_{t-1})$ for a $\beta_t$ and $\|\beta_t\|_2 \leq 1$.

This theorem informs us of something important: any OLO algorithm that suffers a non-decreasing regret against the null competitor must predict in the form of a "vectorial" coin-betting algorithm.

From this connection to online betting, we now derive a lower bound for online convex optimization.

Theorem 5.7. Let $T \in \mathbb{N}$ even and let $\mathcal{A}$ be any online convex optimization algorithm that guarantees regret at most $\epsilon_T$ against the null competitor on any sequence of $T$ linear and $L$-Lipschitz losses $\ell_t : \mathbb{R} \to \mathbb{R}$ for $t = 1, \dots, T$. Let $U > 0$ be such that $1 \leq W\left( \frac{\sqrt{T} UL}{5\epsilon_T} \right) \leq \frac{\sqrt{T}}{2}$. Then, for this algorithm, there exists a sequence of $g_t$ with $\|g_t\|_2 \leq L$ and a competitor $u \in \mathbb{R}^d$ with $\|u\|_2 = U$, such that
$$\sum_{t=1}^T \langle g_t, x_t - u \rangle \geq R_T(U) := UL\sqrt{T} \left( 2W\left( \frac{\sqrt{T} UL}{5\epsilon_T} \right) - 1 \right) - 2UL + \epsilon_T,$$
where $W : \mathbb{R}_{\geq 0} \to \mathbb{R}_{\geq 0}$ is the Lambert function.

Moreover, assuming there exists $K > 0$ such that $\epsilon_T \leq K < \infty$ for all $T$, we have $\lim_{T \to +\infty} \frac{R_T(U)}{UL\sqrt{T \ln T}} = 1$.

Proof. Consider the linear losses $\ell_t(x) = g_t x$, where $|g_t| \leq L$.

Given that the algorithm guarantees a regret of at most $\epsilon_T$ against the null competitor, from the Theorem 5.6 we have that, if we feed the algorithm with linear losses, we can reinterpret the algorithm as a betting algorithm with initial wealth $\epsilon_T$ and that guarantees non-negative wealth, where the wealth at time $T$ is defined as $\epsilon_T - \sum_{t=1}^T g_t x_t$.

Set $A = W \left( \frac{\sqrt{T} UL}{5\epsilon_T} \right)$, $\tilde{q} = \frac{\sqrt{T}}{2} \sqrt{2A - 1}$, and $q = \lfloor \tilde{q} \rfloor$, so that $\tilde{q} \geq q \geq \tilde{q}- 1$.

Observe that the constraint on $U$ assures that
$$q \leq \tilde{q} \leq \frac{T^{3/4}}{2} \leq \frac{T}{2}, \quad (5.1)$$

so we can safely use Theorem 5.4, that tells us that there exists a sequence of $c_t \in \{-L, L\}^T$ such that $|\sum_{t=1}^T c_t| \geq 2qL$ and
$$\sum_{t=1}^T c_t x_t \leq \frac{3}{2} \left( \frac{2q}{\sqrt{T}} + 1 \right) \epsilon_T \exp \left( 2 \frac{q^2}{T} \right) \exp \left( 3.1 \frac{q^4}{T^3} \right) - \epsilon_T.$$

Moreover, observe that
$$\exp \left( 2 \frac{q^2}{T} \right) \exp \left( 3.1 \frac{q^4}{T^3} \right) \leq \exp \left( 2 \frac{\tilde{q}^2}{T} \right) \exp \left( 3.1 \frac{\tilde{q}^4}{T^3} \right) \leq \exp \left( 2 \frac{\tilde{q}^2}{T} \right) \exp \left( 3.1 \frac{1}{16} \right) \leq \frac{4}{3} \exp \left( 2 \frac{\tilde{q}^2}{T} \right),$$
where in second inequality we used $\frac{\tilde{q}^4}{T^3} \leq \frac{1}{16}$ from (5.1).

Set $g_t = -c_t$ and choose $u = -U \text{sign}(\sum_{t=1}^T g_t)$, so we have
$$\sum_{t=1}^T g_t(x_t - u) = U \left| \sum_{t=1}^T g_t \right| + \sum_{t=1}^T g_t x_t \geq U \cdot 2qL - 2 \left( \frac{2\tilde{q}}{\sqrt{T}} + 1 \right) \epsilon_T \exp \left( 2 \frac{\tilde{q}^2}{T} \right) + \epsilon_T \geq UL \cdot (2\tilde{q} - 2) - 2 \left( \frac{2\tilde{q}}{\sqrt{T}} + 1 \right) \epsilon_T \exp \left( 2 \frac{\tilde{q}^2}{T} \right) + \epsilon_T.$$

Now, from the condition on $U$, we have $\frac{2\tilde{q}}{\sqrt{T}} \geq 1$, hence we have
$$\sum_{t=1}^T g_t(x_t - u) = U \left| \sum_{t=1}^T g_t \right| + \sum_{t=1}^T g_t x_t \geq 4 \epsilon_T \left[ \frac{\sqrt{T} UL}{4\epsilon_T} \cdot \frac{2\tilde{q}}{\sqrt{T}} - \frac{2\tilde{q}}{\sqrt{T}} \exp \left( 2 \frac{\tilde{q}^2}{T} \right) \right] - 2UL + \epsilon_T. \quad (5.2)$$

Let $a \geq 1$, then
$$x^\star := \text{argmax}_x ax - x \exp(x^2/2) = \sqrt{2W \left( \frac{a \sqrt{e}}{2} \right) - 1}.$$

So, we have
$$ax^\star - x^\star \exp((x^\star)^2/2) = a \sqrt{2W \left( \frac{a \sqrt{e}}{2} \right) - 1} \left( 1 - \frac{1}{2W \left( \frac{a \sqrt{e}}{2} \right)} \right).$$

Hence, our choice of $\tilde{q}$ maximizes the lower bound in (5.2) and the condition on $U$ assures that $\frac{\sqrt{T} U}{4\epsilon_T} \geq 1$. Hence, we obtain
$$\sum_{t=1}^T g_t(x_t - u) \geq UL \sqrt{T} \left( 2W \left( \frac{\sqrt{T} UL}{4\epsilon_T} \frac{\sqrt{e}}{2} \right) - 1 \right) \left( 1 - \frac{1}{2W \left( \frac{\sqrt{T} UL}{4\epsilon_T} \frac{\sqrt{e}}{2} \right)} \right) - 2UL + \epsilon_T.$$

Simplifying the numerical constants and using the elementary inequality $(1 - 1/(2x))\sqrt{2x - 1} \geq \sqrt{2x - 1}$ for $x \geq 1$ gives the stated bound.

Remark 5.8. The leading constant $\sqrt{2}$ is asymptotically optimal because there exist algorithms with a matching upper bound.

This theorem implies that that OSD with learning rate $\eta = \frac{\alpha}{L\sqrt{T}}$ does not have the optimal dependency on $\|\mathbf{u}\|_2$ for any $\alpha > 0$.

In Chapter 10, we will see that the connection between coin-betting and OLO can also be used to design OLO algorithm. This will give us optimal unconstrained OLO algorithms with the surprising property of not requiring a learning rate at all.

### 5.4 History Bits

The lower bound for OCO is quite standard, the proof presented is a simplified version of the one in Orabona and Pál [2018]. One could also use the function in the lower bound for offline optimization in Nesterov [2004, Section 3.2.1], but it would require the additional assumptions that the $d > T$ and that $x_t$ lies in the span of the previous subgradients. This limitation on $d$ is unavoidable: without it offline convex optimization becomes easier while Online Convex Optimization is equally hard for any number of dimensions.

The lower bound for OSD in Theorem 5.3 is a minor generalization of the one in Orabona and Pál [2015, 2018]. They provide a similar lower bound for the Exponentiated Gradient algorithm.

Strangely enough, both the online learning literature and the optimization one almost ignored the issue of lower bounds for the unconstrained case. The connection between coin-betting and OLO was first unveiled in Orabona and Pál [2016]. Theorem 5.6 is an unpublished result by Ashok Cutkosky, that proved similar and more general results in his PhD thesis [Cutkosky, 2018]. The first lower bound for unconstrained OLO is from Streeter and McMahan [2012], but their proof relied on using the value of $\inf_T \mathbb{P}\{\sum_{i=1}^T X_i \geq \sqrt{T}\}$ where $X_i$ are Rademacher random variables. Streeter and McMahan [2012] claim that this value is $7/64$ that corresponds to $T = 6$, but they do not provide a proof for it. The same value was also conjectured by Hitczenko and Kwapien [1994] and proved formally only in 2023 by Hollom and Portier [2023]. A proof avoiding completely that step was given by Orabona [2013]. Theorem 5.7 is new: it is a more precise version of the lower bound in Orabona [2013] and it has the asymptotically optimal constant $\sqrt{2}$. One way to achieve the optimal constant in lower bounds using the tail of Binomial distributions was shown in Orabona and Pál [2015]. Independently, Zhang et al. [2022] proved a lower bound for unconstrained OCO with the optimal constant but only when $\epsilon_T = \mathcal{O}(\sqrt{T})$, with an algorithm with a matching upper bound. The specific method used here to obtain the optimal constant is new. McMahan and Orabona [2014] proposed an algorithm that matches the lower bound up to a multiplicative constant, while Zhang et al. [2022] matched the multiplicative constant too. More recently, Carmon and Hinder [2024] proved a lower bound of $\Omega(\|x^\star - x_1\| \sqrt{\frac{\ln \|x^\star - x_1\|}{\epsilon_T} T})$ in the stochastic setting on the expected suboptimality gap. Their lower bound implies the same lower bound we stated on the regret of OCO through the online to batch conversion. However, their lower bo... [truncated]

### 5.5 Exercises

Problem 5.1. Fix $U > 0$ and $\mathcal{V} = \mathbb{R}^d$. Mimicking the proof of Theorem 5.1, prove that for any OCO algorithm there exists a $u^\star$ and a sequence of loss functions such that $\text{Regret}_T(u^\star) \geq \frac{1}{2} \|u^\star\|_2 L \sqrt{T}$ where $\|u^\star\|_2 = U$ and the loss functions are $L$-Lipschitz with respect to $\|\cdot\|_2$.

Problem 5.2. Extend the proof of Theorem 5.1 to an arbitrary norm $\|\cdot\|$ to measure the diameter of $\mathcal{V}$ and with $\|g_t\|_\star \leq L$.

# Chapter 6

# Online Mirror Descent

In this chapter, we will introduce the Online Mirror Descent (OMD) algorithm. To explain its genesis, I think it is essential to understand what subgradients do. In particular, the negative subgradients are not always pointing towards a direction that minimizes the function. Then, we will introduce the Bregman divergences to generalize the notion of distance implicitly used in online subgradient descent. Finally, we will see extensions and applications of OMD.

## 6.1 Subgradients are not Informative

We have seen that in online learning we receive a sequence of loss functions and we have to output a vector before observing the loss function on which we will be evaluated. However, we can gain a lot of intuition if we consider the easy case that the sequence of loss functions is always a fixed function, i.e., $\ell_t(x) = \ell(x)$. If our hypothetical online algorithm does not work in this situation, for sure it will not work on the more general case.

Hence, considering the case of fixed loss functions, let's take a look at the key step in the proof of the upper bound to the regret for Online Subgradient Descent (OSD) in Lemma 2.30. We used the following property of the subgradients:
$$\ell(x_t) - \ell(u) \leq \langle g_t, x_t - u \rangle, \quad \forall u \in \mathcal{V}. \quad (6.1)$$

In words, to minimize the left hand side of this equation, it is enough to minimize the right hand side, that is nothing else than the instantaneous linear regret on the linear function $\langle g_t, \cdot \rangle$. This is the only reason why Online Subgradient Descent (OSD) works! However, I am sure you heard a million of times the (wrong) intuition that gradient points towards the minimum, and you might be tempted to think that the same (even more wrong) intuition holds for subgradients. Indeed, I am sure that even if we proved the regret guarantee based on (6.1), in the back of your mind you keep thinking "yeah, sure, it works because the subgradient tells me where to go to minimize the function". Typically this idea is so strong that I have to present explicit counterexamples to fully convince a person.

So, take a look at the following examples that illustrate the fact that a subgradient does not always point in a direction where the function decreases.

Example 6.1. Let $f(x) = \max[-x_1, x_1 - x_2, x_1 + x_2]$, see Figure 6.1. The vector $g = (1, 1)$ is a subgradient in $x = (1, 0)$ of $f(x)$. No matter how we choose the stepsize, moving in the direction of the negative subgradient will not decrease the objective function. An even more extreme example is in Figure 6.2, with the function $f(x) = \max[x_1^2 + (x_2 + 1)^2, x_1^2 + (x_2 - 1)^2]$. Here, in the point $(1, 0)$, any positive step in the direction of the negative subgradient will increase the objective function.

In both examples, one might think that it would be enough to add a little bit of noise to move away from the "corners". However, we have to remember that we are in the adversarial setting. So, assuming the adversary see our randomization, it can always present us a function such that our prediction is exactly on a "corner". This means that our analysis of subgradient descent will not have to use the fact that the subgradients point towards descending directions, because it can be false in each single iteration.

[Screenshot for page 51]
Figure 6.1: 3D plot (left) and level sets (right) of $f(x) = \max[-x_1, x_1 - x_2, x_1 + x_2]$. A negative subgradient is indicated by the black arrow.

Figure 6.2: 3D plot (left) and level sets (right) of $f(x) = \max[x_1^2 + (x_2 + 1)^2, x_1^2 + (x_2 - 1)^2]$. A negative subgradient is indicated by the black arrow.

Remark 6.2. Given the above, one might wonder how much "information" the subgradients carry. It turns out, quite a lot! In fact, for bounded domains in the batch case, i.e., when all the functions are the same, the cutting plane method can optimize the function exponentially fast just using subgradients. However, in the adversarial setting subgradients becomes much weaker than in the batch setting, exactly because the adversary has the power to change the function in each round.

6.2 Reinterpreting the Online Subgradient Descent Algorithm
How Online Subgradient Descent works? It works exactly as I told you before: thanks to (6.1). But, what does that inequality really mean?
A way to understand how the OSD algorithm works is to think that it minimizes a local approximation of the original objective function. This is not unusual for optimization algorithms, for example the Newton's algorithm constructs an approximation with a Taylor expansion truncated to the second term. Thanks to the definition of subgradients, we can immediately build a linear lower bound to a function $f$ around $x_0$:
$$f(x) \geq \tilde{f}(x) := f(x_0) + \langle g, x - x_0 \rangle, \quad \forall x \in \mathcal{V}.$$
So, in our setting, this would mean that we update the online algorithm with the minimizer of a linear approximation

[Screenshot for page 52]
of the loss function you received. Unfortunately, minimizing a linear function is unlikely to give us a good online algorithm. Indeed, over unbounded domains the minimum of a linear function is $-\infty$.
So, let's introduce the other key concept: we constraint the minimization of this lower bound only in a neighborhood of $x_0$, where we have good reason to believe that the approximation is more precise. Moreover, in online learning it makes sense not to go too far from the previous iteration because the losses are different in each step and we do now want to give too much importance to the current loss. Coding the neighborhood constraint with a $L_2$ squared distance from $x_0$ less than some positive number $h$, we might think to use the following update
$$x_{t+1} = \underset{x \in \mathcal{V}}{\operatorname{argmin}} \ f(x_t) + \langle g, x - x_t \rangle$$
$$\text{s.t. } \|x_t - x\|^2 \leq h.$$
Equivalently, for some $\eta > 0$, we can consider the unconstrained formulation
$$\underset{x \in \mathcal{V}}{\operatorname{argmin}} \ \hat{f}(x) := f(x_0) + \langle g, x - x_0 \rangle + \frac{1}{2\eta} \|x_0 - x\|_2^2. \quad (6.2)$$
This is a well-defined update scheme, that hopefully moves $x_t$ closer to the optimum of $f$. See Figure 6.3 for a graphical representation in one-dimension.

Figure 6.3: Approximations of $f(x)$.

And now the final element of our story: the argmin in (6.2) is exactly the update we used in OSD! Indeed, solving the argmin and completing the square, we get
$$\underset{x \in \mathcal{V}}{\operatorname{argmin}} \ \langle g_t, x \rangle + \frac{1}{2\eta_t} \|x_t - x\|_2^2 = \underset{x \in \mathcal{V}}{\operatorname{argmin}} \ \|\eta_t g_t\|^2 + 2\eta_t \langle g_t, x - x_t \rangle + \|x_t - x\|_2^2 \quad (6.3)$$
$$= \underset{x \in \mathcal{V}}{\operatorname{argmin}} \ \|x - x_t + \eta_t g_t\|_2^2$$
$$= \Pi_{\mathcal{V}}(x_t - \eta_t g_t),$$
where we used the fact that the argmin is independent to additive constants and positive rescalings, and $\Pi_{\mathcal{V}}$ is the Euclidean projection onto $\mathcal{V}$, i.e., $\Pi_{\mathcal{V}}(x) = \underset{y \in \mathcal{V}}{\operatorname{argmin}} \|x - y\|_2$.
The new way to write the update of OSD in (6.2) will be the core ingredient for designing Online Mirror Descent. In fact, Online Mirror Descent (OMD) is a strict generalization of that update when we use a different way to measure the locality of $x$ from $x_t$. That is, we measured the distance between to the current point with the squared $L_2$ norm. What happens if we change the norm? Do we even have to use a norm?
To answer these questions we have to introduce another useful mathematical object: the Bregman divergence.

[Screenshot for page 53]
6.3 Convex Analysis Bits: Bregman Divergence
We first give a new definition, a slightly stronger notion of convexity.
Definition 6.3 (Strictly Convex Function). Let $f : \mathcal{V} \subseteq \mathbb{R}^d \to \mathbb{R}$ and $\mathcal{V}$ a convex set. $f$ is strictly convex if
$$f(\alpha x + (1 - \alpha)y) < \alpha f(x) + (1 - \alpha)f(y), \quad \forall x, y \in \mathcal{V}, x \neq y, 0 < \alpha < 1.$$
From the definition, it is immediate to see that strong convexity with respect to any norm implies strict convexity. Note that for a differentiable function, strict convexity also implies that $f(y) > f(x) + \langle \nabla f(x), y - x \rangle$ for $x \neq y$ [Bauschke and Combettes, 2017, Proposition 17.10].
We now define our new notion of "distance".
Definition 6.4 (Bregman Divergence). Let $\psi : \mathcal{X} \to \mathbb{R}$ be strictly convex and differentiable on $\operatorname{int} \mathcal{X} \neq \emptyset$. The Bregman Divergence with respect to $\psi$ is denoted by $B_\psi : \mathcal{X} \times \operatorname{int} \mathcal{X} \to \mathbb{R}$ defined as
$$B_\psi(x; y) = \psi(x) - \psi(y) - \langle \nabla \psi(y), x - y \rangle.$$
In the offline optimization literature, the function $\psi$ associated to $B_\psi$ is often called the distance generating function.
From the definition, we see that the Bregman divergence is always non-negative for $x, y \in \operatorname{int} \mathcal{X}$, from the convexity of $\psi$. However, something stronger holds. By the strict convexity of $\psi$, for fixed a point $y \in \operatorname{int} \mathcal{X}$ we have that $\psi(x) \geq \psi(y) + \langle \nabla \psi(y), x - y \rangle, \forall y \in \mathcal{X}$, with equality only for $y = x$. Hence, the strict convexity allows us to use the Bregman divergence as a similarity measure between $x$ and $y$. Moreover, this similarity measure changes with the reference point $y$. This also implies that, as you can see from the definition, the Bregman divergence is not symmetric.
Let me give you some more intuition on the concept of the Bregman divergence. Consider the case that $\psi$ is twice differentiable in an open ball $B$ around $y$ and $x \in B$. So, by the Taylor's theorem, there exists $0 \leq \alpha \leq 1$ such that
$$B_\psi(x; y) = \psi(x) - \psi(y) - \nabla \psi(y)^\top(x - y) = \frac{1}{2}(x - y)^\top \nabla^2 \psi(z)(x - y),$$
where $z = \alpha x + (1 - \alpha)y$. Hence, we are using a squared local norm that depends on the Hessian of $\psi$. Different areas of the space will have a different value of the Hessian, and so the Bregman will behave differently. We will use this exact idea in the local norm analyses of Online Mirror Descent (Section 6.5) and Follow-the-Regularized-Leader (Section 7.4).
We can also lower bound the Bregman divergence if the function $\psi$ is strongly convex. In particular, if $\psi$ is $\lambda$-strongly convex with respect to a norm $\|\cdot\|$ in $\operatorname{int} \mathcal{X}$, then we have
$$B_\psi(x; y) \geq \frac{\lambda}{2} \|x - y\|^2. \quad (6.4)$$
Example 6.5. If $\psi(x) = \frac{1}{2}\|x\|_2^2$, then $B_\psi(x; y) = \frac{1}{2}\|x\|_2^2 - \frac{1}{2}\|y\|_2^2 - \langle y, x - y \rangle = \frac{1}{2}\|x - y\|_2^2$.
Example 6.6. Let $\mathcal{X} = \mathbb{R}^d_{\geq 0}$ and $\psi(x) = \sum_{i=1}^d x_i \ln x_i$, the negative entropy. Then, for all $x \in \mathcal{X}$ and $y \in \operatorname{int} \mathcal{X}$ we have
$$B_\psi(x; y) = \sum_{i=1}^d (x_i \ln x_i - y_i \ln y_i - (\ln(y_i) + 1)(x_i - y_i)) = \sum_{i=1}^d \left( x_i \ln \frac{x_i}{y_i} - x_i + y_i \right).$$
This is called the generalized Kullback-Leibler divergence, where "generalized" is due to the fact that $x$ and $y$ do not have to be discrete probability distributions.
We also have the following immediate lemma that links the Bregman divergences between 3 points.
Lemma 6.7 ([Chen and Teboulle, 1993]). Let $B_\psi$ the Bregman divergence with respect to $\psi : \mathcal{X} \to \mathbb{R}$. Then, for any three points $x, y \in \operatorname{int} \mathcal{X}$ and $z \in \mathcal{X}$, the following identity holds
$$B_\psi(z; x) + B_\psi(x; y) - B_\psi(z; y) = \langle \nabla \psi(y) - \nabla \psi(x), z - x \rangle.$$

[Screenshot for page 54]
6.4 Online Mirror Descent
Based on what we said before, we can start from the equivalent formulation of the OSD update,
$$x_{t+1} = \underset{x \in \mathcal{V}}{\operatorname{argmin}} \ \langle g_t, x \rangle + \frac{1}{2\eta_t} \|x_t - x\|_2^2,$$
and we can change the last term with another measure of distance. In particular, using the Bregman divergence with respect to a function $\psi$, we have
$$x_{t+1} = \underset{x \in \mathcal{V}}{\operatorname{argmin}} \ \langle g_t, x \rangle + \frac{1}{\eta_t} B_\psi(x; x_t),$$
where we assumed that the argmin exists. These two updates are exactly the same when $\psi(x) = \frac{1}{2}\|x\|_2^2$.
So, we get the Online Mirror Descent algorithm in Algorithm 6.1.

Algorithm 6.1 Online Mirror Descent
Require: Non-empty closed convex $\mathcal{V} \subseteq \mathcal{X} \subseteq \mathbb{R}^d$, $\psi : \mathcal{X} \to \mathbb{R}$ strictly convex and differentiable on $\operatorname{int} \mathcal{X}$, $x_1 \in \operatorname{int} \mathcal{X} \cap \mathcal{V}$, $\eta_1, \dots, \eta_T > 0$
1: for $t = 1$ to $T$ do
2: Output $x_t \in \mathcal{V}$
3: Pay $\ell_t(x_t)$ for $\ell_t : \mathcal{V} \to \mathbb{R}$ subdifferentiable in $\mathcal{V}$
4: Set $g_t \in \partial \ell_t(x_t)$
5: Set $x_{t+1} \in \underset{x \in \mathcal{V}}{\operatorname{argmin}} \ \langle g_t, x \rangle + \frac{1}{\eta_t} B_\psi(x; x_t)$
6: end for

However, without an additional assumption, this algorithm has a problem. Can you see it? The problem is that $x_{t+1}$ might be on the boundary of $\mathcal{V}$ and in the next step we would have to evaluate $B_\psi(x; x_{t+1})$ for a point on the boundary of $\mathcal{V}$. Given that $\mathcal{V} \subseteq \mathcal{X}$, we might end up on the boundary of $\mathcal{X}$ where the Bregman is not defined!
To fix this problem, different sufficient conditions can be used. In the following, we will use either one of the following assumptions:
$$\lim_{\lambda \to 0} \langle \nabla \psi(x + \lambda(y - x)), y - x \rangle = -\infty, \quad \forall x \in \operatorname{bdry} \mathcal{X}, \forall y \in \operatorname{int} \mathcal{X} \quad (6.5)$$
$$\mathcal{V} \subseteq \operatorname{int} \mathcal{X}. \quad (6.6)$$
If either of these conditions is true and the argmin exists on each round then the algorithm is well-defined as proved in the following theorem.
Theorem 6.8. Let $B_\psi$ the Bregman divergence with respect to $\psi : \mathcal{X} \to \mathbb{R}$. Let $\mathcal{V} \subseteq \mathcal{X}$ a non-empty closed convex set. Assume (6.5) or (6.6) hold and, with the notation in Algorithm 6.1, the argmin exists on all rounds. Then, $x_{t+1} \in \operatorname{int} \mathcal{X}$.
Proof. In the case that (6.6) holds, we have that $x_{t+1} \in \mathcal{V}$ implies immediately that $x_{t+1} \in \operatorname{int} \mathcal{X}$.
Let's now assume that (6.5) holds and let's prove it by induction. The base case is true by the definition of $x_1$. Let's now assume that $x_t \in \operatorname{int} \mathcal{X}$ and let's prove that $x_{t+1} \in \operatorname{int} \mathcal{X}$. We will prove it by contradiction. So, assume that $x_{t+1} \in \operatorname{bdry} \mathcal{X}$. Set $z \in \operatorname{int} \mathcal{X} \cap \mathcal{V}$ and define $\phi(\lambda) = \langle \eta_t g_t, (1 - \lambda)x_{t+1} + \lambda z \rangle + B_\psi((1 - \lambda)x_{t+1} + \lambda z; x_t)$ for $\lambda \in (0, 1)$. From (6.5), we have that
$$\lim_{\lambda \to 0} \phi'(\lambda) = \lim_{\lambda \to 0} \langle \eta_t g_t, z - x_{t+1} \rangle + \langle \nabla \psi(x_{t+1} + \lambda(z - x_{t+1})) - \nabla \psi(x_t), z - x_{t+1} \rangle = -\infty.$$
Hence, there exists $\epsilon > 0$ such that
$$\langle \eta_t g_t, x_\epsilon \rangle + B_\psi(x_\epsilon; x_t) = \phi(\epsilon) < \phi(0) = \langle \eta_t g_t, x_{t+1} \rangle + B_\psi(x_{t+1}; x_t),$$
where $x_\epsilon := (1 - \epsilon)x_{t+1} + \epsilon z \in \operatorname{int} \mathcal{X} \cap \mathcal{V}$. However, this contradicts, the definition of $x_{t+1}$ as an argmin, proving that $x_{t+1}$ must be in $\operatorname{int} \mathcal{X}$.

[Screenshot for page 55]
When (6.5) holds, this theorem implies that the predictions of the algorithm always stay in the interior of the feasible set without the need for any projection. If in addition $\mathcal{V} = \mathcal{X}$, the update of the algorithm is the solution of an unconstrained problem because the feasible set is implicit in the Bregman divergence.
Now we have a well-defined algorithm, but does it guarantee a sublinear regret? We know that at least in one case it recovers the OSD algorithm, that does work. So, from an intuitive point of view, how well the algorithm work should depend on some characteristic on $\psi$. In particular, a key property will be the strong convexity of $\psi$. The strong convexity also takes care of the existence of the argmin in the algorithm, thanks to next Theorem.
Theorem 6.9. Let $\lambda > 0$ and $f : \mathbb{R}^d \to (-\infty, +\infty]$ proper, closed, and $\lambda$-strongly convex with respect to $\|\cdot\|$ on its domain. Assume $\operatorname{dom} \partial f \neq \emptyset$. Then, $f$ has exactly one minimizer.
Proof. Let $y \in \operatorname{dom} \partial f$ and $g \in \partial f(y)$. From Lemma 4.2, for any $x \in \mathbb{R}^d$, we have
$$f(x) \geq f(y) + \langle g, x - y \rangle + \frac{\lambda}{2} \|x - y\|^2$$
$$\geq f(y) - \|g\|_\star \|x\| - \langle g, y \rangle + \frac{\lambda}{2} (\|x\| - \|y\|)^2$$
$$= f(y) - \|g\|_\star \|x\| - \langle g, y \rangle + \frac{\lambda}{2} (\|x\|^2 + \|y\|^2 - 2\|x\|\|y\|),$$
where in the second inequality we used the reverse triangle inequality and the definition of dual norms. From the above, we have that $\lim_{\|x\| \to \infty} f(x) = +\infty$. In turn, this implies that the level sets of $f$ are bounded. From the assumption that $f$ is closed, we get that the level sets are compact. Hence, for any $y$ in $\operatorname{dom} f$, the minimum of $f$ is the same of the minimum of $f$ over the set $\{x : f(x) \leq f(y)\}$, that is the minimum over a compact set, that exists by the Weierstrass theorem, Theorem D.8. The uniqueness is given by the fact that strongly convex function are strictly convex.
To analyze OMD, we first prove a one step relationship, similar to the ones we proved for Online Gradient Descent (OGD) and OSD. Note how in this Lemma, we will use a lot of the concepts we introduced till now: strong convexity, dual norms, subgradients, etc. In a way, over the past sections I slowly prepared you to be able to prove this lemma.
Lemma 6.10. Let $B_\psi$ the Bregman divergence with respect to $\psi : \mathcal{X} \to \mathbb{R}$ and assume $\psi$ to be proper, closed, and $\lambda$-strongly convex with respect to $\|\cdot\|$ in $\mathcal{V} \cap \operatorname{int} \mathcal{X}$. Let $\mathcal{V} \subseteq \mathcal{X}$ a non-empty closed convex set. Assume (6.5) or (6.6) hold. Then, with the notation in Algorithm 6.1, for all $t$ we have that $x_{t+1}$ exists, it is unique, and it is in the interior of $\mathcal{X}$. Moreover, $\forall u \in \mathcal{V}$, the following inequality holds
$$\eta_t(\ell_t(x_t) - \ell_t(u)) \leq \eta_t \langle g_t, x_t - u \rangle \leq B_\psi(u; x_t) - B_\psi(u; x_{t+1}) - B_\psi(x_{t+1}; x_t) + \langle \eta_t g_t, x_t - x_{t+1} \rangle$$
$$\leq B_\psi(u; x_t) - B_\psi(u; x_{t+1}) + \frac{\eta_t^2}{2\lambda} \|g_t\|_\star^2.$$
Proof. First of all, in each round $x_{t+1}$ exists using Theorem 6.9 and the fact that $B_\psi(\cdot; x_t)$ is proper, closed, and strongly convex. Moreover, from Theorem 6.8, $x_t \in \operatorname{int} \mathcal{X}$ for all $t$.
Now, from the optimality condition in Theorem 2.8 for the update of OMD, we have
$$\langle \eta_t g_t + \nabla \psi(x_{t+1}) - \nabla \psi(x_t), u - x_{t+1} \rangle \geq 0, \quad \forall u \in \mathcal{V}. \quad (6.7)$$
Hence, we have that
$$\langle \eta_t g_t, x_t - u \rangle$$
$$= \langle \nabla \psi(x_t) - \nabla \psi(x_{t+1}) - \eta_t g_t, u - x_{t+1} \rangle + \langle \nabla \psi(x_{t+1}) - \nabla \psi(x_t), u - x_{t+1} \rangle + \langle \eta_t g_t, x_t - x_{t+1} \rangle$$
$$\leq \langle \nabla \psi(x_{t+1}) - \nabla \psi(x_t), u - x_{t+1} \rangle + \langle \eta_t g_t, x_t - x_{t+1} \rangle$$
$$= B_\psi(u; x_t) - B_\psi(u; x_{t+1}) - B_\psi(x_{t+1}; x_t) + \langle \eta_t g_t, x_t - x_{t+1} \rangle$$
$$\leq B_\psi(u; x_t) - B_\psi(u; x_{t+1}) - \frac{\lambda}{2} \|x_t - x_{t+1}\|^2 + \eta_t \|g_t\|_\star \|x_t - x_{t+1}\|$$
$$\leq B_\psi(u; x_t) - B_\psi(u; x_{t+1}) + \frac{\eta_t^2}{2\lambda} \|g_t\|_\star^2,$$

[Screenshot for page 56]
where in the second inequality we used (6.7), in the second equality we used Lemma 6.7, in the second inequality we used the definition of dual norm and (6.4) because $\psi$ is $\lambda$-strong convex with respect to $\|\cdot\|$, finally in the last inequality we used the fact that $ax - \frac{b}{2} x^2 \leq \frac{a^2}{2b}$ for all $x \in \mathbb{R}$ and $a, b > 0$.
The lower bound with the function values is due, as usual, to the definition of subgradients.
We now see how to use this one step relationship to prove a regret bound, that will finally show us if and when this entire construction is a good idea. In fact, it is worth stressing that the above motivation is not enough in any way to justify the existence of the OMD algorithm.
We can now prove a regret bound for OMD.
Theorem 6.11. Set $x_1 \in \mathcal{V}$ such that $\psi$ is differentiable in $x_1$. Assume $\eta_{t+1} \leq \eta_t$, $t = 1, \dots, T$. Then, under the assumptions of Lemma 6.10 and $\forall u \in \mathcal{V}$, the following regret bounds hold
$$\sum_{t=1}^T (\ell_t(x_t) - \ell_t(u)) \leq \frac{\max_{1\leq t \leq T} B_\psi(u; x_t)}{\eta_T} + \frac{1}{2\lambda} \sum_{t=1}^T \eta_t \|g_t\|_\star^2.$$
Moreover, if $\eta_t$ is constant, i.e., $\eta_t = \eta$ $\forall t = 1, \dots, T$, we have
$$\sum_{t=1}^T (\ell_t(x_t) - \ell_t(u)) \leq \frac{B_\psi(u; x_1)}{\eta} + \frac{\eta}{2\lambda} \sum_{t=1}^T \|g_t\|_\star^2.$$
Proof. Fix $u \in \mathcal{V}$. As in the proof of OGD, dividing the inequality in Lemma 6.10 by $\eta_t$ and summing from $t = 1, \dots, T$, we get
$$\sum_{t=1}^T (\ell_t(x_t) - \ell_t(u)) \leq \sum_{t=1}^T \left( \frac{1}{\eta_t} B_\psi(u; x_t) - \frac{1}{\eta_t} B_\psi(u; x_{t+1}) \right) + \sum_{t=1}^T \frac{\eta_t}{2\lambda} \|g_t\|_\star^2$$
$$= \frac{1}{\eta_1} B_\psi(u; x_1) - \frac{1}{\eta_T} B_\psi(u; x_{T+1}) + \sum_{t=1}^{T-1} \left( \frac{1}{\eta_{t+1}} - \frac{1}{\eta_t} \right) B_\psi(u; x_{t+1}) + \sum_{t=1}^T \frac{\eta_t}{2\lambda} \|g_t\|_\star^2$$
$$\leq \frac{1}{\eta_1} D^2 + D^2 \sum_{t=1}^{T-1} \left( \frac{1}{\eta_{t+1}} - \frac{1}{\eta_t} \right) + \sum_{t=1}^T \frac{\eta_t}{2\lambda} \|g_t\|_\star^2$$
$$= \frac{1}{\eta_1} D^2 + D^2 \left( \frac{1}{\eta_T} - \frac{1}{\eta_1} \right) + \sum_{t=1}^T \frac{\eta_t}{2\lambda} \|g_t\|_\star^2$$
$$= \frac{D^2}{\eta_T} + \sum_{t=1}^T \frac{\eta_t}{2\lambda} \|g_t\|_\star^2,$$
where we denoted by $D^2 = \max_{1\leq t \leq T} B_\psi(u; x_t)$.
The second statement is left as an exercise.
In words, OMD allows us to prove regret guarantees that depend on arbitrary couple of dual norms $\|\cdot\|$ and $\|\cdot\|_\star$. In particular, the primal norm will be used to measure the feasible set $\mathcal{V}$ or the distance between the competitor and the initial point, and the dual norm will be used to measure the gradients. If you happen to know something about these quantities, we can choose the most appropriate couple of norm to guarantee a small regret. The only thing you need is a function $\psi$ that is strongly convex with respect to the primal norm you have chosen.
Overall, the regret bound is still of the order of $\sqrt{T}$ for Lipschitz functions, that only difference is that now the Lipschitz constant is measured with a different norm. Also, everything we did for Online Subgradient Descent can be trivially used here. For example, we can slightly generalize the stepsizes we saw in Section 4.2 to
$$\eta_t = \frac{D}{\sqrt{2\lambda \sum_{i=1}^t \|g_i\|_\star^2}}$$

[Screenshot for page 57]
to achieve a regret upper bound of $\frac{D}{\sqrt{\lambda}} \sqrt{2 \sum_{t=1}^T \|g_t\|_\star^2}$, where we assume $D^2 = \max_{x, y \in \mathcal{V}} B_\psi(x; y) < \infty$.
In Sections 6.6 and 6.7, we will see practical examples of OMD that guarantee strictly better regret than OSD. As we did in the case of AdaGrad, the better guarantee will depend on the shape of the domain and the characteristics of the subgradients.
Next, we see the meaning of the "Mirror", but first we need another mathematical tool: Fenchel conjugates.

6.4.1 Convex Analysis Bits: Fenchel Conjugate
Definition 6.12 (Closed Function). A function $f : \mathcal{V} \subseteq \mathbb{R}^d \to [-\infty, +\infty]$ is closed iff $\{x : f(x) \leq \alpha\}$ is closed for every $\alpha \in \mathbb{R}$.
Note that in any Euclidean space (and more generally in any Hausdorff space) a function is closed iff it is lower semicontinuous [Bauschke and Combettes, 2017, Lemma 1.24].
Example 6.13. The indicator function of a set $\mathcal{V} \subset \mathbb{R}^d$, is closed iff $\mathcal{V}$ is closed.
Definition 6.14 (Fenchel Conjugate). For a function $f : \mathbb{R}^d \to [-\infty, \infty]$, we define the Fenchel conjugate $f^\star : \mathbb{R}^d \to [-\infty, \infty]$ as
$$f^\star(\theta) = \sup_{x \in \mathbb{R}^d} \langle \theta, x \rangle - f(x).$$
From the definition we immediately obtain the Fenchel-Young's inequality for proper functions:
$$\langle \theta, x \rangle \leq f(x) + f^\star(\theta), \quad \forall x, \theta \in \mathbb{R}^d.$$
Moreover, $f^\star$ is always convex and closed, regardless of the convexity of $f$ [Bauschke and Combettes, 2017, Proposition 13.13].
We have the following useful properties for the Fenchel conjugate.
Theorem 6.15 ([Rockafellar, 1970, Theorem 12.2]). Let $f$ be a convex function. Then, $f^\star$ is a closed convex function, proper iff $f$ is proper. Moreover, if $f$ is also closed then $f^{\star\star} = f$.
Theorem 6.16. Let $f : \mathbb{R}^d \to (-\infty, +\infty]$ be proper. Then, the following conditions are equivalent:
(a) $\theta \in \partial f(x)$.
(b) $\langle \theta, y \rangle - f(y)$ achieves its supremum in $y$ at $y = x$.
(c) $f(x) + f^\star(\theta) = \langle \theta, x \rangle$.
Moreover, if $f$ is also convex and closed, we have an additional equivalent condition
(d) $x \in \partial f^\star(\theta)$.
Proof. Let's prove (a) $\iff$ (b). From the definition of subgradient, we have
$$f(y) \geq f(x) + \langle \theta, y - x \rangle, \quad \forall y$$
that is
$$\langle \theta, x \rangle - f(x) \geq \langle \theta, y \rangle - f(y), \quad \forall y.$$
Then, (b) $\iff$ (c) by definition of $f^\star(\theta)$.
If $f$ is also convex and closed, then $f^{\star\star} = f$ is proper by Theorem 6.15. Hence, (c) is equivalent to $f^{\star\star}(x) + f^\star(\theta) = \langle \theta, x \rangle$, that is equivalent to (d) by following the same reasoning as above.

[Screenshot for page 58]
Figure 6.4: A geometric interpretation of the Fenchel conjugate.

In the following, we will not need a geometric intuition of the concept of Fenchel conjugates in order to use them. However, for some people geometric intuitions help remember better, so let's briefly discuss it. Let $f$ be convex, closed, and proper. Let $x \in \operatorname{dom} \partial f$ and use a subgradient $g_x \in \partial f(x)$ to construct a linear lower bound around $x$ passing through $f(x)$ as $\tilde{f}(y) := f(x) + \langle g_x, y - x \rangle$. Moreover, from the point (c) of the previous theorem, we have
$$\tilde{f}(0) = f(x) - \langle g_x, x \rangle = -f^\star(g).$$
Hence, we have $-f^\star(g)$ is the value of $\tilde{f}$ at $x = 0$, see Figure 6.4 (left). This is some times mentioned as a geometric intuition for Fenchel conjugates, but I do not find it very illuminating, so let's dig deeper.
We can show that we can recover $f$ as the point-wise maximum of all its tangents, see Figure 6.4 (right). In turn, the tangents can be expressed using only the knowledge of $f^\star$. Let's consider the family of linear functions $\tilde{f}_g(y) = -f^\star(g) + \langle g, y \rangle$ parametrized by a generic $g$. By Fenchel-Young inequality, we immediately have $\tilde{f}_g(y) \leq f(y)$. Moreover, from point (c) of the previous theorem, if $g \in \partial f(y)$, then $\tilde{f}_g(y) = f(y)$. This means that this family of functions lower bounds $f$ and it contains the tangents to $f$. Now, observe that for any $y$ we obtain
$$\sup_{g \in \mathbb{R}^d} \tilde{f}_g(y) = \sup_{g \in \mathbb{R}^d} -f^\star(g) + \langle g, y \rangle = f^{\star\star}(y) = f(y),$$
where in last equality we used Theorem 6.15. So, the supremum over this family of linear functions is equal to the function $f$. Overall, this means that the Fenchel conjugate allows us to reason about functions using their tangent hyperplanes, whose information is in the Fenchel conjugate, without losing anything.
Another way to quantify the above is the fact that the domain of $f^\star$ contains all the possible subgradients of $f$.
Corollary 6.17. Let $f : \mathbb{R}^d \to (-\infty, +\infty]$ proper. Then, $\{g \in \partial f(x) : x \in \mathbb{R}^d\} \subseteq \operatorname{dom} f^\star$.
Proof. From Theorem 6.16 (a) $\iff$ (c) and the fact that $\operatorname{dom} \partial f \subseteq \operatorname{dom} f$, we have
$$g \in \partial f(x) \iff f(x) + f^\star(g) = \langle g, x \rangle \Rightarrow f^\star(g) < +\infty.$$
Note that there are cases where the domain of $f^\star$ contains vectors that are not subgradients. For example, $f(x) = \exp(-x)$ and $f^\star(0) = \sup_x - \exp(-x) = 0 < \infty$ and $0$ is not a derivative of $f$.
Example 6.18. Let $f(x) = \exp(x)$, hence we have $f^\star(\theta) = \sup_x x\theta - \exp(x)$. Solving the optimization, we have that $x^\star = \ln(\theta)$ if $\theta > 0$. Hence, $f^\star(\theta) = \begin{cases} \theta \ln \theta - \theta, & \text{if } \theta > 0 \\ 0, & \text{if } \theta = 0 \\ +\infty, & \text{if } \theta < 0 \end{cases}$.

[Screenshot for page 59]
Example 6.19 (Conjugate of the inner product). Let $f(x) = \langle z, x \rangle$ where $z \neq 0 \in \mathbb{R}^d$. Then
$$f^\star(\theta) = \sup_{x \in \mathbb{R}^d} \langle \theta - z, x \rangle = \begin{cases} 0, & \theta = z \\ +\infty, & \text{otherwise.} \end{cases}$$
Example 6.20 (Conjugate of the hinge loss). Let $f(x) = \max(1 - \langle z, x \rangle, 0)$ where $z \in \mathbb{R}^d$ and let's calculate $f^\star(\theta)$. If $\theta$ has a component orthogonal to $z$, I can choose $x$ along that component and the supremum in the definition of $f^\star$ is $+\infty$. Hence, let's consider the case that $\theta = \alpha z$. In this case, we have
$$f^\star(\theta) = \sup_{x} \alpha \langle z, x \rangle - \max(1 - \langle z, x \rangle, 0) = \sup_u \alpha u - \max(1 - u, 0).$$
If $\alpha > 0$ or $\alpha < -1$, again the supremum is $+\infty$. Hence, we only need to consider the case that $-1 \leq \alpha \leq 0$. From a case analysis on $u$, it is easy to see that in this case the supremum is attained in $u = 1$. Putting all together, we have
$$f^\star(\theta) = \begin{cases} \alpha, & \text{if } \theta = \alpha z, \alpha \in [-1, 0] \\ +\infty, & \text{otherwise.} \end{cases}$$
Example 6.21 (Conjugate of squared norms). Consider the function $f(x) = \frac{1}{2}\|x\|^2$, where $\|\cdot\|$ is a norm in $\mathbb{R}^d$, with dual norm $\|\cdot\|_\star$. We can show that its conjugate is $f^\star(\theta) = \frac{1}{2}\|\theta\|_\star^2$. Let's see how. First, we have
$$\langle \theta, x \rangle - \frac{1}{2}\|x\|^2 \leq \|\theta\|_\star \|x\| - \frac{1}{2}\|x\|^2$$
for all $x$. The right hand side is a quadratic function of $\|x\|$, which has maximum value $\frac{1}{2}\|\theta\|_\star^2$. Therefore for all $x$, we have
$$\langle \theta, x \rangle - \frac{1}{2}\|x\|^2 \leq \frac{1}{2}\|\theta\|_\star^2,$$
which shows that $f^\star(\theta) \leq \frac{1}{2}\|\theta\|_\star^2$. To show the other inequality, let $x$ be any vector with $\langle \theta, x \rangle = \|\theta\|_\star \|x\|$, scaled so that $\|x\| = \|\theta\|_\star$. Then we have, for this $x$,
$$\langle \theta, x \rangle - \frac{1}{2}\|x\|^2 = \frac{1}{2}\|\theta\|_\star^2,$$
which shows that $f^\star(\theta) \geq \frac{1}{2}\|\theta\|_\star^2$.
Example 6.22 (Young's inequality). Let $p > 1$ and $f : \mathbb{R}_{\geq 0} \to \mathbb{R}_{\geq 0}$ defined as $f(x) = \frac{1}{p} x^p$. Let's calculate the conjugate. We have
$$f^\star(\theta) = \sup_{x \geq 0} \theta x - \frac{1}{p} x^p.$$
If $\theta < 0$, than the supremum is 0. For $\theta \geq 0$, by differentiation, we have that $x^\star = \theta^{\frac{1}{p-1}}$. So, we have
$$f^\star(\theta) = \begin{cases} 0, & \text{if } \theta < 0 \\ \frac{p-1}{p} \theta^{\frac{p}{p-1}}, & \text{if } \theta \geq 0. \end{cases}$$
Denoting by $q > 1$ the positive number such that $\frac{1}{p} + \frac{1}{q} = 1$, we can rewrite it as
$$f^\star(\theta) = \begin{cases} 0, & \text{if } \theta < 0 \\ \frac{1}{q} \theta^q, & \text{if } \theta \geq 0. \end{cases}$$
Using the Fenchel-Young inequality, for $x, y \geq 0$ and $p, q > 1$ such that $1/p + 1/q = 1$, we have
$$xy \leq \frac{1}{p} x^p + \frac{1}{q} y^q,$$
that is called Young's inequality.

[Screenshot for page 60]
Example 6.23 (Conjugate of norms). Consider the function $f(x) = \|x\|$, where $\|\cdot\|$ is a norm in $\mathbb{R}^d$, with dual norm $\|\cdot\|_\star$. Then, we have
$$f^\star(\theta) = \sup_{x \in \mathbb{R}^d} \langle \theta, x \rangle - \|x\| \leq \sup_{x \in \mathbb{R}^d} \|\theta\|_\star \|x\| - \|x\| = \begin{cases} 0, & \|\theta\|_\star \leq 1 \\ +\infty, & \|\theta\|_\star > 1. \end{cases}$$
To show the other inequality, let $x$ be any vector with $\langle \theta, x \rangle = \|\theta\|_\star \|x\|$, scaled so that $\|x\| = \alpha$. Then, we have
$$f^\star(\theta) = \sup_{x \in \mathbb{R}^d} \langle \theta, x \rangle - \|x\| \geq \sup_{\alpha \in \mathbb{R}_{\geq 0}} \alpha \|\theta\|_\star - \alpha = \begin{cases} 0, & \|\theta\|_\star \leq 1 \\ +\infty, & \|\theta\|_\star > 1. \end{cases}$$
Lemma 6.24. Let $f$ be a function and let $f^\star$ be its Fenchel conjugate. For $a > 0$ and $b \in \mathbb{R}$, the Fenchel conjugate of $h(x) = af(x) + b + \langle g, x \rangle$ is $h^\star(\theta) = af^\star((\theta - g)/a) - b$.
Proof. From the definition of conjugate function, we have
$$h^\star(\theta) = \sup_{x \in \mathbb{R}^d} \langle \theta - g, x \rangle - af(x) - b = -b + a \sup_{x \in \mathbb{R}^d} \left\langle \frac{\theta - g}{a}, x \right\rangle - f(x) = -b + af^\star\left(\frac{\theta - g}{a}\right).$$
Lemma 6.25. Let $f_1$ and $f_2$ such that $f_1(x) \leq f_2(x)$ for all $x$. Then, $f_1^\star(\theta) \geq f_2^\star(\theta)$ for all $\theta$.
Proof.
$$f_2^\star(\theta) = \sup_x \langle \theta, x \rangle - f_2(x) \leq \sup_x \langle \theta, x \rangle - f_1(x) = f_1^\star(\theta).$$
Lemma 6.26 ([Bauschke and Combettes, 2017, Example 13.8]). Let $f : \mathbb{R} \to (-\infty, +\infty]$ even, i.e., $f(x) = f(-x)$. Then $(f \circ \|\cdot\|_2)^\star = f^\star \circ \|\cdot\|_2$.
Remark 6.27. There is a tight connection between the dual function in convex optimization and the Fenchel conjugate. Indeed, let $f : \mathbb{R}^d \to (-\infty, +\infty]$ and consider the constrained optimization problem
$$\min_x f(x)$$
$$\text{s.t. } Ax \leq b$$
$$Cx = d,$$
where the inequality is component-wise and $A, C$ are matrices. The dual function is defined as
$$g(\lambda, \nu) = \inf_x (f(x) + \lambda^\top(Ax - b) + \nu^\top(Cx - d))$$
$$= -b^\top \lambda - d^\top \nu + \inf_x (f(x) + (A^\top \lambda + C^\top \nu)^\top x)$$
$$= -b^\top \lambda - d^\top \nu - f^\star(-A^\top \lambda - C^\top \nu).$$
For closed, convex, and proper functions, Theorem 6.16 implies that $x \in \partial f^\star(\theta)$ iff $\theta \in \partial f(x)$, that in words means that $(\partial f)^{-1} = \partial f^\star$ in the sense of multivalued mappings. Now, we show that for strongly convex functions the Fenchel conjugate is smooth and hence differentiable.
Theorem 6.28 (Duality Strong Convexity/Smoothness). Let $f : \mathbb{R}^d \to (-\infty, +\infty]$ be a proper, closed, convex function, and $\operatorname{dom} \partial f$ be non-empty. Then, $f$ is $\lambda > 0$ strongly convex with respect to $\|\cdot\|$ iff $f^\star$ is $\frac{1}{\lambda}$-smooth with respect to $\|\cdot\|_\star$ on $\mathbb{R}^d$.
Proof. Let's first prove the implication from left to right. First, let's show that $f^\star$ is differentiable. Since $f$ is proper, closed, and strongly convex, the maximizer of $\max_x \langle \theta, x \rangle - f(x)$ exists and it is unique by Theorem 6.9. Denote by $x^\star$ the argmax of this expression. Hence, from Theorem 6.16, we have $x^\star \in \partial f^\star(\theta)$. Let's now show that this is the only element in the subdifferential. Assume there exists $x' \in \partial f^\star(\theta)$, then from Theorem 6.16, we have
$$f^\star(\theta) = \langle \theta, x' \rangle - f(x')$$
but from the uniqueness of the maximizer we have that $x^\star = x'$.

[Screenshot for page 61]
Now, let's prove the that gradient of $f^\star$ is $\frac{1}{\lambda}$-Lipschitz with respect to $\|\cdot\|_\star$. For any $\theta_1$ and $\theta_2$, set $x_1 = \nabla f^\star(\theta_1)$ and $x_2 = \nabla f^\star(\theta_2)$. From Theorem 6.16, we have that $\theta_1 \in \partial f(x_1)$ and $\theta_2 \in \partial f(x_2)$. Hence, by Lemma 4.2, we have
$$f(x_2) \geq f(x_1) + \langle \theta_1, x_2 - x_1 \rangle + \frac{\lambda}{2} \|x_1 - x_2\|^2,$$
$$f(x_1) \geq f(x_2) + \langle \theta_2, x_1 - x_2 \rangle + \frac{\lambda}{2} \|x_1 - x_2\|^2.$$
Summing these two inequalities, we have
$$\|\theta_1 - \theta_2\|_\star \|x_1 - x_2\| \geq \langle \theta_2 - \theta_1, x_1 - x_2 \rangle \geq \lambda \|x_1 - x_2\|^2,$$
where in the first inequality we used the definition of dual norms. Solving the inequality we get that
$$\|\theta_1 - \theta_2\|_\star \geq \lambda \|x_1 - x_2\| = \lambda \|\nabla f^\star(\theta_1) - \nabla f^\star(\theta_2)\|.$$
Let's now prove the other direction. Assume that $f^\star$ is $\frac{1}{\lambda}$-smooth with respect to $\|\cdot\|_\star$ on $\mathbb{R}^d$. Set $y \in \operatorname{dom} \partial f$ and $u \in \partial f(y)$. Hence, by Theorem 6.16 and the differentiability of $f^\star$, we also have $y = \nabla f^\star(u)$. Define $\phi(\theta) := f^\star(\theta + u) - f^\star(u) - \langle \theta, \nabla f^\star(u) \rangle$. From the $\frac{1}{\lambda}$-smoothness and Lemma 4.23, we have $\phi(\theta) \leq \frac{1}{2\lambda} \|\theta\|_\star^2$.
From Lemma 6.25 and Example 6.21, we have that $\phi^\star(x) \geq \frac{\lambda}{2} \|x\|^2$. Let's now calculate $\phi^\star(x)$.
$$\phi^\star(x) = \sup_\theta \langle \theta, x \rangle - f^\star(\theta + u) + f^\star(u) + \langle \theta, \nabla f^\star(u) \rangle$$
$$= f^\star(u) - \langle u, x + \nabla f^\star(u) \rangle + \sup_v \langle v, x + \nabla f^\star(u) \rangle - f^\star(v)$$
$$= f^\star(u) - \langle u, x + \nabla f^\star(u) \rangle + f(x + \nabla f^\star(u))$$
$$= -\langle u, x \rangle - f(\nabla f^\star(u)) + f(x + \nabla f^\star(u)),$$
where we used Theorem 6.15 in the third equality and Theorem 6.16 in the last one. Putting all together, we have
$$f(x + y) - f(y) - \langle u, x \rangle \geq \frac{\lambda}{2} \|x\|^2, \quad \forall u \in \partial f(y).$$
We will also use the following theorem on the first-order optimality condition.
Theorem 6.29. Let $f : \mathbb{R}^d \to (-\infty, +\infty]$ proper. Then $x^\star \in \operatorname{argmin}_{x \in \mathbb{R}^d} f(x)$ iff $0 \in \partial f(x^\star)$.
Proof. We have that
$$x^\star \in \operatorname{argmin}_{x \in \mathbb{R}^d} f(x) \iff \forall y \in \mathbb{R}^d, f(y) \geq f(x^\star) = f(x^\star) + \langle 0, y - x^\star \rangle \iff 0 \in \partial f(x^\star).$$

6.4.2 The "Mirror" Interpretation
Here, we explain the "mirror" interpretation of OMD, using the following Theorem.
Theorem 6.30. Let $B_\psi$ the Bregman divergence with respect to $\psi : \mathcal{X} \to \mathbb{R}$, where $\psi$ is $\lambda > 0$ strongly convex and closed. Let $\mathcal{V} \subseteq \mathcal{X}$ a non-empty closed convex set and $x_t \in \mathcal{V}$. Define
$$x_{t+1} = \underset{x \in \mathcal{V}}{\operatorname{argmin}} \ \langle g_t, x \rangle + \frac{1}{\eta_t} B_\psi(x; x_t),$$
and assume $\psi$ to be differentiable in $x_t$ and $x_{t+1}$. Then, for any $g_t \in \mathbb{R}^d$, we have
$$x_{t+1} = \nabla \psi_{\mathcal{V}}^\star (\nabla \psi(x_t) - \eta_t g_t), \quad (6.8)$$
where $\psi_{\mathcal{V}}$ is the restriction of $\psi$ to $\mathcal{V}$, that is, $\psi_{\mathcal{V}} := \psi + i_{\mathcal{V}}$.

[Screenshot for page 62]
Figure 6.5: OMD update in terms of duality mappings.

Proof. We have that
$$x_{t+1} = \underset{x \in \mathcal{V}}{\operatorname{argmin}} \ \langle g_t, x \rangle + \frac{1}{\eta_t} B_\psi(x; x_t)$$
$$= \underset{x \in \mathcal{V}}{\operatorname{argmin}} \ \eta_t \langle g_t, x \rangle + B_\psi(x; x_t)$$
$$= \underset{x \in \mathcal{V}}{\operatorname{argmin}} \ \eta_t \langle g_t, x \rangle + \psi(x) - \psi(x_t) - \langle \nabla \psi(x_t), x - x_t \rangle$$
$$= \underset{x \in \mathcal{V}}{\operatorname{argmin}} \ \langle \eta_t g_t - \nabla \psi(x_t), x \rangle + \psi(x).$$
Now, we use the first-order optimality condition in Theorem 6.29, to have
$$0 \in \eta_t g_t + \nabla \psi(x_{t+1}) - \nabla \psi(x_t) + \partial i_{\mathcal{V}}(x_{t+1}),$$
that is
$$\nabla \psi(x_t) - \eta_t g_t \in (\nabla \psi + \partial i_{\mathcal{V}})(x_{t+1}) \subseteq \partial \psi_{\mathcal{V}}(x_{t+1}),$$
where in the last inclusion we used Theorem 2.22. Hence, from Theorem 6.16, we have
$$x_{t+1} \in \partial \psi_{\mathcal{V}}^\star (\nabla \psi(x_t) - \eta_t g_t).$$
Using that fact that $\psi_{\mathcal{V}} := \psi + i_{\mathcal{V}}$ is $\lambda$-strongly convex, proper, and closed, from Theorem 6.28 we have that $\partial \psi_{\mathcal{V}}^\star = \{\nabla \psi_{\mathcal{V}}^\star\}$. Hence,
$$x_{t+1} = \nabla \psi_{\mathcal{V}}^\star (\nabla \psi(x_t) - \eta_t g_t).$$
Let's explain what this theorem says. We said that Online Mirror Descent extends the Online Subgradient Descent method to non-Euclidean norms. Hence, the regret bound we proved contains dual norms, that measure the iterate and the gradients. We also said that it makes sense to use a dual norm to measure a gradient, because it is a natural way to measure how "big" is the linear functional $x \to \langle \nabla f(y), x \rangle$. In a more correct way, gradients actually live in the dual space, that is in a different space of the predictions $x_t$. Hence, we cannot sum iterates and gradients together, in the same way in which we cannot sum pears and apples together. So, why we were doing it in OSD? The reason is that in that case the dual space coincides with the primal space. But, it is a very particular case due to the fact that we used the $L_2$ norm. Instead, in the general case, iterates and gradients are in two different spaces.
So, in OMD we need a way to go from one space to the other. This is exactly the role of $\nabla \psi$ and $\nabla \psi_{\mathcal{V}}^\star$, that are called duality mappings. We can now understand that the theorem tells us that OMD takes the primal vector $x_t$, transforms it into a dual vector through $\nabla \psi$, does a subgradient descent step in the dual space, and finally transforms the vector back to the primal space through $\nabla \psi_{\mathcal{V}}^\star$. This reasoning is summarized in Figure 6.5.

[Screenshot for page 63]
Example 6.31. Let $\psi : \mathbb{R}^d \to \mathbb{R}$ equal to $\psi(x) = \frac{1}{2}\|x\|_2^2$ and $\mathcal{V} = \{x \in \mathbb{R}^d : \|x\|_2 \leq 1\}$. Define $\psi_{\mathcal{V}} = \psi + i_{\mathcal{V}}$. Then, we have
$$\psi_{\mathcal{V}}^\star(\theta) = \sup_{x \in \mathcal{V}} \langle \theta, x \rangle - \frac{1}{2}\|x\|_2^2.$$
Let's compute this conjugate. First of all, if $\theta = 0$ we have that $\psi_{\mathcal{V}}^\star(\theta) = 0$. So, in the following we assume $\theta \neq 0$. For any $x \in \mathcal{V}$ there exist $q$ and $\alpha$ such that $x = \alpha \frac{\theta}{\|\theta\|_2} + q$ and $\langle q, \theta \rangle = 0$. Hence, we have
$$\sup_{x \in \mathcal{V}} \langle \theta, x \rangle - \frac{1}{2}\|x\|_2^2 = \sup_{\alpha, q : \alpha \frac{\theta}{\|\theta\|_2} + q \in \mathcal{V}, \langle q, \theta \rangle = 0} \alpha \|\theta\|_2 - \frac{\alpha^2}{2} - \frac{1}{2}\|q\|_2^2 = \sup_{-1 \leq \alpha \leq 1} \alpha \|\theta\|_2 - \frac{\alpha^2}{2}.$$
Solving the constrained optimization problem, we have $\alpha^\star = \min(1, \|\theta\|_2)$. Hence, we have
$$\psi_{\mathcal{V}}^\star(\theta) = \begin{cases} \frac{1}{2}\|\theta\|_2^2, & \|\theta\|_2 \leq 1 \\ \|\theta\|_2 - \frac{1}{2}, & \|\theta\|_2 > 1 \end{cases}$$
that is finite everywhere and differentiable.
So, the two duality mappings are $\nabla \psi(x) = x$ and
$$\nabla \psi_{\mathcal{V}}^\star(\theta) = \begin{cases} \theta, & \|\theta\|_2 \leq 1 \\ \frac{\theta}{\|\theta\|_2}, & \|\theta\|_2 > 1 \end{cases} = \Pi_{\mathcal{V}}(\theta).$$
Using (6.8), we obtain exactly the update of projected online subgradient descent.

6.4.3 Yet Another Way to Write the Online Mirror Descent Update
There exists yet another way to write the update of OMD. This third method uses the concept of Bregman projections. Extending the definition of Euclidean projections, we can define the projection with respect to a Bregman divergence. Let $\Pi_{\mathcal{V}, \psi}$ be defined by
$$\Pi_{\mathcal{V}, \psi}(x) = \underset{y \in \mathcal{V}}{\operatorname{argmin}} B_\psi(y; x).$$
In the online learning literature, the OMD algorithm is typically presented with a two-step update: first, solving the argmin over the entire space and then projecting back over $\mathcal{V}$ with respect to the Bregman divergence. In the following, we show that most of the time the two-step update is equivalent to the one-step update in (6.8).
First, we prove a general theorem that allows to break the constrained minimization of functions in the minimization over the entire space plus and Bregman projection step.
Theorem 6.32. Let $f : \mathbb{R}^d \to (-\infty, +\infty]$ proper, closed, strictly convex, and differentiable in $\operatorname{int} \operatorname{dom} f$. Also, let $\mathcal{V} \subset \mathbb{R}^d$ a non-empty, closed convex set with $\mathcal{V} \cap \operatorname{dom} f \neq \emptyset$ and assume that $\tilde{y} = \operatorname{argmin}_{z \in \mathbb{R}^d} f(z)$ exists and $\tilde{y} \in \operatorname{int} \operatorname{dom} f$. Then, we have
1. $\operatorname{argmin}_{z \in \mathcal{V}} f(z)$ contains exactly one element.
2. $\operatorname{argmin}_{z \in \mathcal{V}} f(z) = \operatorname{argmin}_{z \in \mathcal{V}} B_f(z; \tilde{y})$.
Proof. For the first point, from [Bauschke and Combettes, 2017, Proposition 11.13] and the existence of $\tilde{y}$, we have that $f$ is coercive. So, from Bauschke and Combettes [2017, Proposition 11.15], the minimizer of $f$ in $\mathcal{V}$ exists. Given that $f$ is strictly convex, the minimizer must be unique too.
For the second point, denote by $y' = \operatorname{argmin}_{z \in \mathcal{V}} B_f(z; \tilde{y})$ and $y = \operatorname{argmin}_{z \in \mathcal{V}} f(z)$. From the definition of $y$, we have $f(y) \leq f(y')$. On the other hand, from the first-order optimality condition, we have $\nabla f(\tilde{y}) = 0$. So, we have
$$f(y') - f(\tilde{y}) = B_f(y'; \tilde{y}) \leq B_f(y; \tilde{y}) = f(y) - f(\tilde{y}),$$
that is $f(y') \leq f(y)$. Given that $f$ is strictly convex, $y' = y$.

[Screenshot for page 64]
Now, note that, if $\tilde{\psi}(x) = \psi(x) + \langle g, x \rangle$, then
$$B_{\tilde{\psi}}(x; y) = \tilde{\psi}(x) - \tilde{\psi}(y) - \langle \nabla \tilde{\psi}(y), x - y \rangle$$
$$= \psi(x) - \psi(y) - \langle g + \nabla \psi(y), x - y \rangle + \langle g, x - y \rangle$$
$$= \psi(x) - \psi(y) - \langle \nabla \psi(y), x - y \rangle$$
$$= B_\psi(x; y).$$
Now, define $\tilde{f}(x) = \langle \eta_t g_t, x \rangle + B_\psi(x; x_t)$, so that $\tilde{f}(x) = \psi(x) + \langle z, x \rangle + K$ for some $K \in \mathbb{R}$ and $z \in \mathbb{R}^d$. This implies that $B_{\tilde{f}}(x; y) = B_\psi(x; y)$. Hence, under the assumption of the above theorem, we have that $x_{t+1} = \operatorname{argmin}_{x \in \mathcal{V}} \langle g_t, x \rangle + \frac{1}{\eta_t} B_\psi(x; x_t)$ is equivalent to
$$\tilde{x}_{t+1} = \underset{x \in \mathbb{R}^d}{\operatorname{argmin}} \ \langle \eta_t g_t, x \rangle + B_\psi(x; x_t),$$
$$x_{t+1} = \underset{x \in \mathcal{V}}{\operatorname{argmin}} \ B_\psi(x; \tilde{x}_{t+1}).$$
The advantage of this update is that sometimes it gives two easier problems to solve rather than a single difficult one.

6.5 OMD Regret Bound using Local Norms
In Lemma 6.10, strong convexity basically tells us some minimum curvature in all the directions, that allows to upper bound the difference between $x_t$ and $x_{t+1}$. However, it turns out that we can still get a meaningful regret upper bound without this assumption. In particular, we can get an interesting expression for the regret that involves the use of local norms. We will use these ideas in the section on Multi-armed Bandits (Chapter 11).
Lemma 6.33. Let $B_\psi$ the Bregman divergence with respect to $\psi : \mathcal{X} \to \mathbb{R}$ and assume $\psi$ twice differentiable and with the Hessian positive definite in the interior of its domain. Let $\mathcal{V} \subseteq \mathcal{X}$ a non-empty closed convex set. Assume (6.5) or (6.6) to hold. Define $\|x\|_A := \sqrt{x^\top A x}$. Also, with the notation in Algorithm 6.1, assume $x_{t+1}$ and $\tilde{x}_{t+1} \in \operatorname{argmin}_{x \in \mathcal{X}} \langle g_t, x \rangle + \frac{1}{\eta_t} B_\psi(x; x_t)$ exist. Then, $\forall u \in \mathcal{V}$, there exists $z_t$ on the line segments between $x_t$ and $x_{t+1}$, and $z'_t$ on the line segments between $x_t$ and $\tilde{x}_{t+1}$, such that the following inequality holds
$$\eta_t(\ell_t(x_t) - \ell_t(u)) \leq \eta_t \langle g_t, x_t - u \rangle \leq B_\psi(u; x_t) - B_\psi(u; x_{t+1}) + \frac{\eta_t^2}{2} \min \left( \|g_t\|^2_{(\nabla^2 \psi(z_t))^{-1}}, \|g_t\|^2_{(\nabla^2 \psi(z'_t))^{-1}} \right).$$
Proof. First of all, from Theorem 6.8, $x_t$ and $\tilde{x}_t$ are in the interior of $\mathcal{X}$ for all $t \geq 1$. Then, from Lemma 6.10, we have
$$\langle \eta_t g_t, x_t - u \rangle \leq B_\psi(u; x_t) - B_\psi(u; x_{t+1}) - B_\psi(x_{t+1}; x_t) + \langle \eta_t g_t, x_t - x_{t+1} \rangle. \quad (6.9)$$
From the Taylor's theorem, we have said that $B_\psi(x_{t+1}; x_t) = \frac{1}{2}(x_{t+1} - x_t)^\top \nabla^2 \psi(z_t)(x_{t+1} - x_t)$ for some $z_t$ on the line segment between $x_t$ and $x_{t+1}$. Observe that this is $\frac{1}{2}\|x_{t+1} - x_t\|^2_{\nabla^2 \psi(z_t)}$ and it is indeed a norm because we assumed the Hessian of $\psi$ to be positive definite. Hence, by Fenchel-Young inequality and Examples 4.18 and 6.21, we have
$$\langle \eta_t g_t, x_t - x_{t+1} \rangle - B_\psi(x_{t+1}; x_t) \leq \frac{\eta_t^2}{2} \|g_t\|^2_{(\nabla^2 \psi(z_t))^{-1}} + \frac{1}{2}(x_{t+1} - x_t)^\top \nabla^2 \psi(z_t)(x_{t+1} - x_t) - B_\psi(x_{t+1}; x_t)$$
$$= \frac{\eta_t^2}{2} \|g_t\|^2_{(\nabla^2 \psi(z_t))^{-1}},$$
that gives the first term in the minimum.
For the second term in the minimum, we instead observe that
$$\langle \eta_t g_t, x_t - x_{t+1} \rangle - B_\psi(x_{t+1}; x_t) \leq \max_{x \in \mathcal{X}} \langle \eta_t g_t, x_t - x \rangle - B_\psi(x; x_t) = \langle \eta_t g_t, x_t - \tilde{x}_{t+1} \rangle - B_\psi(\tilde{x}_{t+1}; x_t).$$
Then, we proceed as in the first bound.
Despite the apparent more difficult formulation, the second term in the minimum is often easier to use, especially in constrained settings because $\tilde{x}_{t+1}$ is defined over $\mathcal{X}$ rather than over $\mathcal{V}$. Also, under the assumptions of Theorem 6.32, it is easy to recognize that $x_{t+1}$ is the Bregman projection of $\tilde{x}_{t+1}$ onto $\mathcal{V}$.

[Screenshot for page 65]
6.6 Example of OMD: Exponentiated Gradient

Algorithm 6.2 Exponentiated Gradient
Require: $\eta > 0$
1: Set $x_1 = [1/d, \dots, 1/d]$
2: for $t = 1$ to $T$ do
3: Output $x_t \in δ^{d-1}$
4: Pay $\ell_t(x_t)$ for $\ell_t : δ^{d-1} \to \mathbb{R}$ subdifferentiable in $δ^{d-1}$
5: Set $g_t \in \partial \ell_t(x_t)$
6: Set $x_{t+1, j} = \frac{x_{t, j} \exp(-\eta g_{t, j})}{\sum_{i=1}^d x_{t, i} \exp(-\eta g_{t, i})}$, $j = 1, \dots, d$
7: end for

Let $δ^{d-1} = \{x \in \mathbb{R}^d : x_i \geq 0, \|x\|_1 = 1\}$ the probability simplex and set $\mathcal{V} = δ^{d-1}$. So, in words, we want to output discrete probability distributions over $\mathbb{R}^d$. Also, let $\mathcal{X} = \mathbb{R}^d_{\geq 0}$ and $\psi(x) : \mathcal{X} \to \mathbb{R}$ defined as $\psi(x) = \sum_{i=1}^d x_i \ln x_i$, where we define $0 \ln(0) = 0$. Note that the restriction of $\psi$ to $\mathcal{V}$ is the negative entropy of the discrete distributions in $δ^{d-1}$. It is possible to verify that $\psi$ satisfies the first condition in Theorem 6.8, hence the update is well defined.
The Fenchel conjugate $\psi_{\mathcal{V}}^\star(\theta)$ is defined as
$$\psi_{\mathcal{V}}^\star(\theta) = \sup_{x \in \mathcal{V}} \langle \theta, x \rangle - \psi(x) = \sup_{x \in \mathcal{V}} \langle \theta, x \rangle - \sum_{i=1}^d x_i \ln x_i.$$
It is a constrained optimization problem, we could solve it using the Karush‚¬€œKuhn‚¬€œTucker conditions. However, there is a simpler way to do it: We will remove the probability simplex constraint rephrasing the problem over $d - 1$ variables. In fact, the maximization problem is equivalent to
$$\min_{x \in \mathbb{R}^{d-1}} \sum_{i=1}^{d-1} x_i \ln x_i + \left( 1 - \sum_{i=1}^{d-1} x_i \right) \ln \left( 1 - \sum_{i=1}^{d-1} x_i \right) - \sum_{i=1}^{d-1} \theta_i x_i - \theta_d \left( 1 - \sum_{i=1}^{d-1} x_i \right).$$
Note that the constraint on $x_1, \dots, x_{d-1}$ and $1 - \sum_{i=1}^{d-1} x_i$ to be non-negative is enforced by the domain of the logarithm. This is now an unconstrained concave optimization problem, so we can solve it equating the gradient of the objective function to zero. Hence, we have
$$\ln \frac{x_i}{1 - \sum_{j=1}^{d-1} x_j} = \theta_i - \theta_d, \quad i = 1, \dots, d - 1.$$
That is
$$x_i = \exp(\theta_i - \theta_d) \left( 1 - \sum_{j=1}^{d-1} x_j \right), \quad i = 1, \dots, d - 1. \quad (6.10)$$
Summing this equality over $i = 1, \dots, d - 1$, we obtain
$$\sum_{i=1}^{d-1} x_i = \sum_{i=1}^{d-1} \exp(\theta_i - \theta_d) \left( 1 - \sum_{j=1}^{d-1} x_j \right)$$
that can be solved to obtain
$$1 - \sum_{j=1}^{d-1} x_i = \frac{1}{1 + \sum_{j=1}^{d-1} \exp(\theta_j - \theta_d)}.$$

[Screenshot for page 66]
Substituting it back in (6.10), we have
$$x_i = \frac{\exp(\theta_i - \theta_d)}{1 + \sum_{j=1}^{d-1} \exp(\theta_j - \theta_d)} = \frac{\exp(\theta_i)}{\sum_{j=1}^d \exp(\theta_j)}, \quad i = 1, \dots, d. \quad (6.11)$$
Denoting with $\alpha = \sum_{i=1}^d \exp(\theta_i)$, and substituting in the definition of the conjugate function we get
$$\psi_{\mathcal{V}}^\star(\theta) = \sum_{i=1}^d \left( \frac{1}{\alpha} \theta_i \exp(\theta_i) - \frac{1}{\alpha} \exp(\theta_i) (\theta_i - \ln(\alpha)) \right) = \ln(\alpha) \frac{1}{\alpha} \sum_{i=1}^d \exp(\theta_i) = \ln(\alpha) = \ln \left( \sum_{i=1}^d \exp(\theta_i) \right).$$
We also have $(\nabla \psi_{\mathcal{V}}^\star(\theta))_j = \frac{\exp(\theta_j)}{\sum_{i=1}^d \exp(\theta_i)}$ and $(\nabla \psi(x))_j = \ln(x_j) + 1$ for $x \in \mathbb{R}^d_{>0}$. Note that we could have also derived the gradient of $\psi_{\mathcal{V}}^\star$ directly from (6.11), using Theorem 6.16.
Putting all together, we have the online mirror descent update rule for entropic distance generating function.
$$x_{t+1, j} = \frac{\exp(\ln x_{t, j} + 1 - \eta g_{t, j})}{\sum_{i=1}^d \exp(\ln x_{t, i} + 1 - \eta g_{t, i})} = \frac{x_{t, j} \exp(-\eta g_{t, j})}{\sum_{i=1}^d x_{t, i} \exp(-\eta g_{t, i})}.$$
The algorithm is summarized in Algorithm 6.2. This algorithm is called Exponentiated Gradient (EG) because in the update rule we take the component-wise exponential of the (sub)gradient vector.
Let's take a look at the regret bound we get. From Example 6.6, for all $x \in δ^{d-1}$ and $y \in \{x \in \mathbb{R}^d : x_i > 0, \|x\|_1 = 1\}$, we have $\operatorname{KL}(x; y) := B_\psi(x; y) = \sum_{i=1}^d x_i \ln \frac{x_i}{y_i}$, that is the KL divergence between the discrete distributions $x$ and $y$. Now, we prove the strong convexity of $\psi$, through a slightly more general statement. Setting $c_i = 1$ for $i = 1, \dots, d$ gives the result we need here.
Lemma 6.34. Let $c_1, \dots, c_d \in \mathbb{R}_{>0}$. Then, $\psi(x) = \sum_{i=1}^d c_i x_i \ln x_i$ is 1-strongly convex with respect to the $L_1$ norm defined over the set $\mathcal{K} = \{x \in \mathbb{R}^d : x_i > 0, \sum_{i=1}^d x_i/c_i = 1\}$.
Proof. Let $\phi(u) = (u - 1) \ln u - \frac{2(u-1)^2}{u+1}$ for $u > 0$. Observe that $\phi''(u) > 0$ for $u > 0$ so the function is convex. Moreover, $\phi(1) = \phi'(1) = 0$. So, we have $\phi(u) \geq \phi(1) + \phi'(1)(x - 1) = 0$ for all $u > 0$.
Using this inequality, we have
$$\langle \nabla \psi(x) - \nabla \psi(y), x - y \rangle = \sum_{i=1}^d c_i (x_i - y_i) \ln \frac{x_i}{y_i} = \sum_{i=1}^d c_i y_i \left( \frac{x_i}{y_i} - 1 \right) \ln \frac{x_i}{y_i} \geq \sum_{i=1}^d c_i \frac{2(x_i - y_i)^2}{x_i + y_i}$$
$$= \sum_{i=1}^d \frac{x_i + y_i}{2c_i} \left| \frac{x_i - y_i}{\frac{x_i + y_i}{2c_i}} \right|^2 \geq \left( \sum_{i=1}^d \frac{x_i + y_i}{2c_i} \frac{|x_i - y_i|}{\frac{x_i + y_i}{2c_i}} \right)^2 = \|x - y\|_1^2,$$
where in the last inequality we used Jensen's inequality because $\frac{x+y}{2c_i}$ is a valid probability distribution. Using Theorem 4.3 completes the proof.
Another thing to do is to decide the initial point $x_1$. A reasonable choice is to set $x_1$ to be the minimizer of $\psi$ in $\mathcal{V}$. Hence, we set $x_1 = [1/d, \dots, 1/d] \in \mathbb{R}^d$, because the uniform distribution minimizes the negative entropy. Given that the minimizer is in the interior of $\mathcal{X}$, $\nabla \psi(x_1) = 0$ and $B_\psi(u; x_1)$ is equal to $\psi(u) - \min_{x \in \mathcal{V}} \psi(x)$. So, we have $B_\psi(u; x_1) = \sum_{i=1}^d u_i \ln u_i + \ln d \leq \ln d$.
Putting all together, we have
$$\sum_{t=1}^T (\ell_t(x_t) - \ell_t(u)) \leq \frac{\ln d}{\eta} + \frac{\eta}{2} \sum_{t=1}^T \|g_t\|_\infty^2, \quad \forall u \in δ^{d-1}.$$
Assuming $\|g_t\|_\infty \leq L_\infty$, we can set $\eta = \sqrt{\frac{2 \ln d}{L_\infty^2 T}}$, to obtain that an upper bound to the regret of $\sqrt{2} L_\infty \sqrt{T \ln d}$.

[Screenshot for page 67]
Remark 6.35. Note that the time-varying version of OMD with entropic distance generating function would give rise to a vacuous bound, can you see why? In the next chapter, we will see how Follow-The-Regularized-Leader (FTRL) overcomes this issue using a time-varying regularizer rather than a time-varying learning rate.
We can also get a tighter bound using the local norms. Let's use the additional assumption that $g_{t, i} \geq 0$, for all $t = 1, \dots, T$ and $i = 1, \dots, d$. Summing the inequality of Lemma 6.33 from $t = 1$ to $T$, we have for all $u \in \mathcal{V}$ that
$$\sum_{t=1}^T \ell_t(x_t) - \sum_{t=1}^T \ell(u) \leq \frac{\ln d}{\eta} + \frac{\eta}{2} \sum_{t=1}^T \|g_t\|^2_{(\nabla^2 \psi(z'_t))^{-1}},$$
where $z'_t$ is on the line segment between $x_t$ and $\tilde{x}_{t+1}$. In this case, it is easy to calculate $\tilde{x}_{t+1, i}$ as $x_{t, i} \exp(-\eta g_{t, i})$ for $i = 1, \dots, d$. Moreover, $\nabla^2 \psi(z'_t)$ is a diagonal matrix whose elements on the diagonal are $\frac{1}{z'_{t, i}}$, $i = 1, \dots, d$. Hence, we have that
$$\|g_t\|^2_{(\nabla^2 \psi(z'_t))^{-1}} = \sum_{i=1}^d g_{t, i}^2 z'_{t, i} \leq \sum_{i=1}^d g_{t, i}^2 x_{t, i}.$$
Putting all together, the final bound would be
$$\sum_{t=1}^T \ell_t(x_t) - \sum_{t=1}^T \ell(u) \leq \frac{\ln d}{\eta} + \frac{\eta}{2} \sum_{t=1}^T \sum_{i=1}^d g_{t, i}^2 x_{t, i}, \quad \forall u \in δ^{d-1}.$$
This is indeed a tighter bound because $\sum_{i=1}^d g_{t, i}^2 x_{t, i} \leq \|g_t\|_\infty^2$.
How would Online Subgradient Descent (OSD) work on the same problem? First, it is important to realize that nothing prevents us to use OSD on this problem. We just have to implement the Euclidean projection onto the probability simplex, that does not have a closed formula but it can be implemented in $O(d \ln d)$ time, see Condat [2016].
The regret bound we would get from OSD is
$$\sum_{t=1}^T (\ell_t(x_t) - \ell_t(u)) \leq \frac{2}{\eta} + \frac{\eta}{2} \sum_{t=1}^T \|g_t\|_2^2 \leq \frac{2}{\eta} + \frac{\eta}{2} T L_2^2, \quad \forall u \in δ^{d-1},$$
where $L_2 \geq \|g_t\|_2$ for all $t$. Optimally tuning the learning rate, we get
$$\sum_{t=1}^T (\ell_t(x_t) - \ell_t(u)) \leq 2 L_2 \sqrt{T}, \quad \forall u \in δ^{d-1}.$$
To compare it with the bound the bound of Exponentiated Gradient (EG), it is sufficient to observe that
$$\frac{1}{\sqrt{d}} \|g_t\|_2 \leq \|g_t\|_\infty \leq \|g_t\|_2,$$
and the inequalities are tight. Hence, ignoring the numerical constants, the bound of EG is between $\frac{1}{\sqrt{d}}$ and $\sqrt{\frac{\ln d}{d}}$ times the one of OSD, depending on the structure of the subgradients $g_t$. Hence, in a worst case sense with respect to the infinity norm of the subgradients $g_t$, using an entropic distance generating function can transform a dependency on the dimension from $\sqrt{d}$ to $\sqrt{\ln d}$ for Online Convex Optimization (OCO) over the probability simplex. So, as we already saw analyzing AdaGrad, the shape of the domain and the structure of the subgradients are the important ingredients when we move from the Euclidean norm to other norms.

6.7 Example of OMD: $p$-norm Algorithms
Consider the distance generating function $\psi(x) = \frac{1}{2}\|x\|_p^2$, for $1 < p \leq 2$ over $\mathcal{X} = \mathcal{V} = \mathbb{R}^d$. Let's remind the reader that the $p$-norm of a vector $x$ is defined as $(\sum_{i=1}^d |x_i|^p)^{\frac{1}{p}}$. From Examples 4.17 and 6.21, we have that

[Screenshot for page 68]
$\psi_{\mathcal{V}}^\star(\theta) = \frac{1}{2}\|\theta\|_q^2$, where $\frac{1}{p} + \frac{1}{q} = 1$, so that $q \geq 2$. Let's calculate the dual maps: $(\nabla \psi(x))_j = \operatorname{sign}(x_j)|x_j|^{p-1} \|x\|_p^{2-p}$ and $(\nabla \psi_{\mathcal{V}}^\star(x))_j = \operatorname{sign}(x_j)|x_j|^{q-1} \|x\|_q^{2-q}$. Hence, we can write the update rule as
$$\tilde{x}_{t+1, j} = \operatorname{sign}(x_{t, j})|x_{t, j}|^{p-1} \|x_t\|_p^{2-p} - \eta g_{t, j}, \quad j = 1, \dots, d,$$
$$x_{t+1, j} = \operatorname{sign}(\tilde{x}_{t+1, j})|\tilde{x}_{t+1, j}|^{q-1} \|\tilde{x}_{t+1}\|_q^{2-q}, \quad j = 1, \dots, d,$$
where we broke the update in two steps to simplify the notation (and the implementation). Starting from $x_1 = 0$, we have that
$$B_\psi(u; x_1) = \psi(u) - \psi(x_1) - \langle \nabla \psi(x_1), u - x_1 \rangle = \psi(u).$$
The last ingredient is the fact that $\psi(x)$ is $p - 1$ strongly convex with respect to $\|\cdot\|_p$.
Lemma 6.36. $\psi(x) = \frac{1}{2}\|x\|_p^2$ is $(p - 1)$-strongly convex with respect to $\|\cdot\|_p$, for $1 \leq p \leq 2$.
Proof. Observe that for $p = 1$ the function is convex, hence it is 0-strongly convex with respect to any norm. So, we now assume $p > 1$.
We want to show that
$$\psi(y) \geq \psi(x) + \langle \nabla \psi(x), y - x \rangle + \frac{p - 1}{2} \|x - y\|_p^2, \quad \forall x, y \in \mathbb{R}^d. \quad (6.12)$$
Rather than studying $\psi$ directly, we study the Hessian of a surrogate function: $\psi_a(x) = \sum_{i=1}^d (x_i^2 + a)^{p/2} / p$ where $a > 0$. Observe that $\psi_a$ is continuously twice differentiable for $a > 0$. To do it, let $f : \mathbb{R} \to \mathbb{R}$ convex and twice differentiable and $h : \mathbb{R}^d \to \mathbb{R}$ differentiable and define $\psi(x) = f(h(x))$. So, we have
$$\nabla^2 f(x) = f''(h(x))\nabla h(x) \nabla h(x)^\top + f'(h(x))\nabla^2 h(x) \succeq f'(h(x))\nabla^2 h(x),$$
where in the inequality we used the fact that $f$ is convex, so $f''(x) \geq 0$ for all $x \in \mathbb{R}$.
Now, let's specialize the above result to $h(x) = \sum_{i=1}^d (x_i^2 + a)^{p/2}$ and $f(x) = \frac{1}{2}x^{2/p}$ for $x_i \neq 0$. So, $\nabla^2 h(x)$ is a diagonal matrix and the element $i$ on the diagonal is
$$p(x_i^2 + a)^{p/2-1} + p(p - 2)x_i^2 (x_i^2 + a)^{p/2-2} = p(x_i^2 + a)^{p/2-1} \left( 1 + (p - 2) x_i^2 (x_i^2 + a)^{-1} \right)$$
$$\geq p(x_i^2 + a)^{p/2-1} (1 + (p - 2))$$
$$= p(p - 1)(x_i^2 + a)^{p/2-1},$$
where in the inequality we have used the fact that $a > 0$ and $p \leq 2$. Hence, we have
$$\nabla^2 \psi_a(x) \succeq \frac{1}{p} \sum_{i=1}^d (x_i^2 + a)^{p/2} {}^{2/p-1} \operatorname{diag} \{p(p - 1)(x_1^2 + a)^{p/2-1}, \dots, p(p - 1)(x_d^2 + a)^{p/2-1}\}.$$
Hence, denoting $w_i = (x_i^2 + a)^{(2-p)/p} > 0$, we have
$$\langle \nabla^2 \psi_a(x) y, y \rangle \geq (p - 1) \sum_{i=1}^d (x_i^2 + a)^{p/2} {}^{(2-p)/p} \sum_{i=1}^d (x_i^2 + a)^{p/2-1} y_i^2$$
$$= (p - 1) \left[ \sum_{i=1}^d (x_i^2 + a)^{p/2} {}^{(2-p)/2} \sum_{i=1}^d (x_i^2 + a)^{p/2-1} y_i^2 \right]^{2/p}$$
$$= (p - 1) \left[ \sum_{i=1}^d w_i^{2/(2-p)} {}^{(2-p)/2} \left( \sum_{i=1}^d \frac{y_i^2}{w_i^p} \right)^{p/2} \right]^{2/p}$$
$$\geq (p - 1) \sum_{i=1}^d w_i \left( \frac{y_i^p}{w_i} \right)^{2/p} = (p - 1) \|y\|_p^2,$$

[Screenshot for page 69]
where we used Hölder inequality with dual norms $\|\cdot\|_p^2$ and $\|\cdot\|_{2/(2-p)}^2$.
Hence, we have that
$$\psi_a(y) \geq \psi_a(x) + \langle \nabla \psi_a(x), y - x \rangle + \frac{p - 1}{2} \|x - y\|_p^2, \quad \forall x, y \in \mathbb{R}^d.$$
Now, given that $\psi_a$ and $\nabla \psi_a$ are continuous in $a$, taking the limit for $a \to 0^+$, we get (6.12). By Lemma 4.2, this implies the strong convexity of $\psi$.
Hence, the regret bound will be
$$\sum_{t=1}^T (\ell_t(w_t) - \ell_t(u)) \leq \frac{\|u\|_p^2}{2\eta} + \frac{\eta}{2(p - 1)} \sum_{t=1}^T \|g_t\|_q^2.$$
Setting $p = 2$, we get the (unprojected) Online Subgradient Descent. However, we can set $p$ to achieve a logarithmic dependency in the dimension $d$ as in EG. Let's assume again that $\|g_t\|_\infty \leq L_\infty$, so we have
$$\sum_{t=1}^T \|g_t\|_q^2 \leq L_\infty^2 d^{2/q} T.$$
Also, note that $\|u\|_p \leq \|u\|_1$, so we have an upper bound to the regret of
$$\operatorname{Regret}_T(u) \leq \frac{\|u\|_1^2}{2\eta} + \frac{L_\infty^2 d^{2/q} T \eta}{2(p - 1)}, \quad \forall u \in \mathbb{R}^d.$$
Setting $\eta = \frac{\|u\|_1 \sqrt{p-1}}{L_\infty d^{1/q} \sqrt{T}}$, we get an upper bound to the regret of
$$\frac{1}{2} \left( \frac{\|u\|_1^2}{\alpha} + \alpha \right) L_\infty \sqrt{T} d^{1/q} \sqrt{p - 1} = \frac{1}{2} \left( \frac{\|u\|_1^2}{\alpha} + \alpha \right) L_\infty \sqrt{T} \sqrt{q - 1} d^{1/q} \leq \frac{1}{2} \left( \frac{\|u\|_1^2}{\alpha} + \alpha \right) L_\infty \sqrt{T} \sqrt{q d^{1/q}}.$$
Assuming $d \geq 3$, the choice of $q$ that minimizes the last term is $q = 2 \ln d$ that makes the term $\sqrt{q d^{1/q}} = \sqrt{2e \ln d}$.
Hence, we have regret bound of the order of $\mathcal{O}(\sqrt{T \ln d})$ as $T \to \infty$.
So, the $p$-norm allows to interpolate from the behaviour of OSD to the one of EG. Note that here the set $\mathcal{V}$ is the entire space, however we could still set $\mathcal{V} = \{x \in \mathbb{R}^d : x_i \geq 0, \|x\|_1 = 1\}$. While this would allow us to get the same asymptotic bound of EG, the update would not be in a closed form anymore.

6.8 Application: Learning with Expert Advice
Let's introduce a particular Online Convex Optimization (OCO) game called Learning with Expert Advice (LEA).
In this setting, we have $d$ experts that gives us some advice on each round. In turn, in each round we have to decide which expert we want to follow. After we made our choice, the losses associated to each expert are revealed and we pay the loss associated to the expert we picked. The aim of the game is to minimize the losses we make compared to cumulative losses of the best expert. This is a general setting that allows to model many interesting cases. For example, we have a number of different online learning algorithms and we would like to choose to the best among them.
Is this problem solvable? If we put ourselves in the adversarial setting, unfortunately it cannot be solved! Indeed, even with 2 experts, the adversary can force on us linear regret. Let's see how. In each round we have to pick expert 1 or expert 2. In each round, the adversary can decide that the expert we pick has loss 1 and the other one has loss 0. This means that the cumulative loss of the algorithm over $T$ rounds is $T$. On the other hand, the best cumulative loss over expert 1 and 2 is less than $T/2$. This means that our regret, no matter what we do, can be as big as $T/2$.
The problem above is due to the fact that the adversary has too much power. One way to reduce its power is using randomization. We can allow the algorithm to be randomized and force the adversary to decide the losses at

[Screenshot for page 70]
time $t$ without knowing the outcome of the randomization of the algorithm at time $t$ (but it can depend on the past randomization). This is enough to make the problem solvable. Another view to look at it is that randomization makes the problem convex, allowing us to use any OCO algorithm on it.
First, let's write the problem in the original formulation. We set a discrete feasible set $\mathcal{V} = \{e_i\}_{i=1}^d$, where $e_i$ is the vector will all zeros but a 1 in the coordinate $i$. Our predictions and the competitor are from $\mathcal{V}$. The losses are linear losses: $\ell_t(x) = \langle g_t, x \rangle$, for $t = 1, \dots, T$ and $i = 1, \dots, d$. The regret is
$$\operatorname{Regret}_T(e_i) = \sum_{t=1}^T \langle g_t, x_t \rangle - \sum_{t=1}^T \langle g_t, e_i \rangle, \quad i = 1, \dots, d. \quad (6.13)$$
The only thing that makes this problem non-convex is the feasibility set, that is clearly a non-convex one.
Let's now see how the randomization makes this problem convex. Let's extend the feasible set to $\mathcal{V}' = \{x \in \mathbb{R}^d : x_i \geq 0, \|x\|_1 = 1\}$. Note that $e_i \in \mathcal{V}'$. For this problem we can use an OCO algorithm to minimize the regret
$$\operatorname{Regret}'_T(u) = \sum_{t=1}^T \langle g_t, x_t \rangle - \sum_{t=1}^T \langle g_t, u \rangle, \quad \forall u \in \mathcal{V}'.$$
Can we find a way to transform an upper bound to this regret to the one we care in (6.13)? One way is the following one: On each time step, construct a random variable $A_t$ that is equal to $i$ with probability $x_{t, i}$ for $i = 1, \dots, d$. Then, select the expert according to the outcome of $A_t$. Now, using the law of total expectation, we have
$$\mathbb{E}[g_{t, A_t}] = \mathbb{E}_{A_1, \dots, A_{t-1}}[\mathbb{E}_{A_t}[g_{t, A_t} | A_1, \dots, A_{t-1}]] = \mathbb{E}[\langle g_t, x_t \rangle],$$
and
$$\mathbb{E}[\operatorname{Regret}_T(e_i)] = \mathbb{E}[\operatorname{Regret}'_T(e_i)] = \mathbb{E}\left[ \sum_{t=1}^T \langle g_t, x_t \rangle - \sum_{t=1}^T \langle g_t, e_i \rangle \right], \quad \forall i \in \{1, \dots, d\}.$$
This means that we can minimize in expectation the non-convex regret with a randomized OCO algorithm. We can summarize this reasoning in Algorithm 6.3.

Algorithm 6.3 Learning with Expert Advice through Randomization
Require: $x_1 \in \{x \in \mathbb{R}^d : x_i \geq 0, \|x\|_1 = 1\}$
1: for $t = 1$ to $T$ do
2: Draw $A_t$ according to $\mathbb{P}\{A_t = i\} = x_{t, i}$
3: Select expert $A_t$
4: Observe all the experts' losses $g_t$ and pay the loss of the selected expert
5: Update $x_t$ with an OCO algorithm with feasible set $\{x \in \mathbb{R}^d : x_i \geq 0, \|x\|_1 = 1\}$
6: end for

For example, assume that $\|g_t\|_\infty \leq L_\infty$ for all $t$. Then, using the EG algorithm from Section 6.6, we obtain the following update rule
$$x_{t+1, j} = \frac{x_{t, j} \exp(-\eta g_{t, j})}{\sum_{i=1}^d x_{t, i} \exp(-\eta g_{t, i})}, \quad j = 1, \dots, d,$$
where setting $x_1 = [1/d, \dots, 1/d]$ and $\eta = \frac{\sqrt{2 \ln d}}{L_\infty \sqrt{T}}$. For such algorithm, the regret will be
$$\mathbb{E}[\operatorname{Regret}_T(e_i)] \leq \frac{\sqrt{2}}{2} L_\infty \sqrt{T \ln d}, \quad \forall i \in \{1, \dots, d\}.$$
It is worth stressing the importance of the result just obtained: We can design an algorithm that in expectation is close to the best expert in a set, paying only a logarithmic penalty in the size of the set. The $p$-norm Algorithm in Section 6.7 would give a similar guarantee.
Later, in Section 10.6.1, we will see algorithms that achieve the even better regret guarantee of $\mathcal{O}(\sqrt{T \cdot \operatorname{KL}(u; x_1)})$ as $T \to \infty$, for any $u$ in the probability simplex. You should be able to convince yourself that no setting of $\eta$ in EG allows to achieve such regret guarantee. Indeed, these algorithms will be based on a very different strategy.

[Screenshot for page 71]
6.9 Application: Combining Online Algorithms to Adapt to the Learning Rate
In our analysis of Online Subgradient Descent (OSD), we have seen that the choice of the learning rate $\eta_t$ is critical for performance. For instance, with a constant learning rate $\eta$, the optimal choice that minimizes the regret bound is $\eta^\star = \frac{\|u - x_1\|_2}{\sqrt{\sum_{t=1}^T \|g_t\|_2^2}}$, which unfortunately depends on the competitor $u$ and the entire sequence of future gradients. One might be tempted to use a grid of learning rates and select the best one in hindsight, but unfortunately this is not a valid online learning procedure.
In this section, we demonstrate how to use the LEA framework to design a meta-algorithm that automatically adapts to the best learning rate from a given set, paying only a small price in the regret. The core idea is to treat each instance of an online learning algorithm with a fixed learning rate as an "expert". We then use a controller algorithm, such as Exponentiated Gradient (EG), to combine the predictions of these experts. The resulting ensemble algorithm will have a regret guarantee that is close to the regret of the best expert‚¬€and thus the best learning rate‚¬€in hindsight.
Let us consider running $N$ parallel instances of the OSD algorithm, each with a different learning rate $\eta^{(i)}$ for $i = 1, \dots, N$. At each round $t$, each OSD instance $i$ produces a prediction $x_t^{(i)}$. We can view these $N$ predictions as advice from $N$ experts. Our goal is to combine them into a single prediction $x_t$ that performs nearly as well as the best prediction $x_t^{(i^\star)}$ from the best OSD instance $i^\star$.
A straightforward approach would be to compute the loss $\ell_t(x_t^{(i)})$ for each expert $i$ and use this as the loss vector for the EG controller algorithm. The EG algorithm would then produce weights $p_{t+1, i}$ to form the next combined prediction $x_{t+1} = \sum_{i=1}^N p_{t+1, i} x_{t+1}^{(i)}$. However, this would require computing $N$ separate subgradients $g_t^{(i)} \in \partial \ell_t(x_t^{(i)})$ at each round, which can be computationally expensive.
To create an efficient algorithm that requires only one subgradient evaluation per round, we can use the linearization technique from Section 2.3. The controller algorithm forms its combined prediction $x_t = \sum_{i=1}^N p_{t, i} x_t^{(i)}$. We then receive a single subgradient $g_t \in \partial \ell_t(x_t)$ at this combined point. This single subgradient is then used to define a linear surrogate loss, $\tilde{\ell}_t(x) = \langle g_t, x \rangle$, which is passed to all expert algorithms and to the controller algorithm.
Because all experts receive the same linear loss function, they all use the same subgradient $g_t$ for their updates. The loss for the $i$-th expert, used by the controller EG algorithm, is simply $\langle g_t, x_t^{(i)} \rangle$. This procedure is summarized in Algorithm 6.4.

Algorithm 6.4 Combining OSD instances with EG
Require: Non-empty, closed, convex set $\mathcal{V} \subseteq \mathbb{R}^d$, initial point $x_1 \in \mathcal{V}$, set of $N$ learning rates $\{\eta^{(1)}, \dots, \eta^{(N)}\}$, EG learning rate $\beta > 0$
Initialize $N$ copies of OSD: $x_1^{(i)} = x_1$ for $i = 1, \dots, N$
Initialize EG weights: $p_{1, i} = 1/N$ for $i = 1, \dots, N$
for $t = 1$ to $T$ do
Output combined prediction $x_t = \sum_{i=1}^N p_{t, i} x_t^{(i)}$
Pay $\ell_t(x_t)$ for $\ell_t : \mathcal{V} \to \mathbb{R}$ subdifferentiable in $\mathcal{V}$
Set $g_t \in \partial \ell_t(x_t)$
Update each OSD instance: $x_{t+1}^{(i)} = \Pi_{\mathcal{V}}(x_t^{(i)} - \eta^{(i)} g_t)$ for $i = 1, \dots, N$
Define loss vector for EG: $z_{t, i} = \langle g_t, x_t^{(i)} \rangle$ for $i = 1, \dots, N$
Define $\tilde{p}_{t+1, i} = p_{t, i} \exp(-\beta z_{t, i})$ for $i = 1, \dots, N$
Update EG weights: $p_{t+1, i} = \frac{\tilde{p}_{t+1, i}}{\sum_{i=1}^d \tilde{p}_{t+1, i}}$ for $i = 1, \dots, N$
end for

We can now prove a regret bound for this ensemble algorithm.
Theorem 6.37. Let $\mathcal{V} \subseteq \mathbb{R}^d$ be a non-empty closed convex set. Assume the losses $\ell_t$ are convex and $L$-Lipschitz w.r.t. $\|\cdot\|_2$, and the $L_2$ diameter of $\mathcal{V}$ is $D$. Let $\{\eta^{(i)}\}_{i=1}^N$ be the set of learning rates for the OSD experts and $\beta = \frac{\sqrt{2 \ln N}}{2LD \sqrt{T}}$.

[Screenshot for page 72]
Then, the regret of Algorithm 6.4 is bounded as
$$\sum_{t=1}^T (\ell_t(x_t) - \ell_t(u)) \leq \min_{i=1, \dots, N} \left( \frac{\|u - x_1\|_2^2}{2\eta^{(i)}} + \frac{\eta^{(i)}}{2} L^2 T \right) + 2LD \sqrt{2T \ln N}, \quad \forall u \in \mathcal{V}.$$
Proof. By the convexity of the losses $\ell_t$, the regret of the ensemble algorithm can be bounded by the regret on the linearized losses:
$$\sum_{t=1}^T (\ell_t(x_t) - \ell_t(u)) \leq \sum_{t=1}^T \langle g_t, x_t - u \rangle.$$
Let $i^\star = \operatorname{argmin}_{i=1, \dots, N} \sum_{t=1}^T \langle g_t, x_t^{(i)} \rangle$ be the index of the best expert in hindsight for the sequence of linearized losses. We can decompose the regret as
$$\sum_{t=1}^T \langle g_t, x_t - u \rangle = \sum_{t=1}^T \langle g_t, x_t - x_t^{(i^\star)} \rangle + \sum_{t=1}^T \langle g_t, x_t^{(i^\star)} - u \rangle.$$
The second term is the regret of the best OSD expert on the sequence of linear losses $\langle g_t, \cdot \rangle$. From the regret bound of OSD with constant learning rate (Theorem 2.13), we have
$$\sum_{t=1}^T \langle g_t, x_t^{(i^\star)} - u \rangle \leq \min_{i=1, \dots, N} \left( \frac{\|u - x_1\|_2^2}{2\eta^{(i)}} + \frac{\eta^{(i)}}{2} \sum_{t=1}^T \|g_t\|_2^2 \right).$$
For the first term, we have
$$\sum_{t=1}^T \langle g_t, x_t - x_t^{(i^\star)} \rangle = \sum_{t=1}^T \left\langle g_t, \sum_{i=1}^N p_{t, i} x_t^{(i)} - x_t^{(i^\star)} \right\rangle = \sum_{t=1}^T \left( \sum_{i=1}^N p_{t, i} \langle g_t, x_t^{(i)} \rangle - \langle g_t, x_t^{(i^\star)} \rangle \right).$$
This is the regret of the EG master algorithm against the best expert $i^\star$ on the sequence of loss vectors $z_{t, i} = \langle g_t, x_t^{(i)} \rangle$. Since $\ell_t$ is $L$-Lipschitz and $\mathcal{V}$ has diameter $D$, we have $|\langle g_t, x_t^{(i)} - x_t^{(j)} \rangle| \leq \|g_t\|_2 \|x_t^{(i)} - x_t^{(j)}\|_2 \leq LD$. The range of losses for EG is thus at most $2LD$. Using the regret bound for EG, we get
$$\sum_{t=1}^T \langle g_t, x_t - x_t^{(i^\star)} \rangle \leq 2LD \sqrt{2T \ln N}.$$
Combining the bounds for both terms and using $\|g_t\|_2 \leq L$ completes the proof.
This result demonstrates that by combining OSD instances, we can achieve a regret that is close to the one obtained by the best learning rate in the chosen set. The additional price for this adaptivity is the term $2LD \sqrt{2T \ln N}$, which is logarithmic in the number of experts $N$. While the regret is not better than if we would have just tuned the learning rate with the knowledge of $D$ is a single OSD algorithm, we can expect it to work better, especially if $D$ is large.
This technique provides a principled and effective method for automating the selection of learning rates in an online fashion. However, it can only be used in bounded domains, where the worst-case optimal learning rate is known. This limitation is due to the fact that we need the losses passed to EG to be Lipschitz. In Chapter 10, we will see how to obtain parameter-free algorithms for unbounded domains, without the need to combine a number of base online learners.

6.10 Optimistic OMD
Till now, we have mainly considered the adversarial model as our model of the environment. This allowed us to design algorithm that work in this setting, as well as in other more benign settings. However, the world is never completely adversarial. So, we might be tempted to model the environment in some way, but that would leave our

[Screenshot for page 73]
algorithm vulnerable to attacks. An alternative, is to consider the data as generated by some predictable process plus an adversarial signal. In this view, it might be beneficial to try to model the predictable part, without compromising the robustness to the adversarial signal.
In this section, we will explore this possibility through a particular version of OMD, where we predict the next gradient. In very intuitive terms, if our predicted gradient is correct, we can expect the regret to decrease. However, if our prediction is wrong we still want to recover the worst case guarantee. Such algorithm is called Optimistic OMD.
The core idea of Optimistic OMD is to predict the next gradient and use it in the update rule, as summarized in Algorithm 6.5. Here, at round $t$ the algorithm receives a hint $\tilde{g}_{t+1}$ on the next subgradient $g_{t+1}$ and uses it to construct the update. At the same time, you have to remove the hint you used at the previous time step, $\tilde{g}_t$. Note that for the sake of the analysis, it does not matter how the prediction is generated. It can be even generated by another online learning procedure!

Algorithm 6.5 Optimistic Online Mirror Descent
Require: Non-empty closed convex $\mathcal{V} \subset \mathcal{X} \subseteq \mathbb{R}^d$, $\psi : \mathcal{X} \to \mathbb{R}$ strictly convex and differentiable on $\operatorname{int} \mathcal{X}$, $x_1 \in \operatorname{int} \mathcal{X}$, $\eta_1, \dots, \eta_T > 0$
1: $\tilde{g}_1 = 0$
2: for $t = 1$ to $T$ do
3: Output $x_t$
4: Pay $\ell_t(x_t)$ for $\ell_t : \mathcal{V} \to \mathbb{R}$ subdifferentiable in $\mathcal{V}$
5: Set $g_t \in \partial \ell_t(x_t)$
6: Predict next subgradient $\tilde{g}_{t+1} \in \mathbb{R}^d$
7: $x_{t+1} \in \operatorname{argmin}_{x \in \mathcal{V}} \langle g_t - \tilde{g}_t + \tilde{g}_{t+1}, x \rangle + \frac{1}{\eta_t} B_\psi(x; x_t)$
8: end for

To gain some intuition on why this update makes sense, consider the case that $\psi(x) = \frac{1}{2}\|x\|_2^2$, $\eta_t = \eta$, and $\mathcal{V} = \mathbb{R}^d$. In this case, $x_{t+1} = x_t + \eta \tilde{g}_t - \eta g_t - \eta \tilde{g}_{t+1}$. Unrolling the update, we get $x_{t+1} = x_1 - \eta(\tilde{g}_{t+1} + \sum_{i=1}^t g_i)$. Without hints, that is in plain OMD, under the same assumptions the unrolled update would be $x_{t+1} = x_1 - \eta \sum_{i=1}^t g_i$ and $x_{t+2} = x_1 - \eta \sum_{i=1}^{t+1} g_i$. Hence, $\tilde{g}_{t+1}$ acts as a proxy for the next (unknown) subgradient $g_t$.
Note that one might be tempted to multiply $\tilde{g}_t$ by $\eta_t^{-1}$, because in the previous iteration we used the learning rate $\eta_{t-1}$. However, the OMD proof reveals that the correct way to see the update is to think the learning rate as attached to the Bregman divergence rather than to the subgradients.
One might also be tempted to find a way to study this algorithm with a special proof. However, the one-step lemma we proved for OMD is essentially tight: we only used two inequalities, one to deal with the set $\mathcal{V}$ and the other one to linearize the losses. But, but steps can be made tight, considering $\mathcal{V} = \mathbb{R}^d$ and linear losses. Hence, if the update is just OMD with a different sequence of subgradients, the proof must follow from the one of OMD with a different set of subgradients. This is a general rule: If we have a theorem based on a tight inequality, any other proof of the same theorem, no matter how complex, must be looser or in the best case equivalent.
Note that setting $\tilde{g}_1 = 0$ is not a limitation because setting it to any other value would be equivalent to changing the arbitrary initial point $x_1$.
Theorem 6.38. Let $B_\psi$ the Bregman divergence with respect to $\psi : \mathcal{X} \to \mathbb{R}$ and assume $\psi$ to be proper, closed, and $\lambda$-strongly convex with respect to $\|\cdot\|$ in $\mathcal{V}$. Let $\mathcal{V} \subseteq \mathcal{X}$ a non-empty closed convex set. With the notation in Algorithm 6.5, assume $x_{t+1}$ exists, and it is in $\operatorname{int} \mathcal{X}$.
Assume $\eta_{t+1} \leq \eta_t$, $t = 1, \dots, T$. Then, and $\forall u \in \mathcal{V}$, the following regret bounds hold
$$\sum_{t=1}^T (\ell_t(x_t) - \ell_t(u)) \leq \frac{\max_{1\leq t \leq T} B_\psi(u; x_t)}{\eta_T} + \sum_{t=1}^T \left( \langle g_t - \tilde{g}_t, x_t - x_{t+1} \rangle - \frac{1}{\eta_t} B_\psi(x_{t+1}; x_t) \right)$$
$$\leq \frac{\max_{1\leq t \leq T} B_\psi(u; x_t)}{\eta_T} + \frac{1}{2\lambda} \sum_{t=1}^T \eta_t \|g_t - \tilde{g}_t\|_\star^2.$$

[Screenshot for page 74]
Moreover, if $\eta_t$ is constant, i.e., $\eta_t = \eta$ $\forall t = 1, \dots, T$, we have
$$\sum_{t=1}^T (\ell_t(x_t) - \ell_t(u)) \leq \frac{B_\psi(u; x_1)}{\eta} + \sum_{t=1}^T \left( \langle g_t - \tilde{g}_t, x_t - x_{t+1} \rangle - \frac{1}{\eta} B_\psi(x_{t+1}; x_t) \right)$$
$$\leq \frac{B_\psi(u; x_1)}{\eta} + \frac{\eta}{2\lambda} \sum_{t=1}^T \|g_t - \tilde{g}_t\|_\star^2.$$
Proof. We can use Lemma 6.10 with $g_t \to g_t - \tilde{g}_t + \tilde{g}_{t+1}$, to have
$$\langle g_t - \tilde{g}_t + \tilde{g}_{t+1}, x_t - u \rangle \leq \frac{1}{\eta_t} (B_\psi(u; x_t) - B_\psi(u; x_{t+1}) - B_\psi(x_{t+1}; x_t)) + \langle g_t - \tilde{g}_t + \tilde{g}_{t+1}, x_t - x_{t+1} \rangle.$$
Summing over $t = 1, \dots, T$ the l.h.s., we obtain
$$\sum_{t=1}^T \langle g_t - \tilde{g}_t + \tilde{g}_{t+1}, x_t - u \rangle = \sum_{t=1}^T \langle g_t, x_t - u \rangle + \sum_{t=1}^T \langle \tilde{g}_{t+1} - \tilde{g}_t, x_t - u \rangle$$
$$= \sum_{t=1}^T \langle g_t, x_t - u \rangle + \langle \tilde{g}_1 - \tilde{g}_{T+1}, u \rangle + \sum_{t=1}^T \langle \tilde{g}_{t+1} - \tilde{g}_t, x_t \rangle.$$
Summing the r.h.s., we have that
$$\sum_{t=1}^T \langle g_t - \tilde{g}_t + \tilde{g}_{t+1}, x_t - x_{t+1} \rangle = \sum_{t=1}^T \langle g_t - \tilde{g}_t, x_t - x_{t+1} \rangle + \sum_{t=1}^T \langle \tilde{g}_{t+1}, x_t - x_{t+1} \rangle.$$
Finally, observe that
$$\sum_{t=1}^T \langle \tilde{g}_{t+1} - \tilde{g}_t, x_t \rangle - \sum_{t=1}^T \langle \tilde{g}_{t+1}, x_t - x_{t+1} \rangle = \sum_{t=1}^T (\langle \tilde{g}_{t+1}, x_{t+1} \rangle - \langle \tilde{g}_t, x_t \rangle) = \langle \tilde{g}_{T+1}, x_{T+1} \rangle - \langle \tilde{g}_1, x_1 \rangle.$$
Given that the regret on the rounds $t = 1, \dots, T$ does not depend on $\tilde{g}_{T+1}$, we can safely set it to 0.
We defer the discussion of applications of optimistic algorithms to section on Optimistic FTRL in Section 7.12.

6.11 History Bits
The Bregman divergence was introduced by Bregman [1967] as a particular example of a distance-like function satisfying certain properties, to generalize the cyclic projection algorithm to general topological vector spaces. Often people drop the condition on the strict convexity [e.g., Bauschke et al., 2003] but in reality it is part of the original definition by Bregman [1967].
Mirror Descent (MD) was introduced by Nemirovskij and Yudin [1983] in the offline setting. The description of MD with Bregman divergence that I described here (with minor changes) was done by Beck and Teboulle [2003]. The minor changes are in decoupling the domain $\mathcal{X}$ of $\psi$ from the feasibility set $\mathcal{V}$. This allows to use functions $\psi$ that do not satisfy the condition (6.5) but they satisfy (6.6). In the online setting, the mirror descent scheme was used for the first time by Warmuth and Jagota [1997].
Most of the online learning literature for OMD assumes $\psi$ to be Legendre [see, e.g., Cesa-Bianchi and Lugosi, 2006] that corresponds to assuming (6.5) (or $\lim_{x \to \operatorname{bdry} \mathcal{X}} \|\nabla \psi(x)\|_2 = +\infty$, see [Rockafellar, 1970, Theorem 26.1 and Lemma 26.2]). This condition allows to prove that $\nabla \psi_{\mathcal{V}}^\star = (\nabla \psi_{\mathcal{V}})^{-1}$. However, it turns out that the Legendre condition is not necessary and we only need the function $\psi$ to be differentiable on the predictions $x_t$. For example, we only need one of the two conditions in (6.5) or (6.6) to hold. Removing the Legendre assumption makes it easier to use OMD with different combinations of feasibility sets/Bregman divergences. So, I did not introduce the concept

[Screenshot for page 75]
of Legendre functions at all, relying instead on (a minor modification of) OMD as described by Beck and Teboulle [2003]. Theorem 6.8 is derived from [Bauschke and Borwein, 1997, Theorem 3.12].
The proof of Theorem 6.28 is based on the one in Kakade et al. [2009].
The local norms were introduced in Abernethy et al. [2008] for Follow-The-Regularized-Leader with self-concordant regularizers.
The EG algorithm was introduced by Kivinen and Warmuth [1997], but not as a specific instantiation of OMD. Beck and Teboulle [2003] rediscover EG for the offline case as an example of Mirror Descent. Later, Cesa-Bianchi and Lugosi [2006] show that EG is just an instantiation of OMD. The $p$-norm algorithms for online prediction were originally introduced by Grove et al. [1997, 2001]. Lemma 6.36 is well-known, but I could not find a good proof for it, [1] so I wrote one. The trick to set $q = 2 \ln d$ is from Gentile and Littlestone [1999], Gentile [2003] (online learning) and apparently rediscovered in Ben-Tal et al. [2001] (optimization). The LEA setting was introduced by Littlestone and Warmuth [1994] and Vovk [1990]. The ideas in Algorithm 6.3 are based on the Multiplicative Weights algorithm [Littlestone and Warmuth, 1994] and the Hedge algorithm [Freund and Schapire, 1995, 1997]. As a side note, the weighted majority algorithm was also discovered independently in the game theory literature by Fudenberg and Levine [1995]. For two experts with losses in $[0, 1]$, Cover [1965] showed that the minimax regret is $\sqrt{T/2\pi}$ and proposed an algorithm achieving it. Notably, the approach in Cover [1965] is based on online betting. On the other hand, for more than 2 experts and losses in $[0, 1]$, the minimax regret is $(1 + o(1)) \sqrt{\frac{T \ln d}{2}}$, where $o(1) \to 0$ when $d, T \to \infty$ [Cesa-Bianchi et al., 1993, 1997]. By now, the literature on LEA is huge, with tons of variations over algorithms and settings.
The construction in Section 6.9 is inspired by the one in the MetaGrad algorithm [van Erven and Koolen, 2016, van Erven et al., 2021].
The idea of "hallucinating" future losses used in Optimistic OMD is originally from Azoury and Warmuth [2001] in the Forward Algorithm. Apparently, this idea was forgotten and rediscovered by Chiang et al. [2012] that used the previous loss function as an estimate of the next one, showing smaller regret in the case that the losses have small temporal variation. Later, Rakhlin and Sridharan [2013b] generalized this idea in the Optimistic OMD algorithm. Surprisingly enough, the procedure using two Optimistic OGD algorithms to solve saddle-point problems was already proposed by Popov [1980], see also Section 12.6. Optimistic OMD was proposed in Rakhlin and Sridharan [2013b] with a two-step update. It was then simplified to the one-step updates I presented here by Joulani et al. [2017]. However, Malitsky [2015] presented a version of Popov's algorithm for variational inequalities with only one projection that is essentially Optimistic OGD with one projection. The proof I present here is based on the one I proposed for Optimistic FTRL in Section 7.12.

6.12 Exercises
Problem 6.1. Historically, the negative entropy is associated with the use of the probability simplex as feasible set. However, it can also be applied to other feasible sets. So, derive a closed form update for OMD when using $\psi$ of Example 6.6 and $\mathcal{V} = \mathbb{R}^d_{\geq 0}$, that is the non-negative orthant.
Problem 6.2. Prove the three-points equality for Bregman divergences in Lemma 6.7.
Problem 6.3. Let $A \in \mathbb{R}^{d \times d}$ a positive definite matrix. Define $\|x\|_A^2 = x^\top A x$. Prove that $\frac{1}{2}\|x - y\|_A^2$ is the Bregman divergence $B_\psi(x; y)$ associated with $\psi(x) = \frac{1}{2}\|x\|_A^2$.
Problem 6.4. Let $\psi : \mathcal{X} \to \mathbb{R}$ a valid distance generating function and $D_\psi$ its associated Bregman divergence. Fix $y \in \operatorname{int} \mathcal{X}$ and define $f(x) = D_\psi(x, y)$. For any $v \in \operatorname{int} \mathcal{X}$ and $x \in \mathcal{X}$, prove that $D_f(x; v) = D_\psi(x; v)$.
Problem 6.5. We saw the Fenchel-Young inequality: $\langle \theta, x \rangle \leq f(x) + f^\star(\theta)$. Now, we want to show an equality, quantifying the gap in the inequality with a Bregman divergence term. Assume that $f$ and $f^\star$ are differentiable, $f$ strictly convex, and $\operatorname{dom} f = \mathbb{R}^d$. Prove that
$$f(x) + f^\star(\theta) = \langle \theta, x \rangle + B_f(x; \nabla f^\star(\theta)).$$
[1] People often cite [Shalev-Shwartz, 2007, Lemma 17], but it has a wrong proof because it ignores the fact that the function is not twice differentiable.

[Screenshot for page 76]
Problem 6.6. Let $f : \mathbb{R}^d \to (-\infty, +\infty]$ be even. Prove that $f^\star$ is even
Problem 6.7. In proof of Online Mirror Descent, we have the terms
$$-B_\psi(x_{t+1}; x_t) + \langle \eta_t g_t, x_t - x_{t+1} \rangle.$$
Prove that they can be lower bounded by $B_\psi(x_t; x_{t+1})$.
Problem 6.8. Generalize the concept of strong convexity to Bregman functions, instead of norms, and prove a logarithmic regret guarantee for such functions using OMD.
Problem 6.9. Derive the EG update rule and regret bound in the case that the algorithm starts from an arbitrary vector $x_1$ in the probability simplex.
Problem 6.10. Show that EG is invariant to additive constants added to the loss vectors. Use this observation to show that the terms $\sum_{i=1}^d g_{t, i}^2 x_{t, i}$ for $t = 1, \dots, T$ in the regret upper bound can be tightened to $\sum_{i=1}^d (g_{t, i} - m_t)^2 x_{t, i}$ for any $m_t \in \mathbb{R}$.
Problem 6.11. Extend Theorem 5.1 to arbitrary norms, measuring the diameter with respect to a norm $\|\cdot\|$ and considering losses $L$-Lipschitz with respect to the dual norm $\|\cdot\|_\star$.
Problem 6.12. In this problem, we will tackle Online Non-Convex Optimization. Assume that $\mathcal{V} \subset \mathbb{R}^d$ is the feasible set and it is convex and bounded. The losses $\ell_t : \mathbb{R}^d \to [0, 1]$ are non-convex and 1-Lipschitz with respect to $\|\cdot\|_2$. Prove that there exists a randomized algorithm that achieves sublinear regret on this problem, assuming knowledge of the total number of rounds $T$. Hint: Aim for something like $\mathbb{E}[\operatorname{Regret}_T(u)] = \mathcal{O}(\sqrt{dT \ln T})$ and do not worry about efficiency of the algorithm.

[Screenshot for page 77]
Chapter 7
Follow-The-Regularized-Leader
Till now, we focused only on Online Subgradient Descent and its generalization, Online Mirror Descent, with a brief ad-hoc analysis of a Follow-The-Leader (FTL) analysis in the first chapter. In this chapter, we will extend FTL to a powerful and generic algorithm to do online convex optimization: Follow-The-Regularized-Leader (FTRL).
FTRL is a very intuitive algorithm: At each time step it will play the minimizer of the sum of the past losses plus a time-varying regularization. We will see that the regularization is needed to make the algorithm "more stable" with linear losses and avoid the jumping back and forth that we saw in Example 2.10.

7.1 The Follow-the-Regularized-Leader Algorithm

Algorithm 7.1 Follow-the-Regularized-Leader Algorithm
Require: A sequence of regularizers $\psi_1, \dots, \psi_T : \mathcal{X} \to \mathbb{R}$, closed non-empty set $\mathcal{V} \subseteq \mathcal{X} \subseteq \mathbb{R}^d$
1: for $t = 1$ to $T$ do
2: Output $x_t \in \operatorname{argmin}_{x \in \mathcal{V}} \psi_t(x) + \sum_{i=1}^{t-1} \ell_i(x)$
3: Receive $\ell_t : \mathcal{V} \to \mathbb{R}$ and pay $\ell_t(x_t)$
4: end for


