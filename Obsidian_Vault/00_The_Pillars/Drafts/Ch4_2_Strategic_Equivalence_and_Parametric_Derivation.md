---
type: Draft
status: Open
related_pillar: "[[Ch4_Empirical_Study]]"
tags: [thesis, chapter_4, draft]
created: 2026-05-30
---

# Section 4.2: Strategic Equivalence and Parametric Derivation

## 1. The Strategic Degrees of Freedom in $2 \times 2$ Zero-Sum Games
To systematically map the stability of the probability simplex, we require the ability to construct arbitrary game matrices with a known, predetermined Nash Equilibrium $(x^*, y^*)$. A general $2 \times 2$ zero-sum matrix game is defined by four payoff coefficients:
$$ A = \begin{pmatrix} a & b \\ c & d \end{pmatrix} $$
However, the strategic behavior of the game—and the coordinates of its equilibrium—does not depend on all four parameters independently. 

Under standard game theory, two payoff matrices are strategically equivalent if they yield the same best-response correspondences, thereby preserving the exact coordinates of the Nash Equilibrium. In zero-sum games, this equivalence is invariant under positive affine transformations of the players' utilities. This invariance manifests in two properties:

1. **Column/Row Translation Invariance:** Adding a constant $C_j$ to all elements of column $j$ alters the baseline expected payoff of the column player, but does not alter the relative payoff difference between playing action 1 and action 2. The same principle applies to row translations. Subtracting $c$ from column 1 and $d$ from column 2 yields the strategically equivalent game matrix:
   $$ A' = \begin{pmatrix} a - c & b - d \\ 0 & 0 \end{pmatrix} $$
2. **Positive Scaling Invariance:** Multiplying the entire matrix by a positive scalar $\alpha > 0$ scales the expected utilities but preserves all relative preference inequalities.

Applying these transformations reduces the four original payoff coefficients $(a, b, c, d)$ to exactly **two strategic degrees of freedom**. These two degrees of freedom represent the essential parameters required to uniquely position the Nash Equilibrium $(x^*, y^*)$ anywhere within the interior of the strategy simplex.

---

## 2. Derivation of the Parameterized Matrix Family $A_{\lambda, \gamma}$
We derive the parameterized game family $A_{\lambda, \gamma}$ from first principles using the Nash Equilibrium indifference conditions. Let $x^* = (p, 1-p)^T \in \Delta^2$ and $y^* = (q, 1-q)^T \in \Delta^2$ represent the equilibrium strategies. At equilibrium, the players must be indifferent between their actions:
$$ (x^*)^T A \begin{pmatrix} 1 \\ 0 \end{pmatrix} = (x^*)^T A \begin{pmatrix} 0 \\ 1 \end{pmatrix} \quad \text{and} \quad \begin{pmatrix} 1 & 0 \end{pmatrix} A y^* = \begin{pmatrix} 0 & 1 \end{pmatrix} A y^* $$

Substituting the strategy vectors into the payoff matrix $A$ yields the system of algebraic equations:
$$ \begin{cases} a p + c(1-p) = b p + d(1-p) \\ a q + b(1-q) = c q + d(1-q) \end{cases} $$

Grouping the coefficients:
$$ \begin{cases} p(a - b - c + d) = d - c \\ q(a - b - c + d) = d - b \end{cases} $$

Let $S = a - b - c + d > 0$ denote the positive scaling factor. Solving for the equilibrium coordinates yields:
$$ p = \frac{d - c}{S} \quad \text{and} \quad q = \frac{d - b}{S} $$

To resolve the remaining degrees of freedom, we fix the reference payoff $d = 1$. The off-diagonal entries $b$ and $c$ correspond directly to our two strategic degrees of freedom. By mapping these entries to independent parameters, setting $b = \lambda$ and $c = \gamma$, the system simplifies to:
$$ p = \frac{1 - \gamma}{S} \quad \text{and} \quad q = \frac{1 - \lambda}{S} $$

For a normalized scale factor $S = 1$, the Nash Equilibrium coordinates map directly to the parameters:
$$ p = 1 - \gamma \quad \text{and} \quad q = 1 - \lambda $$

This mapping provides complete, decoupled control over the equilibrium's proximity to the simplex boundaries:
* Pushing $\gamma \to 0$ forces $p \to 1$, shifting the $x$-player's strategy to the boundary.
* Pushing $\lambda \to 1$ forces $q \to 0$, shifting the $y$-player's strategy to the boundary.

Substituting these relations back into the scaling factor equation $a = S - 1 + b + c$ yields $a = S - 1 + \lambda + \gamma$. The resulting parameterized matrix family $A_{\lambda, \gamma}$ is defined as:
$$ A_{\lambda, \gamma} = \begin{pmatrix} S - 1 + \lambda + \gamma & \lambda \\ \gamma & 1 \end{pmatrix} $$

### Generalization of the Literature Baseline
Crucially, the family $A_{\lambda, \gamma}$ is a direct generalization of the baseline hard instance $A_\delta$ studied by Cai et al. (2025). By restricting the parameter space to $\gamma = 0$ and $\lambda = 1/2$, and setting the scaling factor $S = 1 + \delta$, the matrix simplifies to:
$$ A_{1/2, 0} = \begin{pmatrix} (1 + \delta) - 1 + \frac{1}{2} + 0 & \frac{1}{2} \\ 0 & 1 \end{pmatrix} = \begin{pmatrix} \frac{1}{2} + \delta & \frac{1}{2} \\ 0 & 1 \end{pmatrix} = A_\delta $$

While $A_\delta$ only pushes the $x$-player's strategy to the boundary as $\delta \to 0$, our generalized family $A_{\lambda, \gamma}$ allows for the simultaneous boundary positioning of both players' strategies by independently tuning $\lambda \to 1$ and $\gamma \to 0$.

---

## 3. Affine Utility Normalization
Although the parameterization $A_{\lambda, \gamma}$ provides analytical control over the equilibrium, sweeping the parameters $\lambda$ and $\gamma$ near their extreme bounds can cause the matrix entries to become negative or exceed $1$. However, the convergence guarantees of online learning dynamics like OMWU require the utility values to be strictly bounded within the interval $[0, 1]$.

To satisfy this theoretical constraint without altering the underlying game dynamics, we apply a positive affine normalization. Let $A$ represent the raw matrix generated by the parameterization, with $m = \min_{i,j}(A_{i,j})$ and $M = \max_{i,j}(A_{i,j})$. We define the normalization mapping $f: \mathbb{R} \to [0, 1]$ as:
$$ f(x) = \frac{x - m}{M - m} $$

The normalized matrix $A_{\text{norm}}$ is constructed by applying $f(x)$ to each coefficient:
$$ (A_{\text{norm}})_{i,j} = \frac{A_{i,j} - m}{M - m} $$

### Invariance Proof of the Nash Equilibrium
We prove that this transformation preserves the Nash Equilibrium coordinates exactly. Let $\alpha = \frac{1}{M-m} > 0$ and $\beta = -\frac{m}{M-m}$. The transformation is represented as $f(x) = \alpha x + \beta$. 

Applying this transformation to the indifference condition of the row player under $A_{\text{norm}}$ yields:
$$ f(a)q + f(b)(1-q) = f(c)q + f(d)(1-q) $$

Substituting $f(x) = \alpha x + \beta$:
$$ (\alpha a + \beta)q + (\alpha b + \beta)(1-q) = (\alpha c + \beta)q + (\alpha d + \beta)(1-q) $$

Expanding the terms:
$$ \alpha \left[ a q + b(1-q) \right] + \beta \left[ q + (1-q) \right] = \alpha \left[ c q + d(1-q) \right] + \beta \left[ q + (1-q) \right] $$

Since $q + (1-q) = 1$, the constant term $\beta$ cancels from both sides:
$$ \alpha \left[ a q + b(1-q) \right] = \alpha \left[ c q + d(1-q) \right] $$

Because the scaling factor is strictly positive ($\alpha > 0$), we divide by $\alpha$ to recover the original indifference condition:
$$ a q + b(1-q) = c q + d(1-q) $$

This confirms that the Nash Equilibrium $(x^*, y^*)$ is invariant under the normalization. The transformation guarantees that the matrix entries strictly satisfy the $[0, 1]$ bound required for convergence proofs while preserving the exact spatial coordinates of the target equilibrium.
