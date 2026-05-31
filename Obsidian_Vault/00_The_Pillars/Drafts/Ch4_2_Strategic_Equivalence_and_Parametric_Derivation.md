---
type: Draft
status: Open
related_pillar: "[[Ch4_Empirical_Study]]"
tags: [thesis, chapter_4, draft]
created: 2026-05-31
---

# Section 4.2: Strategic Equivalence and Parametric Derivation

To exhaustively test OMWU's sensitivity to the minimum equilibrium probability $\delta$, we require the ability to construct arbitrary game matrices with a known, predetermined Nash Equilibrium $(x^*, y^*)$. 

Given a $2 \times 2$ matrix $A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$, the Nash equilibrium $(x^*, y^*)$ must satisfy the indifference conditions. For the $y$-player facing $x^*$, and the $x$-player facing $y^*$, these conditions are respectively:
$$ (x^*)^T A \begin{pmatrix} 1 \\ 0 \end{pmatrix} = (x^*)^T A \begin{pmatrix} 0 \\ 1 \end{pmatrix} \quad \text{and} \quad \begin{pmatrix} 1 & 0 \end{pmatrix} A y^* = \begin{pmatrix} 0 & 1 \end{pmatrix} A y^* $$

Letting $x^* = (p, 1-p)$ and $y^* = (q, 1-q)$, this yields the system of equations:
$$ \begin{cases} a p + c(1-p) = b p + d(1-p), \\ a q + b(1-q) = c q + d(1-q) \end{cases} $$

Solving for $p$ and $q$, we obtain explicit expressions for the Nash equilibrium based on the matrix coefficients:
$$ p = \frac{d-c}{a-b-c+d} \quad \text{and} \quad q = \frac{d-b}{a-b-c+d} $$

To generate a parameterized family of games whose Nash equilibria can be freely moved, we map the equilibrium probabilities to independent parameters $\lambda$ and $\gamma$. Letting $S = a - b - c + d$ be the scaling factor, we have:
$$ p = \frac{d-c}{S} \quad \text{and} \quad q = \frac{d-b}{S} $$

We have one remaining degree of freedom. By arbitrarily fixing $d=1$, we can map the off-diagonal elements directly to our parameters by setting $b = \lambda$ and $c = \gamma$. Substituting these into the scaling factor equation yields $a = S - 1 + \lambda + \gamma$. This gives the formulation for the parameterized matrix family $A_{\lambda, \gamma, S}$:
$$ A_{\lambda, \gamma, S} = \begin{pmatrix} S - 1 + \lambda + \gamma & \lambda \\ \gamma & 1 \end{pmatrix} $$

This construction guarantees that the unique Nash equilibrium is exactly $(p, 1-p)$ and $(q, 1-q)$, where:
$$ p = \frac{1-\gamma}{S} \quad \text{and} \quad q = \frac{1-\lambda}{S} $$

---

## The Degrees of Freedom in $2 \times 2$ Zero-Sum Games
Although the matrix $A_{\lambda, \gamma, S}$ is defined using three parameters $(\lambda, \gamma, S)$, game-theoretic dynamics dictate that a $2 \times 2$ zero-sum game possesses exactly two strategic degrees of freedom. This is due to two structural invariances under positive affine transformations of the payoffs:

1. **Translation Invariance:** Adding a constant to all entries in a column alters the expected payoff of the column player but preserves the relative payoff differences. The Nash Equilibrium is invariant under column translations.
2. **Scaling Invariance:** Multiplying all matrix coefficients by a positive scalar $\alpha > 0$ scales the expected utilities but preserves all relative preference inequalities.

Because of these invariances, one of the three parameters in the family $A_{\lambda, \gamma, S}$ is strategically redundant. The coordinates of the Nash Equilibrium $(p, q)$ occupy a two-dimensional space inside the simplex interior, meaning they can be fully parameterized using only two variables.

---

## Canonical Form ($S = 1$)
To resolve this scaling redundancy, we enforce a scaling normalization constraint by setting:
$$ S = 1 $$

Enforcing $S = 1$ resolves the redundancy and yields the canonical $2 \times 2$ matrix family defined purely by the two strategic parameters $\lambda$ and $\gamma$:
$$ A_{\lambda, \gamma} = \begin{pmatrix} \lambda + \gamma & \lambda \\ \gamma & 1 \end{pmatrix} $$

This canonical matrix provides a linear, decoupled mapping directly to the Nash Equilibrium coordinates:
$$ p = 1 - \gamma \quad \text{and} \quad q = 1 - \lambda $$

---

## Generalization to the $A_\delta$ Baseline
The general 3-parameter derivation is sufficiently robust to recover the baseline game $A_\delta$ studied in the literature. 

Instead of setting $S = 1$, the $A_\delta$ baseline is recovered by freezing the off-diagonal coordinates at $\gamma = 0$ and $\lambda = 1/2$. In this coordinate projection, the scale $S$ is retained as the single free variable. By writing $S = 1 + \delta$, the general matrix $A_{\lambda, \gamma, S}$ simplifies exactly to:
$$ A_{1/2, 0, 1+\delta} = \begin{pmatrix} (1+\delta) - 1 + \frac{1}{2} + 0 & \frac{1}{2} \\ 0 & 1 \end{pmatrix} = \begin{pmatrix} \frac{1}{2} + \delta & \frac{1}{2} \\ 0 & 1 \end{pmatrix} = A_\delta $$

This demonstrates that both $A_\delta$ (a 1-parameter family sweeping one boundary) and our canonical $A_{\lambda, \gamma}$ (a 2-parameter family sweeping both boundaries simultaneously) are mathematically rigorous projections of the same general parameterized game family.
