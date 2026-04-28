---
type: Logic
status: Open
related_pillar: "[[Ch5_Optimization_Application]]"
tags: [thesis, chapter_5, math, logistic_regression]
---
# Theory: Logistic Regression as a Min-Max Game

## Conceptual Logic
To test OMWU and OGDA in a practical setting, we transform the standard Regularized Logistic Regression problem into a saddle-point problem.

Given a dataset $\{(a_i, b_i)\}_{i=1}^n$ where $a_i \in \mathbb{R}^d$ and $b_i \in \{-1, 1\}$, the primal problem is:
$$ \min_{w \in \mathbb{R}^d} \frac{1}{n} \sum_{i=1}^n \log(1 + \exp(-b_i w^T a_i)) + \frac{\alpha}{2} \|w\|^2 $$

Using the Fenchel conjugate of the logistic loss, we can rewrite this as a min-max problem:
$$ \min_{w} \max_{\theta \in [0, 1]^n} \dots $$

## Implementation in Optigame
- **Primal Variables ($x$):** The model weights $w$.
- **Dual Variables ($y$):** The sample weights $\theta$.
- **Constraint:** We must project $w$ and $\theta$ onto their respective domains (Simplex or Box constraints).

## Reference
- See [[Lit_Online_Learning_Orabona]] for the duality derivation.

