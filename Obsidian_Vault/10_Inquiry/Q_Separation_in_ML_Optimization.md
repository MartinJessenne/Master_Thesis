---
type: Inquiry
status: Open
related_pillar: "[[Ch5_Optimization_Application]]"
tags: [thesis, chapter_5, optimization, separation]
---
# Inquiry: Does the Separation Persist in ML?

## The Question
Does the separation between Best-Iterate and Last-Iterate convergence observed in $2 \times 2$ matrix games persist when OMWU is used to train a Logistic Regression model?

## Hypothesis
I expect OMWU to show significant oscillations in the "Last-Iterate" weights (validation accuracy will fluctuate), while the "Best-Iterate" weights will achieve a stable, high accuracy comparable to OGDA.

## Proposed Methodology
1. Implement the Saddle-Point formulation in `implementation.py`.
2. Run `optigame` on a standard dataset (e.g., MNIST or a synthetic ill-conditioned dataset).
3. Compare the Duality Gap trajectories for OMWU and OGDA.
4. Link results to [[R_Optimization_Separation]].

