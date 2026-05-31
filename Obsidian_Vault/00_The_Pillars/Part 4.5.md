---
type:
status: Open
related_pillar: "[[Thesis_Master_Plan]]"
tags:
  - thesis
created: 2026-05-31 17:19
---
# Narration reflection 
- This part should be short. 
- The main question we're asking here is that the previous subparts have exhibited an intrinsic difficulty of OMWU to converge in a last iterate sense toward the Nash Equilibrium whenever it's too close from the simplex boundaries. 
- Now we ask ourselves the question, do the mathematical results presented in Cai et al. On separation between convergence mode still hold? 
- to verify that, we just show a similar version of this plot : ![[Part 4.5.png|613]]
- But I don't know how we're supposed to validate or not the convergence bound? 
- Do we just say that the theory says that is it supposed to converge in $O(T^{-\frac{1}{6}})$ so we compute, for T = 10 000, we're suppposed to have a best iterate that is below $10000^{-\frac{1}{6}} \approx 0.21$ which is what where noticing more or less here?
- Or am I missing something important. 
- Because if that's the case this whole part can just go, I mean it's been mathematically proven in the paper, and they've thoroughly studied $A_\delta$ in the paper, so we know it works, there or maybe just a simple line where we display this graph and explain the previous computation I detailed if that makes sense. 
- But I realize this doesn't even make sense, since big $O$ notation are up to a multiplying constant, so even if it the best iterate of the duality gap was 10000 thousands, it's the value of the ratio between the best iterate and $T^{-\frac{1}{6}}$ that matters. 
- I need you to help me clarify that though. 