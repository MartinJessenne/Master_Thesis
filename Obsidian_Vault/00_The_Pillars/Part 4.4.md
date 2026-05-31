---
type:
status: Open
related_pillar: "[[Thesis_Master_Plan]]"
tags:
  - thesis
created: 2026-05-31 14:55
---
# Part 4.4 organization

> [!info] Quick Summary
> What is this note about?

Drawing on what's been previously said on the previous subparts, the aim is here to draw a more precise exploration / mapping of the pathological behavior of OMWU. 

To do that we have to thoroughly specify the set-up : 
- Which parametric curves we used. 
- What exactly are the graphs displaying


The first ideas I have regarding the structure would be the following : 
- Leveraging the previous results, we see that random concentric shell explorations around A_delta have yielded informative results about the persistence of the pathological behavior in OMWU compared to OGDA see <concentric_exploration_OMWU> and <concentric_exploration_OGDA>. 
- Those results naturally motivate us to investigate further this behavior by explorating along a deterministic parametric curve the neighborhood of A_delta to map more clearly this behavior. 
- The parametric curves we're going to use here in these experiments are circles centered around $A_\delta$ Nash equilibrium. 

Question :
- Is the current experimental setting in analysis.py correct :
	- Chose delta = 0.01
	- create those circle parametric curves circling around $x^* = [\frac{1}{1+δ}, \frac{δ}{1+δ}]$ and $y^* = [\frac{1}{2(1+δ)}, \frac{1+2δ}{2(1+δ)}]$ with radius $0.5 * \frac{\delta}{1+\delta}$
	- plot the results of the runs and their associated extracted metrics along those parametric curves. 
	- Comment and conclude

	Because for each point $(x,y)$ along this circle, we're running the algorithm for the $A_{\lambda = x,\gamma = y}$ matrix whose Nash Equilibrium is precisely on this $(x,y)$ point, but this is our $A_{\lambda, \gamma}$ matrix that we're using here, not the lemma 5 matrix, which might be a more adequate generalization of $A_\delta$ matrix with some properties I'm not aware of that would be conserved using another matrix family instead of my $A_{\lambda, \gamma}$ family of matrices. 

Once this question has been answered we could finish the part like this : 
- Maybe compare the metrics / the duality gap values for both OGDA and OMWU in this setting and tell that while this confirm that there are convergence issues on the OMWU side, we're interested on how the dynamics maps clearly in the "strategies space" that why we're switching from the "duality gap" space to the "strategies space". 
- Display the side by side comparison of theoretical vs computed values for the nash equilibrium in OGDA vs OMWU. 
- Tell that we show that OMWU indeed struggle in this setting to converge. That motivates us to switch our observation from the "duality gap space"
- Display one figure which would be a series of plot_2d_profiles for several positions along the circle to see those orbital issues. 
- This allows us to see clearly and "in real-time" how the algorithm is behaving depending on where it is on the simplex. 
- #improvement Maybe it would be interesting in those plot_2d_graphs to display the shape of the simplex, as well as the shapes of level sets, so that we can have more context on where we are. Let's evaluate how easy or not that would be to implement. And if that's too hard, we can keep that for the thesis defense. 
- Along that we can add some reflection about the relevance of the cumulated total variation to display the chaotic behavior. But we'll have to make a choice here between the total variation set-up and the Distance to L2 setup. But I personally think that since we've "switched" to the "strategies space" we'd better stick to it and keep the L2 distance as a tool to showcase the chaotic / cyclic behavior of OMWU. 
- Once all of that is done we can conclude that we've shown that this pathological behavior of OMWU around A_\delta is a robust convergence pathology linked with the intrinsic structure of OMWU and not just and isolated point that exhibit strange convergence behavior.


# Random thoughts
I think that the main contribution of the overall thesis that we must highlight in the conclusion is clearly that the efforts we've put into design a high-performance optimization framework in rust allowed us to push further the boundaries of what's possible to explore, pushing order of magnitudes above the data that could have been produced with a python only set-up, thus allowing us to confirm this instrinsic behavior of OMWU. 

# Opening questions (which might be approached during the defense)
- Why $A_\delta$, we've partly answered this but what happens for any other payoff matrix whose Nash Equilibrium is situated at the same distance from the Nash Equilibrium boundary. 
- To that two experimental set-ups : 
	- 1. Create one parametric line that is a contour of the simplex triangle, like an inner triangle but that is at a $\frac{\delta}{1+\delta}$ distance from the simplex triangle. 
	- 2. Create one parametric line that is on the same level set of the OMWU regularized that the one $A_\delta$ belongs to. 
	

