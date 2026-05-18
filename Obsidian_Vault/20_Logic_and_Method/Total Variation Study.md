---
type:
status: Open
related_pillar: "[[Thesis_Master_Plan]]"
tags:
  - thesis
created: 2026-05-18 10:16
---
# Total Variation Deeper Study

> [!info] Quick Summary
> What is this note about?
> 
> Currently the total variation is only a scalar, that sums the consecutive distances between all duality gap iterations. While this metric does a good job of explaining how chaotic a run has been, it doesn't answer, when or more precisely from where did it start going chaotic. 

# Description 

Try to develop the computation of the total variation as a cumulative sum, the current total variation would just be the final value of this cumsum. 

# Goal
This should allow us to determine when does the algorithm starts showing chaotic behavior in its convergence, as well as to have a graphic visualization of the magnitude of the chaotic behavior throughout the iterations. 

# Method

Everything is currently happening in optigame/experiments/metrics.rs

If we take the current function : 

`pub(crate) fn compute_single_metric(game_result: &GameResult, method: MetricMethodType) -> f64`

This function return a `f64`. 
While we need to return a `Vec<f64>` for our purpose. 

Here are several design options : 
1. Make `compute_single_metric` return a `Vec<f64>` for every single metric. 
2. Introduce an enum like that : 
```rust
enum MetricOutput:
	Reduced(f64),
	Developed(Vec<f64>),
```
So that 
`pub(crate) fn compute_single_metric(game_result: &GameResult, method: MetricMethodType) -> MetricOutput`

And we add a new variant to MetricMethodType : 
```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MetricMethodType {
    MaxLast(f64),
    VarLast(f64),
    TotalVar,
    DevelopedTotalVar,
}
```
The naming conventions are still to be refined, but the global idea is that all metric method will lead to a Reduced output except for DevelopedTotalVar that will lead to a Developed output. 

Now the question is : How will that break existing workflows? 

#Ask: In a coding project like the current one, how do you methodically lead those types of Audit? Like what's the way to have this holistic view of the software, to see how each function gets called by other, what failures would a change imply? Is there a method else than just fucking around and try to look for where is the function call and manually go back up the call stack? Or are there any methods, documentations we can introduce to make that clearer and easier and have a better map of the codebase? 
#Answer:

3. Create a standalone function `compute total var history`
We can indeed consider that this is an experiment in itself and that it requires its own set of function. The deal would be to just run a single experiment, (that's something I need to implement, because right now it's fully located in the ffi.rs file which is not how it should be), get the GameResult out of it, and return both the GameResult and the TotalVarDeveloped. 


Now the question is, from a standpoint of the best practices in scientific coding, what is the best architecture choice in this context? The one that allows for maximum flexibility, and to experiment as easily and freely as possible across all different set ups? 
