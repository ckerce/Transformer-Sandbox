## Objective
Re-host Lamma2 in a different arrangement of the transformer architecture; investigate signal propagation claims; investigate system identification issues with attention mechanisms. 

There are a number of reasons to think that this should be very challenging:
1. The attention mechanisms are tuned to natural language statistics; the hope is that random excitation can still recover key parameter values.
2. Beyond parallelization, SASP attempts to remove unnecessary weight information.  In the first experiment, where both standard and SASP are trained from scratch, the capacity of SASP seems to have been verified -- but at the expense of very slow convergence.

Relevant files:
- [SASP\_llama2\_conversion.py](../experiments/SASP_llama2_conversion.py)
- [model\_SAS.py](../model_SAS.py)

## Notes

- rand vs. randn probing:  Early layers should respond well to the statistics of the tokenizer (which may already be fairly normal), while latter layers should tend to clean gaussian distributions.  This supposition is driven by the additive mixing and rescaling, which should lead to a central-limit-theorm-driven effect that becomes stonger with deeper layers.
- llama2 has a nonstandard mlp.  That's implemented here for comparison.  Does not seem to matter which version is used, no transfer training effects observed (perhaps weakly in favor of standard ove llama2).
- 
