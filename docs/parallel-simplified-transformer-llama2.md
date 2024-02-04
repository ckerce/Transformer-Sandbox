## Objective
Re-host Lamma2 in a different arrangement of the transformer architecture; investigate signal propagation claims; investigate system identification issues with attention mechanisms. 

## Notes

- rand vs. randn probing:  Early layers should respond well to the statistics of the tokenizer (which may already be fairly normal), while latter layers should tend to clean gaussian distributions.  This supposition is driven by the additive mixing and rescaling, which should lead to a central-limit-theorm-driven effect that becomes stonger with deeper layers.
- llama2 has a nonstandard mlp.  That's implemented here for comparison.  Does not seem to matter which version is used, no transfer training effects observed (perhaps weakly in favor of standard ove llama2).
- 
