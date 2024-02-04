## Transformer Sandbox

I forked Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) to create an easy-to-use analysis environment for experimenting with transformer architectures.  

[In contrast to the goals of nanoGPT](https://github.com/karpathy/nanoGPT), where the focus is on effiecient management of training a GPT-2 sized model, this sandbox is focused on the investigation of alternative sequence-to-sequence architectures (using the GPT-2 scale training environment made available by nanoGPT).

The local setup instructions for this repo are found [here](docs/nanoGPT-README.md). 

I'm preparing baseline material to serve as jumping off points for LLM and Generative AI investigation.  Current topics of interest are the following:
* [Flexible Transfomer Architecture](docs/simplified-transformers_README.md):  Investigation of observations and assertions from the paper Simplified Transformer Blocks ([on arxiv](https://arxiv.org/abs/2311.01906)).
* Signal Propgation Analysis: `IN PROGRESS` using Llama2 
* [Parallel Simplified Transformer Implementation of Llama2](docs/parallel-simplified-transformer-llama2.md) : `IN PROGRESS` 
* Simplified Transformer w/ ALiBi: `TODO` 
* Mixture of Experts - Simplified Transformer: `TODO`
* Diffusion Language Models: `TODO`
* HiPPO and State Space Models: `TODO`

