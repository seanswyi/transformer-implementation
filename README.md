# Transformer Implementation :car:

### Personal implementation of the Transformer paper in PyTorch.

**Original paper: [_Attention is All You Need (Vaswani et al., 2017)_](https://arxiv.org/pdf/1706.03762.pdf)**

Initially started this reimplementation project back in the summer of 2019 when I started graduate school. My programming/computer science background back then was _extremely_ lacking and therefore I wasn't able to complete this. Came back after a long hiatus and took it up as a side project during the fall semester of 2020. Managed to finally finish the main bulk of it (achieved the initial goal of 30.0+ BLEU) and just going to maintain/improve it now.

---

### To-Do.

- [x] ~~Implement data parallelism to speed up process.~~ (Dec. 1st, 2020)
- [x] ~~Implement BLEU score.~~ (Dec. 1st, 2020)
- [x] ~~Implement evaluation code.~~ (Dec. 1st, 2020)
- [x] ~~Visualize results using [W&B](https://www.wandb.com/) or another tool.~~ (Dec. 1st, 2020)
- [x] ~~Modify code so that warmup steps are properly implemented as "iterations" rather than "epochs."~~ (Dec. 3rd, 2020)
- [x] ~~Modify code so that you're observing the "epoch loss" as well as the "iteration loss."~~ (Dec. 3rd, 2020)
- [x] ~~Fix code so that weights are shared among input, output embeddings and pre-softmax linear layer.~~ (Dec. 3rd, 2020)
- [x] ~~Add evaluation during training per every <a href="https://www.codecogs.com/eqnedit.php?latex=x" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x" title="x" /></a> steps.~~ (Dec. 3rd, 2020)
- [x] ~~Check if model and BLEU scores work properly. :arrow_right: Run code on entire training data instead of `debug` setting.~~ (Dec. 3rd, 2020)
- [x] ~~Additionally check if learning rates are properly adjusting.~~ (Dec. 3rd, 2020)
- [x] ~~Debug BLEU score. That is, make sure that you're actually calculating correctly.~~ (Dec. 4th, 2020)
- [x] ~~Debug learning rate scheduling. Refer back to the original paper and check.~~ (Dec. 4th, 2020)
- [x] ~~Debug evaluation code. Make it so that you're getting predictions properly.~~ (Dec. 7th, 2020)
- [x] ~~Modify code so that you're plotting evaluation loss as well.~~ (Dec. 7th, 2020)
- [x] ~~Check performance of model to make sure everything works properly.~~ (Dec. 9th, 2020)
- [x] ~~Configure W&B so that you're also getting gradient information and inspect that your network is training properly.~~ (Dec. 10th, 2020)
- [x] ~~Modify evaluation decoding so that it's autoregressive.~~ (Dec. 15th, 2020)
- [x] ~~Fix masked attention to use `torch.triu` and add instead of `torch.tril` and multiply. The difference is in the softmax operation later.~~ (Dec. 17th, 2020)
- [x] ~~Find problem in code regarding BLEU evaluation.~~ (Dec. 24th, 2020)
  - Problem was that BLEU should be evaluated at the corpus level, not the sentence level. The reason is because calculating BLEU on a sentence level is too noisy and inaccurate.
- [x] ~~Modify code so that the output linear layer shares the weights of the embedding matrix.~~ (Dec. 30th, 2020)
  - I was previously treating the "pre-softmax linear transformation" as a simple matrix multiplication between the embedding layer's weights and the previous layer's output (i.e., input). What I needed to do is to treat the output linear layer as a PyTorch Module (i.e., `torch.nn.Linear`) and to do that I needed to do `output_linear.weight = embedding_layer.weight`).
- [ ] ~~Add docstrings to code.~~ (Jan. 2nd, 2021)
- [ ] Modify decoding so that it stops when the EOS token is output.
- [ ] Add beam search.
- [ ] Add plots to REARDME file.
