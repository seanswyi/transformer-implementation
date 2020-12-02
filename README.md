# Transformer Implementation :car:
Personal implementation of the Transformer paper in PyTorch.

Original paper: [_Attention is All You Need (Vaswani et al., 2017)_](https://arxiv.org/pdf/1706.03762.pdf)

---

### To-Do.

- [x] ~~Implement data parallelism to speed up process.~~ (Dec. 1st, 2020)
- [x] ~~Implement BLEU score.~~ (Dec. 1st, 2020)
- [x] ~~Implement evaluation code.~~ (Dec. 1st, 2020)
- [x] Visualize results using [W&B](https://www.wandb.com/) or another tool. (Dec. 1st, 2020)
- [ ] Modify code so that warmup steps are properly implemented as "iterations" rather than "epochs."
- [ ] Modify code so that you're observing the "epoch loss" as well as the "iteration loss."
- [ ] Fix code so that weights are shared among input, output embeddings and pre-softmax linear layer.
- [ ] Add evaluation during training per every <a href="https://www.codecogs.com/eqnedit.php?latex=x" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x" title="x" /></a> steps.
- [ ] Check if model and BLEU scores work properly. :arrow_right: Run code on entire training data instead of `debug` setting.
