# Transformer Implementation
Personal implementation of the Transformer paper in PyTorch.

Original paper: [_Attention is All You Need (Vaswani et al., 2017)_](https://arxiv.org/pdf/1706.03762.pdf)

---

### To-Do.

- [ ] Fix code so that weights are shared among input, output embeddings and pre-softmax linear layer.
- [x] ~~Implement data parallelism to speed up process.~~ (Dec. 1st, 2020)
- [x] ~~Implement BLEU score.~~ (Dec. 1st, 2020)
- [x] ~~Implement evaluation code.~~ (Dec. 1st, 2020)
- [ ] Check if model and BLEU scores work properly. :arrow_right: Run code on entire training data instead of `debug` setting.
- [ ] Visualize results using [W&B](https://www.wandb.com/) or another tool.
