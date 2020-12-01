# Transformer Implementation
Personal implementation of the Transformer paper in PyTorch.

Original paper: [_Attention is All You Need (Vaswani et al., 2017)_](https://arxiv.org/pdf/1706.03762.pdf)

---

### To-Do.

- [ ] Fix code so that weights are shared among input, output embeddings and pre-softmax linear layer.
- [x] Implement data parallelism to speed up process.
- [x] Implement BLEU score.
- [x] Implement evaluation code.
- [ ] Visualize results using [W&B](https://www.wandb.com/) or another tool.
