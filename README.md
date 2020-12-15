# Transformer Implementation :car:
Personal implementation of the Transformer paper in PyTorch.

Original paper: [_Attention is All You Need (Vaswani et al., 2017)_](https://arxiv.org/pdf/1706.03762.pdf)

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
- [ ] Modify evaluation decoding so that it's autoregressive.
- [ ] After checking performance try adding regularization techniques like Dropout and observe performance.
- [ ] Add plots to REARDME file.
