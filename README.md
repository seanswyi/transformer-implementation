# Transformer Implementation :car:

### Personal implementation of the Transformer paper in PyTorch.

**Original paper: [_Attention is All You Need (Vaswani et al., 2017)_](https://arxiv.org/pdf/1706.03762.pdf)**

Initially started this reimplementation project back in the summer of 2019 when I started graduate school. My programming/computer science background back then was _extremely_ lacking and therefore I wasn't able to complete this. Came back after a long hiatus and took it up as a side project during the fall semester of 2020. Managed to finally finish the main bulk of it (achieved the initial goal of 30.0+ BLEU) and just going to maintain/improve it now.

Please feel free to contact me regarding any questions or comments. Feedback is always appreciated!

## Running the Code

Running the code is fairly simple. Just go into the `src` directory and run `python ./main.py` with the options you'd like. If you're not using [Weights & Biases](https://wandb.ai/) already, I'd strongly recommend using it. It's a great tool to plot metrics and monitor your model's gradients. They have a lot more functionality, but these two are the main ones I use.
