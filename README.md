# Transformer Implementation :car:

### Personal implementation of the Transformer paper in PyTorch.

**Original paper: [_Attention is All You Need (Vaswani et al., 2017)_](https://arxiv.org/pdf/1706.03762.pdf)**

Initially started this reimplementation project back in the summer of 2019 when I started graduate school. My programming/computer science background back then was _extremely_ lacking and therefore I wasn't able to complete this. Came back after a long hiatus and took it up as a side project during the fall semester of 2020. Managed to finally finish the main bulk of it (achieved the initial goal of 30.0+ BLEU) and just going to maintain/improve it now.

Please feel free to contact me regarding any questions or comments. Feedback is always appreciated!

## Downloading the Data

Go to the official website for the [Web Inventory of Transcribed and Translated Talks (WIT^3)](https://wit3.fbk.eu/) and find the appropriate data for IWSLT 2017. I personally used the French-English translation task, but you can choose whichever language you like.

After that, go into the `data` directory and run `bash ./download.sh` to preproces the data and receive the resulting files. I've also uploaded the preprocessed data and tokenizer data filesm, but feel free to do it yourself.

## Running the Code

Running the code is fairly simple. Just go into the `src` directory and run `python ./main.py` with the options you'd like. If you're not using [Weights & Biases](https://wandb.ai/) already, I'd strongly recommend using it. It's a great tool to plot metrics and monitor your model's gradients. They have a lot more functionality, but these two are the main ones I use.

## To-Do

This work is ongoing as there are still things to add for improvement. For example, the decoding strategy I'm using right now is a greedy approach, but I plan to add beam search as well. In addition I'll add an option to load a pre-trained model to directly start translating rather than having to go through the entire training process.

Aside from this, I believe this repository will serve as a good base for future re-implementation projects to come.
