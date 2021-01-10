# Transformer Implementation :car:

### Personal implementation of the Transformer paper in PyTorch.

**Original paper: [_Attention is All You Need (Vaswani et al., 2017)_](https://arxiv.org/pdf/1706.03762.pdf)**

**Thoughts write up: https://seanyi.info/posts/transformer-implementation/**

Please feel free to contact me regarding any questions or comments. Feedback is always appreciated!

## Downloading the Data

Go to the official website for the [Web Inventory of Transcribed and Translated Talks (WIT^3)](https://wit3.fbk.eu/) and find the appropriate data for IWSLT 2017. I personally used the French-English translation task, but you can choose whichever language you like. There used to be a convenient link to download the data directly, but after contacting the website maintainers the link's not working anymore and you'll have to manually download the data.

After that, go into the `data` directory and run `bash ./preprocess.sh` to preproces the data and receive the resulting files. I've also uploaded the preprocessed data and tokenizer data filesm, but feel free to do it yourself.

## Running the Code

Running the code is fairly simple. Just go into the `src` directory and run `python ./main.py` with the options you'd like. If you're not using [Weights & Biases](https://wandb.ai/) already, I'd strongly recommend using it. It's a great tool to plot metrics and monitor your model's gradients. They have a lot more functionality, but these two are the main ones I use.

You can also use the Makefile recipes that are contained in `src/Makefile`. Running `make debug` will 1) Turn off W&B, and 2) Only use the first 100 samples for training. Additionally, running `make run` and `make run_single_gpu` will run the model with all available GPU's or one GPU, respectively. Note, however, that the previously mentioned two Make recipes don't have a `wandb on` statement.

## Example Run

Running the `make run` command should run the code with the default settings along with the option to use multiple GPU's. My personal run uses two NVIDIA Titan XP's and takes approximately 9 hours to finish the entire process. The result should give you something like the following (from W&B dashboard):

<p align="center">
  <img src="https://github.com/seanswyi/transformer-implementation/blob/main/images/transformer_images.png?raw=true" alt="Run Results"/>
</p>

Training loss continues to drop while evaluation loss starts to go up around the 20K step point. Your learning rate should shoot up and exponentially decay if you implement it according to the schedule specified in the paper (my implementation is in `src/utils.py`). Evaluation BLEU stays somewhat consistent around 25-30. The best BLEU score is 32.785 and the corresponding evaluation loss at that point is 13.817 (a little past 20K steps).

## To-Do

This work is ongoing as there are still things to add for improvement. For example, the decoding strategy I'm using right now is a greedy approach, but I plan to add beam search as well. In addition I'll add an option to load a pre-trained model to directly start translating rather than having to go through the entire training process.

Aside from this, I believe this repository will serve as a good base for future re-implementation projects to come.
