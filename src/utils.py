import numpy as np
import torch
import torch.nn.functional as F


def adjust_learning_rate(step_num, d_model, warmup_steps):
    """
    Adjusts the learning rate according to the schedule in the paper.

    Arguments
    ---------
    step_num: <int> The current step in training.
    d_model: <int> Dimensionality of model. 512 as per the paper.
    warmup_steps: <int> The number of "warming up" steps that we take before gradually \
        decreasing the learning rate. 4,000 as per the paper.
    """
    step_num += 1e-20
    term1 = np.power(d_model, -0.5)
    term2 = min(np.power(step_num, -0.5), step_num * np.power(warmup_steps, -1.5))

    return term1 * term2


def decode_autoregressive(model, src):
    """
    Performs decoding in an autoregressive fashion.

    Arguments
    ---------
    model: <models.transformer.Transformer> Transformer model to performing decoding.
    src: <torch.Tensor> Input source sequence to be decoded (i.e., translated).
    """
    outputs = torch.ones(size=(src.shape[0],)).reshape(-1, 1) * 2

    if torch.cuda.is_available():
        src = src.to("cuda")
        outputs = outputs.to("cuda")

    for _ in range(src.shape[1]):
        prediction = F.softmax(model(src, outputs), dim=2)
        prediction = torch.argmax(prediction, dim=2)[:, -1]
        outputs = torch.cat((outputs, prediction.view(-1, 1)), dim=-1)

    return outputs[:, 1:]
