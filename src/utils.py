import numpy as np
import torch
from torch import nn


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


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def xavier_init_model(model):
    for param in model.parameters():
        try:
            nn.init.xavier_uniform_(param)
        except ValueError:
            pass
