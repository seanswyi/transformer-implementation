import torch.nn as nn
import torch.nn.functional as F


class FeedForwardNN(nn.Module):
    """
    Simple two-layer feedforward neural network with ReLU activation in between layers.

    Attributes (in alphabetical order)
    ----------------------------------
    d_ff: <int> Hidden dimension size of hidden layer. 2,048 as per the paper.
    d_model: <int> Dimensionality of model. 512 as per the paper.
    dropout: <torch.nn.modules.dropout.Dropout> Dropout from https://jmlr.org/papers/v15/srivastava14a.html.
    linear1: <torch.nn.modules.linear.Linear> First linear layer.
    linear2: <torch.nn.modules.linear.Linear> Second linear layer.
    """
    def __init__(self, args):
        """
        Basic initialization of FeedForwardNN.

        Arguments
        ---------
        args: <argparse.Namespace> Arguments used for overall process.
        """
        super().__init__()

        self.d_model = args.d_model
        self.d_ff = args.d_ff

        self.linear1 = nn.Linear(in_features=self.d_model, out_features=self.d_ff)
        self.linear2 = nn.Linear(in_features=self.d_ff, out_features=self.d_model)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        """
        Forward pass for the model.

        Arguments
        ---------
        x: <torch.Tensor> Input from previous layer.

        Returns
        -------
        <torch.Tensor> Output after linear -> ReLU -> linear.
        """
        output1 = self.dropout(F.relu(self.linear1(x)))
        output2 = self.linear2(output1)

        return output2
