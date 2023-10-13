import logging

import numpy as np
import torch
import torch.nn as nn


logger = logging.getLogger()


class EmbeddingLayer(nn.Module):
    """
    Object to hold entire embedding layer (including positional encoding).

    Attributes (in alphabetical order)
    ----------------------------------
    embedding_layer: <torch.nn.modules.sparse.Embedding> Embedding layer with 16,000 embeddings, with each embedding being dimension 512. \
        We set `padding_idx` to be 0 because we want to tell our embedding layer which index is reserved for padding sequences.
    positional_encoding_layer: <method> Adds the sinusoidal positional values to the embeddings.
    """

    def __init__(self, args):
        """
        Basic initialization of what we need.

        Arguments
        ---------
        args: <argparse.Namespace> Arguments used for overall process.
        """
        super().__init__()

        self.args = args
        self.embedding_layer = nn.Embedding(
            num_embeddings=args.vocab_size, embedding_dim=args.d_model, padding_idx=0
        )

    def forward(self, x):
        """
        Performs embedding. Passes through embedding layer first, then adds sinusoidal positional values.

        Arguments
        ---------
        x: <torch.Tensor> Data to retrieve embedding values of.

        Returns
        -------
        <torch.Tensor> Embedding values from trained embedding and added sinusoidal positional values.
        """
        x_embedding = self.embedding_layer(x)
        x_embedding += self.positional_encoding_layer(x_embedding)

        return x_embedding

    def positional_encoding_layer(self, x):
        """
        Retrieves the positional values according to the sine and cosine functions.

        Arguments
        ---------
        x: <torch.Tensor> Embedding values.

        Returns
        -------
        Positional encoding values according to index (i.e., size of x).
        """
        positions = np.arange(x.shape[1])
        dimensions = np.arange(x.shape[2])

        denominator = np.power(10000, (2 * dimensions) / self.args.d_model)
        input_angles = positions.reshape(-1, 1) / denominator.reshape(
            1, -1
        )  # We want shape (sequence_length, d_model)

        positional_encoding_values = np.zeros(shape=input_angles.shape)
        positional_encoding_values[:, 0::2] = np.sin(input_angles[:, 0::2])
        positional_encoding_values[:, 1::2] = np.cos(input_angles[:, 1::2])
        positional_encoding_values = torch.tensor(positional_encoding_values)

        if torch.cuda.is_available():
            positional_encoding_values = positional_encoding_values.to("cuda")
        else:
            logger.warning("Not using GPU!")

        return positional_encoding_values
