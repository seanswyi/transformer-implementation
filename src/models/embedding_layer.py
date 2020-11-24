import numpy as np
import torch
import torch.nn as nn


class EmbeddingLayer(nn.Module):
    def __init__(self, args):
        super(EmbeddingLayer, self).__init__()

        self.args = args
        self.embedding_layer = nn.Embedding(num_embeddings=args.vocab_size, embedding_dim=args.d_model, padding_idx=0)

    def forward(self, x):
        x_embedding = self.embedding_layer(x)
        x_embedding += self.positional_encoding_layer(x_embedding)

        return x_embedding

    def positional_encoding_layer(self, x):
        positions = np.arange(x.shape[1])
        dimensions = np.arange(x.shape[2])

        denominator = np.power(10000, (2 * dimensions) / self.args.d_model)
        input_angles = positions.reshape(-1, 1) / denominator.reshape(1, -1) # We want shape (sequence_length, d_model)

        positional_encoding_values = np.zeros(shape=input_angles.shape)
        positional_encoding_values[:, 0::2] = np.sin(input_angles[:, 0::2])
        positional_encoding_values[:, 1::2] = np.cos(input_angles[:, 1::2])

        return torch.tensor(positional_encoding_values)
