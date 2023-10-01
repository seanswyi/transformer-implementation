import argparse

import torch
from torch import nn
from torch.nn import functional as F

from models.decoder import Decoder
from models.embedding_layer import EmbeddingLayer
from models.encoder import Encoder
from models.tokenizer import Tokenizer


class Transformer(nn.Module):
    """
    Object to hold full Transformer model.

    Attributes (in alphabetical order)
    ----------------------------------
    d_model: <int> Dimensionality of model. 512 as per the paper.
    decode: <method> Performs neural decoding.
    decoder_stack: <torch.nn.modules.container.ModuleList> Object to hold each Decoder sublayer. 6 here as per the paper.
    dropout: <torch.nn.modules.dropout.Dropout> Dropout from https://jmlr.org/papers/v15/srivastava14a.html.
    emb: <models.embedding_layer.EmbeddingLayer> Embedding layer that combines embedding and positional embedding.
    encode: <method> Performs neural encoding from the given input.
    encoder_stack: <torch.nn.modules.container.Sequential> Performs sequential forward passes for the number of encoder sublayers.
    output_linear: <torch.nn.modules.linear.Linear> Pre-softmax linear transformation. Shares weights with embedding.
    softmax: <torch.nn.modules.activation.LogSoftmax> Log softmax to convert output to probabilities.
    vocab_size: <int> Size of vocabulary. Default is 16,000 in this implementation.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        tokenizer: Tokenizer,
    ):
        """
        Basic initialization of Transformer.

        Arguments
        ---------
        args: <argparse.Namespace> Arguments used for overall process.
        """
        super().__init__()

        self.args = args
        self.num_stacks = self.args.num_stacks
        self.d_model = self.args.d_model
        self.vocab_size = self.args.vocab_size

        self.tokenizer = tokenizer
        self.bos_token_id = tokenizer.bos_id()

        self.emb = EmbeddingLayer(self.args)

        encoders = [Encoder(self.args) for _ in range(self.num_stacks)]
        self.encoder_stack = nn.Sequential(*encoders)

        decoders = [Decoder(self.args) for _ in range(self.num_stacks)]
        self.decoder_stack = nn.ModuleList(decoders)

        self.output_linear = nn.Linear(
            in_features=self.d_model, out_features=self.vocab_size, bias=False
        )
        self.output_linear.weight = self.emb.embedding_layer.weight

        self.softmax = nn.LogSoftmax(dim=-1)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, src, tgt):
        """
        Forward pass for the model. emb -> encode -> decode -> softmax.

        Arguments
        ---------
        src: <torch.Tensor> Source data.
        tgt: <torch.Tensor> Target data.

        Returns
        -------
        <torch.Tensor> Output passed through LogSoftmax layer. Note that if we're using cross entropy loss \
            that means we'll have to set our loss function to negative log likelihood due to this.
        """
        src_emb = self.emb(src.long())
        tgt_emb = self.emb(tgt.long())

        enc_output = self.encode(src_emb)
        dec_output = self.decode(tgt_emb, enc_output)

        logits = self.dropout(self.output_linear(dec_output))
        return logits

    def encode(self, src):
        """
        Performs neural encoding of the source sequence. Note we can use the Sequential module \
            because there's only one chosen input.

        Arguments
        ---------
        src: <torch.Tensor> Source data.
        """
        output = self.encoder_stack(src)
        return output

    def decode(self, tgt, enc_output):
        """
        Performs neural decoding of the target sequence. Note we have to use ModuleList and loop \
            through because we have two inputs.

        Arguments
        ---------
        tgt: <torch.Tensor> Target data.
        enc_output: <torch.Tensor> Output from the Encoder stack.
        """
        output = tgt
        for i in range(self.num_stacks):
            output = self.decoder_stack[i](output, enc_output)

        return output

    def translate(
        self,
        input_text: str | torch.tensor,
        device: str = "cpu",
    ) -> str:
        """Receives raw text as input and translates it.

        Arguments
        ---------
        input_text: str | torch.tensor
            Text to translate. It may be a raw text or a pre-processed tensor.

        Returns
        -------
        translated_text: str
            Translated input text.
        """
        if isinstance(input_text, str):
            src = self.tokenizer.tokenize(input_text)
            src = torch.tensor(src)
        else:
            src = input_text

        tgt = torch.ones(size=(input_text.shape[0],)).reshape(-1, 1) * self.bos_token_id

        src = src.to(device)
        tgt = tgt.to(device)

        for _ in range(src.shape[0]):
            pred = F.softmax(self(src, tgt), dim=2)
            pred = torch.argmax(pred, dim=2)[:, -1]
            tgt = torch.cat((tgt, pred.view(-1, 1)), dim=-1)

        translated_text = self.tokenizer.decode(tgt[:, 1:])
        return translated_text
