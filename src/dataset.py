import argparse
import logging

import sentencepiece as spm
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


logger = logging.getLogger()


class WMT2014Dataset(Dataset):
    """Object for WMT-2014 dataset."""

    def __init__(
        self,
        args: argparse.Namespace,
        tokenizer: spm.SentencePieceProcessor,
    ):
        """
        Most of the stuff is done inside this method. Every preprocessing step is carried out upon calling a WMT2014Dataset object.

        Arguments
        ---------
        args: <argparse.Namespace> Arguments used for overall process.
        """
        super().__init__()

        self.args = args
        self.debug = args.debug
        self.batch_size = args.batch_size

        self.tokenizer = tokenizer

        self.src_train = self.load(file=self.args.src_train_file)
        self.src_valid = self.load(file=self.args.src_valid_file)
        self.tgt_train = self.load(file=self.args.tgt_train_file)
        self.tgt_valid = self.load(file=self.args.tgt_valid_file)

        assert len(self.src_train) == len(self.tgt_train)
        assert len(self.src_valid) == len(self.tgt_valid)

        self.src_train_input_ids = self.build_input_ids(self.src_train)
        self.src_valid_input_ids = self.build_input_ids(self.src_valid)
        self.tgt_train_input_ids = self.build_input_ids(self.tgt_train)
        self.tgt_valid_input_ids = self.build_input_ids(self.tgt_valid)

        self.train_data = self.create_dataset(
            self.src_train_input_ids, self.tgt_train_input_ids
        )
        self.valid_data = self.create_dataset(
            self.src_valid_input_ids, self.tgt_valid_input_ids
        )

    def load(
        self,
        file: str,
    ) -> list[str]:
        """Reads raw text files."""
        with open(file=file) as f:
            data = [line.lower().strip() for line in f.readlines()]

        if self.debug:
            data = data[:100]

        return data

    def build_input_ids(
        self,
        data: list[str],
        is_src: bool = True,
    ) -> list[torch.Tensor]:
        """Builds inputs from list of strings."""
        processed_data = []
        for sample in tqdm(
            iterable=data,
            desc="Building inputs",
            total=len(data),
        ):
            input_ids = self.tokenizer.build_inputs_with_special_tokens(
                sample, is_src=is_src
            )
            processed_data.append(input_ids)

        return processed_data

    def create_dataset(
        self,
        src: list[torch.Tensor],
        tgt: list[torch.Tensor],
    ) -> list[list[torch.Tensor]]:
        return list(zip(src, tgt))
