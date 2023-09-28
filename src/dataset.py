import argparse
import logging
import multiprocessing as mp

import sentencepiece as spm
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
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
        self.tgt_train_input_ids = self.build_input_ids(self.tgt_train, is_src=False)
        self.tgt_valid_input_ids = self.build_input_ids(self.tgt_valid, is_src=False)

        self.train_data = self.create_dataset(
            self.src_train_input_ids, self.tgt_train_input_ids
        )
        self.valid_data = self.create_dataset(
            self.src_valid_input_ids, self.tgt_valid_input_ids
        )

        num_workers = mp.cpu_count() - 1
        logger.info("Creating DataLoaders with %d workers.", num_workers)

        self.train_dataloader = self.get_dataloader(
            data=self.train_data,
            num_workers=num_workers,
            shuffle=True,
        )
        self.valid_dataloader = self.get_dataloader(
            data=self.valid_data,
            num_workers=num_workers,
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
            if is_src:
                input_ids = self.tokenizer(sample)
            else:
                input_ids = self.tokenizer.build_inputs_with_special_tokens(sample)

            processed_data.append(input_ids)

        return processed_data

    def create_dataset(
        self,
        src: list[torch.Tensor],
        tgt: list[torch.Tensor],
    ) -> list[list[torch.Tensor]]:
        return list(zip(src, tgt))

    def collate_fn(
        self,
        batch: list[tuple[torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        srcs = [b[0] for b in batch]
        tgts = [b[1] for b in batch]

        max_src_len = max(len(src) for src in srcs)
        max_tgt_len = max(len(tgt) for tgt in tgts)
        max_batch_seq_len = max(max_src_len, max_tgt_len)

        padded_srcs = [
            F.pad(
                input=src,
                pad=(0, max_batch_seq_len - len(src)),
                value=self.tokenizer.pad_id(),
            )
            for src in srcs
        ]
        padded_tgts = [
            F.pad(
                input=tgt,
                pad=(0, max_batch_seq_len - len(tgt)),
                value=self.tokenizer.pad_id(),
            )
            for tgt in tgts
        ]

        input_srcs = torch.stack(padded_srcs)
        input_tgts = torch.stack(padded_tgts)

        return {"src": input_srcs, "tgt": input_tgts}

    def get_dataloader(
        self,
        data: list[tuple[torch.Tensor]],
        num_workers: int = 0,
        shuffle: bool = False,
    ) -> DataLoader:
        dataloader = DataLoader(
            dataset=data,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
        )

        return dataloader
