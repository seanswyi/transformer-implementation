import argparse
import logging

import sentencepiece as spm
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


logger = logging.getLogger()


class TextPairDataset:
    def __init__(
        self,
        args: argparse.Namespace,
        tokenizer: spm.SentencePieceProcessor,
    ):
        self.args = args
        self.debug = args.debug
        self.batch_size = args.batch_size

        self.src_train_text = self.load(file=args.src_train_file)
        self.src_valid_text = self.load(file=args.src_valid_file)
        self.tgt_train_text = self.load(file=args.tgt_train_file)
        self.tgt_valid_text = self.load(file=args.tgt_valid_file)

        self.train_data = self.create_pair_data(
            src=self.src_train_text, tgt=self.tgt_train_text
        )
        self.valid_data = self.create_pair_data(
            src=self.src_valid_text, tgt=self.tgt_valid_text
        )

        self.tokenizer = tokenizer

        self.train_dataloader = DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )
        self.valid_dataloader = DataLoader(
            dataset=self.valid_data,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
        )

    def collate_fn(
        self,
        batch: list[list[str, str]],
    ) -> torch.Tensor:
        srcs = [x[0] for x in batch]
        tgts = [x[1] for x in batch]

        src_input_ids = [self.tokenizer(src) for src in srcs]
        max_src_len = max(len(src) for src in src_input_ids)
        src_input_ids_padded = [
            F.pad(
                input=src,
                pad=(0, max_src_len - len(src)),
                value=self.tokenizer.pad_id(),
            )
            for src in src_input_ids
        ]
        src_inputs = torch.vstack(src_input_ids_padded)

        tgt_input_ids = [
            self.tokenizer.build_inputs_with_special_tokens(tgt) for tgt in tgts
        ]
        max_tgt_len = max(len(tgt) for tgt in tgt_input_ids)
        tgt_input_ids_padded = [
            F.pad(
                input=tgt,
                pad=(0, max_tgt_len - len(tgt)),
                value=self.tokenizer.pad_id(),
            )
            for tgt in tgt_input_ids
        ]
        tgt_inputs = torch.vstack(tgt_input_ids_padded)

        return {"src": src_inputs, "tgt": tgt_inputs}

    def load(
        self,
        file: str,
    ) -> list[str]:
        with open(file=file) as f:
            data = [line.lower().strip() for line in f.readlines()]

        if self.debug:
            data = data[:100]

        return data

    def create_pair_data(
        self,
        src: list[str],
        tgt: list[str],
    ) -> list[list[str, str]]:
        assert len(src) == len(tgt)

        empty_src_count = 0
        empty_tgt_count = 0
        both_empty_count = 0

        pairs = []
        pbar = tqdm(
            iterable=zip(src, tgt),
            desc=f"Creating pair data from {len(src)} pairs",
            total=len(src),
        )
        for s, t in pbar:
            if not src and tgt:
                empty_src_count += 1
                continue

            if src and not tgt:
                empty_tgt_count += 1
                continue

            if not src and not tgt:
                both_empty_count += 1
                continue

            pairs.append((s, t))

        logger.info(
            "Created pair data of %d pairs. %d total were dropped.",
            len(pairs),
            len(src) - len(pairs),
        )
        logger.info(
            "%d empty source samples, %d empty target samples, %d both empty samples.",
            empty_src_count,
            empty_tgt_count,
            both_empty_count,
        )

        return pairs
