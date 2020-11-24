import argparse
import logging
import os

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data import WMT2014Dataset
from models.transformer import Transformer
from utils import adjust_learning_rate

logger = logging.getLogger()


def train(args, model, data):
    if torch.cuda.is_available():
        model = model.to('cuda')
    else:
        logger.warning("Not using GPU!")

    for step in range(args.warmup_steps):
        learning_rate = adjust_learning_rate(step, args)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

        step_loss = 0.0

        progress_bar = tqdm(iterable=data.train_data, desc="Training", total=len(data.train_data))
        for batch in progress_bar:
            optimizer.zero_grad()

            src, tgt = batch[:, 0], batch[:, 1]
            bos_tokens = torch.ones(size=(tgt.shape[0],)).reshape(-1, 1) * 2
            tgt_shifted_right = torch.cat((bos_tokens, tgt), dim=1)[:, :-1] # Truncate last token to match size.

            if torch.cuda.is_available():
                src = src.to('cuda')
                tgt = tgt.to('cuda')
                tgt_shifted_right = tgt_shifted_right.to('cuda')
            else:
                logger.warning("Not using GPU!")

            output = model(src, tgt_shifted_right)

            loss = criterion(output.view(-1, args.vocab_size), tgt.view(-1).long())
            step_loss += loss.item()

            loss.backward()
            optimizer.step()

        if step % args.log_step == 0:
            logger.info(f"Step: {step} | Loss: {step_loss} | LR: {learning_rate}")

    return 0


def main(args):
    msg_format = '[%(asctime)s - %(levelname)s - %(filename)s: %(lineno)d (%(funcName)s)] %(message)s'
    logging.basicConfig(format=msg_format, level=logging.INFO, handlers=[logging.StreamHandler()])

    data = WMT2014Dataset(args)
    model = Transformer(args)
    results = train(args, model, data)
    # Run the embedded version into single head attention.
    import pdb; pdb.set_trace()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--beta1', default=0.9, type=float)
    parser.add_argument('--beta2', default=0.98, type=float)
    parser.add_argument('--epsilon', default=10e-9, type=float)
    parser.add_argument('--data_root', default='../data/', type=str)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--d_ff', default=2048, type=int)
    parser.add_argument('--d_model', default=512, type=int)
    parser.add_argument('--d_k', default=512, type=int)
    parser.add_argument('--d_v', default=512, type=int)
    parser.add_argument('--log_step', default=50, type=int)
    parser.add_argument('--num_epochs', default=3, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--num_stacks', default=6, type=int)
    parser.add_argument('--src_train_file', default='../data/train.fr-en_preprocessed.fr', type=str)
    parser.add_argument('--tgt_train_file', default='../data/train.fr-en_preprocessed.en', type=str)
    parser.add_argument('--src_valid_file', default='../data/valid.fr-en_preprocessed.fr', type=str)
    parser.add_argument('--tgt_valid_file', default='../data/valid.fr-en_preprocessed.en', type=str)
    parser.add_argument('--vocab_size', default=20000, type=int)
    parser.add_argument('--tokenizer_filename', default='sentence_piece', type=str)
    parser.add_argument('--warmup_steps', default=4000, type=int)
    args = parser.parse_args()

    main(args)
