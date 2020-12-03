import argparse
import logging
import os

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import wandb

from data import WMT2014Dataset
from models.transformer import Transformer
from utils import adjust_learning_rate, calculate_bleu

logger = logging.getLogger()


def train(args, model, data):
    if torch.cuda.is_available():
        model = model.to('cuda')
    else:
        logger.warning("Not using GPU!")

    model.train()

    epoch_progress_bar = tqdm(iterable=range(args.num_epochs), desc="Epochs", total=args.num_epochs)
    for epoch in epoch_progress_bar:
        adjusted_lr = adjust_learning_rate(0, args) if args.learning_rate == 0 else args.learning_rate
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(params=model.parameters(), lr=adjusted_lr)

        epoch_loss = 0

        step_progress_bar = tqdm(iterable=data.train_data, desc="Training", total=len(data.train_data))
        for step, batch in enumerate(step_progress_bar):
            step_loss = 0.0
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
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

            if step <= args.warmup_steps:
                adjusted_lr = adjust_learning_rate(step, args)
                for i in range(len(optimizer.param_groups)):
                    optimizer.param_groups[i]['lr'] = adjusted_lr

            if step % args.log_step == 0:
                wandb.log({'step_loss': step_loss, 'lr': adjusted_lr}, step=step)
                logger.info(f"Step: {step} | Loss: {step_loss} | LR: {adjusted_lr}")

        wandb.log({'epoch_loss': epoch_loss}, step=epoch)

        if args.evaluate_during_training:
            evaluate(args, model, data, data.tokenizer)

    return 0


def evaluate(args, model, data, tokenizer):
    model.eval()

    with torch.no_grad():
        for step, batch in enumerate(data.valid_data):
            src, tgt = batch[:, 0], batch[:, 1]
            bos_tokens = torch.ones(size=(tgt.shape[0],)).reshape(-1, 1) * 2
            tgt_shifted_right = torch.cat((bos_tokens, tgt), dim=1)[:, :-1]

            if torch.cuda.is_available():
                src = src.to('cuda')
                tgt = tgt.to('cuda')
                tgt_shifted_right = tgt_shifted_right.to('cuda')
            else:
                logger.warning("Not using GPU!")

            output = model(src, tgt_shifted_right)
            bleu_score = calculate_bleu(output, tgt, tokenizer)

            if step % args.log_step == 0:
                wandb.log({'BLEU': bleu_score}, step=step)
                logger.info(f"Step: {step} | BLEU: {bleu_score}")


def main(args):
    msg_format = '[%(asctime)s - %(levelname)s - %(filename)s: %(lineno)d (%(funcName)s)] %(message)s'
    logging.basicConfig(format=msg_format, level=logging.INFO, handlers=[logging.StreamHandler()])

    data = WMT2014Dataset(args)
    model = Transformer(args)

    if args.multiple_gpu:
        logger.info("Using multiple GPU's!")
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
        model = nn.DataParallel(model)

    model = model.to('cuda')
    results = train(args, model, data)
    evaluate(args, model, data, data.tokenizer)

    # Run the embedded version into single head attention.
    import pdb; pdb.set_trace()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--beta1', default=0.9, type=float)
    parser.add_argument('--beta2', default=0.98, type=float)
    parser.add_argument('--epsilon', default=10e-9, type=float)
    parser.add_argument('--evaluate_during_training', action='store_true', default=True)
    parser.add_argument('--data_root', default='../data/', type=str)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--d_ff', default=2048, type=int)
    parser.add_argument('--d_model', default=512, type=int)
    parser.add_argument('--d_k', default=512, type=int)
    parser.add_argument('--d_v', default=512, type=int)
    parser.add_argument('--log_step', default=50, type=int)
    parser.add_argument('--learning_rate', default=0, type=float)
    parser.add_argument('--multiple_gpu', action='store_true', default=False)
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

    wandb.init(project='transformer', config=args)

    main(args)
