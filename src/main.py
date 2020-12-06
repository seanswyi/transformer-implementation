import argparse
import logging
import os
import time

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

    global_step = 0

    epoch_progress_bar = tqdm(iterable=range(args.num_epochs), desc="Epochs", total=args.num_epochs)
    for epoch in epoch_progress_bar:
        adjusted_lr = adjust_learning_rate(global_step, args)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(params=model.parameters(), lr=adjusted_lr)

        epoch_loss = 0.0

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

            adjusted_lr = adjust_learning_rate(global_step, args)
            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = adjusted_lr

            if step % args.log_step == 0:
                logger.info(f"Step: {step} | Loss: {step_loss} | LR: {adjusted_lr}")

            output_probs = F.softmax(output, dim=2)
            predictions = torch.argmax(output_probs, dim=2)
            training_bleu = calculate_bleu(predictions, tgt, data.tokenzier)

            wandb.log({'Step Loss': step_loss}, step=step)
            wandb.log({'Training BLEU': training_bleu}, step=step)
            wandb.log({'Learning Rate': adjusted_lr}, step=global_step)

            global_step += 1

        wandb.log({'Epoch Loss': epoch_loss}, step=epoch)

        if args.evaluate_during_training:
            evaluate(args, model, data, loss)

    return None


def evaluate(args, model, data, loss):
    model.eval()

    with torch.no_grad():
        eval_loss = 0.0
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
            loss = criterion(output.view(-1, args.vocab_size), tgt.view(-1).long())
            eval_loss += loss.item()

            output_probs = F.softmax(output, dim=2)
            predictions = torch.argmax(output_probs, dim=2)
            eval_bleu = calculate_bleu(predictions, tgt, data.tokenzier)

            wandb.log({'Eval Loss': eval_loss}, step=step)
            wandb.log({'Eval BLEU': eval_bleu}, step=step)

            # for i in range(1, data.max_seq_len + 1):
            #     output = model(src, tgt_shifted_right[:, :i])
            #     probs = F.softmax(output, dim=2)
            #     predictions = torch.argmax(probs, dim=2)
            #     prediction_value = torch.argmax(probs, dim=1).detach().item()
            #     predictions.append(prediction_value)

            # bleu_score = calculate_bleu(predictions, tgt, tokenizer)

            if step % args.log_step == 0:
                logger.info(f"Step: {step} | BLEU: {bleu_score}")

            # wandb.log({'BLEU': bleu_score}, step=step)

    return None


def main(args):
    global_process_start = time.time()
    msg_format = '[%(asctime)s - %(levelname)s - %(filename)s: %(lineno)d (%(funcName)s)] %(message)s'
    logging.basicConfig(format=msg_format, level=logging.INFO, handlers=[logging.StreamHandler()])

    data = WMT2014Dataset(args)
    model = Transformer(args)

    if args.multiple_gpu:
        logger.info("Using multiple GPU's!")
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
        model = nn.DataParallel(model)

    model = model.to('cuda')

    train_start = time.time()
    train(args, model, data)
    train_end = time.time()
    logger.info(f"Training took approximately {time.strftime('%H:%M:%S', time.gmtime(train_end - train_start))}")

    evaluation_start = time.time()
    evaluate(args, model, data, data.tokenizer)
    evaluation_end = time.time()
    logger.info(f"Evaluation took approximately {time.strftime('%H:%M:%S', time.gmtime(evaluation_end - evaluation_start))}")

    global_process_end = time.time()
    logger.info(f"End of process. Took approximately {time.strftime('%H:%M:%S', time.gmtime(global_process_end - global_process_start))}")


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

    # wandb.init(project='transformer', config=args)

    main(args)
