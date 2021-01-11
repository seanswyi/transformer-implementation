import argparse
from datetime import datetime
import logging
import os
import time

from sacrebleu import corpus_bleu
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import wandb

from data import WMT2014Dataset
from models.transformer import Transformer
from utils import adjust_learning_rate, decode_autoregressive, translate


logger = logging.getLogger()


def train(args, model, data):
    """
    Function to perform training and (optionally) evaluation.

    Arguments
    ---------
    args: <argparse.Namespace> Arguments used for overall process.
    model: <models.transformer.Transformer> Transformer model.
    data: <data.WMT2014Dataset> Dataset object containing data.

    Returns
    -------
    predictions_and_targets: <list> List containing predictions with their respective ground-truth targets.
    best_epoch: <int> Epoch that the model obtained the best BLEU score.

    Training is fairly straightforward. One caveat is that predictions for training are \
        obtained in a different manner from evaluation.
    Training simply returns the output representations passed through a softmax layer whereas \
        evaluation performs autoregressive decoding for the prediction.

    BLEU score is calculated on the entire predictions and entire targets using SacreBLEU's \
        corpus_bleu function.
    """
    tokenizer = data.tokenizer

    if torch.cuda.is_available():
        model = model.to('cuda')
    else:
        logger.warning("Not using GPU!")

    model.train()

    global_step = 0

    adjusted_lr = adjust_learning_rate(global_step, args.d_model, args.warmup_steps)
    criterion = nn.NLLLoss(ignore_index=0)
    optimizer = optim.Adam(params=model.parameters(), lr=adjusted_lr)

    best_bleu = 0.0
    best_pred = []
    best_epoch = 0

    epoch_progress_bar = tqdm(iterable=range(args.num_epochs), desc="Epochs", total=args.num_epochs)
    for epoch in epoch_progress_bar:
        epoch_loss = 0.0

        # Make sure to shuffle (training) data before every training epoch.
        data.shuffle()

        step_progress_bar = tqdm(iterable=data.train_data, desc="Training", total=len(data.train_data))
        epoch_start_time = time.time()
        for step, batch in enumerate(step_progress_bar):
            step_loss = 0.0
            optimizer.zero_grad()

            src, tgt = batch[:, 0], batch[:, 1]

            # Skip empty cases.
            for sample in tgt:
                if sum(sample).item() == 0:
                    continue

            bos_tokens = torch.ones(size=(tgt.shape[0],)).reshape(-1, 1) * 2
            tgt_shifted_right = torch.cat((bos_tokens, tgt), dim=1)[:, :-1] # Truncate last token to match size.

            # Find case where there is no padding and append EOS token.
            tgt_shifted_right_last_idxs = tgt_shifted_right[:, -1].long().tolist()
            nonzero_indices = [idx for idx, x in enumerate(tgt_shifted_right_last_idxs) if x != 0]
            if nonzero_indices:
                for idx in nonzero_indices:
                    tgt_shifted_right[idx][-1] = tokenizer.eos_id()

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

            adjusted_lr = adjust_learning_rate(global_step, args.d_model, args.warmup_steps)
            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = adjusted_lr

            output_probs = F.softmax(output, dim=-1)
            predictions = torch.argmax(output_probs, dim=-1)

            if (step + 1) % args.log_step == 0:
                logger.info(f"Step: {step} | Loss: {step_loss} | LR: {adjusted_lr}")
                logger.info(f"Target Sample Tokens: {tgt[0].detach().long().tolist()}")
                logger.info(f"Target Shifted Right Tokens: {tgt_shifted_right[0].detach().long().tolist()}")
                logger.info(f"Target Sample: {tokenizer.DecodeIds(tgt[0].detach().long().tolist())}")
                logger.info(f"Prediction Sample Tokens: {predictions[0].detach().long().tolist()}")
                logger.info(f"Prediction Sample: {tokenizer.DecodeIds(predictions[0].detach().long().tolist())}")

            wandb.log({'Training Loss': step_loss}, step=global_step)
            wandb.log({'Learning Rate': adjusted_lr}, step=global_step)

            global_step += 1

        wandb.log({'Epoch Loss': epoch_loss}, step=epoch)
        epoch_end_time = time.time()
        logger.info(f"One training epoch took approximately {time.strftime('%H:%M:%S', time.gmtime(epoch_end_time - epoch_start_time))}")

        if args.evaluate_during_training:
            evaluation_start = time.time()
            eval_loss, predictions_translated, targets_translated = evaluate(args, model, data, criterion)
            evaluation_end = time.time()
            logger.info(f"Evaluation took approximately {time.strftime('%H:%M:%S', time.gmtime(evaluation_end - evaluation_start))}")

            bleu_score = corpus_bleu('\n'.join(predictions_translated), '\n'.join(targets_translated)).score
            wandb.log({"Evaluation BLEU": bleu_score})

            if bleu_score > best_bleu:
                predictions_and_targets = [f"{p}\t{t}" for p, t in zip(best_pred, targets_translated)]
                best_pred = predictions_translated
                best_epoch = epoch

    return predictions_and_targets, best_epoch


def evaluate(args, model, data, criterion):
    """
    Function to perform evaluation.

    Arguments
    ---------
    args: <argparse.Namespace> Arguments used for overall process.
    model: <models.transformer.Transformer> Transformer model.
    data: <data.WMT2014Dataset> Dataset object containing data.
    criterion: <torch.nn.modules.loss.CrossEntropyLoss>  Loss function.

    Returns
    -------
    eval_loss: <float> Loss value for entire evaluation.
    predictions_translated: <list> Predicted indices decoded using the tokenizer.
    targets_translated: <list> Ground truth indices decoded using the tokenizer.

    Evaluation is conducted using an autoregressive decoding strategy. For more information \
        regarding the decoding, please refer to utils.decode_autoregressive.
    """
    valid_data = data.valid_data
    tokenizer = data.tokenizer
    model.eval()

    eval_progress_bar = tqdm(iterable=valid_data, desc="Evaluating", total=len(valid_data))
    eval_loss = 0.0
    predictions_translated = []
    targets_translated = []

    with torch.no_grad():
        for step, batch in enumerate(eval_progress_bar):
            src, tgt = batch[:, 0], batch[:, 1]
            bos_tokens = torch.ones(size=(tgt.shape[0],)).reshape(-1, 1) * 2
            tgt_shifted_right = torch.cat((bos_tokens, tgt), dim=1)[:, :-1] # Truncate last token to match size.

            # Skip empty cases.
            for thing in tgt_shifted_right:
                if sum(thing) == 2:
                    continue

            # Find case where there is no padding and append EOS token.
            tgt_shifted_right_last_idxs = tgt_shifted_right[:, -1].long().tolist()
            nonzero_indices = [idx for idx, x in enumerate(tgt_shifted_right_last_idxs) if x != 0]
            if nonzero_indices:
                for idx in nonzero_indices:
                    tgt_shifted_right[idx][-1] = tokenizer.eos_id()

            if torch.cuda.is_available():
                src = src.to('cuda')
                tgt = tgt.to('cuda')
                tgt_shifted_right = tgt_shifted_right.to('cuda')
            else:
                logger.warning("Not using GPU!")

            output = model(src, tgt_shifted_right)
            loss = criterion(output.view(-1, args.vocab_size), tgt.view(-1).long())
            eval_loss += loss.item()

            decoded_outputs = decode_autoregressive(model=model, src=src)

            predictions = translate(data=decoded_outputs, tokenizer=tokenizer)
            targets = translate(data=tgt, tokenizer=tokenizer)
            predictions_translated.extend(predictions)
            targets_translated.extend(targets)

            logger.info(f"Step: {step}")
            logger.info(f"Target Sample: {tokenizer.DecodeIds(tgt[0].detach().long().tolist())}")
            logger.info(f"Prediction Sample: {tokenizer.DecodeIds(decoded_outputs[0].detach().long().tolist())}")

        wandb.log({'Evaluation Loss': eval_loss})

    return eval_loss, predictions_translated, targets_translated


def main(args):
    """
    Main function for overall process.

    Arguments
    ---------
    args: <argparse.Namespace> Arguments used for overall process.

    The process is fairly straightforward. Training is conducted first with evaluation being \
        conducted at the end of each training epoch.
    The best model is saved as a PyTorch file.
    """
    global_process_start = time.time()
    msg_format = '[%(asctime)s - %(levelname)s - %(filename)s: %(lineno)d (%(funcName)s)] %(message)s'
    logging.basicConfig(format=msg_format, level=logging.INFO, \
        handlers=[logging.FileHandler(filename=args.log_filename), logging.StreamHandler()])

    data = WMT2014Dataset(args)
    model = Transformer(args)

    if args.multiple_gpu:
        logger.info("Using multiple GPU's!")
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
        model = nn.DataParallel(model)

    model = model.to('cuda')
    wandb.watch(model)

    train_start = time.time()
    best_pred, best_epoch = train(args, model, data)
    train_end = time.time()
    logger.info(f"Training took approximately {time.strftime('%H:%M:%S', time.gmtime(train_end - train_start))}")

    # If we evaluated during training, write predictions.
    if best_pred:
        pred_filename = f"../predictions/{args.wandb_name}_pred_epoch{best_epoch}.txt"
        logger.info(f"Writing predictions and targets to {pred_filename}.")
        with open(file=pred_filename, mode='w') as f:
            f.write('\n'.join(best_pred) + '\n')

    model_file_name = args.log_filename.split('/')[-1]
    model_save_file = os.path.join(args.model_save_dir, model_file_name) + '.pt'
    logger.info(f"Saving model in {args.model_save_dir} as {args.log_filename}")
    torch.save(model.state_dict(), model_save_file)

    global_process_end = time.time()
    logger.info(f"End of process. Took approximately {time.strftime('%H:%M:%S', time.gmtime(global_process_end - global_process_start))}")


if __name__ == '__main__':
    right_now = time.time()
    timestamp = datetime.fromtimestamp(right_now).strftime('%m%d%Y-%H%M')

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--beta1', default=0.9, type=float)
    parser.add_argument('--beta2', default=0.98, type=float)
    parser.add_argument('--epsilon', default=10e-9, type=float)
    parser.add_argument('--evaluate_during_training', action='store_true', default=True)
    parser.add_argument('--data_root', default='../data/', type=str)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--d_ff', default=2048, type=int)
    parser.add_argument('--d_model', default=512, type=int)
    parser.add_argument('--log_filename', default='', type=str)
    parser.add_argument('--log_step', default=50, type=int)
    parser.add_argument('--max_seq_len', default=50, type=int)
    parser.add_argument('--model_save_dir', default='./saved_models', type=str)
    parser.add_argument('--multiple_gpu', action='store_true', default=False)
    parser.add_argument('--num_epochs', default=25, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--num_stacks', default=6, type=int)
    parser.add_argument('--src_train_file', default='../data/train.fr-en_preprocessed.fr', type=str)
    parser.add_argument('--tgt_train_file', default='../data/train.fr-en_preprocessed.en', type=str)
    parser.add_argument('--src_valid_file', default='../data/valid.fr-en_preprocessed.fr', type=str)
    parser.add_argument('--tgt_valid_file', default='../data/valid.fr-en_preprocessed.en', type=str)
    parser.add_argument('--vocab_size', default=16000, type=int)
    parser.add_argument('--tokenizer_filename', default='sentence_piece', type=str)
    parser.add_argument('--wandb_name', default='', type=str)
    parser.add_argument('--warmup_steps', default=4000, type=int)
    args = parser.parse_args()

    logger.info(args)

    if args.wandb_name:
        args.log_filename = f"../logs/{args.wandb_name}_{timestamp}"
        wandb.init(project='transformer', name=args.wandb_name, config=args)
    else:
        args.log_filename = f"../logs/{timestamp}"
        wandb.init(project='transformer', config=args)

    main(args)
