import argparse
import logging
import os
import time
from datetime import datetime

import torch
import wandb
from dotenv import load_dotenv
from sacrebleu import corpus_bleu
from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm

from dataset import TextPairDataset
from models import Tokenizer, Transformer
from utils import adjust_learning_rate, decode_autoregressive, translate


load_dotenv()
logger = logging.getLogger()


def train(
    args,
    model,
    data,
    tokenizer,
):
    if torch.cuda.is_available():
        model = model.to("cuda")
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

    train_progress_bar = tqdm(
        iterable=range(args.num_epochs),
        desc="Epochs",
        total=args.num_epochs,
    )
    for epoch in train_progress_bar:
        epoch_loss = 0.0

        step_progress_bar = tqdm(
            iterable=data.train_dataloader,
            desc="Training",
            total=len(data.train_dataloader),
        )
        epoch_start_time = time.time()
        for step, batch in enumerate(step_progress_bar):
            step_loss = 0.0
            optimizer.zero_grad()

            batch["src"] = batch["src"].to("cuda")
            batch["tgt"] = batch["tgt"].to("cuda")

            output = model(**batch)

            loss = criterion(
                output.view(-1, args.vocab_size), batch["tgt"].view(-1).long()
            )
            step_loss += loss.item()
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

            adjusted_lr = adjust_learning_rate(
                step_num=global_step,
                d_model=args.d_model,
                warmup_steps=args.warmup_steps,
            )
            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]["lr"] = adjusted_lr

            output_probs = F.softmax(output, dim=-1)
            predictions = torch.argmax(output_probs, dim=-1)

            if (step + 1) % args.log_step == 0:
                logger.info(f"Step: {step} | Loss: {step_loss} | LR: {adjusted_lr}")
                logger.info(
                    f"Target Sample Tokens: {batch['tgt'][0].detach().long().tolist()}"
                )
                logger.info(
                    f"Target Sample: {tokenizer.DecodeIds(batch['tgt'][0].detach().long().tolist())}"
                )
                logger.info(
                    f"Prediction Sample Tokens: {predictions[0].detach().long().tolist()}"
                )
                logger.info(
                    f"Prediction Sample: {tokenizer.DecodeIds(predictions[0].detach().long().tolist())}"
                )

            wandb.log({"Training Loss": step_loss}, step=global_step)
            wandb.log({"Learning Rate": adjusted_lr}, step=global_step)

            global_step += 1

        wandb.log({"Epoch Loss": epoch_loss}, step=epoch)
        epoch_end_time = time.time()
        logger.info(
            f"One training epoch took approximately {time.strftime('%H:%M:%S', time.gmtime(epoch_end_time - epoch_start_time))}"
        )

        if args.evaluate_during_training:
            evaluation_start = time.time()
            _, predictions_translated, targets_translated = evaluate(
                args, model, data, criterion
            )
            evaluation_end = time.time()
            logger.info(
                f"Evaluation took approximately {time.strftime('%H:%M:%S', time.gmtime(evaluation_end - evaluation_start))}"
            )

            bleu_score = corpus_bleu(predictions_translated, [targets_translated]).score
            wandb.log({"Evaluation BLEU": bleu_score})
            logger.info("BLEU at epoch %d: %.4f", epoch, bleu_score)

            if bleu_score > best_bleu:
                predictions_and_targets = [
                    f"{p}\t{t}" for p, t in zip(best_pred, targets_translated)
                ]
                best_pred = predictions_translated
                best_epoch = epoch

    return predictions_and_targets, best_epoch


def evaluate(
    args,
    model,
    data,
    criterion,
    tokenizer,
):
    model.eval()

    eval_progress_bar = tqdm(
        iterable=data.valid_dataloader,
        desc="Evaluating",
        total=len(data.valid_dataloader),
    )
    eval_loss = 0.0
    predictions_translated = []
    targets_translated = []

    with torch.no_grad():
        for step, batch in enumerate(eval_progress_bar):
            batch["src"] = batch["src"].to("cuda")
            batch["tgt"] = batch["tgt"].to("cuda")

            output = model(**batch)
            loss = criterion(
                output.view(-1, args.vocab_size), batch["tgt"].view(-1).long()
            )
            eval_loss += loss.item()

            decoded_outputs = decode_autoregressive(model=model, src=batch["src"])

            predictions = translate(data=decoded_outputs, tokenizer=tokenizer)
            targets = translate(data=batch["tgt"], tokenizer=tokenizer)
            predictions_translated.extend(predictions)
            targets_translated.extend(targets)

            logger.info(f"Step: {step}")
            logger.info(
                f"Target Sample: {tokenizer.DecodeIds(batch['tgt'][0].detach().long().tolist())}"
            )
            logger.info(
                f"Prediction Sample: {tokenizer.DecodeIds(decoded_outputs[0].detach().long().tolist())}"
            )

        wandb.log({"Evaluation Loss": eval_loss})

    return eval_loss, predictions_translated, targets_translated


def main(args):
    global_process_start = time.time()

    msg_format = "[%(asctime)s - %(levelname)s - %(filename)s: %(lineno)d (%(funcName)s)] %(message)s"
    logging.basicConfig(
        format=msg_format,
        level=logging.INFO,
        handlers=[
            logging.FileHandler(filename=args.log_filename),
            logging.StreamHandler(),
        ],
    )

    tokenizer = Tokenizer(
        tokenizer_name=args.tokenizer_filename,
        train_text_files=",".join([args.src_train_file, args.tgt_train_file]),
        vocab_size=args.vocab_size,
        tokenization_algo=args.tokenization_algo,
    )
    data = TextPairDataset(args=args, tokenizer=tokenizer)
    model = Transformer(args)

    if args.multiple_gpu:
        logger.info("Using multiple GPU's!")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        model = nn.DataParallel(model)

    model = model.to("cuda")
    wandb.watch(model)

    train_start = time.time()
    best_pred, best_epoch = train(args, model, data, tokenizer)
    train_end = time.time()
    logger.info(
        f"Training took approximately {time.strftime('%H:%M:%S', time.gmtime(train_end - train_start))}"
    )

    # If we evaluated during training, write predictions.
    if best_pred:
        results_filename = f"{args.wandb_name}_results-epoch-{best_epoch}.txt"
        results_filepath = os.path.join(args.outputs_dir, results_filename)
        logger.info(f"Writing predictions and targets to {results_filepath}.")
        with open(file=results_filepath, mode="w") as f:
            f.write("\n".join(best_pred) + "\n")

    model_file_name = args.log_filename.split("/")[-1]
    model_name = os.path.splitext(model_file_name)[0]
    model_save_file = f"{os.path.join(args.model_save_dir, model_name)}.pt"

    logger.info(f"Saving model in {args.model_save_dir} as {args.log_filename}")
    torch.save(model.state_dict(), model_save_file)

    global_process_end = time.time()
    logger.info(
        f"End of process. Took approximately {time.strftime('%H:%M:%S', time.gmtime(global_process_end - global_process_start))}"
    )


if __name__ == "__main__":
    right_now = time.time()
    timestamp = datetime.fromtimestamp(right_now).strftime("%m-%d-%Y-%H%M")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.join(current_dir, "..")
    data_dir = os.path.join(parent_dir, "data")
    log_dir = os.path.join(parent_dir, "logs")
    outputs_dir = os.path.join(parent_dir, "outputs")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.98, type=float)
    parser.add_argument("--epsilon", default=10e-9, type=float)
    parser.add_argument("--evaluate_during_training", action="store_true", default=True)
    parser.add_argument("--data_root", default="../data/", type=str)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--d_ff", default=2048, type=int)
    parser.add_argument("--d_model", default=512, type=int)
    parser.add_argument("--log_filename", default="", type=str)
    parser.add_argument("--log_step", default=50, type=int)
    parser.add_argument("--max_seq_len", default=50, type=int)
    parser.add_argument("--model_save_dir", default="./saved_models", type=str)
    parser.add_argument("--multiple_gpu", action="store_true", default=False)
    parser.add_argument("--num_epochs", default=25, type=int)
    parser.add_argument("--num_heads", default=8, type=int)
    parser.add_argument("--num_stacks", default=6, type=int)
    parser.add_argument(
        "--src_train_file", default="../data/train.fr-en_preprocessed.fr", type=str
    )
    parser.add_argument(
        "--tgt_train_file", default="../data/train.fr-en_preprocessed.en", type=str
    )
    parser.add_argument(
        "--src_valid_file", default="../data/valid.fr-en_preprocessed.fr", type=str
    )
    parser.add_argument(
        "--tgt_valid_file", default="../data/valid.fr-en_preprocessed.en", type=str
    )
    parser.add_argument("--vocab_size", default=16000, type=int)
    parser.add_argument(
        "--tokenization_algo",
        default="bpe",
        type=str,
        choices=["unigram", "bpe", "char", "word"],
        help="Tokenization algorithm used to train tokenizer.",
    )
    parser.add_argument(
        "--tokenizer_filename", default="sentence_piece.model", type=str
    )
    parser.add_argument("--wandb_name", default="", type=str)
    parser.add_argument("--warmup_steps", default=4000, type=int)

    args = parser.parse_args()

    if args.wandb_name:
        args.wandb_name = f"{args.wandb_name}_{timestamp}"
    else:
        args.wandb_name = f"transformer_{timestamp}"

    args.current_dir = current_dir
    args.parent_dir = parent_dir
    args.data_dir = data_dir
    args.log_dir = log_dir
    args.outputs_dir = outputs_dir

    args.tokenizer_filename = os.path.join(data_dir, args.tokenizer_filename)

    log_filename = f"{args.wandb_name}.log"
    args.log_filename = os.path.join(log_dir, log_filename)

    wandb.init(
        project="transformer",
        name=args.wandb_name,
        config=args,
    )

    main(args)
