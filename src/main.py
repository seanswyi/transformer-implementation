import argparse
import logging
import os
import time
from datetime import datetime

import torch
import wandb
from torch import nn

from data import WMT2014Dataset
from models.transformer import Transformer
from train import train


logger = logging.getLogger()


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
    msg_format = "[%(asctime)s - %(levelname)s - %(filename)s: %(lineno)d (%(funcName)s)] %(message)s"
    logging.basicConfig(
        format=msg_format,
        level=logging.INFO,
        handlers=[
            logging.FileHandler(filename=args.log_filename),
            logging.StreamHandler(),
        ],
    )

    args_dict = vars(args)
    args_msg = "\n\t".join(f"{arg}: {value}" for arg, value in args_dict.items())
    args_msg = f"\t{args_msg}"
    logger.info("\nArguments:\n%s", args_msg)

    data = WMT2014Dataset(args)
    model = Transformer(args)

    if args.multiple_gpu:
        logger.info("Using multiple GPU's!")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        model = nn.DataParallel(model)

    model = model.to("cuda")
    wandb.watch(model)

    train_start = time.time()
    best_pred, best_epoch = train(args, model, data)
    train_end = time.time()
    logger.info(
        "Training took approximately %s",
        time.strftime("%H:%M:%S", time.gmtime(train_end - train_start)),
    )

    # If we evaluated during training, write predictions.
    if best_pred:
        if not os.path.exists(args.pred_tgt_dir):
            os.makedirs(args.pred_tgt_dir, exist_ok=True)

        pred_filename = f"{args.wandb_name}_pred_epoch{best_epoch}.txt"
        pred_file = os.path.join(args.pred_tgt_dir, pred_filename)
        logger.info(f"Writing predictions and targets to {pred_file}.")
        with open(file=pred_file, mode="w") as f:
            f.write("\n".join(best_pred) + "\n")

    model_file_name = f"{args.log_filename.split('/')[-1]}.pt"
    model_save_file = os.path.join(args.model_save_dir, model_file_name)
    logger.info(f"Saving model in {args.model_save_dir} as {args.log_filename}")
    torch.save(model.state_dict(), model_save_file)

    global_process_end = time.time()
    logger.info(
        f"End of process. Took approximately {time.strftime('%H:%M:%S', time.gmtime(global_process_end - global_process_start))}"
    )


if __name__ == "__main__":
    right_now = time.time()
    timestamp = datetime.fromtimestamp(right_now).strftime("%m%d%Y-%H%M")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.join(current_dir, "..")
    data_dir = os.path.join(parent_dir, "data")
    log_dir = os.path.join(parent_dir, "logs")
    pred_tgt_dir = os.path.join(parent_dir, "outputs")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch_size",
        default=128,
        type=int,
    )
    parser.add_argument(
        "--beta1",
        default=0.9,
        type=float,
    )
    parser.add_argument(
        "--beta2",
        default=0.98,
        type=float,
    )
    parser.add_argument(
        "--epsilon",
        default=10e-9,
        type=float,
    )
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--data_root",
        default=data_dir,
        type=str,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--d_ff",
        default=2048,
        type=int,
    )
    parser.add_argument(
        "--d_model",
        default=512,
        type=int,
    )
    parser.add_argument(
        "--log_filename",
        default="",
        type=str,
    )
    parser.add_argument(
        "--log_step",
        default=50,
        type=int,
    )
    parser.add_argument(
        "--max_seq_len",
        default=50,
        type=int,
    )
    parser.add_argument(
        "--model_save_dir",
        default=os.path.join(parent_dir, "models"),
        type=str,
    )
    parser.add_argument(
        "--multiple_gpu",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--num_epochs",
        default=25,
        type=int,
    )
    parser.add_argument(
        "--num_heads",
        default=8,
        type=int,
    )
    parser.add_argument(
        "--num_stacks",
        default=6,
        type=int,
    )
    parser.add_argument(
        "--src_train_file",
        default=os.path.join(data_dir, "train.fr-en_preprocessed.fr"),
        type=str,
    )
    parser.add_argument(
        "--tgt_train_file",
        default=os.path.join(data_dir, "train.fr-en_preprocessed.en"),
        type=str,
    )
    parser.add_argument(
        "--src_valid_file",
        default=os.path.join(data_dir, "valid.fr-en_preprocessed.fr"),
        type=str,
    )
    parser.add_argument(
        "--tgt_valid_file",
        default=os.path.join(data_dir, "valid.fr-en_preprocessed.en"),
        type=str,
    )
    parser.add_argument(
        "--vocab_size",
        default=16000,
        type=int,
    )
    parser.add_argument(
        "--tokenizer_filename",
        default="sentence_piece",
        type=str,
    )
    parser.add_argument(
        "--wandb_name",
        default="",
        type=str,
    )
    parser.add_argument(
        "--warmup_steps",
        default=4000,
        type=int,
    )

    args = parser.parse_args()

    args.current_dir = current_dir
    args.parent_dir = parent_dir
    args.data_dir = data_dir
    args.log_dir = log_dir
    args.pred_tgt_dir = pred_tgt_dir

    if torch.cuda.is_available():
        args.device = "cuda"
    elif torch.backends.mps.is_available():
        args.device = "mps"
    else:
        args.device = "cpu"

    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir, exist_ok=True)

    if args.wandb_name:
        log_filename = f"transformer_{args.wandb_name}_{timestamp}.log"
        args.log_filename = os.path.join(log_dir, log_filename)
        wandb.init(project="transformer", name=args.wandb_name, config=args)
    else:
        log_filename = f"transformer_{timestamp}.log"
        args.log_filename = os.path.join(log_dir, log_filename)
        wandb.init(project="transformer", config=args)

    main(args)
