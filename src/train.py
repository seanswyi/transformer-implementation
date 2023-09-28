import logging
import time

import wandb
from torch import nn, optim
from tqdm import tqdm, trange

from evaluate import evaluate
from utils import adjust_learning_rate


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

    model = model.to(args.device)
    model.train()

    global_step = 0

    adjusted_lr = adjust_learning_rate(
        step_num=global_step,
        d_model=args.d_model,
        warmup_steps=args.warmup_steps,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id())
    optimizer = optim.Adam(params=model.parameters(), lr=adjusted_lr)

    best_bleu = 0.0
    best_pred = []
    best_epoch = 0

    preds_and_tgts = []
    epoch_progress_bar = trange(
        args.num_epochs,
        desc="Epochs",
        total=args.num_epochs,
    )
    for epoch in epoch_progress_bar:
        epoch_loss = 0.0

        train_progress_bar = tqdm(
            iterable=data.train_dataloader,
            desc="Training",
            total=len(data.train_data),
        )
        epoch_start = time.time()
        for batch in train_progress_bar:
            step_loss = 0.0
            optimizer.zero_grad()

            batch["src"] = batch["src"].to(args.device)
            batch["tgt"] = batch["tgt"].to(args.device)

            output = model(**batch)
            loss = criterion(
                output.view(-1, args.vocab_size), batch["tgt"].view(-1).long()
            )
            step_loss += loss.item()
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

            adjusted_lr = adjust_learning_rate(
                global_step, args.d_model, args.warmup_steps
            )
            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]["lr"] = adjusted_lr

            wandb.log({"Training Loss": step_loss}, step=global_step)
            wandb.log({"Learning Rate": adjusted_lr}, step=global_step)

            global_step += 1

        epoch_end = time.time()
        epoch_duration = epoch_end - epoch_start
        epoch_duration_fmt = time.strftime("%H:%M:%S", time.gmtime(epoch_duration))
        logger.info(
            "Epoch %d took %s and has loss: %.4f",
            epoch,
            epoch_duration_fmt,
            epoch_loss,
        )

        eval_start = time.time()
        _, bleu_score, preds_translated, tgts_translated = evaluate(
            args=args,
            model=model,
            data=data,
            criterion=criterion,
        )
        eval_end = time.time()
        eval_duration = eval_end - eval_start
        eval_duration_fmt = time.strftime("%H:%M:%S", time.gmtime(eval_duration))

        logger.info("Evaluation took approximately %s", eval_duration_fmt)
        logger.info("BLEU score at epoch %d: %.4f", epoch, bleu_score)

        if bleu_score > best_bleu:
            preds_and_tgts = [f"{p}\t{t}" for p, t in zip(best_pred, tgts_translated)]
            best_pred = preds_translated
            best_epoch = epoch

    return preds_and_tgts, best_epoch
