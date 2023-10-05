import logging
import random

import torch
import wandb
from sacrebleu import corpus_bleu
from tqdm import tqdm


logger = logging.getLogger()


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
    tokenizer = data.tokenizer

    model = model.to(args.device)
    model.eval()

    eval_loss = 0.0
    pred_texts = []
    tgts = []

    with torch.no_grad():
        eval_progress_bar = tqdm(
            iterable=data.valid_dataloader,
            desc="Evaluating",
            total=len(data.valid_dataloader),
        )
        for batch in eval_progress_bar:
            batch["src"] = batch["src"].to(args.device)
            batch["tgt"] = batch["tgt"].to(args.device)

            output = model(**batch)

            loss = criterion(
                output.view(-1, args.vocab_size), batch["tgt"].view(-1).long()
            )
            eval_loss += loss.item()

            try:
                translated_texts = model.translate(
                    batch["src"],
                    device=args.device,
                )
            except AttributeError:
                translated_texts = model.module.translate(
                    batch["src"],
                    device=args.device,
                )

            decoded_tgts = tokenizer.decode_ids(batch["tgt"])

            pred_texts.extend(translated_texts)
            tgts.extend(decoded_tgts)

        assert len(pred_texts) == len(tgts), "Lens of preds and tgts don't match!"
        sample_idxs = random.sample(population=range(len(pred_texts)), k=5)
        for idx in sample_idxs:
            sample_str = f"Pred: {pred_texts[idx]}\nTgt: {tgts[idx]}"
            logger.info("Sampled pred and tgt:\n%s", sample_str)

        wandb.log({"Evaluation Loss": eval_loss})

    bleu_score = corpus_bleu(pred_texts, [tgts]).score
    wandb.log({"Evaluation BLEU": bleu_score})

    return eval_loss, bleu_score, pred_texts, tgts
