import logging
import random

import torch
import wandb
from sacrebleu import corpus_bleu
from tqdm import tqdm

from translate import convert_ids_to_tokens
from utils import decode_autoregressive


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
    preds_translated = []
    tgts_translated = []

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

            decoded_outputs = decode_autoregressive(model=model, src=batch["src"])
            predictions = convert_ids_to_tokens(
                data=decoded_outputs,
                tokenizer=tokenizer,
            )
            targets = convert_ids_to_tokens(data=batch["tgt"], tokenizer=tokenizer)
            preds_translated.extend(predictions)
            tgts_translated.extend(targets)

        assert len(preds_translated) == len(
            tgts_translated
        ), "Lens of preds and tgts don't match!"
        sample_idxs = random.sample(population=range(len(preds_translated)), k=5)
        for idx in sample_idxs:
            sample_str = f"Pred: {preds_translated[idx]}\nTgt: {tgts_translated[idx]}"
            logger.info("Sampled pred and tgt:\n%s", sample_str)

        wandb.log({"Evaluation Loss": eval_loss})

    bleu_score = corpus_bleu(
        preds_translated,
        [tgts_translated],
    ).score
    wandb.log({"Evaluation BLEU": bleu_score})

    return eval_loss, bleu_score, preds_translated, tgts_translated
