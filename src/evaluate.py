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
    valid_data = data.valid_data
    tokenizer = data.tokenizer
    model.eval()

    eval_loss = 0.0
    preds_translated = []
    tgts_translated = []

    with torch.no_grad():
        eval_progress_bar = tqdm(
            iterable=valid_data,
            desc="Evaluating",
            total=len(valid_data),
        )
        for batch in eval_progress_bar:
            src, tgt = batch[:, 0], batch[:, 1]
            bos_tokens = torch.ones(size=(tgt.shape[0],)).reshape(-1, 1) * 2
            tgt_shifted_right = torch.cat((bos_tokens, tgt), dim=1)[
                :, :-1
            ]  # Truncate last token to match size.

            # Skip empty cases.
            for thing in tgt_shifted_right:
                if sum(thing) == 2:
                    continue

            # Find case where there is no padding and append EOS token.
            tgt_shifted_right_last_idxs = tgt_shifted_right[:, -1].long().tolist()
            nonzero_indices = [
                idx for idx, x in enumerate(tgt_shifted_right_last_idxs) if x != 0
            ]
            if nonzero_indices:
                for idx in nonzero_indices:
                    tgt_shifted_right[idx][-1] = tokenizer.eos_id()

            if torch.cuda.is_available():
                src = src.to("cuda")
                tgt = tgt.to("cuda")
                tgt_shifted_right = tgt_shifted_right.to("cuda")
            else:
                logger.warning("Not using GPU!")

            output = model(src, tgt_shifted_right)
            loss = criterion(output.view(-1, args.vocab_size), tgt.view(-1).long())
            eval_loss += loss.item()

            decoded_outputs = decode_autoregressive(model=model, src=src)

            predictions = convert_ids_to_tokens(
                data=decoded_outputs,
                tokenizer=tokenizer,
            )
            targets = convert_ids_to_tokens(data=tgt, tokenizer=tokenizer)
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
