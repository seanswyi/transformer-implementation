import logging

import torch
import wandb
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

    eval_progress_bar = tqdm(
        iterable=valid_data, desc="Evaluating", total=len(valid_data)
    )
    eval_loss = 0.0
    predictions_translated = []
    targets_translated = []

    with torch.no_grad():
        for step, batch in enumerate(eval_progress_bar):
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
                data=decoded_outputs, tokenizer=tokenizer
            )
            targets = convert_ids_to_tokens(data=tgt, tokenizer=tokenizer)
            predictions_translated.extend(predictions)
            targets_translated.extend(targets)

            logger.info(f"Step: {step}")
            logger.info(
                f"Target Sample: {tokenizer.DecodeIds(tgt[0].detach().long().tolist())}"
            )
            logger.info(
                f"Prediction Sample: {tokenizer.DecodeIds(decoded_outputs[0].detach().long().tolist())}"
            )

        wandb.log({"Evaluation Loss": eval_loss})

    return eval_loss, predictions_translated, targets_translated
