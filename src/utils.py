import numpy as np
from sacrebleu import sentence_bleu
import torch
import torch.nn.functional as F


def adjust_learning_rate(step_num, args):
    step_num += 1e-20
    term1 = np.power(args.d_model, -0.5)
    term2 = min(np.power(step_num, -0.5), step_num * np.power(args.warmup_steps, -1.5))

    return term1 * term2


def calculate_bleu(predictions, targets, tokenizer):
    predictions = predictions.long().tolist()
    targets = targets.long().tolist()

    predictions_decoded = [tokenizer.DecodeIds(ids) for ids in predictions]
    targets_decoded = [tokenizer.DecodeIds(ids) for ids in targets]

    bleu_scores = [sentence_bleu(prediction, [target]).score for prediction, target in zip(predictions_decoded, targets_decoded)]
    final_bleu_score = sum(bleu_scores) / len(bleu_scores)

    return final_bleu_score


def decode_autoregressive(model, src):
    outputs = torch.ones(size=(src.shape[0],)).reshape(-1, 1) * 2

    if torch.cuda.is_available():
        src = src.to('cuda')
        outputs = outputs.to('cuda')

    for _ in range(src.shape[1]):
        prediction = F.softmax(model(src, outputs), dim=2)
        prediction = torch.argmax(prediction, dim=2)[:, -1]
        outputs = torch.cat((outputs, prediction.view(-1, 1)), dim=-1)

    return outputs[:, 1:]


def translate(data, tokenizer):
    data = data.long().tolist()
    return [tokenizer.DecodeIds(ids) for ids in data]
