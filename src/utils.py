from nltk.translate.bleu_score import sentence_bleu
import numpy as np
# from sacrebleu import sentence_bleu
import torch
import torch.nn.functional as F


def adjust_learning_rate(step_num, args):
    step_num += 1e-20
    term1 = np.power(args.d_model, -0.5)
    term2 = min(np.power(step_num, -0.5), step_num * np.power(args.warmup_steps, -1.5))

    return term1 * term2


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


def calculate_bleu(predictions, targets, tokenizer):
    predictions = predictions.long().tolist()
    targets = targets.long().tolist()

    predictions_decoded = [tokenizer.DecodeIds(ids) for ids in predictions]
    targets_decoded = [tokenizer.DecodeIds(ids) for ids in targets]

    bleu_scores = [sentence_bleu([target], prediction).score for target, prediction in zip(targets_decoded, predictions_decoded)]
    final_bleu_score = sum(bleu_scores) / len(bleu_scores)

    return final_bleu_score


def print_data_stats(og_data, tokenized_data):
    src_longest = max([len(x[0]) for x in tokenized_data])
    tgt_longest = max([len(x[1]) for x in tokenized_data])
    print('==================================================================================================')
    print("len(og_data) = {}".format(len(og_data)))
    print('--------------------------------------------------------------------------------------------------')
    print("Longest sequence for src length: {}".format(src_longest))
    print("Longest sequence for src sentence: {}".format(og_data[np.argmax([len(x[0]) for x in tokenized_data])][0]))
    print('--------------------------------------------------------------------------------------------------')
    print("Shortest sequence for src length: {}".format(min([len(x[0]) for x in tokenized_data])))
    print("Shortest sequence for src sentence: {}".format(og_data[np.argmin([len(x[0]) for x in tokenized_data])][0]))
    print('--------------------------------------------------------------------------------------------------')
    print("Longest sequence for tgt length: {}".format(tgt_longest))
    print("Longest sequence for tgt sentence: {}".format(og_data[np.argmax([len(x[1]) for x in tokenized_data])][1]))
    print('--------------------------------------------------------------------------------------------------')
    print("Shortest sequence for src length: {}".format(min([len(x[1]) for x in tokenized_data])))
    print("Shortest sequence for src sentence: {}".format(og_data[np.argmin([len(x[1]) for x in tokenized_data])][1]))
    print('==================================================================================================')
