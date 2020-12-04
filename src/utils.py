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
    targets = targets.long().tolist()

    predictions_decoded = [tokenizer.DecodeIds(ids) for ids in predictions]
    targets_decoded = [tokenizer.DecodeIds(ids) for ids in targets]

    bleu_scores = [sentence_bleu(prediction, target).score for prediction, target in zip(predictions_decoded, targets_decoded)]
    final_bleu_score = sum(bleu_scores) / len(predictions)

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
