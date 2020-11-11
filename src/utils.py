import numpy as np


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
