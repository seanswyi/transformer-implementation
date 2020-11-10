import logging
import os

import numpy as np
import sentencepiece as spm
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

logger = logging.getLogger()


class WMT2014Dataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.max_seq_len = self.args.max_seq_len

        self.data = self.load()

        if self.args.debug:
            self.data = self.data[:100]

        ############# Load/train tokenizer. #############################################################################
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer_path = os.path.join(self.args.data_root, self.args.tokenizer_filename)
        try:
            self.tokenizer.load(self.tokenizer_path + '.model')
        except OSError:
            logger.info("Training tokenizer and saving in %s" % self.args.tokenizer_filename)
            train_command = '--input={} --model_prefix={} --vocab_size={} --bos_id=2 --eos_id=3 --pad_id=0 --unk_id=1'
            train_file = ','.join([self.args.src_train_file, self.args.tgt_train_file])
            spm.SentencePieceTrainer.train(train_command.format(train_file, self.tokenizer_path, self.args.vocab_size))
            self.tokenizer.load(self.tokenizer_path + '.model')
        #################################################################################################################

        self.tokenized_data = self.tokenize()

        src_longest = max([len(x[0]) for x in self.tokenized_data])
        tgt_longest = max([len(x[1]) for x in self.tokenized_data])
        print('==================================================================================================')
        print("len(self.data) = {}".format(len(self.data)))
        print('--------------------------------------------------------------------------------------------------')
        print("Longest sequence for src length: {}".format(src_longest))
        print("Longest sequence for src sentence: {}".format(self.data[np.argmax([len(x[0]) for x in self.tokenized_data])][0]))
        print('--------------------------------------------------------------------------------------------------')
        print("Shortest sequence for src length: {}".format(min([len(x[0]) for x in self.tokenized_data])))
        print("Shortest sequence for src sentence: {}".format(self.data[np.argmin([len(x[0]) for x in self.tokenized_data])][0]))
        print('--------------------------------------------------------------------------------------------------')
        print("Longest sequence for tgt length: {}".format(tgt_longest))
        print("Longest sequence for tgt sentence: {}".format(self.data[np.argmax([len(x[1]) for x in self.tokenized_data])][1]))
        print('--------------------------------------------------------------------------------------------------')
        print("Shortest sequence for src length: {}".format(min([len(x[1]) for x in self.tokenized_data])))
        print("Shortest sequence for src sentence: {}".format(self.data[np.argmin([len(x[1]) for x in self.tokenized_data])][1]))
        print('==================================================================================================')

        import pdb; pdb.set_trace()

        self.max_seq_len = max(src_longest, tgt_longest)

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        return self.tokenized_data[idx]

    def load(self):
        logger.info("Loading src and tgt from %s | %s" % (self.args.src_train_file, self.args.tgt_train_file))
        with open(file=self.args.src_train_file, mode='r', encoding='utf-8') as f:
            src_data = [line.lower().strip() for line in f.readlines()]

        with open(file=self.args.tgt_train_file, mode='r', encoding='utf-8') as f:
            tgt_data = [line.lower().strip() for line in f.readlines()]

        return [[src, tgt] for src, tgt in zip(src_data, tgt_data)]

    def tokenize(self):
        logger.info("Tokenizing data...")
        progress_bar = tqdm(iterable=self.data, desc="Tokenizing data", total=len(self.data))
        return [[self.tokenizer.EncodeAsIds(src), self.tokenizer.EncodeAsIds(tgt)] for src, tgt in progress_bar]

    def process(self, data):
        pass
