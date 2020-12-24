import logging
import os

import numpy as np
import sentencepiece as spm
import torch
from tqdm import tqdm

from utils import print_data_stats

logger = logging.getLogger()


class WMT2014Dataset():
    def __init__(self, args):
        self.args = args
        self.batch_size = self.args.batch_size

        ############# Load raw data. ####################################################################################
        self.train_data = self.load(mode='train')
        self.valid_data = self.load(mode='valid')
        #################################################################################################################

        ############# Load/train tokenizer. #############################################################################
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer_path = os.path.join(self.args.data_root, self.args.tokenizer_filename)
        try:
            self.tokenizer.load(self.tokenizer_path + '.model')
        except OSError:
            logger.info("Training tokenizer and saving in %s" % self.args.tokenizer_filename)
            train_command = '--input={} --model_prefix={} --vocab_size={} --model_type=bpe --bos_id=2 --eos_id=3 --pad_id=0 --unk_id=1'
            train_file = ','.join([self.args.src_train_file, self.args.tgt_train_file])
            spm.SentencePieceTrainer.train(train_command.format(train_file, self.tokenizer_path, self.args.vocab_size))
            self.tokenizer.load(self.tokenizer_path + '.model')
        #################################################################################################################

        self.train_tokenized_data = self.tokenize(mode='train')
        self.valid_tokenized_data = self.tokenize(mode='valid')

        # print_data_stats(og_data=self.train_data, tokenized_data=self.train_tokenized_data)
        # print_data_stats(og_data=self.valid_data, tokenized_data=self.valid_tokenized_data)

        src_train_longest = max([len(x[0]) for x in self.train_tokenized_data])
        tgt_train_longest = max([len(x[1]) for x in self.train_tokenized_data])

        if self.args.max_seq_len:
            self.max_seq_len = self.args.max_seq_len
        else:
            self.max_seq_len = max(src_train_longest, tgt_train_longest)

        self.src_train_data, self.tgt_train_data = self.process(self.train_tokenized_data)
        self.src_valid_data, self.tgt_valid_data = self.process(self.valid_tokenized_data)

        self.train_data = torch.tensor([[x, y] for x, y in zip(self.src_train_data, self.tgt_train_data)])
        self.valid_data = torch.tensor([[x, y] for x, y in zip(self.src_valid_data, self.tgt_valid_data)])

        ############# Create batches. ##### #############################################################################
        self.train_data = self.create_batches(data=self.train_data)
        self.valid_data = self.create_batches(data=self.valid_data)
        #################################################################################################################

        ############# Shuffle training data. ############################################################################
        shuffle_idxs = torch.randperm(n=self.train_data.shape[1])
        self.train_data = self.train_data[:, shuffle_idxs]
        #################################################################################################################

    def load(self, mode='train'):
        if mode == 'train':
            logger.info("Loading src and tgt from %s | %s" % (self.args.src_train_file, self.args.tgt_train_file))
            with open(file=self.args.src_train_file, mode='r', encoding='utf-8') as f:
                self.src_data = [line.lower().strip() for line in f.readlines()]

            with open(file=self.args.tgt_train_file, mode='r', encoding='utf-8') as f:
                self.tgt_data = [line.lower().strip() for line in f.readlines()]
        elif mode == 'valid':
            logger.info("Loading src and tgt from %s | %s" % (self.args.src_valid_file, self.args.tgt_valid_file))
            with open(file=self.args.src_valid_file, mode='r', encoding='utf-8') as f:
                self.src_data = [line.lower().strip() for line in f.readlines()]

            with open(file=self.args.tgt_valid_file, mode='r', encoding='utf-8') as f:
                self.tgt_data = [line.lower().strip() for line in f.readlines()]
        else:
            raise NotImplementedError

        if self.args.debug:
            self.src_data = self.src_data[:100]
            self.tgt_data = self.tgt_data[:100]

        return [[src, tgt] for src, tgt in zip(self.src_data, self.tgt_data)]

    def tokenize(self, mode='train'):
        if mode == 'train':
            logger.info("Tokenizing train data...")
            progress_bar = tqdm(iterable=self.train_data, desc="Tokenizing train data", total=len(self.train_data))
        elif mode == 'valid':
            logger.info("Tokenizing valid data...")
            progress_bar = tqdm(iterable=self.valid_data, desc="Tokenizing valid data", total=len(self.valid_data))
        else:
            raise NotImplementedError

        return [[self.tokenizer.EncodeAsIds(src), self.tokenizer.EncodeAsIds(tgt)] for src, tgt in progress_bar]

    def process(self, data):
        logger.info("Converting tokenized data into input templates.")
        src_data = [[self.tokenizer.bos_id()] + x[0] + [self.tokenizer.eos_id()] for x in data]
        tgt_data = [x[1] + [self.tokenizer.eos_id()] for x in data]
        src_data_template = np.zeros(shape=(len(data), self.max_seq_len))
        tgt_data_template = np.zeros(shape=(len(data), self.max_seq_len))

        assert src_data_template.shape == tgt_data_template.shape, "src and tgt are different shapes!"

        count = 0
        for i in range(src_data_template.shape[0]):
            try:
                src_data_template[i][:len(src_data[i])] = src_data[i]
                tgt_data_template[i][:len(tgt_data[i])] = tgt_data[i]
            except ValueError:
                count += 1
                continue

        logger.info(f"{count} samples were discarded due to length.")

        return src_data_template, tgt_data_template

    def create_batches(self, data):
        num_batches = len(data) // self.batch_size
        batch_data = data[:(num_batches * self.batch_size)]

        num_discarded_samples = len(data) - (num_batches * self.batch_size)
        if num_discarded_samples:
            logger.info("Discarding %d sample(s)." % (len(data) - (num_batches * self.batch_size)))

        return batch_data.view(num_batches, self.batch_size, 2, self.max_seq_len)
