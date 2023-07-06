import logging
import os

import numpy as np
import sentencepiece as spm
import torch
from tqdm import tqdm


logger = logging.getLogger()


class WMT2014Dataset:
    """
    Object to hold WMT 2014 dataset.

    Attributes (in alphabetical order)
    -----------------------------------
    args: <argparse.Namespace> Arguments used for overall process.
    batch_size: <int> Number of samples to process at once. Default is 128.
    create_batches: <method> Method to create batches of data according to args.batch_size.
    load: <method> Method for loading raw text data into train_data and valid_data.
    max_seq_len: <int> Maximum sequence length for each sample. Default is 50.
    process: <method> Method to convert tokenized input data into NumPy NDarray templates.
    shuffle: <method> Shuffles the training data.
    src_data: <list> Raw source data, French in this case.
    src_train_data: <numpy.ndarray> NumPy array holding the source data from train_tokenized_data.
    src_valid_data: <numpy.ndarray> NumPy array holding the source data from valid_tokenized_data.
    tgt_data: <list> Raw target data, English in this case.
    tgt_train_data: <numpy.ndarray> NumPy array holding the target data from train_tokenized_data.
    tgt_valid_data: <numpy.ndarray> NumPy array holding the target data from valid_tokenized_data.
    tokenize: <method> Method to tokenize raw data.
    tokenizer: <sentencepiece.SentencePieceProcessor> Sentencepiece tokenizer.
    tokenizer_path: <str> Path to directory where vocabulary and tokenizer files are saved.
    train_data: <torch.Tensor> Training data. Shape is [num_batches, batch_size, 2, max_seq_len]. \
        2 refers to source and target.
    train_tokenized_data: <list> List containing tokenized raw training data.
    valid_data: <torch.Tensor> Validation data. Shape is [num_batches, batch_size, 2, max_seq_len]. \
        2 refers to source and target.
    valid_tokenized_data: <list> List containing tokenized raw training data.

    Overall data processing pipeline is:
    load (list) -> tokenize (list) -> process (numpy.ndarray) -> create_batches (torch.Tensor)
    """

    def __init__(self, args):
        """
        Most of the stuff is done inside this method. Every preprocessing step is carried out upon calling a WMT2014Dataset object.

        Arguments
        ---------
        args: <argparse.Namespace> Arguments used for overall process.
        """
        self.args = args
        self.batch_size = self.args.batch_size

        ############# Load raw data. ####################################################################################
        self.train_data = self.load(mode="train")
        self.valid_data = self.load(mode="valid")
        #################################################################################################################

        ############# Load/train tokenizer. #############################################################################
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer_path = os.path.join(
            self.args.data_root, self.args.tokenizer_filename
        )
        try:
            self.tokenizer.load(self.tokenizer_path + ".model")
        except OSError:
            logger.info(
                "Training tokenizer and saving in %s" % self.args.tokenizer_filename
            )
            train_command = "--input={} --model_prefix={} --vocab_size={} --model_type=bpe --bos_id=2 --eos_id=3 --pad_id=0 --unk_id=1"
            train_file = ",".join([self.args.src_train_file, self.args.tgt_train_file])
            spm.SentencePieceTrainer.train(
                train_command.format(
                    train_file, self.tokenizer_path, self.args.vocab_size
                )
            )
            self.tokenizer.load(self.tokenizer_path + ".model")
        #################################################################################################################

        self.train_tokenized_data = self.tokenize(mode="train")
        self.valid_tokenized_data = self.tokenize(mode="valid")

        src_train_longest = max([len(x[0]) for x in self.train_tokenized_data])
        tgt_train_longest = max([len(x[1]) for x in self.train_tokenized_data])

        if self.args.max_seq_len:
            self.max_seq_len = self.args.max_seq_len
        else:
            self.max_seq_len = max(src_train_longest, tgt_train_longest)

        self.src_train_data, self.tgt_train_data = self.process(
            self.train_tokenized_data
        )
        self.src_valid_data, self.tgt_valid_data = self.process(
            self.valid_tokenized_data
        )

        self.train_data = torch.tensor(
            [[x, y] for x, y in zip(self.src_train_data, self.tgt_train_data)]
        )
        self.valid_data = torch.tensor(
            [[x, y] for x, y in zip(self.src_valid_data, self.tgt_valid_data)]
        )

        ############# Create batches. ##### #############################################################################
        self.train_data = self.create_batches(data=self.train_data)
        self.valid_data = self.create_batches(data=self.valid_data)
        #################################################################################################################

    def load(self, mode="train"):
        """
        Method to load data from raw text files. Each src and tgt is loaded according to training or validation.

        Keyword Arguments
        -----------------
        mode: <str> Determines which data to load.

        Returns
        -------
        <list> List of [src, tgt] list pairs.
        """
        if mode == "train":
            logger.info(
                "Loading src and tgt from %s | %s"
                % (self.args.src_train_file, self.args.tgt_train_file)
            )
            with open(file=self.args.src_train_file, mode="r", encoding="utf-8") as f:
                self.src_data = [line.lower().strip() for line in f.readlines()]

            with open(file=self.args.tgt_train_file, mode="r", encoding="utf-8") as f:
                self.tgt_data = [line.lower().strip() for line in f.readlines()]
        elif mode == "valid":
            logger.info(
                "Loading src and tgt from %s | %s"
                % (self.args.src_valid_file, self.args.tgt_valid_file)
            )
            with open(file=self.args.src_valid_file, mode="r", encoding="utf-8") as f:
                self.src_data = [line.lower().strip() for line in f.readlines()]

            with open(file=self.args.tgt_valid_file, mode="r", encoding="utf-8") as f:
                self.tgt_data = [line.lower().strip() for line in f.readlines()]
        else:
            raise NotImplementedError

        # If we're debugging, then we only need a small number of samples. Doesn't have to be 100.
        if self.args.debug:
            self.src_data = self.src_data[:100]
            self.tgt_data = self.tgt_data[:100]

        return [[src, tgt] for src, tgt in zip(self.src_data, self.tgt_data)]

    def tokenize(self, mode="train"):
        """
        Method to tokenize raw text data.

        Keyword Arguments
        -----------------
        mode: <str> Determines which data to load.

        Returns
        -------
        <list> List of tokenized [src, tgt] list pairs.
        """
        if mode == "train":
            logger.info("Tokenizing train data...")
            progress_bar = tqdm(
                iterable=self.train_data,
                desc="Tokenizing train data",
                total=len(self.train_data),
            )
        elif mode == "valid":
            logger.info("Tokenizing valid data...")
            progress_bar = tqdm(
                iterable=self.valid_data,
                desc="Tokenizing valid data",
                total=len(self.valid_data),
            )
        else:
            raise NotImplementedError

        return [
            [self.tokenizer.EncodeAsIds(src), self.tokenizer.EncodeAsIds(tgt)]
            for src, tgt in progress_bar
        ]

    def process(self, data):
        """
        Method to tokenize raw text data.

        Arguments
        ---------
        data: <list> List of tokenized data to process.

        Returns
        -------
        src_data_template: <numpy.ndarray> NumPy array holding the source data inserted into a template of zeros.
        tgt_data_template: <numpy.ndarray> NumPy array holding the target data inserted into a template of zeros.
        """
        logger.info("Converting tokenized data into input templates.")
        src_data = [
            [self.tokenizer.bos_id()] + x[0] + [self.tokenizer.eos_id()] for x in data
        ]
        tgt_data = [x[1] + [self.tokenizer.eos_id()] for x in data]
        src_data_template = np.zeros(shape=(len(data), self.max_seq_len))
        tgt_data_template = np.zeros(shape=(len(data), self.max_seq_len))

        assert (
            src_data_template.shape == tgt_data_template.shape
        ), "src and tgt are different shapes!"

        count = 0
        for i in range(src_data_template.shape[0]):
            try:
                src_data_template[i][: len(src_data[i])] = src_data[i]
                tgt_data_template[i][: len(tgt_data[i])] = tgt_data[i]
            except ValueError:
                count += 1
                continue

        logger.info(f"{count} samples were discarded due to length.")

        return src_data_template, tgt_data_template

    def create_batches(self, data):
        """
        Method to chunk data into the appropriate number of batches depending on batch_size.

        Arguments
        ---------
        data: <torch.Tensor> NumPy NDarrays from the previous stage are pre-converted into PyTorch Tensors.

        Returns
        -------
        <torch.Tensor> Reshaped into size [num_batches, batch_size, 2, max_seq_len].
        """
        num_batches = len(data) // self.batch_size
        batch_data = data[: (num_batches * self.batch_size)]

        num_discarded_samples = len(data) - (num_batches * self.batch_size)
        if num_discarded_samples:
            logger.info(
                "Discarding %d sample(s)."
                % (len(data) - (num_batches * self.batch_size))
            )

        return batch_data.view(num_batches, self.batch_size, 2, self.max_seq_len)

    def shuffle(self):
        """
        In-place method to shuffle training data.

        Keep in mind that there is no inherent way to shuffle PyTorch Tensors, so you have to shuffle them by \
            assigning random indices.
        """
        shuffle_idxs = torch.randperm(n=self.train_data.shape[0])
        self.train_data = self.train_data[shuffle_idxs]
