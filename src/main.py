import argparse
import logging
import os

from data import WMT2014Dataset
from models.transformer import Transformer

logger = logging.getLogger()


def train(args, model, data):
    pass


def main(args):
    msg_format = '[%(asctime)s - %(levelname)s - %(filename)s: %(lineno)d (%(funcName)s)] %(message)s'
    logging.basicConfig(format=msg_format, level=logging.INFO, handlers=[logging.StreamHandler()])

    data = WMT2014Dataset(args)
    model = Transformer(args)
    import pdb; pdb.set_trace()
    thing = model(data.src_train_data[0], data.tgt_train_data[0])
    # Run the embedded version into single head attention.
    import pdb; pdb.set_trace()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--data_root', default='../data/', type=str)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--d_ff', default=2048, type=int)
    parser.add_argument('--d_model', default=512, type=int)
    parser.add_argument('--d_k', default=512, type=int)
    parser.add_argument('--d_v', default=512, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--num_stacks', default=6, type=int)
    parser.add_argument('--src_train_file', default='../data/train.fr-en_preprocessed.fr', type=str)
    parser.add_argument('--tgt_train_file', default='../data/train.fr-en_preprocessed.en', type=str)
    parser.add_argument('--src_valid_file', default='../data/valid.fr-en_preprocessed.fr', type=str)
    parser.add_argument('--tgt_valid_file', default='../data/valid.fr-en_preprocessed.en', type=str)
    parser.add_argument('--vocab_size', default=20000, type=int)
    parser.add_argument('--tokenizer_filename', default='sentence_piece', type=str)
    args = parser.parse_args()

    main(args)
