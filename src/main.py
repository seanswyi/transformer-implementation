import argparse
import os

import sentencepiece as spm

from data import WMT2014Dataset


def main(args):
    import pdb; pdb.set_trace()
    data = WMT2014Dataset(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='../data/', type=str)
    parser.add_argument('--src_file', default='../data/train.fr-en_preprocessed.fr', type=str)
    parser.add_argument('--tgt_file', default='../data/train.fr-en_preprocessed.en', type=str)
    parser.add_argument('--dev_file', default='../data/valid.fr-en_preprocessed.fr', type=str)
    parser.add_argument('--max_seq_len', default=100, type=int)
    parser.add_argument('--tokenizer_file', default='../sentence_piece.model', type=str)
    args = parser.parse_args()

    main(args)
