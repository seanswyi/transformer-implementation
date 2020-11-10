import sentencepiece as spm
from torch.utils.data import Dataset, DataLoader


class WMT2014Dataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.max_seq_len = self.args.max_seq_len

        self.data = self.load()
        import pdb; pdb.set_trace()

        self.tokenizer = spm.SentencePieceProcessor()
        try:
            self.tokenizer.load(args.tokenizer_file)
        except OSError:
            train_command = '--input={} --model_prefix={} --vocab_size={} --bos_id=2 --eos_id=3 --pad_id=0 --unk_id=1'
            train_file = ','.join([self.args.src_file, self.args.tgt_file])
            spm.SentencePieceTrainer.train(train_command.format(train_file, self.args.tokenizer_file, self.args.vocab))
            self.tokenizer.load(args.tokenizer_file)

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    def load(self):
        with open(file=self.args.src_file, mode='r', encoding='utf-8') as f:
            src_data = [line.lower().strip() for line in f.readlines()]

        with open(file=self.args.tgt_file, mode='r', encoding='utf-8') as f:
            tgt_data = [line.lower().strip() for line in f.readlines()]

        return list(zip(src_data, tgt_data))

    def tokenize(self, data):
        pass

    def process(self, data):
        pass
