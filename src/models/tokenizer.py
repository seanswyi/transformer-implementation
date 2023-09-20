import logging

from sentencepiece import SentencePieceProcessor, SentencePieceTrainer


logger = logging.getLogger()


class Tokenizer(SentencePieceProcessor):
    def __init__(
        self,
        tokenizer_name: str = None,
        raw_text_files: str = None,
        vocab_size: int = None,
        tokenizer_algo: str = "bpe",
    ):
        super().__init__()

        if tokenizer_name:
            logger.info("Loading tokenizer model from %s", tokenizer_name)
            self.load(tokenizer_name)
        else:
            if tokenizer_name.endswith(".model"):
                tokenizer_name = tokenizer_name.replace(".model", "")

            logger.info(
                "Model not received, training from scratch using provided text data"
            )

            train_command = (
                f"--input={raw_text_files} "
                f"--model_prefix={tokenizer_name} "
                f"--model_type={tokenizer_algo} "
                f"--bos_id=2 "
                f"--eos_id=3 "
                f"--pad_id=0 "
                f"--unk_id=1"
            )
            SentencePieceTrainer.train(train_command)

            self.load(f"{tokenizer_name}.model")

    def tokenize(self, input_text: str) -> list[int]:
        """Tokenizes text."""
