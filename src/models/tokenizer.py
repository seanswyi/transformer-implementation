import logging

import torch
from sentencepiece import SentencePieceProcessor, SentencePieceTrainer


logger = logging.getLogger()


class Tokenizer(SentencePieceProcessor):
    def __init__(
        self,
        tokenizer_name: str = None,
        train_text_files: str = None,
        vocab_size: int = None,
        tokenization_algo: str = "bpe",
    ):
        super().__init__()

        try:
            logger.info("Loading tokenizer model from %s", tokenizer_name)
            self.load(tokenizer_name)
        except OSError as err:
            logger.error("Caught OSError [%s]. Training from scratch.", err)

            if tokenizer_name.endswith(".model"):
                tokenizer_name = tokenizer_name.replace(".model", "")

            train_command = (
                f"--input={train_text_files} "
                f"--model_prefix={tokenizer_name} "
                f"--model_type={tokenization_algo} "
                f"--bos_id=2 "
                f"--eos_id=3 "
                f"--pad_id=0 "
                f"--unk_id=1"
            )
            SentencePieceTrainer.train(train_command)

            self.load(f"{tokenizer_name}.model")

        self.vocab_size = vocab_size
        self.tokenization_algo = tokenization_algo

    def __call__(
        self,
        input_text: str,
        return_tensors: str = "pt",
    ) -> list[int] | torch.Tensor:
        """Meant to imitate the HuggingFace tokenizers."""
        token_ids = self.tokenize(input_text)

        if return_tensors == "pt":
            token_ids = torch.Tensor(token_ids)

        return token_ids

    def tokenize(self, input_text: str) -> list[int]:
        """Convert text to token IDs."""
        token_ids = self.Tokenize(input_text)
        return token_ids

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        """Convert list of token IDs to token texts."""
        tokens = [self.Decode(id_) for id_ in ids]
        return tokens

    def decode(self, token_ids: list[int]) -> str:
        """Receives token IDs and returns string."""
        decoded_str = self.Decode(token_ids)
        return decoded_str