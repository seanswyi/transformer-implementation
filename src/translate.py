def convert_ids_to_tokens(data, tokenizer):
    """
    Translates token ID's using the tokenizer.

    Arguments
    ---------
    data: <torch.Tensor> Data to be translated.
    tokenizer: <sentencepiece.SentencePieceProcessor> Sentencepiece tokenizer.
    """
    data = data.long().tolist()
    return [tokenizer.DecodeIds(ids) for ids in data]
