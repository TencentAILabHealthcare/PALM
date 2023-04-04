
from transformers import (
    BertConfig,
    BertForMaskedLM,
    ConvBertConfig,
    ConvBertForMaskedLM,
    RoFormerConfig,
    RoFormerForMaskedLM,
)

def get_bert_model(logger, bert_variant: str, vocab_size, pad_token_id, **kwargs):
    """Load the bert model"""
    if bert_variant == "bert":
        logger.info("Loading vanilla BERT model")
        config = BertConfig(
            **kwargs,
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
        )
        model = BertForMaskedLM(config)
    elif bert_variant == "convbert":
        logger.info("Loading ConvBERT model")
        config = ConvBertConfig(
            **kwargs,
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
        )
        model = ConvBertForMaskedLM(config)
    elif bert_variant == "roformer":
        logger.info("Loading RoFormer model")
        config = RoFormerConfig(
            **kwargs,
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
        )
        model = RoFormerForMaskedLM(config)
    else:
        raise ValueError(f"Unrecognized BERT variant: {bert_variant}")
    return model