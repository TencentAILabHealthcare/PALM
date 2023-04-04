# -*- coding: utf-8 -*-

from transformers import EncoderDecoderModel

def get_EncoderDecoder_model(logger, 
                             TransformerVariant, 
                             AntibodyBert_dir, 
                             AntigenBert_dir,
                             antibody_tokenizer,
                             antigen_tokenizer,
                             antibody_max_len,
                             antigen_max_len,
                             resume=None):
    if resume is not None:
        logger.info(f'Loading EncoderDecoder from {resume}')
        model = EncoderDecoderModel.from_pretrained(resume)
        model.config.decoder_start_token_id = antibody_tokenizer.cls_token_id
        model.config.eos_token_id = antibody_tokenizer.sep_token_id
        model.config.pad_token_id = antibody_tokenizer.pad_token_id
        model.config.vocab_size = antibody_tokenizer.vocab_size
        model.config.max_length = antibody_max_len
        return model
    
    """Load the bert model"""
    logger.info(f'Loading AntibodyBert from {AntibodyBert_dir}')
    logger.info(f'Loading AntigenBert from {AntigenBert_dir}')

    if TransformerVariant == 'Antibody-Antigen':
        logger.info("Using AntibodyBert as encoder, AntigenBert as decoder.")
        model = EncoderDecoderModel.from_encoder_decoder_pretrained(
            AntibodyBert_dir, AntigenBert_dir)
        model.config.decoder_start_token_id = antigen_tokenizer.cls_token_id
        model.config.eos_token_id = antigen_tokenizer.sep_token_id
        model.config.pad_token_id = antigen_tokenizer.pad_token_id
        model.config.vocab_size = antigen_tokenizer.vocab_size
        model.config.max_length = antigen_max_len

    else:
        logger.info("Using AntigenBert as encoder, AntibodyBert as decoder.")
        model = EncoderDecoderModel.from_encoder_decoder_pretrained(
            AntigenBert_dir, AntibodyBert_dir)
        model.config.decoder_start_token_id = antibody_tokenizer.cls_token_id
        model.config.eos_token_id = antibody_tokenizer.sep_token_id
        model.config.pad_token_id = antibody_tokenizer.pad_token_id
        model.config.vocab_size = antibody_tokenizer.vocab_size
        model.config.max_length = antibody_max_len

    return model