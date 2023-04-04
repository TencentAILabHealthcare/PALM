#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import collections
import torch
import numpy as np

from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
    )

import data.bert_finetuning_er_seq2seq_dataset as module_data
from model.bert_seq2seq import get_EncoderDecoder_model
from model.metric import Seq2Seq_metrics
from parse_config import ConfigParser



def main(config):
    logger = config.get_logger('train')

    # fix random seeds for reproducibility
    seed = config['data_loader']['args']['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    model_input = config['model']['TransformerVariant'].split('-')[0].lower()
    assert config['data_loader']['args']['encoder_input'] == model_input, "The input of dataloader is different from model input!"

    config['data_loader']['args']['logger'] = logger
    dataset = config.init_obj('data_loader', module_data)
    # train_dataset, valid_dataset = dataset.get_seq2seq_train_dataset()
    train_dataset = dataset.get_train_dataset()
    valid_dataset = dataset.get_valid_dataset()
    
    antibody_tokenizer = dataset.get_antibody_tokenizer()
    antigen_tokenizer = dataset.get_antigen_tokenizer()
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=config._save_dir,
        overwrite_output_dir=True,
        num_train_epochs=config['trainer']['epochs'],
        per_device_train_batch_size=config['trainer']['batch_size'],
        learning_rate=config['trainer']['lr'],
        warmup_ratio=config['trainer']['warmup'],
        evaluation_strategy="steps",
        eval_steps=config['trainer']['eval_steps'],
        
        
        eval_accumulation_steps=config['trainer']['eval_accumulation_steps'] if 'eval_accumulation_steps' in config['trainer'] else None,
        per_device_eval_batch_size=config['trainer']['batch_size'],
        logging_strategy="steps",
        logging_steps=config['trainer']['logging_steps'],
        save_strategy="steps",
        save_steps=config['trainer']['save_steps'],

        save_total_limit=1,
        dataloader_num_workers=1,
        load_best_model_at_end=True,
        no_cuda=False,  # Useful for debugging
        skip_memory_metrics=True,
        disable_tqdm=False,
        greater_is_better=True,
        metric_for_best_model='eval_average_blosum',
        logging_dir=config._log_dir,
        predict_with_generate=True)

    model = get_EncoderDecoder_model(
        logger=logger,
        TransformerVariant=config['model']['TransformerVariant'],
        AntibodyBert_dir=config['model']['AntibodyBert_dir'],
        AntigenBert_dir=config['model']['AntigenBert_dir'],
        antibody_tokenizer=antibody_tokenizer,
        antigen_tokenizer=antigen_tokenizer,
        antibody_max_len=config['data_loader']['args']['antibody_max_len'],
        antigen_max_len=config['data_loader']['args']['antigen_max_len'],
        # resume=config['model']['resume'],
    )
    # logger.info(model)

    trainable_params = model.parameters()
    params = sum([np.prod(p.size()) for p in trainable_params if p.requires_grad])
    logger.info(f'Trainable parameters {params}.')

    logger.info('Setting parameters for beam search decoding')
    model.config.early_stopping = config['model']['beam_search']['early_stopping']
    model.config.num_beams = config['model']['beam_search']['num_beams']
    model.config.no_repeat_ngram_size = config['model']['beam_search']['no_repeat_ngram_size']
    model.config.forced_eos_token_id = model.config.eos_token_id
    
    
    ## set all parameters that name include "cross" to be trainable, else parameters that name not include "cross" to be frozen
    for name, param in model.named_parameters():
        if 'cross' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    seq2seq_metrics = Seq2Seq_metrics(
        logger=logger,
        model_variant=config['model']['TransformerVariant'],
        antibody_tokenizer=antibody_tokenizer,
        antigen_tokenizer=antigen_tokenizer,
        blosum_dir=config['metrics']['blosum_dir'],
        blosum=config['metrics']['blosum']
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,  # Defaults to None, see above
        compute_metrics=seq2seq_metrics.compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=10)]
    )

    trainer.train()
    trainer.save_model(config._save_dir)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='config.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-local_rank', '--local_rank', default=None, type=str,
                      help='local rank for nGPUs training')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
