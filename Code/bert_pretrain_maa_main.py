#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import collections
import torch
import numpy as np

from transformers import (
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback)

import data.bert_pretrain_maa_dataset as module_data
from data.utility import DatasetSplit
from model.bert_pretrain import get_bert_model
from model.metric import MAA_metrics
from parse_config import ConfigParser


def main(config):
    logger = config.get_logger('train')

    # fix random seeds for reproducibility
    seed = config['dataset']['args']['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    holdout = config['dataset']['args']['test_split']
    config['dataset']['args']['config'] = config
    config['dataset']['args']['logger'] = logger
    dataset = config.init_obj('dataset', module_data)
    train_dataset = dataset.get_dataset()
    test_dataset = None

    if holdout is not None:
        assert 0.0 < holdout < 1.0, "Must hold out a fractional proportion of data"
        # No validation since we aren't really doing MLM elsewhere
        test_dataset = DatasetSplit(
            logger=logger, full_dataset=train_dataset, split="test", valid=0, test=holdout
        )
        train_dataset = DatasetSplit(
            logger=logger, full_dataset=train_dataset, split="train", valid=0, test=holdout
        )
    
    tokenizer = dataset.get_tokenizer()
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    
    training_args = TrainingArguments(
        output_dir=config._save_dir,
        overwrite_output_dir=True,
        num_train_epochs=config['trainer']['epochs'],
        per_device_train_batch_size=config['trainer']['batch_size'],
        learning_rate=config['trainer']['lr'],
        warmup_ratio=config['trainer']['warmup'],
        evaluation_strategy="steps",
        eval_steps = 5000,
        save_steps = 5000,
        eval_accumulation_steps=config['trainer']['eval_accumulation_steps'] if 'eval_accumulation_steps' in config['trainer'] else None,
        per_device_eval_batch_size=int(config['trainer']['batch_size'] / 4),
        logging_strategy="steps",
        logging_steps=config['trainer']['logging_steps'],
        save_strategy="steps",
        save_total_limit=5,
        dataloader_num_workers=8,
        load_best_model_at_end=True,
        no_cuda=False,  # Useful for debugging
        skip_memory_metrics=True,
        disable_tqdm=False,
        metric_for_best_model='acc',
        logging_dir=config._log_dir)

    vocab_size = dataset.get_vocab_size()
    pad_token_id = dataset.get_pad_token_id()
    logger.info(f'Vocab size {vocab_size} and pad token id {pad_token_id}')
    model = get_bert_model(
        logger=logger, 
        bert_variant=config['model']['bert'], 
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        **config['model']['args'])
    #logger.info(model)

    trainable_params = model.parameters()
    params = sum([np.prod(p.size()) for p in trainable_params if p.requires_grad])
    logger.info(f'Trainable parameters {params}.')
    
    token_with_special_list = dataset.get_token_list()
    maa_metrics = MAA_metrics(token_with_special_list=token_with_special_list,
                              blosum_dir=config['metrics']['blosum_dir'],
                              blosum=config['metrics']['blosum'])
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,  # Defaults to None, see above
        compute_metrics=maa_metrics.compute_metrics,
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
        CustomArgs(['--bs', '--batch_size'], type=int, target='trainer;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
