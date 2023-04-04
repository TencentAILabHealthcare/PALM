#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import collections
import torch
import numpy as np
import transformers
from os.path import join

import data.bert_finetuning_er_dataset as module_data
# import data.bert_finetuning_er_alphabeta_dataset as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.bert_binding as module_arch
# from trainer.bert_finetuning_er_trainer import BERTERTrainer as Trainer
from trainer.bert_finetuning_er_trainer_4_input import BERTERTrainer as Trainer
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

    # setup data_loader instances
    config['data_loader']['args']['logger'] = logger
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_dataset(valid=True)
    test_data_loader = data_loader.split_dataset(valid=False,test = True)
    logger.info('Number of pairs in train: {}, valid: {}, and test: {}.'.format(
        data_loader.sampler.__len__(), 
        valid_data_loader.sampler.__len__(), 
        test_data_loader.sampler.__len__()
    ))
    antibody_tokenizer = data_loader.get_antibody_tokenizer()
    antigen_tokenizer = data_loader.get_antigen_tokenizer()

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    
    BERT_params = list(map(id, model.CDRModel_a.parameters())) +  list(map(id, model.CDRModel_b.parameters()))  +list( map(id, model.CDRModel_c.parameters())) + list( map(id, model.ABModel.parameters()))
    base_params = filter(lambda p: id(p) not in BERT_params,
                        model.parameters())

    BERT_lr = config["optimizer"]["args"]["BERT_lr"]
    lr = config["optimizer"]["args"]["lr"]
    weight_decay = config["optimizer"]["args"]["weight_decay"]
    optimizer = torch.optim.AdamW(
                [{'params': base_params},
                {'params': model.CDRModel_a.parameters(), 'lr': BERT_lr},
                 {'params': model.CDRModel_b.parameters(), 'lr': BERT_lr},
                 {'params':model.CDRModel_c.parameters(), 'lr': BERT_lr},
                 {'params':model.ABModel.parameters(), 'lr': BERT_lr} ], lr=lr,weight_decay=weight_decay)
    
    
    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    
    metrics = [getattr(module_metric, met) for met in config['metrics']]
    #logger.info(model)
    
    trainable_params = filter(lambda p : p.requires_grad, model.parameters())
    

    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      test_data_loader=test_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()

    """Test."""
    logger = config.get_logger('test')
    
    # load best checkpoint
    resume = str(config.save_dir / 'model_best.pth')
    logger.info('Loading checkpoint: {} ...'.format(resume))
    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    # save four berts
    cdr1_bert_save_dir = join(config.save_dir, 'cdr1_model')
    cdr2_bert_save_dir = join(config.save_dir, 'cdr2_model')
    cdr3_bert_save_dir = join(config.save_dir, 'cdr3_model')
    ab_bert_save_dir = join(config.save_dir, 'ab_model')
    logger.info(f'Saving four berts to {cdr1_bert_save_dir}, {cdr2_bert_save_dir}, {cdr3_bert_save_dir}, and {ab_bert_save_dir}')
    os.makedirs(cdr1_bert_save_dir)
    model.CDRModel_a.save_pretrained(cdr1_bert_save_dir)
    os.makedirs(cdr2_bert_save_dir)
    model.CDRModel_b.save_pretrained(cdr2_bert_save_dir)
    os.makedirs(cdr3_bert_save_dir)
    model.CDRModel_c.save_pretrained(cdr3_bert_save_dir)
    os.makedirs(ab_bert_save_dir)
    model.ABModel.save_pretrained(ab_bert_save_dir)


    test_output = trainer.test(antibody_tokenizer=antibody_tokenizer,
                               antigen_tokenizer=antigen_tokenizer)
    logger.info(test_output)

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
