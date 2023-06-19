#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import argparse
import collections
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import join

from transformers import (
    EncoderDecoderModel)

import data.bert_finetuning_er_seq2seq_dataset as module_data
import model.bert_binding as module_arch
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

    model_input = config['TransformerVariant'].split('-')[0].lower()

    config['data_loader']['args']['logger'] = logger
    dataset_test = config.init_obj('data_loader', module_data)
    use_dataset = dataset_test.pair_df

    antigen_tokenizer = dataset_test.get_antigen_tokenizer()
    antibody_tokenizer = dataset_test.get_antibody_tokenizer()
    antigen_split_fn = dataset_test.get_antigen_split_fn()
    antibody_split_fn = dataset_test.get_antibody_split_fn()

    # load model
    logger.info(f"Loading pre-trained model from {config.resume}")
    model = EncoderDecoderModel.from_pretrained(config.resume).to("cuda")

    log_example = []

    def seq_generate(input_seq, max_length, input_split_fn, input_tokenizer, target_tokenizer, beams, gene_num=1000,gen_max_len = 16,gen_min_len = 16):
        input_tokenized = input_tokenizer(" ".join(input_split_fn(input_seq)),
                                          padding="max_length",
                                          max_length=max_length,
                                          truncation=True,
                                          return_tensors="pt")
        input_ids = input_tokenized.input_ids.to("cuda")
        attention_mask = input_tokenized.attention_mask.to("cuda")
        outputs = model.generate(input_ids, 
                                 max_length=gen_max_len,
                                 min_length=gen_min_len,
                                 attention_mask=attention_mask,
                                 num_beams=beams,
                                 do_sample=False,
                                 diversity_penalty = 0.05,
                                 num_beam_groups =10,
                                 num_return_sequences=gene_num,
                                 forced_eos_token_id =23,
                                 no_repeat_ngram_size = 2,
                                 )
        output_str = target_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_str_nospace = [s.replace(" ", "") for s in output_str]
        output_str_nospace = [s for s in output_str_nospace if s != '']

        if len(log_example) == 0:
            logger.info(f'output {outputs[0]} -> output string {output_str_nospace[0]}')
            log_example.append(1)

        return output_str_nospace

    if model_input == 'antigen':
        logger.info('The input is antigen, generate antibody sequences.')

        ###
        # use = ["PDVDLGDISGINAS"]

        # XBB:
        use = [config["use_antigen"]]
        
        result_dict = {'Antigen': [], 'Generated_CDR_H3': []}
        # for antigen in tqdm(list(set(use_dataset['antigen']))):
        for antigen in tqdm(use):
            predict_seq = seq_generate(input_seq=antigen, 
                                       max_length=config['data_loader']['args']['antigen_max_len'],
                                       input_split_fn=antigen_split_fn,
                                       input_tokenizer=antigen_tokenizer,
                                       target_tokenizer=antibody_tokenizer,
                                       
                                       beams=1000,
                                       gene_num=1000,
                                       gen_max_len = 18,
                                       gen_min_len = 14)

            result_dict['Antigen'] += [antigen] * len(predict_seq)
            result_dict['Generated_CDR_H3'] += predict_seq
    
    else:
        logger.info('The input is antibody, generate antigen sequences.')
        result_dict = {'CDR_H3': [], 'Generated_Antigen': []}
        
        antigen_list = list(set(use_dataset['antigen']))
        used_antigen_list = []
        selected_single_chain_list = []
        random.seed(0)
        for antigen in antigen_list:
            single_chain_list = list(set(use_dataset[use_dataset['antigen']==antigen]['single_chain']))
            if len(single_chain_list) > 100:
                selected_single_chain = random.choices(single_chain_list, k=100)
            else:
                selected_single_chain = single_chain_list
            used_antigen_list += [antigen] * len(selected_single_chain)
            selected_single_chain_list += selected_single_chain
        selected_single_chain_df = pd.DataFrame({'antigen': used_antigen_list,
                                         'single_chain': selected_single_chain_list})
        selected_single_chain_df.to_csv(join(config._log_dir, 'selected_single_chain.csv'), index=False)
        logger.info('For each antigen, random sample 100 single_chains for generation.')
        logger.info('In total, {} antigens needs to be generated'.format(len(selected_single_chain_list)*5))

        for single_chain in tqdm(list(selected_single_chain_list)):
            predict_seq = seq_generate(input_seq=single_chain, 
                                       max_length=config['data_loader']['args']['antibody_max_len'],
                                       input_split_fn=antibody_split_fn,
                                       input_tokenizer=antibody_tokenizer,
                                       target_tokenizer=antigen_tokenizer,
                                       beams=10,
                                       gene_num=5)

            result_dict['CDR_H3'] += [single_chain] * len(predict_seq)
            result_dict['Generated_Antigen'] += predict_seq

    result_df = pd.DataFrame(result_dict)
    # result_df.to_csv(join(config._log_dir, 'result.csv'), index=False)



    origin_seq = config["origin_seq"]
    cdrh3_begin = config["cdrh3_begin"]
    cdrh3_end = config["cdrh3_end"]
    orinig_light = config["origin_light"]
    gen_heavy_list = []
    light_list = []
    for c in result_df["Generated_CDR_H3"]:
        gen_antibody = origin_seq[:cdrh3_begin] + c + origin_seq[cdrh3_end:]
        gen_heavy_list.append(gen_antibody)
        light_list.append(orinig_light)
    result_df["Heavy_Chain"] = gen_heavy_list
    result_df["Light_Chain"] = light_list
    result_df.to_csv(join(config._log_dir, 'result.csv'), index=False)



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