# -*- coding: utf-8 -*-

import math
from tokenize import Token
import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from bert_data_prepare.tokenizer import get_tokenizer


def min_power_greater_than(value, base=2):
        """
        Return the lowest power of the base that exceeds the given value
        >>> min_power_greater_than(3, 4)
        4.0
        >>> min_power_greater_than(48, 2)
        64.0
        """
        p = math.ceil(math.log(value, base))
        return math.pow(base, p)

class SelfSupervisedDataset(Dataset):
    """
    Mostly for compatibility with transformers library
    LineByLineTextDataset returns a dict of "input_ids" -> input_ids
    """
    # Reference: https://github.com/huggingface/transformers/blob/447808c85f0e6d6b0aeeb07214942bf1e578f9d2/src/transformers/data/datasets/language_modeling.py
    def __init__(self, seqs, 
                       split_fun,
                       tokenizer,
                       max_len,
                       logger, 
                       round_len=True):
        self.seqs = seqs
        self.split_fun = split_fun
        self.logger = logger
        self.tokenizer = tokenizer

        self.logger.info(
            f"Creating self supervised dataset with {len(self.seqs)} sequences")
        
        self.max_len = max_len
        self.logger.info(f"Maximum sequence length: {self.max_len}")
        
        if round_len:
            self.max_len = int(min_power_greater_than(self.max_len, 2))
            self.logger.info(f"Rounded maximum length to {self.max_len}")
        self._has_logged_example = False

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, i):
        seq = self.seqs[i]
        retval = self.tokenizer.encode(self._insert_whitespace(self.split_fun(seq)),
                                       truncation=True, max_length=self.max_len)
        if not self._has_logged_example:
            self.logger.info(f"Example of tokenized input: {seq} -> {retval}")
            self._has_logged_example = True
        return {"input_ids": torch.tensor(retval, dtype=torch.long)}

    def merge(self, other):
        """Merge this dataset with the other dataset"""
        all_seqs = self.seqs + other.seqs
        self.logger.info(
            f"Merged two self-supervised datasets of sizes {len(self)} {len(other)} for dataset of {len(all_seqs)}")
        return SelfSupervisedDataset(all_seqs)

    def _insert_whitespace(self, token_list):
        """
        Return the sequence of tokens with whitespace after each char
        """
        return " ".join(token_list)
        

class MAADataset(object):
    def __init__(self, 
                 config, 
                 logger, 
                 seed, 
                 seq_dir,
                 tokenizer_name, 
                 vocab_dir,
                 token_length_list, 
                 seq_name, 
                 max_len=None, 
                 test_split=0.1):
        self.config = config
        self.seq_dir = seq_dir
        self.seq_name = seq_name
        self.logger = logger
        self.seed = seed
        self.test_split = test_split

        self.seq_list = self._load_seq()

        self.logger.info('Start creating tokenizer...')
        self.tokenizer = get_tokenizer(tokenizer_name=tokenizer_name,
                                       add_hyphen=False,
                                       logger=self.logger,
                                       vocab_dir=vocab_dir,
                                       token_length_list=token_length_list)
        self.split_fun = self.tokenizer.split

        if max_len is None:
            self.max_len = max([len(self.split_fun(s)) for s in self.seq_list])
        else:
            self.max_len = max_len
        max_len_rounded = min_power_greater_than(self.max_len, base=2)

        self.bert_tokenizer = self.tokenizer.get_bert_tokenizer(max_len=max_len_rounded)
        self.bert_tokenizer.save_pretrained(config._save_dir)

    def get_token_list(self):
        return self.tokenizer.token_with_special_list

    def get_vocab_size(self):
        return len(self.tokenizer.token2index_dict)

    def get_pad_token_id(self):
        return self.tokenizer.token2index_dict[self.tokenizer.PAD]

    def get_tokenizer(self):
        return self.bert_tokenizer

    def _load_seq(self):
        seq_df = pd.read_csv(self.seq_dir)
        seq_list = list(seq_df[self.seq_name])
        self.logger.info(f'Load {len(seq_list)} form {self.seq_name}.')
        return seq_list

    def _split(self):
        train, test = train_test_split(self.seq_list, test_size=self.test_split, random_state=self.seed)
        return train, test

    def get_dataset(self):
        self_supvervised_dataset = SelfSupervisedDataset(seqs=self.seq_list,
                                                         split_fun=self.split_fun,
                                                         tokenizer=self.bert_tokenizer,
                                                         max_len=self.max_len,
                                                         logger=self.logger,
                                                         round_len=True)
        return self_supvervised_dataset
