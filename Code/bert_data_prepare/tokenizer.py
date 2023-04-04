# -*- coding: utf-8 -*-

import tempfile
import pandas as pd
import numpy as np
from abc import abstractmethod
from os.path import join
from transformers import BertTokenizer , AutoTokenizer

class BaseTokenizer(object):
    def __init__(self, logger, tokenizer_name, add_hyphen, vocab_dir, token_length_list=[2,3]):
        self.PAD = "$"
        self.MASK = "."
        self.UNK = "?"
        self.SEP = "|"
        self.CLS = "*"

        self.logger = logger
        self.logger.info(f'Using {tokenizer_name} tokenizer.')
        self.vocab_dir = vocab_dir
        self.token_length_list = token_length_list
        self.logger.info(f"Using token with length {self.token_length_list}")
        self.add_hyphen = add_hyphen

        self.vocab_df, self.vocab_freq_dict = self._load_vocab()
        self.token_with_special_list, self.token2index_dict = self._get_vocab_dict(add_hyphen=add_hyphen)

    def _load_vocab(self):
        df = pd.read_csv(self.vocab_dir, na_filter=False) # Since there are 'NA' token
        self.logger.info('{} tokens in the vocab'.format(len(df)))
        vocab_dict = {row['token']: row['freq_z_normalized'] for _, row in df.iterrows()}
        return df, vocab_dict

    def _get_vocab_dict(self, add_hyphen=False):
        amino_acids_list = [c for c in 'ACDEFGHIKLMNPQRSTVWY']
        special_tokesn = [self.PAD, self.MASK, self.UNK, self.SEP, self.CLS]

        df_sorted = self.vocab_df.sort_values(by=['freq_z_normalized'], ascending=False)
        token_list = list(df_sorted['token'])

        if add_hyphen:
            self.logger.info('Add hyphen - in the tokenizer')
            token_with_special_list = ['-'] + amino_acids_list + special_tokesn + token_list
        else:
            token_with_special_list = amino_acids_list + special_tokesn + token_list
        token2index_dict = {t: i for i, t in enumerate(token_with_special_list)}

        return token_with_special_list, token2index_dict

    @abstractmethod
    def split(self, seq):
        raise NotImplementedError

    def get_bert_tokenizer(self, max_len=64, tokenizer_dir=None):
        if tokenizer_dir is not None:
            self.logger.info('Loading pre-trained tokenizer...')
            tok = BertTokenizer.from_pretrained(
                tokenizer_dir,
                do_lower_case=False,
                do_basic_tokenize=True,
                tokenize_chinese_chars=False,
                pad_token=self.PAD,
                mask_token=self.MASK,
                unk_token=self.UNK,
                sep_token=self.SEP,
                cls_token=self.CLS,
                padding_side="right")
            return tok

        with tempfile.TemporaryDirectory() as tempdir:
            vocab_fname = self._write_vocab(self.token2index_dict, join(tempdir, "vocab.txt"))
            tok = BertTokenizer(
                vocab_fname,
                do_lower_case=False,
                do_basic_tokenize=True,
                tokenize_chinese_chars=False,
                pad_token=self.PAD,
                mask_token=self.MASK,
                unk_token=self.UNK,
                sep_token=self.SEP,
                cls_token=self.CLS,
                model_max_len=max_len,
                padding_side="right")
        return tok

    def _write_vocab(self, vocab, fname):
        """
        Write the vocabulary to the fname, one entry per line
        Mostly for compatibility with transformer AutoTokenizer
        """
        with open(fname, "w") as sink:
            for v in vocab:
                sink.write(v + "\n")
        return fname


class FMFMTokenizer(BaseTokenizer):
    '''FMFM: Forward Maximum Frequency Matching'''
    def __init__(self, logger, add_hyphen, vocab_dir, token_length_list=[2,3], tokenizer_name='FMFM'):
        super().__init__(logger, tokenizer_name, add_hyphen, vocab_dir, token_length_list)

    def split(self, seq):
        def split_fn(seq):
            tokens = []
            i = 0
            seq_len = len(seq)
            while i < seq_len:
                if i > seq_len - 2:
                    tokens.append(seq[i])
                    i += 1
                else:
                    temp_token_list = list(set([seq[i: i+token_len] for token_len in self.token_length_list]))
                    '''
                    add:
                    '''
                    for token in temp_token_list:
                        if not self.vocab_freq_dict.__contains__(token):
                            self.vocab_freq_dict[token] = 0
                    
                    temp_token_freq = [self.vocab_freq_dict[token] for token in temp_token_list]
                    if sum(temp_token_freq) == 0:
                        tokens.append(seq[i])
                        i += 1
                    else:
                        selected_token = temp_token_list[np.argmax(temp_token_freq)]
                        tokens.append(selected_token)
                        i += len(selected_token)
            return tokens
            
        if self.add_hyphen:
            seq1, seq2 = seq.split('-')
            tokens_seq1 = split_fn(seq1)
            tokens_seq2 = split_fn(seq2)
            tokens = tokens_seq1 + ['-'] + tokens_seq2
        else:
            tokens = split_fn(seq)

        return tokens


class FMFCMTokenizer(BaseTokenizer):
    '''FMFCM: Forward Maximum Frequency Continuous Matching'''
    def __init__(self, logger, add_hyphen, vocab_dir, token_length_list=[2,3], tokenizer_name='FMFCM'):
        super().__init__(logger, tokenizer_name, add_hyphen, vocab_dir, token_length_list)

    def split(self, seq):
        def split_fn(seq):
            tokens = []
            i = 0
            seq_len = len(seq)
            while i < seq_len:
                if i > seq_len - 2:
                    tokens.append(seq[i])
                    i += 1
                else:
                    temp_token_list = list(set([seq[i: i+token_len] for token_len in self.token_length_list]))
                    temp_token_freq = [self.vocab_freq_dict[token] for token in temp_token_list]
                    if sum(temp_token_freq) == 0:
                        tokens.append(seq[i])
                        i += 1
                    else:
                        selected_token = temp_token_list[np.argmax(temp_token_freq)]
                        tokens.append(selected_token)
                        i += 1
            return tokens
        
        if self.add_hyphen:
            seq1, seq2 = seq.split('-')
            tokens_seq1 = split_fn(seq1)
            tokens_seq2 = split_fn(seq2)
            tokens = tokens_seq1 + ['-'] + tokens_seq2
        else:
            tokens = split_fn(seq)

        return tokens


class CommonTokenizer(object):
    def __init__(self, logger, tokenizer_name='common', add_hyphen=False):
        self.PAD = "$"
        self.MASK = "."
        self.UNK = "?"
        self.SEP = "|"
        self.CLS = "*"

        self.logger = logger
        self.logger.info(f'Using {tokenizer_name} tokenizer.')

        self.token_with_special_list, self.token2index_dict = self._get_vocab_dict(add_hyphen)
    
    def _get_vocab_dict(self, add_hyphen=False):
        amino_acids_list = [c for c in 'ACDEFGHIKLMNPQRSTVWY']
        special_tokesn = [self.PAD, self.MASK, self.UNK, self.SEP, self.CLS]

        if add_hyphen:
            self.logger.info('Add hyphen - in the tokenizer')
            token_list = ['-'] + amino_acids_list + special_tokesn
        else:
            token_list = amino_acids_list + special_tokesn
        token2index_dict = {t: i for i, t in enumerate(token_list)}

        return token_list, token2index_dict

    def get_bert_tokenizer(self, max_len=64, tokenizer_dir=None):
        if tokenizer_dir is not None:
            self.logger.info('Loading pre-trained tokenizer...')
            tok = BertTokenizer.from_pretrained(
                tokenizer_dir,
                do_lower_case=False,
                do_basic_tokenize=True,
                tokenize_chinese_chars=False,
                pad_token=self.PAD,
                mask_token=self.MASK,
                unk_token=self.UNK,
                sep_token=self.SEP,
                cls_token=self.CLS,
                padding_side="right"
                )
            return tok

        with tempfile.TemporaryDirectory() as tempdir:
            vocab_fname = self._write_vocab(self.token2index_dict, join(tempdir, "vocab.txt"))
            tok = BertTokenizer(
                vocab_fname,
                do_lower_case=False,
                do_basic_tokenize=True,
                tokenize_chinese_chars=False,
                pad_token=self.PAD,
                mask_token=self.MASK,
                unk_token=self.UNK,
                sep_token=self.SEP,
                cls_token=self.CLS,
                model_max_len=max_len,
                padding_side="right"
                )
        return tok

    def split(self, seq):
        return list(seq)

    def _write_vocab(self, vocab, fname):
        """
        Write the vocabulary to the fname, one entry per line
        Mostly for compatibility with transformer AutoTokenizer
        """
        with open(fname, "w") as sink:
            for v in vocab:
                sink.write(v + "\n")
        return fname 

class TCRBertTokenizer():
    def get_bert_tokenizer(self, max_len=64, tokenizer_dir=None):
        return BertTokenizer.from_pretrained(tokenizer_dir, do_lower_case=False)
    
    def split(self, seq):
        return list(seq)


def get_tokenizer(tokenizer_name, add_hyphen, logger, vocab_dir, token_length_list=[2,3]):
    if tokenizer_name == 'common':
        MyTokenizer = CommonTokenizer(logger=logger, add_hyphen=add_hyphen)

    elif tokenizer_name == 'FMFM':
        MyTokenizer = FMFMTokenizer(logger=logger,
                                    add_hyphen=add_hyphen,
                                    token_length_list=[int(v) for v in token_length_list.split(',')],
                                    vocab_dir=vocab_dir)
    elif tokenizer_name == 'FMFCM':
        MyTokenizer = FMFCMTokenizer(logger=logger,
                                     add_hyphen=add_hyphen,
                                     token_length_list=[int(v) for v in token_length_list.split(',')],
                                     vocab_dir=vocab_dir)
    elif tokenizer_name == 'TCRBert':
        MyTokenizer = TCRBertTokenizer()
    return MyTokenizer