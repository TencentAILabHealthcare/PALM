# -*- coding: utf-8 -*-

import torch
import numpy as np
import pandas as pd
from os.path import join, exists
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from bert_data_prepare.tokenizer import get_tokenizer
from bert_data_prepare.utility import is_valid_aaseq


class Seq2SeqDataset(Dataset):
    def __init__(self, antibody_seqs,
                       antigen_seqs,
                       antibody_split_fun,
                       antigen_split_fun,
                       antibody_tokenizer,
                       antigen_tokenizer,
                       encoder_input,
                       antibody_max_len,
                       antigen_max_len,
                       logger):
        self.antibody_seqs = antibody_seqs
        self.antigen_seqs = antigen_seqs
        self.antibody_split_fun = antibody_split_fun
        self.antigen_split_fun = antigen_split_fun
        self.antibody_tokenizer = antibody_tokenizer
        self.antigen_tokenizer = antigen_tokenizer
        self.encoder_input = encoder_input
        self.antibody_max_len = antibody_max_len
        self.antigen_max_len = antigen_max_len
        self.logger = logger
        self._has_logged_example = False

        self.logger.info(f"The input to the encoder is {encoder_input}")

    def __len__(self):
        return len(self.antibody_seqs)
        
    def __getitem__(self, i):
        antibody, antigen = self.antibody_seqs[i], self.antigen_seqs[i]

        input_data = {}
        antibody_tensor = self.antibody_tokenizer(self._insert_whitespace(self.antibody_split_fun(antibody)),
                                                padding="max_length",
                                                max_length=self.antibody_max_len,
                                                truncation=True)
        antigen_tensor = self.antigen_tokenizer(self._insert_whitespace(self.antigen_split_fun(antigen)),
                                                  padding="max_length",
                                                  max_length=self.antigen_max_len,
                                                  truncation=True)

        # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`. 
        # We have to make sure that the PAD token is ignored
        if self.encoder_input == 'antibody':
            input_data['input_ids'] = antibody_tensor['input_ids']
            input_data['attention_mask'] = antibody_tensor['attention_mask']
            input_data['labels'] = antigen_tensor.input_ids.copy()
            input_data['labels'] = [-100 if token == self.antigen_tokenizer.pad_token_id else token for token in input_data['labels']]
        
        elif self.encoder_input == 'antigen':
            input_data['input_ids'] = antigen_tensor['input_ids']
            input_data['attention_mask'] = antigen_tensor['attention_mask']
            input_data['labels'] = antibody_tensor['input_ids'].copy()
            input_data['labels'] = [-100 if token == self.antibody_tokenizer.pad_token_id else token for token in input_data['labels']]
        else:
            self.logger.info("Wrong encoder input!")
        
        input_data = {k: torch.tensor(v, dtype=torch.long) for k, v in input_data.items()}

        if not self._has_logged_example:
            self.logger.info(f"Example of tokenized antibody: {antibody} -> {antibody_tensor['input_ids']}")
            self.logger.info(f"Example of tokenized antigen: {antigen} -> {antigen_tensor['input_ids']}")
            self.logger.info(f"Example of input_ids {input_data['input_ids']}")
            self.logger.info(f"Example of label: {input_data['labels']}")
            self._has_logged_example = True

        return input_data

    def _insert_whitespace(self, token_list):
        """
        Return the sequence of tokens with whitespace after each char
        """
        return " ".join(token_list)


class AntibodyAntigenSeq2SeqDataset(object):
    def __init__(self, logger, 
                       seed,
                       data_dir, 
                    #    seq_dir,
                    #    neg_pair_save_dir, 
                    #    using_dataset, 
                       
                       antibody_vocab_dir,
                       antigen_vocab_dir,
                       antibody_tokenizer_dir,
                       antigen_tokenizer_dir,
                       antibody_tokenizer_name='common',
                       antigen_tokenizer_name='common',
                       antibody_token_length_list='2,3',
                       antigen_token_length_list='2,3',
                       antibody_max_len=None,
                       antigen_max_len=None,

                       valid_split=0.05,
                       antibody_seq_name='antibody', 
                       antigen_seq_name='heavy',
                       encoder_input='antibody',
                       shuffle=True):
        self.logger = logger
        self.seed = seed
        self.data_dir = data_dir
        # self.seq_dir = seq_dir
        # self.neg_pair_save_dir = neg_pair_save_dir
        # self.using_dataset = list(using_dataset.split(','))
        self.antibody_seq_name = antibody_seq_name
        self.antigen_seq_name = antigen_seq_name
        self.valid_split = valid_split
        self.encoder_input = encoder_input

        self.shuffle = shuffle
        self.antibody_max_len = antibody_max_len
        self.antigen_max_len = antigen_max_len
        self.rng = np.random.default_rng(seed=self.seed)

        self.pair_df = self._create_pair()
        train_pair_df, valid_pair_df, test_pair_df = self._split_dataset()
        self.valid_pair_df = valid_pair_df
        self.test_pair_df = test_pair_df

        self.logger.info(f'Creating {antibody_seq_name} tokenizer...')
        self.AntibodyTokenizer = get_tokenizer(tokenizer_name=antibody_tokenizer_name,
                                              add_hyphen=False,
                                              logger=self.logger,
                                              vocab_dir=antibody_vocab_dir,
                                              token_length_list=antibody_token_length_list)
        self.antibody_tokenizer = self.AntibodyTokenizer.get_bert_tokenizer(
            max_len=self.antibody_max_len, 
            tokenizer_dir=antibody_tokenizer_dir)

        self.logger.info(f'Creating {antigen_seq_name} tokenizer...')
        self.AntigenTokenizer = get_tokenizer(tokenizer_name=antigen_tokenizer_name,
                                               add_hyphen=False,
                                               logger=self.logger,
                                               vocab_dir=antigen_vocab_dir,
                                               token_length_list=antigen_token_length_list)
        self.antigen_tokenizer = self.AntigenTokenizer.get_bert_tokenizer(
            max_len=self.antigen_max_len,
            tokenizer_dir=antigen_tokenizer_dir)

        self.train_dataset = self._get_dataset(pair_df=train_pair_df)
        self.valid_dataset = self._get_dataset(pair_df=valid_pair_df)
        self.test_dataset = self._get_dataset(pair_df=test_pair_df)

    def get_antibody_split_fn(self):
        return self.AntibodyTokenizer.split

    def get_antigen_split_fn(self):
        return self.AntigenTokenizer.split

    def get_valid_pair_df(self):
        return self.valid_pair_df

    def get_test_pair_df(self):
        return self.test_pair_df

    def get_train_dataset(self):
        return self.train_dataset

    def get_valid_dataset(self):
        return self.valid_dataset

    def get_test_dataset(self):
        return self.test_dataset

    def get_antibody_tokenizer(self):
        return self.antibody_tokenizer

    def get_antigen_tokenizer(self):
        return self.antigen_tokenizer

    def _get_dataset(self, pair_df):
        er_dataset = Seq2SeqDataset(antibody_seqs=list(pair_df[self.antibody_seq_name]),
                                    antigen_seqs=list(pair_df[self.antigen_seq_name]),
                                    antibody_split_fun=self.AntibodyTokenizer.split,
                                    antigen_split_fun=self.AntigenTokenizer.split,
                                    antibody_tokenizer=self.antibody_tokenizer,
                                    antigen_tokenizer=self.antigen_tokenizer,
                                    encoder_input=self.encoder_input,
                                    antibody_max_len=self.antibody_max_len,
                                    antigen_max_len=self.antigen_max_len,
                                    logger=self.logger)
        return er_dataset

    def _split_dataset(self):
        train_pair_df , test_pair_df = train_test_split(self.pair_df , test_size=self.valid_split * 2, random_state=self.seed)
        valid_pair_df , test_pair_df = train_test_split(test_pair_df , test_size=0.5, random_state=self.seed)
        self.logger.info(f"{len(train_pair_df)} train and {len(valid_pair_df)} valid and {len(test_pair_df)} test.")

        return train_pair_df, valid_pair_df, test_pair_df

    def _create_pair(self):
        df = pd.read_csv(self.data_dir)
        return df

 







from base import BaseDataLoader
from transformers import AutoTokenizer








class Antibody_Antigen_Dataset_AbDab(BaseDataLoader):
    def __init__(self, logger, 
                       seed,
                       batch_size,
                       validation_split,
                       test_split,
                       num_workers,
                       data_dir, 
                       antibody_vocab_dir,
                       antibody_tokenizer_dir,
                       tokenizer_name='common',
                       #receptor_tokenizer_name='common',
                       token_length_list='2,3',
                       #receptor_token_length_list='2,3',
                       antigen_seq_name='antigen',
                       heavy_seq_name='Heavy',
                       light_seq_name='Light',
                       label_name='Label',
                       #receptor_seq_name='beta',
                       test_antibodys=100,
                       #neg_ratio=1.0,
                       shuffle=True,
                       antigen_max_len=None,
                       heavy_max_len=None,
                       light_max_len=None,):
        self.logger = logger
        self.seed = seed
        self.data_dir = data_dir
        self.heavy_seq_name = heavy_seq_name
        self.light_seq_name = light_seq_name
        self.antigen_seq_name = antigen_seq_name
        self.label_name = label_name

        self.test_antibodys = test_antibodys
        self.shuffle = shuffle
        self.heavy_max_len = heavy_max_len
        self.light_max_len = light_max_len
        self.antigen_max_len = antigen_max_len

        self.rng = np.random.default_rng(seed=self.seed)

        self.pair_df = self._create_pair()

        
        self.HeavyTokenizer = get_tokenizer(tokenizer_name=tokenizer_name,
                                              add_hyphen=False,
                                              logger=self.logger,
                                              vocab_dir=antibody_vocab_dir,
                                              token_length_list=token_length_list)
        self.heavy_tokenizer = self.HeavyTokenizer.get_bert_tokenizer(
            max_len=self.heavy_max_len, 
            tokenizer_dir=antibody_tokenizer_dir)



        self.LightTokenizer = get_tokenizer(tokenizer_name=tokenizer_name,
                                              add_hyphen=False,
                                              logger=self.logger,
                                              vocab_dir=antibody_vocab_dir,
                                              token_length_list=token_length_list)
        self.light_tokenizer = self.LightTokenizer.get_bert_tokenizer(
            max_len=self.light_max_len, 
            tokenizer_dir=antibody_tokenizer_dir)

        
        self.AntigenTokenizer = get_tokenizer(tokenizer_name=tokenizer_name,
                                               add_hyphen=False,
                                               logger=self.logger,
                                               vocab_dir=antibody_vocab_dir,
                                               token_length_list=token_length_list)
        self.antigen_tokenizer = self.AntigenTokenizer.get_bert_tokenizer(
            max_len=self.antigen_max_len,
            tokenizer_dir=antibody_tokenizer_dir)


        esm_dir = 'facebook/esm2_t30_150M_UR50D'
        
        self.antigen_tokenizer = AutoTokenizer.from_pretrained(esm_dir,cache_dir = "../esm2/esm2_150m/",max_len = self.antigen_max_len)
        
        dataset = self._get_dataset(pair_df=self.pair_df)
        super().__init__(dataset, batch_size, seed, shuffle, validation_split, test_split,
                         num_workers)


    def get_heavy_tokenizer(self):
        return self.heavy_tokenizer

    def get_light_tokenizer(self):
        return self.light_tokenizer

    def get_antibody_tokenizer(self):
        return self.heavy_tokenizer

    def get_antigen_tokenizer(self):
        return self.antigen_tokenizer

    def get_test_dataloader(self):
        return self.test_dataloader

    def _get_dataset(self, pair_df):
        abag_dataset = AbAGDataset_CovAbDab(
                                        heavy_seqs = list(pair_df[self.heavy_seq_name]),
                                        light_seqs = list(pair_df[self.light_seq_name]),
                                        antigen_seqs = list(pair_df[self.antigen_seq_name]),
                                        # labels = list(pair_df[self.label_name]),
                                        antibody_split_fun = self.HeavyTokenizer.split,
                                        antigen_split_fun = self.AntigenTokenizer.split,
                                        antibody_tokenizer = self.heavy_tokenizer,
                                        antigen_tokenizer = self.antigen_tokenizer,
                                        antibody_max_len = self.heavy_max_len,
                                        antigen_max_len = self.antigen_max_len,
                                        logger = self.logger
                               )
        return abag_dataset

    def _split_dataset(self):
        # if exists(join(self.neg_pair_save_dir, 'unseen_antibodys-seed-'+str(self.seed)+'.csv')):
        #     test_pair_df = pd.read_csv(join(self.neg_pair_save_dir, 'unseen_antibodys-seed-'+str(self.seed)+'.csv'))
        #     self.logger.info(f'Loading created unseen antibodys for test with shape {test_pair_df.shape}')
        
        antibody_list = list(set(self.pair_df['antibody']))
        selected_antibody_index_list = self.rng.integers(len(antibody_list), size=self.test_antibodys)
        self.logger.info(f'Select {self.test_antibodys} from {len(antibody_list)} antibody')
        selected_antibodys = [antibody_list[i] for i in selected_antibody_index_list]
        test_pair_df = self.pair_df[self.pair_df['antibody'].isin(selected_antibodys)]
        #test_pair_df.to_csv(join(self.neg_pair_save_dir, 'unseen_antibodys-seed-'+str(self.seed)+'.csv'), index=False)

        selected_antibodys = list(set(test_pair_df['antibody']))
        train_valid_pair_df = self.pair_df[~self.pair_df['antibody'].isin(selected_antibodys)]
            
        self.logger.info(f'{len(train_valid_pair_df)} pairs for train and valid and {len(test_pair_df)} pairs for test.')

        return train_valid_pair_df, test_pair_df

    def _create_pair(self):
        pair_df = pd.read_csv(self.data_dir)

        if self.shuffle:
            pair_df = pair_df.sample(frac=1).reset_index(drop=True)
            self.logger.info("Shuffling dataset")
        self.logger.info(f"There are {len(pair_df)} samples")

        return pair_df

    def _load_seq_pairs(self):
        self.logger.info(f'Loading from {self.using_dataset}...')
        self.logger.info(f'Loading {self.antibody_seq_name} and {self.receptor_seq_name}')
        column_map_dict = {'alpha': 'cdr3a', 'beta': 'cdr3b', 'antibody': 'antibody'}
        keep_columns = [column_map_dict[c] for c in [self.antibody_seq_name, self.receptor_seq_name]]
        
        df_list = []
        for dataset in self.using_dataset:
            df = pd.read_csv(join(self.data_dir, dataset, 'full.csv'))
            df = df[keep_columns]
            df = df[(df[keep_columns[0]].map(is_valid_aaseq)) & (df[keep_columns[1]].map(is_valid_aaseq))]
            self.logger.info(f'Loading {len(df)} pairs from {dataset}')
            df_list.append(df[keep_columns])
        df = pd.concat(df_list)
        self.logger.info(f'Current data shape {df.shape}')
        df_filter = df.dropna()
        df_filter = df_filter.drop_duplicates()
        self.logger.info(f'After dropping na and duplicates, current data shape {df_filter.shape}')

        column_rename_dict = {column_map_dict[c]: c for c in [self.antibody_seq_name, self.receptor_seq_name]}
        df_filter.rename(columns=column_rename_dict, inplace=True)

        df_filter['label'] = [1] * len(df_filter)
        df_filter.to_csv(join(self.neg_pair_save_dir, 'pos_pair.csv'), index=False)

        return df_filter

class AbAGDataset_CovAbDab(Dataset):
    def __init__(self, heavy_seqs,
                       light_seqs,
                       antigen_seqs,
                    #    labels,
                       antibody_split_fun,
                       antigen_split_fun,
                       antibody_tokenizer,
                       antigen_tokenizer,
                       antibody_max_len,
                       antigen_max_len,
                       logger):
        self.heavy_seqs = heavy_seqs
        self.light_seqs = light_seqs
        self.antigen_seqs = antigen_seqs
        # self.labels = labels
        self.antibody_split_fun = antibody_split_fun
        self.antigen_split_fun = antigen_split_fun
        self.antibody_tokenizer = antibody_tokenizer
        self.antigen_tokenizer = antigen_tokenizer
        self.antibody_max_len = antibody_max_len
        self.antigen_max_len = antigen_max_len
        self.logger = logger
        self._has_logged_example = False

    def __len__(self):
        return len(self.heavy_seqs)
        
    def __getitem__(self, i):
        heavy,light,antigen = self.heavy_seqs[i], self.light_seqs[i] , self.antigen_seqs[i]
        #label = self.labels[i]
        heavy_tensor = self.antibody_tokenizer(self._insert_whitespace(self.antibody_split_fun(heavy)),
                                                padding="max_length",
                                                max_length=self.antibody_max_len,
                                                truncation=True,
                                                return_tensors="pt")
        light_tensor = self.antibody_tokenizer(self._insert_whitespace(self.antibody_split_fun(light)),
                                                padding="max_length",
                                                max_length=self.antibody_max_len,
                                                truncation=True,
                                                return_tensors="pt")

  
        antigen_tensor = self.antigen_tokenizer(antigen,
                                                  padding="max_length",
                                                  max_length=self.antigen_max_len,
                                                  truncation=True,
                                                  return_tensors="pt")

        

        # label_tensor = torch.FloatTensor(np.atleast_1d(label))


        heavy_tensor = {k: torch.squeeze(v) for k, v in heavy_tensor.items()}
        light_tensor = {k: torch.squeeze(v) for k, v in light_tensor.items()}
        antigen_tensor = {k: torch.squeeze(v) for k,v in antigen_tensor.items()}
        return heavy_tensor, light_tensor, antigen_tensor#, label_tensor




    def _insert_whitespace(self, token_list):
        """
        Return the sequence of tokens with whitespace after each char
        """
        return " ".join(token_list)




class ABAGDataset_4_input(Dataset):
    def __init__(self, antibody_a_seqs,
                       antibody_b_seqs,
                       antibody_c_seqs,
                       receptor_seqs,
                    #    labels,
                       antibody_split_fun,
                       receptor_split_fun,
                       antibody_tokenizer,
                       receptor_tokenizer,
                       antibody_max_len,
                       receptor_max_len,
                       logger):
        self.antibody_a_seqs = antibody_a_seqs
        self.antibody_b_seqs = antibody_b_seqs
        self.antibody_c_seqs = antibody_c_seqs
        self.receptor_seqs = receptor_seqs
        # self.labels = labels
        self.antibody_split_fun = antibody_split_fun
        self.receptor_split_fun = receptor_split_fun
        self.antibody_tokenizer = antibody_tokenizer
        self.receptor_tokenizer = receptor_tokenizer
        self.antibody_max_len = antibody_max_len
        self.receptor_max_len = receptor_max_len
        self.logger = logger
        self._has_logged_example = False

    def __len__(self):
        return len(self.antibody_a_seqs)
        
    def __getitem__(self, i):
        antibody_a,antibody_b,antibody_c, receptor = self.antibody_a_seqs[i], self.antibody_b_seqs[i] ,self.antibody_c_seqs[i], self.receptor_seqs[i]
        #label = self.labels[i]
        antibody_a_tensor = self.antibody_tokenizer(self._insert_whitespace(self.antibody_split_fun(antibody_a)),
                                                padding="max_length",
                                                max_length=self.antibody_max_len,
                                                truncation=True,
                                                return_tensors="pt",
                                                )
        antibody_b_tensor = self.antibody_tokenizer(self._insert_whitespace(self.antibody_split_fun(antibody_b)),
                                                padding="max_length",
                                                max_length=self.antibody_max_len,
                                                truncation=True,
                                                return_tensors="pt")

        antibody_c_tensor = self.antibody_tokenizer(self._insert_whitespace(self.antibody_split_fun(antibody_c)),
                                                padding="max_length",
                                                max_length=self.antibody_max_len,
                                                truncation=True,
                                                return_tensors="pt")
        
        receptor_tensor = self.receptor_tokenizer(self._insert_whitespace(self.receptor_split_fun(receptor)),
                                                  padding="max_length",
                                                  max_length=self.receptor_max_len,
                                                  truncation=True,
                                                  return_tensors="pt")


        # label_tensor = torch.FloatTensor(np.atleast_1d(label))

        antibody_a_tensor = {k: torch.squeeze(v) for k, v in antibody_a_tensor.items()}
        antibody_b_tensor = {k: torch.squeeze(v) for k, v in antibody_b_tensor.items()}
        antibody_c_tensor = {k: torch.squeeze(v) for k, v in antibody_c_tensor.items()}
        receptor_tensor = {k: torch.squeeze(v) for k,v in receptor_tensor.items()}

        # if not self._has_logged_example:
        #     self.logger.info(f"Example of tokenized antibody: {antibody} -> {antibody_tensor}")
        #     self.logger.info(f"Example of tokenized receptor: {receptor} -> {receptor_tensor}")
        #     self.logger.info(f"Example of label: {label} -> {label_tensor}")
        #     self._has_logged_example = True

        return antibody_a_tensor , antibody_b_tensor,antibody_c_tensor, receptor_tensor#, label_tensor

    def _insert_whitespace(self, token_list):
        """
        Return the sequence of tokens with whitespace after each char
        """
        return " ".join(token_list)


class Antibody_Antigen_Dataset_4_input(BaseDataLoader):
    def __init__(self, logger, 
                       seed,
                       batch_size,
                       validation_split,
                       test_split,
                       num_workers,
                       data_dir, 
                       antibody_vocab_dir,
                       antibody_tokenizer_dir,
                       antibody_tokenizer_name='common',
                       antibody_token_length_list='2,3',
                       antibody_seq_a_name = "cdr1",
                       antibody_seq_b_name = "cdr2",
                       antibody_seq_c_name = "cdr3",
                       receptor_seq_name = "heavy",
                       test_antibodys=100,
                       shuffle=True,
                       cdr_max_len=None,
                       ab_max_len=None):
        self.logger = logger
        self.seed = seed
        self.data_dir = data_dir
        
        self.antibody_seq_a_name = antibody_seq_a_name
        self.antibody_seq_b_name = antibody_seq_b_name

        self.antibody_seq_c_name = antibody_seq_c_name
        self.receptor_seq_name = receptor_seq_name



        self.test_antibodys = test_antibodys
        #self.neg_ratio = neg_ratio
        self.shuffle = shuffle
        self.cdr_max_len = cdr_max_len
        self.ab_max_len = ab_max_len
        self.rng = np.random.default_rng(seed=self.seed)

        self.pair_df = self._create_pair()
        #train_valid_pair_df, test_pair_df = self._split_dataset()
        
        self.logger.info(f'Creating {antibody_seq_a_name} tokenizer...')
        self.AntibodyTokenizer_a = get_tokenizer(tokenizer_name=antibody_tokenizer_name,
                                              add_hyphen=False,
                                              logger=self.logger,
                                              vocab_dir=antibody_vocab_dir,
                                              token_length_list=antibody_token_length_list)
        self.antibody_tokenizer_a = self.AntibodyTokenizer_a.get_bert_tokenizer(
            max_len=cdr_max_len,   
            tokenizer_dir=antibody_tokenizer_dir)



        #self.logger.info(f'Creating {antibody_seq_name} tokenizer...')
        self.AntibodyTokenizer_b = get_tokenizer(tokenizer_name=antibody_tokenizer_name,
                                              add_hyphen=False,
                                              logger=self.logger,
                                              vocab_dir=antibody_vocab_dir,
                                              token_length_list=antibody_token_length_list)
        self.antibody_tokenizer_b = self.AntibodyTokenizer_b.get_bert_tokenizer(
            max_len=self.cdr_max_len, 
            tokenizer_dir=antibody_tokenizer_dir)


        self.AntibodyTokenizer_c = get_tokenizer(tokenizer_name=antibody_tokenizer_name,
                                              add_hyphen=False,
                                              logger=self.logger,
                                              vocab_dir=antibody_vocab_dir,
                                              token_length_list=antibody_token_length_list)
        self.antibody_tokenizer_c = self.AntibodyTokenizer_c.get_bert_tokenizer(
            max_len=self.cdr_max_len, 
            tokenizer_dir=antibody_tokenizer_dir)
        
        #self.logger.info(f'Creating {receptor_seq_name} tokenizer...')
        self.antigen_tokenizer = get_tokenizer(tokenizer_name=antibody_tokenizer_name,
                                               add_hyphen=False,
                                               logger=self.logger,
                                               vocab_dir=antibody_vocab_dir,
                                               token_length_list=antibody_token_length_list)
        self.receptor_tokenizer = self.antigen_tokenizer.get_bert_tokenizer(
            max_len=self.ab_max_len,
            tokenizer_dir=antibody_tokenizer_dir)
        
        dataset = self._get_dataset(pair_df=self.pair_df)
        super().__init__(dataset, batch_size, seed, shuffle, validation_split, test_split,
                         num_workers)

    def get_antibody_tokenizer(self):
        return self.antibody_tokenizer_a

    def get_antigen_tokenizer(self):
        return self.receptor_tokenizer

    def get_test_dataloader(self):
        return self.test_dataloader

    def _get_dataset(self, pair_df):
        er_dataset = ABAGDataset_4_input(antibody_a_seqs=list(pair_df[self.antibody_seq_a_name]),
                               antibody_b_seqs=list(pair_df[self.antibody_seq_b_name]), 
                               antibody_c_seqs=list(pair_df[self.antibody_seq_c_name]),
                               receptor_seqs=list(pair_df[self.receptor_seq_name]),
                            #    labels=list(pair_df['affinity']), # ll_cdr
                               antibody_split_fun=self.AntibodyTokenizer_a.split,
                               receptor_split_fun=self.antigen_tokenizer.split,
                               antibody_tokenizer=self.antibody_tokenizer_a,
                               receptor_tokenizer=self.receptor_tokenizer,
                               antibody_max_len=self.cdr_max_len,
                               receptor_max_len=self.ab_max_len,
                               logger=self.logger)
        return er_dataset

    def _split_dataset(self):
        # if exists(join(self.neg_pair_save_dir, 'unseen_antibodys-seed-'+str(self.seed)+'.csv')):
        #     test_pair_df = pd.read_csv(join(self.neg_pair_save_dir, 'unseen_antibodys-seed-'+str(self.seed)+'.csv'))
        #     self.logger.info(f'Loading created unseen antibodys for test with shape {test_pair_df.shape}')
        
        antibody_list = list(set(self.pair_df['antibody']))
        selected_antibody_index_list = self.rng.integers(len(antibody_list), size=self.test_antibodys)
        self.logger.info(f'Select {self.test_antibodys} from {len(antibody_list)} antibody')
        selected_antibodys = [antibody_list[i] for i in selected_antibody_index_list]
        test_pair_df = self.pair_df[self.pair_df['antibody'].isin(selected_antibodys)]
        #test_pair_df.to_csv(join(self.neg_pair_save_dir, 'unseen_antibodys-seed-'+str(self.seed)+'.csv'), index=False)

        selected_antibodys = list(set(test_pair_df['antibody']))
        train_valid_pair_df = self.pair_df[~self.pair_df['antibody'].isin(selected_antibodys)]
            
        self.logger.info(f'{len(train_valid_pair_df)} pairs for train and valid and {len(test_pair_df)} pairs for test.')

        return train_valid_pair_df, test_pair_df

    def _create_pair(self):
        pair_df = pd.read_csv(self.data_dir)

        if self.shuffle:
            pair_df = pair_df.sample(frac=1).reset_index(drop=True)
            self.logger.info("Shuffling dataset")
        self.logger.info(f"There are {len(pair_df)} samples")

        return pair_df

    def _load_seq_pairs(self):
        self.logger.info(f'Loading from {self.using_dataset}...')
        self.logger.info(f'Loading {self.antibody_seq_name} and {self.receptor_seq_name}')
        column_map_dict = {'alpha': 'cdr3a', 'beta': 'cdr3b', 'antibody': 'antibody'}
        keep_columns = [column_map_dict[c] for c in [self.antibody_seq_name, self.receptor_seq_name]]
        
        df_list = []
        for dataset in self.using_dataset:
            df = pd.read_csv(join(self.data_dir, dataset, 'full.csv'))
            df = df[keep_columns]
            df = df[(df[keep_columns[0]].map(is_valid_aaseq)) & (df[keep_columns[1]].map(is_valid_aaseq))]
            self.logger.info(f'Loading {len(df)} pairs from {dataset}')
            df_list.append(df[keep_columns])
        df = pd.concat(df_list)
        self.logger.info(f'Current data shape {df.shape}')
        df_filter = df.dropna()
        df_filter = df_filter.drop_duplicates()
        self.logger.info(f'After dropping na and duplicates, current data shape {df_filter.shape}')

        column_rename_dict = {column_map_dict[c]: c for c in [self.antibody_seq_name, self.receptor_seq_name]}
        df_filter.rename(columns=column_rename_dict, inplace=True)

        df_filter['label'] = [1] * len(df_filter)
        df_filter.to_csv(join(self.neg_pair_save_dir, 'pos_pair.csv'), index=False)

        return df_filter








