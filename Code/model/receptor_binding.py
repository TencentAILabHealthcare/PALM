# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from transformers import BertModel
from base import BaseModel

class BERTBinding(BaseModel):
    def __init__(self, ReceptorBert_dir, emb_dim, dropout):
        super().__init__()
        self.ReceptorBert = BertModel.from_pretrained(ReceptorBert_dir)
        self.binding_predict = nn.Sequential(
            nn.Linear(in_features=emb_dim, out_features=emb_dim),
            nn.Tanh(),
            nn.Dropout(p=dropout),

            nn.Linear(in_features=emb_dim, out_features=1)
        )
        # self.binding_predict = nn.Sequential(
        #     nn.Linear(in_features=32, out_features=32*2),
        #     nn.Tanh(),
        #     nn.Linear(in_features=32*2, out_features=32*4),
        #     nn.Tanh(),
        #     nn.Linear(in_features=32*4, out_features=32*2),
        #     nn.Tanh(),
        #     nn.Linear(in_features=32*2, out_features=32),
        #     nn.Tanh(),
        #     nn.Linear(in_features=32, out_features=1)
        # )


    def forward(self, epitope, receptor):
        # shape: [batch_size, seq_length, emb_dim]
        receptor_encoded = self.ReceptorBert(**receptor).last_hidden_state

        '''
        Using the cls (classification) token as the input to get the score which is borrowed
        from huggingface NextSentencePrediciton implementation
        https://github.com/huggingface/transformers/issues/7540
        https://huggingface.co/transformers/v2.0.0/_modules/transformers/modeling_bert.html
        '''
        # shape: [batch_size, emb_dim]
        receptor_cls = receptor_encoded[:, 0, :]
        # receptor_cls = torch.squeeze(torch.sum(receptor_encoded, dim=1))
        output = self.binding_predict(receptor_cls)
        # output = self.binding_predict(receptor['input_ids'].type(torch.float))

        return output