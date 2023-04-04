# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel,AutoModel,RoFormerModel



class BERTBinding_AbDab_cnn(nn.Module):
    def __init__(self, heavy_dir,light_dir, antigen_dir, emb_dim=256):
        super().__init__()
        self.HeavyModel = AutoModel.from_pretrained(heavy_dir, output_hidden_states=True, return_dict=True)
        self.LightModel = AutoModel.from_pretrained(light_dir, output_hidden_states=True, return_dict=True)
        self.AntigenModel = AutoModel.from_pretrained(antigen_dir, output_hidden_states=True, return_dict=True,cache_dir = "../cache")


        self.cnn1 = MF_CNN(in_channel= 140)
        self.cnn2 = MF_CNN(in_channel = 140)
        self.cnn3 = MF_CNN(in_channel = 300,hidden_size=76)#56)

        self.binding_predict = nn.Sequential(
            nn.Linear(in_features=128 * 3, out_features=emb_dim),
            nn.ReLU(),
            nn.Linear(in_features=emb_dim, out_features=emb_dim),
            nn.ReLU(),
            nn.Linear(in_features=emb_dim, out_features=1)
        )

    def forward(self, heavy, light, antigen):
        heavy_encoded = self.HeavyModel(**heavy).last_hidden_state
        light_encoded = self.LightModel(**light).last_hidden_state
        antigen_encoded = self.AntigenModel(**antigen).last_hidden_state

        heavy_cls = self.cnn1(heavy_encoded)
        light_cls = self.cnn2(light_encoded)
        antigen_cls = self.cnn3(antigen_encoded)
        
        concated_encoded = torch.concat((heavy_cls,light_cls,antigen_cls) , dim = 1)


        output = self.binding_predict(concated_encoded)

        return output


class BERTBinding_biomap_cnn(nn.Module):
    def __init__(self, heavy_dir,light_dir, antigen_dir, emb_dim=256):
        super().__init__()
        self.HeavyModel = AutoModel.from_pretrained(heavy_dir, output_hidden_states=True, return_dict=True)
        self.LightModel = AutoModel.from_pretrained(light_dir, output_hidden_states=True, return_dict=True)
        self.AntigenModel = AutoModel.from_pretrained(antigen_dir, output_hidden_states=True, return_dict=True,cache_dir = "../cache")


        self.cnn1 = MF_CNN(in_channel= 170)
        self.cnn2 = MF_CNN(in_channel = 170)
        self.cnn3 = MF_CNN(in_channel = 512,hidden_size=76)

        self.binding_predict = nn.Sequential(
            nn.Linear(in_features=128 * 3, out_features=emb_dim),
            nn.ReLU(),
            nn.Linear(in_features=emb_dim, out_features=emb_dim),
            nn.ReLU(),
            nn.Linear(in_features=emb_dim, out_features=1)
        )

    def forward(self, heavy, light, antigen):
        heavy_encoded = self.HeavyModel(**heavy).last_hidden_state
        light_encoded = self.LightModel(**light).last_hidden_state
        antigen_encoded = self.AntigenModel(**antigen).last_hidden_state

        heavy_cls = self.cnn1(heavy_encoded)
        light_cls = self.cnn2(light_encoded)
        antigen_cls = self.cnn3(antigen_encoded)
        
        concated_encoded = torch.concat((heavy_cls,light_cls,antigen_cls) , dim = 1)


        output = self.binding_predict(concated_encoded)

        return output






        

class BERTBinding_4_input_cnn(nn.Module):
    def __init__(self, PretrainModel_dir, emb_dim):
        super().__init__()
        self.CDRModel_a = AutoModel.from_pretrained(PretrainModel_dir)
        self.CDRModel_b = AutoModel.from_pretrained(PretrainModel_dir)
        self.CDRModel_c = AutoModel.from_pretrained(PretrainModel_dir)
        self.ABModel = AutoModel.from_pretrained(PretrainModel_dir,cache_dir = "../cache")


        self.cnn1 = MF_CNN(in_channel= 18)
        self.cnn2 = MF_CNN(in_channel = 18)
        self.cnn3 = MF_CNN(in_channel = 18)
        self.cnn4 = MF_CNN(in_channel = 120)

        self.binding_predict = nn.Sequential(
            nn.Linear(in_features=128 * 4, out_features=emb_dim),
            nn.ReLU(),
            nn.Linear(in_features=emb_dim, out_features=emb_dim),
            nn.ReLU(),
            nn.Linear(in_features=emb_dim, out_features=1)
        )

    def forward(self, antibody_a, antibody_b, antibody_c, receptor):
        antibody_a_encoded = self.CDRModel_a(**antibody_a).last_hidden_state
        antibody_b_encoded = self.CDRModel_b(**antibody_b).last_hidden_state
        antibody_c_encoded = self.CDRModel_c(**antibody_c).last_hidden_state
        receptor_encoded = self.ABModel(**receptor).last_hidden_state

        
        antibody_a_cls = self.cnn1(antibody_a_encoded)
        antibody_b_cls = self.cnn2(antibody_b_encoded)
        antibody_c_cls = self.cnn3(antibody_c_encoded)
        receptor_cls = self.cnn4(receptor_encoded)

        concated_encoded = torch.concat((antibody_a_cls,antibody_b_cls,antibody_c_cls , receptor_cls), dim=1)
        

        output = self.binding_predict(concated_encoded)

        return output








class MF_CNN(nn.Module):
    def __init__(self, in_channel=118,emb_size = 20,hidden_size = 92):#189):
        super(MF_CNN, self).__init__()
        
        # self.emb = nn.Embedding(emb_size,128)  # 20*128
        self.conv1 = cnn_liu(in_channel = in_channel,hidden_channel = 64)   # 118*64
        self.conv2 = cnn_liu(in_channel = 64,hidden_channel = 32) # 64*32

        self.conv3 = cnn_liu(in_channel = 32,hidden_channel = 32)

        self.fc1 = nn.Linear(32*hidden_size , 128) # 32*29*512
        self.fc2 = nn.Linear(128 , 128)

        self.fc3 = nn.Linear(128 , 128)

    def forward(self, x):
        #x = x
        # x = self.emb(x)
        
        x = self.conv1(x)
        
        x = self.conv2(x)

        x = self.conv3(x)
        
        x = x.view(x.shape[0] ,-1)
        
        x = nn.ReLU()(self.fc1(x))
        sk = x
        x = self.fc2(x)

        x = self.fc3(x)
        return x +sk





    
    
class cnn_liu(nn.Module):
    def __init__(self, in_channel=2, hidden_channel=2, out_channel=2):
        super(cnn_liu, self).__init__()
        
        self.cnn = nn.Conv1d(in_channel , hidden_channel , kernel_size = 5 , stride = 1) # bs * 64*60
        self.max_pool = nn.MaxPool1d(kernel_size = 2 , stride=2)# bs * 32*30
                               
        self.relu = nn.ReLU()
    
    def forward(self, x):
        
        #x = self.emb(x)
        x = self.cnn(x)
        x = self.max_pool(x)
        x = self.relu(x)
        return x