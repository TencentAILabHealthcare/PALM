# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

def CrossEntropyLoss(output, target):
    return F.cross_entropy(input=output, target=target)

def bce_withlogits_loss(output, target):
    return F.binary_cross_entropy_with_logits(output, target)

def mse_loss(output , target):
    return F.mse_loss(output,target)

def L1Loss(output, target):
    return F.l1_loss(output, target)    

def SmoothL1Loss(output, target):
    return F.smooth_l1_loss(output, target)

def Pearson_loss(preds, labels):
    loss = 0
    for i in range(preds.shape[0]):
        sum_x = torch.sum(preds[i])                # x
        sum_y = torch.sum(labels[i])               # y
        sum_xy = torch.sum(preds[i]*labels[i])        # xy
        sum_x2 = torch.sum(torch.pow(preds[i],2))  # x^2
        sum_y2 = torch.sum(torch.pow(labels[i],2)) # y^2
        N = 1
        pearson = (N*sum_xy - sum_x*sum_y)/(torch.sqrt((N*sum_x2 - torch.pow(sum_x,2))*(N*sum_y2 - torch.pow(sum_y,2))))
        loss += 1 - pearson
    loss = loss/preds.shape[0]
    return loss

class Neg_Pearson(nn.Module):    # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self):
        super(Neg_Pearson,self).__init__()
        return
    def forward(self, preds, labels):       # tensor [Batch, Temporal]
        loss = 0
        for i in range(preds.shape[0]):
            sum_x = torch.sum(preds[i])                # x
            sum_y = torch.sum(labels[i])               # y
            sum_xy = torch.sum(preds[i]*labels[i])        # xy
            sum_x2 = torch.sum(torch.pow(preds[i],2))  # x^2
            sum_y2 = torch.sum(torch.pow(labels[i],2)) # y^2
            N = preds.shape[1]
            pearson = (N*sum_xy - sum_x*sum_y)/(torch.sqrt((N*sum_x2 - torch.pow(sum_x,2))*(N*sum_y2 - torch.pow(sum_y,2))))

            #if (pearson>=0).data.cpu().numpy():    # torch.cuda.ByteTensor -->  numpy
            #    loss += 1 - pearson
            #else:
            #    loss += 1 - torch.abs(pearson)
            
            loss += 1 - pearson
            
            
        loss = loss/preds.shape[0]
        return loss