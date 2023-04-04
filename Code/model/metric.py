# -*- coding: utf-8 -*-

import json
import numpy as np
import pandas as pd
import itertools
from sklearn import metrics
from scipy import stats
from math import sqrt

def accuracy_sample(y_pred, y_true):
    """Compute the accuracy for each sample

    Note that the input y_pred and y_true are both numpy array.
    Args:
        y_pred (numpy.array): shape [seq_len*2, batch_size, ntoken]
        y_true (numpy.array): shape [seq_len*2, batch_size]
    """
    y_pred = y_pred.argmax(axis=2)
    print('Shape of y_pred:', y_pred.shape)
    print('Shape of y_true:', y_true.shape)
    return metrics.accuracy_score(y_pred=y_pred, y_true=y_true)

def accuracy_amino_acid(y_pred, y_true):
    '''Compute teh accuracy for each amino acid.

    Args:
        y_pred (numpy.array): shape [batch_size, seq_len, ntoken]
        y_true (numpy.array): shape [batch_size, seq_len]
    '''
    y_pred = y_pred.argmax(axis=2)
    return metrics.accuracy_score(y_pred=y_pred.flatten(), y_true=y_true.flatten())

def correct_count_seq(y_pred, y_true):
    '''Count the correct prediction for each amino acid.

    Args:
        y_pred (numpy.array): shape [batch_size, seq_len, ntoken]
        y_true (numpy.array): shape [batch_size, seq_len]
    '''
    y_pred = y_pred.argmax(axis=2)
    y_true_copy = y_true.copy()
    y_true_copy[y_true_copy==21] = -100
    return (y_pred == y_true_copy).sum(), np.count_nonzero(y_true_copy != -100)


class MAA_metrics(object):
    def __init__(self, token_with_special_list, blosum_dir, blosum=False):
        self.token_with_special_list = token_with_special_list
        self.blosum_dir = blosum_dir
        self.blosum = blosum
        
        self.AMINO_ACIDS = "ACDEFGHIKLMNPQRSTUVWY"
        self.BLOSUM = self._load_blosum()

    def _load_blosum(self):
        """Return the blosum matrix as a dataframe"""
        with open(self.blosum_dir) as source:
            d = json.load(source)
            retval = pd.DataFrame(d)
        retval = pd.DataFrame(0, index=list(self.AMINO_ACIDS), columns=list(self.AMINO_ACIDS))
        for x, y in itertools.product(retval.index, retval.columns):
            if x == "U" or y == "U":
                continue
            retval.loc[x, y] = d[x][y]
        retval.drop(index="U", inplace=True)
        retval.drop(columns="U", inplace=True)
        return retval

    def compute_metrics(self, pred, top_n=3):
        """
        Compute metrics to report
        top_n controls the top_n accuracy reported
        """
        # labels are -100 for masked tokens and value to predict for masked token
        labels = pred.label_ids.squeeze()  # Shape (n, 32)
        preds = pred.predictions  # Shape (n, 32, 26)

        n_mask_total = 0
        top_one_correct, top_n_correct = 0, 0
        blosum_values = []
        print("1234",labels.shape)
        for i in range(labels.shape[0]):
            masked_idx = np.where(labels[i] != -100)[0]
            n_mask = len(masked_idx)  # Number of masked items
            n_mask_total += n_mask
###
            if n_mask_total == 0:
                print("labels:",labels,labels[i])

            pred_arr = preds[i, masked_idx]
            truth = labels[i, masked_idx]  # The masked token indices
            # argsort returns indices in ASCENDING order
            pred_sort_idx = np.argsort(pred_arr, axis=1)  # apply along vocab axis
            # Increments by number of correct in top 1
            top_one_correct += np.sum(truth == pred_sort_idx[:, -1])
            top_n_preds = pred_sort_idx[:, -top_n:]
            for truth_idx, top_n_idx in zip(truth, top_n_preds):
                # Increment top n accuracy
                top_n_correct += truth_idx in top_n_idx
                # Check BLOSUM score
                if self.blosum:
                    truth_res = self.token_with_special_list[truth_idx]
                    pred_res = self.token_with_special_list[top_n_idx[-1]]
                    for aa_idx in range(min(len(truth_res), len(pred_res))):
                        if truth_res[aa_idx] in self.BLOSUM.index and pred_res[aa_idx] in self.BLOSUM.index:
                            blosum_values.append(self.BLOSUM.loc[truth_res[aa_idx], pred_res[aa_idx]])
        # These should not exceed each other
        assert top_one_correct <= top_n_correct <= n_mask_total
        print("asdfasdf",n_mask_total)
        if self.blosum:
            retval = {
                f"top_{top_n}_acc": top_n_correct / n_mask_total,
                "acc": top_one_correct / n_mask_total,
                "average_blosum": np.mean(blosum_values)}
        else:
            retval = {
                f"top_{top_n}_acc": top_n_correct / n_mask_total,
                "acc": top_one_correct / n_mask_total} 
        return retval


    
def mse(y_pred , y_true):
    return metrics.mean_squared_error(y_true, y_pred)

def r_squared_error(y_pred, y_obs):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))

    return mult / float(y_obs_sq * y_pred_sq)
def get_rm2(ys_line, ys_orig):
    r2 = r_squared_error(ys_line, ys_orig)
    r02 = squared_error_zero(ys_line, ys_orig)

    return r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r02 * r02))))

def get_cindex(P, Y):
    summ = 0
    pair = 0

    for i in range(1, len(Y)):
        for j in range(0, i):
            if i is not j:
                if (Y[i] > Y[j]):
                    pair += 1
                    summ += 1 * (P[i] > P[j]) + 0.5 * (P[i] == P[j])

    if pair != 0:
        return summ / pair
    else:
        return 0
    
def get_pearson(f, y):
    
    
    f = np.array(f).flatten()

    y = np.array(y).flatten()
    
    rp = np.corrcoef(y, f)[0, 1]
    return rp


def get_spearman(f, y):
    #print("spearman input: ",f , y)
    rs = stats.spearmanr(y, f)[0]
    return rs


def accuracy(y_pred, y_true):
    """
    Note that the input y_pred and y_true are both numpy array.
    Args:
        y_pred (numpy.array): 
        y_true (numpy.array): [description]
    """
    return metrics.accuracy_score(y_pred=y_pred.round(), y_true=y_true)

def recall(y_pred, y_true):
    """
    Note that the input y_pred and y_true are both numpy array.
    Args:
        y_pred ([type]): [description]
        y_true ([type]): [description]
    Returns:
        [type]: [description]
    """
    return metrics.recall_score(y_pred=y_pred.round(), y_true=y_true)

def roc_auc(y_pred, y_true):
    """
    The values of y_pred can be decimicals, within 0 and 1.
    Args:
        y_pred ([type]): [description]
        y_true ([type]): [description]
    Returns:
        [type]: [description]
    """
    return metrics.roc_auc_score(y_score=y_pred, y_true=y_true)

class Seq2Seq_metrics(object):
    def __init__(self, 
                 logger, 
                 model_variant, 
                 antibody_tokenizer, 
                 antigen_tokenizer, 
                 blosum_dir, 
                 blosum=False):
        self.logger = logger
        self.blosum_dir = blosum_dir
        self.blosum = blosum

        if model_variant == 'Antibody-Antigen':
            self.logger.info("Using Antigen tokenizer in metrics computation.")
            self.tokenizer = antigen_tokenizer
        elif model_variant == 'Antigen-Antibody':
            self.logger.info("Using Antibody tokenizer in metrics computation.")
            self.tokenizer = antibody_tokenizer
        else:
            self.logger.info("model_variant is not valid!")

        self.AMINO_ACIDS = "ACDEFGHIKLMNPQRSTUVWY"
        self.BLOSUM = self._load_blosum()
        self._has_logged_example = False

    def _load_blosum(self):
        """Return the blosum matrix as a dataframe"""
        with open(self.blosum_dir) as source:
            d = json.load(source)
            retval = pd.DataFrame(d)
        retval = pd.DataFrame(0, index=list(self.AMINO_ACIDS), columns=list(self.AMINO_ACIDS))
        for x, y in itertools.product(retval.index, retval.columns):
            if x == "U" or y == "U":
                continue
            retval.loc[x, y] = d[x][y]
        retval.drop(index="U", inplace=True)
        retval.drop(columns="U", inplace=True)
        return retval

    def compute_metrics(self, pred):
        labels_ids = pred.label_ids # shape (batch_size, max_length)
        pred_ids = pred.predictions # shape (batch_size, max_length)

        pred_str_list = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = self.tokenizer.pad_token_id
        label_str_list = self.tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        if not self._has_logged_example:
            self.logger.info("Predicted token: {} -> string: {}".format(
                pred_ids[0], pred_str_list[0].replace(" ", "")
            ))
            self.logger.info("Target token: {} -> string: {}".format(
                labels_ids[0], label_str_list[0].replace(" ", "")
            ))
            self._has_logged_example = True

        top_one_correct = 0
        blosum_values = []
        for pred_str, label_str in zip(pred_str_list, label_str_list):
            pred_str = pred_str.replace(" ", "")
            label_str = label_str.replace(" ", "")
            for i in range(min(len(pred_str), len(label_str))):
                if pred_str[i] == label_str[i]:
                    top_one_correct += 1
                if self.blosum:
                    if pred_str[i] in self.BLOSUM.index and label_str[i] in self.BLOSUM.index:
                        blosum_values.append(self.BLOSUM.loc[label_str[i], pred_str[i]])

        total_aa = sum([len(s.replace(" ", "")) for s in label_str_list])
        retval = {
            "acc": top_one_correct / total_aa,
            "average_blosum": np.mean(blosum_values)
        }

        ###
        self._has_logged_example = False


        return retval

        