# -*- coding: utf-8 -*-

import os
import json
import gzip
import shutil
import inspect
import numpy as np
from torch.utils.data import Dataset

class DatasetSplit(Dataset):
    """
    Dataset split. Thin wrapper on top a dataset to provide data split functionality.
    Can also enable dynamic example generation for train fold if supported by
    the wrapped dataset (NOT for valid/test folds) via dynamic_training flag

    kwargs are forwarded to shuffle_indices_train_valid_test
    """

    def __init__(self, logger, full_dataset: Dataset, split: str, dynamic_training: bool = False,
                 **kwargs):
        self.logger = logger
        self.dset = full_dataset
        split_to_idx = {"train": 0, "valid": 1, "test": 2}
        assert split in split_to_idx
        self.split = split
        self.dynamic = dynamic_training
        if self.split != "train":
            assert not self.dynamic, "Cannot have dynamic examples for valid/test"
        
        self.idx = self.shuffle_indices_train_valid_test(np.arange(len(self.dset)), **kwargs)[split_to_idx[self.split]]
        self.logger.info(f"Split {self.split} with {len(self)} examples")

    def all_labels(self, **kwargs) -> np.ndarray:
        """Get all labels"""
        if not hasattr(self.dset, "get_ith_label"):
            raise NotImplementedError("Wrapped dataset must implement get_ith_label")
        labels = [
            self.dset.get_ith_label(self.idx[i], **kwargs) for i in range(len(self))
        ]
        return np.stack(labels)

    def all_sequences(self, **kwargs):
        """Get all sequences"""
        if not hasattr(self.dset, "get_ith_sequence"):
            raise NotImplementedError(
                f"Wrapped dataset {type(self.dset)} must implement get_ith_sequence"
            )
        # get_ith_sequence could return a str or a tuple of two str (TRA/TRB)
        sequences = [
            self.dset.get_ith_sequence(self.idx[i], **kwargs) for i in range(len(self))
        ]
        return sequences

    def to_file(self, fname: str, compress: bool = True) -> str:
        """
        Write to the given file
        """
        if not (
            hasattr(self.dset, "get_ith_label")
            and hasattr(self.dset, "get_ith_sequence")
        ):
            raise NotImplementedError(
                "Wrapped dataset must implement both get_ith_label & get_ith_sequence"
            )
        assert fname.endswith(".json")
        all_examples = []
        for idx in range(len(self)):
            seq = self.dset.get_ith_sequence(self.idx[idx])
            label_list = self.dset.get_ith_label(self.idx[idx]).tolist()
            all_examples.append((seq, label_list))

        with open(fname, "w") as sink:
            json.dump(all_examples, sink, indent=4)

        if compress:
            with open(fname, "rb") as source:
                with gzip.open(fname + ".gz", "wb") as sink:
                    shutil.copyfileobj(source, sink)
            os.remove(fname)
            fname += ".gz"
        assert os.path.isfile(fname)
        return os.path.abspath(fname)

    def shuffle_indices_train_valid_test(self, idx:np.ndarray, valid:float=0.15, test:float=0.15, seed:int=1234):
        """
        Given an array of indices, return indices partitioned into train, valid, and test indices
        The following tests ensure that ordering is consistent across different calls
        >>> np.all(shuffle_indices_train_valid_test(np.arange(100))[0] == shuffle_indices_train_valid_test(np.arange(100))[0])
        True
        >>> np.all(shuffle_indices_train_valid_test(np.arange(10000))[1] == shuffle_indices_train_valid_test(np.arange(10000))[1])
        True
        >>> np.all(shuffle_indices_train_valid_test(np.arange(20000))[2] == shuffle_indices_train_valid_test(np.arange(20000))[2])
        True
        >>> np.all(shuffle_indices_train_valid_test(np.arange(1000), 0.1, 0.1)[1] == shuffle_indices_train_valid_test(np.arange(1000), 0.1, 0.1)[1])
        True
        """
        np.random.seed(seed)  # For reproducible subsampling
        indices = np.copy(idx)  # Make a copy because shuffling occurs in place
        np.random.shuffle(indices)  # Shuffles inplace
        num_valid = int(round(len(indices) * valid)) if valid > 0 else 0
        num_test = int(round(len(indices) * test)) if test > 0 else 0
        num_train = len(indices) - num_valid - num_test
        assert num_train > 0 and num_valid >= 0 and num_test >= 0
        assert num_train + num_valid + num_test == len(
            indices
        ), f"Got mismatched counts: {num_train} + {num_valid} + {num_test} != {len(indices)}"

        indices_train = indices[:num_train]
        indices_valid = indices[num_train : num_train + num_valid]
        indices_test = indices[-num_test:] if num_test > 0 else np.array([])
        assert indices_train.size + indices_valid.size + indices_test.size == len(idx)

        return indices_train, indices_valid, indices_test

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, idx: int):
        if (
            self.dynamic
            and self.split == "train"
            and "dynamic" in inspect.getfullargspec(self.dset.__getitem__).args
        ):
            return self.dset.__getitem__(self.idx[idx], dynamic=True)
        return self.dset.__getitem__(self.idx[idx])