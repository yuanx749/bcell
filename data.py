from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import scipy.linalg as linalg
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class Vocab:
    def __init__(
        self,
        alphabet="ACDEFGHIKLMNPQRSTVWY",
        tokens=None,
        max_len=50,
        padding=False,
    ):
        if tokens is None:
            tokens = []
        self.alphabet = alphabet
        self.toks = tokens
        self.max_len = max_len
        self.padding = padding
        self.fix_len = ("<cls>" in tokens) + max_len + ("<eos>" in tokens)
        self.itos = ["<pad>"] + list(self.alphabet) + ["<unk>"] + tokens
        self.stoi = dict(zip(self.itos, range(len(self.itos))))
        self.vocab_size = len(self.itos)

    def tokenize(self, seq):
        tokens = list(seq)
        tokens = tokens[: self.max_len]
        if "<cls>" in self.toks:
            tokens.insert(0, "<cls>")
        if "<eos>" in self.toks:
            tokens.append("<eos>")
        if self.padding:
            tokens.extend(["<pad>"] * (self.max_len - len(seq)))
        return tokens

    def numericalize(self, seq):
        tokens = self.tokenize(seq)
        return [self.stoi.get(token, self.stoi["<unk>"]) for token in tokens]

    @staticmethod
    def read_matrix(filename, data_dir="./data"):
        df = pd.read_csv(Path(data_dir, filename), sep="\s+", index_col=0)
        df = df.sort_index().loc[:, df.columns.sort_values()]
        return df.to_numpy(dtype=float)

    def init_weight(self, mat, emb_size=None):
        mat = mat / linalg.norm(mat) * len(mat)
        weight = np.eye(self.vocab_size, M=emb_size, k=-1)
        weight[1:21, 0:20] = mat
        return torch.tensor(weight, dtype=torch.float)

    def eig_embedding(self, filename, emb_size=None, data_dir="./data"):
        a = self.read_matrix(filename, data_dir)
        w, v = linalg.eigh(np.exp2(a))
        v = v * np.sign(v[0])
        mat = v @ np.diag(w**0.5)
        return self.init_weight(mat, emb_size)


class SequenceBase(Dataset):
    def __init__(self):
        self.input = []
        self.label = []

    def __len__(self):
        return len(self.input)

    def __getitem__(self, key):
        input = self.input[key]
        label = self.label[key]
        return {"input": input, "label": label}

    def collate_fn(self, batch: List[dict]):
        value_lst = list(zip(*[data.values() for data in batch]))
        batch_dict = dict(zip(batch[0].keys(), value_lst))
        return batch_dict


class EpitopeDataset(SequenceBase):
    def __init__(
        self,
        vocab=Vocab(),
        root="./data",
        split="train",
    ):
        super().__init__()
        self.vocab = vocab
        self.df = pd.read_csv(Path(root, f"{split}.csv"), index_col=0)
        self.input = [vocab.numericalize(seq) for seq in self.df["Description"]]
        self.label = self.df["Label"].tolist()

    def get_cls_num_lst(self):
        return np.unique(self.label, return_counts=True)[1]

    def collate_fn(self, batch: List[dict]):
        batch_dict = super().collate_fn(batch)
        d = {}
        input = [torch.tensor(lst) for lst in batch_dict["input"]]
        input = nn.utils.rnn.pad_sequence(
            input, batch_first=True, padding_value=self.vocab.stoi["<pad>"]
        )
        d["input"] = input
        label_ig = torch.tensor(batch_dict["label"])
        d["label_p"] = (label_ig > 0).type_as(label_ig)
        labels = label_ig.unsqueeze(1).expand_as(input).clone()
        labels[input == self.vocab.stoi["<pad>"]] = -1
        labels[labels > 0] = 1
        d["label_r"] = labels[labels != -1]
        d["label"] = labels
        label_ig = label_ig - 1
        label_ig[label_ig == 3] = -1
        d["label_ig"] = label_ig
        return d


class BinaryClassification(SequenceBase):
    def __init__(self, vocab=Vocab()):
        super().__init__()
        self.vocab = vocab

    def collate_fn(self, batch: List[dict]):
        batch_dict = super().collate_fn(batch)
        d = {}
        input = [torch.tensor(lst) for lst in batch_dict["input"]]
        input = nn.utils.rnn.pad_sequence(
            input, batch_first=True, padding_value=self.vocab.stoi["<pad>"]
        )
        d["input"] = input
        label_p = torch.tensor(batch_dict["label"])
        labels = label_p.unsqueeze(1).expand_as(input).clone()
        labels[input == self.vocab.stoi["<pad>"]] = -1
        d["label_r"] = labels[labels != -1]
        d["label_p"] = label_p
        d["label"] = labels
        return d


class FASTA(BinaryClassification):
    def __init__(self, file, root="./data", vocab=Vocab()):
        super().__init__(vocab)
        labels = []
        seqs = []
        seq = ""
        with open(Path(root, file)) as f:
            for line in f:
                if line[0] == ">":
                    _, label = line[1:].strip().split("_")
                    labels.append(int(label))
                    if seq:
                        seqs.append(seq)
                        seq = ""
                else:
                    seq += line.strip()
        seqs.append(seq)
        self.input = [vocab.numericalize(seq) for seq in seqs]
        self.label = [int(label > 0) for label in labels]
