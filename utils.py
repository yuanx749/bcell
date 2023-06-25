from collections import defaultdict
from typing import Dict, List

import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data import Vocab

Tdict = Dict[str, Tensor]


def evaluate(data: DataLoader, model: nn.Module):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    lst = defaultdict(list)
    progress_bar = tqdm(data, ascii=True)
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            batch: Tdict = {k: v.to(device) for k, v in batch.items()}
            output: Tdict = model(**batch)
            for k in batch.keys():
                if k.startswith("label_"):
                    lst[k].extend(batch[k].tolist())
            for k in output.keys():
                lst[k].extend(output[k].tolist())
    return lst


def fasta2df(file) -> pd.DataFrame:
    rows = []
    columns = ["identifier", "sequence"]
    with open(file) as f:
        for line in f:
            if line[0] == ">":
                identifier = line[1:].rstrip()
                break
        else:
            raise ValueError("Empty file.")
        seq = ""
        for line in f:
            if line[0] != ">":
                seq += line.rstrip()
            else:
                rows.append((identifier, seq))
                identifier = line[1:].rstrip()
                seq = ""
        rows.append((identifier, seq))
    return pd.DataFrame.from_records(rows, columns=columns)


class Peptides(Dataset):
    def __init__(self, df: pd.DataFrame, vocab=Vocab()):
        self.vocab = vocab
        self.input = [vocab.numericalize(seq) for seq in df["sequence"]]

    def __len__(self):
        return len(self.input)

    def __getitem__(self, key):
        return self.input[key]

    def collate_fn(self, batch: List[List[int]]):
        input = [torch.tensor(lst) for lst in batch]
        input = nn.utils.rnn.pad_sequence(
            input, batch_first=True, padding_value=self.vocab.stoi["<pad>"]
        )
        return input


def predict(data: DataLoader, model: nn.Module):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    lst = defaultdict(list)
    progress_bar = tqdm(data, ascii=True)
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            batch: Tensor = batch.to(device)
            output: Tdict = model(input=batch)
            for k in output.keys():
                lst[k].extend(output[k].tolist())
    return lst
