from collections import Counter, defaultdict
from itertools import product
from typing import Dict, List

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import Vocab

Tdict = Dict[str, Tensor]


def batchify(seqs: List[str], vocab=Vocab()):
    length = [len(s) for s in seqs]
    batch = [vocab.numericalize(s) for s in seqs]
    batch = [torch.tensor(lst) for lst in batch]
    batch = nn.utils.rnn.pad_sequence(
        batch, batch_first=True, padding_value=vocab.stoi["<pad>"]
    )
    return batch, length


def dipeptide_composition(seq: str, keys=None):
    if keys is None:
        keys = list(map("".join, (product("ACDEFGHIKLMNPQRSTVWY", repeat=2))))
    count = Counter([seq[i : i + 2] for i in range(len(seq) - 1)])
    vector = [count[k] for k in keys]
    vector = [e / sum(vector) for e in vector]
    return vector


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
