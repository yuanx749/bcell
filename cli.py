import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "-i",
    "--input",
    required=True,
    help="path to input FASTA file",
)
parser.add_argument(
    "-o",
    "--output",
    required=True,
    help="path to output csv file",
)
parser.add_argument(
    "--model",
    default=Path(ROOT / "results" / "model.pt"),
    help="path to model",
)
args = parser.parse_args()

import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import Vocab
from model import *
from utils import Peptides, fasta2df, predict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(args.model, map_location=device)
vocab = Vocab(max_len=25)
df_in = fasta2df(args.input)
dataset = Peptides(df_in, vocab)
dataloader = DataLoader(dataset, batch_size=32, collate_fn=dataset.collate_fn)
lst = predict(dataloader, model)


def dict2df(lst: dict) -> pd.DataFrame:
    label = dict(zip(range(3), ["A", "E", "M"]))
    data = {
        "score": lst["score_p"],
        "epitope": lst["prediction_p"],
        "Ig": [label[e] for e in lst["prediction_ig"]],
    }
    return pd.DataFrame(data)


df_out = dict2df(lst)
df = pd.concat([df_in, df_out], axis=1)
df.to_csv(args.output)
