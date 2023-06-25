import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from data import FASTA, EpitopeDataset, Vocab
from model import *
from train import print_metrics
from utils import evaluate


def compute_metrics(lst: dict):
    metrics = {
        "accuracy": balanced_accuracy_score(lst["label_p"], lst["prediction_p"]),
        "auroc": roc_auc_score(lst["label_p"], lst["score_p"]),
        "confusion_matrix_p": confusion_matrix(lst["label_p"], lst["prediction_p"]),
    }
    if "label_ig" in lst.keys():
        label_ig = np.array(lst["label_ig"])
        prediction_ig = np.array(lst["prediction_ig"])
        mask = label_ig != -1
        label_ig = label_ig[mask]
        prediction_ig = prediction_ig[mask]
        metrics.update(
            {
                "balanced_accuracy": balanced_accuracy_score(label_ig, prediction_ig),
                "recall": recall_score(label_ig, prediction_ig, average=None),
                "confusion_matrix": confusion_matrix(label_ig, prediction_ig),
            }
        )
    print_metrics(metrics, "Test")
    return metrics


result_dir = ROOT / "results"
file = "model.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(result_dir / file, map_location=device)
vocab = Vocab(max_len=25)

test_data = EpitopeDataset(vocab=vocab, root=ROOT / "data", split="test")
test_loader = DataLoader(
    test_data,
    batch_size=1024,
    shuffle=False,
    collate_fn=test_data.collate_fn,
)
compute_metrics(evaluate(test_loader, model))

test_data = FASTA("sars.fasta", root=ROOT / "data", vocab=vocab)
test_loader = DataLoader(test_data, batch_size=1, collate_fn=test_data.collate_fn)
lst = evaluate(test_loader, model)
compute_metrics(lst)

models = ["EpiDope", "EpitopeVec", "BepiPred-3.0"]
label = {"Ours": lst["label_p"]}
score = {"Ours": lst["score_p"]}
for name in models:
    df = pd.read_csv(result_dir / f"{name}.csv", index_col=0)
    label[name] = df["label"].tolist()
    score[name] = df["score"].tolist()
models.insert(0, "Ours")

plt.style.use(result_dir / "conf.mplstyle")
fig, ax = plt.subplots()
for name in models:
    fpr, tpr, thresholds = roc_curve(label[name], score[name])
    auc = roc_auc_score(label[name], score[name])
    ax.plot(fpr, tpr, label=f"{name:<13}AUC = {auc:.3f}")
ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
ax.set_xlim([-0.01, 1.0])
ax.set_ylim([0.0, 1.01])
ax.set_xlabel("False Positive Rate (1 - Specificity)")
ax.set_ylabel("True Positive Rate (Sensitivity)")
ax.legend(loc="lower right")
fig.savefig(ROOT / "results" / "roc.pdf")
