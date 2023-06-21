import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from data import EpitopeDataset, Vocab
from model import *
from train import print_metrics
from utils import evaluate


def compute_metrics(lst: dict):
    metrics = {
        "accuracy": balanced_accuracy_score(lst["label_p"], lst["prediction_p"]),
        "auroc": roc_auc_score(lst["label_p"], lst["score_p"]),
    }
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
vocab = Vocab(max_len=25)
valid_set = EpitopeDataset(vocab=vocab, root=ROOT / "data", split="valid")
valid_loader = DataLoader(
    valid_set,
    batch_size=10000,
    shuffle=False,
    collate_fn=valid_set.collate_fn,
)
test_set = EpitopeDataset(vocab=vocab, root=ROOT / "data", split="test")
test_loader = DataLoader(
    test_set,
    batch_size=10000,
    shuffle=False,
    collate_fn=test_set.collate_fn,
)

cls_num_lst = valid_set.get_cls_num_lst()
cls_num_lst = cls_num_lst[1:-1]
prior = cls_num_lst / sum(cls_num_lst)
adjustment = np.log(prior / (1 - prior))

file_lst = range(5)
num = 21
valid_acc = np.zeros((len(file_lst), num))
test_acc = np.zeros_like(valid_acc)
tau_lst = np.linspace(0, 5, num=num)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for i, file in enumerate(file_lst):
    model = torch.load(result_dir / f"{file}.pt", map_location=device)
    valid_lst = evaluate(valid_loader, model)
    compute_metrics(valid_lst)
    test_lst = evaluate(test_loader, model)
    compute_metrics(test_lst)

    for j, tau in enumerate(tau_lst):
        for lst, acc in zip([valid_lst, test_lst], [valid_acc, test_acc]):
            label_ig = np.array(lst["label_ig"])
            logits_ig = lst["logits_ig"] - tau * adjustment
            mask = label_ig != -1
            label_ig = label_ig[mask]
            logits_ig = logits_ig[mask]
            prediction_ig = np.argmax(logits_ig, axis=1)
            acc[i, j] = balanced_accuracy_score(label_ig, prediction_ig)

plt.style.use(result_dir / "conf.mplstyle")
fig, ax = plt.subplots()
valid_acc_mean = valid_acc.mean(axis=0)
valid_acc_std = valid_acc.std(axis=0)
test_acc_mean = test_acc.mean(axis=0)
test_acc_std = test_acc.std(axis=0)
ax.fill_between(
    tau_lst,
    test_acc_mean + test_acc_std / 2,
    test_acc_mean - test_acc_std / 2,
    alpha=0.2,
)
ax.plot(tau_lst, test_acc_mean, label="Test")
ax.fill_between(
    tau_lst,
    valid_acc_mean + valid_acc_std / 2,
    valid_acc_mean - valid_acc_std / 2,
    alpha=0.2,
)
ax.plot(tau_lst, valid_acc_mean, label="Validation")
ax.set_ylim([0.6, 0.75])
ax.set_xlabel(r"Scaling Parameter ($\tau$)")
ax.set_ylabel("Balanced Accuracy")
ax.legend()
ax.grid()
fig.savefig(ROOT / "results" / "tau.pdf")
np.savetxt(ROOT / "results" / "tau.txt", test_acc, fmt="%.4f")
