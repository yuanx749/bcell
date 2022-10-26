import copy
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import transformers
from sklearn.metrics import (
    balanced_accuracy_score,
    recall_score,
    roc_auc_score,
)
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import EpitopeDataset, Vocab
from model import Loss3, Model

Tdict = Dict[str, Tensor]


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@dataclass
class Args:
    lr: float = 0.001
    l2: float = 0.0000
    batch_size: int = 128
    emb_size: int = 32
    hidden_size: int = 128
    output_size: int = 128
    dropout: float = 0.0
    alpha: float = 1
    gamma: float = 1
    tau: float = 1
    max_len: int = 25
    epoch: int = 100


def train_eval_epoch(
    dataloader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scheduler: torch.optim.lr_scheduler.LambdaLR = None,
    is_train=True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if is_train:
        model.train()
    else:
        model.eval()
    total_loss = defaultdict(float)
    count = 0
    lst = defaultdict(list)
    progress_bar = tqdm(dataloader, ascii=True)
    with torch.set_grad_enabled(is_train):
        for batch_idx, batch in enumerate(progress_bar):
            batch: Tdict = {k: v.to(device) for k, v in batch.items()}
            output: Tdict = model(**batch)
            batch_loss: Tdict = criterion(output, batch)
            loss = batch_loss["loss"]
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler:
                    scheduler.step()
            for k, v in batch_loss.items():
                total_loss[k] += v.item() * len(batch["label_p"])
            progress_bar.set_description_str(
                f"Batch: {batch_idx:d}, Loss: {loss.item():.4f}"
            )
            count += len(batch["label_p"])
            for k in batch.keys():
                if k.startswith("label_"):
                    lst[k].extend(batch[k].tolist())
            for k in output.keys():
                if k.startswith("prediction_") or k.startswith("score_"):
                    lst[k].extend(output[k].tolist())
    avg_loss = {k: v / count for k, v in total_loss.items()}
    label_ig = np.array(lst["label_ig"])
    prediction_ig = np.array(lst["prediction_ig"])
    mask = label_ig != -1
    label_ig = label_ig[mask]
    prediction_ig = prediction_ig[mask]
    metrics = {
        "accuracy": balanced_accuracy_score(lst["label_p"], lst["prediction_p"]),
        "auroc": roc_auc_score(lst["label_p"], lst["score_p"]),
        "balanced_acc": balanced_accuracy_score(label_ig, prediction_ig),
        "recall": recall_score(label_ig, prediction_ig, average=None),
    }
    return avg_loss, metrics


def print_metrics(metrics: dict, mode: str):
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{mode} {k}: {v:.4f}.", end=" ")
    print()
    for k, v in metrics.items():
        if not isinstance(v, float):
            print(f"{mode} {k}:\n{v}")


def train(args: Args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = Vocab(tokens=[], max_len=args.max_len)
    dataset = EpitopeDataset
    train_set = dataset(vocab=vocab, split="train")
    valid_set = dataset(vocab=vocab, split="valid")
    test_set = dataset(vocab=vocab, split="test")
    g_cpu = torch.Generator()
    g_cpu.manual_seed(2147483647)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_set.collate_fn,
        drop_last=True,
        generator=g_cpu,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_set.collate_fn,
        generator=g_cpu,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_set.collate_fn,
        generator=g_cpu,
    )
    cls_num_lst = train_set.get_cls_num_lst()
    model = Model(
        vocab=vocab,
        emb_size=args.emb_size,
        hidden_size=args.hidden_size,
        output_size=args.output_size,
        dropout=args.dropout,
        cls_num_lst=cls_num_lst,
        tau=args.tau,
    ).to(device)
    criterion = Loss3(args.gamma, args.alpha)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.l2, amsgrad=True
    )
    scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, 200, 10000)
    # scheduler = None
    best_model = None
    for epoch_idx in range(args.epoch):
        train_losses, train_metrics = train_eval_epoch(
            train_loader, model, criterion, optimizer=optimizer, scheduler=scheduler
        )
        valid_losses, valid_metrics = train_eval_epoch(
            valid_loader, model, criterion, is_train=False
        )
        print(f"Epoch {epoch_idx}")
        print_metrics(train_losses, "Training")
        print_metrics(valid_losses, "Validation")
        print_metrics(train_metrics, "Training")
        print_metrics(valid_metrics, "Validation")
    best_model = copy.deepcopy(model)
    _, test_metrics = train_eval_epoch(
        test_loader, best_model, criterion, is_train=False
    )
    print_metrics(test_metrics, "Test")
    return best_model


if __name__ == "__main__":
    args = Args(
        lr=0.001,
        l2=0.001,
        batch_size=1024,
        emb_size=21,
        hidden_size=128,
        output_size=128,
        dropout=0.2,
        alpha=1,
        gamma=1,
        tau=1,
        max_len=25,
        epoch=100,
    )
    set_seed(0)
    model = train(args)
    torch.save(model, Path("results", "model.pt"))
