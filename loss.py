import torch
import torch.nn.functional as F
from torch import Tensor, nn


class FocalSoftmaxLoss(nn.Module):
    def __init__(self, gamma=0.0, weight: Tensor = None, ignore_index=-1):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, input: Tensor, target: Tensor):
        if target.dim() > 1:
            n_class = input.size(1)
            input = input.permute(0, *range(2, input.dim()), 1).reshape(-1, n_class)
            target = target.flatten()
        mask = target != self.ignore_index
        input = input[mask]
        target = target[mask]
        if len(target) == 0:
            return torch.zeros(1, device=input.device)
        probs = F.softmax(input, dim=1)
        pt = probs.gather(dim=1, index=target.unsqueeze(1)).squeeze(1)
        if self.weight is not None:
            self.weight = self.weight.to(input.device)
        ce = F.cross_entropy(input, target, weight=self.weight, reduction="none")
        loss = (1 - pt) ** self.gamma * ce
        loss = loss.mean()
        return loss


class FocalSigmoidLoss(nn.Module):
    def __init__(self, gamma=0.0, weight: Tensor = None, ignore_index=-1):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, input: Tensor, target: Tensor):
        n_class = input.size(1)
        if target.dim() > 1:
            input = input.permute(0, *range(2, input.dim()), 1).reshape(-1, n_class)
            target = target.flatten()
        mask = target != self.ignore_index
        input = input[mask]
        target = target[mask]
        if len(target) == 0:
            return torch.zeros(1, device=input.device)
        target = F.one_hot(target, num_classes=n_class)
        target = target.type_as(input)
        p = torch.sigmoid(input)
        pt = p * target + (1 - p) * (1 - target)
        if self.weight is not None:
            self.weight = self.weight.to(input.device)
        ce = F.binary_cross_entropy_with_logits(
            input, target, weight=self.weight, reduction="none"
        )
        loss = (1 - pt) ** self.gamma * ce
        loss = loss.mean(dim=1).mean()
        return loss
