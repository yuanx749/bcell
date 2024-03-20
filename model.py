import math
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from data import Vocab
from loss import FocalSigmoidLoss, FocalSoftmaxLoss

Tdict = Dict[str, Tensor]


class Encoder(nn.Module):
    def __init__(self, vocab: Vocab, emb_size, pretrain=True):
        super().__init__()
        if pretrain:
            weight = vocab.eig_embedding("blosum62.txt", emb_size)
            self.encoder = nn.Embedding.from_pretrained(weight)
        else:
            self.encoder = nn.Embedding(
                vocab.vocab_size, emb_size, padding_idx=vocab.stoi["<pad>"]
            )

    def forward(self, input: Tensor):
        return self.encoder(input)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=50, pretrain=False):
        super().__init__()
        self.pretrain = pretrain
        self.dropout = nn.Dropout(p=dropout)
        if pretrain:
            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
            )
            pe = torch.zeros(1, max_len, d_model)
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe)
        else:
            self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: Tensor):
        if self.pretrain:
            x = x + self.pe[:, : x.size(1), :]
        else:
            positions = torch.arange(x.size(1), device=x.device)
            x = x + self.pe(positions)
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, d_model, num_head, hidden_size, num_layers, dropout):
        super().__init__()
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, num_head, hidden_size, dropout, batch_first=True, norm_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self._reset_parameters()

    def forward(self, input: Tensor, mask: Tensor = None):
        output: Tensor = self.transformer_encoder(input, src_key_padding_mask=mask)
        return output

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class PETransformer(TransformerModel):
    def __init__(self, d_model, num_head, hidden_size, num_layers, dropout):
        super().__init__(d_model, num_head, hidden_size, num_layers, dropout)
        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(d_model, dropout)

    def forward(self, input: Tensor, mask: Tensor = None):
        input = input * math.sqrt(self.d_model)
        input = self.pos_encoder(input)
        output: Tensor = super().forward(input, mask)
        return output


class PaddedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, bidirectional=True
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, padded_input: Tensor, mask: Tensor):
        self.lstm.flatten_parameters()
        input_lengths = (~mask).sum(dim=1).tolist()
        total_length = padded_input.size(1)
        packed_input = nn.utils.rnn.pack_padded_sequence(
            padded_input, input_lengths, batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=total_length
        )
        output = self.dropout(output)
        return output


class DotProductAttention(nn.Module):
    def forward(self, q: Tensor, k: Tensor, v: Tensor, key_padding_mask: Tensor = None):
        attn = torch.bmm(q, k.transpose(-2, -1))
        attn = attn / math.sqrt(q.size()[-1])
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1)
            attn.masked_fill_(mask, float("-inf"))
        attn_weights = F.softmax(attn, dim=-1)
        output = torch.bmm(attn_weights, v)
        return output, attn_weights


class Model(nn.Module):
    def __init__(
        self,
        vocab: Vocab,
        emb_size=32,
        hidden_size=128,
        output_size=128,
        dropout=0.0,
        cls_num_lst=np.zeros(1),
        tau=1,
        pretrain=True,
        posthoc=False,
        softmax=False,
    ):
        super().__init__()
        self.vocab = vocab
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        n_class = len(cls_num_lst[1:-1])
        self.lstm = PaddedLSTM(emb_size, output_size // 2, 1, dropout)
        self.transformer_enc = TransformerModel(output_size, 2, hidden_size, 2, dropout)
        self.attn = DotProductAttention()
        self.query = nn.Parameter(torch.randn(1, 1, output_size))
        self.clf = Classifier3(
            hidden_size, dropout, n_class, cls_num_lst[1:-1], tau, posthoc, softmax
        )
        self.encoder = Encoder(vocab, emb_size, pretrain)

    def attention(self, input: Tensor, mask: Tensor = None):
        query = self.query.expand(input.size(0), -1, -1)
        key = input
        output, attn_output_weights = self.attn(query, key, key, key_padding_mask=mask)
        output: Tensor = output.squeeze(1)
        attn_output_weights: Tensor = attn_output_weights.squeeze(1)
        return output, attn_output_weights

    def forward(self, input: Tensor = None, **kwargs):
        mask: Tensor = input == self.vocab.stoi["<pad>"]
        input = self.encoder(input)
        x = self.lstm(input, mask)
        x = self.transformer_enc(x, mask)
        x_s, weights = self.attention(x, mask)
        output = self.clf(x_s, x, mask)
        output["weights"] = weights
        return output


class Classifier(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.LazyLinear(hidden_size),
            nn.Dropout(p=dropout),
            nn.ReLU(),
        )
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x_s: Tensor, x_r: Tensor, mask):
        logits_p: Tensor = self.fc(self.mlp1(x_s))
        pred_p = torch.argmax(logits_p, dim=1)
        score_p = F.softmax(logits_p, dim=1)[:, 1]
        logits: Tensor = self.fc(self.mlp1(x_r))
        logits = logits.transpose(1, 2)
        pred_r = torch.argmax(logits, dim=1)[~mask]
        score_r = F.softmax(logits, dim=1)[:, 1, :][~mask]
        return {
            "logits": logits,
            "prediction_r": pred_r,
            "score_r": score_r,
            "logits_p": logits_p,
            "prediction_p": pred_p,
            "score_p": score_p,
        }


class Classifier3(Classifier):
    def __init__(
        self,
        hidden_size,
        dropout,
        n_class,
        cls_num_lst,
        tau,
        posthoc=False,
        softmax=False,
    ):
        super().__init__(hidden_size, dropout)
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(hidden_size, n_class),
        )
        prior = torch.tensor(cls_num_lst) / sum(cls_num_lst)
        if softmax:
            self.adjustment = torch.log(prior)
        else:
            self.adjustment = torch.log(prior / (1 - prior))
        self.tau = tau
        self.posthoc = posthoc

    def forward(self, x_s: Tensor, x_r: Tensor, mask):
        d = super().forward(x_s, x_r, mask)
        logits_ig = self.mlp2(self.mlp1(x_s))
        self.adjustment = self.adjustment.to(x_s.device)
        if self.posthoc:
            pred_ig = torch.argmax(logits_ig - self.tau * self.adjustment, dim=1)
        else:
            pred_ig = torch.argmax(logits_ig, dim=1)
            logits_ig = logits_ig + self.tau * self.adjustment
        d["logits_ig"] = logits_ig
        d["prediction_ig"] = pred_ig
        return d


class Loss3(nn.Module):
    def __init__(self, gamma=0.0, alpha=1.0, ignore_index=-1, softmax=False):
        super().__init__()
        self.alpha = alpha
        self.loss_fn_2 = nn.CrossEntropyLoss(ignore_index=ignore_index)
        if softmax:
            self.focal_loss_3 = FocalSoftmaxLoss(gamma=gamma, ignore_index=ignore_index)
        else:
            self.focal_loss_3 = FocalSigmoidLoss(gamma=gamma, ignore_index=ignore_index)

    def forward(self, output: Tdict, batch: Tdict):
        loss_p = self.loss_fn_2(output["logits_p"], batch["label_p"])
        loss_ig = self.focal_loss_3(output["logits_ig"], batch["label_ig"]) * self.alpha
        return {
            "loss": loss_p + loss_ig,
            "loss_p": loss_p,
            "loss_ig": loss_ig,
        }
