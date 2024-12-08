# Author: Nam Bui

from __future__ import annotations
from typing import Generator, Any, Callable, Iterable
from numpy.typing import NDArray
from torch import Tensor

# import time
# import math

# from collections import defaultdict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as f
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# from torch.nn.utils import clip_grad_norm
# from torch.optim.lr_scheduler import ExponentialLR

from common import *


class LSTMBinaryClassifier(nn.Module):
    def __init__(
        self,
        nwords: int,
        embed_size: int,
        hidden_size: Iterable[int] | int,
        embed_weights: NDArray | Tensor = None,
        freeze_embed: bool = False,
        embedding_dropout: float = 0.5,
    ):
        super(LSTMBinaryClassifier, self).__init__()

        # Create embedding layer
        self.embedding = nn.Embedding(nwords, embed_size)
        self.embedding_droupout = nn.Dropout(p=embedding_dropout)
        if embed_weights is not None:
            n, d = embed_weights.size()
            # Load pretrained embeddings
            if n == nwords and d == embed_size:
                self.embedding.load_state_dict({"weight": embed_weights})
                # Freeze embeddings
                if freeze_embed:
                    self.embedding.weight.requires_grad = False
                    self.embedding_droupout.p = 0.0
            else:
                print(
                    "Embed size or nwords do not match dimensions of embedding weights."
                )

        # Create hidden layers
        self.fc = nn.ModuleList()
        if isinstance(hidden_size, int):
            hidden_size = [hidden_size]
        hidden_size += [2]  # Output layer size = 2 for binary classification
        for i in range(len(hidden_size) - 1):
            self.fc.append(nn.Linear(hidden_size[i], hidden_size[i + 1]))

        # LSTM layer
        self.rnn = nn.LSTM(embed_size, hidden_size[0])

    def forward(self, x: Tensor, lengths: Tensor = None, h: Tensor = None):
        # Get embeddings
        x = self.embedding(x)
        x = self.embedding_droupout(x)

        # Pack batch for RNN
        if lengths is None:
            lengths = torch.Tensor(len(x))
        x = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # Pass through RNN layer
        x, h = self.rnn(x, h)
        x, _ = pad_packed_sequence(x, batch_first=True)

        # Pass through fully connected layers
        for fc in self.fc[:-1]:
            x = f.relu(fc(x))
        x = self.fc[-1](x)

        # Return last output (classification)
        return x[:, -1, :], h


def calc_loss(
    model: nn.Module,
    sent: torch.Tensor,
    lengths: None | torch.Tensor,
    labels: torch.Tensor,
    criterion=nn.CrossEntropyLoss(reduction="sum"),
    device="cpu",
) -> Tensor:
    model.zero_grad()
    sent = sent.to(device=device)
    labels = labels.to(device=device)

    outputs, _ = model(sent, lengths)

    correct = (torch.argmax(outputs, dim=1) == labels).sum()

    return criterion(outputs, labels), correct
