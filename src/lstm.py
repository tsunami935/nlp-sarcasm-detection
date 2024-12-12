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
                self.embedding = nn.Embedding.from_pretrained(embed_weights, freeze=freeze_embed)
                # self.embedding.load_state_dict({"weight": embed_weights})
                # Freeze embeddings
                if freeze_embed:
                    # self.embedding.weight.requires_grad = False
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

        # Attention
        # self.attention = nn.Linear(hidden_size[0], 1)

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
        x = x[range(len(lengths)), lengths - 1]

        # Pass through fully connected layers
        for fc in self.fc[:-1]:
            x = f.tanh(fc(x))
        x = self.fc[-1](x)

        # Return last output (classification)
        return x, h


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


if __name__ == "__main__":
    import json
    import time
    import datetime
    from argparse import ArgumentParser
    import os

    import pandas as pd
    from torch.utils.data import DataLoader
    import torch.optim as optim
    from torch.nn.utils import clip_grad_norm_
    from gensim.models.keyedvectors import KeyedVectors

    from sklearn.metrics import confusion_matrix

    from tokenized_dataloader import SarcasmDataset

    # Load configuration and hyperparameters
    parser = ArgumentParser()
    parser.add_argument("cfg", type=str, help="JSON file with configuration parameters")
    parser.add_argument("-d", "--device", type=str, help="CUDA device")
    parser.add_argument("-t", "--test", action="store_true", help="Model to test.")
    args = parser.parse_args()
    with open(args.cfg, "r") as fin:
        cfg: dict[str, Any] = json.load(fin)
        DATA_DIR: str = cfg.get("DATA_DIR", ".")
        TRAIN_FN: str = cfg.get("TRAIN_FN", "train_tokenized.json")
        VAL_FN: str = cfg.get("VAL_FN", "val_tokenized.json")
        TEST_FN: str = cfg.get("TEST_FN", "test_tokenized.json")
        W2I_FN: str = cfg.get("W2I_FN", "word2ind.json")
        OUT_DIR: str = cfg.get("OUT_DIR", ".")
        EMBED_FN: str | None = cfg.get("EMBED_FN", None)
        EMBED_SIZE: int = cfg.get("EMBED_SIZE", 100)
        HIDDEN_SIZE: int = cfg.get("HIDDEN_SIZE", 64)
        GRAD_CLIP_VALUE: float = cfg.get("GRAD_CLIP_VALUE", 5.0)
        BATCH_SIZE: int = cfg.get("BATCH_SIZE", 16)
        NUM_EPOCHS: int = cfg.get("NUM_EPOCHS", 50)
        MIN_EPOCHS: int = cfg.get("MIN_EPOCHS", 25)
        EARLY_STOP_THRESHOLD: float = cfg.get("EARLY_STOP_THRESHOLD", 0.05)
        NOISE_SD: float = cfg.get("NOISE_SD", 1e-6)
        LEARNING_RATE: float = cfg.get("LEARNING_RATE", 0.005)
        LR_GAMMA: float = cfg.get("LR_GAMMA", 1.0)
        CHKPT_INTERVAL: int = cfg.get("CHKPT_INTERVAL", 25)
        MODEL_NAME: str = cfg.get("MODEL_NAME", None)

    # Make output directory if it does not exist
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR, exist_ok=True)

    # CUDA device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Load standard vocabulary mapping or from word2vec
    if EMBED_FN is None:
        with open(DATA_DIR + W2I_FN, "r") as fin:
            w2i: dict[str, int] = json.load(fin)
    else:
        word2vec: KeyedVectors = KeyedVectors.load(EMBED_FN)
        word2vec.add_vector(TK_UNK, np.zeros(EMBED_SIZE))
        w2i = word2vec.key_to_index
        word2vec = torch.FloatTensor(word2vec.vectors)

    if args.test == False:
        print("Loading data")

        # Load Train and Validation Data
        train_set = SarcasmDataset.load_json(DATA_DIR + TRAIN_FN, w2i)
        val_set = SarcasmDataset.load_json(DATA_DIR + VAL_FN, w2i)
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)

        # Load embeddings & initialize model
        if EMBED_FN is None:
            model = LSTMBinaryClassifier(len(w2i), EMBED_SIZE, HIDDEN_SIZE).to(device)
        else:
            model = LSTMBinaryClassifier(
                len(w2i), EMBED_SIZE, HIDDEN_SIZE, word2vec, freeze_embed=True
            ).to(device)

        # Learning objective
        criterion = nn.CrossEntropyLoss(reduction="sum")
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_GAMMA)

        # Train model
        start = time.time()
        min_loss = np.inf
        for epoch in range(NUM_EPOCHS):
            print(f"epoch {epoch}:")

            # Train over training set
            running_loss = 0
            correct = 0
            for sid, (sent, lengths, labels) in enumerate(train_loader):
                # Calculate model error and  propogate loss
                loss, corr = calc_loss(
                    model, sent, lengths, labels, criterion, device=device
                )
                running_loss += loss.item()
                correct += corr.item()
                loss.backward()

                # Regularization: gradient clipping and noisy gradients
                clip_grad_norm_(model.parameters(), GRAD_CLIP_VALUE * BATCH_SIZE)
                for layer in model.parameters():
                    if layer.requires_grad:
                        layer.grad += torch.randn_like(layer.grad) * NOISE_SD
                optimizer.step()

            # Evaluate over validation set
            val_loss = 0
            val_correct = 0
            with torch.no_grad():
                for sid, (sent, lengths, labels) in enumerate(val_loader):
                    loss, corr = calc_loss(
                        model, sent, lengths, labels, criterion, device=device
                    )
                    val_loss += loss.item()
                    val_correct += corr.item()

            # Print loss and accuracy
            train_loss = running_loss / len(train_set)
            val_loss = val_loss / len(val_set)
            print(f" train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            print(
                f" train_acc={correct / len(train_set):.4f}, val_acc={val_correct/len(val_set):.4f}"
            )

            # Early stoppage:
            if val_loss < min_loss:
                min_loss = val_loss
            elif epoch > MIN_EPOCHS and val_loss > min_loss + EARLY_STOP_THRESHOLD:
                torch.save(model.state_dict(), f"{OUT_DIR}SD_LSTM_ep{epoch}.pt")
                print(f"Early stopping at epoch {epoch}.")
                break

            # Save checkpoint
            if epoch % CHKPT_INTERVAL == 0 and epoch > 0:
                torch.save(model.state_dict(), f"{OUT_DIR}SD_LSTM_ep{epoch}.pt")

            # Update LR
            scheduler.step()
        end = time.time()
        print(f"Training time: {datetime.timedelta(seconds=end-start)}")

        # Save final model
        torch.save(model.state_dict(), f"{OUT_DIR}SD_LSTM_final.pt")

    # Test model
    if args.test:
        state_dict = torch.load(f"{OUT_DIR}SD_LSTM_final.pt")
        model = LSTMBinaryClassifier(len(w2i), EMBED_SIZE, HIDDEN_SIZE)
        model.load_state_dict(state_dict)
        model = model.to(device)
    test_set = SarcasmDataset.load_json(DATA_DIR + TEST_FN, w2i)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)
    pred = []
    y = []
    with torch.no_grad():
        for sid, (sent, lengths, labels) in enumerate(test_loader):
            sent = sent.to(device=device)
            labels = labels.to(device=device)
            outputs, _ = model(sent, lengths)
            pred.append(torch.argmax(outputs, dim=1).cpu().numpy())
            y.append(labels.cpu().numpy())
    print(f"{len(test_set)} samples tested.")
    pred = np.concatenate(pred)
    y = np.concatenate(y)

    cm = confusion_matrix(y, pred)
    evaluate_confusion_matrix(cm)
