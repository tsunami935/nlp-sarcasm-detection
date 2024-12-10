from __future__ import annotations
from numpy.typing import NDArray
from collections import defaultdict
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from common import *


class SarcasmDataset(Dataset):
    def __init__(
        self, tokens: list[NDArray], labels: list[int], w2i: defaultdict[str, int]
    ):
        super().__init__()
        tokens = [torch.LongTensor(tk) for tk in tokens]
        self.tokens = pad_sequence(tokens, batch_first=True, padding_value=w2i[TK_END])
        self.lengths = torch.LongTensor([len(tk) for tk in tokens])
        self.labels = torch.LongTensor(labels)

    @classmethod
    def load_json(cls, fn: str, w2i: defaultdict[str, int]) -> SarcasmDataset:
        df = pd.read_json(fn, lines=True)
        df["tokens"] = df["tokens"].apply(lambda s: vectorize_tokens(s, w2i))
        return cls(df["tokens"], df["is_sarcastic"], w2i)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx], self.lengths[idx], self.labels[idx]
