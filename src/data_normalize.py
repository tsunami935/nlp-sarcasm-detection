from __future__ import annotations
import pandas as pd
from common import *

import contractions

import nltk
from nltk.tokenize import word_tokenize

try:
    nltk.find("tokenizers/punkt/english.pickle")
except LookupError:
    nltk.download("punkt")


def tokenize_single_quote(tokens: list[str]) -> list[str]:
    size = len(tokens)
    start = None
    end = None
    i = 0
    while i < size:
        if tokens[i].startswith("'"):
            start = i
        if tokens[i].endswith("'") and start is not None:
            end = i
            prev = tokens[:start]
            curr = (
                ["'", tokens[start][1:]]
                + tokens[start + 1 : end]
                + [tokens[end][:-1], "'"]
                if start != end
                else ["'", tokens[i][1:-1], "'"]
            )
            next = tokenize_single_quote(tokens[i + 1 :]) if i + 1 < size else []
            return prev + curr + next
        if tokens[i].endswith(("':", "',")) and start is not None:
            end = i
            prev = tokens[:start]
            curr = (
                ["'", tokens[start][1:]]
                + tokens[start + 1 : end]
                + [tokens[end][:-2], "'", tokens[end][-1]]
                if start != end
                else ["'", tokens[i][1:-2], "'", tokens[end][-1]]
            )
            next = tokenize_single_quote(tokens[i + 1 :]) if i + 1 < size else []
            return prev + curr + next
        i += 1
    return tokens


def tokenize_hyphen(tokens: list[str]) -> list[str]:
    hyphen = " " + TK_HYPH + " "
    size = len(tokens)
    i = 0
    while i < size:
        if "-" in tokens[i]:
            subs = hyphen.join(tokens[i].split("-")).split(" ")
            l = len(subs) - 1
            tokens[i : i + 1] = subs
            size += l
            i += l
        i += 1
    return tokens


def normalize_sentence(sentence: str) -> str:
    # Turn all lowercase
    # Turn contractions to canonical form
    # Tokenize sentences
    tks = tokenize_single_quote(sentence.lower().split(" "))
    tks = tokenize_hyphen(word_tokenize(contractions.fix(" ".join(tks))))
    # Add start and end tokens
    return [TK_START] + tks + [TK_END]


if __name__ == "__main__":
    train_df = pd.read_json("../data/train.json", lines=True)
    val_df = pd.read_json("../data/val.json", lines=True)
    test_df = pd.read_json("../data/test.json", lines=True)

    for df in [train_df, val_df, test_df]:
        df["tokens"] = df["headline"].apply(normalize_sentence)
        print(df.head(5))

    train_df.to_json("../data/train_tokenized.json", orient="records", lines=True)
    val_df.to_json("../data/val_tokenized.json", orient="records", lines=True)
    test_df.to_json("../data/test_tokenized.json", orient="records", lines=True)
