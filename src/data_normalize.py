from __future__ import annotations
from collections.abc import Iterable
from itertools import chain
import sys


import contractions
import nltk
from nltk.tokenize import word_tokenize
import pandas as pd

from common import TK_END, TK_HYPH, TK_START

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


def tokenize_edge_cases(tokens: list[str]) -> Iterable[str]:
    iter_tokens = enumerate(tokens)
    n_tokens = len(tokens)

    for idx, token in iter_tokens:
        # undo mangling by contractions library
        if token == "you.s":
            token = "u.s"
            if idx < (n_tokens - 1) and tokens[idx + 1] == ".":
                _ = next(iter_tokens)  # advance the iterator to skip the "."
                token += "."
        # sometimes "'n" is written "'n'". try to fix this.
        elif token == "'n" and idx < (n_tokens - 1) and tokens[idx + 1] == "'":
            _ = next(iter_tokens)  # advance the iterator to skip the "'"
            token += "'"
        yield token



SQ_WORDS = ("'round", "'til", "'tis", "'n", "'n'")
SQ_SUFFIXES = ("'s", "'nt", "'ve", "'d", "''")


# def tokenize_single_quote(tokens: list[str]) -> Iterable[str]:
#     lquote_stack: list[int] = []
#     quote_indices: list[int] = []
#     for idx, token in enumerate(tokens):
#         if "'" not in token or token in chain(SQ_WORDS, SQ_SUFFIXES):
#             continue

#         # assume these are right-quotes
#         if token == "'" or token.endswith("'"):
#             if lquote_stack:
#                 quote_indices.append(lquote_stack.pop())
#                 quote_indices.append(idx)
#             # assume rquote not following an lquote is actually an lquote
#             else:
#                 lquote_stack.append(idx)
#         elif token.startswith("'"):
#             lquote_stack.append(idx)

#     # if lquote_stack:
#     #     print(f"warning: sentence {tokens} might have unbalanced quotes!",
#     #           file=sys.stderr)
#     for idx, token in enumerate(tokens):
#         if token in ("'", "''"):
#             yield token
#             continue
#         if token.startswith("'") and token.endswith("'"):
#             for t in "'", token[1:-1], "'":
#                 yield t
#             continue
#         if idx in quote_indices:
#             if token.startswith("'"):
#                 for t in "'", token[1:]:
#                     yield t
#             elif token.endswith("'"):
#                 for t in token[:-1], "'":
#                     yield t
#             continue
#         yield token


def normalize_sentence(sentence: str) -> list[str]:
    # Turn all lowercase
    # Turn contractions to canonical form
    # sentences

    tks = tokenize_single_quote(sentence.lower().split(" "))
    tks = tokenize_hyphen(word_tokenize(contractions.fix(" ".join(tks))))
    tks = tokenize_edge_cases(tks)

    # Add start and end tokens
    return [TK_START, *tks, TK_END]


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
