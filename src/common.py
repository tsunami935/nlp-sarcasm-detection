from numpy.typing import NDArray

from collections import defaultdict

import numpy as np
from pandas import DataFrame


TK_START = "<s>"
TK_END = "</s>"
TK_UNK = "<UNK>"
TK_HYPH = "<HYPH>"


def count_tokens(sentence: list[str], word2ind: dict[str, int]) -> NDArray:
    v = np.zeros(len(word2ind), dtype=np.int32)
    i_UNK = word2ind[TK_UNK]
    for tk in sentence:
        v[word2ind.get(tk, i_UNK)] += 1
    return v


def vectorize_tokens(sentence: list[str], word2ind: dict[str, int]) -> NDArray:
    i_UNK = word2ind[TK_UNK]
    return np.array([word2ind.get(s, i_UNK) for s in sentence], dtype=np.int32)


def evaluate_confusion_matrix(cm: NDArray) -> None:
    # Calculate and print metrics based off confusion matrix
    precision = (cm[0][0]) / (cm[0][0] + cm[1][0])
    recall = (cm[0][0]) / (cm[0][0] + cm[0][1])
    print(f"Accuracy: {(cm[0][0] + cm[1][1]) / np.sum(cm):.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {(2 * precision * recall) / (precision + recall):.4f}")
