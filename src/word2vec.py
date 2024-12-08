from argparse import ArgumentParser
from collections.abc import Iterator
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
from gensim.models import Word2Vec

from data_normalize import normalize_sentence


class Sentences:
    def __init__(self, dataset_file: str | Path) -> None:
        self.dataset_file = dataset_file

    def __iter__(self) -> Iterator[list[str]]:
        with open(self.dataset_file, 'r') as file:
            while (entry := file.readline()):
                data = json.loads(entry)
                yield normalize_sentence(data['headline'])


def w2v_train(
    vector_size: int = 300,
    epochs: int = 50,
    window: int = 5,
    min_count: int = 1,
    lockf: float = 0.0,
) -> None:
    DATA_DIR = Path(__file__).parent.parent / 'data'
    GOOGLE_WORD2VEC = DATA_DIR / 'GoogleNews-vectors-negative300.bin'
    DATASET_JSON = DATA_DIR / 'Sarcasm_Headlines_Dataset.json'
    OUTPUT_WORD2VEC = \
        DATA_DIR / f'w2v-headline-embeddings-300d-e{epochs}-lockf{lockf}.bin'

    print(f'-- w2v_train({vector_size=}, {window=}, {min_count=}, {epochs=}, '
          f'{lockf=}) --')

    print('Initializing model...')
    headlines = Sentences(DATASET_JSON)
    model = Word2Vec(vector_size=vector_size, window=window, epochs=epochs,
                     min_count=min_count)

    print('Building vocabulary...')
    model.build_vocab(headlines)

    print('Loading pretrained word2vec embeddings...')
    model.wv.vectors_lockf = np.ones((len(model.wv),), dtype=np.float32)
    model.wv.intersect_word2vec_format(GOOGLE_WORD2VEC, binary=True,
                                       lockf=lockf)

    print('Learning embeddings...')
    model.train(headlines,
                total_examples=model.corpus_count,
                epochs=model.epochs)

    print(f'Saving embeddings to {OUTPUT_WORD2VEC}...')
    model.wv.save(str(OUTPUT_WORD2VEC))


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument('-m', '--min-count', type=int)
    parser.add_argument('-w', '--window', type=int)
    parser.add_argument('-e', '--epochs', type=int)
    parser.add_argument('-l', '--lockf', type=float)
    args = parser.parse_args()

    # this is kinda dumb but it works!
    kwargs = {k: v for k, v in args._get_kwargs() if v is not None}
    if kwargs:
        w2v_train(**kwargs)
        return 0

    # if no arguments were passed...
    epochs = (5, 50, 100, 200)
    lockfs = (0.0, 1.0)
    for epoch, lockf in product(epochs, lockfs):
        w2v_train(epochs=epoch, lockf=lockf)
    return 0


if __name__ == '__main__':
    sys.exit(main())


# sources:
# https://code.google.com/archive/p/word2vec/
# https://radimrehurek.com/gensim/models/word2vec.html
# https://rare-technologies.com/word2vec-tutorial/
# https://phdstatsphys.wordpress.com/2018/12/27/word2vec-how-to-train-and-update-it/
