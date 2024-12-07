from collections.abc import Iterator
import json
from pathlib import Path
import sys

import numpy as np
from gensim.models import KeyedVectors, Word2Vec

from data_normalize import normalize_sentence


class Sentences:
    def __init__(self, dataset_file: str | Path) -> None:
        self.dataset_file = dataset_file

    def __iter__(self) -> Iterator[list[str]]:
        with open(self.dataset_file, 'r') as file:
            while (entry := file.readline()):
                data = json.loads(entry)
                yield normalize_sentence(data['headline'])

def main() -> int:
    DATA_DIR = Path(__file__).parent.parent / 'data'
    GOOGLE_WORD2VEC = DATA_DIR / 'GoogleNews-vectors-negative300.bin'
    DATASET_JSON = DATA_DIR / 'Sarcasm_Headlines_Dataset.json'
    OUTPUT_WORD2VEC = DATA_DIR / 'w2v-headline-embeddings-300d.bin'

    print('Initializing model...')
    headlines = Sentences(DATASET_JSON)
    model = Word2Vec(vector_size=300, window=5, min_count=1)

    print('Building vocabulary...')
    model.build_vocab(headlines)

    print('Loading pretrained word2vec embeddings...')
    model.wv.vectors_lockf = np.ones(len(model.wv))
    model.wv.intersect_word2vec_format(GOOGLE_WORD2VEC, binary=True)

    print('Learning embeddings...')
    model.train(headlines,
                total_examples=model.corpus_count,
                epochs=model.epochs)

    print(f'Saving embeddings to {OUTPUT_WORD2VEC}...')
    model.wv.save(str(OUTPUT_WORD2VEC))

    return 0

if __name__ == '__main__':
    sys.exit(main())


# sources:
# https://code.google.com/archive/p/word2vec/
# https://radimrehurek.com/gensim/models/word2vec.html
# https://rare-technologies.com/word2vec-tutorial/
# https://phdstatsphys.wordpress.com/2018/12/27/word2vec-how-to-train-and-update-it/
