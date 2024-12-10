import pandas as pd

from common import * 

def build_vocabulary(
    df: DataFrame, unk_threshold: int = 5
) -> tuple[list[str], dict[str, int]]:
    # Get raw counts
    vocab = defaultdict(int)
    for _, tk_list in df["tokens"].items():
        for tk in tk_list:
            if tk not in [TK_HYPH, TK_UNK]:
                vocab[tk] += 1

    # Any word occurring less than 5 times becomes unknown
    word_list = [TK_HYPH, TK_UNK]
    word2ind = {word_list[i]: i for i in range(len(word_list))}
    for word, count in vocab.items():
        if count >= unk_threshold:
            word2ind[word] = len(word_list)
            word_list.append(word)
    return word_list, word2ind

if __name__ == "__main__":
    import json
    DATA_DIR = "../data/"
    TRAIN_FN = "train_tokenized.json"
    train_df = pd.read_json(DATA_DIR + TRAIN_FN, lines=True)
    vocab, word2ind = build_vocabulary(train_df)
    with open(DATA_DIR + "vocab.txt", "w") as fout:
        fout.write("\n".join(vocab) + "\n")
    with open(DATA_DIR + "word2ind.json", "w") as fout:
        fout.write(json.dumps(word2ind))