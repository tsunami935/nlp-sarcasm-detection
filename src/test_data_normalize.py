from itertools import chain
import json
import unittest

from common import TK_END, TK_HYPH, TK_START
from data_normalize import normalize_sentence


class DataNormalizeTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nice_examples: dict[str, list[str]] = {
            "5 ways to file your taxes with less stress": [
                TK_START, '5', 'ways', 'to', 'file', 'your', 'taxes',
                'with', 'less', 'stress', TK_END
            ],
            "the quick brown fox jumped over the lazy dog.": [
                TK_START, "the", "quick", "brown", "fox", "jumped", "over",
                "the", "lazy", "dog", ".", TK_END
            ],
            "eat your veggies: 9 deliciously different recipes": [
                TK_START, "eat", "your", "veggies", ":", "9", "deliciously",
                "different", "recipes", TK_END
            ]
        }
        self.evil_examples: dict[str, list[str]] = {
            "'tis a day for joy, the man said, 'ahoy, and sieze the day!'": [
                TK_START, "it", "is", "a", "day", "for", "joy", ",", "the",
                "man", "said", ",", "'", "ahoy", ",", "and", "sieze", "the",
                "day", "!", "'", TK_END
            ],
            "'the big dark': series of storms stretching from China to U.S. "
            "batters northwest": [
                TK_START, "'", "the", "big", "dark", "'", ":", "series", "of",
                "storms", "stretching", "from", "china", "to", "u.s.",
                "batters", "northwest", TK_END
            ],
            "how real is 'macromentum'?": [
                TK_START, "how", "real", "is", "'", "macromentum", "'", "?",
                TK_END
            ],
            "toyota launches 'back to the future'-themed ad campaign": [
                TK_START, "toyota", "launches", "'", "back", "to", "the",
                "future", "'", TK_HYPH, "themed", "ad", "campaign", TK_END
            ],
            # thank The Onion for this next one...
            "'hot 'n' nasty butt cum chixx' to appear as 'creative concepts' "
            "on credit-card bill": [
                TK_START, "'", "hot", "'n'", "nasty", "butt", "cum", "chixx",
                "'", "to", "appear", "as", "'", "creative", "concepts", "'",
                "on", "credit", TK_HYPH, "card", "bill", TK_END
            ],
            "unnamed new gas station struggling to find 'stop 'n go' "
            "variant": [
                TK_START, "unnamed", "new", "gas", "station", "struggling",
                "to", "find", "'", "stop", "'n", "go", "'", "variant", TK_END
            ]
        }

    def test_nice_sentences(self):
        for sent, sent_tokenized in self.nice_examples.items():
            self.assertEqual(normalize_sentence(sent), sent_tokenized)

    def test_evil_sentences(self):
        for sent, sent_tokenized in self.evil_examples.items():
            self.assertEqual(normalize_sentence(sent), sent_tokenized)


def check_bad_sentences():
    with open('../data/Sarcasm_Headlines_Dataset.json', 'r') as f:
        data = [json.loads(line.strip()) for line in f]

    def predicate(tok: str) -> bool:
        decades = (f"'{x:02d}s" for x in range(0, 100, 10))
        years = (f"'{x:02d}" for x in range(0, 100))
        contractions = ("'nt", "'s", "'ve", "'d")
        real_words = ("'round", "'til", "'tis", "'n", "'n'")
        edge_cases = chain("''", contractions, decades, years, real_words)
        conditions = (tok.startswith("'"), len(tok) > 1, tok != "''",
                      tok not in edge_cases)
        return all(conditions)

    for d in data:
        normalized = normalize_sentence(d['headline'])
        if any(predicate(s) for s in normalized):
            print(f'sentence: {d['headline']}')
            print(f'normalized: {normalized}')
