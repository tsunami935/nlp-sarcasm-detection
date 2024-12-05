import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
from nltk.tokenize import word_tokenize

class SarcasmDataset(Dataset):
    def __init__(self, json_file):
        # Load data from the JSON file
        self.data = []
        with open(json_file, 'r') as f:
            for line in f:
                self.data.append(json.loads(line.strip()))

        # Get max token length for padding
        max_length = 0
        for item in self.data:
            tokens = item['tokens']
            if len(tokens) >= max_length: max_length = len(tokens)
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the individual sample
        item = self.data[idx]
        label = torch.tensor(item['is_sarcastic'])
        tokens = item['tokens']
        
        # crop for max length (shouldnt be needed but just in case)
        tokens = tokens[:self.max_length]
        
        # Pad the tokens to the max length if needed
        padding_length = self.max_length - len(tokens)
        tokens.extend(['<pad>'] * padding_length)  # Padding with '<pad>'

        # Return the input_ids, attention_mask, and label as a tuple
        return tokens, label

# Example usage
dataset = SarcasmDataset(json_file="test_tokenized.json")

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Iterate through the DataLoader DEBUG
#for batch in dataloader:
#   print(batch)
#  break
#print(dataset.max_length)

