# @Author: M.Serdar NAZLI and Hasan Taha BAĞCI, Istanbul Technical University. 
# @Date: 27/04/2024 
# Prepared for the NLP project.

import torch 
from torch.utils.data import Dataset

class SentenceDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        original_tokens = torch.tensor(self.data.iloc[idx]['Sentence_tokens'], dtype=torch.long)
        poisoned_tokens = torch.tensor(self.data.iloc[idx]['Poisoned_tokens'], dtype=torch.long)
        return poisoned_tokens, original_tokens


class SentenceDatasetTest(Dataset):
    def __init__(self, sentences_tokens):
        self.sentences_tokens = sentences_tokens

    def __len__(self):
        return len(self.sentences_tokens)

    def __getitem__(self, idx):
        return torch.tensor(self.sentences_tokens[idx], dtype=torch.long)