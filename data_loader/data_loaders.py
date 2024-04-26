# @Author: M.Serdar NAZLI, Istanbul Technical University. 
# @Date: 27/04/2024 
# Prepared for the NLP project.

from torch.utils.data import Dataset 
from base import BaseDataLoader
import csv 


class DiacritizationDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  
            for row in reader:
                _, sentence = row
                self.data.append(sentence)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    

def create_data_loaders(file_path, batch_size, shuffle=True, validation_split=0.0, num_workers=0):
    dataset = DiacritizationDataset(file_path)
    data_loader = BaseDataLoader(dataset, batch_size, shuffle, validation_split, num_workers)
    valid_data_loader = data_loader.split_validation()
    return data_loader, valid_data_loader