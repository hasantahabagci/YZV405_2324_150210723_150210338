# Purpose: This file contains the utility functions for the data loading and preprocessing tasks.
# Prepared for YZV405E Natural Language Processing Istanbul Technical University
# Authors: Muhammet Serdar NAZLI, Hasan Taha BAĞCI

import re 
import pandas as pd

def tr_upper(self):
    self = re.sub(r"i", "İ", self)
    self = re.sub(r"ı", "I", self)
    self = re.sub(r"ç", "Ç", self)
    self = re.sub(r"ş", "Ş", self)
    self = re.sub(r"ü", "Ü", self)
    self = re.sub(r"ğ", "Ğ", self)
    self = self.upper() # for the rest use default upper
    return self


def tr_lower(self):
    self = re.sub(r"İ", "i", self)
    self = re.sub(r"I", "ı", self)
    self = re.sub(r"Ç", "ç", self)
    self = re.sub(r"Ş", "ş", self)
    self = re.sub(r"Ü", "ü", self)
    self = re.sub(r"Ğ", "ğ", self)
    self = self.lower() # for the rest use default lower
    return self


def split_long_sentences(sentence, max_length=500):
    # Splits a sentence into multiple chunks each with a maximum length of `max_length`.
    return [sentence[i:i+max_length] for i in range(0, len(sentence), max_length)]

# Apply the splitting function to each sentence and expand the DataFrame
def expand_sentences(df):
    new_rows = []
    for index, row in df.iterrows():
        if len(row['Sentence']) > 500:
            # Split the long sentence
            parts = split_long_sentences(row['Sentence'])
            for part in parts:
                new_row = row.copy()
                new_row['Sentence'] = part
                new_rows.append(new_row)
        else:
            new_rows.append(row)
    return pd.DataFrame(new_rows) 

def split_by_newline(sentence):
    return [line for line in sentence.split('\n') if line.strip()]


def split_exact_row(row_index, train_df):
    if row_index in train_df.index:
        sentence_to_split = train_df.loc[row_index, 'Sentence']
        new_sentences = split_by_newline(sentence_to_split)

        new_rows = pd.DataFrame([{'ID': train_df.loc[row_index, 'ID'], 'Sentence': sent} for sent in new_sentences if sent.strip()])

        train_df = pd.concat([train_df, new_rows], ignore_index=True)

        train_df = train_df.drop(index=row_index).reset_index(drop=True)
    else:
        print(f"Row index {row_index} not found in DataFrame.")
    return train_df



def preprocess_text(text, conversion_dict, to_removed_chars, mapping, poison=False, is_test=False, sos_eos_tokens=False):
    # Convert to lowercase
    text = tr_lower(text) if not is_test else text.lower()

    if poison:
        conversion_map = {
            'ü': 'u',
            'ö': 'o',
            'ğ': 'g',  
            'ş': 's',
            'ç': 'c',
            'ı': 'i',
        }
        # Convert each character using the map
        text = ''.join(conversion_map.get(char, char) for char in text)
        
    # Apply additional mappings
    for key, value in mapping.items():
        text = text.replace(key, value)
    
    # Remove unwanted characters and sequences
    for sequence in to_removed_chars:
        text = text.replace(sequence, '')
    
    # Strip leading and trailing whitespaces
    text = text.strip()
    
    # Convert characters to tokens using a dictionary
    tokens = [conversion_dict.get(char, conversion_dict['UNK']) for char in text]
    if sos_eos_tokens:
        # EOS token
        tokens.append(conversion_dict['[EOS]'])
        # SOS token 
        tokens = [conversion_dict['[SOS]']] + tokens
    
    return tokens