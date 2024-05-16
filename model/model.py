# Description: This file contains the model class for the diacritization task.
# The model is a simple LSTM model with 4 layers and bidirectional LSTM. 
# The model is defined in the LSTMModel class. 
# The model takes the vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, and dropout_rate as input parameters.
# The model has an embedding layer, an LSTM layer, a dropout layer, and two fully connected layers for prediction.
# Prepared for YZV405E Natural Language Processing Istanbul Technical University

# Authors: Muhammet Serdar NAZLI, Hasan Taha BAĞCI 


import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math 


# Lots of old models we didn't include, but simple LSTM is the BEST!
"""class DiacritizationModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_encoder_layers=3, dim_feedforward=512, dropout=0.1):
        super(DiacritizationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 200, d_model))
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        self.vocab_size = vocab_size 

    def forward(self, src, src_mask=None):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src += self.pos_encoder[:, :src.size(1)]
        output = self.transformer_encoder(src, src_key_padding_mask=src_mask)
        output = self.fc_out(output)
        return output  # Should be [batch_size, seq_len, vocab_size]


    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz, device=self.embedding.weight.device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
        
class SlidingWindowBertModel(nn.Module):
    def __init__(self, vocab_size, max_len, window_size, hidden_size, num_layers, num_heads, dropout):
        super(SlidingWindowBertModel, self).__init__()
        self.window_size = window_size
        
        # BERT configuration
        config = BertConfig(
            vocab_size=vocab_size,
            max_position_embeddings=max_len,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
        )
        
        # BERT model for masked language modeling
        self.bert = BertForMaskedLM(config)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        # Training phase (masked language modeling)
        if labels is not None:
            outputs = self.bert(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            return loss
        
        # Inference phase (sliding window)
        else:
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)  # Add batch dimension if input is 1D
            
            batch_size, seq_len = input_ids.size()
            outputs = []
            
            for i in range(seq_len - self.window_size + 1):
                window_input_ids = input_ids[:, i:i+self.window_size]
                window_attention_mask = attention_mask[:, i:i+self.window_size] if attention_mask is not None else None
                
                window_outputs = self.bert(window_input_ids, attention_mask=window_attention_mask)
                window_logits = window_outputs.logits[:, self.window_size//2, :]
                outputs.append(window_logits)
            
            outputs = torch.stack(outputs, dim=1)
            return outputs
    """
    

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128, output_dim=None, num_layers=4, dropout_rate=0.1):
        super(LSTMModel, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Bi-directional LSTM layer(default # of layers is 4, dropout rate is 0.1)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected layers for prediction
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim) 
        self.fc2 = nn.Linear(hidden_dim, output_dim)      


    def forward(self, x):
        x = self.embedding(x)           # get embeddings
        x, _ = self.lstm(x)             # pass through LSTM
        x = self.dropout(x)             # apply dropout
        x = torch.relu(self.fc1(x))     # pass through first FC layer
        x = self.fc2(x)                 # pass through second FC layer
        return x
