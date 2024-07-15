from typing import Literal

import torch
import torch.nn as nn

class RNNClassifier(nn.Module):
    def __init__(self, 
                 output_dim,
                 vocab_size: int=10000, 
                 rnn_type: Literal['RNN', 'LSTM', 'GRU']='RNN',
                 embedding_dim: int=1024, 
                 hidden_dim: int=512, 
                 num_layers: int=6, 
                 bidirectional: bool=True, 
                 dropout: float=0.5):
        
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if rnn_type == 'RNN':
            rnn_obj = nn.RNN
        elif rnn_type == 'LSTM':
            rnn_obj = nn.LSTM
        elif rnn_type == 'GRU':
            rnn_obj = nn.GRU
        else:
            raise ValueError(f"rnn_type must be one of ['RNN', 'LSTM', 'GRU'], got {rnn_type}")
        
        self.rnn = rnn_obj(embedding_dim, 
                          hidden_dim, 
                          num_layers=num_layers, 
                          bidirectional=bidirectional, 
                          dropout=dropout, 
                          batch_first=True)
        
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        _, hidden = self.rnn(embedded)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])
        return self.fc(hidden)