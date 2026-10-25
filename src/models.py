from typing import Any

import torch
import torch.nn as nn

class ANNModel(nn.Module):
    def __init__(self,input_size,hidden1=256,hidden2=128,dropout=0.3):
        super().__init__()
        self.network=nn.Sequential(nn.Linear(input_size,hidden1),
                                   nn.ReLU(),
                                   nn.Dropout(dropout),
                                   nn.Linear(hidden1,hidden2),
                                   nn.ReLU(),
                                   nn.Dropout(dropout),
                                   nn.Linear(hidden2,1))
    
    def forward(self,x):
        return self.network(x).squeeze(1)
    
    

class LSTMModel(nn.Module):
    def __init__(self,input_size=100,hidden_size=64,num_layers=1,dropout=0.3):
        super().__init__()
        self.lstm=nn.LSTM(input_size,hidden_size,num_layers=num_layers,batch_first=True)
        self.drop=nn.Dropout(dropout)
        self.fc=nn.Linear(hidden_size,1)
    def forward(self,x):
        out,_=self.lstm(x)
        last=out[:,-1,:]
        return self.fc(self.drop(last)).squeeze(1)