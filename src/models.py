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