"""Simple feedforward multilayer perceptron (MLP). 
This is a feedfoward neural network which has linear layers followed by non linearity (activation funcitons).
This portion of transformer models is used to memory facts in it's training datasets. 
(Who is the basket player Lebron James ...)
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor

class FeedForward(nn.Module):
    """This is a feedfoward neural network which has linear layers followed by non linearity (activation funcitons)."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        # the activation funciton let's the model fit non linear relationships in the data
        self.net = nn.Sequential(
            # the 'Attention Is All You Need' paper uses mutiple of 4 which will add
            # greater computation to our system, and growing our layer
            nn.Linear(embedding_dim, 4*embedding_dim),
            nn.ReLU(),
            # projection layer going back into the residual pathway
            # to decrease optimization issues with deep models.
            nn.Linear(4*embedding_dim, embedding_dim),

        )
    
    def forward(self, input_ids: Tensor):
        return self.net(input_ids)