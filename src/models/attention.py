"""Contains the attention layer architecture
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, context_size: int = 1024, dropout: float = 0.3):
        super().__init__()
        # TODO: add a constant file to decided some of the main parameters like embedding dimension
        # or generate a simple pipeline function

        # query token asks 'here's what I am looking for'
        # this might be a noun looking for how words might effect it
        self.query = nn.Linear(64, head_size, bias=False)

        # key token responds 'here's what I offer' 
        # this could be adjectives which would have a large impact on a noun
        # thus would attend to the noun
        self.key = nn.Linear(64, head_size, bias=False)

        # here's what I comminucate' 
        # this is finds out how the adjective impacts the nouns embedding (how it will move it in the embedding vector space)
        self.value = nn.Linear(64, head_size, bias=False)
        
        # since we are dealing with a model whose job is to generate
        # we are dealing with decoder moodel this means we cannot have future tokens 
        # impact the generation of past tokens
        # this masks prevents this by 0 ing all the future tokens
        # you might not always want to do this. If you want a encoder model (sentiment analysis classifier)
        # you want to see how all tokens attend to each other
        self.register_buffer('tril', torch.tril(torch.ones(context_size, context_size)))

        # classic way of preventing overfitting by turning off nodes during training
        # so the model doesn't become dependant on any single node.
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_tensor: Tensor) -> Tensor:
        """
        Compute masked self-attention for a single attention head.

        Args:
            input_tensor (Tensor): Input embeddings of shape (batch_size, sequence_length, embedding_dim_per_head)

        Returns:
            Tensor: Output of attention head with shape (batch_size, sequence_length, embedding_dim_per_head)
        """
        # Unpack input shape
        batch_size, sequence_length, head_size = input_tensor.shape  # (B, T, C)

        # Project input tensor into key and query representations
        key_tensor = self.key(input_tensor)     # shape: (B, T, C)
        query_tensor = self.query(input_tensor) # shape: (B, T, C)

        # Compute raw attention weights using scaled dot-product
        # q @ k^T → (B, T, C) @ (B, C, T) = (B, T, T) attention score matrix
        attention_weights = query_tensor @ key_tensor.transpose(-2, -1)
        attention_weights *= head_size ** -0.5  # scale for stability

        # Apply causal (lower triangular) mask to prevent attending to future tokens
        # tril is (T, T), we broadcast it to all batches
        attention_weights = attention_weights.masked_fill(self.tril[:sequence_length, :sequence_length] == 0, float('-inf'))

        # Normalize attention weights across each row (i.e., each token attends over past tokens)
        attention_weights = F.softmax(attention_weights, dim=-1)  # shape: (B, T, T)
        attention_weights = self.dropout(attention_weights)       # apply dropout to prevent overfitting

        # Project input tensor into value representation
        value_tensor = self.value(input_tensor)  # shape: (B, T, C)

        # Aggregate weighted values using attention weights
        # (B, T, T) @ (B, T, C) = (B, T, C)
        output_tensor = attention_weights @ value_tensor

        return output_tensor