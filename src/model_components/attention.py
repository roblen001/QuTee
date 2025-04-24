"""Contains the attention layer architecture
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor

import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    """
    Implements multi-head self-attention.

    This class creates multiple parallel attention heads, each with its own learnable 
    projection of queries, keys, and values. The outputs from all heads are concatenated 
    and passed through a final linear projection layer to match the model's embedding dimension.

    Having multiple smaller attention blocks allows the model to learn different patterns in the data.

    Args:
        num_heads (int): Number of parallel attention heads.
        embedding_dim (int): Total output dimensionality after concatenating all heads.
                             Should be equal to num_heads * head_dim.
        dropout_prob (float): Dropout probability applied after the final projection layer.
    """
    def __init__(self, num_heads: int, embedding_dim: int, dropout_prob: float = 0.2):
        super().__init__() 
        # Dimensionality of each attention head (i.e., the size of each Q/K/V vector).
        head_size = embedding_dim // num_heads
        self.attention_heads = nn.ModuleList([Head(head_size, embedding_dim=embedding_dim) for _ in range(num_heads)])
        self.output_projection = nn.Linear(num_heads * head_size, embedding_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for multi-head attention.

        Args:
            input_ids (torch.Tensor): Input tensor of shape (batch_size, sequence_length, embedding_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, embedding_dim).
        """
        # Apply each attention head in parallel and concatenate the results
        concatenated_output = torch.cat([head(input_ids) for head in self.attention_heads], dim=-1)
        
        # Project the concatenated result back to the embedding dimension and apply dropout
        projected_output = self.output_projection(concatenated_output)
        return self.dropout(projected_output)

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, context_size: int = 1024, dropout: float = 0.2, embedding_dim : int = 64):
        super().__init__()
        """
        Initializes a single attention head used in a multi-head self-attention mechanism.

        Args:
            head_size (int): The dimensionality of the attention head (i.e., the size of the output vector per token).
            context_size (int, optional): The maximum sequence length (number of tokens) the attention can attend over.
                                        Defaults to 1024.
            dropout (float, optional): Dropout rate applied to the attention weights for regularization. Defaults to 0.3.

        The class typically includes learnable linear projections for keys, queries, and values, 
        as well as a mechanism to compute scaled dot-product attention within the specified context size.
        """

        # query token asks 'here's what I am looking for'
        # this might be a noun looking for how words might effect it
        self.query = nn.Linear(embedding_dim, head_size, bias=False)

        # key token responds 'here's what I offer' 
        # this could be adjectives which would have a large impact on a noun
        # thus would attend to the noun
        self.key = nn.Linear(embedding_dim, head_size, bias=False)

        # here's what I comminucate' 
        # this is finds out how the adjective impacts the nouns embedding (how it will move it in the embedding vector space)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)
        
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
        # makes it a decoder block
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