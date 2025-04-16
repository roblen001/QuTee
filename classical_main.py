"""
Building the simplest Transformer model possible.
"""
from src.data_processing.data_processing import prepare_data

batched_data, tokenizer = prepare_data(
    path="data/raw/shakespeare.txt",
    batch_size=32,
    context_size=16
)
