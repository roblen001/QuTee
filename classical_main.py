"""
Building the simplest Transformer model possible.
"""
import torch

from src.data_processing.data_processing import prepare_data
from src.utils.data_processing_tools import decode
from src.models.GPT import GPT

batched_data, tokenizer = prepare_data(
    path="data/raw/shakespeare.txt",
    batch_size=32,
    context_size=128
)


model = GPT(tokenizer=tokenizer, context_size=8)
# Adam uses 2 moving averages to help optimize
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
model.training_loop(batched_data, optimizer, epochs=5000)
print(decode(model.generate(torch.zeros((1, 1), dtype=torch.int64), max_new_tokens=500), tokenizer)) 