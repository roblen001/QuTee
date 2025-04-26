"""
Building the simplest Transformer model possible.
"""

import os
import torch
import datetime

from src.data_processing.data_processing import prepare_data
from src.utils.data_processing_tools import decode
from src.models.gpt import GPT
from src.model_components.feedforward import ClassicalFeedForward, QuantumFeedForward

from src.configs.constants import (
    CONTEXT_SIZE,
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    MAX_NEW_TOKENS,
    NUM_OF_BLOCKS,
    NUM_HEADS,
    EMBEDDING_DIM,
)

# Prepare data
batched_data, tokenizer = prepare_data(
    path="data/raw/shakespeare.txt",
    batch_size=BATCH_SIZE,
    context_size=CONTEXT_SIZE
)

# Select feedforward class
feedforward_cls = ClassicalFeedForward
ff_kwargs = {}  # No kwargs for classical FF atm

# Create model
model = GPT(
    tokenizer=tokenizer,
    context_size=CONTEXT_SIZE,
    num_heads=NUM_HEADS,
    n_layer=NUM_OF_BLOCKS,
    embedding_dim=EMBEDDING_DIM,
    feedforward_cls=feedforward_cls,
    ff_kwargs=ff_kwargs,
)

# adam uses two moving averages to help optimize
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
results_base_dir = "experimental_results"
results_dir = os.path.join(results_base_dir, f"run_{timestamp}")
os.makedirs(results_dir, exist_ok=True)

model.training_loop(batched_data, optimizer, epochs=EPOCHS, results_dir=results_dir)

params_log_path = os.path.join(results_dir, "run_parameters.txt")
with open(params_log_path, "w", encoding="utf-8") as f:
    f.write("Run Parameters:\n")
    f.write(f"CONTEXT_SIZE: {CONTEXT_SIZE}\n")
    f.write(f"BATCH_SIZE: {BATCH_SIZE}\n")
    f.write(f"LEARNING_RATE: {LEARNING_RATE}\n")
    f.write(f"EPOCHS: {EPOCHS}\n")
    f.write(f"MAX_NEW_TOKENS: {MAX_NEW_TOKENS}\n")
    f.write(f"NUM_OF_BLOCKS: {NUM_OF_BLOCKS}\n")
    f.write(f"NUM_HEADS: {NUM_HEADS}\n")
    f.write(f"EMBEDDING_DIM: {EMBEDDING_DIM}\n")
    f.write(f"FEEDFORWARD_CLASS: {feedforward_cls.__name__}\n")
    f.write(f"Feedforward kwargs: {ff_kwargs}\n")
print(f"Saved run parameters to {params_log_path}")

# Text generation
print(
    decode(
        model.generate(
            torch.zeros((1, 1), dtype=torch.int64),
            max_new_tokens=MAX_NEW_TOKENS
        ),
        tokenizer
    )
)
