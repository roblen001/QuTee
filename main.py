"""
Building the simplest Transformer model possible.
"""
import torch

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

batched_data, tokenizer = prepare_data(
    path="data/raw/shakespeare.txt",
    batch_size=BATCH_SIZE,
    context_size=CONTEXT_SIZE
)

feedforward_cls = ClassicalFeedForward
# no ff_kwargs for classical feedforward atm
ff_kwargs = {}                

model = GPT(
    tokenizer=tokenizer,
    context_size=CONTEXT_SIZE,
    num_heads=NUM_HEADS,
    n_layer=NUM_OF_BLOCKS,
    embedding_dim=EMBEDDING_DIM,
    feedforward_cls=feedforward_cls,
    ff_kwargs=ff_kwargs,
)

# Adam uses 2 moving averages to help optimize
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
model.training_loop(batched_data, optimizer, epochs=EPOCHS)

print(
    decode(
        model.generate(
            torch.zeros((1, 1), dtype=torch.int64),
            max_new_tokens=MAX_NEW_TOKENS
        ),
        tokenizer
    )
)
