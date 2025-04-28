"""Loads model and allows drawing inference from it."""

import torch
from src.models.gpt import GPT
from src.utils.data_processing_tools import decode

def encode_text_to_tensor(
    text: str,
    tokenizer: dict,
    context_size: int,
    pad_token: int = 0
) -> torch.Tensor:
    """
    Converts a raw string into a 1×context_size tensor of token IDs.

    - Falls back to `pad_token` for any character not in the vocab.
    - Truncates longer text to last `context_size` tokens.
    - Left-pads shorter text with `pad_token`.
    """
    enc_map = tokenizer['encoder']
    token_ids = [enc_map.get(ch, pad_token) for ch in text]  # safe lookup
    if len(token_ids) > context_size:
        token_ids = token_ids[-context_size:]
    else:
        token_ids = [pad_token] * (context_size - len(token_ids)) + token_ids
    return torch.tensor([token_ids], dtype=torch.int64)


# Load your fine-tuned model
model: GPT = torch.load(
    "data/finetuning/finetune_2025-04-27_14-57-54/model_full_finetuned.pt",
    map_location="cpu"
)
model.eval()  # disable dropout, batchnorm, etc.

# Few-shot + instruction prompt
prompt = """Write a story about a knight."""

# Encode and generate
input_ids = encode_text_to_tensor(
    prompt,
    tokenizer=model.tokenizer,
    context_size=model.context_size
).to(model.device)

with torch.no_grad():
    out = model.generate(input_ids, max_new_tokens=300)

print(decode(out[0].tolist(), model.tokenizer))
