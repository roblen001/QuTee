"""
This module provides a one‑stop data preparation pipeline:

1. **Reading raw text** from disk.
2. **Tokenization**: mapping each character to a unique integer ID.
3. **Encoding** text into integer sequences.
4. **Train/test split** for next‑token prediction.
5. **Batch generation** with a fixed context window.

Concepts used:

- **Tokenizer (encoder/decoder)**:  
  A bijection between characters and integer IDs so we can feed text into a model.

- **Next‑token labels**:  
  We treat each position’s “next character” as the target, yielding many training signals per sequence.

- **Context window**:  
  We fix each input to `context_size` tokens, truncating longer sequences or padding shorter ones.

- **Batches**:  
  Grouping multiple context windows together for efficient parallel training.
"""
import torch

from ..utils.data_processing_tools  import read_textfile, create_tokenizer, encode, train_test_split, get_batch

def prepare_data(
    path: str,
    batch_size: int,
    context_size: int,
    split_ratio: float = 0.9
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, dict]]:
    """
    One‑stop function to read, tokenize, split, and batch raw text.

    Args:
        path: Path to the raw text file.
        batch_size: Number of samples per batch.
        context_size: Fixed length of each input sequence.
        split_ratio: Fraction of data used for training.

    Returns:
        - batched_data: Dict['train'/'test'] -> {'x':Tensor, 'y':Tensor}
        - tokenizer:    The mapping dict with 'encoder'/'decoder'

    Example:
        batched_data, tokenizer = prepare_data(
            "data/raw/shakespeare.txt",
            batch_size=32,
            context_size=16
        )
        print(batched_data['train']['x'].shape)  # -> (32, 16)
    """
    # 1. Read and tokenize
    raw = read_textfile(path)
    tokenizer = create_tokenizer(raw)

    # 2. Encode entire corpus into integer IDs
    ids = encode(raw, tokenizer)
    tensor_ids = torch.tensor(ids, dtype=torch.int64)

    # 3. Split into train/test streams
    splits = train_test_split(tensor_ids, threshold=split_ratio)

    # 4. Build one batch per split
    batched_data = get_batch(splits, batch_size, context_size)

    return batched_data, tokenizer