import torch
from typing import Dict, List, Tuple

def read_textfile(path: str) -> str:
    """
    Reads an entire UTF-8 text file into memory.

    Args:
        path: Path to the .txt file.

    Returns:
        The file’s contents as a single Python string.
    """
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def create_tokenizer(text: str) -> Dict[str, Dict]:
    """
    Builds a character-level tokenizer from the given text.

    We extract the sorted set of unique characters, then assign each
    a consecutive integer ID. This yields:

    - encoder: char -> int  
    - decoder: int  -> char

    Args:
        text: The raw text corpus.

    Returns:
        A dict with keys 'encoder' and 'decoder', each mapping characters
        to IDs and back again.
    """
    chars = sorted(set(text))
    encoder = {ch: i for i, ch in enumerate(chars)}
    decoder = {i: ch for i, ch in enumerate(chars)}
    return {'encoder': encoder, 'decoder': decoder}


def encode(text: str, tokenizer: Dict[str, Dict]) -> List[int]:
    """
    Converts a string into a list of integer token IDs.

    Args:
        text: The input string.
        tokenizer: The output of create_tokenizer().

    Returns:
        A list of ints, one per character in `text`.
    """
    enc = tokenizer['encoder']
    return [enc[ch] for ch in text]


def decode(tokens: List[int], tokenizer: Dict[str, Dict]) -> str:
    """
    Converts a list of integer token IDs back into a string.

    Args:
        tokens: List of integer IDs or a torch.Tensor.
        tokenizer: The output of create_tokenizer().

    Returns:
        The reconstructed string.
    """
    if isinstance(tokens, torch.Tensor):
        flat_list = tokens.view(-1).tolist()
    else:
        flat_list = tokens  # <--- Fix here: if already a list, use it directly

    dec = tokenizer['decoder']
    return ''.join(dec[t] for t in flat_list)

def train_test_split(
    data: torch.Tensor,
    threshold: float = 0.9
) -> Dict[str, torch.Tensor]:
    """
    Splits a 1D tensor of token IDs into train and test portions.

    Args:
        data: Tensor of shape (N,) containing a long token stream.
        threshold: Fraction of data to use for training (e.g. 0.9).

    Returns:
        A dict with keys 'train' and 'test', each a 1D tensor.
    """
    N = len(data)
    split = int(N * threshold)
    return {
        'train': data[:split],
        'test':  data[split:]
    }

def get_batch(
    dataset: Dict[str, torch.Tensor],
    batch_size: int,
    context_size: int
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Generates a single batch for each split with fixed context windows.

    We randomly sample `batch_size` starting positions `i` from each split,
    then take:

        x := tokens[i : i + context_size]
        y := tokens[i+1 : i + context_size+1]

    so that each position predicts its next token.

    Args:
        dataset: Dict with 'train'/'test' 1D tensors.
        batch_size: Number of windows per batch.
        context_size: Length of each input window.

    Returns:
        A dict mapping each split to a dict with:
          - 'x': LongTensor of shape (batch_size, context_size)
          - 'y': LongTensor of shape (batch_size, context_size)
    """
    batched = {}
    for split, data in dataset.items():
        # choose random start indices so we have room for context+1
        ix = torch.randint(0, len(data) - context_size - 1, (batch_size,))
        x = torch.stack([ data[i : i + context_size] for i in ix ])
        y = torch.stack([ data[i + 1 : i + context_size + 1] for i in ix ])
        batched[split] = {'x': x, 'y': y}
    return batched
