"""
Helper tools for finetuning the Shakespeare model.
"""

import os
import pandas as pd
import torch
from typing import Tuple
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from src.utils.data_processing_tools import read_textfile, train_test_split

class EarlyStopping:
    """
    Early stopping utility to halt training when validation loss stops improving.
    """
    def __init__(self, patience: int = 3):
        self.patience = patience
        self.best_val_loss = float('inf')
        self.counter = 0
        self.should_stop = False
        self.best_model_state = None

    def step(self, val_loss: float, model: torch.nn.Module):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.counter = 0
            self.best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

def get_linear_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int
) -> LambdaLR:
    """
    Build a linear warmup + decay learning rate scheduler.
    """
    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return float(step) / max(1, num_warmup_steps)
        return max(
            0.0,
            float(num_training_steps - step) /
            max(1, num_training_steps - num_warmup_steps)
        )
    return LambdaLR(optimizer, lr_lambda)

def get_pretrained_model_path(base_dir: str, pretrained_dir: str) -> str:
    """
    Locate the pretrained model file:
    1) Check pretrained_dir/model_full.pt
    2) Look for finetune_<timestamp>/model_full.pt under base_dir
    3) Fallback to base_dir/model_full.pt
    """
    exp_path = os.path.join(pretrained_dir, 'model_full.pt')
    if os.path.exists(exp_path):
        return exp_path
    subs = [d for d in os.listdir(base_dir) if d.startswith('finetune_') and os.path.isdir(os.path.join(base_dir, d))]
    if subs:
        latest = sorted(subs)[-1]
        candidate = os.path.join(base_dir, latest, 'model_full.pt')
        if os.path.exists(candidate):
            return candidate
    fallback = os.path.join(base_dir, 'model_full.pt')
    if os.path.exists(fallback):
        return fallback
    raise FileNotFoundError(f"No pretrained model found in {pretrained_dir} or {base_dir}")

def preprocess_data_for_finetuning(input_csv: str, output_txt: str) -> None:
    """
    Read CSV prompts/stories and write to a text file.
    """
    if os.path.exists(output_txt):
        print(f"✅ Preprocessed file already exists: {output_txt}")
        return

    os.makedirs(os.path.dirname(output_txt), exist_ok=True)
    df = pd.read_csv(input_csv)

    if not {"prompt", "story"}.issubset(df.columns):
        raise ValueError("CSV must have 'prompt' and 'story' columns!")

    samples = []
    for idx, row in df.iterrows():
        prompt = str(row['prompt']).strip()
        story = str(row['story']).strip()
        formatted = f"<|PROMPT|> {prompt} <|STORY|> {story} <|EOS|>"
        samples.append(formatted)

    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write("\n".join(samples))

    print(f"✅ Processed data written to {output_txt}")
    print(f"Total samples: {len(samples)}")

def load_data(processed_file: str, output_dir: str, pretrained_dir: str, val_split: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Tokenize processed text into training and validation streams.
    """
    model_path = get_pretrained_model_path(output_dir, pretrained_dir)
    bundle = torch.load(model_path, map_location='cpu')
    encoder = bundle.tokenizer['encoder']
    text = read_textfile(processed_file)
    ids = [encoder.get(ch, encoder.get('<|UNK|>', 0)) for ch in text]
    stream = torch.tensor(ids, dtype=torch.int64)
    splits = train_test_split(stream, threshold=1 - val_split)
    return splits['train'], splits['test']

def encode_prompt(prompt: str, tokenizer: dict, context_size: int) -> torch.Tensor:
    """
    Encode a text prompt into a tensor (1, context_size) with left padding.
    """
    enc_map = tokenizer['encoder']
    prompt_norm = prompt.lower()
    token_ids = [enc_map.get(ch, enc_map.get('<|UNK|>', 0)) for ch in prompt_norm]
    if len(token_ids) > context_size:
        token_ids = token_ids[-context_size:]
    else:
        token_ids = [0] * (context_size - len(token_ids)) + token_ids
    return torch.tensor([token_ids], dtype=torch.int64)
