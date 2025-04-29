"""
Finetuning script: take a pretrained Shakespeare model and further train it
for coherent Elizabethan story generation.
"""

import os
import datetime
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd

from src.utils.data_processing_tools import decode
from src.utils.finetuning_tools import (
    get_linear_schedule_with_warmup,
    get_pretrained_model_path,
    preprocess_data_for_finetuning,
    load_data,
    encode_prompt
)
from src.monitoring.monitoring import TrainingMonitor

from src.configs.finetuning_constants import (
    BATCH_SIZE,
    EPOCHS,
    LR,
    WEIGHT_DECAY,
    WARMUP_RATIO,
    GRAD_CLIP,
    VAL_SPLIT,
    EVAL_EVERY,
    MAX_NEW_TOKENS,
    INPUT_CSV,
    OUTPUT_DIR,
    RUN_TIMESTAMP,
    PROCESSED_FILE,
    PRETRAINED_MODEL_DIR
)

# Prefix for generation prompt to match fine-tuning style
PREFIX = (
    "<|PROMPT|> You are William Shakespeare himself, crafting poetic and eloquent stories. "
    "Write a short story in Shakespearean English about "
)

def finetune_model() -> None:
    """Full-model fine-tuning loop with custom prompt sampling."""
    model_path = get_pretrained_model_path(OUTPUT_DIR, PRETRAINED_MODEL_DIR)
    model = torch.load(model_path, map_location='cpu')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device).train()
    # ensuring gradients will be computed
    for p in model.parameters():
        p.requires_grad = True 

    train_stream, val_stream = load_data(PROCESSED_FILE, OUTPUT_DIR, PRETRAINED_MODEL_DIR, VAL_SPLIT)
    x_all = train_stream[:-1].unsqueeze(1)
    y_all = train_stream[1:].unsqueeze(1)
    loader = DataLoader(TensorDataset(x_all, y_all), batch_size=BATCH_SIZE, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(loader) * EPOCHS
    warmup_steps = int(WARMUP_RATIO * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    # decrease the learning rate when the model plateaus to squeeze any more possbile gains
    reduce_lr_on_plateau = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    out_dir = os.path.join(OUTPUT_DIR, f'finetune_{RUN_TIMESTAMP}')
    os.makedirs(out_dir, exist_ok=True)
    monitor = TrainingMonitor()
    monitor.results_dir = out_dir

    checkpoint_path = os.path.join(out_dir, 'ckpt_latest.pt')  # always overwrite latest checkpoint

    step = 0
    for epoch in range(1, EPOCHS + 1):
        for x_batch, y_batch in loader:
            # reset gradient to zero for each batch
            optimizer.zero_grad()
            x = x_batch.t().to(device)
            y = y_batch.t().to(device)
            _, loss = model(x, y)
            loss.backward()
            # prevent exploding gradients by clipping them to a max norm 
            clip_grad_norm_(model.parameters(), GRAD_CLIP)
            # update the model parameters
            optimizer.step()
            # update the learning rate
            scheduler.step()

            step += 1
            if step % EVAL_EVERY == 0 or step == total_steps:
                percent_complete = (step / total_steps) * 100

                # get the loss for the entire batch
                stats = model.pooled_loss(
                    {'train': {'stream': train_stream}, 'test': {'stream': val_stream}},
                    eval_iters=50,
                    batch_size=BATCH_SIZE,
                    context_size=model.context_size
                )
                monitor.record(train_loss=stats['train'], val_loss=stats['test'], epoch_time=0)
                reduce_lr_on_plateau.step(stats['test'])

                print(f"🔵 Fine-tuning progress: {percent_complete:.2f}% complete")

                # checkpoint save (always overwrite)
                torch.save(model.state_dict(), checkpoint_path)

                # generate from styled prompt
                subject = 'an old man.<|STORY|>'
                prompt = PREFIX + subject
                input_ids = encode_prompt(prompt, model.tokenizer, model.context_size).to(device)
                sample_ids = model.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS)
                generated = decode(sample_ids[0].tolist(), model.tokenizer)
                out_file = os.path.join(out_dir, f'sample_{step}.txt')
                with open(out_file, 'w') as sf:
                    sf.write(f"Prompt: {prompt}\nGenerated:\n{generated}")
                print(f"Step {step} → train={stats['train']:.4f}, val={stats['test']:.4f}")

    # final export
    final = os.path.join(out_dir, 'model_full_finetuned.pt')
    torch.save(model, final)
    print(f"✅ Saved fully fine-tuned model to {final}")

if __name__ == '__main__':
    preprocess_data_for_finetuning(INPUT_CSV, PROCESSED_FILE)
    finetune_model()
