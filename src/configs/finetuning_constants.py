"""Set parameters for finetuning"""
import datetime 
import os

BATCH_SIZE       = 8        # sequences per batch
EPOCHS           = 15       # full data passes
LR               = 2e-5      # base learning rate
WEIGHT_DECAY     = 0.01     # weight decay
WARMUP_RATIO     = 0.1      # warmup fraction
GRAD_CLIP        = 1.0      # max gradient norm
VAL_SPLIT        = 0.1      # train/val split
EVAL_EVERY       = 250      # steps between eval/checkpoint
MAX_NEW_TOKENS   = 250      # tokens to generate in samples

INPUT_CSV           = 'data/finetuning/shakespeare_prompts_with_stories.csv'
OUTPUT_DIR          = 'data/finetuning'
RUN_TIMESTAMP       = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
PROCESSED_FILE      = os.path.join(OUTPUT_DIR, f'processed_finetuning_data_{RUN_TIMESTAMP}.txt')
PRETRAINED_MODEL_DIR = 'experimental_results/run_2025-04-28_18-25-53'