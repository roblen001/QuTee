"""
Run batch job, monitor completion, download results, and prepare next batch
using up to your daily token limit.
"""

import os
import time
import json
import logging
import pandas as pd
import openai

from dotenv import load_dotenv
from src.configs.synthetic_data_constants import (
    MAX_TOKENS,
    DAILY_TOKEN_LIMIT,
    PROMPT_OVERHEAD_TOKENS,
)
from src.utils.synthetic_data_genration_tools import (
    load_data, 
    find_pending_prompts, 
    prepare_batch_file, 
    submit_batch,
    monitor_batch,
    download_and_merge
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

def main():
    """
    Main entrypoint for running batch jobs until all prompts are completed,
    respecting the daily token limit.
    """
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI()

    df_prompts, df_output = load_data()

    while True:
        pending = find_pending_prompts(df_prompts, df_output)
        if not pending:
            logging.info("All done—no pending prompts.")
            break

        tokens_left = DAILY_TOKEN_LIMIT
        per_prompt = MAX_TOKENS + PROMPT_OVERHEAD_TOKENS
        affordable = tokens_left // per_prompt
        if affordable <= 0:
            logging.info("Token limit reached: used %d, need %d more", per_prompt)
            break

        batch_prompts = pending[:affordable]
        logging.info("Next batch: %d prompts (tokens left: %d)", len(batch_prompts), tokens_left)

        prepare_batch_file(batch_prompts)
        batch_id = submit_batch(client)
        output_file_id = monitor_batch(client, batch_id)
        download_and_merge(client, output_file_id)


    logging.info("Script complete. Total tokens used: %d")

if __name__ == "__main__":
    main()
