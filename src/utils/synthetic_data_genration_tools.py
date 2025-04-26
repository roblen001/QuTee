"""Holds utils for synthetic data generation tools

TODO: Super low priority since this script is only going to be run to get
this shakespear data for data distillation, but eventually this
garbage code should be generalized to create and fine tuning dataset
and also cleaned. Lots of weird code I don't love in this one.
"""
import os
import time
import json
import logging
import pandas as pd
import openai

from dotenv import load_dotenv
from src.configs.synthetic_data_constants import (
    MODEL,
    TEMPERATURE,
    MAX_TOKENS,
    INPUT_CSV,
    OUTPUT_CSV,
    BATCH_INPUT_FILE,
    DAILY_TOKEN_LIMIT,
    PROMPT_OVERHEAD_TOKENS,
    CHECK_INTERVAL
)

# TODO: Generalize this code so I can make both prompts and answer for those prompts easily
# to help with any kind of fine tuning task (low priority TODO)

def load_data():
    """
    Load the input prompts and any existing outputs.
    
    Note: The reason the initial prompts are seperated from the story is because
    I got chatgpt (UI interface version not API) to generate a csv
    list of prompts to avoid aditional cost and token limits from on the API.
    
    Returns:
        tuple: (DataFrame of input prompts, DataFrame of existing outputs)
    """
    df_prompts = pd.read_csv(INPUT_CSV)
    if os.path.exists(OUTPUT_CSV):
        df_output = pd.read_csv(OUTPUT_CSV)
    else:
        df_output = pd.DataFrame(columns=["prompt", "story"])
    return df_prompts, df_output

def find_pending_prompts(df_prompts, df_output):
    """
    Identify prompts that are still missing a valid story.

    Note: Sometimes prompts give us errors from the API when we hit a token limit
    I am still adding them and moving the the next prompt if this happens. 
    (I might just have it stop if it hits and error in the future)

    Args:
        df_prompts (DataFrame): Full set of input prompts.
        df_output (DataFrame): Existing outputs with 'story' fields.

    Returns:
        list: Prompts that need to be generated.
    """
    # this merge is required because I currently get the chatGPT UI to generate initial
    # prompt to save on API calls since I already pay for the subscription. I will change
    # this when I generalize this (unsure yet)
    merged = pd.merge(df_prompts, df_output, on="prompt", how="left")
    need_error = merged['story'].str.contains("ERROR:", na=False)
    need_empty = merged['story'].isna()
    return merged.loc[need_error | need_empty, 'prompt'].tolist()

def prepare_batch_file(prompts):
    """
    Create a JSONL file with the next batch of prompts ready for OpenAI batch submission.
    
    Args:
        prompts (list): List of prompts to prepare for the batch.
    """
    os.makedirs(os.path.dirname(BATCH_INPUT_FILE), exist_ok=True)
    with open(BATCH_INPUT_FILE, 'w', encoding='utf-8') as f:
        for idx, prompt in enumerate(prompts):
            entry = {
                "custom_id": str(idx),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL,
                    "messages": [
                        # TODO move the system prompt out when generalizing this funcitonality
                        {"role": "system", "content": "You are William Shakespeare himself, crafting poetic and eloquent tales."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": TEMPERATURE,
                    "max_tokens": MAX_TOKENS
                }
            }
            f.write(json.dumps(entry) + "\n")
    logging.info("Wrote %d prompts to %s", len(prompts), BATCH_INPUT_FILE)

def submit_batch(client):
    """
    Upload the prepared JSONL file and submit it as a batch job to OpenAI.
    
    Args:
        client (openai.OpenAI): Authenticated OpenAI client.

    Returns:
        str: ID of the submitted batch job.
    """
    with open(BATCH_INPUT_FILE, "rb") as f:
        batch_file = client.files.create(
            file=f,
            purpose="batch"
        )
    logging.info("Uploaded batch input file. File ID: %s", batch_file.id)

    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    logging.info("Submitted batch job. Batch ID: %s", batch_job.id)
    return batch_job.id

def monitor_batch(client, batch_id):
    """
    Continuously monitor the batch job until it finishes.

    Args:
        client (openai.OpenAI): Authenticated OpenAI client.
        batch_id (str): Batch job ID.

    Returns:
        str: ID of the output file once the batch is complete.
    """
    while True:
        status = client.batches.retrieve(batch_id).status
        logging.info("Batch %s status: %s", batch_id, status)
        if status == "completed":
            return client.batches.retrieve(batch_id).output_file_id
        if status == "failed":
            raise RuntimeError(f"Batch {batch_id} failed")
        time.sleep(CHECK_INTERVAL)

def download_and_merge(client, output_file_id):
    """
    Download the completed batch output and merge it into the main OUTPUT_CSV.

    Args:
        client (openai.OpenAI): Authenticated OpenAI client.
        output_file_id (str): File ID of the completed batch output.
    """
    data = client.files.retrieve_content(output_file_id)
    if hasattr(data, "content") and callable(data.content):
        raw = data.content()
    elif isinstance(data, bytes):
        raw = data
    else:
        raw = data.encode("utf-8")

    output_path = BATCH_INPUT_FILE.replace("_input.jsonl", "_output.jsonl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(raw)
    logging.info("Downloaded batch output to %s", output_path)

    df_output = pd.read_csv(OUTPUT_CSV)
    batch_lines = [json.loads(line) for line in open(output_path, 'r', encoding='utf-8')]
    pending_idx = df_output[
        df_output['story'].isna() |
        # we want to re run the prompts that errored out for whatever reason 
        # and the prompts that have yet to be ran
        df_output['story'].str.contains("ERROR:", na=False)
    ].index.tolist()

    for rec, idx in zip(batch_lines, pending_idx):
        body = rec.get("response", {}).get("body", {})
        choices = body.get("choices", [])
        if choices:
            story = choices[0].get("message", {}).get("content", "")
        else:
            story = "ERROR: Could not generate story."
        df_output.at[idx, 'story'] = story

    df_output.to_csv(OUTPUT_CSV, index=False)
    logging.info("Merged batch results into %s", OUTPUT_CSV)