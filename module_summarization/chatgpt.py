import argparse
import logging
import os
import sys

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI
from utils.files_utils import load_config
from utils.api_generation import gpt_generation
from utils.context_utils import collect_good_context, trim_context, get_temperature
from prompt_templates import get_prompt
from tqdm.auto import tqdm

def prepare_code_context(row, max_context_toks, tokenizer, **context_kwargs):
    context = collect_good_context(row, config.get("context_strategy", "default"), **context_kwargs)
    if max_context_toks is None:
        return context
    return trim_context(context, tokenizer, max_context_toks)

def generate_one(row, code_context, client, model_name, prompt_version='v1_original'):
    intent = row['intent']
    filename = row['docfile_name']

    # Use the prompt template system
    prompt = get_prompt(prompt_version, intent, filename, code_context)

    temp = get_temperature(config.get("context_strategy", "default"))

    answer = gpt_generation(client, prompt, model_name, temp)
    return answer

def generate_all(config, client):

    # Extract parameters
    hf_api_key = config.get("hf_api_key")
    hf_tokenizer_checkpoint = config.get("hf_tokenizer_checkpoint")
    model_name = config.get("model_name")
    max_context_toks = config.get("max_context_toks", None)
    prompt_version = config.get("prompt_version", "v1_original")
    save_dir = config.get("save_dir")
    if not os.path.exists(f"{save_dir}"):
        os.makedirs(f"{save_dir}")

    # Avoid regenerating existing files
    existing = {int(fn.split('.')[0]) for fn in os.listdir(save_dir) if fn.endswith(".txt") and fn.split('.')[0].isdigit()}
    logging.info(f"Found {len(existing)} existing files in {save_dir}, will skip them.")
    
    # Preparing dataset
    logging.info("Downloading dataset")
    dataset = load_dataset("JetBrains-Research/lca-module-summarization",
                           token=hf_api_key)['test']
    logging.info("Downloading tokenizer to trim context")
    tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_checkpoint, 
                                              token=hf_api_key)

    # Generation
    
    logging.info("Start generation process")
    logging.info(f"Using prompt version: {prompt_version}")
    for row_idx, row in tqdm(enumerate(dataset), total=len(dataset), 
                             position=0, leave=True, 
                             desc="Generation"):
        if row_idx in existing:
            logging.info(f"Skipping {row_idx}: already exists")
            print(f"Skipping {row_idx}: already exists")
            continue
        
        code_context = prepare_code_context(row, max_context_toks, tokenizer)
        generate_res = generate_one(row, code_context, client, model_name, prompt_version)

        with open(f"{save_dir}/{row_idx}.txt", 'w', encoding='utf-8') as f:
            f.write(generate_res)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Script with YAML config and command line arguments."
    )
    # Argument for YAML config file path
    parser.add_argument('--config', type=str, 
                        default="config.yaml", 
                        help='Path to the YAML configuration file')
    args = parser.parse_args()

    config = load_config(args.config)
    logs_dir = config.get("logs_dir")
    if not os.path.exists(f"{logs_dir}"):
        os.makedirs(f"{logs_dir}")
    api_key = config.get("api_key")
    model_name = config.get("model_name")

    logging.basicConfig(
        filename=f'{logs_dir}/openai_gen_{model_name}.log',
        encoding='utf-8',
        level=logging.DEBUG
    )

    logging.info("Creating OpenAI client")
    import os
    os.environ["OPENAI_API_KEY"] = api_key
    client = OpenAI()
    logging.info("Done")

    logging.info("Call generate all function")
    generate_all(config, client)
    logging.info("Work finished")
