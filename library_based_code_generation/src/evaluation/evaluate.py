import json
import os
import pickle
from collections import defaultdict

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from ..context.parsed_file import ParsedFile
from ..metrics.chrf import ChrF
from ..metrics.metric import Metric
from ..metrics.overlap import Overlap
from ..metrics.ngram_match import NGram_Match
from ..metrics.weighted_ngram_match import Weighted_NGram_Match
from ..metrics.syntax_match import Syntax_Match
from ..metrics.dataflow_match import Dataflow_Match 

from ..models.example_generation_model import ExampleGenerationModel
from ..models.openai_model import OpenAIModel
from ..models.together_model import TogetherModel


def extract_code(message):
    if "```python" in message:
        return message.split("```python")[1].split("```")[0].strip()
    if "```" in message:
        return message.split("```")[1].split("```")[0].strip()
    if "<code>" in message:
        return message.split("<code>")[1].split("</code>")[0].strip()
    return message.strip()


def evaluate(model: ExampleGenerationModel, metrics: list[Metric], data_path: str):
    print(f"Evaluating model {model.name()}")

    dataset = load_dataset("JetBrains-Research/lca-library-based-code-generation", split="test")
    n_samples = len(dataset)

    evaluation_result_path = os.path.join(data_path, model.name())
    metadata_path = os.path.join(evaluation_result_path, "metadata.json")
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(evaluation_result_path, exist_ok=True)

    scores = defaultdict(list)

    for i, sample in tqdm(enumerate(dataset)):
        # generated_file = os.path.join(evaluation_result_path, f"{i}.py")
        # if os.path.exists(generated_file):
        #     with open(generated_file, "r", encoding="utf-8") as fin:
        #         generated_example = fin.read()
            
            
        # else:
        generated_example = model.generate(sample["instruction"], sample["project_defined_elements"])
        # with open(generated_file, "w", encoding="utf-8") as fout:
        #     fout.write(generated_example)

        generated_example = extract_code(generated_example)
        for metric in metrics:
            score = metric.score(generated_example, sample["clean_reference"], sample["unique_apis"])
            scores[metric.name()].append(score)

    for metric in metrics:
        print(f"Average score for {metric.name()}: {np.mean(scores[metric.name()]):.3f}")
    print()

    metadata = {
        "metrics": {
            metric.name(): {
                "mean": np.mean(scores[metric.name()]),
            }
            for metric in metrics
        },
        "name": model.name(),
    }
    with open(metadata_path, "w") as fout:
        json.dump(metadata, fout)


def evaluate_openai(model_name, use_bm25=False, n_selections=0):
    evaluate(OpenAIModel(model_name, use_bm25, n_selections), [ChrF(), Overlap(), NGram_Match(), Weighted_NGram_Match(), Syntax_Match(), Dataflow_Match()], "results")


def evaluate_together(model_name, use_bm25=False, n_selections=0):
    evaluate(TogetherModel(model_name, use_bm25, n_selections), [ChrF(), Overlap(), NGram_Match(), Weighted_NGram_Match(), Syntax_Match(), Dataflow_Match()], "results")


if __name__ == "__main__":
    # evaluate_together("codellama/CodeLlama-7b-Instruct-hf")
    # evaluate_together("codellama/CodeLlama-70b-Instruct-hf")
    # evaluate_together("mistralai/Mistral-7B-Instruct-v0.3")
    # evaluate_together("mistralai/Mixtral-8x7B-Instruct-v0.1")
    # evaluate_openai("gpt-3.5-turbo-0125", use_bm25=True)
    # evaluate_openai("gpt-4-0125-preview", use_bm25=True)
    # evaluate_together("codellama/CodeLlama-7b-Instruct-hf", use_bm25=True)
    # evaluate_together("codellama/CodeLlama-70b-Instruct-hf", use_bm25=True)
    # evaluate_together("mistralai/Mistral-7B-Instruct-v0.3", use_bm25=True)
    # evaluate_together("mistralai/Mixtral-8x7B-Instruct-v0.1", use_bm25=True)
    
    

    openai_models = ["gpt-3.5-turbo-0125", "gpt-4-0125-preview", "gpt-4.1-2025-04-14"]
    
    n_selections = [0, 20, 50, 200]
    
    
    for model in openai_models:
        
        for n in n_selections:
            if n_selections == 0:
                evaluate_openai(model, False)
                
            else:
                evaluate_openai(model, True, n)
    
    
    # python -m Autumn2025-JetBrains-LongCodeArena.library_based_code_generation.src.evaluation.evaluate