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

    evaluation_result_path = os.path.join(data_path, model.name())
    metadata_path = os.path.join(evaluation_result_path, "metadata.json")
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(evaluation_result_path, exist_ok=True)

    scores = defaultdict(list)

    for i, sample in tqdm(enumerate(dataset)):
        # To avoid issues with caching, comment out the lines 48-56 below and uncomment line 58:
        generated_file = os.path.join(evaluation_result_path, f"{i}.py")
        if os.path.exists(generated_file):
            with open(generated_file, "r", encoding="utf-8") as fin:
                generated_example = fin.read()
            
        else:
            generated_example = model.generate(sample["instruction"], sample["project_defined_elements"])
            with open(generated_file, "w", encoding="utf-8") as fout:
                fout.write(generated_example)
        # uncomment the line below:
        # generated_example = model.generate(sample["instruction"], sample["project_defined_elements"])
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
    # ===== Update the list of models and n_selections to use desired models instead:
    
    openai_models = ["gpt-3.5-turbo-0125", "gpt-4-0125-preview", "gpt-4.1-2025-04-14"]
    together_ai_models = ["mistralai/Mixtral-8x7B-Instruct-v0.1", "codellama/CodeLlama-70b-Instruct-hf"]
    
    models = {"open_ai": openai_models, "together_ai": together_ai_models}
    n_selections = [0, 20, 50, 200]
    
    
    for model_type in models:
        for model_name in model_type:
        
            for n in n_selections:  
                          
                if n_selections == 0:
                    
                    if model_type == "open_ai":
                        evaluate_openai(model_name, False)
                    
                    else:
                        evaluate_together(model_name, False)
                    
                else:
                    
                    if model_type == "open_ai":
                        evaluate_openai(model_name, True, n)

                    else:
                        evaluate_together(model_name, True, n)
