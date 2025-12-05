# 🏟️ Long Code Arena Baselines
## Module summarization

This directory contains the code for running baselines for the Module summarization task in the Long Code Arena benchmark.

We provide the implementation of baselines running inference via [OpenAI](https://platform.openai.com/docs/overview) and [Together.AI](https://www.together.ai/).
We generate documentation based on an intent (one sentence description of documentation content), target documentation name, and relevant code context. 

# How-to

## 💾 Install dependencies

We provide dependencies via the [Poetry](https://python-poetry.org/docs/) manager. 

* To install dependecies, run `poetry install`

## 🚀 Run 

#### Generation

In order to generate your predictions, add your parameters in the [configs](configs) directory and run: 

* `poetry run python chatgpt.py --config="configs/config_openai.yaml"` if you use [OpenAI](https://platform.openai.com/docs/overview) models;
* `poetry run python togetherai.py --config="configs/config_together.yaml"` if you use [Together.AI](https://www.together.ai/) models.

The script will generate predictions and put them into the `save_dir` directory from config.

#### Metrics 

To compare predicted and ground truth texts, we introduce the new metric based on LLM as an assessor. Our approach involves feeding the LLM with relevant code and two versions of documentation: the ground truth and the model-generated text. To mitigate variance and potential ordering effects in model responses, we calculate the probability that the generated documentation is superior by averaging the results of two queries:

```math
CompScore = \frac{ P(pred | LLM(code, pred, gold)) + P(pred | LLM(code, gold, pred))}{2}
```

In order to evaluate predictions, add your parameters in the [config](configs/config_eval.yaml) and run:
* `poetry run python metrics.py --config="configs/config_eval.yaml"`

The script will evaluate the predictions and save the results into the `results.json` file.


## New Additions:
### 1. Prompt Engineering

**NEW**: We now support multiple prompt versions to improve generation quality! 

To test different prompts and improve documentation quality:

**Available versions**: 7 different prompt templates (v1-v7) with various improvements
#### Example: Test a different prompt

```bash
# Edit your config to include prompt_version
echo "prompt_version: v7_quality_focused" >> configs/config_openai.yaml

# Run generation
poetry run python chatgpt.py --config="configs/config_openai.yaml"
```
You can change run_all_versions.sh to help you run multiple version.

### 2. Different Context Compositions:
**NEW**: You can now select different context compositions for generation! 

#### **Steps:**
In `configs/config_openai.yaml` or `configs/config_together.yaml`, select the desired option for the context strategy that you to be used for generation.

**Available versions**: 
- `"default"`: Original existing method for handling context from datasets. Takes the plain datapoints as inputs with no modifications.
- `"ast_strategy"`: Uses AST library to trim less-relevant parts of the context such as import statements and  main-guard (`if __name__= __main__:` blocks).
- `"bm25"`: Uses BM25 library for utilizing the most relevant context pieces.

Run the files as explained in the **Generation** steps above.


### 3. BERT Score
We introduced a new metric, **BERTScore**, to measure semantic similarity between generated and ground-truth documentation. This metric provided another view of documentation quality.

#### **Steps:**
You can select whether you want to run and include the BERT Score metrics in your results. This can be set in the `configs/config_eval.yaml` file. 
Simply set the `use_bert` key to `True` if you want to include it, or `False` if not.


