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

## 🎯 Prompt Engineering

**NEW**: We now support multiple prompt versions to improve generation quality! 

To test different prompts and improve documentation quality:

1. **Quick start**: See [PROMPT_GUIDE.md](PROMPT_GUIDE.md) for detailed instructions
2. **Available versions**: 7 different prompt templates (v1-v7) with various improvements
3. **Configuration**: Add `prompt_version: "v2_structured"` to your config file
4. **Batch testing**: Use `batch_test_prompts.py` to test multiple versions at once
5. **Comparison**: Use `compare_outputs.py` to compare results side-by-side

### Example: Test a different prompt

```bash
# Edit your config to include prompt_version
echo "prompt_version: v7_quality_focused" >> configs/config_openai.yaml

# Run generation
poetry run python chatgpt.py --config="configs/config_openai.yaml"
```

### Example: Batch test multiple prompts

```bash
# Test 3 different prompt versions automatically
poetry run python batch_test_prompts.py \
  --config configs/config_openai.yaml \
  --versions v1_original v2_structured v7_quality_focused

# Compare outputs
poetry run python compare_outputs.py \
  --dirs predictions-v1_original predictions-v7_quality_focused \
  --sample 0
```

See [PROMPT_GUIDE.md](PROMPT_GUIDE.md) for more details on available prompt versions and optimization strategies.

#### Metrics 

To compare predicted and ground truth texts, we introduce the new metric based on LLM as an assessor. Our approach involves feeding the LLM with relevant code and two versions of documentation: the ground truth and the model-generated text. To mitigate variance and potential ordering effects in model responses, we calculate the probability that the generated documentation is superior by averaging the results of two queries:

```math
CompScore = \frac{ P(pred | LLM(code, pred, gold)) + P(pred | LLM(code, gold, pred))}{2}
```

In order to evaluate predictions, add your parameters in the [config](configs/config_eval.yaml) and run:
* `poetry run python metrics.py --config="configs/config_eval.yaml"`

The script will evaluate the predictions and save the results into the `results.json` file.
