# 🏟️ Long Code Arena Extended Baselines

## What is Long Code Arena? 

[**Long Code Arena**](https://huggingface.co/spaces/JetBrains-Research/long-code-arena) is a suite of benchmarks for code-related tasks with large contexts, up to a whole code repository.
It currently spans six different tasks and contains six datasets:
* 🤗 [Library-based code generation](https://huggingface.co/datasets/JetBrains-Research/lca-library-based-code-generation)
* 🤗 [CI builds repair](https://huggingface.co/datasets/JetBrains-Research/lca-ci-builds-repair)
* 🤗 [Project-level code completion](https://huggingface.co/datasets/JetBrains-Research/lca-project-level-code-completion)
* 🤗 [Commit message generation](https://huggingface.co/datasets/JetBrains-Research/lca-commit-message-generation)
* 🤗 [Bug localization](https://huggingface.co/datasets/JetBrains-Research/lca-bug-localization)
* 🤗 [Module summarization](https://huggingface.co/datasets/JetBrains-Research/lca-module-summarization)

This repository is a fork of the official
[JetBrains Long Code Arena baselines](https://github.com/JetBrains-Research/lca-baselines),
extended with new models, metrics, retrieval strategies, and prompt-engineering experiments across
three benchmarks.

## What This Fork Adds

This fork introduces improvements across three benchmarks:

- **Project-Level Code Completion**
- **Library-Based Code Generation**
- **Module Summarization**

### Project-Level Code Completion

* Evaluated several new models (StarCoder2, Granite-1B, Qwen-7B, GPT-3.5/4o-mini/4o).
* Implemented BM25 context composition strategy to compare it to path-distance context strategy.
* Added full support for OpenAI models in the completion pipeline.


### Library-Based Code Generation

* Added the CodeBLEU metric suite for structural/semantic code evaluation.
* Introduced new retrieval strategies (BM25, BM25 + function headers, expanded windows).
* Evaluated a broader set of modern LLMs across these strategies.


### Module Summarization

* Added BERTScore metric for embedding-based semantic similarity.
* Added structured prompt templates for prompt-engineering experiments.
* Implemented context composition strategies such as BM25 retrieval and code-context cleaning to determine whether cleaner code contexts improve the generated documentation.
* Evaluated multiple newer models on this benchmark.
