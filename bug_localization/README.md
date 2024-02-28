# Bug Localization

This folder contains code for Bug Localization task in Long Code Arena 🏟 benchmark. The task is: 
given an issue with bug description, identify the files within the project that need to be modified
to address the reported bug.

We provide [scripts for data collection and processing](./src/data) as well as several [baselines implementations](./src/baselines).
## 💾 Install dependencies
We provide dependencies for pip dependency manager, so please run the following command to install all required packages:
```shell
pip install -r requirements.txt
```
Bug Localization task: given an issue with bug description, identify the files within the project that need to be modified to address the reported bug

## 🤗 Load data
All data is stored in [HuggingFace 🤗](https://huggingface.co/datasets/JetBrains-Research/lca-bug-localization). It contains:
* Dataset with bug localization data (with issue description, sha of repo with initial state and to the state after issue fixation).
You can access data using [datasets](https://huggingface.co/docs/datasets/en/index) library:
```python3
from datasets import load_dataset

# Select a configuration from ["py", "java", "kt", "mixed"]
configuration = "py"
# Select a split from ["dev", "train", "test"]
split = "dev"
# Load data
dataset = load_dataset("JetBrains-Research/lca-bug-localization", configuration, split=split)
```
* Archived repos (from which we can extract repo content on different stages and get diffs which contains bugs fixations)
They are stored in `.tar.gz` so you need to run script to load them an unzip:
1. Set `repos_path` in [config](./configs/hf_data.yaml) to directory where you want to store repos
2. Run [load_data_from_hf.py](./src/load_data_from_hf.py) which will load all repos from HF and unzip them

## ⚙️ Run Baseline
* [TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn.feature_extraction.text.TfidfVectorizer)
* [GTE](https://huggingface.co/thenlper/gte-large)
* [CodeT5](https://huggingface.co/Salesforce/codet5p-110m-embedding)
* [GPT3.5](https://platform.openai.com/docs/models/gpt-3-5-turbo)
