# Quick Start Guide - Module Summarization Prompt Testing

## 🎯 Objective
Test different prompt versions to find the best one for generating high-quality documentation

## 📝 Prerequisites

1. Install dependencies:
```bash
cd module_summarization
poetry install
```

2. Configure API keys:
Edit `configs/config_openai.yaml` and fill in your API keys:
- `api_key`: OpenAI API key
- `hf_api_key`: HuggingFace API key

## 🚀 Three Ways to Use

### Method 1: Interactive Script (Recommended for Beginners)

Easiest way with interactive menu:

```bash
bash quick_start_prompts.sh
```

Select options:
- `1` - Test single prompt version
- `2` - Batch test multiple versions
- `3` - View existing results
- `4` - View detailed guide

### Method 2: Direct Command Line

#### Test Single Version

```bash
# Use v2_structured prompt
poetry run python chatgpt.py --config="configs/config_openai_v2.yaml"

# Use v7_quality_focused prompt
poetry run python chatgpt.py --config="configs/config_openai_v7.yaml"
```

#### Batch Test Multiple Versions

```bash
# Test 3 versions at once
poetry run python batch_test_prompts.py \
  --config configs/config_openai.yaml \
  --versions v1_original v2_structured v7_quality_focused
```

### Method 3: Custom Configuration File

Create your own config file:

```yaml
# my_config.yaml
hf_api_key: "your_hf_key"
hf_tokenizer_checkpoint: "meta-llama/Llama-2-7b-chat-hf"
api_key: "your_openai_key"
model_name: "gpt-4"  # or gpt-3.5-turbo-0125
logs_dir: "./logs"
save_dir: "./my-predictions"
max_context_toks: 4000
prompt_version: "v7_quality_focused"  # Choose version to test
```

Run:
```bash
poetry run python chatgpt.py --config="my_config.yaml"
```

## 📊 Compare and Analyze Results

### 1. Quick Comparison of Two Versions

```bash
poetry run python compare_outputs.py \
  --dirs predictions-v1_original predictions-v7_quality_focused \
  --sample 0
```

### 2. View Statistical Summary

```bash
poetry run python compare_outputs.py \
  --dirs predictions-v1_original predictions-v2_structured predictions-v7_quality_focused \
  --summary
```

### 3. Generate Detailed Comparison Report

```bash
poetry run python generate_comparison_report.py \
  --dirs predictions-v1_original predictions-v2_structured predictions-v7_quality_focused \
  --samples 0 1 2 3 4 \
  --output my_comparison_report.md
```

Then view the report:
```bash
open my_comparison_report.md
# Or open in browser
```

### 4. Run Quantitative Evaluation

```bash
# Edit configs/config_eval.yaml to set correct prediction directory
poetry run python metrics.py --config="configs/config_eval.yaml"
```

## 💡 Recommended Workflow

### First Time Use (Exploration Phase)

```bash
# 1. Test baseline version
poetry run python chatgpt.py --config="configs/config_openai.yaml"

# 2. Test structured version
poetry run python chatgpt.py --config="configs/config_openai_v2.yaml"

# 3. Quick comparison
poetry run python compare_outputs.py \
  --dirs predictions-chatgpt3-2k predictions-gpt35-v2-structured \
  --sample 0

# 4. If v2 is better, continue testing other versions
poetry run python chatgpt.py --config="configs/config_openai_v7.yaml"
```

### Batch Testing (Recommended)

```bash
# Test multiple versions at once
poetry run python batch_test_prompts.py \
  --config configs/config_openai.yaml \
  --versions v1_original v2_structured v3_detailed v7_quality_focused

# Generate comparison report
poetry run python generate_comparison_report.py \
  --dirs predictions-*-v1_original predictions-*-v2_structured \
           predictions-*-v3_detailed predictions-*-v7_quality_focused \
  --samples 0 1 2 \
  --output batch_comparison.md

# View report
open batch_comparison.md
```

### Fine-Tuning (Advanced)

```bash
# 1. Based on best version, create v8_custom in prompt_templates.py
# 2. Create new config file
# 3. Test new version
# 4. Compare and evaluate
# 5. Iterate for improvement
```

## 🎯 Use Cases for Each Prompt Version

### v1_original - Baseline
```bash
prompt_version: "v1_original"
```
- Use: Benchmark comparison
- Features: Simplest and most direct

### v2_structured - Structured
```bash
prompt_version: "v2_structured"
```
- Use: Well-organized documentation needed
- Features: Clear sections and requirements
- **Recommended for: API docs, module descriptions**

### v3_detailed - Detailed Guidance
```bash
prompt_version: "v3_detailed"
```
- Use: Comprehensive, in-depth documentation needed
- Features: Most detailed guidance and requirements
- **Recommended for: Complex systems, critical modules**

### v4_role_based - Role-Playing
```bash
prompt_version: "v4_role_based"
```
- Use: Deep understanding and explanation needed
- Features: Emphasizes expert role and thought process
- **Recommended for: Algorithm implementations, core logic**

### v5_example_guided - Example-Guided
```bash
prompt_version: "v5_example_guided"
```
- Use: Specific format documentation needed
- Features: Provides output template
- **Recommended for: Standardized docs, team collaboration**

### v6_concise - Concise
```bash
prompt_version: "v6_concise"
```
- Use: Fast generation, cost reduction
- Features: Short but retains core requirements
- **Recommended for: Bulk documentation generation, limited budget**

### v7_quality_focused - Quality-Oriented
```bash
prompt_version: "v7_quality_focused"
```
- Use: High-quality professional documentation
- Features: Emphasizes quality standards and best practices
- **Recommended for: Production environments, user documentation**

## 🔍 Evaluation Criteria

Good documentation should:
- ✅ Accurately reflect code functionality
- ✅ Have clear structure, easy to navigate
- ✅ Include practical examples
- ✅ Use correct markdown formatting
- ✅ Appropriate length (neither too short nor too verbose)
- ✅ Professional and clear language
- ✅ Highlight key points with clear logic

## 📈 Performance Comparison Guidelines

| Metric | How to Measure |
|--------|----------------|
| **Accuracy** | Manual review, check if code is correctly understood |
| **Completeness** | Check if all important features are covered |
| **Format Quality** | Check markdown rendering effect |
| **Length** | Use `--summary` to view average word count |
| **Structure** | Count number of headers and code blocks |
| **CompScore** | Run metrics.py to get quantitative score |

## ⚠️ Common Issues

### Q: Error "API key not found"
**A**: Edit config file and fill in correct API key

### Q: Where is the generated documentation?
**A**: In the directory specified by `save_dir` in config file, e.g., `./predictions-v2_structured/`

### Q: How to test only a few samples?
**A**: Currently needs to run full dataset, can add limits in code

### Q: Are there big differences between models?
**A**: Yes, GPT-4 is usually better than GPT-3.5, recommend testing on both models

### Q: Can I modify the prompts?
**A**: Yes! Edit `prompt_templates.py` to add new versions

### Q: How to select the best version?
**A**: 
1. First view comparison report
2. Run metrics.py to get CompScore
3. Manually review several samples
4. Consider both quality and cost

## 📚 More Resources

- **Detailed Guide**: `PROMPT_GUIDE.md`
- **Complete Instructions**: `README.md`
- **Changes Summary**: `CHANGES_SUMMARY.md`
- **Code Implementation**: `prompt_templates.py`

## 🎓 Best Practices

1. **Start with small-scale testing**: Verify quickly with a few samples
2. **Keep good records**: Document pros and cons of each version
3. **Quantitative + Qualitative**: Combine CompScore with manual review
4. **Iterative optimization**: Continuously improve based on results
5. **Version control**: Save good prompt versions

---

**Good luck finding the best prompt! Check `PROMPT_GUIDE.md` for questions or review log files.**
