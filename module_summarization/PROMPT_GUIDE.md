# Prompt Engineering Guide for Module Summarization

This guide explains how to test and improve different prompts for better documentation generation quality.

## 📋 Available Prompt Versions

We provide 7 different prompt versions, each with unique characteristics:

### `v1_original` - Baseline Version
- **Features**: Simple and direct original prompt
- **Use case**: Benchmark comparison
- **Pros**: Concise
- **Cons**: Lacks structure and guidance

### `v2_structured` - Structured Version
- **Features**: Clear section divisions and formatting requirements
- **Use case**: Well-organized documentation needed
- **Pros**: Clear task structure, markdown formatting guidance
- **Improvements**: Added specific requirement lists

### `v3_detailed` - Detailed Guidance Version
- **Features**: Complete content and style guidelines
- **Use case**: High-quality, comprehensive documentation needed
- **Pros**: Detailed content requirements, style guidelines, output format specifications
- **Improvements**: Most comprehensive guidance, covering multiple aspects

### `v4_role_based` - Role-Playing Version
- **Features**: Emphasizes code understanding and clear explanation
- **Use case**: Documentation requiring in-depth analysis
- **Pros**: Emphasizes expert role, step-by-step approach
- **Improvements**: Added thought process guidance

### `v5_example_guided` - Example-Guided Version
- **Features**: Provides documentation structure template
- **Use case**: Documentation requiring specific format
- **Pros**: Clear output structure example
- **Improvements**: Intuitive format template

### `v6_concise` - Concise Version
- **Features**: Brief but includes key requirements
- **Use case**: Testing if brevity improves quality
- **Pros**: Reduces prompt length, lowers costs
- **Improvements**: Retains core requirements, removes redundancy

### `v7_quality_focused` - Quality-Oriented Version
- **Features**: Emphasizes documentation quality standards and best practices
- **Use case**: Professional-grade documentation needed
- **Pros**: Clear quality standards, style guidelines
- **Improvements**: Added quality checklist

## 🚀 How to Use

### 1. Modify Configuration File

Add the `prompt_version` parameter in YAML files under the `configs/` directory:

```yaml
hf_api_key: "YOUR_KEY_HERE"
hf_tokenizer_checkpoint: "meta-llama/Llama-2-7b-chat-hf"
api_key: "YOUR_API_KEY"
model_name: "gpt-3.5-turbo-0125"
logs_dir: "./logs"
save_dir: "./predictions-gpt35-v2-structured"
max_context_toks: 2000
prompt_version: "v2_structured"  # Specify prompt version
```

### 2. Run Generation

Run with the specified configuration file:

```bash
# Using OpenAI models
poetry run python chatgpt.py --config="configs/config_openai_v2.yaml"

# Using Together.AI models
poetry run python togetherai.py --config="configs/config_together_v2.yaml"
```

### 3. Compare Different Prompts

We provide three pre-configured files for quick testing:

```bash
# Test structured prompt
poetry run python chatgpt.py --config="configs/config_openai_v2.yaml"

# Test detailed guidance prompt
poetry run python chatgpt.py --config="configs/config_openai_v3.yaml"

# Test quality-focused prompt
poetry run python chatgpt.py --config="configs/config_openai_v7.yaml"
```

### 4. Evaluate Results

Use metrics.py to evaluate the quality of different prompt versions:

```bash
poetry run python metrics.py --config="configs/config_eval.yaml"
```

## 🔍 Experimental Recommendations

### Recommended Testing Order

1. **Test baseline first**: Run `v1_original` as benchmark
2. **Test structured**: Run `v2_structured` to see if structure helps
3. **Test quality-focused**: Run `v7_quality_focused` to see impact of quality standards
4. **Compare concise version**: Run `v6_concise` to test brevity effectiveness
5. **Fine-tune**: Select best version based on results for further optimization

### Evaluation Dimensions

When comparing different prompt versions, focus on:

1. **Accuracy**: Does the generated documentation accurately reflect code functionality
2. **Completeness**: Does it cover important features and details
3. **Clarity**: Is the documentation easy to understand
4. **Format**: Is the markdown formatting correct
5. **Usefulness**: Does it include helpful examples and explanations
6. **Consistency**: Is the quality stable across multiple generations

### Debugging Tips

If a prompt version doesn't work well:

1. **Check generation logs**: Review log files in the `logs/` directory
2. **Sample checking**: Randomly review several generated documents
3. **Compare differences**: Compare with outputs from other versions
4. **Adjust details**: Fine-tune prompt content in `prompt_templates.py`

## 🎯 Improvement Suggestions

### Create Custom Prompt

Add a new version in `prompt_templates.py`:

```python
def get_prompt_v8_custom(intent, filename, code_context):
    """
    Your custom prompt description
    """
    prompt = f"""Your custom prompt template here...
    
    Intent: {intent}
    Filename: {filename}
    Code: {code_context}
    """
    return prompt

# Register in the version registry
PROMPT_VERSIONS['v8_custom'] = get_prompt_v8_custom
```

### Prompt Optimization Directions

Based on evaluation results, optimize in these directions:

1. **Increase/decrease level of detail**
2. **Adjust output format requirements**
3. **Add/remove examples**
4. **Change tone and role settings**
5. **Adjust constraints**
6. **Optimize instruction order**

## 📊 Expected Improvements

Different prompt versions may improve in different aspects:

- **v2/v3**: Better document structure and organization
- **v4**: Deeper code understanding and explanation
- **v5**: More consistent output format
- **v6**: Faster generation speed and lower cost
- **v7**: Overall higher documentation quality

## 🔧 Troubleshooting

### Common Issues

**Q: The prompt_version parameter doesn't work?**
A: Make sure the parameter is correctly set in the config file, check spelling

**Q: How to view the currently used prompt?**
A: Check the log files, they will record the prompt version used

**Q: Can I combine features from different prompts?**
A: Yes, create a new combined version in `prompt_templates.py`

## 📝 Next Steps

1. Run experiments to compare different prompt versions
2. Collect quality evaluation data
3. Analyze which improvements are most effective
4. Iteratively optimize the best prompt version
5. Validate on larger datasets

---

**Tip**: Different models (GPT-3.5, GPT-4, Llama, etc.) may respond differently to the same prompt. It's recommended to test your prompt improvements on multiple models.
