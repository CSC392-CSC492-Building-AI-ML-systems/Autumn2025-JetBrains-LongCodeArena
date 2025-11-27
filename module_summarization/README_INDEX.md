# 📚 Module Summarization Prompt Improvements - Complete Guide Index

## 🎯 What Do You Want to Do?

### 1️⃣ I want to quickly start testing different prompts
👉 **Read**: [QUICK_START_EXAMPLES.md](QUICK_START_EXAMPLES.md)  
👉 **Run**: `bash quick_start_prompts.sh`

### 2️⃣ I want to understand available prompt versions and how to improve them
👉 **Read**: [PROMPT_GUIDE.md](PROMPT_GUIDE.md)  
👉 **View code**: `prompt_templates.py`

### 3️⃣ I want to know what changes were made
👉 **Read**: [CHANGES_SUMMARY.md](CHANGES_SUMMARY.md)  
👉 **View architecture**: [ARCHITECTURE.md](ARCHITECTURE.md)

### 4️⃣ I want to understand the system architecture
👉 **Read**: [ARCHITECTURE.md](ARCHITECTURE.md)

### 5️⃣ I want basic usage instructions
👉 **Read**: [README.md](README.md)

---

## 📁 File Descriptions

### Core Code
| File | Description | Purpose |
|------|-------------|---------|
| `prompt_templates.py` | 7 prompt versions | Define different prompt strategies |
| `chatgpt.py` | OpenAI generation script | Generate docs using OpenAI API |
| `togetherai.py` | Together.AI generation script | Generate docs using Together.AI API |
| `metrics.py` | Evaluation script | Calculate CompScore and other metrics |

### Tool Scripts
| File | Description | Command Example |
|------|-------------|-----------------|
| `batch_test_prompts.py` | Batch testing tool | `python batch_test_prompts.py --config ... --versions v1 v2` |
| `compare_outputs.py` | Output comparison tool | `python compare_outputs.py --dirs dir1 dir2 --sample 0` |
| `generate_comparison_report.py` | Report generation tool | `python generate_comparison_report.py --dirs dir1 dir2` |
| `quick_start_prompts.sh` | Interactive startup script | `bash quick_start_prompts.sh` |

### Configuration Files
| File | Description | Prompt Version |
|------|-------------|----------------|
| `configs/config_openai.yaml` | Base configuration | v1_original (default) |
| `configs/config_openai_v2.yaml` | v2 test configuration | v2_structured |
| `configs/config_openai_v3.yaml` | v3 test configuration | v3_detailed |
| `configs/config_openai_v7.yaml` | v7 test configuration | v7_quality_focused |

### Documentation
| File | Description | Target Audience |
|------|-------------|-----------------|
| `QUICK_START_EXAMPLES.md` | Quick start examples | Beginners, quick start needed |
| `PROMPT_GUIDE.md` | Detailed prompt guide | Want to understand each version |
| `CHANGES_SUMMARY.md` | Improvements summary | Want to know what changed |
| `ARCHITECTURE.md` | System architecture | Want to understand overall design |
| `README.md` | Project description | Everyone |
| `README_INDEX.md` | This document index | Don't know where to start |

---

## 🚀 3-Minute Quick Start

```bash
# 1. Enter directory
cd module_summarization

# 2. Install dependencies
poetry install

# 3. Configure API key (edit file)
# Edit configs/config_openai.yaml and fill in your API keys

# 4. Run interactive script
bash quick_start_prompts.sh

# 5. Select option 2 (batch testing)
# This will automatically test 3 different prompt versions

# 6. Compare results
poetry run python compare_outputs.py \
  --dirs predictions-v1_original predictions-v2_structured predictions-v7_quality_focused \
  --summary
```

---

## 📊 Quick Comparison of Prompt Versions

| Version | Features | Advantages | Use Case |
|---------|----------|------------|----------|
| **v1_original** | Baseline, simple | Fast, direct | Benchmark comparison |
| **v2_structured** | Structured | Well-organized | General docs ⭐ |
| **v3_detailed** | Detailed guidance | Comprehensive | Complex modules |
| **v4_role_based** | Role-playing | Deep understanding | Core logic |
| **v5_example_guided** | Template-guided | Uniform format | Standardized docs |
| **v6_concise** | Concise, efficient | Low cost | Bulk generation |
| **v7_quality_focused** | Quality-oriented | High professionalism | Production docs ⭐ |

⭐ = Recommended to test first

---

## 🎯 Recommended Learning Path

### Step 1: Understand Basics (5 minutes)
1. Read first half of `README.md`
2. View `ARCHITECTURE.md` to understand overall structure

### Step 2: Hands-on Practice (15 minutes)
1. Read `QUICK_START_EXAMPLES.md`
2. Run `bash quick_start_prompts.sh`
3. Test 1-2 prompt versions

### Step 3: Deep Understanding (20 minutes)
1. Read `PROMPT_GUIDE.md` to understand each version
2. View `prompt_templates.py` code implementation
3. Compare outputs from different versions

### Step 4: Optimize and Improve (30+ minutes)
1. Read `CHANGES_SUMMARY.md` to understand improvement approach
2. Batch test multiple versions
3. Generate comparison report
4. Optimize prompts based on results

---

## 💡 Quick FAQ

### Q1: Which prompt version should I use?
**A**: First test **v2_structured** and **v7_quality_focused** - they're the most balanced versions.

### Q2: How to quickly compare two versions?
**A**: Use `compare_outputs.py --dirs dir1 dir2 --sample 0`

### Q3: How to batch test multiple versions?
**A**: Use `batch_test_prompts.py --versions v1_original v2_structured v7_quality_focused`

### Q4: Where is the generated documentation?
**A**: In the directory specified by `save_dir` in config file, e.g., `predictions-v2_structured/`

### Q5: How to evaluate which version is best?
**A**: 
1. Use `compare_outputs.py --summary` to see statistics
2. Use `generate_comparison_report.py` to generate detailed report
3. Use `metrics.py` to calculate CompScore
4. Manually review several samples

### Q6: Can I create my own prompt version?
**A**: Yes! Add new function in `prompt_templates.py`, register in `PROMPT_VERSIONS`

### Q7: Are there big differences between models?
**A**: Yes, GPT-4 is usually better than GPT-3.5, recommend testing both

### Q8: What about cost?
**A**: Depends on model and prompt length. v6_concise can reduce costs.

---

## 🔗 Quick Command Reference

```bash
# Interactive start
bash quick_start_prompts.sh

# Test single version
poetry run python chatgpt.py --config="configs/config_openai_v2.yaml"

# Batch testing
poetry run python batch_test_prompts.py \
  --config configs/config_openai.yaml \
  --versions v1_original v2_structured v7_quality_focused

# Quick comparison
poetry run python compare_outputs.py \
  --dirs predictions-v1_original predictions-v7_quality_focused \
  --sample 0

# Statistics
poetry run python compare_outputs.py \
  --dirs predictions-* \
  --summary

# Generate report
poetry run python generate_comparison_report.py \
  --dirs predictions-v1_original predictions-v7_quality_focused \
  --samples 0 1 2 \
  --output report.md

# Run evaluation
poetry run python metrics.py --config="configs/config_eval.yaml"
```

---

## 📈 Successful Workflow Example

```
1. Configure API Keys
   └─> Edit configs/config_openai.yaml

2. Batch Testing (Automated)
   └─> bash quick_start_prompts.sh → Option 2
   └─> Wait for generation to complete

3. Quick Review
   └─> python compare_outputs.py --dirs predictions-* --summary

4. Detailed Comparison
   └─> python generate_comparison_report.py --dirs predictions-* --output report.md
   └─> open report.md

5. Quantitative Evaluation
   └─> python metrics.py --config=configs/config_eval.yaml

6. Select Best Version
   └─> Based on report + CompScore + manual review

7. Production Use
   └─> Set best prompt_version in config
   └─> Large-scale generation
```

---

## 🎓 Advanced Topics

### Creating Custom Prompts
Refer to "Create Custom Prompt" section in `PROMPT_GUIDE.md`

### Tuning Tips
Refer to "Prompt Engineering Best Practices" in `CHANGES_SUMMARY.md`

### Few-Shot Learning
Can add examples in prompt, refer to v5_example_guided design

### Iterative Optimization Process
1. Baseline test → 2. Hypothesis improvement → 3. Implement and test → 4. Evaluate and compare → 5. Repeat

---

## 🆘 Need Help?

1. **View documentation**: Check corresponding `.md` files first
2. **Check logs**: Review log files in `logs/` directory
3. **Run examples**: Follow `QUICK_START_EXAMPLES.md` instructions
4. **View code**: All code has comments

---

## ✅ Next Steps

- [ ] Read `QUICK_START_EXAMPLES.md`
- [ ] Configure API keys
- [ ] Run `bash quick_start_prompts.sh`
- [ ] Test v2_structured and v7_quality_focused
- [ ] Generate comparison report
- [ ] Select best version
- [ ] Start large-scale generation!

---

**Good luck finding the best prompt to generate high-quality module documentation! 🚀**
