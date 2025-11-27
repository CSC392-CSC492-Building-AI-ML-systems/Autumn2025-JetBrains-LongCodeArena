# Module Summarization - Prompt Improvement Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Module Summarization                         │
│                   Prompt Engineering System                     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Core Components                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  📝 prompt_templates.py                                          │
│  ├── v1_original       (Baseline version)                       │
│  ├── v2_structured     (Structured + clear requirements)        │
│  ├── v3_detailed       (Detailed guidance + style guidelines)   │
│  ├── v4_role_based     (Role-playing + thought process)         │
│  ├── v5_example_guided (Output template + format examples)      │
│  ├── v6_concise        (Concise + cost optimized)               │
│  └── v7_quality_focused(Quality standards + best practices)     │
│                                                                  │
│  🔧 Generation Scripts (Modified)                                │
│  ├── chatgpt.py        (Supports prompt_version parameter)      │
│  └── togetherai.py     (Supports prompt_version parameter)      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Configuration Files                                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  configs/                                                        │
│  ├── config_openai.yaml      (Base configuration)               │
│  ├── config_openai_v2.yaml   (Test v2_structured)               │
│  ├── config_openai_v3.yaml   (Test v3_detailed)                 │
│  └── config_openai_v7.yaml   (Test v7_quality_focused)          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Tool Scripts                                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  🚀 batch_test_prompts.py                                        │
│     Batch test multiple prompt versions                         │
│     Output: batch_test_results_*.json                           │
│                                                                  │
│  📊 compare_outputs.py                                           │
│     Compare outputs from different versions                     │
│     Features: Statistics + sample comparison                    │
│                                                                  │
│  📄 generate_comparison_report.py                                │
│     Generate detailed Markdown comparison report                │
│     Output: comparison_report.md                                │
│                                                                  │
│  🎯 quick_start_prompts.sh                                       │
│     Interactive quick start script                              │
│     Menu: Test / Batch / View / Help                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Documentation                                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  📖 PROMPT_GUIDE.md           (Detailed usage guide)             │
│  📖 CHANGES_SUMMARY.md        (Improvements and architecture)    │
│  📖 QUICK_START_EXAMPLES.md   (Quick start examples)             │
│  📖 README.md                 (Updated with prompt testing info) │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Workflow                                                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1️⃣  Configure API Keys                                         │
│      └─> Edit configs/config_openai.yaml                        │
│                                                                  │
│  2️⃣  Choose Testing Method                                      │
│      ├─> Interactive: bash quick_start_prompts.sh               │
│      ├─> Single version: poetry run python chatgpt.py --config  │
│      └─> Batch: poetry run python batch_test_prompts.py ...    │
│                                                                  │
│  3️⃣  Generate Documentation                                     │
│      └─> Output to predictions-{version}/ directory             │
│                                                                  │
│  4️⃣  Compare and Analyze                                        │
│      ├─> Quick compare: compare_outputs.py --dirs ... --sample  │
│      ├─> Statistics: compare_outputs.py --dirs ... --summary    │
│      └─> Report: generate_comparison_report.py --dirs ...       │
│                                                                  │
│  5️⃣  Quantitative Evaluation                                    │
│      └─> poetry run python metrics.py --config=...              │
│                                                                  │
│  6️⃣  Select Best Version & Iterate                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Prompt Improvement Strategies                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  v1 (Baseline)                                                   │
│   └─> Simple direct instructions                                │
│                                                                  │
│  v2 (Structured)                                                 │
│   ├─> Added clear sections                                      │
│   ├─> Used numbered lists                                       │
│   └─> Clarified format requirements                             │
│                                                                  │
│  v3 (Detailed)                                                   │
│   ├─> Refined content requirements                              │
│   ├─> Style guidelines                                          │
│   └─> Output format specifications                              │
│                                                                  │
│  v4 (Role-based)                                                 │
│   ├─> Expert role setting                                       │
│   ├─> Step-by-step approach                                     │
│   └─> Emphasized understanding depth                            │
│                                                                  │
│  v5 (Example-guided)                                             │
│   ├─> Provided output template                                  │
│   ├─> Structured headings                                       │
│   └─> Format examples                                           │
│                                                                  │
│  v6 (Concise)                                                    │
│   ├─> Retained core requirements                                │
│   ├─> Reduced redundancy                                        │
│   └─> Lowered token costs                                       │
│                                                                  │
│  v7 (Quality-focused)                                            │
│   ├─> Quality standards checklist                               │
│   ├─> Best practices                                            │
│   └─> Specific style guidance                                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Evaluation Dimensions                                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ✓ Accuracy     - Correctly understands code                     │
│  ✓ Completeness - Covers important features                      │
│  ✓ Clarity      - Easy to understand                             │
│  ✓ Format       - Correct markdown formatting                    │
│  ✓ Usefulness   - Includes helpful examples                      │
│  ✓ Consistency  - Stable across multiple generations             │
│  ✓ CompScore    - LLM-based quantitative evaluation              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Quick Command Reference                                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  # Interactive start                                             │
│  bash quick_start_prompts.sh                                    │
│                                                                  │
│  # Test single version                                           │
│  poetry run python chatgpt.py \                                 │
│    --config="configs/config_openai_v2.yaml"                     │
│                                                                  │
│  # Batch testing                                                 │
│  poetry run python batch_test_prompts.py \                      │
│    --config configs/config_openai.yaml \                        │
│    --versions v1_original v2_structured v7_quality_focused     │
│                                                                  │
│  # Compare outputs                                               │
│  poetry run python compare_outputs.py \                         │
│    --dirs predictions-v1 predictions-v7 --sample 0             │
│                                                                  │
│  # Generate report                                               │
│  poetry run python generate_comparison_report.py \              │
│    --dirs predictions-v1 predictions-v7 \                       │
│    --samples 0 1 2 --output report.md                          │
│                                                                  │
│  # Run evaluation                                                │
│  poetry run python metrics.py \                                 │
│    --config="configs/config_eval.yaml"                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Custom Extensions                                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Add new version in prompt_templates.py:                        │
│                                                                  │
│  def get_prompt_v8_custom(intent, filename, code_context):      │
│      """Your custom prompt"""                                   │
│      prompt = f"""                                               │
│      Your innovative design here...                             │
│      """                                                         │
│      return prompt                                               │
│                                                                  │
│  # Register                                                      │
│  PROMPT_VERSIONS['v8_custom'] = get_prompt_v8_custom            │
│                                                                  │
│  # Use                                                           │
│  prompt_version: "v8_custom"                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

                        🎯 Start Optimizing Your Prompts!
```
