#!/bin/bash
# Batch run all prompt versions for GPT-5-nano

echo "======================================"
echo "Batch Generation: All 7 Prompt Versions"
echo "======================================"
echo ""

cd /Users/aceyang/Downloads/CSC492/repo/Autumn2025-JetBrains-LongCodeArena/module_summarization

versions=("v1" "v2" "v3" "v4" "v5" "v6" "v7")
names=("original" "structured" "detailed" "role-based" "example-guided" "concise" "quality-focused")

for i in "${!versions[@]}"; do
    version="${versions[$i]}"
    name="${names[$i]}"
    
    echo ""
    echo "======================================"
    echo "Running ${version} - ${name}"
    echo "======================================"
    echo ""
    
    poetry run python chatgpt.py --config="configs/config_gpt5nano_${version}.yaml"
    
    if [ $? -eq 0 ]; then
        echo "✅ ${version} completed successfully"
    else
        echo "❌ ${version} failed"
    fi
done

echo ""
echo "======================================"
echo "All generations completed!"
echo "======================================"
echo ""
echo "Generated directories:"
ls -d predictions-gpt5nano-* 2>/dev/null

echo ""
echo "To evaluate all results, run:"
echo "  export OPENAI_API_KEY='your_key'"
echo "  poetry run python metrics.py --config='configs/config_eval.yaml'"
