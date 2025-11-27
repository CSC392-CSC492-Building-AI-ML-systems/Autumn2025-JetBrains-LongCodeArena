#!/bin/bash
# Quick start script for testing different prompts
# Usage: bash quick_start_prompts.sh

echo "=========================================="
echo "Module Summarization - Prompt Testing"
echo "=========================================="
echo ""

# Check if poetry is installed
if ! command -v poetry &> /dev/null
then
    echo "❌ Poetry is not installed. Please install it first:"
    echo "   curl -sSL https://install.python-poetry.org | python3 -"
    exit 1
fi

echo "✓ Poetry found"
echo ""

# Check if dependencies are installed
echo "📦 Installing dependencies..."
poetry install
echo ""

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs
mkdir -p temp_configs
echo "✓ Directories created"
echo ""

# Check if API key is configured
if grep -q "YOUR_KEY_HERE" configs/config_openai.yaml; then
    echo "⚠️  Warning: API keys not configured!"
    echo "   Please edit configs/config_openai.yaml and add your API keys"
    echo ""
    echo "   Required keys:"
    echo "   - api_key: Your OpenAI API key"
    echo "   - hf_api_key: Your HuggingFace API key"
    echo ""
    read -p "Press Enter to continue anyway or Ctrl+C to exit..."
    echo ""
fi

# Show available prompt versions
echo "📝 Available Prompt Versions:"
echo "   v1_original      - Baseline (simple)"
echo "   v2_structured    - Clear structure with sections"
echo "   v3_detailed      - Comprehensive guidelines"
echo "   v4_role_based    - Expert role emphasis"
echo "   v5_example_guided- With output template"
echo "   v6_concise       - Short and direct"
echo "   v7_quality_focused - Quality standards emphasis"
echo ""

# Ask user what to do
echo "What would you like to do?"
echo "  1) Test a single prompt version"
echo "  2) Compare multiple prompt versions (batch test)"
echo "  3) View existing results"
echo "  4) Show detailed guide"
echo ""
read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        echo ""
        read -p "Enter prompt version (e.g., v2_structured): " version
        
        # Create temp config
        temp_config="temp_configs/config_test_$version.yaml"
        cp configs/config_openai.yaml "$temp_config"
        
        # Update config with prompt version
        if grep -q "prompt_version:" "$temp_config"; then
            sed -i.bak "s/prompt_version:.*/prompt_version: \"$version\"/" "$temp_config"
        else
            echo "prompt_version: \"$version\"" >> "$temp_config"
        fi
        
        # Update save directory
        sed -i.bak "s|save_dir:.*|save_dir: \"./predictions-$version\"|" "$temp_config"
        
        echo ""
        echo "🚀 Running generation with prompt version: $version"
        echo "   Config: $temp_config"
        echo "   Output: ./predictions-$version"
        echo ""
        
        poetry run python chatgpt.py --config="$temp_config"
        
        echo ""
        echo "✓ Generation complete!"
        echo "   Check outputs in: ./predictions-$version"
        ;;
        
    2)
        echo ""
        echo "Testing multiple versions: v1_original, v2_structured, v7_quality_focused"
        echo ""
        
        poetry run python batch_test_prompts.py \
            --config configs/config_openai.yaml \
            --versions v1_original v2_structured v7_quality_focused
        
        echo ""
        echo "✓ Batch testing complete!"
        echo ""
        echo "Compare results:"
        echo "  poetry run python compare_outputs.py \\"
        echo "    --dirs predictions-v1_original predictions-v2_structured predictions-v7_quality_focused \\"
        echo "    --sample 0"
        ;;
        
    3)
        echo ""
        echo "📊 Analyzing existing results..."
        echo ""
        
        # Find all prediction directories
        pred_dirs=$(find . -maxdepth 1 -type d -name "predictions-*" 2>/dev/null)
        
        if [ -z "$pred_dirs" ]; then
            echo "No prediction directories found."
            echo "Run generation first!"
        else
            echo "Found prediction directories:"
            echo "$pred_dirs"
            echo ""
            
            # Show summary
            for dir in $pred_dirs; do
                count=$(find "$dir" -name "*.txt" 2>/dev/null | wc -l)
                echo "  $dir: $count files"
            done
            
            echo ""
            echo "To compare outputs:"
            echo "  poetry run python compare_outputs.py --dirs [dir1] [dir2] --sample 0"
        fi
        ;;
        
    4)
        echo ""
        echo "📖 Opening detailed guide..."
        echo ""
        
        if command -v less &> /dev/null; then
            less PROMPT_GUIDE.md
        elif command -v more &> /dev/null; then
            more PROMPT_GUIDE.md
        else
            cat PROMPT_GUIDE.md
        fi
        ;;
        
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Done! Next steps:"
echo "  1. Review generated documentation"
echo "  2. Run metrics evaluation:"
echo "     poetry run python metrics.py --config=configs/config_eval.yaml"
echo "  3. See PROMPT_GUIDE.md for more details"
echo "=========================================="
