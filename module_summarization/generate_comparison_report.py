#!/usr/bin/env python3
"""
Visual comparison tool for prompt outputs.
Generates a markdown report comparing different prompt versions.
Usage: python generate_comparison_report.py --dirs predictions-v1 predictions-v2 --output report.md
"""

import argparse
import os
from pathlib import Path
from datetime import datetime
import json


def read_output(directory, sample_id):
    """Read output file from a directory"""
    filepath = Path(directory) / f"{sample_id}.txt"
    if not filepath.exists():
        return None
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"[Error: {e}]"


def analyze_output(content):
    """Analyze output characteristics"""
    if content is None or content.startswith("[Error"):
        return None
    
    lines = content.splitlines()
    words = content.split()
    
    # Count markdown elements
    headers = sum(1 for line in lines if line.strip().startswith('#'))
    code_blocks = content.count('```')
    lists = sum(1 for line in lines if line.strip().startswith(('-', '*', '1.')))
    
    return {
        'length': len(content),
        'lines': len(lines),
        'words': len(words),
        'headers': headers,
        'code_blocks': code_blocks // 2,  # pairs of ```
        'lists': lists,
        'avg_line_length': len(content) / len(lines) if lines else 0
    }


def get_directory_stats(directory):
    """Get statistics for all files in a directory"""
    path = Path(directory)
    if not path.exists():
        return None
    
    files = list(path.glob("*.txt"))
    if not files:
        return None
    
    stats = []
    for file in files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
                analysis = analyze_output(content)
                if analysis:
                    stats.append(analysis)
        except:
            continue
    
    if not stats:
        return None
    
    # Calculate averages
    return {
        'count': len(stats),
        'avg_length': sum(s['length'] for s in stats) / len(stats),
        'avg_words': sum(s['words'] for s in stats) / len(stats),
        'avg_lines': sum(s['lines'] for s in stats) / len(stats),
        'avg_headers': sum(s['headers'] for s in stats) / len(stats),
        'avg_code_blocks': sum(s['code_blocks'] for s in stats) / len(stats),
        'avg_lists': sum(s['lists'] for s in stats) / len(stats),
    }


def generate_report(directories, samples, output_file):
    """Generate markdown comparison report"""
    
    report = []
    report.append("# Module Summarization - Prompt Comparison Report\n")
    report.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"**Directories**: {len(directories)}\n")
    report.append(f"**Sample IDs**: {', '.join(map(str, samples))}\n\n")
    
    # Overall statistics
    report.append("## 📊 Overall Statistics\n\n")
    report.append("| Directory | Files | Avg Length | Avg Words | Avg Lines | Avg Headers | Avg Code Blocks |\n")
    report.append("|-----------|-------|------------|-----------|-----------|-------------|----------------|\n")
    
    dir_stats = {}
    for directory in directories:
        stats = get_directory_stats(directory)
        dir_stats[directory] = stats
        
        if stats:
            report.append(f"| `{Path(directory).name}` | {stats['count']} | "
                        f"{stats['avg_length']:.0f} | {stats['avg_words']:.0f} | "
                        f"{stats['avg_lines']:.0f} | {stats['avg_headers']:.1f} | "
                        f"{stats['avg_code_blocks']:.1f} |\n")
        else:
            report.append(f"| `{Path(directory).name}` | - | - | - | - | - | - |\n")
    
    report.append("\n")
    
    # Sample comparisons
    report.append("## 📝 Sample Comparisons\n\n")
    
    for sample_id in samples:
        report.append(f"### Sample #{sample_id}\n\n")
        
        outputs = {}
        for directory in directories:
            content = read_output(directory, sample_id)
            outputs[directory] = content
        
        # Show side-by-side comparison
        for i, (directory, content) in enumerate(outputs.items(), 1):
            report.append(f"#### Version {i}: `{Path(directory).name}`\n\n")
            
            if content is None:
                report.append("*File not found*\n\n")
                continue
            
            # Show analysis
            analysis = analyze_output(content)
            if analysis:
                report.append(f"**Stats**: {analysis['words']} words, "
                            f"{analysis['lines']} lines, "
                            f"{analysis['headers']} headers, "
                            f"{analysis['code_blocks']} code blocks\n\n")
            
            # Show preview (first 500 chars)
            preview_length = 500
            if len(content) > preview_length:
                preview = content[:preview_length] + "\n\n*... (truncated)*"
            else:
                preview = content
            
            report.append("**Content Preview**:\n\n")
            report.append("```\n")
            report.append(preview)
            report.append("\n```\n\n")
            
            # Full content in collapsible section
            report.append("<details>\n")
            report.append("<summary>📄 View Full Content</summary>\n\n")
            report.append("```\n")
            report.append(content)
            report.append("\n```\n")
            report.append("</details>\n\n")
        
        report.append("---\n\n")
    
    # Recommendations
    report.append("## 💡 Analysis & Recommendations\n\n")
    
    if dir_stats:
        # Find version with most structure
        structured = max(
            [(d, s) for d, s in dir_stats.items() if s],
            key=lambda x: x[1]['avg_headers'] + x[1]['avg_code_blocks'],
            default=(None, None)
        )
        
        if structured[0]:
            report.append(f"- **Most structured output**: `{Path(structured[0]).name}` "
                        f"({structured[1]['avg_headers']:.1f} headers, "
                        f"{structured[1]['avg_code_blocks']:.1f} code blocks)\n")
        
        # Find most comprehensive
        comprehensive = max(
            [(d, s) for d, s in dir_stats.items() if s],
            key=lambda x: x[1]['avg_words'],
            default=(None, None)
        )
        
        if comprehensive[0]:
            report.append(f"- **Most comprehensive**: `{Path(comprehensive[0]).name}` "
                        f"({comprehensive[1]['avg_words']:.0f} words on average)\n")
        
        # Find most concise
        concise = min(
            [(d, s) for d, s in dir_stats.items() if s],
            key=lambda x: x[1]['avg_words'],
            default=(None, None)
        )
        
        if concise[0]:
            report.append(f"- **Most concise**: `{Path(concise[0]).name}` "
                        f"({concise[1]['avg_words']:.0f} words on average)\n")
    
    report.append("\n### Next Steps\n\n")
    report.append("1. Review the sample outputs above\n")
    report.append("2. Run quantitative evaluation: `poetry run python metrics.py --config configs/config_eval.yaml`\n")
    report.append("3. Based on CompScore and manual review, select the best prompt version\n")
    report.append("4. Consider creating a hybrid prompt combining strengths of multiple versions\n")
    
    # Write report
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(report)
    
    print(f"✓ Report generated: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate comparison report for different prompt versions"
    )
    parser.add_argument(
        '--dirs',
        nargs='+',
        required=True,
        help='Directories to compare'
    )
    parser.add_argument(
        '--samples',
        nargs='+',
        type=int,
        default=[0, 1, 2],
        help='Sample IDs to include in report (default: 0 1 2)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='comparison_report.md',
        help='Output report filename (default: comparison_report.md)'
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("Generating Comparison Report")
    print(f"{'='*60}\n")
    print(f"Directories: {len(args.dirs)}")
    print(f"Samples: {args.samples}")
    print(f"Output: {args.output}\n")
    
    generate_report(args.dirs, args.samples, args.output)
    
    print(f"\n{'='*60}")
    print("Report ready! Open it to compare prompt versions:")
    print(f"  open {args.output}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
