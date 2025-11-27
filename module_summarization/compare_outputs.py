#!/usr/bin/env python3
"""
Simple script to compare outputs from different prompt versions side-by-side.
Usage: python compare_outputs.py --dirs predictions-v1 predictions-v2 --sample 0
"""

import argparse
import os
from pathlib import Path


def read_output(directory, sample_id):
    """Read output file from a directory"""
    filepath = Path(directory) / f"{sample_id}.txt"
    if not filepath.exists():
        return f"[File not found: {filepath}]"
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except Exception as e:
        return f"[Error reading file: {e}]"


def compare_outputs(directories, sample_id):
    """Compare outputs from multiple directories"""
    print(f"\n{'='*80}")
    print(f"Comparing Sample #{sample_id}")
    print(f"{'='*80}\n")
    
    for i, directory in enumerate(directories, 1):
        print(f"\n{'-'*80}")
        print(f"Directory {i}: {directory}")
        print(f"{'-'*80}\n")
        
        content = read_output(directory, sample_id)
        
        # Print first 1000 characters for preview
        if len(content) > 1000:
            print(content[:1000])
            print(f"\n... (truncated, {len(content)} total characters)")
        else:
            print(content)
        
        print(f"\nLength: {len(content)} characters")
        print(f"Lines: {len(content.splitlines())}")


def analyze_directory(directory):
    """Analyze outputs in a directory"""
    path = Path(directory)
    
    if not path.exists():
        return {
            'exists': False,
            'count': 0,
            'avg_length': 0,
            'files': []
        }
    
    files = list(path.glob("*.txt"))
    
    if not files:
        return {
            'exists': True,
            'count': 0,
            'avg_length': 0,
            'files': []
        }
    
    lengths = []
    for file in files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                lengths.append(len(f.read()))
        except:
            pass
    
    return {
        'exists': True,
        'count': len(files),
        'avg_length': sum(lengths) / len(lengths) if lengths else 0,
        'min_length': min(lengths) if lengths else 0,
        'max_length': max(lengths) if lengths else 0,
        'files': [f.name for f in files[:5]]  # First 5 files
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare outputs from different prompt versions"
    )
    parser.add_argument(
        '--dirs',
        nargs='+',
        required=True,
        help='Directories to compare'
    )
    parser.add_argument(
        '--sample',
        type=int,
        default=0,
        help='Sample ID to compare (default: 0)'
    )
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Show summary statistics for all directories'
    )
    
    args = parser.parse_args()
    
    if args.summary:
        print(f"\n{'='*80}")
        print(f"Directory Summary Statistics")
        print(f"{'='*80}\n")
        
        for directory in args.dirs:
            stats = analyze_directory(directory)
            
            print(f"\nDirectory: {directory}")
            print(f"{'-'*80}")
            
            if not stats['exists']:
                print("  Status: Directory does not exist")
            elif stats['count'] == 0:
                print("  Status: No output files found")
            else:
                print(f"  Status: ✓ Active")
                print(f"  Files: {stats['count']}")
                print(f"  Avg Length: {stats['avg_length']:.0f} characters")
                print(f"  Min Length: {stats['min_length']} characters")
                print(f"  Max Length: {stats['max_length']} characters")
                print(f"  Sample Files: {', '.join(stats['files'][:3])}")
    else:
        compare_outputs(args.dirs, args.sample)
        
        print(f"\n{'='*80}")
        print(f"Tips:")
        print(f"  - Use --sample N to view different samples")
        print(f"  - Use --summary to see statistics for all directories")
        print(f"  - Compare multiple samples to assess consistency")
        print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
