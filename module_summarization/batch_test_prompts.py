#!/usr/bin/env python3
"""
Batch test script to run multiple prompt versions and compare results.
Usage: python batch_test_prompts.py --config configs/config_openai.yaml --versions v1_original v2_structured v7_quality_focused
"""

import argparse
import os
import yaml
import subprocess
import json
from datetime import datetime


def load_config(config_path):
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_config(config, output_path):
    """Save configuration to YAML file"""
    with open(output_path, 'w') as f:
        yaml.dump(config, f)


def run_generation(config_path, script_name='chatgpt.py'):
    """Run generation with a specific config"""
    cmd = f"poetry run python {script_name} --config={config_path}"
    print(f"\n{'='*60}")
    print(f"Running: {cmd}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running {cmd}")
        print(f"STDERR: {result.stderr}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Batch test different prompt versions"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Base config file path'
    )
    parser.add_argument(
        '--versions',
        nargs='+',
        default=['v1_original', 'v2_structured', 'v7_quality_focused'],
        help='Prompt versions to test'
    )
    parser.add_argument(
        '--script',
        type=str,
        default='chatgpt.py',
        choices=['chatgpt.py', 'togetherai.py'],
        help='Script to run (chatgpt.py or togetherai.py)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to process (for testing)'
    )
    
    args = parser.parse_args()
    
    # Load base config
    base_config = load_config(args.config)
    
    # Create temp configs directory
    temp_dir = "temp_configs"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Track results
    results = {
        'timestamp': datetime.now().isoformat(),
        'base_config': args.config,
        'script': args.script,
        'versions_tested': [],
        'results': {}
    }
    
    print(f"\n{'#'*60}")
    print(f"# Batch Prompt Testing")
    print(f"# Base config: {args.config}")
    print(f"# Testing versions: {', '.join(args.versions)}")
    print(f"# Script: {args.script}")
    print(f"{'#'*60}\n")
    
    # Test each version
    for version in args.versions:
        print(f"\n{'='*60}")
        print(f"Testing prompt version: {version}")
        print(f"{'='*60}\n")
        
        # Create modified config
        test_config = base_config.copy()
        test_config['prompt_version'] = version
        
        # Modify save_dir to include version
        original_save_dir = base_config.get('save_dir', './predictions')
        test_config['save_dir'] = f"{original_save_dir}-{version}"
        
        # Save temporary config
        temp_config_path = os.path.join(temp_dir, f"config_{version}.yaml")
        save_config(test_config, temp_config_path)
        
        # Run generation
        success = run_generation(temp_config_path, args.script)
        
        results['versions_tested'].append(version)
        results['results'][version] = {
            'success': success,
            'save_dir': test_config['save_dir'],
            'config_path': temp_config_path
        }
        
        if success:
            print(f"✓ Version {version} completed successfully")
            print(f"  Output saved to: {test_config['save_dir']}")
        else:
            print(f"✗ Version {version} failed")
    
    # Save results summary
    results_file = f"batch_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"\nResults saved to: {results_file}\n")
    
    for version in args.versions:
        result = results['results'][version]
        status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
        print(f"{version:20s}: {status}")
        if result['success']:
            print(f"{'':20s}  → {result['save_dir']}")
    
    print(f"\n{'='*60}")
    print(f"Next steps:")
    print(f"  1. Review generated documentation in each output directory")
    print(f"  2. Run metrics evaluation:")
    print(f"     poetry run python metrics.py --config configs/config_eval.yaml")
    print(f"  3. Compare quality across different prompt versions")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
