#!/usr/bin/env python3
"""
Simple runner script for GEN baseline experiments
Usage: python run_gen_baseline.py [--config CONFIG_PATH] [--method METHOD]
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    parser = argparse.ArgumentParser(description='Run GEN baseline for OOD intent detection')
    parser.add_argument('--config', type=str, default='gen_config.json',
                       help='Path to configuration file')
    parser.add_argument('--method', type=str, 
                       choices=['random_token_replacement', 'random_shuffle', 
                               'random_sampling', 'class_mixing'],
                       help='Generation method to use (overrides config)')
    parser.add_argument('--dataset', type=str, choices=['snips', 'atis'],
                       help='Dataset to use (overrides config)')
    parser.add_argument('--evaluate-only', action='store_true',
                       help='Only run evaluation on existing model')
    parser.add_argument('--all-methods', action='store_true',
                       help='Run all generation methods for comparison')
    
    args = parser.parse_args()
    
    # Load configuration
    if not os.path.exists(args.config):
        print(f"Config file {args.config} not found. Creating default config...")
        create_default_config(args.config)
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Override config with command line arguments
    if args.method:
        config['gen_method'] = args.method
    if args.dataset:
        config['dataset'] = args.dataset
    
    print("Starting GEN baseline experiment...")
    print(f"Dataset: {config['dataset']}")
    print(f"Method: {config.get('gen_method', 'random_token_replacement')}")
    print(f"BERT Model: {config['bert']}")
    
    # Import here to avoid circular imports
    try:
        from gen_baseline import GENBaseline, run_gen_baseline
        from gen_evaluation import run_comprehensive_gen_evaluation
    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("Make sure all required files are in the same directory")
        return 1
    
    try:
        if args.all_methods:
            # Run all methods
            print("Running all generation methods...")
            results = run_gen_baseline(args.config)
            
        elif args.evaluate_only:
            # Only run evaluation
            print("Running evaluation only...")
            results, comparison = run_comprehensive_gen_evaluation(args.config)
            
        else:
            # Run single method
            gen_baseline = GENBaseline(config)
            results = gen_baseline.run_full_pipeline(
                generation_method=config.get('gen_method', 'random_token_replacement')
            )
            
            # Run detaile