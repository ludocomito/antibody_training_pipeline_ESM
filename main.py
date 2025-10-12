#!/usr/bin/env python3
"""
Main script for the Antibody Classification Pipeline using ESM-1V embeddings.

This script orchestrates the complete training pipeline:
1. Load configuration from YAML file
2. Setup data and model parameters
3. Train the binary classifier on ESM embeddings
4. Evaluate performance and save results

Usage:
    python main.py [config_path]
    
If no config_path is provided, defaults to 'config.yaml'
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from train import train_model

def setup_directories(config_path: str = "config.yaml"):
    """Create necessary directories for the pipeline"""
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create directories
    directories = [
        os.path.dirname(config['training']['log_file']),
        config['data']['embeddings_cache_dir'],
        os.path.dirname(config['training']['model_save_dir']),
    ]
    
    for directory in directories:
        if directory:
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")

def validate_config(config_path: str):
    """Validate that the configuration file exists and has required fields"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check for required fields
    required_fields = [
        ['model', 'name'],
        ['model', 'device'],
        ['classifier', 'max_iter'],
        ['classifier', 'random_state'],
        ['training', 'metrics'],
        ['data', 'sequence_column'],
        ['data', 'label_column'],
    ]
    
    for field_path in required_fields:
        current = config
        for field in field_path:
            if field not in current:
                raise ValueError(f"Missing required configuration field: {'.'.join(field_path)}")
            current = current[field]
    
    print(f"Configuration validated successfully: {config_path}")
    return config

def print_pipeline_info(config):
    """Print information about the pipeline configuration"""
    print("=" * 60)
    print("ANTIBODY CLASSIFICATION PIPELINE")
    print("=" * 60)
    print(f"ESM Model: {config['model']['name']}")
    print(f"Device: {config['model']['device']}")
    print(f"Classifier: {config['classifier']['type']}")
    print(f"Random State: {config['classifier']['random_state']}")
    print(f"Max Iterations: {config['classifier']['max_iter']}")
    print(f"Cross-validation Folds: {config['classifier']['cv_folds']}")
    print(f"Metrics: {', '.join(config['training']['metrics'])}")
    print(f"Log Level: {config['training']['log_level']}")
    
    if config['data']['train_file']:
        print(f"Training Data: {config['data']['train_file']}")
    else:
        print("Training Data: Not specified (will need to be set in config)")
    
    print("=" * 60)

def main():
    """Main function to run the antibody classification pipeline"""
    parser = argparse.ArgumentParser(
        description="Train antibody binary classifier using ESM-1V embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                    # Use default config.yaml
    python main.py my_config.yaml     # Use custom config file
    
The pipeline will:
1. Load protein sequences and binary labels from your data files
2. Extract ESM-1V embeddings for each sequence
3. Perform k-fold cross-validation on training data
4. Train final logistic regression classifier on full training set
5. Save trained model
        """
    )
    
    parser.add_argument(
        'config', 
        nargs='?', 
        default='config.yaml',
        help='Path to configuration YAML file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate configuration without running training'
    )
    
    parser.add_argument(
        '--setup-dirs',
        action='store_true',
        help='Setup required directories and exit'
    )
    
    
    args = parser.parse_args()
    
    try:
        # Validate configuration
        print(f"Using configuration file: {args.config}")
        config = validate_config(args.config)
        
        # Print pipeline information
        print_pipeline_info(config)
        
        # Setup directories
        if args.setup_dirs:
            setup_directories(args.config)
            print("Directory setup completed.")
            return
        
        # If only validating, exit here
        if args.validate_only:
            print("Configuration validation completed successfully.")
            return
        
        # Check if training data is specified
        if not config['data']['train_file']:
            print("\nWARNING: No training data file specified in configuration.")
            print("Please set 'data.train_file' in your config.yaml to point to your training data.")
            print("The training data should be a CSV file with columns:")
            print(f"  - '{config['data']['sequence_column']}': protein sequences")
            print(f"  - '{config['data']['label_column']}': binary labels (0/1)")
            print("\nExample CSV format:")
            print(f"{config['data']['sequence_column']},{config['data']['label_column']}")
            print("MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNEL,1")
            print("MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG,0")
            return
        
        # Setup directories
        setup_directories(args.config)
        
        # Run training pipeline
        print("\nStarting training pipeline...")
        results = train_model(args.config)
        
        # Print final results summary
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        # Print results
        if 'cv_metrics' in results:
            print("\nCross-Validation Results:")
            for metric, values in results['cv_metrics'].items():
                print(f"  {metric.upper()}: {values['mean']:.4f} (+/- {values['std']*2:.4f})")
        
        if 'train_metrics' in results:
            print("\nFinal Training Results:")
            for metric, value in results['train_metrics'].items():
                print(f"  {metric.upper()}: {value:.4f}")
        
        if results.get('model_path'):
            print(f"\nModel saved to: {results['model_path']}")
        
        print("\nCheck the log file for detailed training information.")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(1)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nFor help, run: python main.py --help")
        sys.exit(1)

if __name__ == "__main__":
    main()
