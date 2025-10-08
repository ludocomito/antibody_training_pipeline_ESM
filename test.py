#!/usr/bin/env python3
"""
Test Script for Antibody Classification Pipeline

This script provides testing functionality for trained antibody classifiers:
1. Load trained models from pickle files
2. Evaluate on test datasets with performance metrics
3. Generate confusion matrices and logging

Usage:
    python test.py --model models/antibody_classifier.pkl --data sample_data.csv
    python test.py --config test_config.yaml
"""

import os
import sys
import argparse
import logging
import pickle
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime

# Scikit-learn imports
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix,
    average_precision_score
)

# Local imports
from model import ESMEmbeddingExtractor
from classifier import BinaryClassifier
from data import preprocess_raw_data, load_preprocessed_data

# Configure matplotlib for better plots
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
sns.set_palette("husl")

@dataclass
class TestConfig:
    """Configuration for testing pipeline"""
    model_paths: List[str]
    data_paths: List[str]
    sequence_column: str = "sequence"  # Column name for sequences in dataset
    label_column: str = "label"        # Column name for labels in dataset
    output_dir: str = "./test_results"
    metrics: List[str] = None
    save_predictions: bool = True
    batch_size: int = 32  # Batch size for embedding extraction
    device: str = "cpu"  # Device to use for inference [cuda, cpu, mps]
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]

class ModelTester:
    """Model testing class"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.results = {}
        self.cached_embedding_files = []  # Track cached files for cleanup
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        # Create output directory if it doesn't exist
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        log_file = os.path.join(self.config.output_dir, f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        return logging.getLogger(__name__)
    
    def load_model(self, model_path: str) -> BinaryClassifier:
        """Load trained model from pickle file"""
        self.logger.info(f"Loading model from {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        if not isinstance(model, BinaryClassifier):
            raise ValueError(f"Expected BinaryClassifier, got {type(model)}")
        
        # Update device if different from config
        if hasattr(model, 'embedding_extractor') and model.embedding_extractor.device != self.config.device:
            self.logger.info(f"Updating device from {model.embedding_extractor.device} to {self.config.device}")
            # Recreate embedding extractor with new device
            from model import ESMEmbeddingExtractor
            batch_size = getattr(model, 'batch_size', 32)
            model.embedding_extractor = ESMEmbeddingExtractor(model.model_name, self.config.device, batch_size)
            model.device = self.config.device
        
        # Update batch_size if different from config
        if hasattr(model, 'embedding_extractor') and model.embedding_extractor.batch_size != self.config.batch_size:
            self.logger.info(f"Updating batch_size from {model.embedding_extractor.batch_size} to {self.config.batch_size}")
            model.embedding_extractor.batch_size = self.config.batch_size
        
        self.logger.info(f"Model loaded successfully: {model_path} on device: {model.embedding_extractor.device}")
        return model
    
    def load_dataset(self, data_path: str) -> Tuple[List[str], List[int]]:
        """Load dataset from CSV file using configured column names"""
        self.logger.info(f"Loading dataset from {data_path}")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset file not found: {data_path}")
        
        df = pd.read_csv(data_path)
        
        sequence_col = self.config.sequence_column
        label_col = self.config.label_column
        
        if sequence_col not in df.columns:
            raise ValueError(f"Sequence column '{sequence_col}' not found in dataset. Available columns: {list(df.columns)}")
        if label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found in dataset. Available columns: {list(df.columns)}")
        
        sequences = df[sequence_col].tolist()
        labels = df[label_col].tolist()
        
        self.logger.info(f"Loaded {len(sequences)} samples from {data_path} (sequence_col='{sequence_col}', label_col='{label_col}')")
        return sequences, labels
    
    def embed_sequences(self, sequences: List[str], model: BinaryClassifier, dataset_name: str) -> np.ndarray:
        """Extract embeddings for sequences using the model's embedding extractor"""
        cache_file = os.path.join(self.config.output_dir, f"{dataset_name}_test_embeddings.pkl")
        
        # Track this file for cleanup
        if cache_file not in self.cached_embedding_files:
            self.cached_embedding_files.append(cache_file)
        
        # Try to load from cache
        if os.path.exists(cache_file):
            self.logger.info(f"Loading cached embeddings from {cache_file}")
            with open(cache_file, 'rb') as f:
                embeddings = pickle.load(f)
            
            if len(embeddings) == len(sequences):
                return embeddings
            else:
                self.logger.warning("Cached embeddings size mismatch, recomputing...")
        
        # Extract embeddings
        self.logger.info(f"Extracting embeddings for {len(sequences)} sequences...")
        embeddings = model.embedding_extractor.extract_batch_embeddings(sequences)
        
        # Cache embeddings
        with open(cache_file, 'wb') as f:
            pickle.dump(embeddings, f)
        self.logger.info(f"Embeddings cached to {cache_file}")
        
        return embeddings
    
    def evaluate_pretrained(self, model: BinaryClassifier, X: np.ndarray, y: np.ndarray,
                           model_name: str, dataset_name: str) -> Dict[str, Any]:
        """Evaluate pretrained model directly on test set (no retraining)"""
        self.logger.info(f"Evaluating pretrained model {model_name} on {dataset_name}")
        
        # Get predictions using the pretrained model
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]
        
        # Calculate metrics
        results = {
            'test_scores': {},
            'predictions': {
                'y_true': y,
                'y_pred': y_pred,
                'y_proba': y_proba
            },
            'confusion_matrix': confusion_matrix(y, y_pred),
            'classification_report': classification_report(y, y_pred, output_dict=True)
        }
        
        # Calculate all requested metrics
        if 'accuracy' in self.config.metrics:
            results['test_scores']['accuracy'] = accuracy_score(y, y_pred)
        if 'precision' in self.config.metrics:
            results['test_scores']['precision'] = precision_score(y, y_pred, zero_division=0)
        if 'recall' in self.config.metrics:
            results['test_scores']['recall'] = recall_score(y, y_pred, zero_division=0)
        if 'f1' in self.config.metrics:
            results['test_scores']['f1'] = f1_score(y, y_pred, zero_division=0)
        if 'roc_auc' in self.config.metrics:
            results['test_scores']['roc_auc'] = roc_auc_score(y, y_proba)
        if 'pr_auc' in self.config.metrics:
            results['test_scores']['pr_auc'] = average_precision_score(y, y_proba)
        
        # Log results
        self.logger.info(f"Test results for {model_name} on {dataset_name}:")
        for metric, value in results['test_scores'].items():
            self.logger.info(f"  {metric}: {value:.4f}")
        
        return results
    
    
    def plot_confusion_matrix(self, results: Dict[str, Dict], dataset_name: str):
        """Create confusion matrix visualization"""
        self.logger.info(f"Creating confusion matrix for {dataset_name}")
        
        model_names = list(results.keys())
        n_models = len(model_names)
        
        # Create figure with subplots for confusion matrices
        fig_width = min(5 * n_models, 20)
        fig_height = 5
        fig, axes = plt.subplots(1, n_models, figsize=(fig_width, fig_height))
        
        # Handle single model case
        if n_models == 1:
            axes = [axes]
        
        # Plot confusion matrix for each model
        for idx, model_name in enumerate(model_names):
            cm = results[model_name]['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Negative', 'Positive'],
                       yticklabels=['Negative', 'Positive'],
                       ax=axes[idx])
            axes[idx].set_title(f'Confusion Matrix - {model_name}')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(self.config.output_dir, f'confusion_matrix_{dataset_name}.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Confusion matrix saved to {plot_file}")
    
    def save_detailed_results(self, results: Dict[str, Dict], dataset_name: str):
        """Save detailed results to files"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save main results
        results_file = os.path.join(self.config.output_dir, f'detailed_results_{dataset_name}_{timestamp}.yaml')
        with open(results_file, 'w') as f:
            yaml.dump({
                'dataset': dataset_name,
                'config': self.config.__dict__,
                'results': results
            }, f, default_flow_style=False)
        
        # Save predictions if requested
        if self.config.save_predictions:
            for model_name, model_results in results.items():
                if 'predictions' in model_results:
                    pred_file = os.path.join(self.config.output_dir, 
                                           f'predictions_{model_name}_{dataset_name}_{timestamp}.csv')
                    pred_df = pd.DataFrame({
                        'y_true': model_results['predictions']['y_true'],
                        'y_pred': model_results['predictions']['y_pred'],
                        'y_proba': model_results['predictions']['y_proba']
                    })
                    pred_df.to_csv(pred_file, index=False)
        
        self.logger.info(f"Detailed results saved to {results_file}")
    
    def cleanup_cached_embeddings(self):
        """Delete cached embedding files"""
        self.logger.info("Cleaning up cached embedding files...")
        for cache_file in self.cached_embedding_files:
            if os.path.exists(cache_file):
                try:
                    os.remove(cache_file)
                    self.logger.info(f"Deleted cached embeddings: {cache_file}")
                except Exception as e:
                    self.logger.warning(f"Failed to delete {cache_file}: {e}")
    
    def run_comprehensive_test(self):
        """Run testing pipeline"""
        self.logger.info("Starting model testing")
        self.logger.info(f"Models to test: {self.config.model_paths}")
        self.logger.info(f"Datasets to test: {self.config.data_paths}")
        
        all_results = {}
        
        try:
            # Test each dataset
            for data_path in self.config.data_paths:
                dataset_name = Path(data_path).stem
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"Testing on dataset: {dataset_name}")
                self.logger.info(f"{'='*60}")
                
                # Load dataset
                try:
                    sequences, labels = self.load_dataset(data_path)
                    labels = np.array(labels)
                except Exception as e:
                    self.logger.error(f"Failed to load dataset {data_path}: {e}")
                    continue
                
                dataset_results = {}
                
                # Test each model
                for model_path in self.config.model_paths:
                    model_name = Path(model_path).stem
                    self.logger.info(f"\nTesting model: {model_name}")
                    
                    try:
                        # Load model
                        model = self.load_model(model_path)
                        
                        # Extract embeddings
                        X_embedded = self.embed_sequences(sequences, model, f"{dataset_name}_{model_name}")
                        
                        # Direct evaluation on test set using pretrained model
                        test_results = self.evaluate_pretrained(model, X_embedded, labels, model_name, dataset_name)
                        dataset_results[model_name] = test_results
                        
                    except Exception as e:
                        self.logger.error(f"Failed to test model {model_path}: {e}")
                        continue
                
                # Create visualizations
                self.plot_confusion_matrix(dataset_results, dataset_name)
                
                # Save detailed results
                self.save_detailed_results(dataset_results, dataset_name)
                
                all_results[dataset_name] = dataset_results
            
            self.results = all_results
            self.logger.info(f"\nTesting completed. Results saved to: {self.config.output_dir}")
            
        finally:
            # Always cleanup cached embeddings
            self.cleanup_cached_embeddings()
        
        return all_results

def load_config_file(config_path: str) -> TestConfig:
    """Load test configuration from YAML file"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return TestConfig(**config_dict)

def create_sample_test_config():
    """Create a sample test configuration file"""
    sample_config = {
        'model_paths': ['./models/antibody_classifier.pkl'],
        'data_paths': ['./sample_data.csv'],
        'sequence_column': 'sequence',
        'label_column': 'label',
        'output_dir': './test_results',
        'metrics': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc'],
        'save_predictions': True
    }
    
    with open('test_config.yaml', 'w') as f:
        yaml.dump(sample_config, f, default_flow_style=False)
    
    print("Sample test configuration created: test_config.yaml")

def main():
    """Main function for the test script"""
    parser = argparse.ArgumentParser(
        description="Testing for antibody classification models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Test single model on single dataset
    python test.py --model models/antibody_classifier.pkl --data sample_data.csv
    
    # Test multiple models on multiple datasets
    python test.py --model models/model1.pkl models/model2.pkl --data dataset1.csv dataset2.csv
    
    # Use configuration file
    python test.py --config test_config.yaml
    
    # Override device and batch size
    python test.py --config test_config.yaml --device cuda --batch-size 64
    
    # Create sample configuration
    python test.py --create-config
        """
    )
    
    parser.add_argument('--model', nargs='+', help='Path(s) to trained model pickle files')
    parser.add_argument('--data', nargs='+', help='Path(s) to test dataset CSV files')
    parser.add_argument('--config', help='Path to test configuration YAML file')
    parser.add_argument('--output-dir', default='./test_results', help='Output directory for results')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], help='Device to use for inference (overrides config)')
    parser.add_argument('--batch-size', type=int, help='Batch size for embedding extraction (overrides config)')
    parser.add_argument('--create-config', action='store_true', help='Create sample configuration file')
    
    args = parser.parse_args()
    
    # Create sample config if requested
    if args.create_config:
        create_sample_test_config()
        return
    
    # Load configuration
    if args.config:
        config = load_config_file(args.config)
    else:
        if not args.model or not args.data:
            parser.error("Either --config or both --model and --data must be specified")
        
        config = TestConfig(
            model_paths=args.model,
            data_paths=args.data,
            output_dir=args.output_dir
        )
    
    # Override config with command line arguments
    if args.device:
        config.device = args.device
    if args.batch_size:
        config.batch_size = args.batch_size
    
    # Run testing
    try:
        tester = ModelTester(config)
        results = tester.run_comprehensive_test()
        
        print(f"\n{'='*60}")
        print("TESTING COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"Results saved to: {config.output_dir}")
        
        # Print summary
        for dataset_name, dataset_results in results.items():
            print(f"\nDataset: {dataset_name}")
            print("-" * 40)
            for model_name, model_results in dataset_results.items():
                print(f"Model: {model_name}")
                if 'test_scores' in model_results:
                    for metric, value in model_results['test_scores'].items():
                        print(f"  {metric}: {value:.4f}")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
