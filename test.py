#!/usr/bin/env python3
"""
Test Script for Antibody Classification Pipeline

This script provides comprehensive testing functionality for trained antibody classifiers:
1. Load trained models from pickle files
2. Perform k-fold cross-validation on multiple datasets
3. Generate detailed performance metrics and visualizations
4. Compare multiple models statistically

Usage:
    python test.py --model models/antibody_classifier.pkl --data sample_data.csv
    python test.py --config test_config.yaml
    python test.py --model models/model1.pkl --compare models/model2.pkl --data dataset1.csv dataset2.csv
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
from sklearn.model_selection import (
    StratifiedKFold, KFold, cross_val_score, cross_validate
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import StandardScaler
from scipy import stats

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
    output_dir: str = "./test_results"
    cv_folds: int = 5
    cv_strategy: str = "stratified"  # "stratified" or "normal"
    random_state: int = 42
    metrics: List[str] = None
    plot_results: bool = True
    save_predictions: bool = True
    statistical_tests: bool = True
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]

class ModelTester:
    """Comprehensive model testing class"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.results = {}
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
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
        
        self.logger.info(f"Model loaded successfully: {model_path}")
        return model
    
    def load_dataset(self, data_path: str, sequence_col: str = "sequence", label_col: str = "label") -> Tuple[List[str], List[int]]:
        """Load dataset from CSV file"""
        self.logger.info(f"Loading dataset from {data_path}")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset file not found: {data_path}")
        
        df = pd.read_csv(data_path)
        
        if sequence_col not in df.columns:
            raise ValueError(f"Sequence column '{sequence_col}' not found in dataset")
        if label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found in dataset")
        
        sequences = df[sequence_col].tolist()
        labels = df[label_col].tolist()
        
        self.logger.info(f"Loaded {len(sequences)} samples from {data_path}")
        return sequences, labels
    
    def embed_sequences(self, sequences: List[str], model: BinaryClassifier, dataset_name: str) -> np.ndarray:
        """Extract embeddings for sequences using the model's embedding extractor"""
        cache_file = os.path.join(self.config.output_dir, f"{dataset_name}_test_embeddings.pkl")
        
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
    
    def perform_kfold_cv(self, model: BinaryClassifier, X: np.ndarray, y: np.ndarray, 
                        model_name: str, dataset_name: str) -> Dict[str, Any]:
        """Perform k-fold cross-validation"""
        self.logger.info(f"Performing {self.config.cv_folds}-fold CV for {model_name} on {dataset_name}")
        
        # Setup cross-validation
        if self.config.cv_strategy == "stratified":
            cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, 
                               random_state=self.config.random_state)
        else:
            cv = KFold(n_splits=self.config.cv_folds, shuffle=True, 
                      random_state=self.config.random_state)
        
        # Define scoring metrics
        scoring_metrics = {
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall', 
            'f1': 'f1',
            'roc_auc': 'roc_auc'
        }
        
        # Perform cross-validation
        cv_results = cross_validate(
            model.classifier, X, y, cv=cv, 
            scoring=list(scoring_metrics.values()),
            return_train_score=True,
            return_estimator=False
        )
        
        # Calculate additional metrics manually for each fold
        fold_metrics = {metric: [] for metric in self.config.metrics}
        all_y_true = []
        all_y_pred = []
        all_y_proba = []
        
        for train_idx, test_idx in cv.split(X, y):
            X_train_fold, X_test_fold = X[train_idx], X[test_idx]
            y_train_fold, y_test_fold = y[train_idx], y[test_idx]
            
            # Create a new classifier for this fold
            temp_classifier = BinaryClassifier({
                'model_name': model.embedding_extractor.model_name,
                'device': model.embedding_extractor.device,
                'random_state': self.config.random_state,
                'max_iter': 1000
            })
            
            # Fit on training fold
            temp_classifier.fit(X_train_fold, y_train_fold)
            
            # Predict on test fold
            y_pred_fold = temp_classifier.predict(X_test_fold)
            y_proba_fold = temp_classifier.predict_proba(X_test_fold)[:, 1]
            
            # Store predictions for overall metrics
            all_y_true.extend(y_test_fold)
            all_y_pred.extend(y_pred_fold)
            all_y_proba.extend(y_proba_fold)
            
            # Calculate metrics for this fold
            if 'accuracy' in self.config.metrics:
                fold_metrics['accuracy'].append(accuracy_score(y_test_fold, y_pred_fold))
            if 'precision' in self.config.metrics:
                fold_metrics['precision'].append(precision_score(y_test_fold, y_pred_fold, zero_division=0))
            if 'recall' in self.config.metrics:
                fold_metrics['recall'].append(recall_score(y_test_fold, y_pred_fold, zero_division=0))
            if 'f1' in self.config.metrics:
                fold_metrics['f1'].append(f1_score(y_test_fold, y_pred_fold, zero_division=0))
            if 'roc_auc' in self.config.metrics:
                fold_metrics['roc_auc'].append(roc_auc_score(y_test_fold, y_proba_fold))
            if 'pr_auc' in self.config.metrics:
                fold_metrics['pr_auc'].append(average_precision_score(y_test_fold, y_proba_fold))
        
        # Compile results
        results = {
            'cv_scores': {},
            'fold_predictions': {
                'y_true': np.array(all_y_true),
                'y_pred': np.array(all_y_pred),
                'y_proba': np.array(all_y_proba)
            },
            'confusion_matrix': confusion_matrix(all_y_true, all_y_pred),
            'classification_report': classification_report(all_y_true, all_y_pred, output_dict=True)
        }
        
        # Calculate mean and std for each metric
        for metric in self.config.metrics:
            if metric in fold_metrics:
                scores = np.array(fold_metrics[metric])
                results['cv_scores'][metric] = {
                    'mean': scores.mean(),
                    'std': scores.std(),
                    'scores': scores.tolist()
                }
        
        # Log results
        self.logger.info(f"Cross-validation results for {model_name} on {dataset_name}:")
        for metric, values in results['cv_scores'].items():
            self.logger.info(f"  {metric}: {values['mean']:.4f} (+/- {values['std']*2:.4f})")
        
        return results
    
    def compare_models_statistically(self, results: Dict[str, Dict]) -> Dict[str, Any]:
        """Perform statistical tests to compare models"""
        if not self.config.statistical_tests or len(results) < 2:
            return {}
        
        self.logger.info("Performing statistical comparison of models")
        
        model_names = list(results.keys())
        comparisons = {}
        
        # Compare each pair of models
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names[i+1:], i+1):
                comparison_key = f"{model1}_vs_{model2}"
                comparisons[comparison_key] = {}
                
                # Compare each metric
                for metric in self.config.metrics:
                    if metric in results[model1]['cv_scores'] and metric in results[model2]['cv_scores']:
                        scores1 = results[model1]['cv_scores'][metric]['scores']
                        scores2 = results[model2]['cv_scores'][metric]['scores']
                        
                        # Paired t-test
                        t_stat, p_value = stats.ttest_rel(scores1, scores2)
                        
                        # Wilcoxon signed-rank test (non-parametric alternative)
                        w_stat, w_p_value = stats.wilcoxon(scores1, scores2)
                        
                        comparisons[comparison_key][metric] = {
                            't_test': {'statistic': t_stat, 'p_value': p_value},
                            'wilcoxon': {'statistic': w_stat, 'p_value': w_p_value},
                            'significant': p_value < 0.05,
                            'effect_size': (np.mean(scores1) - np.mean(scores2)) / np.sqrt((np.var(scores1) + np.var(scores2)) / 2)
                        }
        
        return comparisons
    
    def plot_results(self, results: Dict[str, Dict], dataset_name: str):
        """Create visualization plots for results"""
        if not self.config.plot_results:
            return
        
        self.logger.info(f"Creating visualization plots for {dataset_name}")
        
        n_models = len(results)
        model_names = list(results.keys())
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. ROC Curves
        plt.subplot(3, 3, 1)
        for model_name in model_names:
            if 'fold_predictions' in results[model_name]:
                y_true = results[model_name]['fold_predictions']['y_true']
                y_proba = results[model_name]['fold_predictions']['y_proba']
                fpr, tpr, _ = roc_curve(y_true, y_proba)
                auc_score = roc_auc_score(y_true, y_proba)
                plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Precision-Recall Curves
        plt.subplot(3, 3, 2)
        for model_name in model_names:
            if 'fold_predictions' in results[model_name]:
                y_true = results[model_name]['fold_predictions']['y_true']
                y_proba = results[model_name]['fold_predictions']['y_proba']
                precision, recall, _ = precision_recall_curve(y_true, y_proba)
                pr_auc = average_precision_score(y_true, y_proba)
                plt.plot(recall, precision, label=f'{model_name} (PR-AUC = {pr_auc:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Cross-validation scores comparison
        metrics_to_plot = ['accuracy', 'f1', 'roc_auc']
        for idx, metric in enumerate(metrics_to_plot):
            plt.subplot(3, 3, 3 + idx)
            
            model_scores = []
            model_labels = []
            
            for model_name in model_names:
                if metric in results[model_name]['cv_scores']:
                    scores = results[model_name]['cv_scores'][metric]['scores']
                    model_scores.append(scores)
                    model_labels.append(model_name)
            
            if model_scores:
                plt.boxplot(model_scores, labels=model_labels)
                plt.ylabel(metric.upper())
                plt.title(f'{metric.upper()} Distribution')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
        
        # 4. Confusion matrices
        for idx, model_name in enumerate(model_names[:4]):  # Limit to 4 models
            plt.subplot(3, 3, 6 + idx)
            cm = results[model_name]['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Negative', 'Positive'],
                       yticklabels=['Negative', 'Positive'])
            plt.title(f'Confusion Matrix - {model_name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(self.config.output_dir, f'results_{dataset_name}.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Plots saved to {plot_file}")
    
    def save_detailed_results(self, results: Dict[str, Dict], dataset_name: str, 
                            statistical_comparisons: Dict[str, Any]):
        """Save detailed results to files"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save main results
        results_file = os.path.join(self.config.output_dir, f'detailed_results_{dataset_name}_{timestamp}.yaml')
        with open(results_file, 'w') as f:
            yaml.dump({
                'dataset': dataset_name,
                'config': self.config.__dict__,
                'results': results,
                'statistical_comparisons': statistical_comparisons
            }, f, default_flow_style=False)
        
        # Save predictions if requested
        if self.config.save_predictions:
            for model_name, model_results in results.items():
                if 'fold_predictions' in model_results:
                    pred_file = os.path.join(self.config.output_dir, 
                                           f'predictions_{model_name}_{dataset_name}_{timestamp}.csv')
                    pred_df = pd.DataFrame({
                        'y_true': model_results['fold_predictions']['y_true'],
                        'y_pred': model_results['fold_predictions']['y_pred'],
                        'y_proba': model_results['fold_predictions']['y_proba']
                    })
                    pred_df.to_csv(pred_file, index=False)
        
        self.logger.info(f"Detailed results saved to {results_file}")
    
    def run_comprehensive_test(self):
        """Run comprehensive testing pipeline"""
        self.logger.info("Starting comprehensive model testing")
        self.logger.info(f"Models to test: {self.config.model_paths}")
        self.logger.info(f"Datasets to test: {self.config.data_paths}")
        
        all_results = {}
        
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
                    
                    # Perform k-fold cross-validation
                    cv_results = self.perform_kfold_cv(model, X_embedded, labels, model_name, dataset_name)
                    dataset_results[model_name] = cv_results
                    
                except Exception as e:
                    self.logger.error(f"Failed to test model {model_path}: {e}")
                    continue
            
            # Statistical comparisons
            statistical_comparisons = self.compare_models_statistically(dataset_results)
            
            # Create visualizations
            self.plot_results(dataset_results, dataset_name)
            
            # Save detailed results
            self.save_detailed_results(dataset_results, dataset_name, statistical_comparisons)
            
            all_results[dataset_name] = {
                'model_results': dataset_results,
                'statistical_comparisons': statistical_comparisons
            }
        
        self.results = all_results
        self.logger.info(f"\nTesting completed. Results saved to: {self.config.output_dir}")
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
        'output_dir': './test_results',
        'cv_folds': 5,
        'cv_strategy': 'stratified',
        'random_state': 42,
        'metrics': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc'],
        'plot_results': True,
        'save_predictions': True,
        'statistical_tests': True
    }
    
    with open('test_config.yaml', 'w') as f:
        yaml.dump(sample_config, f, default_flow_style=False)
    
    print("Sample test configuration created: test_config.yaml")

def main():
    """Main function for the test script"""
    parser = argparse.ArgumentParser(
        description="Comprehensive testing for antibody classification models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Test single model on single dataset
    python test.py --model models/antibody_classifier.pkl --data sample_data.csv
    
    # Test multiple models on multiple datasets
    python test.py --model models/model1.pkl models/model2.pkl --data dataset1.csv dataset2.csv
    
    # Use configuration file
    python test.py --config test_config.yaml
    
    # Create sample configuration
    python test.py --create-config
        """
    )
    
    parser.add_argument('--model', nargs='+', help='Path(s) to trained model pickle files')
    parser.add_argument('--data', nargs='+', help='Path(s) to test dataset CSV files')
    parser.add_argument('--config', help='Path to test configuration YAML file')
    parser.add_argument('--output-dir', default='./test_results', help='Output directory for results')
    parser.add_argument('--cv-folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--random-state', type=int, default=42, help='Random state for reproducibility')
    parser.add_argument('--create-config', action='store_true', help='Create sample configuration file')
    parser.add_argument('--no-plots', action='store_true', help='Disable plotting')
    parser.add_argument('--no-stats', action='store_true', help='Disable statistical tests')
    
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
            output_dir=args.output_dir,
            cv_folds=args.cv_folds,
            random_state=args.random_state,
            plot_results=not args.no_plots,
            statistical_tests=not args.no_stats
        )
    
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
            for model_name, model_results in dataset_results['model_results'].items():
                print(f"Model: {model_name}")
                for metric, values in model_results['cv_scores'].items():
                    print(f"  {metric}: {values['mean']:.4f} (+/- {values['std']*2:.4f})")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
