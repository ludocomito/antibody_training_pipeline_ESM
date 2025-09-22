"""
Training Script for Antibody Classification Pipeline
Trains a binary classifier on ESM-1V embeddings of antibody sequences.
"""

import os
import logging
import pickle
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix
)

from model import ESMEmbeddingExtractor
from classifier import BinaryClassifier
from data import preprocess_raw_data, store_preprocessed_data, load_preprocessed_data

def setup_logging(config: Dict) -> logging.Logger:
    """Setup logging configuration"""
    log_level = getattr(logging, config['training']['log_level'].upper())
    log_file = config['training']['log_file']
    
    # Create log directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_data(config: Dict, logger: logging.Logger) -> Tuple[List[str], List[int], Optional[List[str]], Optional[List[int]]]:
    """
    Load training and validation data from files
    
    Returns:
        X_train, y_train, X_val, y_val (validation data may be None)
    """
    data_config = config['data']
    
    if data_config['train_file'] is None:
        raise ValueError("train_file must be specified in config")
    
    # Load training data
    logger.info(f"Loading training data from {data_config['train_file']}")
    train_df = pd.read_csv(data_config['train_file'])
    
    X_train = train_df[data_config['sequence_column']].tolist()
    y_train = train_df[data_config['label_column']].tolist()
    
    logger.info(f"Loaded {len(X_train)} training samples")
    
    # Load validation data if provided
    X_val, y_val = None, None
    if data_config['val_file'] is not None:
        logger.info(f"Loading validation data from {data_config['val_file']}")
        val_df = pd.read_csv(data_config['val_file'])
        X_val = val_df[data_config['sequence_column']].tolist()
        y_val = val_df[data_config['label_column']].tolist()
        logger.info(f"Loaded {len(X_val)} validation samples")
    
    return X_train, y_train, X_val, y_val

def create_train_val_split(
    X: List[str], 
    y: List[int], 
    config: Dict, 
    logger: logging.Logger
) -> Tuple[List[str], List[str], List[int], List[int]]:
    """Create train/validation split if no validation file provided"""
    val_split = config['data']['validation_split']
    random_state = config['classifier']['random_state']
    
    logger.info(f"Creating train/validation split with {val_split:.1%} validation data")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=val_split, 
        random_state=random_state, 
        stratify=y
    )
    
    logger.info(f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}")
    return X_train, X_val, y_train, y_val

def get_or_create_embeddings(
    sequences: List[str], 
    embedding_extractor: ESMEmbeddingExtractor, 
    cache_path: str, 
    dataset_name: str,
    logger: logging.Logger
) -> np.ndarray:
    """Get embeddings from cache or create them"""
    cache_file = os.path.join(cache_path, f"{dataset_name}_embeddings.pkl")
    
    if os.path.exists(cache_file):
        logger.info(f"Loading cached embeddings from {cache_file}")
        with open(cache_file, 'rb') as f:
            embeddings = pickle.load(f)
        
        if len(embeddings) == len(sequences):
            logger.info(f"Using cached embeddings for {len(sequences)} sequences")
            return embeddings
        else:
            logger.warning("Cached embeddings size mismatch, recomputing...")
    
    logger.info(f"Computing embeddings for {len(sequences)} sequences...")
    embeddings = embedding_extractor.extract_batch_embeddings(sequences)
    
    # Cache the embeddings
    os.makedirs(cache_path, exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(embeddings, f)
    logger.info(f"Cached embeddings to {cache_file}")
    
    return embeddings

def evaluate_model(
    classifier: BinaryClassifier, 
    X: np.ndarray, 
    y: np.ndarray, 
    dataset_name: str,
    metrics: List[str],
    logger: logging.Logger
) -> Dict[str, float]:
    """Evaluate model performance"""
    logger.info(f"Evaluating model on {dataset_name} set")
    
    # Get predictions
    y_pred = classifier.predict(X)
    y_pred_proba = classifier.predict_proba(X)[:, 1]  # Probability of positive class
    
    # Calculate metrics
    results = {}
    
    if 'accuracy' in metrics:
        results['accuracy'] = accuracy_score(y, y_pred)
    
    if 'precision' in metrics:
        results['precision'] = precision_score(y, y_pred, average='binary')
    
    if 'recall' in metrics:
        results['recall'] = recall_score(y, y_pred, average='binary')
    
    if 'f1' in metrics:
        results['f1'] = f1_score(y, y_pred, average='binary')
    
    if 'roc_auc' in metrics:
        results['roc_auc'] = roc_auc_score(y, y_pred_proba)
    
    # Log results
    logger.info(f"{dataset_name} Results:")
    for metric, value in results.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Log classification report
    logger.info(f"\n{dataset_name} Classification Report:")
    logger.info(f"\n{classification_report(y, y_pred)}")
    
    return results

def perform_cross_validation(
    classifier: BinaryClassifier,
    X: np.ndarray,
    y: np.ndarray,
    config: Dict,
    logger: logging.Logger
) -> Dict[str, float]:
    """Perform cross-validation"""
    cv_config = config['classifier']
    cv_folds = cv_config['cv_folds']
    random_state = cv_config['random_state']
    stratify = cv_config['stratify']
    
    logger.info(f"Performing {cv_folds}-fold cross-validation")
    
    # Setup cross-validation
    if stratify:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    else:
        from sklearn.model_selection import KFold
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    # Perform cross-validation for different metrics
    cv_results = {}
    
    # Create a new classifier instance for CV (to avoid fitting on full data)
    cv_params = config['classifier'].copy()
    cv_params['model_name'] = config['model']['name']
    cv_params['device'] = config['model']['device']
    cv_classifier = BinaryClassifier(cv_params)
    
    # Accuracy
    scores = cross_val_score(cv_classifier.classifier, X, y, cv=cv, scoring='accuracy')
    cv_results['cv_accuracy'] = {'mean': scores.mean(), 'std': scores.std()}
    
    # F1 score
    scores = cross_val_score(cv_classifier.classifier, X, y, cv=cv, scoring='f1')
    cv_results['cv_f1'] = {'mean': scores.mean(), 'std': scores.std()}
    
    # ROC AUC
    scores = cross_val_score(cv_classifier.classifier, X, y, cv=cv, scoring='roc_auc')
    cv_results['cv_roc_auc'] = {'mean': scores.mean(), 'std': scores.std()}
    
    # Log results
    logger.info("Cross-validation Results:")
    for metric, values in cv_results.items():
        logger.info(f"  {metric}: {values['mean']:.4f} (+/- {values['std']*2:.4f})")
    
    return cv_results

def save_model(classifier: BinaryClassifier, config: Dict, logger: logging.Logger):
    """Save trained model"""
    if not config['training']['save_model']:
        return
    
    model_path = config['training']['model_save_path']
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    logger.info(f"Saving model to {model_path}")
    
    # Save the entire classifier (including scaler and fitted model)
    with open(model_path, 'wb') as f:
        pickle.dump(classifier, f)
    
    logger.info("Model saved successfully")

def train_model(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Main training function
    
    Args:
        config_path: Path to configuration YAML file
        
    Returns:
        Dictionary containing training results and metrics
    """
    # Load configuration
    config = load_config(config_path)
    
    # Setup logging
    logger = setup_logging(config)
    logger.info("Starting antibody classification training")
    logger.info(f"Configuration loaded from {config_path}")
    
    try:
        # Load data
        X_train, y_train, X_val, y_val = load_data(config, logger)
        
        # Create train/val split if no validation data provided
        if X_val is None:
            X_train, X_val, y_train, y_val = create_train_val_split(X_train, y_train, config, logger)
        
        # Initialize embedding extractor and classifier
        logger.info("Initializing ESM embedding extractor and classifier")
        classifier_params = config['classifier'].copy()
        classifier_params['model_name'] = config['model']['name']  # Map 'name' to 'model_name'
        classifier_params['device'] = config['model']['device']
        classifier = BinaryClassifier(classifier_params)
        
        # Get or create embeddings
        cache_dir = config['data']['embeddings_cache_dir']
        
        X_train_embedded = get_or_create_embeddings(
            X_train, classifier.embedding_extractor, cache_dir, "train", logger
        )
        
        X_val_embedded = get_or_create_embeddings(
            X_val, classifier.embedding_extractor, cache_dir, "val", logger
        )
        
        # Convert labels to numpy arrays
        y_train = np.array(y_train)
        y_val = np.array(y_val)
        
        # Train the classifier
        logger.info("Training classifier...")
        classifier.fit(X_train_embedded, y_train)
        logger.info("Training completed")
        
        # Evaluate on training and validation sets
        metrics = config['training']['metrics']
        train_results = evaluate_model(classifier, X_train_embedded, y_train, "Training", metrics, logger)
        val_results = evaluate_model(classifier, X_val_embedded, y_val, "Validation", metrics, logger)
        
        # Perform cross-validation
        cv_results = perform_cross_validation(classifier, X_train_embedded, y_train, config, logger)
        
        # Save model
        save_model(classifier, config, logger)
        
        # Compile results
        results = {
            'train_metrics': train_results,
            'val_metrics': val_results,
            'cv_metrics': cv_results,
            'config': config,
            'model_path': config['training']['model_save_path'] if config['training']['save_model'] else None
        }
        
        logger.info("Training pipeline completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    import sys
    
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    results = train_model(config_path)
    print("Training completed successfully!")
