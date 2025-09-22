import logging
import numpy as np
from typing import Dict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from model import ESMEmbeddingExtractor

logger = logging.getLogger(__name__)

class BinaryClassifier:
    """Binary classifier for protein sequences using ESM-1V embeddings"""
    
    def __init__(self, params: Dict):
        """
        Initialize the binary classifier
        
        Args:
            params: Dictionary containing the parameters for the classifier
        """
        random_state = params['random_state']
        
        self.embedding_extractor = ESMEmbeddingExtractor(params['model_name'], params['device'])
        self.scaler = StandardScaler()
        
        self.classifier = LogisticRegression(
            random_state=params['random_state'], 
            max_iter=params['max_iter']
        )
        self.random_state = random_state
        self.is_fitted = False
        self.device = self.embedding_extractor.device
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the classifier to the data
        
        Args:
            X: Array of ESM-1V embeddings
            y: Array of labels
        """
        # Scale the embeddings
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit the classifier
        self.classifier.fit(X_scaled, y)
        self.is_fitted = True
        logger.info(f"Classifier fitted on {len(X)} samples")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the labels for the data
        
        Args:
            X: Array of ESM-1V embeddings
            
        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before making predictions")
            
        X_scaled = self.scaler.transform(X)
        return self.classifier.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for the data
        
        Args:
            X: Array of ESM-1V embeddings
            
        Returns:
            Predicted probabilities
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before making predictions")
            
        X_scaled = self.scaler.transform(X)
        return self.classifier.predict_proba(X_scaled)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the mean accuracy on the given test data and labels
        
        Args:
            X: Array of ESM-1V embeddings
            y: Array of true labels
            
        Returns:
            Mean accuracy
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before scoring")
            
        X_scaled = self.scaler.transform(X)
        return self.classifier.score(X_scaled, y)