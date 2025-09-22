import logging
import pickle
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

def preprocess_raw_data(
    X: List[str],
    y: List[Any],
    embedding_extractor,
    scaler: Optional[StandardScaler] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a dataset of input sequences (X) and labels (y), embed the sequences and return
    the embedded dataset and labels.

    Args:
        X: List of input protein sequences (strings)
        y: List or array of labels
        embedding_extractor: An instance of ESMEmbeddingExtractor or similar with an 'embed_sequence' or 'extract_batch_embeddings' method
        scaler: Optional StandardScaler to fit/transform the embeddings

    Returns:
        X_embedded: np.ndarray of embedded input sequences
        y: np.ndarray of labels (unchanged)
    """
    logger.info(f"Embedding {len(X)} sequences...")
    # Try to use batch embedding if available
    if hasattr(embedding_extractor, "extract_batch_embeddings"):
        X_embedded = embedding_extractor.extract_batch_embeddings(X)
    else:
        X_embedded = np.array([embedding_extractor.embed_sequence(seq) for seq in X])

    if scaler is not None:
        logger.info("Scaling embeddings...")
        X_embedded = scaler.fit_transform(X_embedded)

    return X_embedded, np.array(y)

def store_preprocessed_data(
    X: List[str] = None,
    y: List[Any] = None,
    X_embedded: Optional[np.ndarray] = None,
    filename: str = None
):
    """
    Store the preprocessed data to a pickle file.
    You can provide either raw sequences (X) and labels (y), or embedded data (X_embedded) and labels (y).
    If X_embedded is provided, it will be stored as the embedded data.
    """
    data = {}
    if X_embedded is not None:
        data['X_embedded'] = X_embedded
    if X is not None:
        data['X'] = X
    if y is not None:
        data['y'] = y
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_preprocessed_data(filename: str) -> Dict:
    """
    Load the preprocessed data from a pickle file.
    Returns a dictionary with keys: 'X', 'y', and/or 'X_embedded'
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)