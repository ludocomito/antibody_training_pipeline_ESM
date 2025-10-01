"""
ESM Model Module
Contains the ESM-1V embedding extractor for protein sequences.
"""

import logging
import torch
import numpy as np
from transformers import AutoModel,AutoModelForMaskedLM, AutoTokenizer
from tqdm import tqdm
from typing import List

logger = logging.getLogger(__name__)

class ESMEmbeddingExtractor:
    
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device
        # Load model with output_hidden_states enabled
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.model.to(device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        logger.info(f"ESM model {model_name} loaded on {device}")
        
    def embed_sequence(self, sequence: str) -> np.ndarray:
        """ Extract ESM-1V embedding for a given protein sequence."""
        try:
            # Validate sequence
            valid_aas = set("ACDEFGHIKLMNPQRSTVWYX")
            sequence = sequence.upper().strip()
            
            if not all(aa in valid_aas for aa in sequence):
                raise ValueError("Invalid amino acid characters in sequence")
            
            if len(sequence) < 1:
                raise ValueError("Sequence too short")
            
            # Tokenize the sequence
            inputs = self.tokenizer(sequence, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                # Check if hidden_states are available
                if outputs.hidden_states is None:
                    raise RuntimeError("Model did not output hidden_states. Please retrain the model with the fixed version.")
                embeddings = outputs.hidden_states[-1]  # (batch, seq_len, hidden_dim)
                
                # Use attention mask to properly exclude padding and special tokens
                attention_mask = inputs['attention_mask'].unsqueeze(-1)  # (batch, seq_len, 1)
                
                # Mask out special tokens (first and last)
                attention_mask[:, 0, :] = 0  # CLS token
                attention_mask[:, -1, :] = 0  # EOS token
                
                # Masked mean pooling
                masked_embeddings = embeddings * attention_mask
                sum_embeddings = masked_embeddings.sum(dim=1)  # Sum over sequence
                sum_mask = attention_mask.sum(dim=1)  # Count valid tokens
                mean_embeddings = sum_embeddings / sum_mask  # Average
                
                return mean_embeddings.squeeze(0).cpu().numpy()
            
        except Exception as e:
            logger.error(f"Error getting embeddings for sequence: {e}")
            raise
        
    def extract_batch_embeddings(self, sequences: List[str]) -> np.ndarray:
        """Extract embeddings for a batch of sequences"""
        embeddings_list = []
        
        logger.info(f"Extracting embeddings for {len(sequences)} sequences...")
        for i, seq in enumerate(tqdm(sequences, desc="Processing sequences")):
            try:
                embedding = self.embed_sequence(seq)
                embeddings_list.append(embedding)
                
                # Clear GPU cache periodically to prevent OOM
                if (i + 1) % 100 == 0:
                    self._clear_gpu_cache()
                    
            except Exception as e:
                logger.warning(f"Failed to process sequence {i}: {e}")
                # Add zero embedding for failed sequences
                embeddings_list.append(np.zeros(1280))  # ESM-1V embedding dimension
        
        # Final cache clear
        self._clear_gpu_cache()
        return np.array(embeddings_list)
    
    def _clear_gpu_cache(self):
        """Clear GPU cache if using CUDA"""
        if str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()

