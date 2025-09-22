# Antibody Classification Pipeline Usage Guide

This guide explains how to use the antibody classification pipeline with ESM-1V embeddings.

## Quick Start

1. **Prepare your data**:
   Create a CSV file with your antibody sequences and binary labels:
   ```csv
   sequence,label
   QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYYMHWVRQAPGQGLEWMGWINPNSGGTNYAQKFQGRVTMTRDTSISTAYMELSRLRSDDTAVYYCARSTYYGGDWYFNVWGQGTLVTVSS,1
   EVQLLESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARQHYDWPWGQGTLVTVSS,0
   ```

2. **Update configuration**:
   Edit `config.yaml` to point to your data file:
   ```yaml
   data:
     train_file: "path/to/your/training_data.csv"
   ```

3. **Run training**:
   ```bash
   python main.py
   ```

## Configuration Options

### Model Configuration
```yaml
model:
  name: "facebook/esm1v_t33_650M_UR90S_1"  # ESM model to use
  device: "cuda"  # or "cpu"
```

### Data Configuration
```yaml
data:
  train_file: "data/train.csv"           # Required: training data
  val_file: null                         # Optional: validation data
  sequence_column: "sequence"            # Column name for sequences
  label_column: "label"                  # Column name for labels
  validation_split: 0.2                  # Train/val split if no val_file
  save_embeddings: true                  # Cache embeddings
  embeddings_cache_dir: "./embeddings_cache"
```

### Classifier Configuration
```yaml
classifier:
  type: "logistic_regression"
  max_iter: 1000                         # Max iterations for LogReg
  random_state: 42                       # Reproducibility seed
  cv_folds: 5                           # Cross-validation folds
  stratify: true                        # Stratified CV
```

### Training Configuration
```yaml
training:
  metrics: ["accuracy", "precision", "recall", "f1", "roc_auc"]
  save_model: true
  model_save_path: "./models/antibody_classifier.pkl"
  log_level: "INFO"
  log_file: "./logs/training.log"
```

## Command Line Usage

### Basic Usage
```bash
# Use default config.yaml
python main.py

# Use custom config file
python main.py my_config.yaml
```

### Additional Options
```bash
# Only validate configuration
python main.py --validate-only

# Setup directories without training
python main.py --setup-dirs

# Get help
python main.py --help
```

## Data Format

Your training data should be a CSV file with at least two columns:

1. **Sequence column**: Contains antibody sequences (amino acid strings)
2. **Label column**: Contains binary labels (0 or 1)

Example:
```csv
sequence,label,description
QVQLVQSGAEVKKPGASVKVSCKASGYTFT...,1,developable
EVQLLESGGGLVQPGGSLRLSCAASGFTFS...,0,non-developable
```

### Data Requirements
- Sequences should contain only valid amino acid characters (A-Z)
- Labels should be binary (0/1)
- No missing values in sequence or label columns
- Sequences will be truncated to 1024 tokens (ESM limit)

## Output Files

After training, the pipeline creates several output files:

### Model Files
- `models/antibody_classifier.pkl`: Trained classifier (includes scaler and model)

### Cache Files
- `embeddings_cache/train_embeddings.pkl`: Cached training embeddings
- `embeddings_cache/val_embeddings.pkl`: Cached validation embeddings

### Log Files
- `logs/training.log`: Detailed training logs

## Using the Trained Model

```python
import pickle
import numpy as np

# Load the trained model
with open('models/antibody_classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)

# Predict on new sequences
sequences = ["QVQLVQSGAEVK...", "EVQLLESGGGLV..."]
embeddings = classifier.embedding_extractor.extract_batch_embeddings(sequences)

# Get predictions
predictions = classifier.predict(embeddings)
probabilities = classifier.predict_proba(embeddings)

print(f"Predictions: {predictions}")
print(f"Probabilities: {probabilities}")
```

## Performance Monitoring

The pipeline provides comprehensive evaluation:

### Metrics Calculated
- **Accuracy**: Overall correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve

### Cross-Validation
- Stratified K-fold cross-validation
- Configurable number of folds
- Mean and standard deviation reported

### Output Example
```
Training Results:
  accuracy: 0.8542
  precision: 0.8421
  recall: 0.8667
  f1: 0.8542
  roc_auc: 0.9123

Cross-validation Results:
  cv_accuracy: 0.8456 (+/- 0.0234)
  cv_f1: 0.8398 (+/- 0.0267)
  cv_roc_auc: 0.9087 (+/- 0.0156)
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**:
   - Reduce batch size in config
   - Use CPU instead of GPU
   - Clear cache more frequently

2. **Invalid sequences**:
   - Check for non-standard amino acids
   - Remove sequences with ambiguous characters
   - Ensure sequences are not empty

3. **Poor performance**:
   - Check data quality and balance
   - Increase max_iter for logistic regression
   - Try different train/validation splits

### Memory Management
- Embeddings are cached to avoid recomputation
- GPU cache is cleared periodically
- Use CPU if GPU memory is limited

### Logging
Set `log_level` to `"DEBUG"` for detailed information:
```yaml
training:
  log_level: "DEBUG"
```

## Advanced Usage

### Custom Metrics
Add custom evaluation metrics by modifying the `evaluate_model` function in `train.py`.

### Different Models
Change the ESM model in configuration:
```yaml
model:
  name: "facebook/esm2_t33_650M_UR50D"  # Different ESM variant
```

### Hyperparameter Tuning
Modify classifier parameters:
```yaml
classifier:
  max_iter: 2000      # More iterations
  cv_folds: 10        # More CV folds
```

## Support

For issues or questions:
1. Check the log files for detailed error messages
2. Validate your configuration with `--validate-only`
3. Ensure your data format matches the requirements
4. Check GPU/CPU memory usage
