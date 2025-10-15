# Antibody Classification Pipeline Usage Guide

This guide explains how to use the antibody classification pipeline with ESM-1V embeddings.

## Quick Start

# Configure Dataset source
## Option 1: local dataset
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
3. **update source**
  Edit load_data in `train.py` 
  ```python
  load_data(config, source='local')
  ```

## Option 2: HuggingFace dataset

1. **Update configuration**:
  Edit `config.yaml` with dataset name:
  ```yaml
  data:
    dataset_name: "VH_dataset"
  ```

2. **update source**
  Edit load_data in `train.py` 
  ```python
  load_data(config, source='hf')
  ```

# Run training pipeline
   ```bash
   python main.py
   ```

## Configuration Options

### Model Configuration
```yaml
model:
  name: "facebook/esm1v_t33_650M_UR90S_1"  # ESM model to use
  device: "cuda"  # or "cpu"/"mps"
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
```

### Training Configuration
```yaml
training:
  metrics: ["accuracy", "precision", "recall", "f1", "roc_auc"]
  save_model: true
  model_name: "antibody_classifier"  # Name for the saved model (without extension)
  model_save_dir: "./models"          # Directory to save models
  log_level: "INFO"
  log_file: "./logs/training.log"
```

**Note**: The trained model will be saved as `{model_save_dir}/{model_name}.pkl`. Change `model_name` to give your model a custom name.

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

## Testing the Trained Model

After training, you can test your model using the comprehensive testing framework. The testing pipeline performs k-fold cross-validation and generates detailed performance metrics and visualizations.

### Test Configuration

Create or edit `test_config.yaml` to specify testing parameters:

```yaml
# Test Configuration for Antibody Classification Pipeline

# Model configuration
model_paths:
  - ./models/antibody_classifier.pkl  # Path(s) to trained model checkpoint(s)

# Data configuration
data_paths:
  - ./sample_data.csv  # Path(s) to test dataset(s)

# Column names in the dataset
sequence_column: "sequence"  # Column name containing protein sequences
label_column: "label"        # Column name containing binary labels

# Cross-validation configuration
cv_folds: 5
cv_strategy: stratified  # "stratified" or "normal"
random_state: 42

# Metrics to evaluate
metrics:
  - accuracy
  - precision
  - recall
  - f1
  - roc_auc
  - pr_auc

# Output configuration
output_dir: ./test_results
plot_results: true
save_predictions: true
statistical_tests: true
```

### Running Tests

```bash
# Test with configuration file
python test.py --config test_config.yaml

# Or test directly with command line arguments
python test.py --model models/antibody_classifier.pkl --data test_data.csv

# Test multiple models on multiple datasets
python test.py --model models/model1.pkl models/model2.pkl --data data1.csv data2.csv

# Create a sample test configuration
python test.py --create-config
```

### Test Output

The testing pipeline generates:

1. **Detailed results**: YAML file with all metrics and cross-validation scores
2. **Predictions**: CSV files with true labels, predictions, and probabilities
3. **Visualizations**: PNG files with:
   - ROC curves
   - Precision-Recall curves
   - Cross-validation score distributions (boxplots)
   - Confusion matrices
4. **Statistical comparisons**: T-tests and Wilcoxon tests between models (if testing multiple models)
5. **Log file**: Detailed testing logs with timestamps

### Test Results Example

```
Testing on dataset: test_data
Model: antibody_classifier
Cross-validation results:
  accuracy: 0.8456 (+/- 0.0234)
  precision: 0.8398 (+/- 0.0267)
  recall: 0.8523 (+/- 0.0189)
  f1: 0.8459 (+/- 0.0245)
  roc_auc: 0.9087 (+/- 0.0156)
  pr_auc: 0.8923 (+/- 0.0178)

Plots saved to: ./test_results/results_test_data.png
```

### Custom Column Names

If your test dataset uses different column names, specify them in `test_config.yaml`:

```yaml
sequence_column: "vh_sequence"
label_column: "developability"
```

## Using the Trained Model Programmatically

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

### Training Output Example
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

Saving model to ./models/antibody_classifier.pkl
Model saved successfully
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

