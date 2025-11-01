# Antibody Non-Specificity Prediction Pipeline using ESM
This repository provides a machine learning pipeline to predict the non-specificity of antibodies using embeddings from the ESM-1v Protein Language Model(PLM). The project is an implementation of the methods described in the paper *"Prediction of Antibody Non-Specificity using Protein Language Models and Biophysical Parameters"* by Sakhnini et al.

---

# Project Description
Non-specific binding of therapeutic antibodies can lead to faster clearance from the body and other unwanted side effects, compromising their effectiveness and safety. Predicting this property, also known as polyreactivity, from an antibody's amino acid sequence is a critical step in drug development.

This project offers a computational pipeline to tackle this challenge. It leverages the power of the ESM-1v, a state-of-the-art PLM, to convert antibody's amino acid sequences meaningful numerical representations (embeddings). These embeddings capture complex biophysical and evolutionary information, which is then used to train a machine learning classifier to predict non-specificity. The pipeline is designed to be modular, allowing for easy adaptation to different datasets and models. 

---

# Model Architecture
The model's architecture is a two-stage process designed for both power and interpretability:
1. **Sequence Embedding with ESM-1v**: The amino acid sequence of an antibody's Variable Heavy(VH) domain is fed into the pre-trained ESM-1v model. ESM-1v, trained on millions of diverse protein sequences, generates a high-dimensional vector(embedding) for the antibody. This vector represents the learned structural and functional properties of the sequence.
2. **Classification**: The generated embedding vector is then used as input for a simpler, classical machine learning model. The original paper found that a **Logistic Regression** classifier performed best, achieving up to 71% accuracy in 10-fold cross-validation. This second two-stage learns to map the sequence features captured by ESM-1v to a binary outcome: **specific** or **non-specific**

This hybrid approach combines the deep contextual understanding of a PLM with the efficiency and interpretability of a linear classifier.

---

# Features
### Implemented
- **Data Processing**: Scripts to load, clean, and process antibody datasets, including the Boughter et al. (2020) dataset used for training.

- **Sequence Annotation**: Annotation of Complementarity-Determining Regions (CDRs) and extraction of the VH domain from full antibody sequences.

- **ESM-1v Embedding**: A module to generate embeddings for antibody sequences using the ESM-1v model.

- **Model Training**: A complete training pipeline for a Logistic Regression classifier on the generated embeddings.

- **Model Evaluation**: Standard evaluation metrics, including k-fold cross-validation, accuracy, sensitivity, and specificity, are implemented to assess model performance.

### To-Be Implemented
- **Prediction Script**: A user-friendly script to quickly get non-specificity predictions for new antibody sequences.

- **Biophysical Descriptor Module**: A feature to calculate and incorporate key biophysical parameters, such as the isoelectric point (pI), which was identified as a major driver of non-specificity.

- **Support for Other PLMs**: Integration of other antibody-specific language models like AbLang or AntiBERTy for performance comparison.

- **Web Application Interface**: A simple frontend application to make the prediction tool accessible to users without a programming background.

---
# Installation & Setup
To get started, clone the repository and set up the Python environment.
1. Clone the Repository
```bash
git clone https://github.com/ludocomito/antibody_training_pipeline_ESM.git
cd antibody_training_pipeline_ESM
```
2. Create the Environment
This project uses [uv](https://github.com/astral-sh/uv) for fast Python package management with virtual environments.

Install `uv` if you don't have it:

 - *For Linux/macOS*

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
 - *For Windows(use pip)*

```bash
pip install uv
```
3. Set up the project

- *On Linux/macOS*
```bash
uv venv 
source .venv/bin/activate 

uv sync
```
- *On Windows*
```bash
uv venv 
venv\Scripts\activate

uv sync
```

---

# Datasets

## Shehata Dataset

The Shehata dataset (Shehata et al. 2019) provides a critical test set of 398 human antibodies with polyspecific reagent (PSR) assay measurements. This dataset is used to evaluate model performance on fragment-level predictions.

**Quick Start:**

1. Download the raw data from the [Sakhnini et al. 2025 paper supplementary materials](https://doi.org/10.1016/j.cell.2024.12.025) (files: `mmc2.xlsx`)
2. Place the Excel file in `test_datasets/` directory
3. Run Phase 1 preprocessing (Excel → CSV conversion):

   ```bash
   python3 scripts/convert_shehata_excel_to_csv.py
   ```

4. Run Phase 2 preprocessing (fragment extraction):

   ```bash
   python3 preprocessing/process_shehata.py
   ```

**Output Files:**

- `test_datasets/shehata.csv` - Full paired VH+VL sequences (398 antibodies)
- `test_datasets/shehata/*.csv` - 16 fragment-specific files (VH, VL, H-CDR1, H-CDR2, H-CDR3, L-CDR1, L-CDR2, L-CDR3, H-CDRs, L-CDRs, H-FWRs, L-FWRs, VH+VL, All-CDRs, All-FWRs, Full)

**Methodology:** All sequences are annotated using ANARCI with IMGT numbering scheme, following the exact procedure described in Sakhnini et al. 2025 (Section 4.3). For detailed information about data sources and preprocessing steps, see [`docs/shehata_data_sources.md`](docs/shehata_data_sources.md).

---

# Citation
This work is an implementation of the research conducted at Novo Nordisk and the University of Cambridge. If you use this code or its methodology in your research, please cite the original paper:
```bibtex
@unpublished{Sakhnini2025preprint,
  author  = {Sakhnini, Laila I. and Beltrame, Ludovica and Fulle, Simone and Sormanni, Pietro and Henriksen, Anette and Lorenzen, Nikolai and Vendruscolo, Michele and Granata, Daniele},
  title   = {Prediction of Antibody Non-Specificity using Protein Language Models and Biophysical Parameters},
  note    = {Preprint posted on bioRxiv},
  year    = {2025},
  month   = {May},
  doi     = {10.1101/2025.04.28.650927},
  url     = {https://www.biorxiv.org/content/10.1101/2025.04.28.650927v1}
}
```
