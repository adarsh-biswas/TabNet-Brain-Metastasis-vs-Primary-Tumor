# TabNet: Cross-Pathway Interaction Attention for Brain Metastasis Prediction

## Overview

This repository contains the implementation of a pathway-aware deep learning framework for predicting brain metastasis using gene expression microarray datasets. The model integrates pathway-based feature learning, attention mechanisms, contrastive learning, and TabNet classification to improve metastasis detection performance.

The framework introduces a Cross-Pathway Interaction Attention (CPIA) mechanism to capture biologically meaningful interactions between molecular pathways.

This repository is provided for research reproducibility purposes.

---

# Methodology

The proposed framework consists of the following stages:

1. GEO dataset loading and preprocessing
2. Probe-to-gene mapping
3. Pathway-aware gene selection
4. Within-pathway attention learning
5. Cross-Pathway Interaction Attention (CPIA)
6. Contrastive representation learning
7. TabNet-based classification

The model combines biological pathway information with deep learning-based feature extraction to improve predictive performance on imbalanced metastasis datasets.

---

# Novel Contribution

The primary contribution of this work is the Cross-Pathway Interaction Attention (CPIA) module.

Given pathway embeddings:

\[
S_{ij} = \frac{e_i \cdot e_j}{||e_i|| ||e_j|| + \epsilon}
\]

\[
G_{ij} = \sigma(MLP([S_{ij}, S_{ij}^2, S_{ij}f_i, S_{ij}f_j]))
\]

\[
A_{ij} = softmax_j \left(\frac{G_{ij} \cdot S_{ij}}{\sqrt{d}}\right)
\]

\[
h_i = \sum_j A_{ij} e_j
\]

The module learns biologically relevant cross-pathway interactions to improve separation between primary tumor and metastatic samples.

---

# Dataset Information

The study uses publicly available datasets from the NCBI Gene Expression Omnibus (GEO).

## GEO Datasets Used

| Dataset | Description | Label |
|----------|-------------|-------|
| GSE50161 | Primary brain tumor samples | 0 |
| GSE108474 | Primary tumor samples | 0 |
| GSE52604 | Brain metastasis samples | 1 |

## Dataset Source

NCBI GEO Database:

https://www.ncbi.nlm.nih.gov/geo/

## Dataset Characteristics

- Data Type: Gene expression microarray data
- Platform: GEO microarray platforms
- Biological Domain: Brain tumor and metastasis analysis
- Classes:
  - Primary tumors
  - Brain metastases

## Data Processing

The preprocessing pipeline includes:

- GEO dataset download using GEOparse
- Probe-to-gene symbol mapping
- Missing value imputation
- Z-score normalization
- Standard scaling
- Pathway-aware gene filtering
- Stratified train-test splitting

---

# Architecture

The framework consists of two major attention stages:

## Stage 1: Within-Pathway Attention

Learns gene-level importance inside each biological pathway.

## Stage 2: Cross-Pathway Interaction Attention (CPIA)

Learns interactions between pathway embeddings using gated cosine attention.

---

# Model Components

## Attention Encoder

- PathwayAttention
- CrossPathwayInteractionAttention (CPIA)

## Classifier

- TabNetClassifier

## Loss Functions

- BCEWithLogitsLoss
- Contrastive interaction loss

---

# Technologies Used

- Python
- PyTorch
- TabNet
- Scikit-learn
- GEOparse
- NumPy
- Pandas

---

# Required Libraries

```txt
torch
numpy
pandas
scikit-learn
geoparse
pytorch-tabnet
gseapy
```

---

# Running the Project

Run the main script:

```bash
python main.py
```

The pipeline automatically:

1. Downloads GEO datasets
2. Performs preprocessing
3. Builds pathway-aware representations
4. Trains the attention encoder
5. Trains the TabNet classifier
6. Evaluates performance
7. Saves output files

---

# Output Files

The following files are generated after execution:

| File | Description |
|------|-------------|
| v4_2_results.csv | Sample-level predictions |
| v4_2_metrics.csv | Performance metrics |
| v4_2_interaction_heatmap.csv | Cross-pathway interaction matrix |

---

# Evaluation Metrics

The framework evaluates:

- ROC-AUC
- PR-AUC
- MCC
- Balanced Accuracy
- Sensitivity
- Specificity
- Precision
- F1-score
- Brier Score
- Log-loss

---

# Reproducibility

To ensure reproducibility:

- Random seed is fixed
- Public GEO datasets are used
- Full preprocessing pipeline is included
- Complete training pipeline is provided

---

# Citation

If you use this work, please cite:

```bibtex
@article{yourcitation,
  title={TabNet: Cross-Pathway Interaction Attention for Brain Metastasis Prediction},
  author={Author Name},
  journal={Journal Name},
  year={2026}
}
```

---

# License

This project is intended for academic and research purposes.

---

# Contact

For questions regarding the implementation or reproducibility, please contact the authors.
