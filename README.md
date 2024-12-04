# Stress-Strain Adaptive Predictive Model (SSAPM)

## Overview
This repository implements the **Stress-Strain Adaptive Predictive Model (SSAPM)** for accurate stress-strain prediction in structural mechanics. The model combines mechanistic modeling and hybrid attention modules with data-driven corrections to address complex material behaviors.

## Features
- **Hybrid Attention Module**: Integrates self-attention and channel attention mechanisms.
- **Custom Loss Function**: Combines physical constraints and data-driven loss for enhanced performance.
- **Data Augmentation**: Includes noise addition, random scaling, and flipping.
- **Flexible Data Pipeline**: Supports CSV-based datasets with train/validation/test splits.
- **Visualization Tools**: Generate comparison plots for true and predicted values.

---

## File Structure
```plaintext
.
├── README.md                      # Project documentation
├── requirements.txt               # Dependencies
├── main.py                        # Entry point for training, validation, and testing
├── config.yaml                    # Configuration file for model and data
├── src/
│   ├── models/                    # Model-related code
│   │   ├── ssapm.py               # Stress-Strain Adaptive Predictive Model
│   │   ├── attention.py           # Hybrid attention module
│   │   ├── loss_functions.py      # Custom loss functions
│   │   └── optimization.py        # Optimizer configurations
│   ├── utils/                     # Utility functions
│   │   ├── visualization.py       # Visualization tools
│   │   └── logging.py             # Logging utilities
├── data/
│   ├── raw/                       # Raw datasets
│   └── processed/                 # Preprocessed datasets
├── tests/                         # Unit tests
│   ├── test_models.py             # Tests for model initialization and forward pass
│   ├── test_data_processing.py    # Tests for data loading and augmentation
│   └── test_training.py           # Tests for training pipeline
