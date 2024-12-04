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

---

## Usage

Step 1: Prepare Data
Place raw datasets in the data/raw/ directory.
Use dataset_loader.py to preprocess the data into PyTorch tensors.
Save processed datasets in the data/processed/ directory.
Step 2: Configure the Model
Modify the config.yaml file to adjust parameters such as input size, hidden layers, batch size, and learning rate.

Step 3: Train the Model
Run the main.py script to train the model:  python main.py

Step 4: Test the Model
After training, the best model is saved in the specified directory (config["train"]["save_path"]). Testing will automatically run at the end of the script.

##  Configuration
config.yaml

## Testing
Run unit tests to ensure the code works as expected:

python -m unittest discover -s tests

##  Features in Detail

Hybrid Attention
The attention module combines:

Self-Attention: Captures dependencies across input sequences.
Channel Attention: Focuses on feature importance.
Custom Loss
The hybrid loss function balances:

Mean Squared Error (MSE): For data-driven accuracy.
Physical Residuals: To enforce physical consistency.
Data Augmentation
Enhancements include:

Adding Gaussian noise.
Random scaling and flipping.
Visualization
Use plot_predictions() from utils/visualization.py to visualize true vs. predicted values.

##  Example Output

Trained Model: Saved in outputs/models/.
Logs: Training and test logs in outputs/logs/.
Visualizations: True vs. predicted plots in outputs/results/.

##  License
This project is licensed under the MIT License.

