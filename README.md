# Stress-Strain Adaptive Predictive Model (SSAPM)

## Overview
This repository implements the **Stress-Strain Adaptive Predictive Model (SSAPM)** for accurate stress-strain prediction in structural mechanics. The model combines mechanistic modeling and hybrid attention modules with data-driven corrections to address complex material behaviors.

## Features
- **Hybrid Attention Module**: Integrates self-attention and channel attention mechanisms.
- **Custom Loss Function**: Combines physical constraints and data-driven loss for enhanced performance.
- **Data Augmentation**: Includes noise addition, random scaling, and flipping.
- **Flexible Data Pipeline**: Supports CSV-based datasets with train/validation/test splits.
- **Visualization Tools**: Generate comparison plots for true and predicted values.


## File Structure

### Root Directory
- **README.md**: Project documentation.
- **requirements.txt**: Dependencies.
- **main.py**: Entry point for training, validation, and testing.
- **config.yaml**: Configuration file for model and data.

### `src/` Directory
#### Models
- **ssapm.py**: Stress-Strain Adaptive Predictive Model.
- **attention.py**: Hybrid attention module.
- **loss_functions.py**: Custom loss functions.
- **optimization.py**: Optimizer configurations.

#### Utilities
- **visualization.py**: Visualization tools.
- **logging.py**: Logging utilities.

### `data/` Directory
- **raw/**: Raw datasets.
- **processed/**: Preprocessed datasets.

### `tests/` Directory
- **test_models.py**: Tests for model initialization and forward pass.
- **test_data_processing.py**: Tests for data loading and augmentation.
- **test_training.py**: Tests for training pipeline.


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


## Contributions
This work introduces several key contributions to structural mechanics and computational modeling:

Hybrid Attention Design: Integrates domain-specific feature attention with deep learning, improving interpretability and performance for stress-strain predictions.
Custom Loss Framework: Combines mechanistic principles with data-driven losses to enforce physical realism in predictions.
Efficient Data Processing: Implements augmentation techniques and a flexible pipeline for multi-sensor data integration.
Modular Framework: Provides a reusable structure for extensions in other applications, such as material failure prediction or real-time simulations.

## Future Work
This project opens up several avenues for future research and development:

Real-Time Adaptability: Incorporate dynamic optimization strategies to adapt the model to changing material properties in real-time.
Extended Sensor Integration: Explore additional data sources, such as acoustic and thermal sensors, for multi-modal fusion.
Scalability: Improve the computational efficiency for large-scale datasets and 3D modeling scenarios.
Explainability: Develop tools to interpret attention mechanisms, making the model outputs more accessible to domain experts.
Transfer Learning: Adapt the model for applications in related domains, such as biomechanics or aerospace engineering.

##  License
This project is licensed under the MIT License.

