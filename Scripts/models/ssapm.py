import torch
import torch.nn as nn
from models.attention import HybridAttention


class StressStrainModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, attention_dim):
        """
        Stress-Strain Adaptive Predictive Model (SSAPM)
        Args:
            input_size (int): Input feature size.
            hidden_size (int): Size of the hidden layers.
            output_size (int): Output size (e.g., stress-strain predictions).
            attention_dim (int): Dimension of the attention mechanisms.
        """
        super(StressStrainModel, self).__init__()

        # Linear transformations
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer1 = nn.Linear(hidden_size, hidden_size)
        self.hidden_layer2 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

        # Attention mechanisms
        self.attention = HybridAttention(hidden_size, attention_dim)

        # Activation function
        self.activation = nn.ReLU()

        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        """
        Forward pass for the SSAPM.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            torch.Tensor: Predicted output tensor.
        """
        # Input transformation
        x = self.activation(self.input_layer(x))

        # First hidden layer
        x = self.activation(self.hidden_layer1(x))

        # Apply attention mechanism
        attention_output, attention_weights = self.attention(x)

        # Second hidden layer
        x = self.activation(self.hidden_layer2(attention_output))

        # Dropout regularization
        x = self.dropout(x)

        # Output layer
        return self.output_layer(x), attention_weights


# Helper function for model initialization
def initialize_ssapm(config):
    """
    Initializes the Stress-Strain Model with given configuration.
    Args:
        config (dict): Configuration dictionary containing model parameters.
    Returns:
        nn.Module: Initialized model.
    """
    model = StressStrainModel(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        output_size=config["output_size"],
        attention_dim=config["attention_dim"]
    )
    return model


if __name__ == "__main__":
    # Example configuration
    config = {
        "input_size": 128,
        "hidden_size": 256,
        "output_size": 64,
        "attention_dim": 128
    }

    # Create a model
    model = initialize_ssapm(config)

    # Example input
    example_input
