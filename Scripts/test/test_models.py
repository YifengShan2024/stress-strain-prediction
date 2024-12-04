import torch
from models.ssapm import initialize_ssapm

def test_model_initialization():
    """
    Test the initialization of the Stress-Strain Adaptive Predictive Model.
    """
    config = {
        "input_size": 128,
        "hidden_size": 256,
        "output_size": 64,
        "attention_dim": 128
    }
    model = initialize_ssapm(config)
    assert model is not None, "Model initialization failed"
    print("Model initialized successfully")

def test_model_forward_pass():
    """
    Test the forward pass of the model with random inputs.
    """
    config = {
        "input_size": 128,
        "hidden_size": 256,
        "output_size": 64,
        "attention_dim": 128
    }
    model = initialize_ssapm(config)
    example_input = torch.rand((32, config["input_size"]))
    output, attention_weights = model(example_input)
    assert output.shape == (32, config["output_size"]), "Output shape mismatch"
    assert attention_weights is not None, "Attention weights missing"
    print("Model forward pass test passed")

if __name__ == "__main__":
    test_model_initialization()
    test_model_forward_pass()
