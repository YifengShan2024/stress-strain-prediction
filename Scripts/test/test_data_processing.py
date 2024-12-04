import torch
from torch.utils.data import DataLoader, TensorDataset

def test_data_loading():
    """
    Test loading data and creating a DataLoader.
    """
    # Simulate data
    inputs = torch.rand((100, 128))
    targets = torch.rand((100, 64))
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Check data shapes
    for x, y in dataloader:
        assert x.shape[0] <= 32, "Batch size exceeds limit"
        assert x.shape[1] == 128, "Input feature size mismatch"
        assert y.shape[1] == 64, "Target size mismatch"
    print("Data loading test passed")

def test_data_shapes():
    """
    Test the shape consistency between inputs and targets.
    """
    inputs = torch.rand((50, 128))
    targets = torch.rand((50, 64))
    assert inputs.size(0) == targets.size(0), "Mismatch in data and target lengths"
    print("Data shapes test passed")

if __name__ == "__main__":
    test_data_loading()
    test_data_shapes()
