import torch
from models.ssapm import initialize_ssapm
from utils.logging import setup_logger
from main import train_model

def test_training():
    """
    Test the training process on dummy data.
    """
    # Configuration
    config = {
        "input_size": 128,
        "hidden_size": 256,
        "output_size": 64,
        "attention_dim": 128,
        "batch_size": 16,
        "epochs": 2,
        "optimizer": {"type": "adam", "lr": 0.001},
        "save_path": "test_model.pth"
    }

    # Dummy data
    inputs = torch.rand((100, 128))
    targets = torch.rand((100, 64))
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(inputs, targets),
        batch_size=config["batch_size"]
    )
    val_loader = train_loader

    # Model and logger
    model = initialize_ssapm(config)
    logger = setup_logger("test_training.log")

    # Training
    train_model(model, train_loader, val_loader, config, logger)
    print("Training test passed")

if __name__ == "__main__":
    test_training()
