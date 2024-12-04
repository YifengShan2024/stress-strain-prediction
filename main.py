import os
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset
from models.ssapm import initialize_ssapm
from models.loss_functions import hybrid_loss
from models.optimization import get_optimizer
from utils.logging import setup_logger
from utils.visualization import plot_predictions


# Load configuration
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


# Load data
def load_data(config):
    train_data = torch.load(config["train_path"])
    val_data = torch.load(config["val_path"])
    test_data = torch.load(config["test_path"])

    train_loader = DataLoader(TensorDataset(*train_data), batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(TensorDataset(*val_data), batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(TensorDataset(*test_data), batch_size=config["batch_size"], shuffle=False)

    return train_loader, val_loader, test_loader


# Train the model
def train_model(model, train_loader, val_loader, config, logger):
    optimizer = get_optimizer(model, config["optimizer"])
    criterion = hybrid_loss
    best_val_loss = float("inf")
    save_path = config["save_path"]

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            predictions, attention_weights = model(x)
            loss = criterion(predictions, y, torch.zeros_like(predictions))  # Example physical residual
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1}/{config['epochs']} - Training Loss: {avg_loss:.4f}")

        # Validate the model
        val_loss = validate_model(model, val_loader, criterion)
        logger.info(f"Epoch {epoch + 1}/{config['epochs']} - Validation Loss: {val_loss:.4f}")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            logger.info(f"Best model saved at epoch {epoch + 1}")


# Validate the model
def validate_model(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            predictions, _ = model(x)
            loss = criterion(predictions, y, torch.zeros_like(predictions))
            total_loss += loss.item()

    return total_loss / len(val_loader)


# Test the model
def test_model(model, test_loader, config, logger):
    model.load_state_dict(torch.load(config["save_path"]))
    model.eval()

    predictions = []
    true_values = []
    with torch.no_grad():
        for x, y in test_loader:
            preds, _ = model(x)
            predictions.append(preds)
            true_values.append(y)

    predictions = torch.cat(predictions, dim=0)
    true_values = torch.cat(true_values, dim=0)

    # Visualize results
    plot_predictions(true_values.numpy(), predictions.numpy())

    # Log final metrics
    test_loss = hybrid_loss(predictions, true_values, torch.zeros_like(predictions)).item()
    logger.info(f"Test Loss: {test_loss:.4f}")


# Main entry point
if __name__ == "__main__":
    # Load configuration
    config = load_config("config.yaml")

    # Set up logging
    logger = setup_logger(config["log_path"])
    logger.info("Starting Stress-Strain Prediction Model...")

    # Load data
    logger.info("Loading data...")
    train_loader, val_loader, test_loader = load_data(config["data"])

    # Initialize model
    logger.info("Initializing model...")
    model = initialize_ssapm(config["model"])
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters.")

    # Train the model
    logger.info("Starting training...")
    train_model(model, train_loader, val_loader, config["train"], logger)

    # Test the model
    logger.info("Testing the model...")
    test_model(model, test_loader, config["test"], logger)

    logger.info("Model training and testing complete.")
