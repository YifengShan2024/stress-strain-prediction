import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

def load_csv_data(file_path, target_column):
    """
    Load dataset from a CSV file and split into inputs and targets.
    Args:
        file_path (str): Path to the CSV file.
        target_column (str): Name of the target column.
    Returns:
        inputs (torch.Tensor): Input features as a tensor.
        targets (torch.Tensor): Target values as a tensor.
    """
    df = pd.read_csv(file_path)
    targets = torch.tensor(df[target_column].values, dtype=torch.float32)
    inputs = torch.tensor(df.drop(columns=[target_column]).values, dtype=torch.float32)
    return inputs, targets


def split_data(inputs, targets, test_size=0.2, val_size=0.1):
    """
    Split data into training, validation, and test sets.
    Args:
        inputs (torch.Tensor): Input features.
        targets (torch.Tensor): Target values.
        test_size (float): Proportion of the dataset to include in the test split.
        val_size (float): Proportion of the training dataset to include in the validation split.
    Returns:
        train_set, val_set, test_set: Tuple of TensorDatasets.
    """
    x_train, x_test, y_train, y_test = train_test_split(inputs, targets, test_size=test_size, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_size, random_state=42)

    train_set = TensorDataset(x_train, y_train)
    val_set = TensorDataset(x_val, y_val)
    test_set = TensorDataset(x_test, y_test)
    return train_set, val_set, test_set


def get_data_loaders(train_set, val_set, test_set, batch_size):
    """
    Create DataLoaders for training, validation, and testing.
    Args:
        train_set (TensorDataset): Training dataset.
        val_set (TensorDataset): Validation dataset.
        test_set (TensorDataset): Testing dataset.
        batch_size (int): Batch size for DataLoaders.
    Returns:
        Tuple of DataLoaders: train_loader, val_loader, test_loader.
    """
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Example usage
    inputs, targets = load_csv_data("data/raw/sample.csv", target_column="target")
    train_set, val_set, test_set = split_data(inputs, targets, test_size=0.2, val_size=0.1)
    train_loader, val_loader, test_loader = get_data_loaders(train_set, val_set, test_set, batch_size=32)

    for batch_idx, (x, y) in enumerate(train_loader):
        print(f"Batch {batch_idx}: Inputs shape: {x.shape}, Targets shape: {y.shape}")
