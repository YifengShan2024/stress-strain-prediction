import torch
import random

def add_noise(inputs, noise_level=0.01):
    """
    Add Gaussian noise to the input data.
    Args:
        inputs (torch.Tensor): Input tensor.
        noise_level (float): Standard deviation of the Gaussian noise.
    Returns:
        torch.Tensor: Tensor with added noise.
    """
    noise = torch.randn_like(inputs) * noise_level
    return inputs + noise


def random_scaling(inputs, scaling_range=(0.8, 1.2)):
    """
    Apply random scaling to the input data.
    Args:
        inputs (torch.Tensor): Input tensor.
        scaling_range (tuple): Range of scaling factors (min, max).
    Returns:
        torch.Tensor: Scaled input tensor.
    """
    scale_factor = random.uniform(*scaling_range)
    return inputs * scale_factor


def random_flipping(inputs, flip_prob=0.5):
    """
    Randomly flip the input data along a specified axis.
    Args:
        inputs (torch.Tensor): Input tensor.
        flip_prob (float): Probability of flipping.
    Returns:
        torch.Tensor: Flipped input tensor.
    """
    if random.random() < flip_prob:
        return torch.flip(inputs, dims=[-1])  # Flip along the last dimension
    return inputs


def apply_augmentations(inputs, augmentations):
    """
    Apply a list of augmentations to the input data.
    Args:
        inputs (torch.Tensor): Input tensor.
        augmentations (list): List of augmentation functions.
    Returns:
        torch.Tensor: Augmented input tensor.
    """
    for aug in augmentations:
        inputs = aug(inputs)
    return inputs


if __name__ == "__main__":
    # Example input tensor
    example_inputs = torch.rand((10, 5))  # 10 samples, 5 features each

    # Define augmentations
    augmentations = [
        lambda x: add_noise(x, noise_level=0.05),
        lambda x: random_scaling(x, scaling_range=(0.9, 1.1)),
        lambda x: random_flipping(x, flip_prob=0.3),
    ]

    # Apply augmentations
    augmented_inputs = apply_augmentations(example_inputs, augmentations)

    print("Original Inputs:")
    print(example_inputs)
    print("Augmented Inputs:")
    print(augmented_inputs)
