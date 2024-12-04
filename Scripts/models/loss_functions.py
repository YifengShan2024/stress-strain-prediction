import torch

def hybrid_loss(predicted, target, physical_residual, alpha=0.5):
    mse_loss = torch.nn.functional.mse_loss(predicted, target)
    physical_loss = torch.norm(physical_residual, p=2)
    return alpha * mse_loss + (1 - alpha) * physical_loss
