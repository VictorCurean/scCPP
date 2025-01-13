import torch
import torch.nn as nn
import torch.nn.functional as F

class ControlDivergenceLoss(nn.Module):
    def __init__(self, alpha=0.1):
        """
        Custom loss function that combines prediction error with a penalty
        for predictions being too similar to the input.

        Args:
        - alpha (float): Weight of the similarity penalty term.
        """
        super(ControlDivergenceLoss, self).__init__()
        self.alpha = alpha

    def forward(self, predictions, targets, inputs):
        """
        Compute the custom loss.

        Args:
        - predictions (Tensor): Model predictions (batch_size, output_dim).
        - targets (Tensor): Ground truth (batch_size, output_dim).
        - inputs (Tensor): Input embeddings used in the model (batch_size, input_dim).

        Returns:
        - Tensor: Combined loss value.
        """
        # Compute the primary loss (e.g., MSE loss)
        primary_loss = F.mse_loss(predictions, targets)

        # Compute the similarity penalty (using cosine similarity)
        similarity = F.cosine_similarity(predictions, inputs, dim=-1)
        similarity_penalty = torch.mean(similarity)  # Penalize high similarity

        # Combine the losses
        total_loss = primary_loss + self.alpha * similarity_penalty
        return total_loss