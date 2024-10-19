import torch
import torch.nn as nn


class Quantizer(nn.Module):
    def __init__(self, config):
        super(Quantizer, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(config['codebook_size'], config['latent_dim'])  # Codebook embeddings

    def forward(self, x):
        # Assume x is of shape (batch_size, latent_dim)
        B, D = x.shape  # B = batch_size, D = latent dimension (1280)

        # Compute distance between input x and the codebook embeddings
        # Reshape self.embedding.weight to (1, codebook_size, latent_dim) for broadcasting
        dist = torch.cdist(x.unsqueeze(1), self.embedding.weight[None, :, :])  # Shape: (B, 1, codebook_size)

        # Find the nearest codebook embedding for each input vector
        min_encoding_indices = torch.argmin(dist, dim=-1).squeeze(1)  # Shape: (B,)

        # Select the corresponding embeddings based on the nearest indices
        quant_out = self.embedding(min_encoding_indices)  # Shape: (B, latent_dim)

        # Calculate losses
        commitment_loss = torch.mean((quant_out.detach() - x) ** 2)  # Encourage output to stay close to input
        codebook_loss = torch.mean((quant_out - x.detach()) ** 2)  # Encourage output to represent input well
        quantize_losses = {
            'codebook_loss': codebook_loss,
            'commitment_loss': commitment_loss
        }

        # Apply straight-through estimator (for backpropagation)
        quant_out = x + (quant_out - x).detach()  # Shape remains (B, D)

        return quant_out, quantize_losses, min_encoding_indices

    def quantize_indices(self, indices):
        """
        Quantizes a given set of indices back into their corresponding embeddings.
        """
        return self.embedding(indices)  # Shape: (batch_size, latent_dim)


def get_quantizer(config):
    return Quantizer(config=config['model_params'])