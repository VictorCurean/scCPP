import torch
import torch.nn as nn
import torch.nn.functional as F


class CVAE(nn.Module):
    def __init__(self, input_dim, cond_dim, latent_dim, output_dim, hidden_dim=128):
        """
        Parameters:
            input_dim (int): Dimension of control_emb.
            cond_dim (int): Dimension of drug_emb + logdose.
            latent_dim (int): Dimension of the latent space.
            output_dim (int): Dimension of treated_emb.
            hidden_dim (int): Hidden layer dimension in encoder/decoder.
        """
        super(CVAE, self).__init__()

        # Encoder: Takes control_emb and conditioning vector (drug_emb + logdose)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

        # Decoder: Takes latent z and conditioning vector (drug_emb + logdose)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def encode(self, control_emb, cond):
        """Encodes the input into latent space."""
        x = torch.cat([control_emb, cond], dim=-1)  # Concatenate control_emb and condition
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick to sample z from q(z|x)."""
        std = torch.exp(0.5 * logvar)  # Standard deviation
        eps = torch.randn_like(std)  # Random noise
        return mu + eps * std

    def decode(self, z, cond):
        """Decodes the latent vector back to the output space."""
        x = torch.cat([z, cond], dim=-1)  # Concatenate z and condition
        return self.decoder(x)

    def forward(self, control_emb, drug_emb, logdose):
        """
        Forward pass for the CVAE.
        control_emb: Input (conditioned).
        drug_emb: Drug embedding (conditioning input).
        logdose: Dose level (conditioning input).
        """
        # Concatenate conditioning inputs
        cond = torch.cat([drug_emb, logdose.unsqueeze(-1)], dim=-1)

        # Encode to latent space
        mu, logvar = self.encode(control_emb, cond)

        # Sample from latent space
        z = self.reparameterize(mu, logvar)

        # Decode to output space
        treated_emb = self.decode(z, cond)

        return treated_emb, mu, logvar