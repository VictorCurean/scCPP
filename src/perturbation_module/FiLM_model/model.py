import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMModel(nn.Module):
    def __init__(self, config):
        """
        Args:
        - control_dim (int): Dimension of the control embedding.
        - drug_dim (int): Dimension of the drug embedding.
        - hidden_dim (int): Dimension of the hidden layers.
        - output_dim (int): Dimension of the output embedding (treated embedding).
        """
        super(FiLMModel, self).__init__()

        control_dim = config['model_params']['control_dim']
        drug_dim = config['model_params']['drug_emb_dim']
        hidden_dim = config['model_params']['hidden_dim']
        output_dim = config['model_params']['output_dim']


        # FiLM layers to generate gamma and beta from drug embedding and dose
        self.gamma_layer = nn.Sequential(
            nn.Linear(drug_dim + 1, hidden_dim),  # +1 for logdose
            nn.ReLU(),
            nn.Linear(hidden_dim, control_dim)  # Match the dimension of control_emb
        )
        self.beta_layer = nn.Sequential(
            nn.Linear(drug_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, control_dim)
        )

        # Decoder to generate the treated embedding
        self.decoder = nn.Sequential(
            nn.Linear(control_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, control_emb, drug_emb, logdose):
        """
        Forward pass for the FiLM model.

        Args:
        - control_emb (Tensor): Control embedding of shape (batch_size, control_dim).
        - drug_emb (Tensor): Drug embedding of shape (batch_size, drug_dim).
        - logdose (Tensor): Log dose of shape (batch_size, 1).

        Returns:
        - treated_emb (Tensor): Predicted treated embedding of shape (batch_size, output_dim).
        """
        # Concatenate drug embedding and log dose
        drug_input = torch.cat([drug_emb, logdose], dim=-1)  # Shape: (batch_size, drug_dim + 1)

        # Compute FiLM modulation parameters
        gamma = self.gamma_layer(drug_input)  # Shape: (batch_size, control_dim)
        beta = self.beta_layer(drug_input)  # Shape: (batch_size, control_dim)

        # Apply FiLM modulation to the control embedding
        modulated_control = gamma * control_emb + beta  # Shape: (batch_size, control_dim)

        # Decode the modulated embedding into the treated embedding
        treated_emb = self.decoder(modulated_control)  # Shape: (batch_size, output_dim)

        return treated_emb