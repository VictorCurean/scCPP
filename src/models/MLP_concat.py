import torch
import torch.nn as nn

class MLPModel(nn.Module):
    def __init__(self, input_dim, drug_dim, output_dim, hidden_dims, dropout):
        super(MLPModel, self).__init__()
        # Combined input: cell + drug
        combined_dim = input_dim + drug_dim 

        # Build hidden layers dynamically
        layers = []
        in_features = combined_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(),
                nn.Dropout(dropout)
            ])
            in_features = hidden_dim

        # Final projection back to cell embedding space
        layers.append(nn.Linear(in_features, output_dim))
        layers.append(nn.ReLU())

        self.mlp = nn.Sequential(*layers)

    def forward(self, control_cell, drug_emb):
        # Concatenate all inputs
        x = torch.cat([control_cell, drug_emb], dim=-1)
        return self.mlp(x)