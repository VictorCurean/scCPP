import torch
import torch.nn as nn

class CompoundDecoder(nn.Module):
    def __init__(self, config):
        super(CompoundDecoder, self).__init__()
        drug_emb_dim = config['model_params']['drug_emb_dim']
        cell_onehot_dim = config['model_params']['cell_onehot_dim']
        output_dim = config['model_params']['output_dim']

        self.hidden_layers = config['model_params']['hidden_dims']
        dropout = config['model_params']['dropout']


        # Build hidden layers dynamically
        layers = []
        in_features = drug_emb_dim + cell_onehot_dim
        for hidden_dim in self.hidden_layers:
            layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            in_features = hidden_dim

        # Final projection
        layers.append(nn.Linear(in_features, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, drug_emb_dim, cell_onehot):
        x = torch.cat([drug_emb_dim, cell_onehot], dim=-1)
        return self.mlp(x)