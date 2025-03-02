import torch
import torch.nn as nn

class CellDecoder(nn.Module):
    def __init__(self, config):
        super(CellDecoder, self).__init__()
        input_dim = config['model_params']['control_dim']
        output_dim = config['model_params']['output_dim']
        self.hidden_layers = config['model_params']['hidden_dims']
        dropout = config['model_params']['dropout']


        # Build hidden layers dynamically
        layers = []
        in_features = input_dim
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

    def forward(self, input):
        return self.mlp(input)