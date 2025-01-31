import torch
import torch.nn as nn


class FiLM(nn.Module):
    def __init__(self, config):
        super(FiLM, self).__init__()
        condition_dim = config['model_params']['drug_emb_dim']
        hidden_dim = config['model_params']['hidden_dim']

        # More robust conditioning network with regularization
        self.gamma = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(config['model_params']['dropout']),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.beta = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(config['model_params']['dropout']),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Initialize for stable L1 optimization
        nn.init.uniform_(self.gamma[-1].weight, 0.9, 1.1)  # Start near identity
        nn.init.normal_(self.beta[-1].weight, 0, 0.1)      # Small initial shifts

    def forward(self, x, condition):
        condition = torch.cat([condition], dim=-1)
        return self.gamma(condition) * x + self.beta(condition)


class FiLMModel(nn.Module):
    def __init__(self, config):
        super(FiLMModel, self).__init__()
        input_dim = config['model_params']['control_dim']
        hidden_dim = config['model_params']['hidden_dim']

        # Smoother dimensionality reduction
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(config['model_params']['dropout']),
            nn.Linear(512, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # FiLM modulation layers
        self.film_layers = nn.ModuleList([
            nn.Sequential(
                FiLM(config),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ) for _ in range(config['model_params']['num_layers'])
        ])

        # Expanded reconstruction
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(config['model_params']['dropout']),
            nn.Linear(512, input_dim)
        )

    def forward(self, input, condition):
        # Progressive input projection
        x = self.input_proj(input)

        # Residual FiLM modulation
        for film_block in self.film_layers:
            residual = x
            x = film_block[0](x, condition)
            x = film_block[1](x)  # BatchNorm
            x = film_block[2](x)  # ReLU
            x = x + residual  # Preserve information

        # Gradual output reconstruction
        return self.output_proj(x)