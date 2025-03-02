import torch
import torch.nn as nn


class FiLM(nn.Module):
    def __init__(self, config):
        super(FiLM, self).__init__()
        condition_dim = config['model_params']['drug_emb_dim']
        latent_dim = config['model_params']['latent_dim']

        # More robust conditioning network with regularization
        self.gamma = nn.Sequential(
            nn.Linear(condition_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(config['model_params']['dropout']),
            nn.Linear(512, latent_dim)
        )
        self.beta = nn.Sequential(
            nn.Linear(condition_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(config['model_params']['dropout']),
            nn.Linear(latent_dim, latent_dim)
        )

        # #Initialize for stable L1 optimization
        # nn.init.uniform_(self.gamma[-1].weight, 0.9, 1.1)  # Start near identity
        # nn.init.normal_(self.beta[-1].weight, 0, 0.1)      # Small initial shifts

    def forward(self, control_cell, drug_emb):
        #condition = torch.cat([condition], dim=-1)
        return self.gamma(drug_emb) * control_cell + self.beta(drug_emb)


class FiLMModel(nn.Module):
    def __init__(self, config):
        super(FiLMModel, self).__init__()
        input_dim = config['model_params']['control_dim']
        latent_dim = config['model_params']['latent_dim']
        output_dim = config['model_params']['output_dim']

        # Smoother dimensionality reduction
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(config['model_params']['dropout']),
            nn.Linear(512, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU()
        )

        # FiLM modulation layers
        self.film_layers = nn.ModuleList([
            nn.Sequential(
                FiLM(config),
                nn.LayerNorm(latent_dim),
                nn.GELU()
            ) for _ in range(config['model_params']['num_layers'])
        ])

        # Expanded reconstruction
        self.output_proj = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(config['model_params']['dropout']),
            nn.Linear(512, output_dim)
        )

    def forward(self, control_cell, drug_emb):
        # Progressive input projection
        x = self.input_proj(control_cell)

        for film_block in self.film_layers:
            residual = x
            # Correct order: FiLM → Residual → LayerNorm → ReLU
            x = film_block[0](x, drug_emb)
            x = x + residual
            x = film_block[1](x)  # LayerNorm after residual
            x = film_block[2](x)  # ReLU
        return self.output_proj(x)

        # Gradual output reconstruction
        return self.output_proj(x)