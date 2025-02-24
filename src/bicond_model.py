import torch
import torch.nn as nn

class MutualFiLMLayer(nn.Module):
    def __init__(self, dim_A, dim_B, hidden_dim=128):
        super().__init__()
        # FiLM generators for A → B modulation
        self.film_A2B = nn.Sequential(
            nn.Linear(dim_A, hidden_dim),  # Project to hidden_dim
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * dim_B)  # Output γ and β for B
        )

        # FiLM generators for B → A modulation
        self.film_B2A = nn.Sequential(
            nn.Linear(dim_B, hidden_dim),  # Project to hidden_dim
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * dim_A)  # Output γ and β for A
        )

        # Feature fusion with hidden_dim
        self.fuse = nn.Sequential(
            nn.Linear(dim_A + dim_B, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim_A + dim_B)
        )

    def forward(self, z_A, z_B):
        # A → B modulation
        film_params_A2B = self.film_A2B(z_A)  # [batch, 2*dim_B]
        γ_A2B, β_A2B = torch.chunk(film_params_A2B, 2, dim=-1)
        z_B_mod = γ_A2B * z_B + β_A2B  # [batch, dim_B]

        # B → A modulation
        film_params_B2A = self.film_B2A(z_B)  # [batch, 2*dim_A]
        γ_B2A, β_B2A = torch.chunk(film_params_B2A, 2, dim=-1)
        z_A_mod = γ_B2A * z_A + β_B2A  # [batch, dim_A]

        # Fuse and update with residual connections
        z_fused = self.fuse(torch.cat([z_A_mod, z_B_mod], dim=-1))
        z_A_new, z_B_new = torch.split(z_fused, [z_A.size(-1), z_B.size(-1)], dim=-1)

        return z_A + z_A_new, z_B + z_B_new


class MutualFiLMModel(nn.Module):
    def __init__(self, dim_cell, dim_mol, hidden_dim=128, num_layers=4):
        super().__init__()
        # Project molecule + dosage to dim_mol
        self.mol_dosage_encoder = nn.Sequential(
            nn.Linear(dim_mol + 1, hidden_dim),  # +1 for dosage
            nn.GELU(),
            nn.Linear(hidden_dim, dim_mol)
        )
        # Initialize mutual FiLM layers
        self.layers = nn.ModuleList([
            MutualFiLMLayer(dim_cell, dim_mol, hidden_dim)
            for _ in range(num_layers)
        ])
        # Final projection to post-perturbation embedding
        self.output_proj = nn.Linear(dim_cell + dim_mol, dim_cell)

    def forward(self, z_initial, z_mol, dosage):
        # Encode molecule + dosage
        z_mol_d = self.mol_dosage_encoder(
            torch.cat([z_mol, dosage.unsqueeze(-1)], dim=-1)
        )
        # Iterate through mutual modulation layers
        z_A, z_B = z_initial, z_mol_d
        for layer in self.layers:
            z_A, z_B = layer(z_A, z_B)
        # Predict post-perturbation state
        return self.output_proj(torch.cat([z_A, z_B], dim=-1))