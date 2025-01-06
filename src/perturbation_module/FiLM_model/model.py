import torch
import torch.nn as nn


# Utility function to create an MLP dynamically
def create_mlp(input_dim, hidden_dims, output_dim, activation=nn.ReLU):
    layers = []
    dims = [input_dim] + hidden_dims
    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(activation())
    layers.append(nn.Linear(dims[-1], output_dim))  # Final output layer without activation
    return nn.Sequential(*layers)


class FiLMModulator(nn.Module):
    def __init__(self, drug_dim, control_dim, hidden_dims):
        super(FiLMModulator, self).__init__()
        input_dim = drug_dim + 1  # +1 for logdose

        # Shared MLP for both gamma and beta
        self.shared_mlp = create_mlp(input_dim, hidden_dims, 2 * control_dim)  # Output dimension is 2 times control_dim

    def forward(self, drug_emb, logdose):
        # Concatenate drug embedding and logdose
        drug_input = torch.cat([drug_emb, logdose], dim=-1)

        # Compute gamma and beta using the same MLP
        modulated_output = self.shared_mlp(drug_input)

        # Split into gamma and beta
        gamma, beta = torch.chunk(modulated_output, 2, dim=-1)  # Both will match control_dim

        return gamma, beta


# Residual block with FiLM modulation and non-linearity
class FiLMResidualBlock(nn.Module):
    def __init__(self, control_dim, drug_dim, hidden_dim, modulator_hidden_dims):
        super(FiLMResidualBlock, self).__init__()
        self.modulator = FiLMModulator(drug_dim, control_dim, modulator_hidden_dims)
        self.fc1 = nn.Linear(control_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, control_dim)
        self.bn2 = nn.BatchNorm1d(control_dim)

    def forward(self, control_emb, drug_emb, logdose):
        """
        Forward pass with FiLM modulation inside the block.

        Args:
        - control_emb (Tensor): Control embedding of shape (batch_size, control_dim).
        - drug_emb (Tensor): Drug embedding of shape (batch_size, drug_dim).
        - logdose (Tensor): Log dose of shape (batch_size, 1).

        Returns:
        - Tensor: Modulated and processed control embedding of shape (batch_size, control_dim).
        """
        # Compute gamma and beta for this block
        gamma, beta = self.modulator(drug_emb, logdose)

        # Apply FiLM modulation
        modulated_control = gamma * control_emb + beta

        # Pass through block layers
        out = self.fc1(modulated_control)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.bn2(out)

        # Add skip connection
        out = out + control_emb

        return out


# Decoder to generate the treated embedding
class FiLMDecoder(nn.Module):
    def __init__(self, control_dim, output_dim, hidden_dims):
        super(FiLMDecoder, self).__init__()
        self.decoder = create_mlp(control_dim, hidden_dims, output_dim)

    def forward(self, modulated_control):
        return self.decoder(modulated_control)


# Full FiLM Model with residual blocks
class FiLMResidualModel(nn.Module):
    def __init__(self, config):
        super(FiLMResidualModel, self).__init__()
        control_dim = config['model_params']['control_dim']
        drug_dim = config['model_params']['drug_emb_dim']
        modulator_hidden_dims = config['model_params']['hidden_dims_modulator']
        block_hidden_dim = config['model_params']['block_hidden_dim']
        num_blocks = config['model_params']['num_blocks']
        output_dim = config['model_params']['output_dim']
        dropout_rate = config['model_params']['dropout_rate']

        # Residual blocks with FiLM modulation
        self.blocks = nn.ModuleList([
            FiLMResidualBlock(control_dim, drug_dim, block_hidden_dim, modulator_hidden_dims)
            for _ in range(num_blocks)
        ])

        # Final decoder
        self.decoder = FiLMDecoder(control_dim, output_dim, [])

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
        for block in self.blocks:
            control_emb = block(control_emb, drug_emb, logdose)

        treated_emb = self.decoder(control_emb)
        return treated_emb


# # Example configuration
# if __name__ == "__main__":
#     config = {
#         'model_params': {
#             'control_dim': 1280,
#             'drug_emb_dim': 256,
#             'modulator_hidden_dims': [512, 256],
#             'block_hidden_dim': 640,
#             'num_blocks': 4,
#             'output_dim': 1280
#         }
#     }
#     # Instantiate the model
#     model = FiLMResidualModel(config)
#
#     # Example inputs
#     control_emb = torch.randn(16, 1280)  # Batch size of 16, control dimension 1280
#     drug_emb = torch.randn(16, 256)  # Batch size of 16, drug embedding dimension 256
#     logdose = torch.randn(16, 1)  # Batch size of 16, log dose dimension 1
#
#     # Forward pass
#     output = model(control_emb, drug_emb, logdose)
#     print(output.shape)  # Expected: (16, 1280)