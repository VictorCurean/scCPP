import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.latent_dim = config['latent_dim']  # Encoded dimension (e.g., 1280)
        self.output_dim = config['output_dim']  # Output dimension (1280)

        # Adjust the positional encoding for the latent space
        # If you want to maintain the positional encoding for single embeddings, it can be simplified
        self.positional_encoding = self._generate_positional_encoding(self.latent_dim)  # Remove seq_len

        # Attention layers for decoding
        self.attention_layers = nn.ModuleList([
            nn.Sequential(
                nn.MultiheadAttention(embed_dim=self.latent_dim, num_heads=config['num_heads'], batch_first=True),
                nn.LayerNorm(self.latent_dim),
                nn.GELU(),
            )
            for _ in range(config['attention_blocks'])
        ])

        # Final projection layer to expand the latent dimension to the output dimension
        self.to_output = nn.Linear(self.latent_dim, self.output_dim)

    def _generate_positional_encoding(self, embed_dim):
        """
        Generate positional encoding for the latent space.
        Since we do not have a sequence dimension, we can generate a single positional vector.
        """
        position = torch.zeros(1, embed_dim)  # Shape (1, embed_dim)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        position[0, 0::2] = torch.sin(div_term)
        position[0, 1::2] = torch.cos(div_term)
        return position  # Shape (1, embed_dim)

    def forward(self, x):
        # Add positional encoding to latent input
        x = x + self.positional_encoding.to(x.device)  # Adjust based on the shape

        # Apply attention layers
        # Since the input is (batch_size, latent_dim), we need to reshape it for attention
        x = x.unsqueeze(1)  # Shape (batch_size, 1, latent_dim)
        for attention_layer in self.attention_layers:
            attn_out, _ = attention_layer[0](x, x, x)  # Shape remains (batch_size, 1, latent_dim)
            x = attention_layer[1](attn_out + x)
            x = attention_layer[2](x)

        # Remove the extra dimension added for attention
        x = x.squeeze(1)  # Shape (batch_size, latent_dim)

        # Expand back to output dimension
        output = self.to_output(x)  # Shape (batch_size, output_dim)
        return output

def get_decoder(config):
    decoder = Decoder(
        config=config['model_params']
    )
    return decoder