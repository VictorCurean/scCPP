import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.latent_dim = config['latent_dim']  # Encoded dimension
        self.input_dim = config['input_dim']    # Input/Output dimension (1280)

        # Positional encoding for input sequence
        self.positional_encoding = self._generate_positional_encoding(self.input_dim, self.input_dim)

        # Attention layers to process the input sequence
        self.attention_layers = nn.ModuleList([
            nn.Sequential(
                nn.MultiheadAttention(embed_dim=self.input_dim, num_heads=config['num_heads'], batch_first=True),
                nn.LayerNorm(self.input_dim),
                nn.GELU(),
            )
            for _ in range(config['attention_blocks'])
        ])

        # Final projection layer to compress the input sequence to latent_dim (e.g., 1280 to 512)
        self.to_latent = nn.Linear(self.input_dim, self.latent_dim)

    def _generate_positional_encoding(self, embed_dim, seq_len):
        """
        Generate positional encoding for input sequence.
        """
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe = torch.zeros(seq_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # Shape (1, seq_len, embed_dim)

    def forward(self, x):
        # Check if x is 2D and unsqueeze to add a sequence dimension
        if len(x.shape) == 2:  # x has shape (batch_size, input_dim)
            x = x.unsqueeze(1)  # Now x has shape (batch_size, 1, input_dim)

        # Expand positional encoding to match the batch size and sequence length of x
        pos_encoding = self.positional_encoding[:, :x.size(1), :]  # Adjust positional encoding for sequence length
        pos_encoding = pos_encoding.expand(x.size(0), -1, -1)

        # Add positional encoding to input
        x = x + pos_encoding.to(x.device)

        # Apply attention layers
        for attention_layer in self.attention_layers:
            attn_out, _ = attention_layer[0](x, x, x)
            x = attention_layer[1](attn_out + x)
            x = attention_layer[2](x)

        # Compress the sequence to the latent dimension
        latent = self.to_latent(x.squeeze(1))  # Squeeze back the sequence dimension before projection
        return latent

def get_encoder(config):
    encoder = Encoder(
        config=config['model_params']
    )
    return encoder