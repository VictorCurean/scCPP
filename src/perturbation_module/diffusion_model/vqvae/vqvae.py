import torch
import torch.nn as nn
from encoder import get_encoder
from decoder import get_decoder
from quantizer import get_quantizer


class VQVAE(nn.Module):
    def __init__(self, config):
        super(VQVAE, self).__init__()
        self.encoder = get_encoder(config)

        # Linear layer before quantization
        self.pre_quant_fc = nn.Linear(config['model_params']['latent_dim'],
                                      config['model_params']['latent_dim'])

        # Quantizer
        self.quantizer = get_quantizer(config)

        # Linear layer after quantization
        self.post_quant_fc = nn.Linear(config['model_params']['latent_dim'],
                                       config['model_params']['latent_dim'])

        # Decoder
        self.decoder = get_decoder(config)

    def forward(self, x):
        # Encoder: Produces a latent sequence (batch_size, seq_len, latent_dim)
        enc = self.encoder(x)

        # Pre-quantization transformation (using a Linear layer instead of Conv)
        quant_input = self.pre_quant_fc(enc)

        # Quantization step
        quant_output, quant_loss, quant_idxs = self.quantizer(quant_input)

        # Post-quantization transformation (using a Linear layer instead of Conv)
        dec_input = self.post_quant_fc(quant_output)

        # Decoder to reconstruct the output
        out = self.decoder(dec_input)

        return {
            'generated_output': out,
            'quantized_output': quant_output,
            'quantized_losses': quant_loss,
            'quantized_indices': quant_idxs
        }

    def decode_from_codebook_indices(self, indices):
        # Convert indices back to quantized embeddings
        quantized_output = self.quantizer.quantize_indices(indices)

        # Post-quantization transformation
        dec_input = self.post_quant_fc(quantized_output)

        # Pass through the decoder to get the final output
        return self.decoder(dec_input)


def get_model(config):
    print(config)
    model = VQVAE(config=config)
    return model