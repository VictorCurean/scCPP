import torch
import torch.nn as nn


class ConditionalFeedForwardNN(nn.Module):
    def __init__(self, config):
        super(ConditionalFeedForwardNN, self).__init__()

        # Define the input size (concatenated cell and condition embeddings + 1 for log of dose)
        input_dim = config['model_params']['cell_embedding_dim'] + config['model_params']['condition_dim'] + 1

        # Create the network layers
        layers = []
        hidden_dims = config['model_params']['hidden_dims']

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dims[i - 1], hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim  # Update input_dim for next layer

        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], config['model_params']['output_dim']))

        # Define the sequential feedforward model
        self.network = nn.Sequential(*layers)

        # Define meta-data for model
        self.model_config = config['model_params']
        self.train_config = config['train_params']

    def get_meta(self):
        return self.model_config, self.train_config


    def forward(self, input):
        # Pass through the network
        output = self.network(input)
        return output