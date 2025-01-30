import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLM(nn.Module):
    def __init__(self, config):

        condition_dim = config['model_params']['drug_emb_dim'] + 1 #for dose
        input_dim = config['model_params']['control_dim']

        super(FiLM, self).__init__()
        self.gamma = nn.Linear(condition_dim, input_dim)
        self.beta = nn.Linear(condition_dim, input_dim)

    def forward(self, x, condition, dose):
        condition = torch.cat([condition, dose], dim=-1)
        gamma = self.gamma(condition)
        beta = self.beta(condition)
        return gamma * x + beta


class FiLMModel(nn.Module):
    def __init__(self, config):

        input_dim = config['model_params']['control_dim']
        condition_dim = config['model_params']['drug_emb_dim'] + 1 # for dose
        num_film_layers = config['model_params']['num_layers']

        super(FiLMModel, self).__init__()
        self.num_film_layers = num_film_layers
        self.film_layers = nn.ModuleList([FiLM(config) for _ in range(num_film_layers)])
        self.fc_out = nn.Linear(input_dim, input_dim)

    def forward(self, input, condition, dose):
        for film in self.film_layers:
            x = film(input, condition, dose)
        x = self.fc_out(x)
        return x

