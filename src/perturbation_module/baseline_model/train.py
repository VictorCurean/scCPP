import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import math
import yaml
from torch.utils.data.dataloader import DataLoader

from model import ConditionalFeedForwardNN
from dataset import SciplexDatasetBaseline

def train(model, dataloader, optimizer, criterion, num_epochs, device):
    model = model.to(device)
    model.train()  # Set the model to training mode

    for epoch in range(num_epochs):
        losses = list()

        for input, output_actual in tqdm(dataloader):
            # Move tensors to the specified device
            input = input.to(device)
            output_actual = output_actual.to(device)

            optimizer.zero_grad()

            output = model(input)

            loss = criterion(output, output_actual)

            loss.backward()

            optimizer.step()

            losses.append(loss.item())


        # Print statistics for the epoch
        avg_loss = np.mean(losses)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

    print("Training completed.")
    return model

def test(model, dataloader, criterion, device):
    model.eval()
    test_losses = list()

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_losses.append(loss.item())

        avg_loss = np.mean(test_losses)
        print(f"Test Loss: {avg_loss}")


if __name__ == "__main__":
    ROOT = 'C:\\Users\\curea\\Documents\\bioFM for drug discovery\\dege-fm\\'

    with open(ROOT+ "config\\baseline.yaml", 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    print("Initializing ...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConditionalFeedForwardNN(config)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    print("Reading dataset ...")
    sciplex_dataset = SciplexDatasetBaseline(config['dataset_params']['train_adata_path'], config['dataset_params']['seed'])
    sciplex_loader = DataLoader(sciplex_dataset, batch_size=config['train_params']['batch_size'], shuffle=True, num_workers=0)

    print("Training model ...")
    num_epochs = config['train_params']['num_epochs']
    trained_model = train(model, sciplex_loader, optimizer, criterion, num_epochs, device)

    print("Loading test data ...")
    sciplex_dataset_test = SciplexDatasetBaseline(config['dataset_params']['test_adata_path'], config['dataset_params']['seed'])
    sciplex_loader_test = DataLoader(sciplex_dataset_test, batch_size=config['train_params']['batch_size'], shuffle=True, num_workers=0)

    print("Evaluating on test set ...")
    loss = test(model, sciplex_loader_test, criterion, device)
