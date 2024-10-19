import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import math
import yaml
from torch.utils.data.dataloader import DataLoader

from dataset import SciplexDatasetBaseline

def test(dataloader, criterion):
    test_losses = list()

    for inputs, targets in dataloader:

        outputs = inputs[:, :1280] #predict the input embedding

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

    print("Loading test data ...")
    sciplex_dataset_test = SciplexDatasetBaseline(config['dataset_params']['test_adata_path'], config['dataset_params']['seed'])
    sciplex_loader_test = DataLoader(sciplex_dataset_test, batch_size=config['train_params']['batch_size'], shuffle=True, num_workers=0)

    print("Evaluating on test set ...")
    criterion = nn.MSELoss()
    loss = test(sciplex_loader_test, criterion)
