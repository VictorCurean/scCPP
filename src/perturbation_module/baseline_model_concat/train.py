import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import math
import yaml
from torch.utils.data.dataloader import DataLoader
import pickle as pkl
import seaborn as sns
import matplotlib.pyplot as plt

from model import ConditionalFeedForwardNN
from dataset import SciplexDatasetBaseline
from dataset_zhao import ZhaoDatasetBaseline

def train(model, dataloader, optimizer, criterion, num_epochs, device):
    model = model.to(device)
    model.train()  # Set the model to training mode
    losses = list()

    for epoch in range(num_epochs):


        for input, output_actual, meta in tqdm(dataloader):
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
    return model, losses

def test(model, dataloader, criterion, device):
    model.eval()
    test_losses = list()
    res = list()
    model_meta = model.get_meta()

    with torch.no_grad():
        for inputs, targets, meta in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            res.append({"input": inputs, "targets": targets, "predicted": outputs, "meta": meta})

            loss = criterion(outputs, targets)

            test_losses.append(loss.item())

        avg_loss = np.mean(test_losses)
        print(f"Test Loss: {avg_loss}")
    return res


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

    with open(ROOT + "data\\sciplex\\drugs_train_list.txt", "r") as f:
        drugs_train = [line.strip() for line in f]

    with open(ROOT + "data\\sciplex\\drugs_validation_list.txt", "r") as f:
        drugs_validation = [line.strip() for line in f]

    print("Reading dataset ...")
    sciplex_dataset = SciplexDatasetBaseline(config['dataset_params']['sciplex_adata_path'], drugs_train)
    sciplex_loader = DataLoader(sciplex_dataset, batch_size=config['train_params']['batch_size'], shuffle=True, num_workers=0)

    print("Training model ...")
    num_epochs = config['train_params']['num_epochs']
    trained_model, losses = train(model, sciplex_loader, optimizer, criterion, num_epochs, device)

    #plot the losses
    index_losses = list(range(len(losses)))
    sns.lineplot(x=index_losses, y=losses)
    plt.ylabel("MSE")
    plt.title("Train Loss")
    plt.show()

    print("Loading test data ...")
    sciplex_dataset_test = SciplexDatasetBaseline(config['dataset_params']['sciplex_adata_path'], drugs_validation)
    sciplex_loader_test = DataLoader(sciplex_dataset_test, batch_size=config['train_params']['batch_size'], shuffle=True, num_workers=0)

    print("Evaluating on test set ...")
    results_test = test(trained_model, sciplex_loader_test, criterion, device)


    print("Saving ... ")

    torch.save(trained_model.state_dict(), ROOT + "results\\baseline\\model_weights.pth")

    with open(ROOT + "results\\baseline\\sciplex_predictions.pkl", "wb") as f:
        pkl.dump(results_test, f)

    # with open(ROOT + "results\\zhao_predictions.pkl", "wb") as f:
    #     pkl.dump(results_zhao, f)