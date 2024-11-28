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

import anndata as ad
import pandas as pd
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_curve, auc, precision_score, recall_score
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_curve, auc, precision_score, recall_score
import seaborn as sns

def train(model, dataloader, optimizer, criterion, num_epochs, device):
    model = model.to(device)
    model.train()  # Set the model to training mode
    losses = list()

    for epoch in range(num_epochs):


        for control_emb, drug_emb, logdose, treated_emb, meta in tqdm(dataloader):
            # Move tensors to the specified device

            input = torch.cat([control_emb, drug_emb, logdose], dim=-1)

            input = input.to(device)
            treated_emb = treated_emb.to(device)

            optimizer.zero_grad()

            output = model(input)

            loss = criterion(output, treated_emb)

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

    with torch.no_grad():
        for control_emb, drug_emb, logdose, treated_emb, meta in dataloader:

            input = torch.cat([control_emb, drug_emb, logdose], dim=-1)
            input = input.to(device)
            treated_emb = treated_emb.to(device)

            output = model(input)
            res.append({"input": input, "targets": treated_emb, "predicted": output, "meta": meta})

            loss = criterion(output, treated_emb)

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
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()

    with open(ROOT + "data\\sciplex\\drugs_train_list_TEST.txt", "r") as f:
        drugs_train = [line.strip() for line in f]

    with open(ROOT + "data\\sciplex\\drugs_validation_list_TEST.txt", "r") as f:
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

    # print("Loading test data ...")
    # sciplex_dataset_test = SciplexDatasetBaseline(config['dataset_params']['sciplex_adata_path'], drugs_validation)
    # sciplex_loader_test = DataLoader(sciplex_dataset_test, batch_size=config['train_params']['batch_size'], shuffle=True, num_workers=0)
    #
    # print("Evaluating on test set ...")
    # results_test = test(trained_model, sciplex_loader_test, criterion, device)

    adata_zhao = ad.read_h5ad(self.ROOT + "data\\zhao\\zhao_preprocessed.h5ad")

    #sample PW052, pert_name = etoposide

    adata_zhao = adata_zhao[adata_zhao.obs['sample'] == "PW052"]
    adata_zhao = adata_zhao[adata_zhao.obs['perturbation'].isin(["control", "etoposide"])]

    X_control = adata_zhao[adata_zhao.obs['perturbation'] == "control"].obsm['X_uce']
    y_control = [0 for _ in range(X_control.shape[0])]

    X_prt = adata_zhao[adata_zhao.obs['perturbation'] == "etoposide"].obsm['X_uce']
    y_prt = [1 for _ in range(X_prt.shape[0])]

    X = np.vstack((X_control, X_prt))
    y = np.array(y_control + y_prt)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1702)
    classifier = LogisticRegression(random_state=1702, max_iter=1000, class_weight='balanced')
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    y_pred_proba = classifier.predict_proba(X_test)[:, 1]

    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)

    # Calculate confusion matrix and plot it
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 4))

    # Confusion Matrix Plot
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Control', 'Treatment'],
                yticklabels=['Control', 'Treatment'])
    plt.title(f"Confusion Matrix - {sample} ({prt})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # Precision-Recall Curve Plot
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {sample} ({prt})")
    plt.legend()

    # Show both plots together for each treatment
    plt.tight_layout()
    plt.show()

    #see how model predictions affect classifier
    zhao_dataset = ZhaoDatasetBaseline(adata_zhao)
    zhao_loader = DataLoader(zhao_dataset)

    validation_results_zhao = list()

    with torch.no_grad():
        for inputs, targets, meta in zhao_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            outputs = self.trained_model(inputs)
            validation_results_zhao.append(
                {"input": inputs, "targets": targets, "predicted": outputs, "meta": meta})




