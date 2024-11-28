import math
import yaml
import numpy as np
import seaborn as sns
import anndata as add
import pandas as pd
import scanpy as sc
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_curve, auc, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier

from model import ConditionalFeedForwardNN
from dataset import SciplexDatasetBaseline
from dataset_zhao import ZhaoDatasetBaseline


class BaselineModelEvaluator():

    def __init__(self, config_path):
        # load config file
        self.__read_config(config_path)

        #prepare model
        self.__prepare_model()

        #read data
        self.__read_data()

    def __read_config(self, config_path):
        with open(config_path, 'r') as file:
            try:
                self.config = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)
                raise RuntimeError(exc)

    def __prepare_model(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ConditionalFeedForwardNN(self.config)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['train_params']['lr'])
        #self.criterion = nn.MSELoss()
        self.criterion = nn.L1Loss()

        self.model = self.model.to(self.device)

    def __read_data(self):

        #list of drugs to include in the training and test splot

        with open(self.config['dataset_params']['sciplex_drugs_train'], "r") as f:
            drugs_train = [line.strip() for line in f]

        with open(self.config['dataset_params']['sciplex_drugs_test'], "r") as f:
            drugs_validation = [line.strip() for line in f]

        print("Loading train dataset ...")
        sciplex_dataset_train = SciplexDatasetBaseline(self.config['dataset_params']['sciplex_adata_path'],
                                                      drugs_train)
        self.sciplex_loader_train = DataLoader(sciplex_dataset_train, batch_size=self.config['train_params']['batch_size'],
                                         shuffle=True,
                                         num_workers=0)

        print("Loading sciplex test dataset ...")
        sciplex_dataset_test = SciplexDatasetBaseline(self.config['dataset_params']['sciplex_adata_path'], drugs_validation)
        self.sciplex_loader_test = DataLoader(sciplex_dataset_test, batch_size=self.config['train_params']['batch_size'],
                                         shuffle=True, num_workers=0)

    def train(self):
        print("Begin training ... ")
        self.model.train()  # Set the model to training mode
        losses = list()

        num_epochs = self.config['train_params']['num_epochs']

        for epoch in range(num_epochs):

            for control_emb, drug_emb, logdose, treated_emb, meta in tqdm(self.sciplex_loader_train):

                input = torch.cat([control_emb, drug_emb, logdose], dim=-1)
                # Move tensors to the specified device

                input = input.to(self.device)
                treated_emb = treated_emb.to(self.device)

                self.optimizer.zero_grad()

                output = self.model(input)

                loss = self.criterion(output, treated_emb)
                #loss = euclidean(output.cpu(), treated_emb.cpu())


                print(loss)

                loss.backward()

                self.optimizer.step()

                losses.append(loss.item())

        self.losses_train = losses
        self.trained_model = self.model

        print("Training completed ...")

    def test(self):
        self.control_embeddings = list()
        self.treated_embeddings = list()
        self.model_output = list()
        self.compounds = list()
        self.doses = list()
        self.cell_types = list()


        self.trained_model.eval()

        with torch.no_grad():
            for control_emb, drug_emb, logdose, treated_emb, meta in tqdm(self.sciplex_loader_test):
                input = torch.cat([control_emb, drug_emb, logdose], dim=-1)
                input = input.to(self.device)
                treated_emb = treated_emb.to(self.device)

                output = self.trained_model(input)


                loss = self.criterion(output, treated_emb)

                #decompose it into lists
                control_emb = [x.cpu() for x in list(torch.unbind(control_emb, dim=0))]
                #drug_emb = torch.unbind(drug_emb, dim=0)
                #logdose = torch.unbind(logdose, dim=0)
                treated_emb = [x.cpu() for x in list(torch.unbind(treated_emb, dim=0))]
                output = [x.cpu() for x in list(torch.unbind(output, dim=0))]

                compounds = meta['compound']
                doses = meta['dose']
                cell_types = meta['cell_type']

                self.control_embeddings += control_emb
                self.treated_embeddings += treated_emb
                self.model_output += output
                self.compounds += compounds
                self.doses += doses
                self.cell_types += cell_types

        self.test_results = pd.DataFrame({
            "ctrl_emb": self.control_embeddings,
            "pert_emb": self.treated_embeddings,
            "pred_emb": self.model_output,
            "compound": self.compounds,
            "dose": self.doses,
            "cell_type": self.cell_types,
        })

    def plot_stats(self):
        dist_ctrl_pert = list()
        dist_ctrl_pred = list()
        dist_pert_pred = list()

        for x in self.test_results.iterrows():
            dist_ctrl_pert.append(euclidean(row['ctrl_emb'], row['pert_emb']))
            dist_ctrl_pred.append(euclidean(row['ctrl_emb'], row['pred_emb']))
            dist_pert_pred.append(euclidean(row['pert_emb'], row['pred_emb']))

        data = pd.DataFrame({
            "Value": dist_ctrl_pert + dist_ctrl_pred + dist_pert_pred,
            "Category": (
                    ["ctrl_pert"] * len(dist_ctrl_pert) +
                    ["ctrl_pred"] * len(dist_ctrl_pred) +
                    ["pert_pred"] * len(dist_pert_pred)
            )
        })

        # Create the boxplot
        plt.figure(figsize=(8, 6))
        sns.boxplot(x="Category", y="Value", data=data, palette="Set2")

        # Add titles and labels
        plt.title("Distribution of Distances", fontsize=16)
        plt.xlabel("Samples", fontsize=14)
        plt.ylabel("Euclidean Distance between embeddings", fontsize=14)
        plt.show()

            