import yaml
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt

import torch
import pandas as pd
import pickle as pkl
from torch.utils.data import Dataset
import random
import numpy as np
import math
import anndata as ad
import ast
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

from model import FiLMModel
from dataset import SciplexDatasetUnseenPerturbations

def loss_fn(pred, target, control):
    # L1 loss (primary term)
    l1_loss = F.l1_loss(pred, target)
    
    pred_dir = pred - control
    target_dir = target - control
    cos_loss = 1 - F.cosine_similarity(pred_dir, target_dir).mean()
    
    return l1_loss + 0.3 * cos_loss


class FiLMModelEvaluator():

    def __init__(self, config_path, model, sciplex_dataset_train, sciplex_dataset_validation, sciplex_dataset_test):
        # load config file
        self.__read_config(config_path)

        #prepare model
        self.__prepare_model(model)

        self.sciplex_loader_train = DataLoader(sciplex_dataset_train,
                                               batch_size=self.config['train_params']['batch_size'],
                                               shuffle=True,
                                               num_workers=0)

        self.sciplex_loader_validation = DataLoader(sciplex_dataset_validation,
                                               batch_size=self.config['train_params']['batch_size'],
                                               shuffle=True,
                                               num_workers=0)

        self.sciplex_loader_test = DataLoader(sciplex_dataset_test,
                                              batch_size=self.config['train_params']['batch_size'],
                                              shuffle=True, num_workers=0)

    def __read_config(self, config_path):
        with open(config_path, 'r') as file:
            try:
                self.config = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)
                raise RuntimeError(exc)



    def __prepare_model(self, model):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model(self.config)
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.config['train_params']['lr'],
                                    weight_decay=self.config['train_params']['weight_decay'])
        self.criterion = nn.L1Loss()

        self.model = self.model.to(self.device)

    def train(self):
        print("Begin training ...")
        self.model.train()  # Set the model to training mode
        losses = []

        num_epochs = self.config['train_params']['num_epochs']
        device = self.device  # Target device (e.g., 'cuda' or 'cpu')

        iteration = 0
        every_n = 10

        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")

            for control_emb, drug_emb, treated_emb, meta in self.sciplex_loader_train:
                # Move tensors to the specified device
                control_emb = control_emb.to(device)
                drug_emb = drug_emb.to(device)
                treated_emb = treated_emb.to(device)

                # Zero the gradients
                self.optimizer.zero_grad()

                # Forward pass through the model
                output = self.model(control_emb, drug_emb)

                # Compute the loss
                #loss = self.criterion(output, treated_emb)
                loss = loss_fn(output, treated_emb, control_emb)

                # Backpropagation
                loss.backward()

                # Update model parameters
                self.optimizer.step()

                # Track the loss
                losses.append(loss.item())

                iteration += 1

                #############VALIDATION LOOP#################

                if iteration % every_n == 0:


                    validation_losses = list()
                    with torch.no_grad():
                        for control_emb, drug_emb, treated_emb, meta in self.sciplex_loader_validation:
                            control_emb, drug_emb, treated_emb = (
                                control_emb.to(device),
                                drug_emb.to(device),
                                treated_emb.to(device),
                            )

                            # Forward pass
                            output_validation = self.model(control_emb, drug_emb)

                            # Compute loss
                            validation_loss = loss_fn(output_validation, treated_emb, control_emb)

                            # Track validation loss
                            validation_losses.append(validation_loss.item())


                    print("Iteration:", iteration, "Test Loss:", loss.item(), "Avg. Validation Loss:", np.mean(validation_losses))

                #############VALIDATION LOOP#################

        self.losses_train = losses
        self.trained_model = self.model

        print("Training completed.")

    def test(self, save_path=None):
        """
        Test the FiLMResidualModel and collect results.
        """
        control_embeddings = []
        treated_embeddings = []
        model_output = []
        compounds_list = []
        cell_types_list = []

        self.trained_model.eval()  # Set the model to evaluation mode

        with torch.no_grad():  # Disable gradient computation
            for control_emb, drug_emb, treated_emb, meta in tqdm(self.sciplex_loader_test):
                # Move tensors to the specified device
                control_emb = control_emb.to(self.device)
                drug_emb = drug_emb.to(self.device)
                treated_emb = treated_emb.to(self.device)

                # Forward pass through the model
                output = self.trained_model(control_emb, drug_emb)

                # Convert tensors to lists of NumPy arrays for DataFrame compatibility
                control_emb_list = [x.cpu().numpy() for x in torch.unbind(control_emb, dim=0)]
                treated_emb_list = [x.cpu().numpy() for x in torch.unbind(treated_emb, dim=0)]
                output_list = [x.cpu().numpy() for x in torch.unbind(output, dim=0)]

                # Meta information
                compounds = meta['compound']
                cell_types = meta['cell_type']

                # Append results to lists
                control_embeddings.extend(control_emb_list)
                treated_embeddings.extend(treated_emb_list)
                model_output.extend(output_list)
                compounds_list.extend(compounds)
                cell_types_list.extend(cell_types)

        # Save results into a DataFrame
        self.test_results = pd.DataFrame({
            "ctrl_emb": control_embeddings,
            "pert_emb": treated_embeddings,
            "pred_emb": model_output,
            "compound": compounds_list,
            "cell_type": cell_types_list,
        })

        print("Testing completed. Results stored in 'self.test_results'.")

        # Save to file if save_path is provided
        if save_path:
            self.save_results(save_path)

    def save_results(self, save_path):
        """
        Saves test results to a file. Supports multiple formats (CSV, JSON, Pickle).
        """
        file_extension = save_path.split('.')[-1]

        if file_extension == 'csv':
            # Convert embeddings to lists for CSV compatibility
            df_to_save = self.test_results.copy()
            df_to_save['ctrl_emb'] = df_to_save['ctrl_emb'].apply(list)
            df_to_save['pert_emb'] = df_to_save['pert_emb'].apply(list)
            df_to_save['pred_emb'] = df_to_save['pred_emb'].apply(list)
            df_to_save.to_csv(save_path, index=False)
        elif file_extension == 'json':
            # Save to JSON
            self.test_results.to_json(save_path, orient='records')
        elif file_extension == 'pkl':
            # Save to Pickle for preserving object types like tensors
            self.test_results.to_pickle(save_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

        print(f"Results saved to {save_path}.")

    def plot_training_loss(self):
        plt.figure(figsize=(8, 6))
        index_losses = list(range(len(self.losses_train)))
        sns.lineplot(x=index_losses, y=self.losses_train)
        plt.ylabel("MAE")
        plt.xlabel("Iteration")
        plt.title("Train Loss")
        plt.show()


    def get_test_results(self):
        return self.test_results