import math
import yaml
import numpy as np
import seaborn as sns
import anndata as ad
import pandas as pd
import scanpy as sc
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from scipy.stats import spearmanr

from model import FiLMResidualModel
from dataset import SciplexDatasetBaseline


class FiLMModelEvaluator():

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
        self.model = FiLMResidualModel(self.config)
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.config['train_params']['lr'],
                                    weight_decay=self.config['train_params']['weight_decay'])
        #self.criterion = nn.MSELoss()
        self.criterion = nn.L1Loss()

        self.model = self.model.to(self.device)

    def __read_data(self):

        #list of drugs to include in the training and test splot

        with open(self.config['dataset_params']['sciplex_drugs_train'], "r") as f:
            drugs_train = [line.strip() for line in f]

        with open(self.config['dataset_params']['sciplex_drugs_test'], "r") as f:
            drugs_validation = [line.strip() for line in f]

        print("Loading sciplex train dataset ...")
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
        print("Begin training ...")
        self.model.train()  # Set the model to training mode
        losses = []

        num_epochs = self.config['train_params']['num_epochs']
        device = self.device  # Target device (e.g., 'cuda' or 'cpu')

        iteration = 0
        every_n = 10

        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")

            for control_emb, drug_emb, logdose, treated_emb, meta in self.sciplex_loader_train:
                # Move tensors to the specified device
                control_emb = control_emb.to(device)
                drug_emb = drug_emb.to(device)
                logdose = logdose.to(device)
                treated_emb = treated_emb.to(device)

                # Zero the gradients
                self.optimizer.zero_grad()

                # Forward pass through the model
                output = self.model(control_emb, drug_emb, logdose)

                # Compute the loss
                loss = self.criterion(output, treated_emb)

                # Backpropagation
                loss.backward()

                # Update model parameters
                self.optimizer.step()

                # Track the loss
                losses.append(loss.item())

                iteration += 1

                if iteration % every_n == 0:
                    print("Iteration:", iteration, "Loss:", loss.item())

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
        doses_list = []
        cell_types_list = []

        self.trained_model.eval()  # Set the model to evaluation mode

        with torch.no_grad():  # Disable gradient computation
            for control_emb, drug_emb, logdose, treated_emb, meta in tqdm(self.sciplex_loader_test):
                # Move tensors to the specified device
                control_emb = control_emb.to(self.device)
                drug_emb = drug_emb.to(self.device)
                logdose = logdose.to(self.device)
                treated_emb = treated_emb.to(self.device)

                # Forward pass through the model
                output = self.trained_model(control_emb, drug_emb, logdose)

                # Convert tensors to lists of NumPy arrays for DataFrame compatibility
                control_emb_list = [x.cpu().numpy() for x in torch.unbind(control_emb, dim=0)]
                treated_emb_list = [x.cpu().numpy() for x in torch.unbind(treated_emb, dim=0)]
                output_list = [x.cpu().numpy() for x in torch.unbind(output, dim=0)]

                # Meta information
                compounds = meta['compound']
                doses = meta['dose']
                doses = [d.item() for d in doses]
                cell_types = meta['cell_type']

                # Append results to lists
                control_embeddings.extend(control_emb_list)
                treated_embeddings.extend(treated_emb_list)
                model_output.extend(output_list)
                compounds_list.extend(compounds)
                doses_list.extend(doses)
                cell_types_list.extend(cell_types)

        # Save results into a DataFrame
        self.test_results = pd.DataFrame({
            "ctrl_emb": control_embeddings,
            "pert_emb": treated_embeddings,
            "pred_emb": model_output,
            "compound": compounds_list,
            "dose": doses_list,
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

    def get_stats(self, dist_func):
        pcp_results = self.get_PCP(dist_func)
        pr_results_perturbed, pr_results_predicted = self.get_PR(dist_func)
        loss_results = self.get_validation_loss(dist_func)
        null_model_loss_results = self.get_null_model_loss(dist_func)

        print("Test PCP:", pcp_results)
        print("Test PR perturbed:", pr_results_perturbed)
        print("Test PR predicted:", pr_results_predicted)
        print("Test Loss:", loss_results)
        print("Null Model Loss:", null_model_loss_results)


    def get_PCP(self, dist_func):
        """
        Get PCP (procentage closer to perturb) based on a given dist function.
        The dist function takes as input two centroids
        """

        results = {"A549": None, "K562": None, "MCF7": None}

        for cell_type in self.test_results['cell_type'].unique():
            no_closer_to_reference = 0
            no_closer_to_perturbed = 0

            for compound in tqdm(list(self.test_results['compound'].unique())):
                for dose in self.test_results['dose'].unique():

                    df_subset = self.test_results[(self.test_results['cell_type'] == cell_type) &
                                                  (self.test_results['compound'] == compound) &
                                                  (self.test_results['dose'] == dose)]

                    centroid_reference =  np.array(df_subset['ctrl_emb'].tolist()).mean(axis=0)
                    centroid_perturbed = np.array(df_subset['pert_emb'].tolist()).mean(axis=0)
                    centroid_predicted = np.array(df_subset['pred_emb'].tolist()).mean(axis=0)

                    if dist_func(centroid_reference, centroid_predicted) < dist_func(centroid_perturbed, centroid_predicted):
                        no_closer_to_reference += 1
                    else:
                        no_closer_to_perturbed += 1



            # Compute the proportion
            total_comparisons = no_closer_to_reference + no_closer_to_perturbed
            pcp = no_closer_to_perturbed / total_comparisons

            results[cell_type] = pcp



        return results

    def get_PR(self, dist_func):
        """
        Get PR (prediction robustness), which measures the spearman correlation between dose and distance to control
        """

        results_perturbed = {"A549": None, "K562": None, "MCF7": None}
        results_predicted = {"A549": None, "K562": None, "MCF7": None}

        for cell_type in self.test_results['cell_type'].unique():
            correlations_perturbed = list()
            correlation_predicted = list()

            for compound in tqdm(list(self.test_results['compound'].unique())):


                df_subset = self.test_results[(self.test_results['cell_type'] == cell_type) &
                                              (self.test_results['compound'] == compound)]



                centroid_reference = np.array(df_subset['ctrl_emb'].tolist()).mean(axis=0)

                doses = sorted(list(df_subset['dose'].unique()))
                perturbed_dist_to_reference = list()
                predicted_dist_to_reference = list()

                for dose in doses:
                    df_dose = df_subset[df_subset['dose'] == dose]

                    centroid_perturbed = np.array(df_dose['pert_emb'].tolist()).mean(axis=0)
                    centroid_predicted = np.array(df_dose['pred_emb'].tolist()).mean(axis=0)

                    dist_reference_perturbed = dist_func(centroid_reference, centroid_perturbed)
                    dist_reference_predicted = dist_func(centroid_reference, centroid_predicted)

                    perturbed_dist_to_reference.append(dist_reference_perturbed)
                    predicted_dist_to_reference.append(dist_reference_predicted)

                corr_perturbed, _ = spearmanr(doses, perturbed_dist_to_reference)
                corr_predicted, _ = spearmanr(doses, predicted_dist_to_reference)

                correlations_perturbed.append(corr_perturbed)
                correlation_predicted.append(corr_predicted)

            corr_perturbed_mean = np.mean(correlations_perturbed)
            corr_predicted_mean = np.mean(correlation_predicted)

            results_perturbed[cell_type] = corr_perturbed_mean
            results_predicted[cell_type] = corr_predicted_mean

        return results_perturbed, results_predicted

    def get_validation_loss(self, dist_func):
        """
        Get average distance between reference and predicted for each cell type
        """

        results = {"A549": None, "K562": None, "MCF7": None}

        for cell_type in self.test_results['cell_type'].unique():
            losses = list()

            df_subset = self.test_results[self.test_results['cell_type'] == cell_type]

            for i, row in df_subset.iterrows():
                losses.append(dist_func(np.array(row['pert_emb']), np.array(row['pred_emb'])))

            avg_loss = np.mean(losses)

            results[cell_type] = avg_loss

        return results


    def get_null_model_loss(self, dist_func):
        """
        Get null model loss for each cell type. In this case, a null model predicts no change so the loss becomes
        the distance function between paired control and treated cells
        """

        results = {"A549": None, "K562": None, "MCF7": None}

        for cell_type in self.test_results['cell_type'].unique():
            losses = list()

            df_subset = self.test_results[self.test_results['cell_type'] == cell_type]

            for i, row in df_subset.iterrows():
                losses.append(dist_func(np.array(row['ctrl_emb']), np.array(row['pert_emb'])))

            avg_loss = np.mean(losses)

            results[cell_type] = avg_loss

        return results


    def plot_training_loss(self):
        plt.figure(figsize=(8, 6))
        index_losses = list(range(len(self.losses_train)))
        sns.lineplot(x=index_losses, y=self.losses_train)
        plt.ylabel("MAE")
        plt.xlabel("Iteration")
        plt.title("Train Loss")
        plt.show()


    def get_model(self):
        return self.trained_model
