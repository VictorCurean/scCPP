import yaml
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from abstract_evaluator import AbstractEvaluator

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm


class FiLMModelEvaluator(AbstractEvaluator):

    def __init__(self, config_path, model, sciplex_dataset_train, sciplex_dataset_validation, sciplex_dataset_test):
        # load config file
        self.read_config(config_path)

        #load model
        self.model = model(self.config)

        #prepare model
        self.prepare_model()


        #load training, validation and test data in
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

    def read_config(self, config_path):
        with open(config_path, 'r') as file:
            try:
                self.config = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)
                raise RuntimeError(exc)



    def prepare_model(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.config['train_params']['lr'],
                                    weight_decay=self.config['train_params']['weight_decay'])

        self.model = self.model.to(self.device)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                              mode=self.config['train_params']['scheduler_mode'],
                                                              factor=self.config['train_params']['scheduler_factor'],
                                                              patience=self.config['train_params']['scheduler_patience'])

    def train(self, loss_fn):
        self.model.train()

        num_epochs = self.config['train_params']['num_epochs']
        device = self.device

        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")

            for control, drug_emb, target, meta in self.sciplex_loader_train:
                # Move tensors to the specified device
                control = control.to(device)
                drug_emb = drug_emb.to(device)
                target = target.to(device)

                # Zero the gradients
                self.optimizer.zero_grad()

                # Forward pass through the model
                output = self.model(control, drug_emb)

                # Compute the loss
                loss = loss_fn(output, target, control)

                # Backprop
                loss.backward()

                # Update model parameters
                self.optimizer.step()


            ### VALIDATION LOOP ###

            validation_losses = list()
            with torch.no_grad():
                for control, drug_emb, target, meta in self.sciplex_loader_validation:

                    control, drug_emb, target = (
                        control.to(device),
                        drug_emb.to(device),
                        target.to(device)
                    )

                    # Forward pass
                    output_validation = self.model(control, drug_emb)

                    # Compute loss
                    validation_loss = loss_fn(output_validation, target, control)

                    # Track validation loss
                    validation_losses.append(validation_loss.item())

            self.scheduler.step(np.mean(validation_losses))
            print(f"Epoch {epoch + 1}/{num_epochs}", "Validation Loss:", np.mean(validation_losses))

            ### VALIDATION LOOP###

        self.trained_model = self.model


    def test(self):
        control_embeddings = []
        treated_embeddings = []
        model_output = []
        compounds_list = []
        cell_types_list = []
        doses_list = []

        self.trained_model.eval()  # Set the model to evaluation mode

        with torch.no_grad():  # Disable gradient computation
            for control, drug_emb, target, meta in tqdm(self.sciplex_loader_test):
                # Move tensors to the specified device
                control = control.to(self.device)
                drug_emb = drug_emb.to(self.device)
                target = target.to(self.device)

                # Forward pass through the model
                output = self.trained_model(control, drug_emb)

                # Convert tensors to lists of NumPy arrays for DataFrame compatibility
                control_emb_list = [x.cpu().numpy() for x in torch.unbind(control, dim=0)]
                treated_emb_list = [x.cpu().numpy() for x in torch.unbind(target, dim=0)]
                output_list = [x.cpu().numpy() for x in torch.unbind(output, dim=0)]

                # Meta information
                compounds = meta['compound']
                cell_types = meta['cell_type']
                doses = [d.item() for d in meta['dose']]

                # Append results to lists
                control_embeddings.extend(control_emb_list)
                treated_embeddings.extend(treated_emb_list)
                model_output.extend(output_list)
                compounds_list.extend(compounds)
                cell_types_list.extend(cell_types)
                doses_list.extend(doses)

        # Save results into a DataFrame
        self.test_results = pd.DataFrame({
            "ctrl_emb": control_embeddings,
            "pert_emb": treated_embeddings,
            "pred_emb": model_output,
            "compound": compounds_list,
            "cell_type": cell_types_list,
            "dose": doses_list
        })

    def get_test_results(self):
        return self.test_results