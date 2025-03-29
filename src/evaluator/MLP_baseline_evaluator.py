import yaml
import optuna
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from evaluator.abstract_evaluator import AbstractEvaluator

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm


class MLPBaselineEvaluator(AbstractEvaluator):
    def __init__(self, config_path, model, train_dataset, val_dataset, test_dataset, trial):
        self.config = self.read_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize model, optimizer, and scheduler
        self.model = model(self.config).to(self.device)
        self.test_model = model(self.config).to(self.device)

        self.optimizer, self.scheduler, self.batch_size, self.model_patience = self.prepare_model(trial)

        # Load data
        self.train_loader = self.create_dataloader(train_dataset,  self.batch_size)
        self.val_loader = self.create_dataloader(val_dataset,  self.batch_size)
        self.test_loader = self.create_dataloader(test_dataset,  self.batch_size)

    @staticmethod
    def read_config(config_path):
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    @staticmethod
    def create_dataloader(dataset, batch_size):
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    def prepare_model(self, trial):
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
        weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
        scheduler_factor = trial.suggest_loguniform('scheduler_factor', 0.1, 0.5)
        scheduler_patience = trial.suggest_int('scheduler_patience', 1, 20)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256])

        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode=self.config['train_params']['scheduler_mode'],
                                                         factor=scheduler_factor,
                                                         patience=scheduler_patience)

        model_patience = self.config['train_params']['model_patience']
        return optimizer, scheduler, batch_size, model_patience

    def train(self, loss_fn, trial):
        self.model.train()
        best_model_weights = self.model.state_dict()
        best_loss = float('inf')
        epochs_without_improvement = 0

        for epoch in range(self.config['train_params']['num_epochs']):
            for control, drug_emb, target, _ in self.train_loader:
                control, drug_emb, target = control.to(self.device), drug_emb.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                loss = loss_fn(self.model(control, drug_emb), target, control)
                loss.backward()
                self.optimizer.step()

            validation_loss = self.validate(loss_fn)
            trial.report(validation_loss, epoch)

            if validation_loss < best_loss:
                best_loss = validation_loss
                best_model_weights = self.model.state_dict()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.model_patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    self.model.load_state_dict(best_model_weights)
                    return best_loss

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        self.model.load_state_dict(best_model_weights)
        return best_loss

    def validate(self, loss_fn):
        self.model.eval()
        validation_losses = []

        with torch.no_grad():
            for control, drug_emb, target, _ in self.val_loader:
                control, drug_emb, target = control.to(self.device), drug_emb.to(self.device), target.to(self.device)
                validation_loss = loss_fn(self.model(control, drug_emb), target, control)
                validation_losses.append(validation_loss.item())

        avg_loss = np.mean(validation_losses)
        self.scheduler.step(avg_loss)
        return avg_loss

    def test(self):
        self.model.eval()
        results = {'ctrl_emb': [], 'pert_emb': [], 'pred_emb': [], 'compound': [], 'cell_type': [], 'dose': []}

        with torch.no_grad():
            for control, drug_emb, target, meta in tqdm(self.test_loader):
                control, drug_emb, target = control.to(self.device), drug_emb.to(self.device), target.to(self.device)
                output = self.model(control, drug_emb)

                results['ctrl_emb'].extend(control.cpu().numpy())
                results['pert_emb'].extend(target.cpu().numpy())
                results['pred_emb'].extend(output.cpu().numpy())
                results['compound'].extend(meta['compound'])
                results['cell_type'].extend(meta['cell_type'])
                results['dose'].extend([d.item() for d in meta['dose']])

        return pd.DataFrame(results)