import optuna
import torch.optim as optim
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle as pkl
from torch.utils.data.dataloader import DataLoader

from src.models.compound_decoder import DecoderModel
from src.dataset.dataset_unseen_compounds import SciplexDatasetUnseenPerturbations
from src.utils import get_model_stats


cell_type_onehot = {"A549": torch.tensor([1, 0, 0]),
                    "K562": torch.tensor([0, 1, 0]),
                    "MCF7": torch.tensor([0, 0, 1]),}



class DecoderEvaluator():
    def __init__(self, train_dataset, val_dataset, test_dataset, params, add_relu=True):
        self.MODEL_PATIENCE = 10
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_epochs = 100

        self.model = DecoderModel(params['input_dim'], params['drug_dim'], params['output_dim'], params['hidden_dims'],
                              params['dropout'], add_relu=add_relu)
        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                         mode=params['scheduler_mode'],
                                                         factor=params['scheduler_factor'],
                                                         patience=params['scheduler_patience'])


        self.train_loader = self.create_dataloader(train_dataset,  params['batch_size'])

        if val_dataset is not None:
            self.val_loader = self.create_dataloader(val_dataset,  params['batch_size'])

        if test_dataset is not None:
            self.test_loader = self.create_dataloader(test_dataset,  params['batch_size'])


    @staticmethod
    def create_dataloader(dataset, batch_size):
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)


    def train_with_validation(self, loss_fn, trial):
        self.model.train()
        best_model_weights = self.model.state_dict()
        best_loss = float('inf')
        best_epoch = -1
        epochs_without_improvement = 0

        for epoch in range(self.max_epochs):
            self.model.train()
            for _, drug_emb, target, meta in self.train_loader:

                cell_type_encoding = torch.stack([cell_type_onehot[x] for x in meta['cell_type']])

                cell_type_encoding, drug_emb, target = cell_type_encoding.to(self.device), drug_emb.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                loss = loss_fn(self.model(cell_type_encoding, drug_emb), target, cell_type_encoding)
                loss.backward()
                self.optimizer.step()

            validation_loss = self.validate(loss_fn)
            print("Epoch:\t", epoch, "Val Loss:\t", validation_loss)
            self.scheduler.step(validation_loss)
            trial.report(validation_loss, epoch)

            if validation_loss < best_loss:
                best_loss = validation_loss
                best_model_weights = self.model.state_dict()
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                trial.set_user_attr('best_epoch', best_epoch)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.MODEL_PATIENCE:
                    self.model.load_state_dict(best_model_weights)
                    return best_loss

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        self.model.load_state_dict(best_model_weights)

        trial.set_user_attr('best_epoch', best_epoch)
        return best_loss

    def train(self, loss_fn, num_epochs=None):
        self.model.train()

        for epoch in range(num_epochs):
            for _, drug_emb, target, meta in self.train_loader:
                cell_type_encoding = torch.stack([cell_type_onehot[x] for x in meta['cell_type']])
                cell_type_encoding, drug_emb, target = cell_type_encoding.to(self.device), drug_emb.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                pred = self.model(cell_type_encoding, drug_emb)
                loss = loss_fn(pred, target, cell_type_encoding)
                loss.backward()
                self.optimizer.step()

    def validate(self, loss_fn):
        self.model.eval()
        validation_losses = []

        with torch.no_grad():
            for _, drug_emb, target, meta in self.val_loader:
                cell_type_encoding = torch.stack([cell_type_onehot[x] for x in meta['cell_type']])

                cell_type_encoding, drug_emb, target = cell_type_encoding.to(self.device), drug_emb.to(self.device), target.to(self.device)
                validation_loss = loss_fn(self.model(cell_type_encoding, drug_emb), target, cell_type_encoding)
                validation_losses.append(validation_loss.item())

        avg_loss = np.mean(validation_losses)
        return avg_loss

    def test(self):
        self.model.eval()
        results = {'ctrl_emb': [], 'pert_emb': [], 'pred_emb': [], 'compound': [], 'cell_type': [], 'dose': []}

        with torch.no_grad():
            for control, drug_emb, target, meta in tqdm(self.test_loader):
                cell_type_encoding = torch.stack([cell_type_onehot[x] for x in meta['cell_type']])

                control, drug_emb, target = control.to(self.device), drug_emb.to(self.device), target.to(self.device)
                cell_type_encoding = cell_type_encoding.to(self.device)
                output = self.model(cell_type_encoding, drug_emb)

                results['ctrl_emb'].extend(control.cpu().numpy())
                results['pert_emb'].extend(target.cpu().numpy())
                results['pred_emb'].extend(output.cpu().numpy())
                results['compound'].extend(meta['compound'])
                results['cell_type'].extend(meta['cell_type'])
                results['dose'].extend([d.item() for d in meta['dose']])

        return pd.DataFrame(results)


def objective(trial, dataset_train=None, dataset_validation=None,
              input_dim=0, output_dim=0, drug_dim=0, scheduler_mode='min', loss_fn=None, add_relu=True):

    lr = trial.suggest_categorical('lr', [1e-6, 1e-5, 1e-4, 1e-3])
    weight_decay = trial.suggest_categorical('weight_decay', [1e-6, 1e-5, 1e-4, 1e-3])
    scheduler_factor = trial.suggest_categorical('scheduler_factor', [0.1, 0.3, 0.5, 0.8])
    scheduler_patience = trial.suggest_categorical('scheduler_patience', [1, 5, 10, 20])
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256])
    dropout = trial.suggest_categorical('dropout', [0.05, 0.1, 0.15, 0.2])
    hidden_dims = trial.suggest_categorical('hidden_dims', [64, 128, 256, 512, 1024])


    params = {
        'input_dim': input_dim,
        'output_dim' : output_dim,
        'drug_dim' : drug_dim,
        'dropout' : dropout,
        'scheduler_mode': scheduler_mode,
        'lr': lr,
        'weight_decay': weight_decay,
        'scheduler_factor': scheduler_factor,
        'scheduler_patience': scheduler_patience,
        'batch_size': batch_size,
        'hidden_dims' : (hidden_dims,),
    }
    ev = DecoderEvaluator(dataset_train, dataset_validation, None, params, add_relu=add_relu)

    return ev.train_with_validation(loss_fn, trial)

def get_models_results(drug_splits=None, loss_function=None, adata=None, input_dim=None,
                            output_dim=None, drug_rep_name=None, drug_emb_size=None, n_trials=None,
                            scheduler_mode=None, run_name=None, save_path=None, add_relu=True):

    print("Loading Datasets ...")

    drugs_train = drug_splits['train']
    drugs_validation = drug_splits['valid']
    drugs_test = drug_splits['test']

    #Optimize Hyperparamteres
    dataset_train = SciplexDatasetUnseenPerturbations(adata, drugs_train, drug_rep_name, drug_emb_size,)
    dataset_validation = SciplexDatasetUnseenPerturbations(adata, drugs_validation, drug_rep_name, drug_emb_size,)

    print("Optimizing Hyperparameters with Optuna ...")

    study = optuna.create_study(direction='minimize', study_name=run_name, storage="sqlite:///optuna_study.db", load_if_exists=True)
    study.optimize(lambda trial: objective(trial,
                                           dataset_train=dataset_train, dataset_validation=dataset_validation,
                                           input_dim=input_dim, output_dim=output_dim,
                                           drug_dim=drug_emb_size, loss_fn=loss_function, add_relu=add_relu), n_trials=n_trials)
    best_trial = study.best_trial
    optimal_params = best_trial.params
    best_epoch = best_trial.user_attrs["best_epoch"]

    del dataset_train
    del dataset_validation

    print("Training model with best parameters on train+validation ...")

    #Retrain the model on validation + train set with the best parameters
    drugs_train_final = list(drugs_train) + list(drugs_validation)

    dataset_train_final = SciplexDatasetUnseenPerturbations(adata, drugs_train_final, drug_rep_name, drug_emb_size)
    dataset_test = SciplexDatasetUnseenPerturbations(adata, drugs_test, drug_rep_name, drug_emb_size)

    optimal_params['input_dim'] = input_dim
    optimal_params['output_dim'] = output_dim
    optimal_params['drug_dim'] = drug_emb_size
    optimal_params['scheduler_mode'] = scheduler_mode
    optimal_params['hidden_dims'] = (optimal_params['hidden_dims'],)

    final_ev = DecoderEvaluator(dataset_train_final, None, dataset_test, optimal_params, add_relu=add_relu)
    final_ev.train(loss_function, num_epochs=best_epoch)

    print("Getting test set predictions and saving results ...")

    #Get model performance metrics
    predictions = final_ev.test()

    with open(save_path, 'wb') as f:
        pkl.dump(predictions, f)