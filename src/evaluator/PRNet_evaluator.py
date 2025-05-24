import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.distributions import normal
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import optuna
import pickle as pkl

import math
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from src.dataset.dataset_unseen_compounds import SciplexDatasetUnseenPerturbations
from src.models.PRNet import PRnet
from src.utils import get_model_stats



class PRnetEvaluator:
    def __init__(self, train_dataset=None, valid_dataset=None, test_dataset=None, params=None, add_relu=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        self.model = PRnet(gene_vector_dim=params['input_dim'], hidden_layer_sizes=params['hidden_dims'],
                           agg_latent_dim=params['agg_latent_dim'], adaptor_layer_sizes=params['hidden_dims_adapter'],
                           drug_latent_dim=params['drug_latent_dim'], comb_num=params['comb_num'], drug_dimension=params['drug_dimension'],
                           dr_rate=params['dropout'], add_relu=add_relu)


        self.modelPGM = self.model.get_PGM()
        self.modelPGM = self.modelPGM.to(self.device)

        self.modelPGM.apply(self.__weight_init)
        print(self.modelPGM)

        self.max_epochs = params['max_epochs']
        self.MODEL_PATIENCE = params['model_patience']

        self.optimizer = optim.Adam(self.modelPGM.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                              mode=params['scheduler_mode'],
                                                              factor=params['scheduler_factor'],
                                                              patience=params['scheduler_patience'])

        self.train_loader = self.create_dataloader(train_dataset,  params['batch_size'])

        if valid_dataset is not None:
            self.val_loader = self.create_dataloader(valid_dataset, params['batch_size'])

        if test_dataset is not None:
            self.test_loader = self.create_dataloader(test_dataset, params['batch_size'])

        self.criterion = nn.GaussianNLLLoss()



    @staticmethod
    def create_dataloader(dataset, batch_size):
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    def train_with_validation(self, loss, trial):
        best_model_weights = self.modelPGM.state_dict()
        best_loss = float('inf')
        best_epoch = -1
        epochs_without_improvement = 0

        for epoch in range(self.max_epochs):
            self.modelPGM.train()
            for control, drug_emb, target, _ in self.train_loader:
                control, drug_emb, target = control.to(self.device), drug_emb.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()

                b_size = control.size(0)
                noise = self.__make_noise(b_size, 10)

                gene_reconstructions = self.modelPGM(control, drug_emb, noise)
                dim = gene_reconstructions.size(1) // 2

                gene_means = gene_reconstructions[:, :dim]
                gene_vars = gene_reconstructions[:, dim:]
                gene_vars = F.softplus(gene_vars)

                reconstruction_loss = self.criterion(input=gene_means, target=target, var=gene_vars)
                reconstruction_loss.backward()

                # Update PGM
                self.optimizer.step()

            validation_loss = self.validate()
            print("Epoch:", epoch, "\t Validation Loss:", validation_loss)
            self.scheduler.step(validation_loss)
            trial.report(validation_loss, epoch)

            if validation_loss < best_loss:
                best_loss = validation_loss
                best_model_weights = self.modelPGM.state_dict()
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                trial.set_user_attr('best_epoch', best_epoch)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.MODEL_PATIENCE:
                    self.modelPGM.load_state_dict(best_model_weights)
                    return best_loss

        self.modelPGM.load_state_dict(best_model_weights)

        trial.set_user_attr('best_epoch', best_epoch)
        return best_loss

    def train(self, num_epochs=None):
        self.modelPGM.train()
        for epoch in range(num_epochs):
            for control, drug_emb, target, _ in self.train_loader:

                control, drug_emb, target = control.to(self.device), drug_emb.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()

                b_size = control.size(0)
                noise = self.__make_noise(b_size, 10)

                gene_reconstructions = self.modelPGM(control, drug_emb, noise)
                dim = gene_reconstructions.size(1) // 2

                gene_means = gene_reconstructions[:, :dim]
                gene_vars = gene_reconstructions[:, dim:]
                gene_vars = F.softplus(gene_vars)

                reconstruction_loss = self.criterion(input=gene_means, target=target, var=gene_vars)
                reconstruction_loss.backward()

                # Update PGM
                self.optimizer.step()


    def validate(self):
        self.modelPGM.eval()
        validation_losses = []

        with torch.no_grad():
            for control, drug_emb, target, _ in self.val_loader:
                control, drug_emb, target = control.to(self.device), drug_emb.to(self.device), target.to(self.device)

                b_size = control.size(0)
                noise = self.__make_noise(b_size, 10)
                gene_reconstructions = self.modelPGM(control, drug_emb, noise)
                dim = gene_reconstructions.size(1) // 2
                gene_means = gene_reconstructions[:, :dim]
                gene_vars = gene_reconstructions[:, dim:]
                gene_vars = F.softplus(gene_vars)


                dist = normal.Normal(
                    torch.clamp(
                        torch.Tensor(gene_means),
                        min=1e-3,
                        max=1e3,
                    ),
                    torch.clamp(
                        torch.Tensor(gene_vars.sqrt()),
                        min=1e-3,
                        max=1e3,
                    )
                )

                nb_sample = dist.sample().cpu()
                y_true = target.cpu()

                mse_score = mean_squared_error(y_true, nb_sample)
                validation_losses.append(mse_score)

        avg_loss = np.mean(validation_losses)
        return avg_loss

    def test(self):
        self.modelPGM.eval()
        results = {'ctrl_emb': [], 'pert_emb': [], 'pred_emb': [], 'compound': [], 'cell_type': [], 'dose': []}

        with torch.no_grad():

            for control, drug_emb, target, meta in tqdm(self.test_loader):
                control, drug_emb, target = control.to(self.device), drug_emb.to(self.device), target.to(self.device)

                b_size = control.size(0)

                noise = self.__make_noise(b_size, 10)
                gene_reconstructions = self.modelPGM(control, drug_emb, noise).detach()
                dim = gene_reconstructions.size(1) // 2
                gene_means = gene_reconstructions[:, :dim]
                gene_vars = gene_reconstructions[:, dim:]
                gene_vars = F.softplus(gene_vars)

                #GUSS
                dist = normal.Normal(
                    torch.clamp(
                        torch.Tensor(gene_means),
                        min=1e-3,
                        max=1e3,
                    ),
                    torch.clamp(
                        torch.Tensor(gene_vars.sqrt()),
                        min=1e-3,
                        max=1e3,
                    )
                )


                output = dist.sample()

                results['ctrl_emb'].extend(control.cpu().numpy())
                results['pert_emb'].extend(target.cpu().numpy())
                results['pred_emb'].extend(output.cpu().numpy())
                results['compound'].extend(meta['compound'])
                results['cell_type'].extend(meta['cell_type'])
                results['dose'].extend([d.item() for d in meta['dose']])

        return pd.DataFrame(results)


    def __make_noise(self, batch_size, shape, volatile=False):
        noise = torch.randn(batch_size, shape, device=self.device, dtype=torch.float32)
        return noise

    def __weight_init(self, m):
        # initialize the weights of the model
        if isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.normal_(1, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()


def objective(trial, dataset_train=None, dataset_validation=None,
              input_dim=0, output_dim=0, drug_dim=0, scheduler_mode='min', loss=None, add_relu=True):

    lr = trial.suggest_categorical('lr', [1e-6, 1e-5, 1e-4, 1e-3])
    weight_decay = trial.suggest_categorical('weight_decay', [1e-6, 1e-5, 1e-4, 1e-3])
    scheduler_factor = trial.suggest_categorical('scheduler_factor', [0.1, 0.3, 0.5, 0.8])
    scheduler_patience = trial.suggest_categorical('scheduler_patience', [1, 5, 10, 20])
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256, 512])
    dropout = trial.suggest_categorical('dropout', [0.05, 0.1, 0.15, 0.2])



    params = {
        'input_dim': input_dim,
        'output_dim' : output_dim,
        'drug_dimension' : drug_dim,
        'dropout' : dropout,
        'scheduler_mode': scheduler_mode,
        'lr': lr,
        'weight_decay': weight_decay,
        'scheduler_factor': scheduler_factor,
        'scheduler_patience': scheduler_patience,
        'batch_size': batch_size,
        'hidden_dims' : [128],
        'hidden_dims_adapter': [128],
        'drug_latent_dim': 64,
        'agg_latent_dim': 64,
        'comb_num': 1,
        'model_patience': 20,
        'max_epochs': 100,
        'loss': loss

    }
    ev = PRnetEvaluator(dataset_train, dataset_validation, None, params, add_relu=add_relu)

    return ev.train_with_validation(loss, trial)

def get_models_results(drug_splits=None, loss=None, adata=None, input_dim=None,
                            output_dim=None, drug_rep_name=None, drug_emb_size=None, n_trials=None,
                            scheduler_mode=None, run_name=None, save_path=None, add_relu=True):

    print("Loading Datasets ...")

    drugs_train = drug_splits['train']
    drugs_validation = drug_splits['valid']
    drugs_test = drug_splits['test']

    dataset_train = SciplexDatasetUnseenPerturbations(adata, drugs_train, drug_rep_name, drug_emb_size)
    dataset_validation = SciplexDatasetUnseenPerturbations(adata, drugs_validation, drug_rep_name, drug_emb_size)

    print("Optimizing Hyperparameters with Optuna ...")

    study = optuna.create_study(direction='minimize', study_name=run_name, storage="sqlite:///optuna_study.db", load_if_exists=True)
    study.optimize(lambda trial: objective(trial,
                                           dataset_train=dataset_train, dataset_validation=dataset_validation,
                                           input_dim=input_dim, output_dim=output_dim,
                                           drug_dim=drug_emb_size, loss=loss), n_trials=n_trials, add_relu=add_relu)
    best_trial = study.best_trial
    optimal_params = best_trial.params
    best_epoch = best_trial.user_attrs["best_epoch"]

    print("Training model with best parameters on train+validation ...")

    del dataset_train
    del dataset_validation

    #Retrain the model on validation + train set with the best parameters
    drugs_train_final = list(drugs_train) + list(drugs_validation)

    dataset_train_final = SciplexDatasetUnseenPerturbations(adata, drugs_train_final, drug_rep_name, drug_emb_size)
    dataset_test = SciplexDatasetUnseenPerturbations(adata, drugs_test, drug_rep_name, drug_emb_size)

    optimal_params['input_dim'] = input_dim
    optimal_params['output_dim'] = output_dim
    optimal_params['scheduler_mode'] = scheduler_mode
    optimal_params['drug_dimension'] = drug_emb_size

    optimal_params['batch_size']=  512
    optimal_params['hidden_dims']= [128]
    optimal_params['hidden_dims_adapter'] = [128]
    optimal_params['drug_latent_dim'] = 64
    optimal_params['agg_latent_dim'] = 64
    optimal_params['comb_num'] = 1
    optimal_params['model_patience'] = 20
    optimal_params['max_epochs'] = 100

    final_ev = PRnetEvaluator(dataset_train_final, None, dataset_test, optimal_params, add_relu=add_relu)
    final_ev.train(num_epochs=best_epoch)

    print("Getting test set predictions and saving results ...")

    predictions = final_ev.test()

    with open(save_path, 'wb') as f:
        pkl.dump(predictions, f)


