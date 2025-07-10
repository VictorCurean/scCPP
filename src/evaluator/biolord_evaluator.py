
import numpy as np
import pandas as pd
import torch
import anndata as ad

from tqdm import tqdm
import ast
import optuna
import pickle as pkl

import biolord

class BiolordEvaluator:

    def __init__(self, adata):
        self.adata = adata.copy()

        biolord.Biolord.setup_anndata(
            self.adata,
            ordered_attributes_keys=["fmfp_dose"],
            categorical_attributes_keys=["cell_type"],
            retrieval_attribute_key=None,
        )


    def train_with_validation(self, trial, module_params, trainer_params, extra_params):
        self.model = biolord.Biolord(
            adata=self.adata,
            n_latent=256,
            model_name="sciplex3_hyperparam_optimization",
            module_params=module_params,
            train_classifiers=False,
            split_key="split_random",
            train_split='train',
            valid_split='valid',
            test_split='test'
        )

        self.model.train(
            max_epochs=extra_params['max_epochs'],
            batch_size=extra_params['batch_size'],
            plan_kwargs=trainer_params,
            early_stopping=True,
            early_stopping_patience=extra_params['early_stopping_patience'],
            check_val_every_n_epoch=5,
            num_workers=0,
            enable_checkpointing=False
        )

        # Get epoch history
        history = self.model.training_plan.epoch_history

        # Filter for validation epochs
        val_biolord_metrics = [
            metric for mode, metric in zip(history["mode"], history["biolord_metric"]) if mode == "valid"
        ]

        best_val_metric = max(val_biolord_metrics)

        return best_val_metric


    def train(self, module_params_best, trainer_params_best, extra_params_best):
        self.model_final = biolord.Biolord(
            adata=self.adata,
            n_latent=256,
            model_name="sciplex3_final",
            module_params=module_params_best,
            train_classifiers=False,
            split_key="split_random",
            train_split='train',
            valid_split='valid',
            test_split='test'
        )

        self.model_final.train(
            max_epochs=extra_params_best['max_epochs'],
            batch_size=extra_params_best['batch_size'],
            plan_kwargs=trainer_params_best,
            early_stopping=True,
            early_stopping_patience=extra_params_best['early_stopping_patience'],
            check_val_every_n_epoch=5,
            num_workers=0,
            enable_checkpointing=False
        )


    def test(self):
        idx_test_control = np.where((self.adata.obs["product_name"] == "Vehicle"))[0]

        adata_test_control = self.adata[idx_test_control].copy()
        idx_ood = np.where((self.adata.obs["split_random"] == "test"))[0]

        adata_ood = self.adata[idx_ood].copy()
        dataset_control = self.model_final.get_dataset(adata_test_control)
        dataset_ood = self.model_final.get_dataset(adata_ood)

        results = self.__compute_prediction(model=self.model_final,
                                            adata=adata_ood,
                                            dataset=dataset_ood,
                                            dataset_control=dataset_control)

        return pd.DataFrame(results)


    def __bool2idx(self, x):
        """
        Returns the indices of the True-valued entries in a boolean array `x`
        """
        return np.where(x)[0]

    def __repeat_n(self, x, n):
        """
        Returns an n-times repeated version of the Tensor x,
        repetition dimension is axis 0
        """
        # copy tensor to device BEFORE replicating it n times
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return x.to(device).view(1, -1).repeat(n, 1)


    def __compute_prediction(
            self,
            model,
            adata,
            dataset,
            cell_lines=None,
            dataset_control=None
    ):
        pert_categories_index = pd.Index(adata.obs["drug_celltype_dose"].values, dtype="category")


        cl_dict = {
            torch.Tensor([0.]): "A549",
            torch.Tensor([1.]): "K562",
            torch.Tensor([2.]): "MCF7",
        }

        if cell_lines is None:
            cell_lines = ["A549", "K562", "MCF7"]

        print(cell_lines)
        layer = "X" if "X" in dataset else "layers"


        results = {'pert_emb': [], 'pred_emb': [], 'compound': [], 'cell_type': [], 'dose': []}

        for cell_drug_dose_comb, category_count in tqdm(
                zip(*np.unique(pert_categories_index.values, return_counts=True))
        ):

            if category_count <= 5:
                continue

            if "vehicle" in cell_drug_dose_comb.lower():
                continue


            bool_category = pert_categories_index.get_loc(cell_drug_dose_comb)
            idx_all = self.__bool2idx(bool_category)
            idx = idx_all[0]
            y_true = dataset[layer][idx_all, :].to(model.device)

            dataset_comb = {}
            if dataset_control is None:
                n_obs = y_true.size(0).to(model.device)
                for key, val in dataset.items():
                    dataset_comb[key] = val[idx_all].to(model.device)
            else:
                n_obs = dataset_control[layer].size(0)
                dataset_comb[layer] = dataset_control[layer].to(model.device)
                dataset_comb["ind_x"] = dataset_control["ind_x"].to(model.device)
                for key in dataset_control:
                    if key not in [layer, "ind_x"]:
                        dataset_comb[key] = self.__repeat_n(dataset[key][idx, :], n_obs)

            stop = False
            for tensor, cl in cl_dict.items():
                if (tensor == dataset["cell_type"][idx]).all():
                    if cl not in cell_lines:
                        stop = True
            if stop:
                continue

            y_pred, _ = model.module.get_expression(dataset_comb)

            cell_type = cell_drug_dose_comb.split("_")[0]
            drug = cell_drug_dose_comb.split("_")[1]
            dose = cell_drug_dose_comb.split("_")[2]

            results['pert_emb'].extend(y_true.cpu().numpy())
            results['pred_emb'].extend(y_pred.cpu().numpy())
            results['compound'].extend([drug] * y_true.shape[0])
            results['cell_type'].extend([cell_type * y_true.shape[0]])
            results['dose'].extend([dose] * y_true.shape[0])

        return results


def __preprocess_adata_biolord(adata_path, drugs_train, drugs_valid, drugs_test, params):

    adata = ad.read_h5ad(adata_path)

    # setup chemical representation of molecules
    drug_emb_matrix = list()

    for idx in tqdm(list(adata.obs_names)):
        cell_meta = adata.obs.loc[idx]

        if cell_meta['product_name'] == 'Vehicle':
            drug_emb = np.zeros(params.drug_emb_size)
            drug_emb_matrix.append(drug_emb)

        else:
            # get drug embedding
            drug_emb = ast.literal_eval(cell_meta[params.drug_emb_column])

            # get dose
            dose = float(cell_meta['dose'])

            # multiply drug emb with does with the dose
            drug_emb = np.array(drug_emb) * np.log10(1 + dose)
            drug_emb_matrix.append(drug_emb)

    drug_emb_matrix = np.array(drug_emb_matrix)

    assert drug_emb_matrix.shape[0] == adata.n_obs

    adata.obsm['fmfp_dose'] = drug_emb_matrix

    # setup train-valid-split
    split_optimization = list()

    for idx in tqdm(list(adata.obs_names)):
        cell_meta = adata.obs.loc[idx]

        if cell_meta['product_name'] == 'Vehicle':
            split_optimization.append('train')

        elif cell_meta['product_name'] in drugs_train:
            split_optimization.append('train')

        elif  cell_meta['product_name'] in drugs_valid:
            split_optimization.append('valid')

        elif cell_meta['product_name'] in drugs_test:
            split_optimization.append('test')

    adata.obs['split_random'] = split_optimization

    return adata


def objective(trial, adata=None, gene_likelihood='normal'):

    #fixed params
    decoder_width = 4096
    decoder_depth = 4
    attribute_nn_width = 2048
    attribute_nn_depth = 2
    n_latent_attribute_ordered = 256
    n_latent_attribute_categorical = 3
    train_classifiers = False
    cosine_scheduler = True
    use_batch_norm = False
    use_layer_norm = False

    #optimizable params
    lr = trial.suggest_categorical('lr', [1e-6, 1e-5, 1e-4, 1e-3])
    attribute_nn_lr = trial.suggest_categorical('attribute_nn_lr', [1e-5, 1e-4, 1e-3, 1e-2])
    weight_decay = trial.suggest_categorical('weight_decay', [1e-6, 1e-5, 1e-4, 1e-3])
    dropout = trial.suggest_categorical('dropout', [0.05, 0.1, 0.15, 0.2])
    reconstruction_penalty = trial.suggest_categorical('reconstruction_penalty', [1e+1, 1e+2, 1e+3, 1e+4])
    attribute_nn_wd = trial.suggest_categorical('attribute_nn_wd', [4e-8, 1e-8, 1e-7])
    unknown_attribute_noise_param = trial.suggest_categorical('unknown_attribute_noise_param', [2, 5, 10, 20])
    unknown_attribute_penalty = trial.suggest_categorical('unknown_attribute_penalty', [0.1, 0.5, 1, 2, 5])
    scheduler_final_lr = trial.suggest_categorical('scheduler_final_lr', [1e-4, 1e-5, 1e-6])
    step_size_lr = trial.suggest_categorical('step_size_lr', [45, 90, 180])


    module_params = {
        "decoder_width": decoder_width,
        "decoder_depth": decoder_depth,
        "attribute_nn_width": attribute_nn_width,
        "attribute_nn_depth": attribute_nn_depth,
        "use_batch_norm": use_batch_norm,
        "use_layer_norm": use_layer_norm,
        "unknown_attribute_noise_param": unknown_attribute_noise_param,
        "seed": 42,
        "n_latent_attribute_ordered": n_latent_attribute_ordered,
        "n_latent_attribute_categorical": n_latent_attribute_categorical,
        "gene_likelihood": gene_likelihood,
        "reconstruction_penalty": reconstruction_penalty,
        "unknown_attribute_penalty": unknown_attribute_penalty,
        "attribute_dropout_rate": dropout,
    }

    trainer_params = {
        "n_epochs_warmup": 0,
        "latent_lr": lr,
        "latent_wd": weight_decay,
        "decoder_lr": lr,
        "decoder_wd": weight_decay,
        "attribute_nn_lr": attribute_nn_lr,
        "attribute_nn_wd": attribute_nn_wd,
        "step_size_lr": step_size_lr,
        "cosine_scheduler": cosine_scheduler,
        "scheduler_final_lr": scheduler_final_lr
    }

    extra_params = {
        'max_epochs' : 200,
        'batch_size' : 512,
        'early_stopping_patience': 20
    }

    ev = BiolordEvaluator(adata)

    return ev.train_with_validation(trial, module_params, trainer_params, extra_params)

def get_models_results(adata=None, n_trials=None, run_name=None, save_path=None, gene_likelihood='normal'):


    print("Optimizing Hyperparameters with Optuna ...")

    study = optuna.create_study(direction='minimize', study_name=run_name, storage="sqlite:///optuna_study.db", load_if_exists=True)
    study.optimize(lambda trial: objective(trial, adata=adata, gene_likelihood=gene_likelihood), n_trials=n_trials)

    best_trial = study.best_trial
    optimal_params = best_trial.params

    # fixed params
    decoder_width = 4096
    decoder_depth = 4
    attribute_nn_width = 2048
    attribute_nn_depth = 2
    n_latent_attribute_ordered = 256
    n_latent_attribute_categorical = 3
    train_classifiers = False
    cosine_scheduler = True
    use_batch_norm = False
    use_layer_norm = False

    #optuna dervied optimally params
    lr = optimal_params['lr']
    attribute_nn_lr = optimal_params['attribute_nn_lr']
    weight_decay = optimal_params['weight_decay']
    dropout = optimal_params['dropout']
    reconstruction_penalty = optimal_params['reconstruction_penalty']
    attribute_nn_wd = optimal_params['attribute_nn_wd']
    unknown_attribute_noise_param = optimal_params['unknown_attribute_noise_param']
    unknown_attribute_penalty = optimal_params['unknown_attribute_penalty']
    scheduler_final_lr = optimal_params['scheduler_final_lr']
    step_size_lr = optimal_params['step_size_lr']

    module_params = {
        "decoder_width": decoder_width,
        "decoder_depth": decoder_depth,
        "attribute_nn_width": attribute_nn_width,
        "attribute_nn_depth": attribute_nn_depth,
        "use_batch_norm": use_batch_norm,
        "use_layer_norm": use_layer_norm,
        "unknown_attribute_noise_param": unknown_attribute_noise_param,
        "seed": 42,
        "n_latent_attribute_ordered": n_latent_attribute_ordered,
        "n_latent_attribute_categorical": n_latent_attribute_categorical,
        "gene_likelihood": gene_likelihood,
        "reconstruction_penalty": reconstruction_penalty,
        "unknown_attribute_penalty": unknown_attribute_penalty,
        "attribute_dropout_rate": dropout,
    }

    trainer_params = {
        "n_epochs_warmup": 0,
        "latent_lr": lr,
        "latent_wd": weight_decay,
        "decoder_lr": lr,
        "decoder_wd": weight_decay,
        "attribute_nn_lr": attribute_nn_lr,
        "attribute_nn_wd": attribute_nn_wd,
        "step_size_lr": step_size_lr,
        "cosine_scheduler": cosine_scheduler,
        "scheduler_final_lr": scheduler_final_lr
    }

    extra_params = {
        'max_epochs': 200,
        'batch_size': 512,
        'early_stopping_patience': 20
    }




    final_ev = BiolordEvaluator(adata)
    final_ev.train()

    print("Getting test set predictions and saving results ...")

    #Get model performance metrics
    final_ev.train(module_params, trainer_params, extra_params)
    predictions = final_ev.test()

    with open(save_path, 'wb') as f:
        pkl.dump(predictions, f)











