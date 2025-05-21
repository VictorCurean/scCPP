import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from torch.utils.data.dataloader import DataLoader
from src.dataset.dataset_unseen_compounds import SciplexDatasetUnseenPerturbations
from src.utils import get_model_stats


class MeanEvaluator:

    def __init__(self,train_dataset, test_dataset):
        self.mean_predictions = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.train_loader = self.create_dataloader(train_dataset, 16)
        self.test_loader = self.create_dataloader(test_dataset, 16)

    @staticmethod
    def create_dataloader(dataset, batch_size):
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    def train_with_validation(self):
        pass

    def train(self):
        results = {'ctrl_emb': [], 'pert_emb': [], 'compound': [], 'cell_type': [], 'dose': []}
        for control, drug_emb, target, meta in tqdm(self.train_loader):
            control, drug_emb, target = control.to(self.device), drug_emb.to(self.device), target.to(self.device)

            results['ctrl_emb'].extend(control.cpu().numpy())
            results['pert_emb'].extend(target.cpu().numpy())
            results['compound'].extend(meta['compound'])
            results['cell_type'].extend(meta['cell_type'])
            results['dose'].extend([d.item() for d in meta['dose']])

        train_data = pd.DataFrame(results)
        mean_predictions = dict()
        for cell_type in train_data['cell_type'].unique():
            df_subset = train_data[train_data['cell_type'] == cell_type]
            mean_predictions[cell_type] = np.mean(np.array(df_subset['pert_emb'].tolist()), axis=0)

        self.mean_predictions = mean_predictions


    def test(self):
        results = {'ctrl_emb': [], 'pert_emb': [], 'pred_emb': [], 'compound': [], 'cell_type': [], 'dose': []}


        for control, drug_emb, target, meta in tqdm(self.test_loader):
            control, drug_emb, target = control.to(self.device), drug_emb.to(self.device), target.to(self.device)


            results['ctrl_emb'].extend(control.cpu().numpy())
            results['pert_emb'].extend(target.cpu().numpy())

            mean_preds = list()
            for cell_type in meta['cell_type']:
                mean_preds.append(self.mean_predictions[cell_type])

            results['pred_emb'].extend(mean_preds)
            results['compound'].extend(meta['compound'])
            results['cell_type'].extend(meta['cell_type'])
            results['dose'].extend([d.item() for d in meta['dose']])

        return pd.DataFrame(results)

    @staticmethod
    def objective():
        pass

    @staticmethod
    def get_models_results(drug_splits=None, adata=None,
                                drug_rep_name=None, drug_emb_size=None, save_path=None):

        print("Loading Datasets ...")

        drugs_train = drug_splits['train']
        drugs_validation = drug_splits['valid']
        drugs_test = drug_splits['test']

        drugs_train_final = list(drugs_train) + list(drugs_validation)
        dataset_train_final = SciplexDatasetUnseenPerturbations(adata, drugs_train_final, drug_rep_name,
                                                                drug_emb_size)
        dataset_test = SciplexDatasetUnseenPerturbations(adata, drugs_test, drug_rep_name, drug_emb_size)



        final_ev = MeanEvaluator(dataset_train_final, dataset_test)
        final_ev.train()

        print("Getting test set predictions and saving results ...")

        predictions = final_ev.test()

        with open(save_path, 'wb') as f:
            pkl.dump(predictions, f)

