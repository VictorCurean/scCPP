import torch
import pandas as pd
from tqdm import tqdm

from torch.utils.data.dataloader import DataLoader

from src.dataset.dataset_unseen_compounds import SciplexDatasetUnseenPerturbations
from src.utils import get_model_stats


class NullEvaluator:

    def __init__(self, test_dataset):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.test_loader = self.create_dataloader(test_dataset, 16)

    @staticmethod
    def create_dataloader(dataset, batch_size):
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    def train_with_validation(self):
        pass

    def train(self):
        pass

    def validate(self):
        pass

    def test(self):
        results = {'ctrl_emb': [], 'pert_emb': [], 'pred_emb': [], 'compound': [], 'cell_type': [], 'dose': []}


        for control, drug_emb, target, meta in tqdm(self.test_loader):
            control, drug_emb, target = control.to(self.device), drug_emb.to(self.device), target.to(self.device)


            results['ctrl_emb'].extend(control.cpu().numpy())
            results['pert_emb'].extend(target.cpu().numpy())
            results['pred_emb'].extend(control.cpu().numpy())
            results['compound'].extend(meta['compound'])
            results['cell_type'].extend(meta['cell_type'])
            results['dose'].extend([d.item() for d in meta['dose']])

        return pd.DataFrame(results)

    @staticmethod
    def objective():
        pass

    @staticmethod
    def cross_validation_models(drug_splits=None, adata=None, input_name=None,
                                output_name=None, drug_rep_name=None, drug_emb_size=None,
                                gene_names_key=None, run_name=None):
        output = dict()
        for i in range(5):
            drugs_test = drug_splits[f'drug_split_{i}']['test']


            dataset_test = SciplexDatasetUnseenPerturbations(adata, drugs_test, drug_rep_name, drug_emb_size, input_name, output_name)


            final_ev = NullEvaluator(dataset_test)

            #Get model performance metrics
            adata_control = adata[adata.obs['product_name'] == "Vehicle"]
            gene_names = adata_control.uns[gene_names_key]
            predictions = final_ev.test()

            performance = get_model_stats(predictions, adata_control, output_name, gene_names, run_name)
            output[i] = performance

        return output

