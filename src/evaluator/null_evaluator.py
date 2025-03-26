from torch.utils.data.dataloader import DataLoader
from evaluator.abstract_evaluator import AbstractEvaluator

import torch
import pandas as pd
from tqdm import tqdm


class NullEvaluator(AbstractEvaluator):

    def __init__(self, sciplex_dataset_test):
        self.sciplex_loader_test = DataLoader(sciplex_dataset_test,
                                              batch_size=self.config['train_params']['batch_size'],
                                              shuffle=True, num_workers=0)

    def read_config(self, config_path):
        pass


    def prepare_model(self):
        pass

    def train(self, loss_fn):
        pass


    def test(self):
        control_embeddings = []
        treated_embeddings = []
        model_output = []
        compounds_list = []
        cell_types_list = []
        doses_list = []

        for control, drug_emb, target, meta in tqdm(self.sciplex_loader_test):
            # Move tensors to the specified device
            control = control.to(self.device)
            target = target.to(self.device)

            # Convert tensors to lists of NumPy arrays for DataFrame compatibility
            control_emb_list = [x.cpu().numpy() for x in torch.unbind(control, dim=0)]
            treated_emb_list = [x.cpu().numpy() for x in torch.unbind(target, dim=0)]
            output_list = [x.cpu().numpy() for x in torch.unbind(control, dim=0)]

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