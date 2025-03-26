from torch.utils.data.dataloader import DataLoader
from evaluator.abstract_evaluator import AbstractEvaluator

import pandas as pd
import numpy as np
from tqdm import tqdm


class MeanEvaluator(AbstractEvaluator):

    def __init__(self, sciplex_dataset_train, sciplex_dataset_test):

        #load training, validation and test data in
        self.sciplex_loader_train = DataLoader(sciplex_dataset_train,
                                               batch_size=self.config['train_params']['batch_size'],
                                               shuffle=True,
                                               num_workers=0)


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

        mean_A549 = None
        mean_K562 = None
        mean_MCF7 = None

        all_targets = list()
        all_cell_types = list()

        for control, drug_emb, target, meta in tqdm(self.sciplex_loader_train):
            all_targets.extend([x for x in target])
            all_cell_types.extend(meta['cell_type'])

            for i in range(len(all_cell_types)):
                if all_cell_types[i] == 'A549':
                    mean_A549 += all_targets[i]

                elif all_cell_types[i] == 'K562':
                    mean_K562 += all_targets[i]

                elif all_cell_types[i] == 'MCF7':
                    mean_MCF7 += all_targets[i]

        mean_A549 = np.mean(mean_A549)
        mean_K562 = np.mean(mean_K562)
        mean_MCF7 = np.mean(mean_MCF7)

        for control, drug_emb, target, meta in tqdm(self.sciplex_loader_test):
            control_emb_list = [x  for x in control]
            treated_emb_list = [x for x in target]

            # Meta information
            compounds = meta['compound']
            cell_types = meta['cell_type']
            doses = [d.item() for d in meta['dose']]

            output_list = []
            for i in range(len(cell_types)):
                if cell_types[i] == 'A549':
                    output_list.append(mean_A549)
                elif cell_types[i] == 'K562':
                    output_list.append(mean_K562)
                elif cell_types[i] == 'MCF7':
                    output_list.append(mean_MCF7)

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