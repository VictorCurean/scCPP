import torch
import pandas as pd
import pickle as pkl
from torch.utils.data import Dataset
import random
import numpy as np
import math
import anndata as ad
import ast
from tqdm import tqdm
from sklearn.model_selection import train_test_split


class ZhaoDatasetBaseline(Dataset):
    def __init__(self, adata_file):
        self.adata = ad.read_h5ad(adata_file)
        self.data_processed = list()
        self.__match_control_to_treated()

    def __match_control_to_treated(self):
        adata = self.adata

        data_list = list()

        for sample in tqdm(list(adata.obs['sample'].unique())):
            adata_subset = adata[adata.obs['sample'] == sample]

            control_X = adata_subset[adata_subset.obs['perturbation'] == "control"].obsm['X_uce']

            perturbations = list(adata_subset.obs['perturbation'].unique())
            perturbations.remove("control")

            if len(perturbations) == 0:
                print(sample, "has no perturbations, skipping ...")

            for pert in perturbations:

                adata_perturbed = adata_subset[adata_subset.obs['perturbation'] == pert]

                for idx in range(adata_perturbed.n_obs):
                    cell_emb = adata_perturbed.obsm['X_uce'][idx]
                    cell_meta = adata_perturbed.obs.iloc[idx]

                    random_row_idx = np.random.choice(control_X.shape[0])
                    matched_control = control_X[random_row_idx]

                    dose_raw = cell_meta['dose_value']
                    dose_unit = cell_meta['dose_unit']

                    dose_raw = float(dose_raw)

                    if dose_unit == "uM":
                        dose = dose_raw * 1000
                    elif dose_unit == "nM":
                        dose = dose_raw
                    else:
                        raise RuntimeError("Unrecognized dose unit")

                    dose = math.log1p(dose)

                    drug_emb =  ast.literal_eval(cell_meta['sm_emb'])

                    meta = dict()
                    meta['compound'] = cell_meta['perturbation']
                    meta['dose'] = dose
                    meta['sample'] = cell_meta['sample']

                    data_list.append({
                        "idx": idx,
                        "treated_emb": torch.tensor(cell_emb, dtype=torch.float),
                        "matched_control_emb": torch.tensor(matched_control, dtype=torch.float),
                        "drug_emb": torch.tensor(drug_emb, dtype=torch.float),
                        "logdose": torch.tensor([dose], dtype=torch.float),
                        "meta": meta,
                        "drug_name": meta['compound']
                    })

        self.data_processed = data_list

    def __len__(self):
        # Return the number of samples
        return len(self.data_processed)

    def __getitem__(self, idx):
        val = self.data_processed[idx]

        #concatenate control, drug embedding, dose
        input = torch.cat([val['matched_control_emb'], val['drug_emb'], val['logdose']], dim=0)
        output = val['treated_emb']
        meta = val['meta']

        return input, output, meta






