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


class SciplexDatasetBaseline(Dataset):
    def __init__(self, adata_file, seed):
        self.adata = ad.read_h5ad(adata_file)
        self.seed = seed
        self.data_processed = list()
        self.__match_control_to_treated()


    def __len__(self):
        # Return the number of samples
        # TODO change this
        return len(self.data_processed)

    def __match_control_to_treated(self):
        #np.random.seed(self.seed)
        adata = self.adata

        control_A549 = adata[(adata.obs['cell_type'] == "A549") & (adata.obs['product_name'] == "Vehicle")].X
        control_K562 = adata[ (adata.obs['cell_type'] == "K562") & (adata.obs['product_name'] == "Vehicle")].X
        control_MCF7 = adata[ (adata.obs['cell_type'] == "MCF7") & (adata.obs['product_name'] == "Vehicle")].X

        data_list = list() #list of dict object

        for idx in tqdm(range(adata.n_obs)):
            cell_meta = adata.obs.iloc[idx]
            cell_emb = adata.X[idx]

            if cell_meta['product_name'] == 'Vehicle':
                continue

            else:
                matched_control = None
                if cell_meta['cell_type'] == "A549":
                    control_pool = control_A549
                elif cell_meta['cell_type'] == "K562":
                    control_pool = control_K562
                elif cell_meta['cell_type'] == "MCF7":
                    control_pool = control_MCF7
                else:
                    raise ValueError(f"Unknown cell type: {cell_meta['cell_type']}")

                # Randomly select a control cell from the relevant pool
                random_row_idx = np.random.choice(control_pool.shape[0])
                matched_control = control_pool[random_row_idx]

                #get drug embedding
                drug_emb = ast.literal_eval(cell_meta['sm_embedding'])

                #natural log of dose
                dose = math.log1p(cell_meta['dose'])

                #metadata
                meta = dict()
                meta['compound'] = cell_meta['product_name']
                meta['dose'] = cell_meta['dose']
                meta['cell_type'] = cell_meta['cell_type']


                # Store the treated and matched control metadata
                data_list.append({
                    "idx": idx,
                    "treated_emb": torch.tensor(cell_emb, dtype=torch.float),
                    "matched_control_emb": torch.tensor(matched_control, dtype=torch.float),
                    "drug_emb": torch.tensor(drug_emb, dtype=torch.float),
                    "logdose": torch.tensor([dose], dtype=torch.float),
                    "meta": meta
                })

        self.data_processed = data_list

    def __getitem__(self, idx):
        val = self.data_processed[idx]

        #concatenate control, drug embedding, dose
        input = torch.cat([val['matched_control_emb'], val['drug_emb'], val['logdose']], dim=0)
        output = val['treated_emb']
        meta = val['meta']

        return input, output, meta
