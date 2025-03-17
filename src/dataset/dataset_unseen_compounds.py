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

class SciplexDatasetUnseenPerturbations(Dataset):
    def __init__(self, adata_file, drug_list, sm_emb_column, sm_emb_dim, input_type, output_type):
        self.drug_list = drug_list
        self.sm_emb_dim = sm_emb_dim # dimension of the drug embedding used
        self.sm_emb_column = sm_emb_column #column of the sm_emb used
        self.input_type = input_type # type of input (gene expression / scFM embedding)
        self.output_type = output_type # type of output (gene expression / scFM embedding / pathway activation)

        self.adata = ad.read_h5ad(adata_file)
        self.data_processed = list()
        self.__match_control_to_treated()

    def __len__(self):
        # Return the number of samples
        return len(self.data_processed)

    def __match_control_to_treated(self):
        adata = self.adata

        control_A549 = adata[(adata.obs['cell_type'] == "A549") & (adata.obs['product_name'] == "Vehicle")].obsm[self.input_type]
        control_K562 = adata[ (adata.obs['cell_type'] == "K562") & (adata.obs['product_name'] == "Vehicle")].obsm[self.input_type]
        control_MCF7 = adata[ (adata.obs['cell_type'] == "MCF7") & (adata.obs['product_name'] == "Vehicle")].obsm[self.input_type]

        data_list = list() #list of dict object

        for idx in tqdm(range(adata.n_obs)):
            cell_meta = adata.obs.iloc[idx]

            if cell_meta['product_name'] == 'Vehicle':
                continue

            if cell_meta['product_name'] not in self.drug_list:
                continue

            cell_vector = adata.obsm[self.output_type][idx]

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
            drug_emb = ast.literal_eval(cell_meta[self.sm_emb_column])

            #get dose
            dose = float(cell_meta['dose'])

            #multiply drug emb with does with the dose
            drug_emb = np.array(drug_emb) * np.log10(1+dose)

            #metadata
            meta = dict()
            meta['compound'] = cell_meta['product_name']
            meta['cell_type'] = cell_meta['cell_type']
            meta['dose'] = float(cell_meta['dose'])


            # Store the treated and matched control metadata
            data_list.append({
                "idx": idx,
                "treated_emb": torch.tensor(cell_vector, dtype=torch.float),
                "matched_control_emb": torch.tensor(matched_control, dtype=torch.float),
                "drug_emb": torch.tensor(drug_emb, dtype=torch.float),
                "dose": dose,
                "meta": meta
            })

            print(type(dose))
            print(type(meta['dose']))

        self.data_processed = data_list

    def __getitem__(self, idx):
        val = self.data_processed[idx]

        #concatenate control, drug embedding,
        control_emb = val['matched_control_emb']
        drug_emb = val['drug_emb']
        treated_emb = val['treated_emb']
        meta = val['meta']

        return control_emb, drug_emb, treated_emb, meta