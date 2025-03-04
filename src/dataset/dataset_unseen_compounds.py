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
    def __init__(self, adata_file, drug_list, pct_treatement_negative,
                 pct_dosage_negative, sm_emb_column, sm_emb_dim, input_type, output_type, scfm_obsm_name):
        self.drug_list = drug_list
        self.pct_treatement_negative = pct_treatement_negative # percentage of treatement negative i.e. drug emb set to zero
        self.pct_dosage_negative = pct_dosage_negative # percentage of dosage negative i.e. dosage set to zero
        self.sm_emb_dim = sm_emb_dim # dimension of the drug embedding used
        self.sm_emb_column = sm_emb_column #column of the sm_emb used
        self.input_type = input_type # type of input (gene expression / scFM embedding)
        self.output_type = output_type # type of output (gene expression / scFM embedding / pathway activation)
        self.scfm_obsm_name = scfm_obsm_name

        self.adata = ad.read_h5ad(adata_file)
        self.data_processed = list()
        self.__match_control_to_treated()
        self.add_treatement_negative()
        self.add_dosage_negative()

    def __len__(self):
        # Return the number of samples
        return len(self.data_processed)

    def __match_control_to_treated(self):
        adata = self.adata

        if self.input_type == "gene_expression":

            control_A549 = adata[(adata.obs['cell_type'] == "A549") & (adata.obs['product_name'] == "Vehicle")].X
            control_K562 = adata[ (adata.obs['cell_type'] == "K562") & (adata.obs['product_name'] == "Vehicle")].X
            control_MCF7 = adata[ (adata.obs['cell_type'] == "MCF7") & (adata.obs['product_name'] == "Vehicle")].X

        elif self.input_type == "scfm_embedding":

            control_A549 = adata[(adata.obs['cell_type'] == "A549") & (adata.obs['product_name'] == "Vehicle")].obsm[self.scfm_obsm_name]
            control_K562 = adata[ (adata.obs['cell_type'] == "K562") & (adata.obs['product_name'] == "Vehicle")].obsm[self.scfm_obsm_name]
            control_MCF7 = adata[ (adata.obs['cell_type'] == "MCF7") & (adata.obs['product_name'] == "Vehicle")].obsm[self.scfm_obsm_name]

        data_list = list() #list of dict object

        for idx in tqdm(range(adata.n_obs)):
            cell_meta = adata.obs.iloc[idx]

            if cell_meta['product_name'] == 'Vehicle':
                continue

            if cell_meta['product_name'] not in self.drug_list:
                continue

            #use preferred cell representation vector

            if self.output_type == "gene_expression":
                cell_vector = adata.X[idx]
            elif self.output_type == "scfm_embedding":
                cell_vector = adata.obsm[self.scfm_obsm_name][idx]


            matched_control = None
            if cell_meta['cell_type'] == "A549":
                control_pool = control_A549
            elif cell_meta['cell_type'] == "K562":
                control_pool = control_K562
            elif cell_meta['cell_type'] == "MCF7":
                control_pool = control_MCF7
            else:
                raise ValueError(f"Unknown cell type: {cell_meta['cell_type']}")

            for i in range(self.n_match):

                # Randomly select a control cell from the relevant pool
                random_row_idx = np.random.choice(control_pool.shape[0])
                matched_control = control_pool[random_row_idx]

                #get drug embedding
                drug_emb = ast.literal_eval(cell_meta[sm_emb_column])

                #get dose
                dose = float(cell_meta['dose'])

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

        self.data_processed = data_list

    def add_treatement_negative(self):
        if self.pct_treatement_negative == 0:
            return
        else:
            # calculate how many negative pairs to add per cell type
            no_examples_to_add_total = round(self.pct_treatement_negative * len(self.data_processed))
            no_examples_to_add_per_celltype = round(no_examples_to_add_total/3)

            control_A549 = adata[(adata.obs['cell_type'] == "A549") & (adata.obs['product_name'] == "Vehicle")].X
            control_K562 = adata[(adata.obs['cell_type'] == "K562") & (adata.obs['product_name'] == "Vehicle")].X
            control_MCF7 = adata[(adata.obs['cell_type'] == "MCF7") & (adata.obs['product_name'] == "Vehicle")].X

            control_pools = [control_A549, control_K562, control_MCF7]

            data_list_negative = list()

            idx = 10000000

            #for each cell type, add negative pairs
            for control in control_pools:
                cell_type = list(control.obs['cell_type'].unique())[0]
                for x in range(no_examples_to_add_per_celltype):
                    random_control = control[np.random.choice(control.shape[0])]
                    idx += 1

                    data_list_negative.append({
                        "idx": idx,
                        "treated_emb": torch.tensor(random_control, dtype=torch.float),
                        "matched_control_emb": torch.tensor(random_control, dtype=torch.float),
                        "drug_emb": torch.zeros(self.sm_emb_dim),
                        "dose": 0,
                        "meta": {"compound": None, "cell_type": cell_type, dose: 0}
                    })

            self.data_processed.extend(data_list_negative)

    def add_dosage_negative(self):
        if self.pct_dosage_negative == 0:
            return
        else:
            # TODO
            pass


    def __getitem__(self, idx):
        val = self.data_processed[idx]

        #concatenate control, drug embedding,
        control_emb = val['matched_control_emb']
        drug_emb = val['drug_emb']
        treated_emb = val['treated_emb']
        dose = val['dose']
        meta = val['meta']

        return control_emb, drug_emb, treated_emb, dose, meta