import torch
from torch.utils.data import Dataset

import numpy as np
import ast
from tqdm import tqdm
from scipy.sparse import issparse


class SciplexDatasetUnseenPerturbations(Dataset):
    def __init__(self, adata, drug_list, sm_emb_column, sm_emb_dim, input_type, output_type):
        self.drug_list = drug_list
        self.sm_emb_dim = sm_emb_dim # dimension of the drug embedding used
        self.sm_emb_column = sm_emb_column #column of the sm_emb used
        self.input_type = input_type # type of input (gene expression / scFM embedding)
        self.output_type = output_type # type of output (gene expression / scFM embedding / pathway activation)

        self.adata = adata
        self.data_processed = list()
        #convert obsm to array:
        for key in list({input_type, output_type}):
            if issparse(adata.obsm[key]):
                adata.obsm[key] = adata.obsm[key].toarray()

        self.__match_control_to_treated()



    def __len__(self):
        # Return the number of samples
        return len(self.data_processed)

    def __match_control_to_treated(self):
        data_list = list() #list of dict object

        for idx in tqdm(list(self.adata.obs_names)):

            cell_meta = self.adata.obs.loc[idx]

            if cell_meta['product_name'] == 'Vehicle':
                continue

            if cell_meta['product_name'] not in self.drug_list:
                continue

            idx_position = self.adata.obs.index.get_loc(idx)
            cell_vector = self.adata.obsm[self.output_type][idx_position]

            matched_control_index = cell_meta['match_index']
            idx_position_match = self.adata.obs.index.get_loc(matched_control_index)
            matched_control = self.adata.obsm[self.input_type][idx_position_match]

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
                "meta": meta
            })

        self.data_processed = data_list

    def __getitem__(self, idx):
        val = self.data_processed[idx]

        #concatenate control, drug embedding,
        control_emb = val['matched_control_emb']
        drug_emb = val['drug_emb']
        treated_emb = val['treated_emb']
        meta = val['meta']

        return control_emb, drug_emb, treated_emb, meta