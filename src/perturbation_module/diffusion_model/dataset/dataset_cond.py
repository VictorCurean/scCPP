import torch
import pandas as pd
import pickle as pkl
from torch.utils.data import Dataset


class SciplexDatasetCond(Dataset):
    def __init__(self, embedding_file, label_file, drug_embedding_column):
        def __init__(self, adata_file):
            self.adata = ad.read_h5ad(adata_file)

    def __len__(self):
        # Return the number of samples
        return len(self.embeddings)

    def __getitem__(self, idx):
        # Get the embedding and the corresponding drug embedding
        emembedding = torch.tensor(self.adata.X[idx, :], dtype=torch.float)
        drug_emb = torch.tensor(self.adata.obs[idx, drug_embedding_column],
                             dtype=torch.long)

        return embedding, drug_emb
