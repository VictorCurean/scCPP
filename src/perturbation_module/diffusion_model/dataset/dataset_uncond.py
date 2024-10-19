import torch
from torch.utils.data import Dataset
import anndata as ad


class SciplexDatasetUncond(Dataset):
    def __init__(self, adata_file):
        self.adata = ad.read_h5ad(adata_file)

    def __len__(self):
        # Return the number of samples
        return len(list(self.adata.obs_names))

    def __getitem__(self, idx):
        # Get the embedding and the corresponding drug embedding
        embedding = torch.tensor(self.adata.X[idx, :], dtype=torch.float)

        return embedding
