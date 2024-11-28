import anndata as ad
import scanpy as sc
import pandas as pd
from tqdm import tqdm
from scipy.stats import energy_distance

def calculate_e_distance(adata):

    def __get_energy_distance(treated, control):
        samples_treated = treated.X.tolist()
        samples_control = control.X.tolist()

        e_dist = energy_distance(samples_control, samples_treated)
        return e_dist

    results = list()

    control_A549 = adata[(adata.obs['cell_type'] == "A549") & (adata.obs['product_name'] == "Vehicle")]
    control_K562 = adata[(adata.obs['cell_type'] == "K562") & (adata.obs['product_name'] == "Vehicle")]
    control_MCF7 = adata[(adata.obs['cell_type'] == "MCF7") & (adata.obs['product_name'] == "Vehicle")]

    for compound in tqdm(list(adata.obs['product_name'].unique())):
        # if compound == "Vehicle":
        #     continue

        for cell_type in list(adata.obs['cell_type'].unique()):
            for dose in list(adata.obs['dose'].unique()):

                adata_subset = adata[
                    (adata.obs['product_name'] == compound) |
                    (adata.obs['cell_type'] == cell_type) |
                    (adata.obs['dose'] == dose)
                ]

                reference = None

                if cell_type == "A549":
                    reference = control_A549
                elif cell_type == "K562":
                    reference = control_K562
                elif cell_type == "MCF7":
                    reference = control_MCF7
                else:
                    raise RuntimeError("Invalid Cell Type")


                #calculate statistics between adata_subset and reference
                e_dist = __get_energy_distance(adata_subset, reference)

                size_treated = adata_subset.n_obs


                results.append({"compound": compound, "dose": dose, "cell_type": cell_type, "e_dist": e_dist, 'sample_size': size_treated})

    results = pd.DataFrame(results)
    return results



