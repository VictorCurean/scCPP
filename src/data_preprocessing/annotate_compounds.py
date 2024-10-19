import pandas as pd
import anndata as ad
import numpy as np
#import pertpy

def split_dataset_vqvae(label_file):

    #add full info column
    labels_df = pd.read_csv(label_file, sep="\t")
    product_dose = list(labels_df['product_dose']) #contains both product name and dose
    cell_type = list(labels_df['cell_type'])
    assert len(product_dose) == len(cell_type)

    obs_len = len(product_dose)

    label = [product_dose[x] + "_" + cell_type[x] for x in range(obs_len)]
    labels_df['label'] = label

    #annotate compounds based on name
    adata = ad.AnnData(X=np.zeros((obs_len, 10)))
    adata.obs = labels_df
    cp = pertpy.metadata.Compound()
    adata = cp.annotate_compounds(adata=adata, query_id='product_name', query_id_type='name', verbosity=5)

    pp_annotations = adata.obs
    no_nan = list(adata.obs['smiles']).count(np.nan)
    no_vehicle = len([x for x in list(adata.obs['product_name']) if "ehicle" in x])
    smiles = list(set(list(adata.obs['smiles'])))

    pp_annotations.to_csv("C:\\Users\\curea\\Documents\\bioFM for drug discovery\\dege-fm\\raw-data\\sciplex_obs_annot_with_smiles.tsv")



#label_file = "C:\\Users\\curea\\Documents\\bioFM for drug discovery\\dege-fm\\raw-data\\sciplex_obs_annot.tsv"
#split_dataset_vqvae(label_file)

def do_manual_annotation():
    pp_annotations = pd.read_csv("/raw-data/sciplex_obs_annot_with_smiles.tsv")

    missing_vals_dict = {'Bisindolylmaleimide IX (Ro 31-8220 Mesylate)': 'Cn1cc(C2=C(c3cn(CCCSC(=N)N)c4ccccc34)C(=O)NC2=O)c2ccccc21',
     'Glesatinib?(MGCD265)': 'COCCNCc1ccc(-c2cc3nccc(Oc4ccc(NC(=S)NC(=O)Cc5ccc(F)cc5)cc4F)c3s2)nc1',
     'Ivosidenib (AG-120)':'N#Cc1ccnc(N2C(=O)CC[C@H]2C(=O)N(c2cncc(F)c2)[C@H](C(=O)NC2CC(F)(F)C2)c2ccccc2Cl)c1',
     'Dacinostat (LAQ824)': 'O=C(/C=C/c1ccc(CN(CCO)CCc2c[nH]c3ccccc23)cc1)NO'}

    for x in missing_vals_dict.keys():
        pp_annotations.loc[pp_annotations['product_name'] == x, 'smiles'] = missing_vals_dict[x]

    list_prod_name = list(pp_annotations['product_name'])
    list_smiles = list(pp_annotations['smiles'])
    not_found_prods = [list_prod_name[i] for i in range(len(list_prod_name)) if type(list_smiles[i]) != str]
    not_found_prods = list(set(not_found_prods))

    print(not_found_prods) # should be only Vehicle

    pp_annotations.to_csv("C:\\Users\\curea\\Documents\\bioFM for drug discovery\\dege-fm\\raw-data\\sciplex_obs_annot_with_smiles_manual.tsv")

#do_manual_annotation()

def get_unique_compound_names():
    adata_obs = pd.read_csv("/raw-data/sciplex_obs_annot_with_smiles_manual.tsv")
    df_subset = adata_obs[['product_name', 'smiles', 'pubchem_name']].drop_duplicates()
    df_subset.to_csv("C:\\Users\\curea\\Documents\\bioFM for drug discovery\\dege-fm\\raw-data\\compound_names.csv")


get_unique_compound_names()