import numpy as np
import pandas as pd
import seaborn as sns
import scanpy as sc
import anndata as ad
import itertools

from sklearn.metrics import pairwise_distances
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler


def format_test_results(test_results_raw):
    """
    Eliminate negative data points from the results
    """

    test_results_formatted = test_results_raw[test_results_raw['compound'] != None]
    test_results_formatted = test_results_formatted[test_results_formatted['dose'] != 0]

    return test_results_formatted

#dist_functions

def __get_edistance(X, Y):
    """
    Calculate edistances between two matrices
    """
    sigma_X = pairwise_distances(X, X, metric="sqeuclidean").mean()
    sigma_Y = pairwise_distances(Y, Y, metric="sqeuclidean").mean()
    delta = pairwise_distances(X, Y, metric="sqeuclidean").mean()
    return 2 * delta - sigma_X - sigma_Y

def __get_r2_score(X, Y):
    """
    Calculate correlation of determination between the row avergae of 2 matrices
    """
    x = np.mean(X, axis=0)
    y = np.mean(Y, axis=0)

    return r2_score(x, y)

def __get_mse(X, Y):
    """
    Calculate MSE between the row average of 2 matrices
    """
    x = np.mean(X, axis=0)
    y = np.mean(Y, axis=0)

    return mean_squared_error(x, y)


def __get_model_performance_pairwise(formatted_test_results, dist_func):
    """
    Get the pairwise distances between matched predicted and perturbed values
    """
    results = dict()
    stdevs = dict()

    for cell_type in formatted_test_results['cell_type'].unique():
        losses = list()

        df_subset = formatted_test_results[formatted_test_results['cell_type'] == cell_type]

        results_per_cell = list()

        for i, row in df_subset.iterrows():

            emb_pert = row['pert_emb'].tolist()
            emb_pred = row['pred_emb'].tolist()

            dist = dist_func(emb_pert, emb_pred)

            results_per_cell.append(dist)

        results[cell_type] = np.mean(results_per_cell)
        stdevs[cell_type] = np.std(results_per_cell)

    return results, stdevs


def __get_model_performance_aggregated(formatted_test_results, dist_func):
    """
    Get the distance between per covariate combination aggregated matrices of perturbed and predicted values
    """
    results = dict()
    stdevs = dict()

    for cell_type in formatted_test_results['cell_type'].unique():
        results_per_cell = list()

        for compound in formatted_test_results['compound'].unique():
            df_subset = formatted_test_results[formatted_test_results['cell_type'] == cell_type]
            df_subset = df_subset[df_subset['compound'] == compound]

            X_pert = np.array(df_subset['pert_emb'].tolist())
            X_pred = np.array(df_subset['pred_emb'].tolist())

            dist = dist_func(X_pert, X_pred)
            results_per_cell.append(dist)

        results[cell_type] = np.mean(results_per_cell)
        stdevs[cell_type] = np.std(results_per_cell)

    return results, stdevs


def __get_results__fc(results, adata_control, obsm_key, gene_names):
    """
    Get fold changes results
    """

    # create adata control
    adata_ctrl = ad.AnnData(adata_control.obsm[obsm_key])
    adata_ctrl.obs['cell_type'] = list(adata_control.obs['cell_type'])
    adata_ctrl.obs['compound'] = list(adata_control.obs['product_name'])
    adata_ctrl.obs['dose'] = 0
    adata_ctrl.obs['condition'] = 'control'

    # create adata perturbed
    adata_pert = ad.AnnData(np.array(results['pert_emb'].tolist()))
    adata_pert.obs['cell_type'] = list(results['cell_type'])
    adata_pert.obs['compound'] = list(results['compound'])
    adata_pert.obs['dose'] = list(results['dose'])
    adata_pert.obs['condition'] = 'perturbed'

    # create adata pred
    adata_pred = ad.AnnData(np.array(results['pred_emb'].tolist()))
    adata_pred.obs['cell_type'] = list(results['cell_type'])
    adata_pred.obs['compound'] = list(results['compound'])
    adata_pred.obs['dose'] = list(results['dose'])
    adata_pred.obs['condition'] = 'predicted'

    adata = ad.concat([adata_ctrl, adata_pert, adata_pred])
    adata.obs_names_make_unique()
    adata.var_names = gene_names

    logFC_results = []

    # calculate FC

    for (cell_type, compound, dose) in adata.obs[['cell_type', 'compound', 'dose']].drop_duplicates().itertuples(
            index=False):

        if compound == "Vehicle":
            continue

        # subset adata for the specific group
        subset1 = adata[(adata.obs['cell_type'] == cell_type) & (adata.obs['compound'] == compound) & (
                adata.obs['dose'] == dose)].copy()
        subset2 = adata[(adata.obs['cell_type'] == cell_type) & (adata.obs['compound'] == "Vehicle")].copy()

        subset = ad.concat([subset1, subset2])

        assert len(list(subset.obs['condition'].unique())) == 3

        sc.tl.rank_genes_groups(subset, groupby='condition', reference='control', method='wilcoxon')

        df_perturbed = sc.get.rank_genes_groups_df(subset, "perturbed")
        df_predicted = sc.get.rank_genes_groups_df(subset, "predicted")

        df_perturbed = df_perturbed.sort_values(by="names")
        df_predicted = df_predicted.sort_values(by="names")

        logFC_pert = df_perturbed['logfoldchanges']
        logFC_pred = df_predicted['logfoldchanges']

        logFC_results.append({
            'cell_type': cell_type,
            'compound': compound,
            'dose': dose,
            'logFC_pert': logFC_pert.tolist(),
            'logFC_pred': logFC_pred.tolist()
        })

    return pd.DataFrame(logFC_results)


def __get_logFC_rank_score(res_logfc_full):
    """
    Get the rank based on logFC results
    """
    results = dict()
    stdevs = dict()

    for cell_type in res_logfc_full['cell_type'].unique():

        res_logfc = res_logfc_full[res_logfc_full['cell_type'] == cell_type]

        scores = list()
        for i, row in res_logfc.iterrows():
            pert_logfc = row['logFC_pert']

            cosine_sim_per_pert = list()
            for j, row2 in res_logfc.iterrows():
                pred_logfc = row2['logFC_pred']

                cos_sim = np.dot(pred_logfc, pert_logfc) / (np.linalg.norm(pred_logfc) * np.linalg.norm(pert_logfc))
                cosine_sim_per_pert.append([j, cos_sim])

            cosine_sim_per_pert = sorted(cosine_sim_per_pert, key=lambda x: x[1], reverse=True)
            position = next((x for x, sublist in enumerate(cosine_sim_per_pert) if sublist[0] == i), -1)
            rank = position / (len(cosine_sim_per_pert) - 1)
            scores.append(rank)

        results[cell_type] = np.mean(scores)
        stdevs[cell_type] = np.std(scores)
    return results, stdevs

def get_all_results(formatted_test_results, adata_control, output_name, gene_names, model_name):

    #pairwise MSE
    res_mse_pairwise, std_mse_pairwise =  __get_model_performance_pairwise(formatted_test_results, mean_squared_error)

    #aggregated MSE
    res_mse_agg, std_mse_agg = __get_model_performance_aggregated(formatted_test_results, __get_mse)

    #pairwise R2
    res_r2_pairwise, std_r2_pairwise = __get_model_performance_pairwise(formatted_test_results, r2_score)

    #aggregated R2
    res_r2_agg, std_r2_agg = __get_model_performance_aggregated(formatted_test_results, __get_r2_score)

    #rank all logFC
    lfc = __get_results__fc(formatted_test_results, adata_control, output_name, gene_names)
    res_rank_logfc, std_rank_logfc = __get_logFC_rank_score(lfc)

    return {"key": model_name, "mse_pw_A549": }













