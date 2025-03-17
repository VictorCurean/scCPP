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


def __get_results__fc(results, adata_control):
    """
    Get fold changes results
    """

    #create adata control
    adata_ctrl = ad.AnnData(adata_control.X)
    adata_ctrl.obs['cell_type'] = adata_control.obs['cell_type']
    adata_ctrl.obs['compound'] = adata_control.obs['product_name']
    adata_ctrl.obs['dose'] = 0
    adata_ctrl.obs['condition'] = 'control'

    #create adata perturbed
    adata_pert = ad.AnnData(np.array(results['pert_emb'].tolist()))
    adata_pert.obs['cell_type'] = list(results['cell_type'])
    adata_pert.obs['compound'] = list(results['compound'])
    adata_pert.obs['dose'] = list(results['dose'])
    adata_pert.obs['condition'] = 'perturbed'

    #create adata pred
    adata_pred = ad.AnnData(np.array(results['pred_emb'].tolist()))
    adata_pred.obs['cell_type'] = list(results['cell_type'])
    adata_pred.obs['compound'] = list(results['compound'])
    adata_pred.obs['dose'] = list(results['dose'])
    adata_pred.obs['condition'] = 'predicted'


    adata = ad.concat([adata_ctrl, adata_pert, adata_pred])

    logFC_results = []

    #calculate FC

    for (cell_type, compound, dose) in adata.obs[['cell_type', 'compound', 'dose']].drop_duplicates().itertuples(
            index=False):
        #subset adata for the specific group
        subset = adata[(adata.obs['cell_type'] == cell_type) & (adata.obs['compound'] == compound) & (
                    adata.obs['dose'] == dose)].copy()

        assert len(list(subset['condition'].unique())) == 3

        sc.tl.rank_genes_groups(subset, groupby='condition', reference='control', method='wilcoxon')

        names_pert = subset.uns['rank_genes_groups']['names']['perturbed']
        names_pred = subset.uns['rank_genes_groups']['names']['predicted']

        assert names_pert == names_pred

        logFC_pert = subset.uns['rank_genes_groups']['logfoldchanges']['perturbed']
        logFC_pred = subset.uns['rank_genes_groups']['logfoldchanges']['predicted']

        logFC_results.append({
            'cell_type': cell_type,
            'compound': compound,
            'dose': dose,
            'logFC_pert': logFC_pert.tolist(),
            'logFC_pred': logFC_pred.tolist()
        })

    return pd.DataFrame(logFC_results)


def __get_topn_genes_similarity(logfc_res, dist_func, top_n):
    distances = {}

    # Group by the unique conditions (cell_type, compound, dose)
    conditions = logfc_res.groupby(['cell_type', 'compound', 'dose'])




def __get_rank_similarity(logfc_res, dist_func, reverse):
    """
    Calculate rank similarity metric proposed in the PerturBench paper.
    """
    results = list()

    #iterate over all perturbed logFC values
    for i, row1 in logfc_res.iterrows():

        fc_pert = np.array(row1['logFC_pert'])

        order = list()

        #iterate over all predicted logFC values
        for j, row2 in logfc_res.iterrows():

            fc_pred = np.array(row2['logFC_pred'])

            dist = dist_func(fc_pert, fc_pred)

            order.append([j, dist])

        order = sorted(order, key=lambda x: x[1], reverse=reverse)

        index = next((x for x, val in enumerate(order) if val[0] == j), None)

        total_conditions = logfc_res.shape[0]

        score = index / rank

        results.append(score)

    return np.mean(results), np.std(results)










