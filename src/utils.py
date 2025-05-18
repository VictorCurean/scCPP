import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad


from sklearn.metrics import pairwise_distances
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr



def format_test_results(test_results_raw):
    """
    Eliminate negative data points from the results
    """

    test_results_formatted = test_results_raw[test_results_raw['compound'] != None]
    test_results_formatted = test_results_formatted[test_results_formatted['dose'] != 0]

    return test_results_formatted

def clamp_negative_values(df, col_name):
    df = df.copy()  # avoid modifying the original DataFrame
    df[col_name] = df[col_name].apply(
        lambda arr: [x if x > 0 else 0 for x in arr]
    )
    return df


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


def __get_model_performance_aggregated(formatted_test_results, dist_func):
    """
    Get the distance between per covariate combination aggregated matrices of perturbed and predicted values
    """
    results = dict()

    for cell_type in formatted_test_results['cell_type'].unique():
        results_per_cell = list()

        for compound in formatted_test_results['compound'].unique():
            for dose in formatted_test_results['dose'].unique():
                df_subset = formatted_test_results[formatted_test_results['cell_type'] == cell_type]
                df_subset = df_subset[df_subset['compound'] == compound]
                df_subset = df_subset[df_subset['dose'] == dose]

                X_pert = np.array(df_subset['pert_emb'].tolist())
                X_pred = np.array(df_subset['pred_emb'].tolist())

                dist = dist_func(X_pert, X_pred)
                results_per_cell.append(dist)

        results[cell_type] = np.mean(results_per_cell)

    return results

def __get_predicted_bio_rep(formatted_test_results, control_adata, output_name):
    """
    Get the Spearman correlation between the distances of predicted-control values and incremental dosages
    """
    results = dict()

    for cell_type in formatted_test_results['cell_type'].unique():
        results_per_cell = list()
        adata_control_subset = control_adata[control_adata.obs['cell_type'] == cell_type]
        x_control = np.mean(adata_control_subset.obsm[output_name], axis=0)

        for compound in formatted_test_results['compound'].unique():
            distances = list()
            doses = list()

            for dose in sorted(list(formatted_test_results['dose'].unique()), reverse=False):
                df_subset = formatted_test_results[formatted_test_results['cell_type'] == cell_type]
                df_subset = df_subset[df_subset['compound'] == compound]
                df_subset = df_subset[df_subset['dose'] == dose]
                doses.append(dose)

                x_pred = np.mean(np.array(df_subset['pred_emb'].tolist()), axis=0)

                dist = mean_squared_error(x_control, x_pred)
                distances.append(dist)

            corr, _ = spearmanr(distances, doses)
            results_per_cell.append(corr)
        results[cell_type] = np.mean(results_per_cell)

    return results



def __get_results__fc(results, adata_control, gene_names):
    """
    Get fold changes results
    """

    #clamp negative values in predicted values
    results = clamp_negative_values(results, 'pred_emb')


    # create adata control
    adata_ctrl = ad.AnnData(adata_control.X.toarray())
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

        logFC_pert = df_perturbed['logfoldchanges'].tolist()
        logFC_pred = df_predicted['logfoldchanges'].tolist()

        names_pert = df_perturbed['names'].tolist()
        names_pred = df_predicted['names'].tolist()

        pvals_pert = df_perturbed['pvals_adj'].tolist()
        pvals_pred = df_predicted['pvals_adj'].tolist()

        assert names_pert == names_pred

        logFC_results.append({
            'cell_type': cell_type,
            'compound': compound,
            'dose': dose,
            'logFC_pert': logFC_pert,
            'logFC_pred': logFC_pred,
            'names': names_pert,
            'pvals_pert': pvals_pert,
            'pvals_pred': pvals_pred
        })

    return pd.DataFrame(logFC_results)

def __get_top_logfc_correlation_score(res_logfc_full, topn=50):
    """
    Get the pearson correlation between the top DEGs between predicted and target
    """
    results = dict()
    for cell_type in res_logfc_full['cell_type'].unique():
        res_logfc = res_logfc_full[res_logfc_full['cell_type'] == cell_type]

        scores = list()
        for i, row in res_logfc.iterrows():

            #perturbed top 50 logfc
            combined_pert = list(zip(row['pvals_pert'], row['logFC_pert'], row['names']))
            sorted_combined_pert = sorted(combined_pert, key=lambda x: x[0])
            top50 = sorted_combined_pert[:topn]
            top50_logFC_pert = [x[1] for x in top50]
            top50_genes = [x[2] for x in top50]

            #predicted logfc for same 50 genes
            combined_pred = dict(zip(row['names'], row['logFC_pred']))
            top50_logFC_pred = [combined_pred[gene] for gene in top50_genes]

            corr, _ = pearsonr(top50_logFC_pert, top50_logFC_pred)
            scores.append(corr)
        results[cell_type] = np.mean(scores)
    return results

def __get_logfc_correlation_score(res_logfc_full):
    """
    Get the Pearson correlation between all DEGs between predicted and target
    """
    results = dict()
    for cell_type in res_logfc_full['cell_type'].unique():
        res_logfc = res_logfc_full[res_logfc_full['cell_type'] == cell_type]

        scores = list()
        for i, row in res_logfc.iterrows():

            logFC_pert = row['logFC_pert']
            logFC_pred = row['logFC_pred']

            corr, _ = pearsonr(logFC_pert, logFC_pred)
            scores.append(corr)
        results[cell_type] = np.mean(scores)
    return results

def __get_logFC_rank_score(res_logfc_full):
    """
    Get the rank based on logFC results
    """
    results = dict()

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
    return results

def get_model_stats(formatted_test_results, adata_control, output_name, gene_names, key):

    #aggregated MSE
    res_mse_agg = __get_model_performance_aggregated(formatted_test_results, __get_mse)

    #aggregated R2
    res_r2_agg = __get_model_performance_aggregated(formatted_test_results, __get_r2_score)

    #aggregated E-distance
    res_edistance_agg = __get_model_performance_aggregated(formatted_test_results, __get_edistance)

    lfc = __get_results__fc(formatted_test_results, adata_control, output_name, gene_names)

    #rank all logFC
    res_rank_logfc = __get_logFC_rank_score(lfc)

    #corr all logFC
    res_logfc_corr = __get_logfc_correlation_score(lfc)

    #corr top logFC
    res_top_logfc_corr = __get_top_logfc_correlation_score(lfc, topn=50)

    #dosage correlation
    res_predicted_bio_rep = __get_predicted_bio_rep(formatted_test_results, adata_control, output_name)

    return {"key": key,
            "mse_A549": res_mse_agg['A549'],
            "mse_K562": res_mse_agg['K562'],
            "mse_MCF7": res_mse_agg['MCF7'],
            "r2_A549": res_r2_agg['A549'],
            "r2_K562": res_r2_agg['K562'],
            "r2_MCF7": res_r2_agg['MCF7'],
            "rank_logfc_A549": res_rank_logfc['A549'],
            "rank_logfc_K562": res_rank_logfc['K562'],
            "rank_logfc_MCF7": res_rank_logfc['MCF7'],
            "edistance_A549": res_edistance_agg['A549'],
            "edistance_K562": res_edistance_agg['K562'],
            "edistance_MCF7": res_edistance_agg['MCF7'],
            "logfc_corr_A549": res_logfc_corr['A549'],
            "logfc_corr_K562": res_logfc_corr['K562'],
            "logfc_corr_MCF7": res_logfc_corr['MCF7'],
            "top_logfc_corr_A549": res_top_logfc_corr['A549'],
            "top_logfc_corr_K562": res_top_logfc_corr['K562'],
            "top_logfc_corr_MCF7": res_top_logfc_corr['MCF7'],
            "predicted_bio_rep_A549": res_predicted_bio_rep['A549'],
            "predicted_bio_rep_K562": res_predicted_bio_rep['K562'],
            "predicted_bio_rep_MCF7": res_predicted_bio_rep['MCF7'],
            }
















