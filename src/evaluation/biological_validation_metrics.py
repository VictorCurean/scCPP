import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error

def get_top_logfc_correlation_score(res_logfc_full, topn=20):
    """
    Get the pearson correlation between the top DEGs between predicted and target
    """
    results = dict()
    for cell_type in res_logfc_full['cell_type'].unique():
        res_logfc = res_logfc_full[res_logfc_full['cell_type'] == cell_type]

        scores = list()
        for i, row in res_logfc.iterrows():

            #perturbed top 20 logfc
            combined_pert = list(zip(row['pvals_pert'], row['logFC_pert'], row['names']))
            sorted_combined_pert = sorted(combined_pert, key=lambda x: x[0])
            top_genes = sorted_combined_pert[:topn]
            top_genes_logFC_pert = [x[1] for x in top_genes]
            top_genes_names = [x[2] for x in top_genes]

            #predicted logfc for same 20 genes
            combined_pred = dict(zip(row['names'], row['logFC_pred']))
            top_genes_logFC_pred = [combined_pred[gene] for gene in top_genes_names]

            corr, _ = pearsonr(top_genes_logFC_pert, top_genes_logFC_pred)
            scores.append(corr)
        results[cell_type] = scores

    return results

def get_top_delta_correlation_score_deg(res_delta_full, topn=20):
    """
    Pearson correlation between Δ_pert and Δ_pred restricted to the top-N
    DEGs by adjusted p-value from sc.tl.rank_genes_groups (perturbed vs control).
    """
    results = {}

    for cell_type in res_delta_full['cell_type'].unique():
        df = res_delta_full[res_delta_full['cell_type'] == cell_type]
        scores = []

        for _, row in df.iterrows():
            delta_pert = np.asarray(row['delta_pert'])
            delta_pred = np.asarray(row['delta_pred'])
            pvals      = np.asarray(row['pvals_pert'])

            # indices of top-N by significance (smallest p-values)
            idx = np.argsort(pvals)[:topn]

            r, _ = pearsonr(delta_pert[idx], delta_pred[idx])
            scores.append(r)

        results[cell_type] = scores

    return results


def get_biorep_delta(formatted_test_results, adata_control):
    """
    Get the Spearman correlation between the distances of predicted-control values and incremental dosages
    """
    results = dict()

    ctrl_per_cell_type = {
        "A549": np.asarray(np.mean(adata_control[adata_control.obs.cell_type == "A549"].copy().X, axis=0)).flatten(),
        "K562": np.asarray(np.mean(adata_control[adata_control.obs.cell_type == "K562"].copy().X, axis=0)).flatten(),
        "MCF7": np.asarray(np.mean(adata_control[adata_control.obs.cell_type == "MCF7"].copy().X, axis=0)).flatten(),
    }

    for cell_type in formatted_test_results['cell_type'].unique():
        results_per_cell = list()

        for compound in formatted_test_results['compound'].unique():
            distances_pert = list()
            distances_pred = list()
            doses = list()

            df_validation = formatted_test_results[(formatted_test_results['cell_type'] == cell_type) & (formatted_test_results['compound'] == compound)]

            if len(list(df_validation['dose'].unique())) != 4: #make sure there are observations for all doses
                continue

            x_ctrl = ctrl_per_cell_type[cell_type]

            for dose in sorted(list(formatted_test_results['dose'].unique()), reverse=False):
                df_subset = formatted_test_results[formatted_test_results['cell_type'] == cell_type]
                df_subset = df_subset[df_subset['compound'] == compound]
                df_subset = df_subset[df_subset['dose'] == dose]
                doses.append(dose)

                x_pred = np.asarray(np.mean(np.array(df_subset['pred_emb'].tolist()), axis=0))
                x_pert = np.asarray(np.mean(np.array(df_subset['pert_emb'].tolist()), axis=0))

                dist_pred = mean_squared_error(x_ctrl, x_pred)
                dist_pert = mean_squared_error(x_ctrl, x_pert)

                distances_pred.append(dist_pred)
                distances_pert.append(dist_pert)

            corr_pred, _ = spearmanr(distances_pred, doses)
            corr_pert, _ = spearmanr(distances_pert, doses)

            corr_delta = corr_pert - corr_pred #calculate difference between actual biorep and true biorep

            results_per_cell.append(corr_delta)

        results[cell_type] = results_per_cell

    return results

def get_gene_regulation_agreement(lfc, pval_threshold=0.05, lfc_threshold=0.1):
    results = {}

    for cell_type in lfc['cell_type'].unique():
        cell_type_results = lfc[lfc['cell_type'] == cell_type]
        pct_agreements = []


        for _, row in cell_type_results.iterrows():
            # Initialize category arrays
            true_categories = np.full(len(row['names']), 'non_DE', dtype=object)
            pred_categories = np.full(len(row['names']), 'non_DE', dtype=object)

            # Categorize true perturbation
            for i, (pval, lfc_val) in enumerate(zip(row['pvals_pert'], row['logFC_pert'])):
                if pval < pval_threshold:
                    if lfc_val > lfc_threshold:
                        true_categories[i] = 'up'
                    elif lfc_val < -lfc_threshold:
                        true_categories[i] = 'down'

            # Categorize predicted perturbation
            for i, (pval, lfc_val) in enumerate(zip(row['pvals_pred'], row['logFC_pred'])):
                if pval < pval_threshold:
                    if lfc_val > lfc_threshold:
                        pred_categories[i] = 'up'
                    elif lfc_val < -lfc_threshold:
                        pred_categories[i] = 'down'


            # Calculate Agreement
            no_agreements = 0
            no_true_deg_genes = 0

            for i in range(len(true_categories)):
                if true_categories[i] != 'non_DE':
                    no_true_deg_genes += 1
                    if true_categories[i] == pred_categories[i]:
                        no_agreements += 1

            if no_true_deg_genes == 0:
                continue
            else:
                pct_agreements.append(no_agreements / no_true_deg_genes)

        results[cell_type] = pct_agreements

    return results