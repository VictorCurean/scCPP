import numpy as np
import pandas as pd

from src.evaluation.utils import *

def get_logFC_rank_similarity_score(res_logfc_full):
    """
    Get the rank similarity metric (see Perturbench) based on logFC/condition and cosine similarity
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

        results[cell_type] = scores
    return results


def get_expr_rank_similarity_score(formatted_test_results, adata_control):
    """
    Get the rank similarity metric (see Perturbench) based on pseudobulked expression deltas and cosine similarity
    """
    ctrl_per_cell_type = {
        "A549": np.asarray(np.mean(adata_control[adata_control.obs.cell_type == "A549"].copy().X, axis=0)).flatten(),
        "K562": np.asarray(np.mean(adata_control[adata_control.obs.cell_type == "K562"].copy().X, axis=0)).flatten(),
        "MCF7": np.asarray(np.mean(adata_control[adata_control.obs.cell_type == "MCF7"].copy().X, axis=0)).flatten(),
    }

    #Create exp deltas
    pseudobulk_results = list()

    for cell_type in formatted_test_results['cell_type'].unique():
        for compound in formatted_test_results['compound'].unique():
            for dose in formatted_test_results['dose'].unique():
                df_subset = formatted_test_results[formatted_test_results['cell_type'] == cell_type]
                df_subset = df_subset[df_subset['compound'] == compound]
                df_subset = df_subset[df_subset['dose'] == dose]

                if df_subset.shape[0] == 0:
                    continue

                x_pert = np.mean(np.array(df_subset['pert_emb'].tolist()), axis=0)
                x_pred = np.mean(np.array(df_subset['pred_emb'].tolist()), axis=0)
                x_ctrl = ctrl_per_cell_type[cell_type]

                pseudobulk_results.append({'cell_type': cell_type, 'compound': compound, 'dose': dose, "x_pert_delta": x_pert- x_ctrl, "x_pred_delta": x_pred- x_ctrl})

    pseudobulk_results = pd.DataFrame(pseudobulk_results)
    results = dict()

    #Calculate ranks
    for cell_type in pseudobulk_results['cell_type'].unique():

        res_logfc = pseudobulk_results[pseudobulk_results['cell_type'] == cell_type]

        scores = list()
        for i, row in res_logfc.iterrows():
            pert_delta = row['x_pert_delta']

            cosine_sim_per_pert = list()
            for j, row2 in res_logfc.iterrows():
                pred_delta = row2['x_pred_delta']

                cos_sim = np.dot(pred_delta, pert_delta) / (np.linalg.norm(pred_delta) * np.linalg.norm(pert_delta))
                cosine_sim_per_pert.append([j, cos_sim])

            cosine_sim_per_pert = sorted(cosine_sim_per_pert, key=lambda x: x[1], reverse=True)
            position = next((x for x, sublist in enumerate(cosine_sim_per_pert) if sublist[0] == i), -1)
            rank = position / (len(cosine_sim_per_pert) - 1)
            scores.append(rank)

        results[cell_type] = scores
    return results






