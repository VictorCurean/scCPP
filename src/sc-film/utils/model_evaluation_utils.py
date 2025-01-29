import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import pairwise_distances
from scipy.stats import spearmanr
from tqdm import tqdm


def calculate_edistance(X, Y)
    """
    Calculate edistances between two matrices
    """
    sigma_X = pairwise_distances(X, X, metric="sqeuclidean").mean()
    sigma_Y = pairwise_distances(Y, Y, metric="sqeuclidean").mean()
    delta = pairwise_distances(X, Y, metric="sqeuclidean").mean()
    return 2 * delta - sigma_X - sigma_Y

def format_test_results(test_results_raw):
    """
    Eliminate negative data points from the results
    """

    test_results_formatted = test_results_raw[test_results_raw['compound'] != None]
    test_results_formatted = test_results_formatted[test_results_formatted['dose'] != 0]

    return test_results_formatted


def get_model_stats(formatted_test_results):
    """
    Calculate test results statistics
    """

    results_pred_loss = {"A549": None, "K562": None, "MCF7": None}
    results_null_loss = {"A549": None, "K562": None, "MCF7": None}
    results_PCP = {"A549": None, "K562": None, "MCF7": None}
    results_PR = {"A549": None, "K562": None, "MCF7": None}


    for cell_type in formatted_test_results['cell_type'].unique():
        losses = list()

        df_subset = formatted_test_results[formatted_test_results['cell_type'] == cell_type]

        pred_losses = list()
        null_losses = list()
        PCP = list()
        PR = list()


        for compound in tqdm(list(formatted_test_results['compound'].unique())):
            distances_to_ctrl = list()
            for dose in sorted(formatted_test_results['dose'].unique()):
                df_subset = formatted_test_results[(formatted_test_results['cell_type'] == cell_type) &
                                         (formatted_test_results['compound'] == compound) &
                                         (formatted_test_results['dose'] == dose)]

                ctrl_X = np.array(df_subset['ctrl_emb'].tolist())
                pert_X = np.array(df_subset['pert_emb'].tolist())
                pred_X = np.array(df_subset['pred_emb'].tolist())

                edist_ctrl_pert = calculate_edistance(ctrl_X, pert_X)
                edist_ctrl_pred = calculate_edistance(ctrl_X, pred_X)
                edist_pert_pred = calculate_edistance(pert_X, pred_X)

                distances_to_ctrl.append(edist_ctrl_pred)

                if edist_pert_pred < edist_ctrl_pred:
                    PCP.append(1)

                pred_losses.append(edist_pert_pred)
                null_losses.append(edist_ctrl_pert)

            corr, _ = spermanr(list(formatted_test_results['dose'].unique()), distances_to_ctrl)
            PR.append(corr)

        results_pred_loss[cell_type] = pred_losses
        results_null_loss[cell_type] = null_losses
        results_PCP[cell_type] = PCP
        results_PR[cell_type] = PR

    return results_pred_loss, results_null_loss, results_PCP, results_PR

