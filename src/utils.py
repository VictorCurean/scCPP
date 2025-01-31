import numpy as np
import pandas as pd
import seaborn as sns
import itertools

from sklearn.metrics import pairwise_distances
from scipy.stats import spearmanr
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

def calculate_edistance(X, Y):
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

    results_pred_loss = dict()
    results_null_loss = dict()
    results_similarity_loss = dict()

    for cell_type in formatted_test_results['cell_type'].unique():
        losses = list()

        df_subset = formatted_test_results[formatted_test_results['cell_type'] == cell_type]

        for compound in tqdm(list(formatted_test_results['compound'].unique())):
            df_subset = formatted_test_results[(formatted_test_results['cell_type'] == cell_type) &
                                               (formatted_test_results['compound'] == compound)]

            ctrl_X = np.array(df_subset['ctrl_emb'].tolist())
            pert_X = np.array(df_subset['pert_emb'].tolist())
            pred_X = np.array(df_subset['pred_emb'].tolist())

            edist_ctrl_pert = calculate_edistance(ctrl_X, pert_X)
            edist_ctrl_pred = calculate_edistance(ctrl_X, pred_X)
            edist_pert_pred = calculate_edistance(pert_X, pred_X)

            key = cell_type + "_" + compound
            results_pred_loss[key] = edist_pert_pred
            results_null_loss[key] = edist_ctrl_pert
            results_similarity_loss[key] = edist_ctrl_pred

    return results_pred_loss, results_null_loss, results_similarity_loss


def get_res_stratified(results, cell_type):
    out = list()
    for key, value in results.items():
        ct = key.split("_")[0]
        if ct == cell_type:
            out.append(value)

    return out


def plot_results(results_formatted, cell_type):
    predloss = get_res_stratified(results_formatted[0], cell_type)
    nullloss = get_res_stratified(results_formatted[1], cell_type)
    similarityloss = get_res_stratified(results_formatted[2], cell_type)


    # Group the lists into pairs
    data = [predloss, nullloss, similarityloss]

    print("Avg Pred Loss:", np.mean(predloss))
    print("Avg Null Loss:", np.mean(nullloss))
    print("Avg Similarity Loss:", np.mean(similarityloss))

    # Create boxplot
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=data, width=0.15, showmeans=True)

    # Adjust x-ticks to group pairs
    plt.xticks(ticks=range(8),
               labels=["Predicted-Perturbed", "Control-Perturbed", "Predicted Control"])
    plt.title(cell_type)
    plt.xlabel("Distance")
    plt.ylabel("E-distance")

    plt.show()