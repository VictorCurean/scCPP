import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

def __get_r2_delta(X, Y, ctrl):
    """
    Calculate correlation of determination between the row avergae delta of 2 matrices
    """
    x = np.asarray(np.mean(X, axis=0) - ctrl).flatten()
    y = np.asarray(np.mean(Y, axis=0) - ctrl).flatten()


    return r2_score(x, y)


def __get_css_delta(X, Y, ctrl):
    """
    Calculate cosine similarity between the row-average delta of two matrices.

    Returns: scalar cosine similarity between perturbation vectors
    """
    x = np.asarray(np.mean(X, axis=0) - ctrl).flatten()
    y = np.asarray(np.mean(Y, axis=0) - ctrl).flatten()

    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def __get_pearson_delta(X, Y, ctrl):
    """
    Calculate Pearson correlation between the row-average delta of two matrices.

    Returns: scalar Pearson correlation coefficient between perturbation vectors
    """
    x = np.asarray(np.mean(X, axis=0) - ctrl).flatten()
    y = np.asarray(np.mean(Y, axis=0) - ctrl).flatten()
    r, _ = pearsonr(x, y)
    return r

def get_goodness_of_fit_metrics(df_predictions, adata_control):
    """
    Get the distance between per covariate combination aggregated matrices of perturbed and predicted values
    """
    results_r2 = dict()
    results_css = dict()
    results_pearson = dict()

    #generate control vectors
    ctrl_per_cell_type = {
        "A549": np.mean(adata_control[adata_control.obs.cell_type == "A549"].copy().X, axis=0),
        "K562": np.mean(adata_control[adata_control.obs.cell_type == "K562"].copy().X, axis=0),
        "MCF7": np.mean(adata_control[adata_control.obs.cell_type == "MCF7"].copy().X, axis=0)
    }

    for cell_type in df_predictions['cell_type'].unique():
        results_per_cell_r2 = list()
        results_per_cell_css = list()
        results_per_cell_pearson = list()

        for compound in df_predictions['compound'].unique():
            for dose in df_predictions['dose'].unique():

                df_subset = df_predictions[df_predictions['cell_type'] == cell_type]
                df_subset = df_subset[df_subset['compound'] == compound]
                df_subset = df_subset[df_subset['dose'] == dose]

                if df_subset.shape[0] == 0:
                    continue

                X_pert = np.array(df_subset['pert_emb'].tolist())
                X_pred = np.array(df_subset['pred_emb'].tolist())

                ctrl = ctrl_per_cell_type[cell_type]

                r2 = __get_r2_delta(X_pert, X_pred, ctrl)
                css = __get_css_delta(X_pert, X_pred, ctrl)
                pearsonr = __get_pearson_delta(X_pert, X_pred, ctrl)


                results_per_cell_r2.append(r2)
                results_per_cell_css.append(css)
                results_per_cell_pearson.append(pearsonr)


        results_r2[cell_type] = results_per_cell_r2
        results_css[cell_type] = results_per_cell_css
        results_pearson[cell_type] = results_per_cell_pearson

    return results_r2, results_css, results_pearson