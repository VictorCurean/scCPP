import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def __get_mse(X, Y):
    """
    Calculate MSE between the row average of 2 matrices
    """
    x = np.mean(X, axis=0)
    y = np.mean(Y, axis=0)

    return mean_squared_error(x, y)

def __get_mae(X, Y):
    """
    Calculate MAE between the row average of 2 matrices
    """
    x = np.mean(X, axis=0)
    y = np.mean(Y, axis=0)
    return mean_absolute_error(x, y)

def __get_rmse(X, Y):
    """
    Calculate RMSE between the row average of 2 matrices
    """
    x = np.mean(X, axis=0)
    y = np.mean(Y, axis=0)
    mse = mean_squared_error(x, y)
    return np.sqrt(mse)

def __get_l2norm(X, Y):
    """
    Calculate L2 norm distance between the row average of 2 matrices
    """
    x = np.mean(X, axis=0)
    y = np.mean(Y, axis=0)
    return np.linalg.norm(x - y)  # L2 norm is Euclidean distance


def get_error_metrics(df_predictions):
    """
    Get the distance between per covariate combination aggregated matrices of perturbed and predicted values
    """
    results_mse = dict()
    results_mae = dict()
    results_rmse = dict()
    results_l2norm = dict()

    for cell_type in df_predictions['cell_type'].unique():
        results_per_cell_mse = list()
        results_per_cell_mae = list()
        results_per_cell_rmse = list()
        results_per_cell_l2norm = list()

        for compound in df_predictions['compound'].unique():
            for dose in df_predictions['dose'].unique():

                df_subset = df_predictions[df_predictions['cell_type'] == cell_type]
                df_subset = df_subset[df_subset['compound'] == compound]
                df_subset = df_subset[df_subset['dose'] == dose]

                if df_subset.shape[0] == 0:
                    continue

                X_pert = np.array(df_subset['pert_emb'].tolist())
                X_pred = np.array(df_subset['pred_emb'].tolist())


                mse = __get_mse(X_pert, X_pred)
                mae = __get_mae(X_pert, X_pred)
                rmse = __get_rmse(X_pert, X_pred)
                l2norm = __get_l2norm(X_pert, X_pred)

                results_per_cell_mse.append(mse)
                results_per_cell_mae.append(mae)
                results_per_cell_rmse.append(rmse)
                results_per_cell_l2norm.append(l2norm)

        results_mse[cell_type] = results_per_cell_mse
        results_mae[cell_type] = results_per_cell_mae
        results_rmse[cell_type] = results_per_cell_rmse
        results_l2norm[cell_type] = results_per_cell_l2norm

    return results_mse, results_mae, results_rmse, results_l2norm

