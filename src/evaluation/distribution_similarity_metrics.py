import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from scipy.stats import wasserstein_distance


def __compute_mmd(X, Y, gamma=None):
    """
    Calculate Maximum Mean Discrepancy (MMD) between two sample sets X and Y.
    Uses RBF kernel.

    Parameters:
    - X: array-like, shape (n_samples_x, n_features)
    - Y: array-like, shape (n_samples_y, n_features)
    - gamma: float, optional RBF kernel width parameter (1 / (2 * sigma^2))

    Returns:
    - MMD^2 (float)
    """
    if gamma is None:
        # Use median heuristic for gamma
        Z = np.vstack([X, Y])
        dists = np.linalg.norm(Z[:, None] - Z[None, :], axis=-1)
        gamma = 1.0 / (2 * np.median(dists) ** 2 + 1e-6)

    K_xx = rbf_kernel(X, X, gamma=gamma)
    K_yy = rbf_kernel(Y, Y, gamma=gamma)
    K_xy = rbf_kernel(X, Y, gamma=gamma)

    m = X.shape[0]
    n = Y.shape[0]

    mmd = np.sum(K_xx) / (m * m) + np.sum(K_yy) / (n * n) - 2 * np.sum(K_xy) / (m * n)
    return mmd


def __compute_wasserstein(X, Y):
    """
    Compute the average Wasserstein Distance across each feature dimension.

    Parameters:
    - X: array-like, shape (n_samples, n_features)
    - Y: array-like, shape (n_samples, n_features)

    Returns:
    - Average 1D Wasserstein distance across all features
    """
    X = np.asarray(X)
    Y = np.asarray(Y)

    assert X.shape[1] == Y.shape[1]

    distances = [
        wasserstein_distance(X[:, i], Y[:, i]) for i in range(X.shape[1])
    ]
    return np.mean(distances)


def get_distribution_similarity_metrics(df_predictions):
    """
    Get the distance between per covariate combination aggregated matrices of perturbed and predicted values
    """
    results_mmd = dict()
    results_wasserstein = dict()

    for cell_type in df_predictions['cell_type'].unique():
        results_per_cell_mmd = list()
        results_per_cell_wasserstein = list()


        for compound in df_predictions['compound'].unique():
            for dose in df_predictions['dose'].unique():

                df_subset = df_predictions[df_predictions['cell_type'] == cell_type]
                df_subset = df_subset[df_subset['compound'] == compound]
                df_subset = df_subset[df_subset['dose'] == dose]

                if df_subset.shape[0] == 0:
                    continue

                X_pert = np.array(df_subset['pert_emb'].tolist())
                X_pred = np.array(df_subset['pred_emb'].tolist())


                mmd = __compute_mmd(X_pert, X_pred)
                wasserstein = __compute_wasserstein(X_pert, X_pred)


                results_per_cell_mmd.append(mmd)
                results_per_cell_wasserstein.append(wasserstein)


        results_mmd[cell_type] = results_per_cell_mmd
        results_wasserstein[cell_type] = results_per_cell_wasserstein

    return results_mmd, results_wasserstein