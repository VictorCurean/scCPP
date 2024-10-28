import anndata as ad
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
import pandas as pd


def __train_test_split(X, y, test_size=0.2, random_seed=None):
    """
    Splits input data X and response variable y into training and testing sets.

    Parameters:
        X (np.array): Input features.
        y (list): Response variable.
        test_size (float): Proportion of the dataset to include in the test split.
        random_seed (int, optional): Seed for reproducibility.

    Returns:
        X_train, X_test, y_train, y_test: Split datasets for training and testing.
    """
    # Set random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)

    # Calculate the number of test samples
    total_samples = len(X)
    test_samples = int(total_samples * test_size)

    # Shuffle the indices
    indices = np.arange(total_samples)
    np.random.shuffle(indices)

    # Split indices for training and testing
    test_indices = indices[:test_samples]
    train_indices = indices[test_samples:]

    # Split the data and response variable
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = [y[i] for i in train_indices], [y[i] for i in test_indices]

    return X_train, X_test, y_train, y_test

def calculate_classification_stats(adata, savefile=None):
    results = list()
    control_A549 = adata[(adata.obs['cell_type'] == "A549") & (adata.obs['product_name'] == "Vehicle")].X
    control_K562 = adata[(adata.obs['cell_type'] == "K562") & (adata.obs['product_name'] == "Vehicle")].X
    control_MCF7 = adata[(adata.obs['cell_type'] == "MCF7") & (adata.obs['product_name'] == "Vehicle")].X

    len_A549 = adata[(adata.obs['cell_type'] == "A549") & (adata.obs['product_name'] == "Vehicle")].n_obs
    len_K562 = adata[(adata.obs['cell_type'] == "K562") & (adata.obs['product_name'] == "Vehicle")].n_obs
    len_MCF7 = adata[(adata.obs['cell_type'] == "MCF7") & (adata.obs['product_name'] == "Vehicle")].n_obs

    for compound in tqdm(list(adata.obs['product_name'].unique())):
        if compound == "Vehicle":
            continue

        for cell_type in list(adata.obs['cell_type'].unique()):
            for dose in list(adata.obs['dose'].unique()):
                if dose == 0.0:
                    continue

                adata_subset = adata[
                    (adata.obs['product_name'] == compound) &
                    (adata.obs['cell_type'] == cell_type) &
                    (adata.obs['dose'] == dose)
                ]

                X_control = None

                if cell_type == "A549":
                    X_control = control_A549
                    y_control = [0 for _ in range(len_A549)]
                elif cell_type == "K562":
                    X_control = control_K562
                    y_control  = [0 for _ in range(len_K562)]
                elif cell_type == "MCF7":
                    X_control = control_MCF7
                    y_control = [0 for _ in range(len_MCF7)]
                else:
                    raise RuntimeError("Invalid Cell Type")

                X_treated = adata_subset.X
                y_treated = [1 for _ in range(adata_subset.n_obs)]

                #hold 20% of control and treated samples for testing

                X_train_treated, X_test_treated, y_train_treated, y_test_treated = __train_test_split(X_treated, y_treated)
                X_train_control, X_test_control, y_train_control, y_test_control = __train_test_split(X_control, y_control)


                X_train = np.vstack((X_train_treated, X_train_control))
                y_train = np.array(y_train_treated + y_train_control)

                X_test = np.vstack((X_test_treated, X_test_control))
                y_test = np.array(y_test_treated + y_test_control)

                log_reg_cv = LogisticRegression(class_weight='balanced')

                log_reg_cv.fit(X_train, y_train)

                y_pred = log_reg_cv.predict(X_test)

                report_dict = classification_report(y_test, y_pred, output_dict=True)['1']
                report_dict['compound'] = compound
                report_dict['dose'] = dose
                report_dict['cell_type'] = cell_type
                report_dict['sample_size_treated'] = len(y_treated)
                report_dict['sample_size_control'] = len(y_control)
                report_dict['sample_size_total'] = len(y_treated) + len(y_control)

                print(report_dict)

                results.append(report_dict)

    if savefile is not None:
        results = pd.DataFrame(results)
        results.to_csv(savefile, index=False)


if __name__ == "__main__":
    ROOT = 'C:\\Users\\curea\\Documents\\bioFM for drug discovery\\dege-fm\\'
    adata = ad.read_h5ad(ROOT + "data\\adata_preprocessed.h5ad")
    calculate_classification_stats(adata, ROOT + "results\\logreg_results_allcovariates.csv")









