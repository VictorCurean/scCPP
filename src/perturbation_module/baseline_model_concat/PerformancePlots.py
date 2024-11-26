import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_curve, auc, precision_score, recall_score
import seaborn as sns

def plot_css(adata):
    """
    Compare r-squared values between the null model and predictions
        ** compare r-squared between control-target and predicted-target pairs **
    TODO: this assumes that the order of pairs is preserved
    """

    css_model = list()
    css_null = list()

    for condition in tqdm(list(adata.obs['condition'].unique())):
        adata_subset = adata[adata.obs['condition'] == condition]

        X_control = adata_subset[adata_subset.obs['data_type'] == "input"].X
        X_target = adata_subset[adata_subset.obs['data_type'] == "target"].X
        X_predicted = adata_subset[adata_subset.obs['data_type'] == "predicted"].X

        for i in range(X_control.shape[0]):
            x_control = X_control[i,]
            x_target = X_target[i,]
            x_predicted = X_predicted[i,]

            cosine_similarity_model = np.dot(x_target, x_predicted) / (
                        np.linalg.norm(x_target) * np.linalg.norm(x_predicted))

            cosine_similarity_null = np.dot(x_target, x_control) / (np.linalg.norm(x_target) * np.linalg.norm(x_control))

            css_model.append(cosine_similarity_model)
            css_null.append(cosine_similarity_null)

    data = pd.DataFrame({
        "css": css_model + css_null,
        "model": ["model"] * len(css_model) + ["null"] * len(css_null)
    })

    plt.figure(figsize=(8, 6))  # Set the figure size
    sns.boxplot(x='model', y='css', data=data)  # Create the boxplot
    plt.xlabel('Predictor')  # X-axis label
    plt.ylabel('CSS')  # Y-axis label
    plt.show()  # Show the plot

def plot_mse(adata):
    """
    Compare r-squared values between the null model and predictions
        ** compare r-squared between control-target and predicted-target pairs **
    TODO: this assumes that the order of pairs is preserved
    """
    mse_model_list = list()
    mse_null_list = list()


    for condition in tqdm(list(adata.obs['condition'].unique())):
        adata_subset = adata[adata.obs['condition'] == condition]

        X_control = adata_subset[adata_subset.obs['data_type'] == "input"].X
        X_target = adata_subset[adata_subset.obs['data_type'] == "target"].X
        X_predicted = adata_subset[adata_subset.obs['data_type'] == "predicted"].X


        for i in range(X_control.shape[0]):
            x_control = X_control[i,]
            x_target = X_target[i,]
            x_predicted = X_predicted[i,]


            mse_model = np.mean((x_target - x_predicted) ** 2)

            mse_null = np.mean((x_target - x_control) ** 2)

            mse_model_list.append(mse_model)
            mse_null_list.append(mse_null)

    data = pd.DataFrame({
        "mse": mse_model_list + mse_null_list,
        "model": ["model"] * len(mse_model_list) + ["null"] * len(mse_null_list)
    })

    plt.figure(figsize=(8, 6))  # Set the figure size
    sns.boxplot(x='model', y='mse', data=data)  # Create the boxplot
    plt.xlabel('Predictor')  # X-axis label
    plt.ylabel('MSE')  # Y-axis label
    plt.show()  # Show the plot


def plot_r2(adata):
    """
    Compare R-squared values between the null model and predictions.
    """
    r2_model_list = list()
    r2_null_list = list()

    for condition in tqdm(list(adata.obs['condition'].unique())):
        adata_subset = adata[adata.obs['condition'] == condition]

        X_control = adata_subset[adata_subset.obs['data_type'] == "input"].X
        X_target = adata_subset[adata_subset.obs['data_type'] == "target"].X
        X_predicted = adata_subset[adata_subset.obs['data_type'] == "predicted"].X

        for i in range(X_control.shape[0]):
            x_control = X_control[i,]
            x_target = X_target[i,]
            x_predicted = X_predicted[i,]

            # Compute SS_tot and SS_res for model and null
            ss_tot = np.sum((x_target - np.mean(x_target)) ** 2)
            ss_res_model = np.sum((x_target - x_predicted) ** 2)
            ss_res_null = np.sum((x_target - x_control) ** 2)

            # Compute R-squared
            r2_model = 1 - (ss_res_model / ss_tot)
            r2_null = 1 - (ss_res_null / ss_tot)

            r2_model_list.append(r2_model)
            r2_null_list.append(r2_null)

    # Prepare data for plotting
    data = pd.DataFrame({
        "R-squared": r2_model_list + r2_null_list,
        "model": ["model"] * len(r2_model_list) + ["null"] * len(r2_null_list)
    })

    # Plot
    plt.figure(figsize=(8, 6))  # Set the figure size
    sns.boxplot(x='model', y='R-squared', data=data)  # Create the boxplot
    plt.xlabel('Predictor')  # X-axis label
    plt.ylabel('R-squared')  # Y-axis label
    plt.title('Comparison of R-squared Values')  # Title
    plt.show()  # Show the plot


def evaluate_classifier_logreg(adata_zhao_original, trained_model, criterion):
    for sample in adata_zhao_original.obs['sample'].unique():

        ad = adata_zhao_original[adata_zhao_original.obs['sample'] == sample]

        perturbations = list(ad.obs['perturbation'].unique())
        print(perturbations)
        #build control, perturbed samples
        X_control = ad[ad.obs['perturbation'] == "control"].obsm['X_uce']
        y_control = [0 for _ in range(X_control.shape[0])]

        #for each individual treatment, build a classifier
        for prt in perturbations:
            if prt == 'control':
                continue

            X_prt = ad[ad.obs['perturbation'] == prt].obsm['X_uce']
            y_prt = [1 for _ in range(X_prt.shape[0])]

            #get SM embeddings
            sm_embeddings = list(ad[ad.obs['perturbation'] == prt].obs['sm_emb'])
            dose_values = list(ad[ad.obs['perturbation'] == prt].obs['dose_value'])
            dose_units = list(ad[ad.obs['perturbation'] == prt].obs['dose_unit'])

            X = np.vstack((X_control, X_prt))
            y = np.array(y_control + y_prt)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1702)

            classifier = LogisticRegression(random_state=1702, max_iter=1000, class_weight='balanced')
            classifier.fit(X_train, y_train)

            y_pred = classifier.predict(X_test)
            y_pred_proba = classifier.predict_proba(X_test)[:, 1]

            # Calculate performance metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)

            print("Classification Report:\n", classification_report(y_test, y_pred, target_names=["Control", "Perturbed"]))

            # Calculate confusion matrix and plot it
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(10, 4))

            # Confusion Matrix Plot
            plt.subplot(1, 2, 1)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Control', 'Treatment'], yticklabels=['Control', 'Treatment'])
            plt.title(f"Confusion Matrix - {sample} ({prt})")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")

            # Precision-Recall Curve Plot
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            pr_auc = auc(recall, precision)
            plt.subplot(1, 2, 2)
            plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.3f}")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"Precision-Recall Curve - {sample} ({prt})")
            plt.legend()

            # Plot control + drug predictions
            plt.subplot(1, 2, 3)

            #TODO plot model predictions
            X_predictions = __make_predictions(X_prt, sm_embeddings, dose_values, dose_units, trained_model, criterion)

            # Show both plots together for each treatment
            plt.tight_layout()
            plt.show()


def __make_predictions(X_prt, sm_embeddings, dose_values, dose_units, trained_model, criterion):
    pass