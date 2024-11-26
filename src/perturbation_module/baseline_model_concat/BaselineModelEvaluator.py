import math
import yaml
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
import pickle as pkl
import seaborn as sns
import matplotlib.pyplot as plt
import anndata as ad

from model import ConditionalFeedForwardNN
from dataset import SciplexDatasetBaseline
from dataset_zhao import ZhaoDatasetBaseline

from ModelEvaluator import ModelEvaluator
from PerformancePlots import plot_css, plot_mse, plot_r2
from src.VARS import VARS


class BaselineModelEvaluator(ModelEvaluator):

    def __init__(self):
        self.ROOT = VARS.ROOT
        self.initialize()

    def initialize(self):
        #load config file
        self.__read_config()

        #prepare model
        self.__prepare_model()

        #read data
        self.__read_data()


    def __read_config(self):
        # load config file
        print("Reading config file ...")
        with open(self.ROOT + "config\\baseline.yaml", 'r') as file:
            try:
                self.config = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)
                raise RuntimeError(exc)

    def __prepare_model(self):
        print("Preparing model ...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ConditionalFeedForwardNN(self.config)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.model = self.model.to(self.device)

    def __read_data(self):
        with open(self.ROOT + "data\\sciplex\\drugs_train_list.txt", "r") as f:
            drugs_train = [line.strip() for line in f]

        with open(self.ROOT + "data\\sciplex\\drugs_validation_list.txt", "r") as f:
            drugs_validation = [line.strip() for line in f]

        print("Loading train dataset ...")
        sciplex_dataset = SciplexDatasetBaseline(self.config['dataset_params']['sciplex_adata_path'],
                                                      drugs_train)
        self.sciplex_loader = DataLoader(sciplex_dataset, batch_size=self.config['train_params']['batch_size'],
                                         shuffle=True,
                                         num_workers=0)

        print("Loading sciplex test dataset ...")
        sciplex_dataset_test = SciplexDatasetBaseline(self.config['dataset_params']['sciplex_adata_path'], drugs_validation)
        self.sciplex_loader_test = DataLoader(sciplex_dataset_test, batch_size=self.config['train_params']['batch_size'],
                                         shuffle=True, num_workers=0)

        # print("Loading zhao test dataset ...")
        # zhao_dataset = ZhaoDatasetBaseline(self.config['dataset_params']['zhao_adata_path'])
        # self.zhao_loader = DataLoader(zhao_dataset, batch_size=self.config['train_params']['batch_size'], shuffle=True, num_workers=0)


    def train(self):
        print("Begin training ... ")
        self.model.train()  # Set the model to training mode
        losses = list()

        num_epochs = self.config['train_params']['num_epochs']

        for epoch in range(num_epochs):

            for input, output_actual, meta in tqdm(self.sciplex_loader):
                # Move tensors to the specified device
                input = input.to(self.device)
                output_actual = output_actual.to(self.device)

                self.optimizer.zero_grad()

                output = self.model(input)

                loss = self.criterion(output, output_actual)

                loss.backward()

                self.optimizer.step()

                losses.append(loss.item())

        self.losses_train = losses
        self.trained_model = self.model

        print("Training completed ...")

    def __validate_sciplex(self):

        print("Inferring on sciplex test dataset ...")

        self.trained_model.eval()
        validation_results_sciplex = list()

        with torch.no_grad():
            for inputs, targets, meta in self.sciplex_loader_test:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.trained_model(inputs)
                validation_results_sciplex.append({"input": inputs, "targets": targets, "predicted": outputs, "meta": meta})

        targets = list()
        predicted = list()
        input = list()
        meta = list()

        for x in tqdm(validation_results_sciplex):

            assert x['targets'].shape == x['predicted'].shape

            for i in range(x['targets'].shape[0]):
                targets.append(x['targets'][i].cpu().numpy())

            for i in range(x['predicted'].shape[0]):
                predicted.append(x['predicted'][i].cpu().numpy())

            for i in range(x['input'].shape[0]):
                input.append(x['input'][i].cpu().numpy()[:1280])

            compound_list = x['meta']['compound']
            dose_list = x['meta']['dose'].tolist()
            cell_type = x['meta']['cell_type']

            for i in range(len(compound_list)):
                meta.append(compound_list[i] + "_" + str(dose_list[i]) + "_" + cell_type[i])

        df_targets = pd.DataFrame(targets)
        df_predicted = pd.DataFrame(predicted)
        df_input = pd.DataFrame(input)

        df_targets['data_type'] = "target"
        df_predicted['data_type'] = "predicted"
        df_input['data_type'] = "input"

        df_targets['condition'] = meta
        df_predicted['condition'] = meta
        df_input['condition'] = meta

        del meta

        df = pd.concat([df_targets, df_predicted, df_input], axis=0, ignore_index=True)

        del df_targets
        del df_predicted
        del df_input

        X = df.iloc[:, :1280].values
        obs = df[['condition', 'data_type', 'cell_type']]

        del df

        self.adata_zhao_sciplex= ad.AnnData(X=X, obs=obs)

        del X
        del obs



    def __validate_zhao(self):
        print("Inferring on sciplex test dataset ...")

        self.trained_model.eval()

        adata_zhao = ad.read_h5ad(self.ROOT + "data\\zhao\\zhao_preprocessed.h5ad")

        for sample in adata.obs['sample'].unique():
            ad = adata[adata.obs['sample'] == sample]

            perturbations = list(ad.obs['perturbation'].unique())

            # build control, perturbed samples
            X_control = ad[ad.obs['perturbation'] == "control"].obsm['X_uce']
            y_control = [0 for _ in range(X_control.shape[0])]

            # for each individual treatment, build a classifier
            for prt in perturbations:
                if prt == 'control':
                    continue

                classifier = self.__per_sample_classifier(ad, sample, prt)

                self.__per_sample_prediction(ad, sample, prt, classifier)


    def __per_sample_classifier(self, ad, sample, perturbation):

            X_prt = ad[ad.obs['perturbation'] == prt].obsm['X_uce']
            y_prt = [1 for _ in range(X_prt.shape[0])]

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

            print("Classification Report:\n",
                  classification_report(y_test, y_pred, target_names=["Control", "Perturbed"]))

            # Calculate confusion matrix and plot it
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(10, 4))

            # Confusion Matrix Plot
            plt.subplot(1, 2, 1)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Control', 'Treatment'],
                        yticklabels=['Control', 'Treatment'])
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

            # Show both plots together for each treatment
            plt.tight_layout()
            plt.show()

            return classifier

    def __per_sample_prediction(self, ad, sample, prt, classifier):

        adata = ad[ad.obs['perturbation'] == prt]

        zhao_dataset = ZhaoDatasetBaseline(adata)
        zhao_loader = DataLoader(zhao_loader, batch_size=32)

        validation_results_zhao = list()


        with torch.no_grad():
            for inputs, targets, meta in zhao_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.trained_model(inputs)
                validation_results_zhao.append(
                    {"input": inputs, "targets": targets, "predicted": outputs, "meta": meta})

        targets = list()
        predicted = list()
        input = list()
        meta = list()

        for x in tqdm(validation_results_zhao):

            assert x['targets'].shape == x['predicted'].shape

            for i in range(x['targets'].shape[0]):
                targets.append(x['targets'][i].cpu().numpy())

            for i in range(x['predicted'].shape[0]):
                predicted.append(x['predicted'][i].cpu().numpy())

            for i in range(x['input'].shape[0]):
                input.append(x['input'][i].cpu().numpy()[:1280])

            compound_list = x['meta']['compound']
            dose_list = x['meta']['dose'].tolist()
            sample = x['meta']['sample'].tolist()

            for i in range(len(compound_list)):
                meta.append(compound_list[i] + "_" + str(dose_list[i]) + "_" + cell_type[i])

        df_targets = pd.DataFrame(targets)
        df_predicted = pd.DataFrame(predicted)
        df_input = pd.DataFrame(input)

        df_targets['data_type'] = "target"
        df_predicted['data_type'] = "predicted"
        df_input['data_type'] = "input"

        df_targets['condition'] = meta
        df_predicted['condition'] = meta
        df_input['condition'] = meta

        df = pd.concat([df_targets, df_predicted, df_input], axis=0, ignore_index=True)

        X = df.iloc[:, :1280].values
        obs = df[['condition', 'data_type']]

        adata_zhao_results = ad.AnnData(X=X, obs=obs)


        plot_css(self.adata_zhao_results)
        plot_mse(self.adata_zhao_results)
        plot_r2(self.adata_zhao_results)

        #see classifier performance
        X_model = adata_zhao_results[adata_zhao_results.obs['data_type'] == "predicted"]
        pred_y = classifier.predict(X_model)

        data_barplot = pd.DataFrame({
            'Category': ['Predicted Treated' if y == 1 else 'Predicted Control' for y in pred_y]
        })

        # Count the occurrences of each category
        counts = data_barplot['Category'].value_counts().reset_index()
        counts.columns = ['Category', 'Count']

        # Plot the bar chart
        sns.barplot(x='Category', y='Count', data=counts, palette='pastel')
        plt.title('Distribution of Predicted Categories')
        plt.ylabel('Count')
        plt.xlabel('Category')
        plt.show()



    def model_report_zhao(self):
        self.__validate_zhao()
        plot_css(self.adata_zhao_results)
        plot_mse(self.adata_zhao_results)
        plot_r2(self.adata_zhao_results)

    def model_report_sciplex(self):
        self.__validate_sciplex()
        plot_css(self.adata_zhao_results)
        plot_mse(self.adata_zhao_results)
        plot_r2(self.adata_zhao_results)


