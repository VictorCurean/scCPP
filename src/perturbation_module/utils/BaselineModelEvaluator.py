from ModelEvaluator import ModelEvaluator

class BaselineModelEvaluator(ModelEvaluator):
    self.ROOT = 'C:\\Users\\curea\\Documents\\bioFM for drug discovery\\dege-fm\\'


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
        self.model = ConditionalFeedForwardNN(config)
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.model = model.to(self.device)

    def __read_data(self):
        with open(ROOT + "data\\sciplex\\drugs_train_list.txt", "r") as f:
            drugs_train = [line.strip() for line in f]

        with open(ROOT + "data\\sciplex\\drugs_validation_list.txt", "r") as f:
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

        print("Loading zhao test dataset ...")
        zhao_dataset = ZhaoDatasetBaseline(config['dataset_params']['zhao_adata_path'])
        self.zhao_loader = DataLoader(zhao_dataset, batch_size=config['train_params']['batch_size'], shuffle=True, num_workers=0)


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

        self.trained_model.eval()
        res = list()

        with torch.no_grad():
            for inputs, targets, meta in self.sciplex_loader_test:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.trained_model(inputs)
                res.append({"input": inputs, "targets": targets, "predicted": outputs, "meta": meta})

            print(f"Test Loss: {avg_loss}")

        self.validation_results_sciplex = res

    def __validate_zhao(self):
        self.trained_model.eval()
        res = list()

        with torch.no_grad():
            for inputs, targets, meta in self.zhao_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.trained_model(inputs)
                res.append({"input": inputs, "targets": targets, "predicted": outputs, "meta": meta})

        self.validation_results_zhao = res

    def model_report(self):
        pass

