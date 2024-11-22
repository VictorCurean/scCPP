from ModelEvaluator import ModelEvaluator

class BaselineModelEvaluator(ModelEvaluator):
    ROOT = 'C:\\Users\\curea\\Documents\\bioFM for drug discovery\\dege-fm\\'


    def initialize(self):
        #load config file
        with open(ROOT + "config\\baseline.yaml", 'r') as file:
            try:
                self.config = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)
                raise RuntimeError(exc)

        #prepare model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = ConditionalFeedForwardNN(config)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()



    def train_loader(self):
        pass

    def validation_loader(self):
        pass

    def train(self):
        pass

    def validate(self):
        pass

    def model_report(self):
        pass

