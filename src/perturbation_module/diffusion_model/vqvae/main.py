from src.perturbation_module.diffusion_model.dataset import SciplexDataset
import yaml

def train(config_file):
    # Read the config file
    with open(config_file, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    # read dataset
    dataset_params = config['dataset_params']
    embedding_file = dataset_params['embeddings']
    annotations = dataset_params['annotations']
    label_column = dataset_params['label_column']

    dataset = SciplexDataset(embedding_file, annotations, label_column)

    a = 0




config_file = "C:\\Users\\curea\\Documents\\bioFM for drug discovery\\dege-fm\\config\\sciplex_uce_emb.yaml"

train(config_file)