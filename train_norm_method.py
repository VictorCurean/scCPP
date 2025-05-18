import anndata as ad
import pickle as pkl

from src.evaluator.MLP_baseline_evaluator import get_models_results
from src.evaluator.evaluator_utils import l2_loss
from src.utils import get_model_stats

def train_different_normalization(adata_path, run_name, res_savename, stats_savename):
    DRUG_ENCODING_NAME = "fmfp"
    DRUG_ENCODING_SIZE = 1024
    N_TRIALS = 50
    SCHEDULER_MODE = 'min'

    with open("./data/drug_splits/train_drugs_rand.pkl", 'rb') as f:
        drugs_train_rand = pkl.load(f)

    with open("./data/drug_splits/val_drugs_rand.pkl", 'rb') as f:
        drugs_val_rand = pkl.load(f)

    with open("./data/drug_splits/test_drugs_rand.pkl", 'rb') as f:
        drugs_test_rand = pkl.load(f)

    drug_splits = dict()
    drug_splits['train'] = drugs_train_rand
    drug_splits['valid'] = drugs_val_rand
    drug_splits['test'] = drugs_test_rand

    adata = ad.read_h5ad(adata_path)

    get_models_results(drug_splits=drug_splits,
                          loss_function=l2_loss,
                          adata=adata,
                          input_dim=1869,
                          output_dim=1869,
                          drug_rep_name=DRUG_ENCODING_NAME,
                          drug_emb_size=DRUG_ENCODING_SIZE,
                          n_trials=N_TRIALS,
                          scheduler_mode=SCHEDULER_MODE,
                          run_name=run_name,
                          save_path=res_savename
                      )

    with open(res_savename, 'rb') as f:
        res_raw = pkl.load(f)

    adata_control = adata[adata.obs.product_name == 'Vehicle'].copy()
    gene_names = list(adata_control.var_names)
    raw_stats = get_model_stats(res_raw, adata_control, gene_names, run_name)

    with open(stats_savename, 'wb') as f:
        pkl.dump(raw_stats, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate model with different normalization strategies.")
    parser.add_argument("adata_path", type=str, help="Path to the .h5ad data file.")
    parser.add_argument("run_name", type=str, help="Name for the run")
    parser.add_argument("res_savename", type=str, help="Path to save raw results (pickle).")
    parser.add_argument("stats_savename", type=str, help="Path to save statistics (pickle).")

    args = parser.parse_args()

    train_different_normalization(
        adata_path=args.adata_path,
        run_name=args.run_name,
        res_savename=args.res_savename,
        stats_savename=args.stats_savename
    )




