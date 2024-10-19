import torch
import pandas as pd
from coati.models.io.coati import load_e3gnn_smiles_clip_e2e
from coati.generative.coati_purifications import embed_smiles
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
from tqdm import tqdm

#load sm annotations
adata_obs = pd.read_csv("/raw-data/sciplex_obs_annot_with_smiles_manual.tsv")
smiles_set = list(set(adata_obs['smiles']))


# load pretrained sm encoder model
encoder, tokenizer = load_e3gnn_smiles_clip_e2e(
    freeze=True,
    device=torch.device("cuda:0"),
    # model parameters to load.
    doc_url="s3://terray-public/models/barlow_closed.pkl",
)

smiles_embeddings = dict()
adata_obs['sm_embedding'] = None

exclude_list = [""]

for sm in smiles_set:
    if type(sm) == str:
        try:
            #remove salts and steroechemistry
            mol = Chem.MolFromSmiles(sm)
            Chem.MolToSmiles(mol)

            remover = SaltRemover()
            stripped = remover.StripMol(mol)

            Chem.RemoveStereochemistry(stripped)
            smiles = Chem.MolToSmiles(stripped)
            smiles = Chem.CanonSmiles(smiles)
            vector = embed_smiles(smiles, encoder, tokenizer)
            smiles_embeddings[sm] = vector
        except UnboundLocalError:
            smiles_embeddings[sm] = "MIXTURE_OF_COMPOUNDS"

print("Assigning embeddings ... ")
for sm in tqdm(list(set(list(adata_obs['smiles'])))):

    try:
        # smiles has embedding
        if torch.is_tensor(smiles_embeddings[sm]):
            emb = smiles_embeddings[sm].tolist()
            adata_obs.loc[adata_obs['smiles'] == sm, 'sm_embedding'] = adata_obs.loc[adata_obs['smiles'] == sm, 'smiles'].apply(lambda x: emb)

        # mixture of multiple compounds
        elif smiles_embeddings[sm] == "MIXTURE_OF_COMPOUNDS":
            adata_obs.loc[adata_obs['smiles'] == sm, 'sm_embedding'] = "MIXTURE_OF_COMPOUNDS"

    except KeyError:
        #vehicle / control
        if type(sm) != str:
            adata_obs.loc[adata_obs['smiles'] == sm, 'sm_embedding'] = "VEHICLE"
        else:
            raise RuntimeError("CASE NOT HANDLED")


adata_obs.to_csv("C:\\Users\\curea\\Documents\\bioFM for drug discovery\\dege-fm\\raw-data\\sciplex_obs_annot_with_smiles_manual_plus_embeddings.tsv", sep="\t")

a = 1
