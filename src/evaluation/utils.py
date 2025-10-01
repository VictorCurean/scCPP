import anndata as ad
import pandas as pd
import numpy as np
import scanpy as sc

def __clamp_negative_values(df, col_name):
    df = df.copy()  # avoid modifying the original DataFrame
    df[col_name] = df[col_name].apply(
        lambda arr: [x if x > 0 else 0 for x in arr]
    )
    return df

def get_results__fc(results, adata_control, gene_names, method='wilcoxon'):
    """
    Get fold changes scoring
    """

    #clamp negative values in predicted values
    results = __clamp_negative_values(results, 'pred_emb')


    # create adata control
    adata_ctrl = ad.AnnData(adata_control.X)
    adata_ctrl.obs['cell_type'] = list(adata_control.obs['cell_type'])
    adata_ctrl.obs['compound'] = list(adata_control.obs['product_name'])
    adata_ctrl.obs['dose'] = 0
    adata_ctrl.obs['condition'] = 'control'

    # create adata perturbed
    adata_pert = ad.AnnData(np.array(results['pert_emb'].tolist()))
    adata_pert.obs['cell_type'] = list(results['cell_type'])
    adata_pert.obs['compound'] = list(results['compound'])
    adata_pert.obs['dose'] = list(results['dose'])
    adata_pert.obs['condition'] = 'perturbed'

    # create adata pred
    adata_pred = ad.AnnData(np.array(results['pred_emb'].tolist()))
    adata_pred.obs['cell_type'] = list(results['cell_type'])
    adata_pred.obs['compound'] = list(results['compound'])
    adata_pred.obs['dose'] = list(results['dose'])
    adata_pred.obs['condition'] = 'predicted'

    adata = ad.concat([adata_ctrl, adata_pert, adata_pred])
    adata.obs_names_make_unique()
    adata.var_names = gene_names

    logFC_results = []

    # calculate FC

    for (cell_type, compound, dose) in adata.obs[['cell_type', 'compound', 'dose']].drop_duplicates().itertuples(
            index=False):

        if compound == "Vehicle":
            continue

        if dose == 0:
            continue

        # subset adata for the specific group
        subset1 = adata[(adata.obs['cell_type'] == cell_type) & (adata.obs['compound'] == compound) & (
                adata.obs['dose'] == dose)].copy()
        subset2 = adata[(adata.obs['cell_type'] == cell_type) & (adata.obs['compound'] == "Vehicle")].copy()

        subset = ad.concat([subset1, subset2])

        if len(list(subset.obs.condition.unique())) != 3:
            continue

        assert len(list(subset.obs['condition'].unique())) == 3

        sc.tl.rank_genes_groups(subset, groupby='condition', reference='control', method=method)

        df_perturbed = sc.get.rank_genes_groups_df(subset, "perturbed")
        df_predicted = sc.get.rank_genes_groups_df(subset, "predicted")

        df_perturbed = df_perturbed.sort_values(by="names")
        df_predicted = df_predicted.sort_values(by="names")

        logFC_pert = df_perturbed['logfoldchanges'].tolist()
        logFC_pred = df_predicted['logfoldchanges'].tolist()

        names_pert = df_perturbed['names'].tolist()
        names_pred = df_predicted['names'].tolist()

        pvals_pert = df_perturbed['pvals_adj'].tolist()
        pvals_pred = df_predicted['pvals_adj'].tolist()

        assert names_pert == names_pred

        logFC_results.append({
            'cell_type': cell_type,
            'compound': compound,
            'dose': dose,
            'logFC_pert': logFC_pert,
            'logFC_pred': logFC_pred,
            'names': names_pert,
            'pvals_pert': pvals_pert,
            'pvals_pred': pvals_pred
        })

    return pd.DataFrame(logFC_results)


def get_results__logp(results, adata_control, gene_names, method='t-test-overestim-var', min_p=1e-24):
    # clamp negatives in predicted values (unchanged)
    results = __clamp_negative_values(results, 'pred_emb')

    # control
    adata_ctrl = ad.AnnData(adata_control.X)
    adata_ctrl.obs['cell_type'] = list(adata_control.obs['cell_type'])
    adata_ctrl.obs['compound']  = list(adata_control.obs['product_name'])
    adata_ctrl.obs['dose']      = 0
    adata_ctrl.obs['condition'] = 'control'

    # perturbed
    adata_pert = ad.AnnData(np.array(results['pert_emb'].tolist()))
    adata_pert.obs['cell_type'] = list(results['cell_type'])
    adata_pert.obs['compound']  = list(results['compound'])
    adata_pert.obs['dose']      = list(results['dose'])
    adata_pert.obs['condition'] = 'perturbed'

    # predicted
    adata_pred = ad.AnnData(np.array(results['pred_emb'].tolist()))
    adata_pred.obs['cell_type'] = list(results['cell_type'])
    adata_pred.obs['compound']  = list(results['compound'])
    adata_pred.obs['dose']      = list(results['dose'])
    adata_pred.obs['condition'] = 'predicted'

    # concat
    adata = ad.concat([adata_ctrl, adata_pert, adata_pred])
    adata.obs_names_make_unique()
    adata.var_names = gene_names

    out = []

    for (cell_type, compound, dose) in adata.obs[['cell_type','compound','dose']].drop_duplicates().itertuples(index=False):
        if compound == "Vehicle" or dose == 0:
            continue

        subset1 = adata[(adata.obs['cell_type']==cell_type) & (adata.obs['compound']==compound) & (adata.obs['dose']==dose)].copy()
        subset2 = adata[(adata.obs['cell_type']==cell_type) & (adata.obs['compound']=="Vehicle")].copy()
        subset  = ad.concat([subset1, subset2])

        if len(list(subset.obs.condition.unique())) != 3:
            continue
        assert len(list(subset.obs['condition'].unique())) == 3

        sc.tl.rank_genes_groups(subset, groupby='condition', reference='control', method=method)

        df_pert = sc.get.rank_genes_groups_df(subset, group='perturbed').sort_values('names')
        df_pred = sc.get.rank_genes_groups_df(subset, group='predicted').sort_values('names')

        # align genes
        names_pert = df_pert['names'].tolist()
        names_pred = df_pred['names'].tolist()
        assert names_pert == names_pred

        # signed -ln(p) using RAW p-values (not adjusted), sign from test statistic 'scores'
        logp_pert = np.sign(df_pert['scores'].to_numpy()) * (-np.log(df_pert['pvals'].to_numpy() + min_p))
        logp_pred = np.sign(df_pred['scores'].to_numpy()) * (-np.log(df_pred['pvals'].to_numpy() + min_p))

        out.append({
            'cell_type': cell_type,
            'compound': compound,
            'dose': dose,
            'logp_pert': logp_pert,
            'logp_pred': logp_pred,
            'names': names_pert
        })

    return pd.DataFrame(out)


def get_results__delta_deg(results, adata_control, gene_names, method='wilcoxon'):
    """
    For each (cell_type, compound, dose):
      - run sc.tl.rank_genes_groups(groupby='condition', reference='control', method=method)
      - take pvals_adj for the 'perturbed' group (significance)
      - compute Δ_pert = mean(perturbed) - mean(control)
                Δ_pred = mean(predicted) - mean(control)
      - return pvals aligned to `gene_names` so selection is consistent with Δ vectors
    """
    results = __clamp_negative_values(results, 'pred_emb')

    # control
    adata_ctrl = ad.AnnData(adata_control.X)
    adata_ctrl.obs['cell_type'] = list(adata_control.obs['cell_type'])
    adata_ctrl.obs['compound']  = list(adata_control.obs['product_name'])
    adata_ctrl.obs['dose']      = 0
    adata_ctrl.obs['condition'] = 'control'

    # observed perturbed
    adata_pert = ad.AnnData(np.array(results['pert_emb'].tolist()))
    adata_pert.obs['cell_type'] = list(results['cell_type'])
    adata_pert.obs['compound']  = list(results['compound'])
    adata_pert.obs['dose']      = list(results['dose'])
    adata_pert.obs['condition'] = 'perturbed'

    # predicted
    adata_pred = ad.AnnData(np.array(results['pred_emb'].tolist()))
    adata_pred.obs['cell_type'] = list(results['cell_type'])
    adata_pred.obs['compound']  = list(results['compound'])
    adata_pred.obs['dose']      = list(results['dose'])
    adata_pred.obs['condition'] = 'predicted'

    # combine just to share var_names and subset
    adata = ad.concat([adata_ctrl, adata_pert, adata_pred])
    adata.obs_names_make_unique()
    adata.var_names = gene_names

    out = []

    for (cell_type, compound, dose) in adata.obs[['cell_type','compound','dose']].drop_duplicates().itertuples(index=False):
        if compound == "Vehicle" or dose == 0:
            continue

        subset1 = adata[(adata.obs['cell_type']==cell_type) &
                        (adata.obs['compound']==compound) &
                        (adata.obs['dose']==dose)].copy()
        subset2 = adata[(adata.obs['cell_type']==cell_type) &
                        (adata.obs['compound']=="Vehicle")].copy()
        subset  = ad.concat([subset1, subset2])

        if len(list(subset.obs['condition'].unique())) != 3:
            continue
        assert len(list(subset.obs['condition'].unique())) == 3

        # DEG testing (perturbed vs control); keep default n_genes
        sc.tl.rank_genes_groups(subset, groupby='condition', reference='control', method=method)

        # take pvals_adj for the true 'perturbed' group and align to `gene_names`
        df_pert = sc.get.rank_genes_groups_df(subset, group='perturbed')
        pval_map = dict(zip(df_pert['names'].tolist(), df_pert['pvals_adj'].tolist()))
        pvals_aligned = np.array([pval_map[g] for g in gene_names])  # assumes all present

        # means per gene
        def _mean_row(mat):
            m = mat.mean(axis=0)
            return m.A1 if hasattr(m, "A1") else np.asarray(m).ravel()

        ctrl_mask = subset.obs['condition'] == 'control'
        pert_mask = subset.obs['condition'] == 'perturbed'
        pred_mask = subset.obs['condition'] == 'predicted'

        mean_ctrl = _mean_row(subset[ctrl_mask].X)
        mean_pert = _mean_row(subset[pert_mask].X)
        mean_pred = _mean_row(subset[pred_mask].X)

        delta_pert = mean_pert - mean_ctrl
        delta_pred = mean_pred - mean_ctrl

        out.append({
            'cell_type':  cell_type,
            'compound':   compound,
            'dose':       dose,
            'delta_pert': delta_pert,
            'delta_pred': delta_pred,
            'names':      list(gene_names),
            'pvals_pert': pvals_aligned
        })

    return pd.DataFrame(out)

