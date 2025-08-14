
from src.evaluation.utils import get_results__fc
from src.evaluation.error_metrics import get_error_metrics
from src.evaluation.goodness_of_fit_metrics import get_goodness_of_fit_metrics
from src.evaluation.distribution_similarity_metrics import get_distribution_similarity_metrics
from src.evaluation.rank_similarity_metrics import get_expr_rank_similarity_score, get_logFC_rank_similarity_score
from src.evaluation.biological_validation_metrics import get_gene_regulation_agreement, get_top_logfc_correlation_score, get_biorep_delta


def get_model_stats(test_results=None,
                    adata_control=None,
                    gene_names=None,
                    model_name=None,
                    experiment_key=None,
                    experiment_condition=None,
                    dose_subset=None,
                    method_deg='wilcoxon'):

        #ERROR Metrics
        results_mse, results_mae, results_rmse, results_l2norm = get_error_metrics(test_results)

        if dose_subset is not None:
                test_results = test_results[test_results['dose'] == dose_subset]

        #GOODNESS of FIT METRICS
        results_r2, results_css, results_pearson = get_goodness_of_fit_metrics(test_results, adata_control)

        #Distribution similarity metrics
        results_mmd, results_wasserstein = get_distribution_similarity_metrics(test_results)

        #Rank similarity metrics
        lfc = get_results__fc(test_results, adata_control, gene_names, method=method_deg)
        results_logfc_rank_similarity = get_logFC_rank_similarity_score(lfc)
        results_expr_rank_similarity = get_expr_rank_similarity_score(test_results, adata_control)

        results_gra = get_gene_regulation_agreement(lfc)
        results_top_logfc_correlation = get_top_logfc_correlation_score(lfc)
        results_biorep_delta = get_biorep_delta(test_results, adata_control)

        return {"model": model_name,
                "experiment_key": experiment_key,
                "experiment_condition": experiment_condition,
                "dose_subset": dose_subset,

                "mse_A549": results_mse['A549'],
                "mse_K562": results_mse['K562'],
                "mse_MCF7": results_mse['MCF7'],

                "mae_A549": results_mae['A549'],
                "mae_K562": results_mae['K562'],
                "mae_MCF7": results_mae['MCF7'],

                "rmse_A549": results_rmse['A549'],
                "rmse_K562": results_rmse['K562'],
                "rmse_MCF7": results_rmse['MCF7'],

                "l2norm_A549": results_l2norm['A549'],
                "l2norm_K562": results_l2norm['K562'],
                "l2norm_MCF7": results_l2norm['MCF7'],

                "r2_A549": results_r2['A549'],
                "r2_K562": results_r2['K562'],
                "r2_MCF7": results_r2['MCF7'],

                "css_A549": results_css['A549'],
                "css_K562": results_css['K562'],
                "css_MCF7": results_css['MCF7'],

                "pearson_A549": results_pearson['A549'],
                "pearson_K562": results_pearson['K562'],
                "pearson_MCF7": results_pearson['MCF7'],

                "mmd_A549": results_mmd["A549"],
                "mmd_K562": results_mmd["K562"],
                "mmd_MCF7": results_mmd["MCF7"],

                "wasserstein_A549": results_wasserstein["A549"],
                "wasserstein_K562": results_wasserstein["K562"],
                "wasserstein_MCF7": results_wasserstein["MCF7"],

                "logfc_rank_simiarity_A549": results_logfc_rank_similarity["A549"],
                "logfc_rank_simiarity_K562": results_logfc_rank_similarity["K562"],
                "logfc_rank_simiarity_MCF7": results_logfc_rank_similarity["MCF7"],

                "expr_rank_similarity_A549": results_expr_rank_similarity["A549"],
                "expr_rank_similarity_K562": results_expr_rank_similarity["K562"],
                "expr_rank_similarity_MCF7": results_expr_rank_similarity["MCF7"],

                "gra_A549": results_gra["A549"],
                "gra_K562": results_gra["K562"],
                "gra_MCF7": results_gra["MCF7"],

                "logfc_corr_A549": results_top_logfc_correlation["A549"],
                "logfc_corr_K562": results_top_logfc_correlation["K562"],
                "logfc_corr_MCF7": results_top_logfc_correlation["MCF7"],

                "biorep_delta_A549": results_biorep_delta["A549"],
                "biorep_delta_K562": results_biorep_delta["K562"],
                "biorep_delta_MCF7": results_biorep_delta["MCF7"],
            }



