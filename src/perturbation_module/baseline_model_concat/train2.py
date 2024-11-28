from BaselineModelEvaluator import BaselineModelEvaluator

if __name__ == "__main__":
    eval = BaselineModelEvaluator("C:\\Users\\curea\\Documents\\bioFM for drug discovery\\dege-fm\\config\\baseline.yaml")
    eval.train()
    eval.test()
    eval.plot_stats()
