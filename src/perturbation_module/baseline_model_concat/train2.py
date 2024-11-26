from BaselineModelEvaluator import BaselineModelEvaluator

if __name__ == "__main__":
    eval = BaselineModelEvaluator()
    eval.train()
    eval.model_report_sciplex()
    eval.model_report_zhao()