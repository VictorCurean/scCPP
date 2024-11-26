from abc import ABC, abstractmethod


class ModelEvaluator(ABC):
    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def model_report_sciplex(self):
        pass

    @abstractmethod
    def model_report_zhao(self):
        pass

