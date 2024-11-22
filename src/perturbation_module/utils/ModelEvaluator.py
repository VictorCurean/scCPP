from abc import ABC, abstractmethod


class ModelEvaluator(ABC):
    @abstractmethod
    def initialize(self):
        pass

    @property
    @abstractmethod
    def train_loader(self):
        pass

    @property
    @abstractmethod
    def validation_loader(self):
        pass


    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def validate(self):
        pass

    @abstractmethod
    def model_report(self):
        pass

