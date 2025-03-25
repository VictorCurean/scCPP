from abc import ABC, abstractmethod


class AbstractEvaluator(ABC):

    @abstractmethod
    def read_config(self, path):
        pass

    @abstractmethod
    def prepare_model(self):
        pass

    @abstractmethod
    def train(self, loss_fn):
        pass

    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def get_test_results(self):
        pass