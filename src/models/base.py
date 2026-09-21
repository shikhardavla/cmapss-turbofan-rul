from abc import ABC, abstractmethod

class BaseModel(ABC):

    @abstractmethod
    def train (self, X_train, y_train, **kwargs):
        pass

    @abstractmethod
    def predict (self, X):
        pass