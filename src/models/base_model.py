from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):
    @abstractmethod
    def fit(self, Phi: np.ndarray, y: np.ndarray) -> None:
        pass
    @abstractmethod
    def predict_proba(self, Phi: np.ndarray) -> np.ndarray:
        pass