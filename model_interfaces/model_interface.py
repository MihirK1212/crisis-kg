from abc import ABC, abstractmethod
from typing import List, Any, Dict, Tuple
import torch

from data.schemas import DataItemSchema
from enums import SplitRunType

class ModelInterface(ABC):
    def __init__(self, model) -> None:
        self.model = model

    @abstractmethod
    def _get_model_output(self, data_item: DataItemSchema) -> Any:
        """_get_model_output must be implemented by the subclass"""
        raise NotImplementedError("Subclasses must implement _get_model_output")

    @abstractmethod
    def _get_prediction_from_model_output(
        self, model_output: Any
    ) -> Dict[str, torch.Tensor]:
        """_get_prediction_from_model_output must be implemented by the subclass"""
        raise NotImplementedError(
            "Subclasses must implement _get_prediction_from_model_output"
        )
    
    def _check_prediction(
        self, prediction: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]
    ) -> bool:
        if prediction.keys() != target.keys():
            return False
        for key in prediction:
            if prediction[key].size() != target[key].size():
                return False
        return True

    def predict(self, data_item: DataItemSchema) -> Tuple[Any, Dict[str, torch.Tensor]]:
        model_output = self._get_model_output(data_item)
        prediction = self._get_prediction_from_model_output(model_output)
        assert self._check_prediction(prediction, target=data_item.label)
        return model_output, prediction
    
    def save_model(self):
        pass