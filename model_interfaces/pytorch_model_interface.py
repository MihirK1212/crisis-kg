from abc import ABC, abstractmethod
from typing import List, Any, Dict, Tuple
import torch

import constants
from enums import SplitRunType
from model_interfaces import ModelInterface
from data.schemas import DataItemSchema, PytorchModelOutputSchema
from utils import config_utils, gpu_utils



class PytorchModelInterface(ModelInterface):
    def __init__(
        self,
        model: torch.nn.Module,
        loss_function: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        rank: int = gpu_utils.get_device(),
        model_save_path = None
    ) -> None:
        super().__init__(model=model)
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.rank = rank
        self.model_save_path = model_save_path

    def prepare_model(self, split_run_type: SplitRunType) -> None:
        self.model.to(self.rank)
        if split_run_type == SplitRunType.TRAIN:
            print("----- Setting model mode to TRAIN -----")
            self.model.train()
        else:
            print("----- Setting model mode to EVAL -----")
            self.model.eval()

    def _get_model_output(self, data_item: DataItemSchema) -> PytorchModelOutputSchema:
        return self.model(data_item)
    
    def _get_prediction_from_model_output(
        self, model_output: PytorchModelOutputSchema
    ) -> Dict[str, torch.Tensor]:
        predictions_dict: Dict[str, torch.Tensor] = dict({})
        for label, probabilities in model_output.pred_logits.items():
            _, prediction = torch.max(probabilities, dim=1)
            predictions_dict[label] = prediction
        return predictions_dict
    
    def get_loss(self, model_output: PytorchModelOutputSchema, target: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.loss_function(model_output, target)
    
    def fit(self, loss: torch.Tensor):
        config = config_utils.load_config()
        self.optimizer.zero_grad()
        loss.backward()
        if config[constants.FIELD_APPLY_GRADIENT_CLIPPING]:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

    def predict(self, data_item: DataItemSchema) -> Tuple[PytorchModelOutputSchema, Dict[str, torch.Tensor]]:
        return super().predict(data_item)
    
    def scheduler_step(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def save_model(self):
        if self.model_save_path is not None:
            torch.save(self.model.state_dict(), self.model_save_path)


