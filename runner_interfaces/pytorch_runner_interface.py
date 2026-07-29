from abc import ABC, abstractmethod
from typing import List, Any, Dict, Tuple
import torch
from tqdm import tqdm
import time

import constants
from model_interfaces import ModelInterface, PytorchModelInterface
from data.schemas import DataItemSchema, PytorchModelOutputSchema
from enums import SplitRunType
from runner_interfaces import RunnerInterface
from utils import gpu_utils, data_utils, config_utils, metric_utils

class PytorchRunnerInterface(RunnerInterface):
    def __init__(self, model_interfaces: Dict[str, PytorchModelInterface], dataloader: Any, split_run_type: SplitRunType) -> None:
        super().__init__(model_interfaces, dataloader, split_run_type)
        assert len(model_interfaces) == 1
        assert constants.FIELD_PYTROCH_MODEL_INTERFACE in model_interfaces
        self.model_interface = self._load_model_interface()
        self.max_accuracy = 0

        self._config = config_utils.load_config()

    def _load_model_interface(self) -> PytorchModelInterface:
        return self.model_interfaces[constants.FIELD_PYTROCH_MODEL_INTERFACE]

    def run_epoch(self, epoch: int) -> None | Dict[str, Any]:
        torch.cuda.empty_cache()

        self.model_interface.prepare_model(self.split_run_type)

        num_steps = 0
        total_loss = 0
        batch_probabilities: List[Dict[str, torch.Tensor]] = []
        batch_predictions: List[Dict[str, torch.Tensor]] = []
        batch_targets: List[Dict[str, torch.Tensor]] = []
 
        total_inference_time_s: float = 0.0
        total_inference_samples: int = 0

        for _, batch_item in tqdm(enumerate(self.dataloader, 0)):
            gpu_utils.clear_cuda_cache()
            
            data_item = DataItemSchema(**batch_item)
            data_item = data_utils.load_data_item_to_device(data_item, device=gpu_utils.get_device())

            # Determine batch size from labels
            batch_size = next(iter(data_item.label.values())).shape[0]

            # Time the model inference (forward pass)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _t0 = time.perf_counter()
            model_output, prediction = self.model_interface.predict(data_item)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _t1 = time.perf_counter()
            total_inference_time_s += (_t1 - _t0)
            total_inference_samples += batch_size
            target = data_item.label

            loss = self.model_interface.get_loss(model_output, target)

            if self.split_run_type == SplitRunType.TRAIN:
                self.model_interface.fit(loss)

            probabilities = data_utils.load_tensor_dict_to_device(model_output.pred_logits, device=constants.DEVICE_CPU)
            prediction = data_utils.load_tensor_dict_to_device(prediction, device=constants.DEVICE_CPU, data_type=torch.long)
            target = data_utils.load_tensor_dict_to_device(target, device=constants.DEVICE_CPU, data_type=torch.long)

            num_steps+=1
            total_loss+=loss.item()
            batch_probabilities.append(probabilities)
            batch_predictions.append(prediction)
            batch_targets.append(target)

        if self._config[constants.FIELD_USE_LR_SCHEDULER]:
            self.model_interface.scheduler_step()
        
        epoch_loss = total_loss / num_steps
        probabilities = data_utils.concatenate_tensor_dict_list(batch_probabilities)
        predictions = data_utils.concatenate_tensor_dict_list(batch_predictions)
        targets = data_utils.concatenate_tensor_dict_list(batch_targets)

        # Compute avg inference time per sample across the epoch
        avg_inference_time_per_sample_s = (total_inference_time_s / total_inference_samples) if total_inference_samples > 0 else 0.0
        # Compute number of trainable parameters in the model
        trainable_params = sum(p.numel() for p in self.model_interface.model.parameters() if p.requires_grad)

        metrics_dict  = self.epoch_analysis(
            epoch=epoch,
            probabilities=probabilities,
            predictions=predictions,
            targets=targets,
            other_values={
                'epoch_loss': epoch_loss,
                'inference_time_per_sample_s': avg_inference_time_per_sample_s,
                'trainable_params': trainable_params,
            }
        )
        
        del epoch_loss
        torch.cuda.empty_cache()

        if self.split_run_type == SplitRunType.TEST and metrics_dict['accuracy'] > self.max_accuracy:
            if epoch > 0 and self.max_accuracy < 0.01:
                raise ValueError("something wrong with accuracy!")
            self.max_accuracy = metrics_dict['accuracy']
            self.model_interface.save_model()

        if self.split_run_type == SplitRunType.TEST:
            return {
                'probabilities': probabilities,
                'predictions': predictions,
                'targets': targets,
                'avg_inference_time_per_sample_s': avg_inference_time_per_sample_s,
            }
    
