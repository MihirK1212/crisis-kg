import torch
from typing import Dict

from data.schemas import PytorchModelOutputSchema
from utils import gpu_utils


class VisionTextPipelineCrossEntropyLoss(torch.nn.Module):
    def __init__(self, rank: int = gpu_utils.get_device(), class_weights=None) -> None:
        super(VisionTextPipelineCrossEntropyLoss, self).__init__()
        self.classification_loss_function = torch.nn.CrossEntropyLoss(
            weight=class_weights.to(rank) if class_weights is not None else None
        )

    def forward(
        self, model_output: PytorchModelOutputSchema, target: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        pred_logits = model_output.pred_logits
        assert pred_logits.keys() == target.keys()
        loss = 0
        for k in pred_logits.keys():
            loss += self.classification_loss_function(pred_logits[k], target[k])
        return loss
