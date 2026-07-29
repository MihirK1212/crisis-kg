import torch

from data.datasets import MedicDisasterDataset
from models import MultimodalConceptTopicGraphModel
from model_interfaces import PytorchModelInterface
from model_meta_components.loss_functions import (
    VisionTextPipelineCrossEntropyLoss,
)
from runner_interfaces import (
    RunnerInterface,
    PytorchRunnerInterface,
)

import constants
from data.dataloader_interface import get_dataloader
from enums import SplitRunType
from typing import Dict
from utils import config_utils

config = config_utils.load_config()
MULTIMODAL_CONCEPT_GRAPH_MODEL_SAVE_PATH = f"./models/multimodal_concept_topic_graph/saved_model/multimodal_concept_topic_graph_{config[constants.FIELD_DATASET_TO_USE]}.pth"


def get_multimodal_concept_topic_graph_model_runners(
    use_pretrained_model: bool = False,
) -> Dict[SplitRunType, RunnerInterface]:
    config = config_utils.load_config()
    dataset_cls = MedicDisasterDataset
    train_dataloader = get_dataloader(
        dataset_cls=dataset_cls, split_run_type=SplitRunType.TRAIN
    )
    validation_dataloader = get_dataloader(
        dataset_cls=dataset_cls, split_run_type=SplitRunType.VALIDATION
    )
    test_dataloader = get_dataloader(
        dataset_cls=dataset_cls, split_run_type=SplitRunType.TEST
    )

    class_weights = torch.tensor(
        [0.00136054, 0.00194932, 0.0078125, 0.00362319, 0.00163399, 0.00229885, 0.00027886]
    )  # MEDIC dataset class weights

    model = MultimodalConceptTopicGraphModel(
        num_classes=len(constants.LABELS_MEDIC_DISASTER_TYPES),
        category_name=constants.FIELD_DISASTER_TYPE,
    )
    if use_pretrained_model:
        model.load_state_dict(torch.load(MULTIMODAL_CONCEPT_GRAPH_MODEL_SAVE_PATH))

    loss_function = VisionTextPipelineCrossEntropyLoss(class_weights=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.00007)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
    model_interface = PytorchModelInterface(
        model=model,
        loss_function=loss_function,
        optimizer=optimizer,
        scheduler=scheduler,
        model_save_path=MULTIMODAL_CONCEPT_GRAPH_MODEL_SAVE_PATH,
    )

    model_interfaces = {constants.FIELD_PYTROCH_MODEL_INTERFACE: model_interface}
    return {
        SplitRunType.TRAIN: PytorchRunnerInterface(
            model_interfaces=model_interfaces,
            dataloader=train_dataloader,
            split_run_type=SplitRunType.TRAIN,
        ),
        SplitRunType.VALIDATION: PytorchRunnerInterface(
            model_interfaces=model_interfaces,
            dataloader=validation_dataloader,
            split_run_type=SplitRunType.VALIDATION,
        ),
        SplitRunType.TEST: PytorchRunnerInterface(
            model_interfaces=model_interfaces,
            dataloader=test_dataloader,
            split_run_type=SplitRunType.TEST,
        ),
    }
