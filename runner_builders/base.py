from runner_interfaces import RunnerInterface

import constants
from enums import SplitRunType
from typing import Dict
from utils import config_utils

from .multimodal_graph_topic_model_runners import (
    get_multimodal_concept_topic_graph_model_runners,
)


def get_runners(**kwargs) -> Dict[SplitRunType, RunnerInterface]:
    config = config_utils.load_config()
    model_name_to_runner_func_map = {
        constants.MODEL_MULTIMODAL_CONCEPT_TOPIC_GRAPH: get_multimodal_concept_topic_graph_model_runners,
    }
    return model_name_to_runner_func_map[config[constants.FIELD_MODEL_TO_USE]](**kwargs)
