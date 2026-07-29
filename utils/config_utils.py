import yaml
from typing import Dict, Any

import constants


def load_config() -> Dict[str, Any]:
    with open(constants.PATH_CONFIG_FILE, "r") as file:
        config = yaml.safe_load(file)
    return config


def save_config(config: Dict[str, Any]) -> None:
    with open(constants.PATH_CONFIG_FILE, "w") as file:
        yaml.dump(config, file, default_flow_style=False)
