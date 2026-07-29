import json
import os

import yaml

COMPLETED_COMBINATIONS_PATH = os.path.join("combination_tracker", "completed_combinations.json")


def is_tracking_enabled(combo: dict) -> bool:
    return False
    """Tracking is only meaningful when at least one component is actually disabled."""
    return bool(combo) and any(combo.values())


def load_completed_combinations() -> list[dict]:
    if not os.path.exists(COMPLETED_COMBINATIONS_PATH):
        return []
    with open(COMPLETED_COMBINATIONS_PATH, "r") as f:
        data = json.load(f)
    return data.get("completed", [])


def save_completed_combination(combo: dict):
    completed = load_completed_combinations()
    completed.append(combo)
    with open(COMPLETED_COMBINATIONS_PATH, "w") as f:
        json.dump({"completed": completed}, f, indent=2)


def is_combination_done(combo: dict, completed: list[dict]) -> bool:
    combo_items = frozenset(combo.items())
    return any(combo_items.issubset(frozenset(c.items())) for c in completed)


def update_config_yaml(update_config_dict: dict):
    print("update_config_dict", update_config_dict)

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    for k, v in update_config_dict.items():
        config[k] = v

    with open("config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
