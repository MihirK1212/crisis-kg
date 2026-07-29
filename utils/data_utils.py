import pandas as pd
from typing import Tuple, Dict, List, Any
import torch
from datetime import datetime

from utils import config_utils, gpu_utils
import constants
from enums import SplitRunType
from data.schemas import DataItemSchema, LogLineSchema, TokenizedTextInputsSchema


def split_dataframe(
    dataframe: pd.DataFrame, frac: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    config = config_utils.load_config()
    df1 = dataframe.sample(frac=frac, random_state=config[constants.FIELD_SEED])
    df2 = dataframe.drop(df1.index).reset_index(drop=True)
    df1 = df1.reset_index(drop=True)
    return df1, df2


def get_dataframes_split(dataframe: pd.DataFrame) -> Dict[SplitRunType, pd.DataFrame]:
    config = config_utils.load_config()
    train_data, validation_data = split_dataframe(
        dataframe, config[constants.FIELD_TRAIN_SPLIT_RATIO]
    )
    validation_data, test_data = split_dataframe(validation_data, 0.5)
    return {
        SplitRunType.TRAIN: train_data,
        SplitRunType.VALIDATION: validation_data,
        SplitRunType.TEST: test_data,
    }


def preprocess_text_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    return dataframe


def get_dataloader_params(split_run_type: SplitRunType):
    config = config_utils.load_config()
    if split_run_type == SplitRunType.TRAIN:
        return {
            "batch_size": config[constants.FIELD_TRAIN_BATCH_SIZE],
            "shuffle": True,
            "num_workers": 0,
        }
    elif split_run_type == SplitRunType.VALIDATION:
        return {
            "batch_size": config[constants.FIELD_VALIDATION_BATCH_SIZE],
            "shuffle": False,
            "num_workers": 0,
        }
    elif split_run_type == SplitRunType.TEST:
        return {
            "batch_size": config[constants.FIELD_TEST_BATCH_SIZE],
            "shuffle": False,
            "num_workers": 0,
        }


def load_tensor_dict_to_device(
    tensor_dict: Dict[str, torch.Tensor],
    device: str,
    data_type=None,
) -> Dict[str, torch.Tensor]:
    return {k: v.to(device=device, dtype=data_type) for k, v in tensor_dict.items()}


def load_data_item_to_device(data_item: DataItemSchema, device: str) -> DataItemSchema:
    def helper(val):
        if isinstance(val, dict):
            for k, v in val.items():
                val[k] = helper(v)
            return val
        elif isinstance(val, torch.Tensor):
            return val.to(device)
        elif isinstance(val, list):
            return [helper(v) for v in val]
        else:
            return val

    data_item_dict = data_item.model_dump()
    data_item_dict = helper(data_item_dict)
    return DataItemSchema(**data_item_dict)


def concatenate_tensor_dict_list(
    tensor_dict_list: List[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    if len(tensor_dict_list) == 0:
        return tensor_dict_list
    result = {}
    keys = tensor_dict_list[0].keys()
    for key in keys:
        result[key] = torch.cat([d[key] for d in tensor_dict_list])
    return result


def get_timestamp():
    return datetime.now().strftime("%Y%m%d%H%M%S")


def write_to_epoch_log(log_file_path: str, epoch_log_lines: List[LogLineSchema], mode="a+"):
    with open(log_file_path, mode) as log_file:
        log_file.write("################################################\n")
        for log_line in epoch_log_lines:
            for (key, value) in log_line.data:
                log_file.write(f"{key}: {value}; ")
                if log_line.new_line_in_between:
                    log_file.write("\n")
            if not log_line.new_line_in_between:
                log_file.write("\n")
            if log_line.blank_line_after:
                    log_file.write("\n")
        log_file.write("################################################\n\n")

def parse_tokenizer_output(tokenizer_output) -> TokenizedTextInputsSchema:
    ids = tokenizer_output["input_ids"]
    attention_mask = tokenizer_output["attention_mask"]
    token_type_ids = tokenizer_output["token_type_ids"]
    return TokenizedTextInputsSchema(**{
        "ids": torch.tensor(ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
    })