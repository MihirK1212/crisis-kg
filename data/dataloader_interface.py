import torch
from typing import Type, Any, Dict

from data.dataset_interface import DatasetInterface
from enums import SplitRunType
from utils import data_utils, gpu_utils, config_utils
from torch.utils.data import DistributedSampler


class LoaderInstanceInterface(torch.utils.data.Dataset):
    def __init__(self, dataset: DatasetInterface) -> None:
        self.dataset: DatasetInterface = dataset

    def __len__(self) -> int:
        return self.dataset.get_data_length()

    def __getitem__(self, idx) -> Dict[str, Any]:
        return self.dataset.get_data_item(idx).model_dump()


def get_dataloader(
    dataset_cls: Type[DatasetInterface],
    split_run_type: SplitRunType,
    dataset_params={},
    **kwargs
):
    config = config_utils.load_config()

    dataset = dataset_cls(split_run_type, **dataset_params)
    dataset_loader_instance = LoaderInstanceInterface(dataset)

    rank = kwargs.get("rank", None)

    if (
        (rank is not None)
        and isinstance(rank, int)
        and rank >= 0
        and config["run_parallel"]
    ):
        world_size = gpu_utils.get_device_count()
        sampler = DistributedSampler(
            dataset_loader_instance,
            num_replicas=world_size,
            rank=rank,
            shuffle=data_utils.get_dataloader_params(split_run_type)["shuffle"],
        )
        return (
            torch.utils.data.DataLoader(
                dataset_loader_instance,
                sampler=sampler,
                batch_size=data_utils.get_dataloader_params(split_run_type)[
                    "batch_size"
                ],
            ),
            sampler,
        )
    else:
        return torch.utils.data.DataLoader(
            dataset_loader_instance, **data_utils.get_dataloader_params(split_run_type)
        )
