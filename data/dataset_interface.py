from abc import ABC, abstractmethod
from typing import List

from data.schemas import DataItemSchema
from enums import SplitRunType


class DatasetInterface(ABC):

    split_run_type: SplitRunType = None

    def __init__(self, split_run_type: SplitRunType) -> None:
        if split_run_type == None or not isinstance(split_run_type, SplitRunType):
            raise ValueError("split_run_type must be one of TRAIN, TEST or VALID")
        self.split_run_type = split_run_type

    @abstractmethod
    def get_data_length(self) -> int:
        """get_data_length must be implemented by the subclass"""
        raise NotImplementedError("Subclasses must implement get_data_length")

    @abstractmethod
    def get_data_item(self, idx=-1) -> DataItemSchema:
        """get_data_item must be implemented by the subclass"""
        raise NotImplementedError("Subclasses must implement get_data_item")
