import torch

import constants

def get_device():
    return torch.cuda.current_device() if torch.cuda.is_available() else constants.DEVICE_CPU

def get_device_count():
    return torch.cuda.device_count()

def clear_cuda_cache():
    if get_device() != constants.DEVICE_CPU:
        torch.cuda.empty_cache()