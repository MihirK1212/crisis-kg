import os

import torch
import torch.distributed as dist

import constants
import runner_builders
from enums import SplitRunType
from utils import config_utils, gpu_utils


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def get_dummy_rank_parameter():
    config = config_utils.load_config()
    if config.get(constants.FIELD_USE_DUMMY_RANK_FOR_PARALLEL_ENABLED_MODEL):
        return {"rank": 0}
    return {}


def run(rank, world_size):
    config = config_utils.load_config()
    if config.get(constants.FIELD_RUN_PARALLEL):
        setup(rank, world_size)
        runners = runner_builders.get_runners(rank=rank)
    else:
        runners = runner_builders.get_runners(**get_dummy_rank_parameter())

    print("DEVICE:", gpu_utils.get_device())

    for epoch in range(config.get(constants.FIELD_NUM_EPOCHS)):
        train_runner = runners.get(SplitRunType.TRAIN, None)
        if train_runner:
            train_runner.run_epoch(epoch)

        if config[constants.FIELD_RUN_PARALLEL]:
            dist.barrier()

        with torch.no_grad():
            validation_runner = runners.get(SplitRunType.VALIDATION, None)
            if validation_runner:
                validation_runner.run_epoch(epoch)

            test_runner = runners[SplitRunType.TEST]
            test_runner.run_epoch(epoch)

    if config[constants.FIELD_RUN_PARALLEL]:
        cleanup()


if __name__ == "__main__":
    config = config_utils.load_config()

    if config[constants.FIELD_RUN_PARALLEL]:
        world_size = torch.cuda.device_count()
        torch.multiprocessing.spawn(run, args=(world_size,), nprocs=world_size)
    else:
        run(None, None)
