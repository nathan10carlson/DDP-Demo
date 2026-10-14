import os

def get_distributed_info():
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ

    if distributed:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        rank = 0
        local_rank = 0
        world_size = 1

    return distributed, rank, local_rank, world_size