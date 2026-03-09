import torch.distributed as dist


def is_rank0():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0
