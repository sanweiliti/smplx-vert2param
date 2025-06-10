import torch as th
import torch.distributed as dist


used_device = 0

def setup_dist(device=0):
    """
    Setup a distributed process group.
    """
    global used_device
    used_device = device
    if dist.is_initialized():
        return


def dev():
    """
    Get the device to use for torch.distributed.
    """
    global used_device
    if th.cuda.is_available() and used_device>=0:
        return th.device(f"cuda:{used_device}")
    return th.device("cpu")