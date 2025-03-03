"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
import torch as th
import torch.distributed as dist
from tqdm import tqdm

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3


def setup_dist():
    """
    Setup a distributed process group.
    """
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    print(f"local_rank = {local_rank}, device count = {th.cuda.device_count()}")
    th.cuda.set_device(f"cuda:{local_rank}")


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    chunk_size = 2**30  # MPI has a relatively small size limit
    local_rank = int(os.environ["LOCAL_RANK"])
    print("load_state_dict: local_rank = ", local_rank)
    if local_rank == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
        num_chunks = len(data) // chunk_size
        if len(data) % chunk_size:
            num_chunks += 1
        num_chunks = [num_chunks]
        dist.broadcast_object_list(num_chunks)
        for i in range(0, len(data), chunk_size):
            temp_data = [data[i : i + chunk_size]]
            dist.broadcast_object_list(temp_data)
    else:
        num_chunks = [None]
        dist.broadcast_object_list(num_chunks)
        num_chunks = num_chunks[0]
        data = bytes()
        for _ in range(num_chunks):
            temp_data = [None]
            dist.broadcast_object_list(temp_data)
            data += temp_data[0]
    print("load_state_dict: local_rank = ", local_rank, " done")
    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    local_rank = int(os.environ["LOCAL_RANK"])
    iter_params = tqdm(params) if local_rank == 0 else params
    for p in iter_params:
        with th.no_grad():
            dist.broadcast(p, 0)


def rank_zero_only(fn):
    """
    Decorator to make a function only run on rank 0.
    """

    def wrapped(*args, **kwargs):
        local_rank = int(os.environ["LOCAL_RANK"])
        if local_rank == 0:
            return fn(*args, **kwargs)

    return wrapped
