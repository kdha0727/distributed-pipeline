# Copyright (c) 2023 kdha0727
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Helpers for distributed training.
This module's function is compatible even though script is not running in torch.distributed.run environment.
Write code as though you are using torch.distributed.run - if you directly run scripts, it works!
"""
__author__ = "https://github.com/kdha0727"

import io
import os
import contextlib
import functools
from typing import overload, Sequence, Optional, Union, ContextManager, Any

import torch
import torch.distributed as dist
from torch.cuda import is_available as _cuda_available


RANK: int = 0
WORLD_SIZE: int = 1


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                       Setup Tools                                       #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def is_initialized() -> bool:
    # if pytorch isn't compiled with c10d, is_initialized is omitted from namespace.
    # this function wraps
    """
    Returns c10d (distributed) runtime is initialized.
    """
    return dist.is_available() and getattr(dist, "is_initialized", lambda: False)()


@overload
def setup_dist(temp_dir: str, rank: int, world_size: int) -> None: ...


@overload
def setup_dist() -> None: ...


def setup_dist(*args):
    """
    Set up a distributed process group.
    Usage
        1. setup_dist(temp_dir, rank, world_size)
            : if you want to init by file, call this function with three args (temp_dir, rank, world_size).
        2. setup_dist()
            : if you want to init by env (by torchrun), call this function without args.
    """
    if is_initialized():
        return

    backend = "gloo" if (not _cuda_available()) or (os.name == "nt") else "nccl"

    if len(args) == 3:
        temp_dir, rank, world_size = args
        assert os.path.isdir(temp_dir), f"temp_dir {temp_dir} is not a directory"
        assert isinstance(rank, int), f"rank {rank} must be int"
        assert isinstance(world_size, int), f"world_size {world_size} must be int"
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
        else:
            init_method = f'file://{init_file}'
        dist.init_process_group(backend=backend, init_method=init_method, rank=rank, world_size=world_size)
    elif len(args) == 0:
        assert os.getenv("LOCAL_RANK", None) is not None, "environ LOCAL_RANK is not set"
        dist.init_process_group(backend=backend, init_method="env://")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        raise TypeError("setup_dist() takes 0 or 3 arguments")

    global RANK, WORLD_SIZE
    RANK = rank
    WORLD_SIZE = world_size

    if _cuda_available():
        torch.cuda.set_device(dev())
        torch.cuda.empty_cache()


def cleanup_dist():
    """
    Clean up a distributed process group.
    """
    if is_initialized():
        dist.destroy_process_group()


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                      General Tools                                      #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

@functools.lru_cache(maxsize=None)
def get_rank(group: Optional[dist.ProcessGroup] = None) -> int:
    """
    Wrapper of torch.distributed.get_rank.
    Get the rank of current process.
    """
    if group is not None and is_initialized():
        return dist.get_rank(group=group)
    return RANK


@functools.lru_cache(maxsize=None)
def get_world_size(group: Optional[dist.ProcessGroup] = None) -> int:
    """
    Wrapper of torch.distributed.get_world_size.
    Get the world size of current process.
    """
    if group is not None and is_initialized():
        return dist.get_world_size(group=group)
    return WORLD_SIZE


def barrier(*args: ..., **kwargs: ...) -> None:
    """
    Wrapper for torch.distributed.barrier.
    Synchronizes all processes.
    see `torch.distributed.distributed_c10d.barrier.__doc__` for more information.
    """
    if is_initialized():
        return dist.barrier(*args, **kwargs)


@contextlib.contextmanager
def synchronized() -> ContextManager[None]:
    """
    context manager version of barrier() function.
    """
    barrier()
    yield
    barrier()


@functools.lru_cache(maxsize=None)
def dev(group: Optional[dist.ProcessGroup] = None) -> torch.device:
    """
    Get the device to use for torch.distributed.
    """
    if _cuda_available():
        return torch.device("cuda:{}".format(get_rank(group)))
    return torch.device("cpu")


try:
    def load_state_dict(local_or_remote_path: Union[str, os.PathLike], **kwargs: ...) -> Any:
        """
        Load a PyTorch file.
        """
        with bf.BlobFile(local_or_remote_path, "rb") as f:
            data = f.read()
        return torch.load(io.BytesIO(data), **kwargs)
    import blobfile as bf  # NOQA: F401
except ImportError:
    def load_state_dict(local_or_remote_path: Union[str, os.PathLike], **kwargs: ...) -> Any:
        """
        Load a PyTorch file.
        """
        return torch.load(local_or_remote_path, **kwargs)


def broadcast(
        tensor: Sequence[torch.Tensor],
        src: Optional[int] = 0,
        group: Optional[dist.ProcessGroup] = None,
        async_op: bool = False
) -> None:
    """
    Synchronize a Tensor across ranks from {src} rank. (default=0)
    :param tensor: torch.Tensor.
    :param src: source rank to sync params from. default is 0.
    :param group:
    :param async_op:
    """
    if not is_initialized():
        return
    with torch.no_grad():
        dist.broadcast(tensor, src, group=group, async_op=async_op)


def sync_params(
        params: Sequence[torch.Tensor],
        src: Optional[int] = 0,
        group: Optional[dist.ProcessGroup] = None,
        async_op: bool = False
) -> None:
    """
    Synchronize a sequence of Tensors across ranks from {src} rank. (default=0)
    :param params: Sequence of torch.Tensor.
    :param src: source rank to sync params from. default is 0.
    :param group:
    :param async_op:
    """
    if not is_initialized():
        return
    for p in params:
        broadcast(p, src, group=group, async_op=async_op)


def sequential_print(*args: ..., **kwargs: ...) -> None:
    """
    Print argument sequentially by rank order.
    Arguments are passed to print function.
    """
    for i in range(get_world_size()):
        if i == get_rank():
            print(*args, **kwargs)
        barrier()


def print_master_node(*args: ..., **kwargs: ...) -> None:
    """
    Print argument only on master node.
    Arguments are passed to print function.
    """
    if get_rank() == 0:
        print(*args, **kwargs)
    barrier()


@contextlib.contextmanager
def dummy_context() -> ContextManager[None]:
    """
    Dummy context manager.
    """
    yield


try:
    # patch some os module functions - since file io uses os.sync
    import nt  # NOQA
    os.sync = nt.sync = lambda: None  # signature: () -> None
except ImportError:
    pass
