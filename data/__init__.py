def load_data_from_args(
        split,
        data_dir,
        # TODO: add dataset args on your own
        batch_size: int,
        deterministic: bool = False,
        loop: bool = True,
        seed: ... = 0,
        window_size: float = 0.5,
        num_loader_proc: int = 1,
):
    from basic_utils import dist_util
    from basic_utils.sampler import InfiniteSampler
    from torch.utils.data.distributed import DistributedSampler
    from torch.utils.data import DataLoader
    from .dataset import CustomDataset

    rank = dist_util.get_rank()
    num_replicas = dist_util.get_world_size()

    # TODO: add your data-loading function from split & data_dir & your arguments
    dataset_kwargs = dict(data_dir=data_dir, split=split)  # TODO
    dataset = CustomDataset(**dataset_kwargs)

    sampler_kwargs = dict(
        num_replicas=num_replicas,
        rank=rank,
        shuffle=not deterministic,
        seed=hash(seed)
    )
    if loop:
        sampler = InfiniteSampler(dataset, **sampler_kwargs, window_size=window_size)
    else:
        sampler = DistributedSampler(dataset, **sampler_kwargs, drop_last=False)

    return DataLoader(
        dataset,
        batch_size=batch_size // num_replicas,
        sampler=sampler,
        num_workers=num_loader_proc,
        persistent_workers=num_loader_proc > 0,
        # pin_memory=True,
    )
