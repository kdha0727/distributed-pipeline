def load_data_from_args(
        split,
        data_dir,
        batch_size,
        # TODO: add args on your own
        deterministic=False,
        loop=True,
        num_loader_proc=1,
):
    from torch.utils.data import DataLoader
    from .dataset import CustomDataset

    # TODO: add your data-loading function from split & data_dir & your arguments
    dataset_kwargs = ...

    data = CustomDataset(**dataset_kwargs)
    loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=not deterministic,
        num_workers=num_loader_proc,
        persistent_workers=num_loader_proc > 0,
    )
    if loop:
        return infinite_loader_from_iterable(loader)
    else:
        return loader


def infinite_loader_from_object(obj):
    import copy
    while True:
        yield copy.deepcopy(obj)


def infinite_loader_from_iterable(iterable):
    while True:
        yield from iterable
