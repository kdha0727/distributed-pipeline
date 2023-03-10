# Add your dataset configuration
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    # TODO: implement this class

    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
