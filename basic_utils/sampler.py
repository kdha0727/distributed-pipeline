import torch


class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(
        self, 
        dataset, num_replicas: int = 1,
        rank: int = None, shuffle: bool = True, 
        seed: int = 0, window_size: float = 0.5
    ) -> None:
        from . import dist_util
        assert len(dataset) > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        assert 0 <= window_size <= 1
        super().__init__(dataset)
        self.dataset = dataset
        self.num_replicas = num_replicas if num_replicas is not None else dist_util.get_world_size()
        self.rank = rank if rank is not None else dist_util.get_rank()
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size

    def __iter__(self):
        length = len(self.dataset)
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            order = torch.randperm(length, generator=g).tolist()  # type: ignore[arg-type]
            window = round(length * self.window_size)
        else:
            g = None
            order = list(range(length))  # type: ignore[arg-type]
            window = 0

        idx = 0
        while True:
            i = idx % length
            if idx % self.num_replicas == self.rank:
                yield order[i]
            if window >= 2:
                j = (i - torch.randint(window, size=(), generator=g)) % length
                order[i], order[j] = order[j], order[i]
            idx += 1
