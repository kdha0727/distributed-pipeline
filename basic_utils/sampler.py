import numpy as np
import torch


class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, rank: int = 0, num_replicas: int = 1, shuffle: bool = True, seed: int = 0, window_size: float = 0.5):
        assert len(dataset) > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        assert 0 <= window_size <= 1
        super().__init__(dataset)
        self.dataset = dataset
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size

    def __iter__(self):
        length = len(self.dataset)
        order = np.arange(length)
        rnd = None
        window = 0
        if self.shuffle:
            rnd = np.random.RandomState(self.seed)
            rnd.shuffle(order)
            window = int(np.rint(length * self.window_size))

        idx = 0
        while True:
            i = idx % length
            if idx % self.num_replicas == self.rank:
                yield order[i]
            if window >= 2:
                j = (i - rnd.randint(window)) % length
                order[i], order[j] = order[j], order[i]
            idx += 1
