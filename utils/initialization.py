def seed_all(seed, deterministic=False):
    import random
    import numpy as np
    import torch
    from basic_utils.dist_util import get_rank
    if deterministic:
        seed = int(seed)
        torch.backends.cudnn.deterministic = True  # NOQA
        torch.backends.cudnn.benchmark = False  # NOQA
    else:
        seed = int(seed) + get_rank()  # Make seed differ by node rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # contains torch.cuda.manual_seed_all
    # TODO: Add your seeder if needed


def create_model_from_config(
        *,
        argument1,
        argument2,
        argument3,
        **_
):
    # TODO: Implement this function
    model = ...
    return model
