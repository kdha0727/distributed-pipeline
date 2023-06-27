# Pytorch pipeline with `torch.distributed`

`utils.trainer.TrainLoop` will run training loop - compatible with torch.distributed.run

## All you need to do

* Complete `config/train.py`'s `TrainSettings`(or `YourSettings`) class.
  * this setting class is compatible with argparse and json.
* Complete `data/__init__.py`'s `load_data_from_args` function.
* Complete `model` package.
* Complete `utils/initialization.py`'s `create_model_from_config` function.
* Complete some method of `utils/trainer.py`'s `TrainLoop` class.
  * `log_loss_dict` method: logging function of loss values dict.
  * `compute_losses` method: calculate `losses` from `micro_batch` and TrainLoop vars
  * `backward_from_losses` method: make single `loss` from `losses`, and run `loss.backward()`
  * `__init__` method: add your extra values to TrainLoop vars if needed.
* Complete `train.py` to make sense with all code signatures you modified.
* Modify setting json file, after copying default train settings with command,
  ```
  python3 -c "from config import TrainSettings as T; print(T().json(indent=2))" >> train_config.json
  ```

<details>
<summary>View simplest train.py script example:</summary>

```python
from torch.distributed.elastic.multiprocessing.errors import record


def main():

    import os
    import torch
    from basic_utils import dist_util

    if os.getenv("LOCAL_RANK", None) and not dist_util.is_initialized():
        dist_util.setup_dist()
        with dist_util.with_dist_cleanup():
            main()
        return
    rank = dist_util.get_rank()
    dist_util.barrier()

    class Model(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.param = torch.nn.Parameter(torch.ones(1))

        def forward(self, x):
            return self.param * x

    model = Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)

    model.to(dist_util.dev())
    dist_util.barrier()

    if dist_util.is_initialized():
        ddp_kwargs = dict(
            broadcast_buffers=False,
            bucket_cap_mb=128,
            find_unused_parameters=False,
        )
        if torch.cuda.is_available():
            ddp_kwargs.update(device_ids=[dist_util.dev()], output_device=dist_util.dev())
        ddp_model = torch.nn.parallel.DistributedDataParallel(model, **ddp_kwargs)
    else:
        ddp_model = model

    dist_util.sequential_print("rank", rank, "param :", model.param.data.item())
    dist_util.print_master_node()

    data = torch.ones(1, device=dist_util.dev()) * (dist_util.get_rank() + 1)
    target = torch.ones(1, device=dist_util.dev()) * 0.5 * (dist_util.get_rank() + 1)

    with ddp_model.no_sync() if dist_util.is_initialized() else dist_util.dummy_context():
        loss = (target - ddp_model(data)) ** 2
        dist_util.sequential_print("rank", rank, "loss :", loss.item())
        dist_util.print_master_node()

        loss.backward()
        dist_util.sequential_print("rank", rank, "grad :", model.param.grad.item())
        dist_util.print_master_node()

    loss = (target - ddp_model(data)) ** 2
    dist_util.sequential_print("rank", rank, "loss :", loss.item())
    dist_util.print_master_node()

    loss.backward()
    dist_util.sequential_print("rank", rank, "sync_grad :", model.param.grad.item())
    dist_util.print_master_node()

    optimizer.step()
    dist_util.sequential_print("rank", rank, "updated_param :", model.param.data.item())
    dist_util.barrier()


if __name__ == "__main__":
    record(main)()

```

Execute it with...

```bash
torchrun --nproc_per_node gpu train.py
```

Or without distributed training...

```bash
python3 train.py
```

</details>

## How to run

after completion, you can run train script with

```bash
torchrun --nproc_per_node gpu train.py --config_json train_config.json
```

## Citations

```bibtex
@inproceedings{gong2022diffuseq,
  author = {Gong, Shansan and Li, Mukai and Feng, Jiangtao and Wu, Zhiyong and Kong, Lingpeng},
  booktitle = {International Conference on Learning Representations, ICLR},
  title = {{DiffuSeq}: Sequence to Sequence Text Generation with Diffusion Models},
  year = 2023
}
```
