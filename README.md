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
* Complete `run/train.py` to make sense with all code signatures you modified.
* Modify setting json file, after copying default train settings with command,
  ```
  python3 -c "from config;train import TrainSettings as T; print(T.json(indent=2))" >> train_config.json
  ```

## How to run

after completion, you can run train script with

```bash
python3 -m run.train --distributed --config_json train_config.json
```
