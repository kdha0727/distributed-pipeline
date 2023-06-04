torchrun --nproc_per_node gpu train.py --config_json train_config.json
# python -m torch.distributed.launch --use_env --nproc_per_node 4 train.py --config_json train_config.json  # <=1.8
