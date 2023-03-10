from typing import final
from argparse import ArgumentParser as Ap, ArgumentDefaultsHelpFormatter as Df
from .base import S, Choice, Item as _


class GeneralSettings(S):
    lr: float \
        = _(1e-4, "Learning Rate")
    batch_size: int \
        = _(2048, "Batch size of running step and optimizing")
    microbatch: int \
        = _(64, "Batch size for forward and backward")
    learning_steps: int \
        = _(320000, "Steps for whole iteration")
    log_interval: int \
        = _(20, "Steps per log")
    save_interval: int \
        = _(2000, "Steps per save")
    eval_interval: int \
        = _(1000, "Steps per eval")
    ema_rate: str \
        = _("0.5,0.9,0.99", "EMA rate. separate rates by comma(',').")
    seed: int \
        = _(102, "Seed for train or test.")
    resume_checkpoint: str \
        = _("", "Checkpoint path(.pt) to resume training")
    checkpoint_path: str \
        = _("", "! This will be automatically updated while training !")
    gradient_clipping: float \
        = _(0., "Gradient clipping (>0), default: 0 (no clipping). ")
    weight_decay: float \
        = _(0., "Weight decay.")


class DataSettings(S):
    dataset: str \
        = _("dataset", "Name of dataset.")
    data_dir: str \
        = _("datasets/dataset", "Path for dataset to be saved.")
    data_loader_workers: int \
        = _(2, "num_workers for DataLoader.")


class YourSettings(S):
    # TODO: add extra settings on your own
    pass


@final
class TrainSettings(
        YourSettings,
        # TODO: inherit setting classes "reversely", due to python mro.
        DataSettings,
        GeneralSettings
):

    @classmethod
    def to_argparse(cls, parser_or_group=None, add_json=False):
        if not add_json:
            return super(TrainSettings, cls).to_argparse(parser_or_group)
        if parser_or_group is None:
            parser_or_group = Ap(formatter_class=Df)
        setting_group = parser_or_group.add_argument_group(title="settings")
        setting_group.add_mutually_exclusive_group().add_argument(
            "--config_json", type=str, required=False,
            help="You can alter arguments all below by config_json file.")
        super(TrainSettings, cls).to_argparse(setting_group.add_mutually_exclusive_group())
        return parser_or_group

    @classmethod
    def from_argparse(cls, namespace, __top=True):
        if getattr(namespace, "config_json", None):
            return cls.parse_file(namespace.config_json)
        else:
            if hasattr(namespace, "config_json"):
                delattr(namespace, "config_json")
            return super(TrainSettings, cls).from_argparse(namespace)


__all__ = ('TrainSettings', )
