from typing import Literal
try:
    from typing import get_args
except ImportError:
    from typing_extensions import get_args
from argparse import ArgumentParser as Ap, ArgumentDefaultsHelpFormatter as Df
from pydantic import BaseModel, Field as _
from pydantic.validators import bool_validator
import yaml
import json


class ArgparseCompatibleBaseModel(BaseModel):

    @classmethod
    def from_argparse(cls, argparse_args):
        kwargs = vars(argparse_args)
        child_models = {}
        for name, field in cls.__fields__.items():
            child_models[name] = field.type_(**kwargs)
        kwargs.update(child_models)
        return cls(**kwargs)

    @classmethod
    def to_argparse(cls, parser_or_group=None):
        if parser_or_group is None:
            parser_or_group = Ap(formatter_class=Df)
        for name, field in cls.__fields__.items():
            if isinstance(field.type_, type) and issubclass(field.type_, BaseModel):
                group = parser_or_group.add_argument_group(name)
                if issubclass(field.type_, ArgparseCompatibleBaseModel):
                    field.type_.to_argparse(group)
                else:
                    ArgparseCompatibleBaseModel.to_argparse.__func__(field.type_, group)  # NOQA
                continue
            kw = dict(dest=name, type=field.type_, default=field.default,
                      help=field.field_info.description, required=field.required)
            if getattr(field.type_, '__origin__', None) is Literal:
                choices = tuple(get_args(field.outer_type_))
                s = "def {name}(arg):\n    for ch in __CHOICES__:\n" \
                    "        if str(ch) == arg:\n            return ch\n    raise ValueError" \
                    .format(name=name)
                n = {"__CHOICES__": choices, "__name__": name}
                exec(s, n)  # caster = n[name]
                kw.update(type=n[name], choices=choices, metavar="{"+", ".join(map(str, choices))+"}")
            elif isinstance(field.type_, type) and issubclass(field.type_, bool):
                kw.update(type=bool_validator, metavar="{true, false}")
            parser_or_group.add_argument("--" + name, **kw)
        return parser_or_group

    @classmethod
    def from_argv(cls):
        return cls.from_argparse(cls.to_argparse().parse_args())


S = ArgparseCompatibleBaseModel


if __name__ == '__main__':  # Some example
    
    class Config1(S):
        a: int = _(1, description='this is a')
        b: int = _(2, description='this is b')


    class Config2(S):
        c: Literal['choice1', 'choice2'] = _('choice2', description='this is c')
        d: bool = _(True, description='this is d')

    class Config(S):
        conf1: Config1 = Config1()
        conf2: Config2 = Config2()

    Config.to_argparse().print_help()
    yaml.dump(Config.from_argv().dict(), __import__('sys').stdout)
