import sys
import pydoc
from typing import Optional

from ruamel.yaml import YAML, CommentedMap
from fire import Fire
from addict import Dict


class CfgDict(Dict):
    """Modified addict.Dict class without blank {} returns while missing."""

    def __missing__(self, key):
        return None


class MLConfig:
    """
    Main config class with addict cfg for interaction and yaml one for safe dumping.

    Args:
        yaml_config (ruamel.yaml.CommentedMap): safe loaded with ruamel yaml config.
    """

    def __init__(self, yaml_config: CommentedMap):
        self.__yaml_config = yaml_config
        self.__cfg = CfgDict(yaml_config)

    def __getattr__(self, item):
        return getattr(self.__cfg, item)

    def __getitem__(self, key):
        return self.__cfg[key]

    def __setitem__(self, key, value):
        return setattr(self.__cfg, key, value)

    def pretty_print(self) -> None:
        yaml = YAML()
        yaml.dump(self.__yaml_config, sys.stdout)

    def dump(self, path: str):
        yaml = YAML()

        with open(path, "w") as f:
            yaml.dump(self.__yaml_config, f)


def update_config(config: CommentedMap, params: dict):
    """Updates base config with params from new one --config and some specified --params."""
    for k, v in params.items():
        *path, key = k.split(".")

        updating_config = config

        if path:
            for p in path:
                if p not in updating_config:
                    updating_config[p] = {}
                updating_config = updating_config[p]

        updating_config.update({key: v})
    return config


def fit(**kwargs) -> CommentedMap:
    """Loads base config and updates it with specified new one --config with some others --params."""
    yaml = YAML()

    with open("configs/base.yml", "r") as f:
        base_config = yaml.load(f)

    if "config" in kwargs:
        cfg_path = kwargs.pop("config")
        with open(cfg_path, "r") as f:
            cfg_yaml = yaml.load(f)

        merged_cfg = update_config(base_config, cfg_yaml)
    else:
        merged_cfg = base_config
    update_cfg = update_config(merged_cfg, kwargs)
    return update_cfg


def object_from_dict(d: Optional[CfgDict], parent=None, **default_kwargs):
    if d is not None:
        kwargs = dict(d).copy()
        object_type = kwargs.pop("type")
        params = kwargs.pop("params", None)

        for name, value in default_kwargs.items():
            params.setdefault(name, value)
        if parent is not None:
            if params is not None:
                return getattr(parent, object_type)(**params)
            else:
                return getattr(parent, object_type)
        else:
            try:
                if params is not None:
                    return pydoc.locate(object_type)(**params)
                else:
                    return pydoc.locate(object_type)
            except:
                raise ImportError("Check module installed and correct params for", object_type)


def load_config() -> MLConfig:
    yaml_config: CommentedMap = fit(**Fire(lambda **kwargs: kwargs))
    cfg = MLConfig(yaml_config)
    return cfg
