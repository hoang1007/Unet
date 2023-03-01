from functools import partial
from omegaconf.omegaconf import Dict
import importlib


def instantiate_from_config(config: Dict):
    if config is None:
        return None

    target_key = "_target_"
    partial_key = "_partial_"

    if target_key not in config:
        raise KeyError(f"Missing {target_key} key in config")

    target_name = config.pop(target_key)

    if config.pop(partial_key, False):
        return partial(get_obj_from_str(target_name), **config)

    return get_obj_from_str(target_name)(**config)


def get_obj_from_str(string: str):
    module, cls = string.rsplit(".", 1)

    return getattr(importlib.import_module(module, package=None), cls)
