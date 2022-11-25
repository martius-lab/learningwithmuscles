import yaml
import os
import time
from types import SimpleNamespace

import numpy as np


# TODO remove redundancy from loaders and put into unified dicts,
# you then only load those ones and do the rest model agnostic


def create_log_dict(*args):
    log_dict = {}
    for id_var, var in enumerate(args):
        for id_el, el in enumerate(var):
            log_dict[f"var_{id_var}_el_{id_el}"] = el
    return log_dict


def load_default_params(path=None):
    if path is None:
        path = "../param_files/default_params.yaml"
    with open(path) as f:
        params = yaml.safe_load(f)
    params["model_dir"] = "."
    params = SimpleNamespace(**params)
    args = SimpleNamespace(**params.params_env)
    return params, args


def save_metrics(params, args, metrics):
    paths = [
        f"{params.model_dir}/{args.folder_name}/{key}.npy" for key in metrics.keys()
    ]
    for key, path in zip(metrics.keys(), paths):
        np.save(f"{params.model_dir}/{args.folder_name}/{key}.npy", metrics[key])
    print("Saving metrics")


class DummyLogger:
    def __init__(self):
        pass

    def log_data(self, *args, **kwargs):
        pass

    def write_separate(self, *args, **kwargs):
        pass

    def reset(self, *args, **kwargs):
        pass
