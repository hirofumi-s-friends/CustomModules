import argparse
import inspect
from functools import wraps
from azureml.designer.modules.recommendation.dnn.common.entry_param import EntryParam


def build_cli_args(func):
    params = inspect.signature(func).parameters
    args = {}
    for p_name, p in params.items():
        # skip *args and **kwargs parameter
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        # set default values
        if p.default == p.empty:
            args[p_name] = None
        else:
            args[p_name] = p.default

    parser = argparse.ArgumentParser()
    for name, default_value in args.items():
        parser.add_argument(f"--{name.replace('_', '-')}", default=default_value)

    kwargs = vars(parser.parse_known_args()[0])
    return kwargs


def params_loader(func):
    @wraps(func)
    def wrapper(obj, **kwargs):
        # filter func necessary kwargs, because ds gives extra parameters
        known_params = inspect.signature(func).parameters
        known_params = [p_name for p_name, p in known_params.items()]
        kwargs = {p_name: p_value for p_name, p_value in kwargs.items() if p_name in known_params}

        for p_name, p_type in inspect.getfullargspec(func).annotations.items():
            if p_name in kwargs:
                kwargs[p_name] = EntryParam.load(p_type, kwargs[p_name])
        return func(obj, **kwargs)

    return wrapper
