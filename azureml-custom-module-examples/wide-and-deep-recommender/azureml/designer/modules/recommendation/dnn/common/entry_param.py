from abc import ABCMeta
from azureml.designer.modules.recommendation.dnn.common.constants import TUPLE_SEP


class EntryParam(ABCMeta):
    _LOADER = 'load'
    _ENTRY_PARAM_LOADER = "_ENTRY_PARAM_LOADER"
    param_loaders = {}

    def __new__(mcs, name, bases, class_dict):
        cls = ABCMeta.__new__(mcs, name, bases, class_dict)
        loader_name = class_dict.get(EntryParam._ENTRY_PARAM_LOADER, EntryParam._LOADER)
        EntryParam.param_loaders[cls] = getattr(cls, loader_name)
        return cls

    @staticmethod
    def load(p_type, p_value):
        """Try to load p_value to the specified p_type type.

        If p_value is None or has already been of p_type, just return, or try to find loaders in params_loader.
        """
        if p_value is None or isinstance(p_value, p_type):
            return p_value
        loader = EntryParam.param_loaders.get(p_type, p_type)
        return loader(p_value)


class IntTuple(metaclass=EntryParam):
    @staticmethod
    def load(seq_str: str):
        integers = seq_str.split(TUPLE_SEP)
        try:
            integers = tuple([int(integer) for integer in integers])
        except TypeError:
            raise TypeError(f"Cannot convert {seq_str} to int tuple")
        return integers


class Boolean(metaclass=EntryParam):
    @staticmethod
    def load(bool_str: str):
        if bool_str not in ["True", "False"]:
            raise TypeError(f"Cannot convert {bool_str} to bool type")
        return bool_str == "True"
