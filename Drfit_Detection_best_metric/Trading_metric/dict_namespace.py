from typing import Dict, Any, Callable, TypeAlias
from types import SimpleNamespace as sn

# TYPE ALIASES
KAFKA_DICT: TypeAlias = Dict[str, Any]
KAFKA_PUSH_FUNC: TypeAlias = Callable[[str, KAFKA_DICT], None]

# CONVERT DICT TO CLASS NAMESPACE
def DICT_NAMESPACE(d):
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = DICT_NAMESPACE(value)
    return sn(**d)