import os
import yaml

_config = None


def load_config():
    global _config
    if _config is not None:
        return _config

    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as f:
        _config = yaml.safe_load(f)

    return _config
