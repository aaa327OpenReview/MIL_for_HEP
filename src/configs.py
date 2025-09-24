# import yaml
import os
from omegaconf import OmegaConf


def get_configs():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "configs.yaml")
    try:
        config = OmegaConf.load(config_path)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at: {config_path}")
