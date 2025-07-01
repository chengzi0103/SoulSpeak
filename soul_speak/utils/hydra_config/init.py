from hydra import initialize, compose
import os
from hydra.core.hydra_config import HydraConfig

from omegaconf import OmegaConf


def load_hydra_config(version_base=None, config_path=f"../../conf", config_name="config.yaml"):
    with initialize(version_base=version_base, config_path=config_path):
        cfg = compose(config_name=config_name, return_hydra_config=True)
        # ✅ 正确设置 HydraConfig
        HydraConfig.instance().set_config(cfg)
    return cfg

def init_env(cfg):
    if "env" in cfg.env:
        for item in cfg.env:
            value = item.value
            os.environ[item.name] = value

conf = load_hydra_config()
init_env(conf)

