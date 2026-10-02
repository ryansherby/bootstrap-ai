from pathlib import Path

import yaml
import OmegaConf
    
config_path = Path("configs/config.yaml").resolve()

OmegaConf.register_new_resolver(
    "project_root",
    lambda: str(config_path.parent.parent),
    replace=True,
)

CONFIG = OmegaConf.load(config_path)
OmegaConf.resolve(CONFIG)