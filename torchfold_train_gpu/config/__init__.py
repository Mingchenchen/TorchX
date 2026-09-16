# torchfold config system.
#
# parse_configs / ConfigManager build an argparse-driven ConfigDict from the
# nested sentinel-type config dicts in configs/configs_base.py + configs_data.py.
# Re-exported here so callers can `from torchfold.config import parse_configs`.
from torchfold.config.config import (
    ConfigManager,
    load_config,
    parse_configs,
    parse_sys_args,
    save_config,
)

__all__ = [
    "ConfigManager",
    "load_config",
    "parse_configs",
    "parse_sys_args",
    "save_config",
]
