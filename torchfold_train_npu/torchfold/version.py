import re

VERSION = "0.0.1"
__version__ = '3.0.1'

def is_release_version():
    return bool(re.match(r"^\d+\.\d+\.\d+$", VERSION))