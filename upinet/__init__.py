"""U-PINet neural network backbone."""
from .geometry import TargetGeometry, prepare_target
from .model import UPINet, UPINetConfig

__version__ = "0.1.0"
__all__ = ["UPINet", "UPINetConfig", "TargetGeometry", "prepare_target"]
