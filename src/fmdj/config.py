from dataclasses import dataclass
from .variants import VariantConfig

@dataclass(frozen=True)
class OpeningBarnesAndHut:
    opening_angle : float = 0.8

@dataclass(frozen=True)
class OpeningRelative:
    relative_accuracy : float = 0.01

@dataclass(frozen=True)
class Config(VariantConfig):
    opening: OpeningBarnesAndHut | OpeningRelative = OpeningBarnesAndHut()