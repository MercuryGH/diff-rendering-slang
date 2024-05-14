from abc import abstractmethod
from attr import dataclass
import numpy as np


@dataclass
class Camera:
    eye: list[float]
    dir: list[float]
    up: list[float]


@dataclass
class PerspectiveCamera(Camera):
    fov: float
    aspect: float
    near: float
    far: float
