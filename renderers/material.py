from abc import abstractmethod
from attr import dataclass
from utils.vectors import Vector3


class Material:
    @abstractmethod
    def serialize(self):
        pass


@dataclass
class CookTorrance(Material):
    roughness: float
    metallic: float

    def serialize(self):
        return {
            "metallic": self.metallic,
            "roughness": self.roughness,
        }
