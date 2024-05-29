from attr import dataclass
import math


@dataclass
class Vector3:
    x: float
    y: float
    z: float

    def __add__(self, other: "Vector3"):
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vector3"):
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other: "Vector3"):
        return Vector3(self.x * other.x, self.y * other.y, self.z * other.z)

    def normalize(self):
        norm = math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
        return Vector3(self.x / norm, self.y / norm, self.z / norm)

    def to_list(self) -> list[float]:
        return [self.x, self.y, self.z]


def dot(v1: Vector3, v2: Vector3) -> float:
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z


def cross(v1: Vector3, v2: Vector3) -> Vector3:
    return Vector3(
        v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x
    )
