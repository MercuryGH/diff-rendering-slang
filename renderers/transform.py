from attr import dataclass
import math

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.vectors import Vector3


@dataclass
class Quaternion:
    x: float
    y: float
    z: float
    w: float

    def __add__(self, other: "Quaternion"):
        return Quaternion(
            self.x + other.x, self.y + other.y, self.z + other.z, self.w + other.w
        )

    def __sub__(self, other: "Quaternion"):
        return Quaternion(
            self.x - other.x, self.y - other.y, self.z - other.z, self.w - other.w
        )

    def __mul__(self, other: "Quaternion"):
        x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
        y = self.w * other.y + self.y * other.w + self.z * other.x - self.x * other.z
        z = self.w * other.z + self.z * other.w + self.x * other.y - self.y * other.x
        w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
        return Quaternion(x, y, z, w)

    def norm(self):
        return math.sqrt(
            self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w
        )

    def conjugate(self):
        return Quaternion(-self.x, -self.y, -self.z, self.w)

    def inverse(self):
        norm_squared = (
            self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w
        )
        return Quaternion(
            -self.x / norm_squared,
            -self.y / norm_squared,
            -self.z / norm_squared,
            self.w / norm_squared,
        )

    def normalize(self):
        norm = self.norm()
        return Quaternion(self.x / norm, self.y / norm, self.z / norm, self.w / norm)


def rotate_to_quaternion(axis: Vector3, angle: float) -> Quaternion:
    axis = axis.normalize()
    half_angle = angle / 2
    return Quaternion(
        axis.x * math.sin(half_angle),
        axis.y * math.sin(half_angle),
        axis.z * math.sin(half_angle),
        math.cos(half_angle),
    )


@dataclass
class Transform:
    rotation: Quaternion
    position: Vector3
    scaling: Vector3

    def serialize(self):
        return (
            [self.rotation.x, self.rotation.y, self.rotation.z, self.rotation.w],
            self.position.to_list(),
            self.scaling.to_list(),
        )

if __name__ == "__main__":
    import numpy as np
    from torch.autograd import Function
    import slangtorch

    axis = Vector3(0.0, 1.0, 0.0)
    angle = 60.0 / 180.0 * np.pi

    rotation = rotate_to_quaternion(axis, angle)

    transform = Transform(
        rotation=rotation,
        position=Vector3(-2.0, 3.0, 1.0),
        scaling=Vector3(2.0, 3.0, 7.0),
    )

    transform_module = slangtorch.loadModule("shaders/soft_ras/transform.slang", verbose=True)

    class TestTransform(Function):
        @staticmethod
        def forward(ctx, transform: Transform):
            ctx.save_for_backward(transform)

            transform_module.TestTransform(transform=transform.serialize()).launchRaw(
                blockSize=(1, 1, 1), gridSize=(1, 1, 1)
            )

    dispatcher = TestTransform()
    dispatcher.apply(transform)
