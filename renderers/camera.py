from abc import abstractmethod
import numpy as np


class Camera:
    def __init__(self, eye: list[float], dir: list[float], up: list[float]) -> None:
        self.eye = eye
        self.dir = dir
        self.up = up

    def get_view_matrix(self):
        return np.matrix(
            [
                [self.dir[0], self.dir[1], self.dir[2], -self.eye[0]],
                [self.up[0], self.up[1], self.up[2], -self.eye[1]],
                [-self.dir[0], -self.dir[1], -self.dir[2], -self.eye[2]],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

    @abstractmethod
    def get_projection_matrix(self):
        pass


class PerspectiveCamera(Camera):
    def __init__(self, eye, dir, up, fov, aspect, near, far) -> None:
        super().__init__(eye, dir, up)
        self.fov = fov
        self.aspect = aspect
        self.near = near
        self.far = far
    
    def get_view_matrix(self):
        return super().get_view_matrix()

    def get_projection_matrix(self):
        return np.matrix(
            [
                [
                    1 / (self.aspect * np.tan(self.fov / 2)),
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    1 / np.tan(self.fov / 2),
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    (self.far + self.near) / (self.near - self.far),
                    2 * self.far * self.near / (self.near - self.far),
                ],
                [
                    0,
                    0,
                    -1,
                    0,
                ],
            ],
            dtype=np.float32,
        )
