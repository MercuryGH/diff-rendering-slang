from attr import dataclass

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.vectors import Vector3


@dataclass
class Camera:
    eye: Vector3
    dir: Vector3
    up: Vector3


@dataclass
class PerspectiveCamera(Camera):
    fov: float
    near: float
    far: float
    width: int
    height: int

    # align with the camera in the shader
    def serialize(self):
        return (
            self.eye.to_list(),
            self.dir.to_list(),
            self.up.to_list(),
            self.fov,
            float(self.width) / float(self.height),
            self.near,
            self.far,
        )


if __name__ == "__main__":
    import numpy as np
    from torch.autograd import Function
    import slangtorch

    camera = PerspectiveCamera(
        eye=Vector3(0.0, 0.0, 3.0),
        dir=Vector3(0.0, 0.0, -1.0),
        up=Vector3(0.0, 1.0, 0.0),
        fov=60.0 / 180.0 * np.pi,
        near=0.1,
        far=100,
        width=512,
        height=512,
    )

    camera_module = slangtorch.loadModule("shaders/soft_ras/camera.slang", verbose=True)

    class TestPerspectiveCamera(Function):
        @staticmethod
        def forward(ctx, camera: PerspectiveCamera):
            ctx.save_for_backward(camera)

            camera_module.TestPerspectiveCamera(camera=camera.serialize()).launchRaw(
                blockSize=(1, 1, 1), gridSize=(1, 1, 1)
            )

    dispatcher = TestPerspectiveCamera()
    dispatcher.apply(camera)
