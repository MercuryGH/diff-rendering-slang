import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from renderers.soft_ras import SoftRas
from resources.resource import spot_obj
from utils.util import wrap_float_tensor
from renderers.camera import PerspectiveCamera
from utils.vectors import Vector3

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def single_forward(renderer, camera, face_vertices, RenderParams) -> torch.Tensor:
    # Create an instance of RenderParams
    # params = RenderParams(width, height, sigma)
    # Pass it to the forward function
    output = renderer.apply(camera, face_vertices, RenderParams)
    return output


def main():
    renderer = SoftRas()
    # width = height = 512

    face_vertices = spot_obj.face_vertices
    print(face_vertices.shape)

    params = {
        "sigma": 1e-6,
        "epsilon": 1e-3,
        "gamma": 1e-4,
        "distance_epsilon": 1e-5,
        "fg_color": [0.7, 0.8, 0.9],
        "bg_color": [0.3, 0.2, 0.1],
    }

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

    output = single_forward(renderer, camera, face_vertices, params)
    print(output[0, 0, :])
    plt.rcParams["figure.figsize"] = (camera.width, camera.height)
    plt.imshow(output.cpu().numpy())
    plt.show()


if __name__ == "__main__":
    main()
