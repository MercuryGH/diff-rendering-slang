import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from renderers.soft_ras import SoftRas
from resources.resource import spot_obj, tex_image
from utils.util import wrap_float_tensor
from renderers.camera import PerspectiveCamera
from renderers.light import PointLight
from utils.vectors import Vector3
from renderers.transform import Transform, rotate_to_quaternion
from renderers.material import CookTorrance

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def single_forward(
    renderer, camera, mesh, transform, point_light, material, img, RenderParams
) -> torch.Tensor:
    # Create an instance of RenderParams
    # params = RenderParams(width, height, sigma)
    # Pass it to the forward function
    output = renderer.apply(
        camera, mesh, transform, point_light, material, img, RenderParams
    )
    return output


def main():
    renderer = SoftRas()

    # width = height = 512

    # face_vertices = spot_obj.face_vertices
    # print(face_vertices.shape)

    params = {
        "sigma": 1e-6,
        "epsilon": 1e-3,
        "gamma": 1e-4,
        "distance_epsilon": 1e-5,
        # "fg_color": [0.7, 0.8, 0.9],
        "ambient_light": [0.02, 0.02, 0.02],
        "bg_color": [0.4, 0.4, 0.4],
        "gamma_correction": True,
    }

    camera = PerspectiveCamera(
        eye=Vector3(0.0, 0.0, -3.0),
        dir=Vector3(0.0, 0.0, 1.0),
        up=Vector3(0.0, 1.0, 0.0),
        fov=60.0 / 180.0 * np.pi,
        near=0.1,
        far=100,
        width=512,
        height=384,
    )

    transform = Transform(
        rotation=rotate_to_quaternion(Vector3(0.0, 1.0, 0.0), 0.3 * np.pi),
        position=Vector3(0.0, 0.0, 0.0),
        scaling=Vector3(1.0, 1.5, 1.0),
    )

    point_light = PointLight(
        position=Vector3(1.0, 2.0, -3.0),
        color=Vector3(1.0, 1.0, 1.0),
        attenuation=Vector3(1.0, 0.0, 0.0),
    )

    material = CookTorrance(
        roughness=0.6,
        metallic=0.04,
    )

    image = wrap_float_tensor(tex_image)

    output = single_forward(
        renderer, camera, spot_obj, transform, point_light, material, image, params
    )
    print(output.shape)
    print(output[256, 256])
    print(output.max())
    # plt.rcParams["figure.figsize"] = (camera.width, camera.height)
    # plt.imshow(output.cpu().numpy(), origin="lower")
    plt.imsave("output.png", output.cpu().numpy(), origin="lower")
    plt.show()


if __name__ == "__main__":
    main()
