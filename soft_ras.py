import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from renderers.soft_ras import SoftRas
from resources.resource import spot_obj, tex_image, cube_obj
from utils.util import wrap_float_tensor
from renderers.camera import PerspectiveCamera
from renderers.light import PointLight
from utils.vectors import Vector3
from renderers.transform import Transform, rotate_to_quaternion
from renderers.material import CookTorrance
from IPython.display import HTML

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

WIDTH = 1024
HEIGHT = 768


def update_frame(frame: int, img):
    T = 32
    angle = 2 * np.pi * (frame % T) / T
    transform = Transform(
        rotation=rotate_to_quaternion(Vector3(0.0, 1.0, 1.0), angle)
        * rotate_to_quaternion(Vector3(1.0, 0.0, 0.0), angle),
        position=Vector3(0.0, 0.0, 0.0),
        scaling=Vector3(0.7, 0.7, 0.7),
    )

    camera = PerspectiveCamera(
        eye=Vector3(0.0, 0.0, 3.0),
        dir=Vector3(0.0, 0.0, -1.0),
        up=Vector3(0.0, 1.0, 0.0),
        fov=60.0 / 180.0 * np.pi,
        near=0.1,
        far=100,
        width=WIDTH,
        height=HEIGHT,
    )

    point_light = PointLight(
        position=Vector3(1.0, 2.0, 3.0),
        color=Vector3(1.0, 1.0, 1.0),
        attenuation=Vector3(1.0, 0.0, 0.0),
    )

    material = CookTorrance(
        roughness=0.6,
        metallic=0.1,
    )

    renderer = SoftRas()

    params = {
        "sigma": 1e-6,
        "epsilon": 1e-3,
        "gamma": 1e-4,
        "distance_epsilon": 1e-5,
        # "fg_color": [0.7, 0.8, 0.9],
        "ambient_light": [0.01, 0.01, 0.01],
        "bg_color": [0.4, 0.4, 0.4],
        "gamma_correction": True,
    }

    output: torch.Tensor = renderer.apply(
        camera,
        cube_obj,
        transform,
        point_light,
        material,
        wrap_float_tensor(tex_image),
        params,
    )  # type: ignore

    print(output.shape)
    # print(output[512, 512])
    # print(output.max())
    # plt.rcParams["figure.figsize"] = (camera.width, camera.height)
    # plt.imshow(output.cpu().numpy(), origin="lower")
    # plt.imsave("output.png", output.cpu().numpy(), origin="lower")
    # plt.show()

    new_image = output.cpu().numpy()
    img.set_array(new_image)

    return (img,)


def main():
    fig, ax = plt.subplots()
    initial_image = np.zeros((HEIGHT, WIDTH, 3))
    img = ax.imshow(initial_image, animated=True)
    ani = animation.FuncAnimation(
        fig, update_frame, fargs=(img,), frames=32, interval=100, blit=True
    )

    ani.save("output.gif", writer="imagemagick")  # Save as GIF


if __name__ == "__main__":
    main()
